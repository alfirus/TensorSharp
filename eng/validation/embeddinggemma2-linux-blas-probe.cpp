// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone, TensorSharp-owned F32 matmul probe; no ggml linkage/patches.
// g++ -std=c++17 -O3 -pthread -I TensorSharp.GGML.Native this.cpp -ldl -o <ignored-output>
#include "embedding_linux_blas.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace tsg_embeddings;

static std::string quote(const std::string & value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { result += '\\'; result += char(c); }
        else if (c < 32) { char escaped[8]; std::snprintf(escaped, sizeof(escaped), "\\u%04x", c); result += escaped; }
        else result += char(c);
    }
    return result + "\"";
}

// Persistent worker cohort, matching an existing ggml pool rather than timing
// thread creation for every projection. All tests join it before plan.close.
class cohort {
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_, finished_;
    std::function<void(int)> task_;
    int generation_ = 0, remaining_ = 0, count_;
    bool stopping_ = false;
public:
    explicit cohort(int count) : count_(count) {
        for (int ith = 1; ith < count; ith++) workers_.emplace_back([this, ith] {
            int previous = 0;
            std::unique_lock<std::mutex> lock(mutex_);
            while (true) {
                ready_.wait(lock, [&] { return stopping_ || generation_ != previous; });
                if (stopping_) return;
                previous = generation_; auto work = task_;
                lock.unlock(); work(ith); lock.lock();
                if (--remaining_ == 0) finished_.notify_one();
            }
        });
    }
    void run(const std::function<void(int)> & task) {
        { std::lock_guard<std::mutex> lock(mutex_); task_ = task; remaining_ = count_ - 1; generation_++; }
        ready_.notify_all(); task(0);
        std::unique_lock<std::mutex> lock(mutex_); finished_.wait(lock, [&] { return remaining_ == 0; });
    }
    ~cohort() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        ready_.notify_all(); for (auto & worker : workers_) worker.join();
    }
};

struct problem {
    int k, m, n, ws, xs, ys;
    std::vector<float> w, x, y;
    static constexpr float canary = 37;
    problem(int inputs, int outputs, int rows, bool padded, float input_scale = 1, float weight_scale = 1)
        : k(inputs), m(outputs), n(rows), ws(k + (padded ? 5 : 0)), xs(k + (padded ? 7 : 0)), ys(m + (padded ? 3 : 0)),
          w(size_t(ws) * m + 5, canary), x(size_t(xs) * n + 5, canary), y(size_t(ys) * n + 5, canary) {
        for (int row = 0; row < m; row++) for (int i = 0; i < k; i++) w[1 + size_t(row) * ws + i] = weight_scale * std::sin(float(row * 17 + i) * .13791f) * .03717f;
        for (int row = 0; row < n; row++) for (int i = 0; i < k; i++) x[1 + size_t(row) * xs + i] = input_scale * std::cos(float(row * 13 + i) * .17317f) * .19137f;
    }
    blas_const_matrix weights() const { return {w.data() + 1, w.size() - 1, m, k, ws}; }
    blas_const_matrix input() const { return {x.data() + 1, x.size() - 1, n, k, xs}; }
    blas_matrix output() { return {y.data() + 1, y.size() - 1, n, m, ys}; }
    bool guards() const {
        if (y.front() != canary) return false;
        for (int row = 0; row < n; row++) for (int i = m; i < ys; i++) if (y[1 + size_t(row) * ys + i] != canary) return false;
        for (size_t i = 1 + size_t(n) * ys; i < y.size(); i++) if (y[i] != canary) return false;
        return true;
    }
};

static void counts_json(const linux_blas_counts & c) {
    std::printf("{\"gemm_calls\":%llu,\"output_channel_calls\":%llu,\"token_row_calls\":%llu,\"output_elements\":%llu,\"empty_partitions\":%llu,\"worker_mask\":%llu,\"shape_rejections\":%llu,\"policy_rejections\":%llu,\"active_calls\":%llu}",
        (unsigned long long)c.gemm_calls, (unsigned long long)c.output_channel_calls, (unsigned long long)c.token_row_calls,
        (unsigned long long)c.output_elements, (unsigned long long)c.empty_partitions, (unsigned long long)c.worker_mask,
        (unsigned long long)c.shape_rejections, (unsigned long long)c.policy_rejections,(unsigned long long)c.active_calls);
}

static void gate(linux_blas_plan & plan, cohort & pool, int workers, problem & p, blas_partition axis) {
    std::atomic<bool> success{true};
    pool.run([&](int ith) { if (!plan.try_partition(p.weights(), p.input(), p.output(), ith, workers, axis)) success = false; });
    if (!success || !p.guards()) throw std::runtime_error("partition-or-output-canary-failed");
}

struct oracle_result { double double_max_abs, scalar_max_abs; size_t outputs_checked; };
static oracle_result check_double(const problem & p) {
    double max_error = 0, max_scalar_error = 0; size_t checked = 0;
    std::vector<int> rows, outputs;
    if (int64_t(p.m) * p.n * p.k <= 2000000) {
        for (int r = 0; r < p.n; r++) rows.push_back(r);
        for (int m = 0; m < p.m; m++) outputs.push_back(m);
    } else { rows = {0, 1, p.n / 2, p.n - 1}; outputs = {0, 1, p.m / 2, p.m - 2, p.m - 1}; }
    for (int row : rows) for (int output : outputs) {
        if (row >= p.n || output >= p.m) continue;
        double expected = 0, absolute = 0; float scalar = 0;
        for (int i = 0; i < p.k; i++) {
            double term = double(p.w[1 + size_t(output) * p.ws + i]) * p.x[1 + size_t(row) * p.xs + i];
            expected += term; absolute += std::abs(term);
            volatile float rounded_term = p.w[1 + size_t(output) * p.ws + i] * p.x[1 + size_t(row) * p.xs + i];
            scalar += rounded_term; // Independent ordered F32 product+sum.
        }
        double actual = p.y[1 + size_t(row) * p.ys + output], error = std::abs(actual - expected);
        // Original F32 operands, unrounded independent Double oracle. Range
        // cases use a forward-error bound proportional to sum(|products|),
        // rather than hiding cancellation behind a relative-to-output check.
        double bound = std::max(2e-5, absolute * 1e-6);
        if (!std::isfinite(actual) || error > bound) throw std::runtime_error("independent-double-gate-failed");
        max_error = std::max(max_error, error); max_scalar_error = std::max(max_scalar_error, std::abs(actual - scalar)); checked++;
    }
    return {max_error, max_scalar_error, checked};
}

int main(int argc, char ** argv) {
    const char * path = nullptr; bool benchmark = false, rotate_weights = false; int workers = 8;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--library" && i + 1 < argc) path = argv[++i];
        else if (std::string(argv[i]) == "--workers" && i + 1 < argc) workers = std::stoi(argv[++i]);
        else if (std::string(argv[i]) == "--benchmark") benchmark = true;
        else if (std::string(argv[i]) == "--rotate-weights") rotate_weights = true;
        else { std::fprintf(stderr, "unknown/incomplete argument\n"); return 2; }
    }
    linux_blas_plan plan(path, workers);
    if (!plan.available()) { std::printf("{\"completed\":false,\"available\":false,\"reason\":%s,\"library\":%s,\"config\":%s}\n", quote(plan.reason()).c_str(), quote(plan.library_path()).c_str(), quote(plan.library_config()).c_str()); return 2; }
    try {
        if (workers < 1 || workers > 8) throw std::runtime_error("invalid-worker-budget");
        cohort pool(workers);
        std::printf("{\"available\":true,\"library\":%s,\"config\":%s,\"actual_blas_workers\":1,\"existing_worker_budget\":%d,\"cases\":[", quote(plan.library_path()).c_str(), quote(plan.library_config()).c_str(), workers);
        bool comma = false;
        for (auto shape : {std::vector<int>{65,19,3}, {129,37,13}, {257,131,29}, {768,12288,13}, {768,6144,29}, {768,2304,256}, {768,768,512}})
        for (auto axis : {blas_partition::output_channels, blas_partition::token_rows}) {
            problem p(shape[0], shape[1], shape[2], true);
            gate(plan, pool, workers, p, axis); auto checked = check_double(p);
            if (comma) std::printf(","); comma = true;
            std::printf("{\"k\":%d,\"m\":%d,\"n\":%d,\"axis\":%s,\"padded\":true,\"double_max_abs\":%.17g,\"ordered_scalar_max_abs\":%.17g,\"outputs_checked\":%zu}", p.k,p.m,p.n, quote(axis == blas_partition::output_channels ? "channels" : "tokens").c_str(),checked.double_max_abs,checked.scalar_max_abs,checked.outputs_checked);
        }
        // F32 retains activation/weight exponent range beyond Half.
        for (auto scales : {std::pair<float,float>{80000,1}, {1,80000}}) {
            problem p(65,19,3,true,scales.first,scales.second);
            gate(plan,pool,workers,p,blas_partition::output_channels); auto checked = check_double(p);
            std::printf(",{\"k\":65,\"m\":19,\"n\":3,\"range_input\":%.9g,\"range_weight\":%.9g,\"double_max_abs\":%.17g,\"ordered_scalar_max_abs\":%.17g,\"outputs_checked\":%zu}",scales.first,scales.second,checked.double_max_abs,checked.scalar_max_abs,checked.outputs_checked);
        }
        problem cancel(1024,2,3,true);
        for (int row=0;row<cancel.n;row++) for(int i=0;i<cancel.k;i++) cancel.x[1+size_t(row)*cancel.xs+i]=1;
        for (int row=0;row<cancel.m;row++) for(int i=0;i<cancel.k;i++) cancel.w[1+size_t(row)*cancel.ws+i]= i<256?8.f:i<768?1.f/4096:-8.f;
        gate(plan,pool,workers,cancel,blas_partition::output_channels);
        for (int row=0;row<3;row++) for(int m=0;m<2;m++) if(cancel.y[1+size_t(row)*cancel.ys+m] != .125f) throw std::runtime_error("F32-cancellation-gate-failed");
        std::printf("],\"cancellation_passed\":true");
        problem invalid(65,19,3,true); auto original=invalid.y; auto output=invalid.output();
        if (plan.try_partition(invalid.weights(),invalid.input(),output,0,9,blas_partition::token_rows)) throw std::runtime_error("worker-policy-rejection-failed");
        output.elements=1;
        if(plan.try_partition(invalid.weights(),invalid.input(),output,0,workers,blas_partition::output_channels)) throw std::runtime_error("tail-bounds-rejection-failed");
        output=invalid.output();output.data=invalid.x.data()+1;output.elements=invalid.x.size()-1;
        if(plan.try_partition(invalid.weights(),invalid.input(),output,0,workers,blas_partition::token_rows)) throw std::runtime_error("alias-rejection-failed");
        if(invalid.y != original) throw std::runtime_error("rejection-modified-output");
        linux_blas_plan peer(path,workers);
        if(!peer.available()) throw std::runtime_error("peer-load-failed");
        plan.close(); plan.close();
        if(plan.try_partition(invalid.weights(),invalid.input(),invalid.output(),0,workers,blas_partition::token_rows)) throw std::runtime_error("closed-plan-executed");
        gate(peer,pool,workers,invalid,blas_partition::output_channels); check_double(invalid);
        linux_blas_plan in_flight(path,workers);problem pending(257,131,8192,false);std::atomic<bool>done{false},executed{false};
        std::thread running([&]{executed=in_flight.try_partition(pending.weights(),pending.input(),pending.output(),0,1,blas_partition::token_rows);done=true;});
        while(!done && in_flight.counts().active_calls==0)std::this_thread::yield();
        bool observed_active=in_flight.counts().active_calls>0;in_flight.close();running.join();
        if(!observed_active||!executed||in_flight.counts().active_calls!=0)throw std::runtime_error("in-flight-loader-lifetime-not-proven");
        check_double(pending);if(!pending.guards())throw std::runtime_error("in-flight-canary-failed");
        std::printf(",\"guard_lifetime_passed\":true,\"in_flight_close_observed\":true,\"counts\":");counts_json(plan.counts());std::printf(",\"peer_counts\":");counts_json(peer.counts());
        if (benchmark) {
            std::printf(",\"timings\":[");bool timed_comma=false;
            // Actual pinned text geometry. The broad correctness cases above
            // deliberately remain generic operators, not a model workload.
            for(auto shape:{std::vector<int>{512,12288,13},{512,12288,29},
                           {512,12288,128},{512,12288,256},{512,12288,512},
                           {512,2048,128},{512,4096,128},{1024,512,128},{2048,512,128},
                           {512,2048,256},{512,3072,512},
                           {512,4096,256},{512,4096,512},{1024,512,256},{2048,512,512},
                           {512,2048,8192},{512,4096,8192},{2048,512,8192}})
            for(int mode=0;mode<3;mode++) {
                // A small rotating cohort avoids assuming every layer's weights
                // fit in the shared LLC. This is still a synthetic cache regime,
                // not a substitute for complete-model measurements.
                const size_t weight_bytes = size_t(shape[0]) * shape[1] * sizeof(float);
                const int rotations = rotate_weights && shape[2] <= 512
                    ? std::max(8, std::min(64, int((64 * 1024 * 1024 + weight_bytes - 1) / weight_bytes))) : 1;
                std::vector<std::unique_ptr<problem>> problems;
                for (int i = 0; i < rotations; ++i) problems.push_back(std::make_unique<problem>(shape[0],shape[1],shape[2],false));
                size_t current = 0;
                auto call=[&]{auto & p=*problems[current++ % problems.size()];if(mode==0){if(!peer.try_partition(p.weights(),p.input(),p.output(),0,1,blas_partition::token_rows))throw std::runtime_error("single-call-failed");}else gate(peer,pool,workers,p,mode==1?blas_partition::output_channels:blas_partition::token_rows);};
                for(int w=0;w<3;w++)call();std::vector<double>times;auto start=std::chrono::steady_clock::now();
                do{auto t=std::chrono::steady_clock::now();call();times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}while(times.size()<20||std::chrono::steady_clock::now()-start<std::chrono::milliseconds(250));
                for (const auto & p : problems) check_double(*p);
                const auto & p = *problems[0];
                std::sort(times.begin(),times.end());double median=times[times.size()/2];
                if(timed_comma)std::printf(",");timed_comma=true;
                std::printf("{\"workload\":\"embeddinggemma2-text\",\"k\":%d,\"m\":%d,\"n\":%d,\"mode\":%s,\"weight_rotations\":%d,\"weight_cohort_bytes\":%zu,\"wall_median_ms\":%.17g,\"samples\":%zu}",p.k,p.m,p.n,quote(mode==0?"single-blas-worker":mode==1?"partition-channels":"partition-tokens").c_str(),rotations,p.w.size()*sizeof(float)*rotations,median,times.size());
            }
            std::printf("]");
        }
        std::printf(",\"completed\":true,\"scope\":\"owned standalone F32 helper; no GGML/model execution or default promotion\",\"timing_limitations\":\"CPU wall includes cohort issue/wait for partitions; single BLAS uses one worker; excludes runtime load/data preparation; no graph-pool/model speed claim\"}\n");return 0;
    } catch(const std::exception & failure){std::fprintf(stderr,"probe failure: %s\n",failure.what());return 1;}
}
