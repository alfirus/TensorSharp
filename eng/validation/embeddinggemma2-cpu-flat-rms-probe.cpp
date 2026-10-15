// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Isolated operator screen; never modify or rebuild upstream ggml.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
constexpr float epsilon = 1e-6f;
void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
std::string quote(const std::string & value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { result += '\\'; result += char(c); }
        else if (c < 32) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); result += b; }
        else result += char(c);
    }
    return result + '"';
}
std::string environment(const char * name) { const auto * v = std::getenv(name); return v ? quote(v) : "null"; }
struct options {
    std::string output;
    std::vector<int> rows{113,311,912,8192};
    int threads = 8, warmup = 3, repeats = 20;
    double seconds = .1;
};
options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        require(i + 1 < argc, "Each option needs a value");
        const std::string key = argv[i], value = argv[++i];
        if (key == "--output") o.output = value;
        else if (key == "--threads") o.threads = std::stoi(value);
        else if (key == "--warmup") o.warmup = std::stoi(value);
        else if (key == "--repeats") o.repeats = std::stoi(value);
        else if (key == "--minimum-seconds") o.seconds = std::stod(value);
        else if (key == "--rows") {
            o.rows.clear();
            size_t start = 0;
            do {
                const auto end = value.find(',', start);
                const auto part = value.substr(start, end == std::string::npos ? end : end - start);
                size_t parsed = 0;
                const int rows = std::stoi(part, &parsed);
                require(parsed == part.size() && rows >= 1 && rows <= 8192 &&
                    std::find(o.rows.begin(), o.rows.end(), rows) == o.rows.end(), "Invalid/duplicate --rows value");
                o.rows.push_back(rows);
                if (end == std::string::npos) break;
                start = end + 1;
            } while (true);
        }
        else throw std::runtime_error("Unknown option " + key);
    }
    require(!o.output.empty() && !std::filesystem::exists(o.output), "Require a new --output path");
    require(o.threads >= 1 && o.threads <= 8 && o.warmup >= 1 && o.warmup <= 1000 &&
        o.repeats >= 1 && o.repeats <= 10000 && std::isfinite(o.seconds) && o.seconds >= 0 && o.seconds <= 60,
        "Invalid timing/thread policy");
    require(std::getenv("GGML_CPU_DISABLE_FUSION") == nullptr, "Current fused baseline requires GGML_CPU_DISABLE_FUSION absent");
    return o;
}
struct guarded {
    static constexpr size_t border = 64;
    static constexpr float canary = 9876.5f;
    float * storage = nullptr;
    size_t count, allocated;
    explicit guarded(size_t n) : count(n), allocated((n + 2 * border + 15) / 16 * 16) {
        storage = static_cast<float *>(std::aligned_alloc(64, allocated * sizeof(float)));
        if (!storage) throw std::bad_alloc();
        std::fill(storage, storage + allocated, canary);
    }
    ~guarded() { std::free(storage); }
    guarded(const guarded &) = delete;
    float * data() const { return storage + border; }
    bool guards() const {
        for (size_t i = 0; i < border; ++i) if (storage[i] != canary) return false;
        for (size_t i = border + count; i < allocated; ++i) if (storage[i] != canary) return false;
        return true;
    }
    uint64_t fingerprint() const {
        uint64_t result = 14695981039346656037ULL;
        const auto * bytes = reinterpret_cast<const unsigned char *>(storage);
        for (size_t i = 0; i < allocated * sizeof(float); ++i) result = (result ^ bytes[i]) * 1099511628211ULL;
        return result;
    }
};
struct session {
    ggml_backend_t backend = nullptr;
    ggml_threadpool_t pool = nullptr;
    explicit session(int threads) {
        backend = ggml_backend_cpu_init(); require(backend, "Cannot create CPU backend");
        auto params = ggml_threadpool_params_default(threads);
        pool = ggml_threadpool_new(&params);
        if (!pool) { ggml_backend_free(backend); backend = nullptr; throw std::runtime_error("Cannot create CPU pool"); }
        ggml_backend_cpu_set_n_threads(backend, threads);
        ggml_backend_cpu_set_threadpool(backend, pool);
        ggml_backend_cpu_set_use_ref(backend, false);
    }
    ~session() { if (backend) ggml_backend_free(backend); if (pool) ggml_threadpool_free(pool); }
};
struct shape { const char * name; int width, heads, packed_heads, offset_heads; bool weighted; };
struct problem {
    shape s;
    int tokens;
    guarded inputs, weights;
    uint64_t input_hash, weight_hash;
    std::vector<double> oracle;
    problem(shape value, int n) : s(value), tokens(n), inputs(size_t(s.width) * s.packed_heads * n),
        weights(s.width), oracle(size_t(s.width) * s.heads * n) {
        uint32_t state = 1729;
        auto random = [&] { state = state * 1664525U + 1013904223U; return float(int(state >> 8) - 8388608) / 8388608.f; };
        for (size_t i = 0; i < inputs.count; ++i) inputs.data()[i] = random() * (i % 11 == 0 ? .000013f : 1.000021f);
        for (int i = 0; i < s.width; ++i) weights.data()[i] = .83f + random() * .21f;
        input_hash = inputs.fingerprint(); weight_hash = weights.fingerprint();
        // True Double squares and Double normalization are independent of the
        // required F32-square/float-mean/sqrtf implementation being screened.
        for (int token = 0; token < n; ++token) for (int head = 0; head < s.heads; ++head) {
            const float * x = input(token, head);
            double sum = 0;
            for (int i = 0; i < s.width; ++i) sum += double(x[i]) * double(x[i]);
            const double scale = 1. / std::sqrt(sum / s.width + double(epsilon));
            for (int i = 0; i < s.width; ++i)
                oracle[(size_t(token) * s.heads + head) * s.width + i] = double(x[i]) * scale * (s.weighted ? weights.data()[i] : 1.);
        }
    }
    const float * input(int token, int head) const {
        return inputs.data() + (size_t(token) * s.packed_heads + s.offset_heads + head) * s.width;
    }
    bool immutable() const { return inputs.guards() && weights.guards() && inputs.fingerprint() == input_hash && weights.fingerprint() == weight_hash; }
};
struct row_counters {
    std::atomic<bool> enabled{false};
    std::array<std::atomic<uint64_t>, 8> rows{};
    void reset() { for (auto & count : rows) count.store(0); }
};
void flat_rows(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * w, int ith, int nth, row_counters & counters) {
    const int64_t rows = ggml_nrows(a);
    uint64_t completed = 0;
    for (int64_t row = ith; row < rows; row += nth) {
        const int64_t i1 = row % a->ne[1], i2 = row / a->ne[1] % a->ne[2], i3 = row / (a->ne[1] * a->ne[2]);
        const float * x = reinterpret_cast<const float *>(static_cast<const char *>(a->data) + i1 * a->nb[1] + i2 * a->nb[2] + i3 * a->nb[3]);
        float * y = reinterpret_cast<float *>(static_cast<char *>(dst->data) + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
        double sum = 0;
        for (int64_t i = 0; i < a->ne[0]; ++i) sum += double(x[i] * x[i]);
        const float mean = sum / a->ne[0];
        const float scale = 1.0f / sqrtf(mean + epsilon);
        if (w) {
            const float * weight = reinterpret_cast<const float *>(static_cast<const char *>(w->data) +
                (i1 % w->ne[1]) * w->nb[1] + (i2 % w->ne[2]) * w->nb[2] + (i3 % w->ne[3]) * w->nb[3]);
            for (int64_t i = 0; i < a->ne[0]; ++i) y[i] = x[i] * scale * weight[i];
        } else {
            for (int64_t i = 0; i < a->ne[0]; ++i) y[i] = x[i] * scale;
        }
        ++completed;
    }
    if (counters.enabled.load(std::memory_order_relaxed)) counters.rows[size_t(ith)].fetch_add(completed, std::memory_order_relaxed);
}
void flat_one(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    flat_rows(dst, a, nullptr, ith, nth, *static_cast<row_counters *>(userdata));
}
void flat_two(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * w, int ith, int nth, void * userdata) {
    flat_rows(dst, a, w, ith, nth, *static_cast<row_counters *>(userdata));
}
struct route {
    guarded output_values;
    row_counters counters;
    ggml_backend_t backend;
    ggml_context * ctx = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_buffer_t input_buffer = nullptr, weight_buffer = nullptr, output_buffer = nullptr, intermediates = nullptr;
    bool candidate, fused = false;
    ggml_tensor * input = nullptr, * output = nullptr;
    route(session & owner, problem & p, bool owned) : output_values(p.oracle.size()), backend(owner.backend), candidate(owned) {
        try {
            ctx = ggml_init({1024 * 1024, nullptr, true}); require(ctx, "Cannot create graph context");
            auto * packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.s.width * p.s.packed_heads, p.tokens);
            auto * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.s.width);
            input_buffer = ggml_backend_cpu_buffer_from_ptr(p.inputs.storage, p.inputs.allocated * sizeof(float));
            weight_buffer = ggml_backend_cpu_buffer_from_ptr(p.weights.storage, p.weights.allocated * sizeof(float));
            require(input_buffer && weight_buffer, "Cannot wrap guarded operands");
            require(ggml_backend_tensor_alloc(input_buffer, packed, p.inputs.data()) == GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(weight_buffer, w, p.weights.data()) == GGML_STATUS_SUCCESS, "Cannot bind guarded operands");
            input = ggml_view_3d(ctx, packed, p.s.width, p.s.heads, p.tokens, p.s.width * sizeof(float),
                p.s.width * p.s.packed_heads * sizeof(float), p.s.offset_heads * p.s.width * sizeof(float));
            require(ggml_backend_view_init(input) == GGML_STATUS_SUCCESS, "Cannot bind packed QKV view");
            ggml_set_input(packed); ggml_set_input(w);
            ggml_tensor * rms = nullptr;
            if (owned) output = p.s.weighted ? ggml_map_custom2(ctx, input, w, flat_two, GGML_N_TASKS_MAX, &counters) :
                ggml_map_custom1(ctx, input, flat_one, GGML_N_TASKS_MAX, &counters);
            else { rms = ggml_rms_norm(ctx, input, epsilon); output = p.s.weighted ? ggml_mul(ctx, rms, w) : rms; }
            require(ggml_is_contiguous(output) && output->nb[0] == sizeof(float) && ggml_are_same_shape(input, output), "Output layout differs");
            ggml_set_output(output);
            output_buffer = ggml_backend_cpu_buffer_from_ptr(output_values.storage, output_values.allocated * sizeof(float));
            require(output_buffer && ggml_backend_tensor_alloc(output_buffer, output, output_values.data()) == GGML_STATUS_SUCCESS, "Cannot bind guarded output");
            auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, output);
            if (!owned && p.s.weighted) {
                for (int i = 0; i < graph->n_nodes; ++i) if (graph->nodes[i] == rms) {
                    const ggml_op ops[] = {GGML_OP_RMS_NORM, GGML_OP_MUL};
                    fused = ggml_can_fuse(graph, i, ops, 2);
                }
                require(fused, "Original RMS+MUL baseline is not fusion-eligible");
            }
            // Only the ordinary weighted graph needs its normal RMS intermediate.
            // Candidate adds no packing, permute, contiguous-copy or row scratch.
            intermediates = ggml_backend_alloc_ctx_tensors(ctx, backend);
            require(!(!owned && p.s.weighted) || intermediates, "Cannot allocate baseline RMS intermediate");
            plan = ggml_backend_graph_plan_create(backend, graph); require(plan, "Cannot create persistent CPU graph plan");
        } catch (...) { release(); throw; }
    }
    void release() {
        if (plan) { ggml_backend_graph_plan_free(backend, plan); plan = nullptr; }
        ggml_backend_buffer_free(intermediates); intermediates = nullptr;
        ggml_backend_buffer_free(output_buffer); output_buffer = nullptr;
        ggml_backend_buffer_free(weight_buffer); weight_buffer = nullptr;
        ggml_backend_buffer_free(input_buffer); input_buffer = nullptr;
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
    }
    ~route() { release(); }
    void compute() { require(ggml_backend_graph_plan_compute(backend, plan) == GGML_STATUS_SUCCESS, "CPU graph compute failed"); }
};
struct quality { bool passed = true, bits = true; double max_error = 0, max_ratio = 0; };
quality check(problem & p, const route & r, const std::vector<float> & baseline) {
    require(p.immutable() && r.output_values.guards(), "Source/output canary or fingerprint changed");
    quality q;
    q.bits = std::memcmp(r.output_values.data(), baseline.data(), baseline.size() * sizeof(float)) == 0;
    q.passed = q.bits;
    for (size_t i = 0; i < p.oracle.size(); ++i) {
        const float actual = r.output_values.data()[i];
        require(std::isfinite(actual) && actual != guarded::canary, "Nonfinite/unwritten complete output");
        const double error = std::abs(double(actual) - p.oracle[i]), bound = 2e-6 + 2e-6 * std::abs(p.oracle[i]);
        q.max_error = std::max(q.max_error, error); q.max_ratio = std::max(q.max_ratio, error / bound);
        q.passed &= error <= bound;
    }
    return q;
}
struct timing { std::vector<double> samples; double median = 0; quality warm, final; std::array<uint64_t, 8> worker_rows{}; };
timing measure(problem & p, route & r, const std::vector<float> & baseline, const options & o) {
    std::fill(r.output_values.data(), r.output_values.data() + r.output_values.count, guarded::canary);
    r.counters.reset(); r.counters.enabled.store(true);
    for (int i = 0; i < o.warmup; ++i) r.compute();
    r.counters.enabled.store(false);
    timing t; t.warm = check(p, r, baseline);
    for (size_t i = 0; i < t.worker_rows.size(); ++i) t.worker_rows[i] = r.counters.rows[i].load();
    if (r.candidate) {
        uint64_t total = 0;
        const uint64_t rows = uint64_t(p.s.heads) * p.tokens;
        for (int i = 0; i < o.threads; ++i) {
            const uint64_t expected = (rows / o.threads + (uint64_t(i) < rows % o.threads)) * o.warmup;
            require(t.worker_rows[size_t(i)] == expected, "Candidate worker row coverage differs");
            total += t.worker_rows[size_t(i)];
        }
        require(total == uint64_t(p.s.heads) * p.tokens * o.warmup, "Candidate complete row coverage differs");
    }
    const auto start = clock_type::now();
    do { const auto before = clock_type::now(); r.compute(); t.samples.push_back(std::chrono::duration<double, std::milli>(clock_type::now() - before).count()); }
    while (t.samples.size() < size_t(o.repeats) || std::chrono::duration<double>(clock_type::now() - start).count() < o.seconds);
    t.final = check(p, r, baseline);
    auto sorted = t.samples; std::sort(sorted.begin(), sorted.end()); t.median = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
    return t;
}
void write_phase(std::ostream & out, const char * name, const route & r, const timing & t) {
    out << "{\"name\":" << quote(name) << ",\"route\":" << quote(r.candidate ? "owned-flat-logical-rows" : "unchanged-upstream-rms")
        << ",\"median_ms\":" << t.median << ",\"repeats\":" << t.samples.size() << ",\"warm_passed\":" << (t.warm.passed ? "true" : "false")
        << ",\"final_passed\":" << (t.final.passed ? "true" : "false") << ",\"full_baseline_bit_identity\":" << (t.warm.bits && t.final.bits ? "true" : "false")
        << ",\"full_double_max_error\":" << std::max(t.warm.max_error, t.final.max_error) << ",\"full_double_max_error_over_bound\":" << std::max(t.warm.max_ratio, t.final.max_ratio)
        << ",\"input_output_canaries_and_immutability\":true,\"qualification_worker_rows\":[";
    for (size_t i = 0; i < t.worker_rows.size(); ++i) { if (i) out << ','; out << t.worker_rows[i]; }
    out << "],\"samples_ms\":["; for (size_t i = 0; i < t.samples.size(); ++i) { if (i) out << ','; out << t.samples[i]; } out << "]}";
}
} // namespace

int main(int argc, char ** argv) try {
    const auto o = parse(argc, argv); std::ofstream out(o.output); require(bool(out), "Cannot create report"); out << std::setprecision(17);
    session owner(o.threads);
    out << "{\"threads\":" << o.threads << ",\"warmup\":" << o.warmup << ",\"minimum_repeats\":" << o.repeats << ",\"minimum_seconds\":" << o.seconds
        << ",\"GGML_CPU_DISABLE_FUSION\":" << environment("GGML_CPU_DISABLE_FUSION") << ",\"OMP_NUM_THREADS\":" << environment("OMP_NUM_THREADS")
        << ",\"TS_GGML_CPU_THREADS\":" << environment("TS_GGML_CPU_THREADS") << ",\"environment\":{";
    bool environment_comma = false;
    for (const auto * key : {"GGML_CPU_TILED_MM", "GGML_CPU_TILED_MM_FORCE", "GGML_KLEIDIAI_SME", "GGML_TOTAL_THREADS",
        "GGML_KLEIDIAI_CHUNK_MULTIPLIER", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME", "OPENBLAS_NUM_THREADS"}) {
        if (environment_comma) out << ','; environment_comma = true; out << quote(key) << ':' << environment(key);
    }
    out << "},\"selected_token_counts\":[";
    for (size_t i = 0; i < o.rows.size(); ++i) out << (i ? "," : "") << o.rows[i];
    out << "],\"persistent_pool\":true,\"persistent_graph_plans\":true,\"cases\":[";
    bool comma = false, passed = true; size_t cases = 0;
    for (const auto s : {shape{"q-swa",256,4,8,0,true},shape{"k-swa",256,2,8,4,true},shape{"v-swa",256,2,8,6,false},
        shape{"q-global",512,4,6,0,true},shape{"k-global",512,1,6,4,true},shape{"v-global",512,1,6,5,false}})
        for (int tokens : o.rows) {
            problem p(s, tokens); route baseline(owner, p, false), candidate(owner, p, true);
            baseline.compute(); std::vector<float> original(baseline.output_values.data(), baseline.output_values.data() + baseline.output_values.count);
            const auto initial = check(p, baseline, original); passed &= initial.passed;
            const auto a = measure(p, baseline, original, o), b = measure(p, candidate, original, o),
                c = measure(p, candidate, original, o), d = measure(p, baseline, original, o);
            passed &= a.warm.passed && a.final.passed && b.warm.passed && b.final.passed && c.warm.passed && c.final.passed && d.warm.passed && d.final.passed;
            if (comma) out << ','; comma = true; ++cases;
            out << "{\"family\":" << quote(s.name) << ",\"head_width\":" << s.width << ",\"heads\":" << s.heads << ",\"tokens\":" << tokens
                << ",\"weighted\":" << (s.weighted ? "true" : "false") << ",\"packed_token_stride_bytes\":" << candidate.input->nb[2]
                << ",\"upstream_active_row_workers_source_bound\":" << std::min(o.threads, s.heads) << ",\"upstream_rms_mul_fusion_eligible\":" << (baseline.fused ? "true" : "false")
                << ",\"candidate_over_baseline_ratios\":[" << b.median / a.median << ',' << c.median / d.median << "],\"phases\":[";
            write_phase(out,"baseline-a",baseline,a); out << ','; write_phase(out,"candidate-a",candidate,b); out << ',';
            write_phase(out,"candidate-b",candidate,c); out << ','; write_phase(out,"baseline-b",baseline,d); out << "]}"; out.flush();
        }
    out << "],\"completed\":true,\"passed\":" << (passed ? "true" : "false") << ",\"cases_executed\":" << cases
        << ",\"model_qualified\":false,\"production_promoted\":false,\"limits\":\"complete baseline bit identity plus full independent Double RMS error <=2e-6+2e-6*abs(reference)\""
        << ",\"limitations\":[\"Synthetic F32 inputs and nonuniform weights at checkpoint Q/K/V dimensions; no model or service execution.\",\"Original fusion eligibility and active-worker limit are grounded in unchanged dependency source; candidate warmup worker counts are measured.\",\"Counters are disabled in timed calls; callback includes one relaxed atomic enable check per worker.\",\"No packing/permutation/contiguous copies in candidate graph; source QKV token stride is retained.\",\"Input fingerprints, output guards, complete outputs and warm/final numerical checks are outside timings.\"]}\n";
    return passed ? 0 : 1;
} catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 2; }
