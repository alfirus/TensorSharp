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
constexpr int window = 12, heads = 8;
constexpr float cap = 50.0f;
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
    std::vector<int> lengths{9,25,100,500,1500};
    int threads = 8, warmup = 3, repeats = 20;
    double seconds = .1;
};
int integer(const std::string& text) {
    size_t used = 0; const auto value = std::stoi(text, &used);
    require(used == text.size(), "Invalid integer suffix"); return value;
}
options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        require(i + 1 < argc, "Each option needs a value");
        const std::string key = argv[i], value = argv[++i];
        if (key == "--output") o.output = value;
        else if (key == "--threads") o.threads = integer(value);
        else if (key == "--warmup") o.warmup = integer(value);
        else if (key == "--repeats") o.repeats = integer(value);
        else if (key == "--minimum-seconds") {
            size_t used = 0; o.seconds = std::stod(value, &used); require(used == value.size(), "Invalid duration suffix");
        } else if (key == "--lengths") {
            o.lengths.clear(); size_t start = 0;
            do {
                const auto end = value.find(',', start);
                const int n = integer(value.substr(start, end == std::string::npos ? end : end - start));
                require(n >= 1 && n <= 8192 && std::find(o.lengths.begin(), o.lengths.end(), n) == o.lengths.end(), "Invalid/duplicate length");
                o.lengths.push_back(n); if (end == std::string::npos) break; start = end + 1;
            } while (true);
        } else throw std::runtime_error("Unknown option " + key);
    }
    require(!o.output.empty() && !std::filesystem::exists(o.output), "Require a fresh --output path");
    require((o.threads == 7 || o.threads == 8) && o.warmup >= 1 && o.warmup <= 1000 && o.repeats >= 1 && o.repeats <= 10000 &&
        std::isfinite(o.seconds) && o.seconds >= 0 && o.seconds <= 60, "Invalid thread/timing policy");
    require(std::getenv("GGML_CPU_DISABLE_FUSION") == nullptr, "Baseline requires GGML_CPU_DISABLE_FUSION absent");
    return o;
}
struct guarded {
    static constexpr size_t border = 64;
    static constexpr float canary = 9876.5f;
    size_t count, allocated;
    std::vector<float> backing;
    float * storage;
    explicit guarded(size_t n) : count(n), allocated((n + 2 * border + 15) / 16 * 16),
        backing(allocated + 15),
        storage(reinterpret_cast<float *>((reinterpret_cast<uintptr_t>(backing.data()) + 63) & ~uintptr_t(63))) {
        std::fill(storage, storage + allocated, canary);
    }
    guarded(const guarded&) = delete;
    float * data() const { return storage + border; }
    bool guards() const {
        for (size_t i = 0; i < border; ++i) if (storage[i] != canary) return false;
        for (size_t i = border + count; i < allocated; ++i) if (storage[i] != canary) return false;
        return true;
    }
    uint64_t fingerprint() const {
        uint64_t hash = 14695981039346656037ULL;
        const auto * bytes = reinterpret_cast<const unsigned char *>(storage);
        for (size_t i = 0; i < allocated * sizeof(float); ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
        return hash;
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
struct problem {
    int tokens;
    guarded scores, mask;
    uint64_t score_hash, mask_hash;
    std::vector<double> oracle;
    explicit problem(int n) : tokens(n), scores(size_t(window) * heads * n), mask(size_t(window) * n), oracle(scores.count) {
        uint32_t state = 1729;
        auto random = [&] { state = state * 1664525U + 1013904223U; return float(int(state >> 8) - 8388608) / 8388608.f; };
        for (int time = 0; time < n; ++time) {
            for (int key = 0; key < window; ++key) {
                const int actual_key = time + key - (window - 1);
                mask.data()[size_t(time) * window + key] = actual_key < 0 || actual_key >= n ? -1.e9f : 0.f;
            }
            for (int head = 0; head < heads; ++head) {
                const size_t row = size_t(time) * heads + head;
                double maximum = -INFINITY;
                for (int key = 0; key < window; ++key) {
                    const float noise = random();
                    const float value = row % 5 == 0 ? noise * 150.f : row % 5 == 1 ? noise * 1e-6f :
                        row % 5 == 2 ? (key % 3 ? -5000.f : 5000.f) : row % 5 == 3 ? 8.f + noise * 1e-5f : (key % 2 ? -0.f : 0.f);
                    scores.data()[row * window + key] = value;
                    const double score = double(cap) * std::tanh(double(value) / double(cap)) + double(mask.data()[size_t(time) * window + key]);
                    oracle[row * window + key] = score; maximum = std::max(maximum, score);
                }
                double sum = 0;
                for (int key = 0; key < window; ++key) sum += std::exp(oracle[row * window + key] - maximum);
                for (int key = 0; key < window; ++key) oracle[row * window + key] = std::exp(oracle[row * window + key] - maximum) / sum;
            }
        }
        score_hash = scores.fingerprint(); mask_hash = mask.fingerprint();
    }
    bool immutable() const { return scores.guards() && mask.guards() && scores.fingerprint() == score_hash && mask.fingerprint() == mask_hash; }
};
struct route {
    guarded output_values;
    ggml_backend_t backend;
    ggml_context * ctx = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_buffer_t score_buffer = nullptr, mask_buffer = nullptr, output_buffer = nullptr, intermediates = nullptr;
    bool candidate;
    int graph_nodes = 0;
    ggml_tensor * softmax = nullptr, * output = nullptr;
    route(session& owner, problem& p, bool reshape) : output_values(p.oracle.size()), backend(owner.backend), candidate(reshape) {
        try {
            ctx = ggml_init({1024 * 1024, nullptr, true}); require(ctx, "Cannot create graph context");
            auto * scores = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, window, 1, heads, p.tokens);
            auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, window, 1, 1, p.tokens);
            score_buffer = ggml_backend_cpu_buffer_from_ptr(p.scores.storage, p.scores.allocated * sizeof(float));
            mask_buffer = ggml_backend_cpu_buffer_from_ptr(p.mask.storage, p.mask.allocated * sizeof(float));
            require(score_buffer && mask_buffer, "Cannot wrap guarded inputs");
            require(ggml_backend_tensor_alloc(score_buffer, scores, p.scores.data()) == GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(mask_buffer, mask, p.mask.data()) == GGML_STATUS_SUCCESS, "Cannot bind guarded inputs");
            ggml_set_input(scores); ggml_set_input(mask);
            // Identical actual audio graph prefix: F32 soft cap followed by a
            // causal finite padding mask. Only the softmax logical shape differs.
            auto * capped = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, scores, 1.f / cap)), cap);
            auto * masked = ggml_add(ctx, capped, mask);
            require(ggml_is_contiguous(masked), "Expected contiguous scores before shape-only view");
            auto * softmax_input = reshape ? ggml_reshape_2d(ctx, masked, window, int64_t(heads) * p.tokens) : masked;
            softmax = ggml_soft_max(ctx, softmax_input);
            output = reshape ? ggml_reshape_4d(ctx, softmax, window, 1, heads, p.tokens) : softmax;
            require(ggml_are_same_shape(output, scores) && ggml_is_contiguous(output), "Changed output shape/strides");
            // Bind the non-view softmax destination. The final shape-only view
            // aliases this exact guarded output after ordinary backend allocation.
            output_buffer = ggml_backend_cpu_buffer_from_ptr(output_values.storage, output_values.allocated * sizeof(float));
            require(output_buffer && ggml_backend_tensor_alloc(output_buffer, softmax, output_values.data()) == GGML_STATUS_SUCCESS, "Cannot bind guarded output");
            ggml_set_output(output);
            auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, output);
            graph_nodes = graph->n_nodes;
            intermediates = ggml_backend_alloc_ctx_tensors(ctx, backend); require(intermediates, "Cannot allocate graph intermediates");
            require(output->data == output_values.data(), "Final reshape introduced another output buffer");
            plan = ggml_backend_graph_plan_create(backend, graph); require(plan, "Cannot create persistent CPU graph plan");
        } catch (...) { release(); throw; }
    }
    void release() {
        if (plan) { ggml_backend_graph_plan_free(backend, plan); plan = nullptr; }
        ggml_backend_buffer_free(intermediates); intermediates = nullptr;
        ggml_backend_buffer_free(output_buffer); output_buffer = nullptr;
        ggml_backend_buffer_free(mask_buffer); mask_buffer = nullptr;
        ggml_backend_buffer_free(score_buffer); score_buffer = nullptr;
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
    }
    ~route() { release(); }
    void compute() { require(ggml_backend_graph_plan_compute(backend, plan) == GGML_STATUS_SUCCESS, "CPU graph compute failed"); }
};
struct quality {
    bool passed = true, bits = true, finite = true, probability = true, masked_zero = true;
    double max_error = 0, max_ratio = 0, max_sum_error = 0;
};
quality check(problem& p, const route& r, const std::vector<float>& baseline) {
    require(p.immutable() && r.output_values.guards(), "Input/output canary or fingerprint changed");
    quality q;
    q.bits = std::memcmp(r.output_values.data(), baseline.data(), baseline.size() * sizeof(float)) == 0;
    for (size_t row = 0; row < size_t(heads) * p.tokens; ++row) {
        double sum = 0;
        for (int key = 0; key < window; ++key) {
            const size_t i = row * window + key;
            const float actual = r.output_values.data()[i];
            q.finite &= std::isfinite(actual) && actual != guarded::canary;
            if (!std::isfinite(actual)) continue;
            const double error = std::abs(double(actual) - p.oracle[i]), bound = 2e-6 + 2e-6 * std::abs(p.oracle[i]);
            q.max_error = std::max(q.max_error, error); q.max_ratio = std::max(q.max_ratio, error / bound);
            q.passed &= error <= bound; sum += actual;
            q.probability &= actual >= 0 && actual <= 1.f;
            if (p.mask.data()[(row / heads) * window + key] < 0) q.masked_zero &= actual == 0.f;
        }
        q.max_sum_error = std::max(q.max_sum_error, std::abs(sum - 1.));
        q.probability &= std::abs(sum - 1.) <= 5e-6;
    }
    q.passed &= q.bits && q.finite && q.probability && q.masked_zero;
    return q;
}
struct timing { std::vector<double> samples; double median = 0; quality warm, final; };
timing measure(problem& p, route& r, const std::vector<float>& baseline, const options& o) {
    std::fill(r.output_values.data(), r.output_values.data() + r.output_values.count, guarded::canary);
    for (int i = 0; i < o.warmup; ++i) r.compute();
    timing t; t.warm = check(p, r, baseline);
    const auto start = clock_type::now();
    do { const auto before = clock_type::now(); r.compute(); t.samples.push_back(std::chrono::duration<double, std::milli>(clock_type::now() - before).count()); }
    while (t.samples.size() < size_t(o.repeats) || std::chrono::duration<double>(clock_type::now() - start).count() < o.seconds);
    t.final = check(p, r, baseline);
    auto sorted = t.samples; std::sort(sorted.begin(), sorted.end()); t.median = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
    return t;
}
void write_phase(std::ostream& out, const char * name, const route& r, const timing& t) {
    out << "{\"name\":" << quote(name) << ",\"route\":" << quote(r.candidate ? "unchanged-softmax-flat-rows" : "unchanged-softmax-original-4d")
        << ",\"median_ms\":" << t.median << ",\"repeats\":" << t.samples.size() << ",\"warm_passed\":" << (t.warm.passed ? "true" : "false")
        << ",\"final_passed\":" << (t.final.passed ? "true" : "false") << ",\"full_baseline_bit_identity\":" << (t.warm.bits && t.final.bits ? "true" : "false")
        << ",\"finite\":" << (t.warm.finite && t.final.finite ? "true" : "false") << ",\"probabilities\":" << (t.warm.probability && t.final.probability ? "true" : "false")
        << ",\"masked_exact_zero\":" << (t.warm.masked_zero && t.final.masked_zero ? "true" : "false")
        << ",\"full_double_max_error\":" << std::max(t.warm.max_error, t.final.max_error) << ",\"full_double_max_error_over_bound\":" << std::max(t.warm.max_ratio, t.final.max_ratio)
        << ",\"max_probability_sum_error\":" << std::max(t.warm.max_sum_error, t.final.max_sum_error)
        << ",\"input_output_canaries_and_immutability\":true,\"samples_ms\":[";
    for (size_t i = 0; i < t.samples.size(); ++i) out << (i ? "," : "") << t.samples[i]; out << "]}";
}
} // namespace
int main(int argc, char ** argv) try {
    const auto o = parse(argc, argv); std::ofstream out(o.output); require(bool(out), "Cannot create report"); out << std::setprecision(17);
    session owner(o.threads);
    out << "{\"threads\":" << o.threads << ",\"window\":" << window << ",\"heads\":" << heads << ",\"cap\":" << cap
        << ",\"warmup\":" << o.warmup << ",\"minimum_repeats\":" << o.repeats << ",\"minimum_seconds\":" << o.seconds << ",\"environment\":{";
    bool comma = false;
    for (const auto * key : {"GGML_CPU_DISABLE_FUSION","GGML_CPU_TILED_MM","GGML_CPU_TILED_MM_FORCE","GGML_TOTAL_THREADS",
        "GGML_KLEIDIAI_SME","GGML_KLEIDIAI_CHUNK_MULTIPLIER","OMP_NUM_THREADS","OMP_WAIT_POLICY","GOMP_SPINCOUNT","KMP_BLOCKTIME",
        "TS_GGML_CPU_THREADS","OPENBLAS_NUM_THREADS"}) {
        out << (comma ? "," : "") << quote(key) << ':' << environment(key); comma = true;
    }
    out << "},\"persistent_pool\":true,\"persistent_graph_plans\":true,\"cases\":[";
    comma = false; bool passed = true;
    for (const int n : o.lengths) {
        problem p(n); route baseline(owner,p,false), candidate(owner,p,true);
        baseline.compute(); const std::vector<float> original(baseline.output_values.data(), baseline.output_values.data() + baseline.output_values.count);
        const auto initial = check(p,baseline,original); passed &= initial.passed;
        const auto a = measure(p,baseline,original,o), b = measure(p,candidate,original,o), c = measure(p,candidate,original,o), d = measure(p,baseline,original,o);
        passed &= a.warm.passed && a.final.passed && b.warm.passed && b.final.passed && c.warm.passed && c.final.passed && d.warm.passed && d.final.passed;
        out << (comma ? "," : "") << "{\"tokens\":" << n << ",\"complete_elements\":" << p.oracle.size()
            << ",\"initial_baseline_passed\":" << (initial.passed ? "true" : "false")
            << ",\"baseline_graph_nodes\":" << baseline.graph_nodes << ",\"candidate_graph_nodes\":" << candidate.graph_nodes
            << ",\"baseline_softmax_ne1\":" << baseline.softmax->ne[1] << ",\"candidate_softmax_ne1\":" << candidate.softmax->ne[1]
            << ",\"active_softmax_workers_source_bound\":[1," << std::min(o.threads,heads*n) << "]"
            << ",\"candidate_over_baseline_ratios\":[" << b.median/a.median << ',' << c.median/d.median << "],\"phases\":[";
        write_phase(out,"baseline-a",baseline,a); out << ','; write_phase(out,"candidate-a",candidate,b); out << ',';
        write_phase(out,"candidate-b",candidate,c); out << ','; write_phase(out,"baseline-b",baseline,d); out << "]}"; out.flush(); comma = true;
    }
    out << "],\"completed\":true,\"passed\":" << (passed ? "true" : "false") << ",\"cases_executed\":" << o.lengths.size()
        << ",\"model_qualified\":false,\"production_promoted\":false,\"limits\":\"complete original4D baseline bit identity, independentDouble error<=2e-6+2e-6*abs(reference), sumerror<=5e-6,maskedexactzero\""
        << ",\"limitations\":[\"Synthetic scores at actual audio window12/heads8; no learned projection, model,frontend or service execution.\",\"Timed prefix includes unchanged scale/tanh/scale/ADD(mask); only softmax reshape differs. TANH and scales remain single-task baseline nodes.\",\"Active worker limits are source-derived, not instrumented worker counts.\",\"Graph creation,allocation,oracle/quality/canary/fingerprint checks excluded from timing.\",\"Candidate adds only two shape-only views, no tensor copy or arithmetic kernel.\"]}\n";
    require(bool(out), "Report write failed"); return passed ? 0 : 1;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
