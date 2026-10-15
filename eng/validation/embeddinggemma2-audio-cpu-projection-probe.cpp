// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone graph probe. Link unchanged ggml CPU/base; never build/patch ggml.
// See embeddinggemma2-audio-cpu-projection-probe.md for compile/run instructions.
#include "embedding_cpu_matmul.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
using tsg_embeddings::linux_blas_plan;

std::string quote(const std::string & value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { result += '\\'; result += char(c); }
        else if (c < 32) { char escaped[8]; std::snprintf(escaped, sizeof(escaped), "\\u%04x", c); result += escaped; }
        else result += char(c);
    }
    return result + "\"";
}

std::string environment(const char * name) {
    const char * value = std::getenv(name);
    return value ? quote(value) : "null";
}

struct options {
    std::string output, library;
    int threads = 8, warmup = 3, repeats = 20;
    double minimum_seconds = 1;
};

options parse(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("Every option requires a value");
        const std::string value = argv[++i];
        if (name == "--output") result.output = value;
        else if (name == "--library") result.library = value;
        else if (name == "--threads") result.threads = std::stoi(value);
        else if (name == "--warmup") result.warmup = std::stoi(value);
        else if (name == "--repeats") result.repeats = std::stoi(value);
        else if (name == "--minimum-seconds") result.minimum_seconds = std::stod(value);
        else throw std::runtime_error("Unknown option: " + name);
    }
    if (result.output.empty() || result.threads < 1 || result.threads > 8 ||
        result.warmup < 1 || result.warmup > 10000 || result.repeats < 1 || result.repeats > 10000 ||
        !std::isfinite(result.minimum_seconds) || result.minimum_seconds < 0 || result.minimum_seconds > 60)
        throw std::runtime_error("Require --output, threads 1..8, warmup 1..10000, repeats 1..10000, minimum-seconds 0..60");
    return result;
}

struct guarded {
    static constexpr size_t guard = 64;
    static constexpr float canary = 9876.5f;
    float * storage = nullptr;
    size_t elements, allocated;
    explicit guarded(size_t count) : elements(count), allocated((count + 2 * guard + 15) / 16 * 16) {
        storage = static_cast<float *>(std::aligned_alloc(64, allocated * sizeof(float)));
        if (!storage) throw std::bad_alloc();
        std::fill(storage, storage + allocated, canary);
    }
    ~guarded() { std::free(storage); }
    guarded(const guarded &) = delete;
    guarded & operator=(const guarded &) = delete;
    float * data() { return storage + guard; }
    const float * data() const { return storage + guard; }
    bool guards() const {
        for (size_t i = 0; i < guard; ++i) if (storage[i] != canary) return false;
        for (size_t i = guard + elements; i < allocated; ++i) if (storage[i] != canary) return false;
        return true;
    }
    uint64_t fingerprint() const {
        const auto * bytes = reinterpret_cast<const unsigned char *>(storage);
        uint64_t result = 14695981039346656037ULL;
        for (size_t i = 0; i < allocated * sizeof(float); ++i) result = (result ^ bytes[i]) * 1099511628211ULL;
        return result;
    }
};

struct session {
    ggml_backend_t backend = nullptr;
    ggml_threadpool_t pool = nullptr;
    explicit session(int threads) {
        backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("Cannot initialize ggml CPU");
        auto parameters = ggml_threadpool_params_default(threads);
        pool = ggml_threadpool_new(&parameters);
        if (!pool) { ggml_backend_free(backend); backend = nullptr; throw std::runtime_error("Cannot initialize persistent CPU threadpool"); }
        ggml_backend_cpu_set_n_threads(backend, threads);
        ggml_backend_cpu_set_threadpool(backend, pool);
    }
    ~session() {
        if (backend) ggml_backend_free(backend);
        if (pool) ggml_threadpool_free(pool);
    }
};

struct problem {
    int k, m, n;
    guarded weights, inputs;
    uint64_t weight_fingerprint, input_fingerprint;
    problem(int inner, int columns, int rows, uint32_t seed) :
        k(inner), m(columns), n(rows), weights(size_t(inner) * columns), inputs(size_t(inner) * rows) {
        auto random = [&] { seed = seed * 1664525U + 1013904223U; return float(int(seed >> 8) - 8388608) / 8388608.0f; };
        for (size_t i = 0; i < weights.elements; ++i) weights.data()[i] = random() * .03125f;
        for (size_t i = 0; i < inputs.elements; ++i) inputs.data()[i] = random();
        weight_fingerprint = weights.fingerprint(); input_fingerprint = inputs.fingerprint();
    }
    void check_inputs() const {
        if (!weights.guards() || !inputs.guards() || weights.fingerprint() != weight_fingerprint ||
            inputs.fingerprint() != input_fingerprint) throw std::runtime_error("Source operands or canaries changed");
    }
};

struct route {
    guarded values;
    ggml_context * context = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_t backend;
    ggml_backend_buffer_t weight_buffer = nullptr, input_buffer = nullptr, output_buffer = nullptr;
    ggml_tensor * output = nullptr;
    linux_blas_plan * blas_plan = nullptr;
    std::string name;
    route(session & owner, problem & p, linux_blas_plan * blas) :
        values(size_t(p.m) * p.n), backend(owner.backend), blas_plan(blas), name(blas ? "owned-lp64-openblas" : "unchanged-ggml-f32") {
        try {
            context = ggml_init({1024 * 1024, nullptr, true});
            if (!context) throw std::runtime_error("Cannot allocate graph metadata");
            auto * w = ggml_new_tensor_2d(context, GGML_TYPE_F32, p.k, p.m);
            auto * x = ggml_new_tensor_2d(context, GGML_TYPE_F32, p.k, p.n);
            ggml_set_input(w); ggml_set_input(x);
            output = blas ? tsg_embeddings::cpu_matmul(context, w, x, 1, blas) : ggml_mul_mat(context, w, x);
            if (!output) throw std::runtime_error("Required owned CPU matmul route rejected this graph");
            if (!blas) ggml_prec_set_acc(output, GGML_PREC_F32);
            ggml_set_output(output);
            weight_buffer = ggml_backend_cpu_buffer_from_ptr(p.weights.storage, p.weights.allocated * sizeof(float));
            input_buffer = ggml_backend_cpu_buffer_from_ptr(p.inputs.storage, p.inputs.allocated * sizeof(float));
            output_buffer = ggml_backend_cpu_buffer_from_ptr(values.storage, values.allocated * sizeof(float));
            if (!weight_buffer || !input_buffer || !output_buffer ||
                ggml_backend_tensor_alloc(weight_buffer, w, p.weights.data()) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(input_buffer, x, p.inputs.data()) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(output_buffer, output, values.data()) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Cannot bind guarded CPU graph operands");
            auto * graph = ggml_new_graph_custom(context, 32, false);
            ggml_build_forward_expand(graph, output);
            if (!ggml_backend_supports_op(backend, output)) throw std::runtime_error("CPU backend rejected graph output");
            plan = ggml_backend_graph_plan_create(backend, graph);
            if (!plan) throw std::runtime_error("Cannot create persistent graph plan");
        } catch (...) { release(); throw; }
    }
    void release() {
        if (plan) { ggml_backend_graph_plan_free(backend, plan); plan = nullptr; }
        ggml_backend_buffer_free(output_buffer); output_buffer = nullptr;
        ggml_backend_buffer_free(input_buffer); input_buffer = nullptr;
        ggml_backend_buffer_free(weight_buffer); weight_buffer = nullptr;
        if (context) { ggml_free(context); context = nullptr; }
    }
    ~route() { release(); }
    void compute() {
        if (ggml_backend_graph_plan_compute(backend, plan) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Persistent graph computation failed: " + name);
    }
};

struct quality {
    size_t sampled = 0;
    double max_absolute_error = 0, max_error_over_bound = 0;
};

quality check(const problem & p, const route & r, int threads) {
    if (!r.values.guards()) throw std::runtime_error("Output guard canary changed");
    for (size_t i = 0; i < r.values.elements; ++i)
        if (!std::isfinite(r.values.data()[i]) || r.values.data()[i] == guarded::canary)
            throw std::runtime_error("Output has a nonfinite or unwritten element");
    p.check_inputs();
    std::set<int> rows{0, 1, p.n / 2, p.n - 1}, columns{0, 1, 7, 15, 16, p.m / 2, p.m - 1};
    for (int worker = 1; worker < threads; ++worker)
        for (int offset : {-1, 0, 1}) columns.insert(p.m * worker / threads + offset);
    uint32_t state = 1729;
    for (int i = 0; i < 32; ++i) { state = state * 1664525U + 1013904223U; columns.insert(int(state % p.m)); }
    quality result;
    for (int row : rows) for (int column : columns) {
        if (row < 0 || row >= p.n || column < 0 || column >= p.m) continue;
        double expected = 0, absolute_products = 0;
        for (int inner = 0; inner < p.k; ++inner) {
            const double product = double(p.weights.data()[size_t(column) * p.k + inner]) * p.inputs.data()[size_t(row) * p.k + inner];
            expected += product; absolute_products += std::abs(product);
        }
        const double actual = r.values.data()[size_t(row) * p.m + column];
        const double error = std::abs(actual - expected), bound = std::max(2e-5, absolute_products * 1e-6);
        result.max_absolute_error = std::max(result.max_absolute_error, error);
        result.max_error_over_bound = std::max(result.max_error_over_bound, error / bound);
        ++result.sampled;
        if (error > bound) throw std::runtime_error("Independent double product gate failed: " + r.name);
    }
    return result;
}

struct timing {
    std::vector<double> samples;
    double wall_ms = 0, median = 0, p95 = 0;
    quality checked;
    uint64_t blas_gemm_calls = 0, blas_output_elements = 0, blas_worker_mask = 0;
};

timing measure(route & r, problem & p, const options & config) {
    const auto counts_before = r.blas_plan ? r.blas_plan->counts() : tsg_embeddings::linux_blas_counts{};
    // Refill outside timing, so every phase independently proves its output was
    // written instead of accidentally accepting a previous phase's values.
    std::fill(r.values.data(), r.values.data() + r.values.elements, guarded::canary);
    for (int i = 0; i < config.warmup; ++i) r.compute();
    check(p, r, config.threads);
    timing result;
    const auto start = clock_type::now();
    do {
        const auto before = clock_type::now();
        r.compute();
        result.samples.push_back(std::chrono::duration<double, std::milli>(clock_type::now() - before).count());
    } while (result.samples.size() < size_t(config.repeats) ||
        std::chrono::duration<double>(clock_type::now() - start).count() < config.minimum_seconds);
    result.wall_ms = std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
    result.checked = check(p, r, config.threads);
    auto sorted = result.samples; std::sort(sorted.begin(), sorted.end());
    const size_t middle = sorted.size() / 2;
    result.median = sorted.size() % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) * .5;
    result.p95 = sorted[size_t(std::ceil(.95 * sorted.size())) - 1];
    if (r.blas_plan) {
        const auto after = r.blas_plan->counts();
        const uint64_t calls = uint64_t(config.warmup) + result.samples.size();
        result.blas_gemm_calls = after.gemm_calls - counts_before.gemm_calls;
        result.blas_output_elements = after.output_elements - counts_before.output_elements;
        result.blas_worker_mask = after.worker_mask;
        if (result.blas_gemm_calls != calls * uint64_t(config.threads) ||
            result.blas_output_elements != calls * p.m * p.n ||
            after.output_channel_calls - counts_before.output_channel_calls != result.blas_gemm_calls ||
            after.token_row_calls != counts_before.token_row_calls ||
            after.empty_partitions != counts_before.empty_partitions ||
            after.shape_rejections != counts_before.shape_rejections ||
            after.policy_rejections != counts_before.policy_rejections || after.active_calls ||
            after.worker_mask != ((uint64_t(1) << config.threads) - 1))
            throw std::runtime_error("Owned BLAS did not execute every complete configured-worker projection partition");
    }
    return result;
}

void write_timing(std::ostream & out, const timing & result, const std::string & route_name, const char * phase) {
    out << "{\"phase\":" << quote(phase) << ",\"route\":" << quote(route_name)
        << ",\"median_ms\":" << result.median << ",\"p95_ms\":" << result.p95
        << ",\"measured_wall_ms\":" << result.wall_ms << ",\"actual_repeats\":" << result.samples.size()
        << ",\"sampled_double_outputs\":" << result.checked.sampled
        << ",\"double_max_absolute_error\":" << result.checked.max_absolute_error
        << ",\"max_error_over_bound\":" << result.checked.max_error_over_bound
        << ",\"blas_gemm_calls\":" << result.blas_gemm_calls
        << ",\"blas_output_elements\":" << result.blas_output_elements
        << ",\"blas_worker_mask\":" << result.blas_worker_mask
        << ",\"input_output_canaries_passed\":true,\"immutable_input_fingerprints_passed\":true,\"all_outputs_finite\":true,\"samples_ms\":[";
    for (size_t i = 0; i < result.samples.size(); ++i) { if (i) out << ','; out << result.samples[i]; }
    out << "]}";
}
} // namespace

int main(int argc, char ** argv) {
    options config;
    try { config = parse(argc, argv); }
    catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 2; }
    std::ofstream out(config.output);
    if (!out) { std::fprintf(stderr, "Cannot open --output\n"); return 2; }
    out << std::setprecision(17);
    linux_blas_plan blas(config.library.empty() ? nullptr : config.library.c_str(), config.threads);
    if (!blas.available()) {
        out << "{\"completed\":false,\"passed\":false,\"blas_available\":false,\"reason\":" << quote(blas.reason()) << "}\n";
        return 2;
    }
    out << "{\"blas_available\":true,\"provider\":" << quote(blas.library_path())
        << ",\"provider_config\":" << quote(blas.library_config()) << ",\"actual_blas_workers\":1"
        << ",\"ggml_threads\":" << config.threads << ",\"persistent_threadpool\":true,\"persistent_graph_plans\":true"
        << ",\"warmup_per_phase\":" << config.warmup << ",\"minimum_repeats_per_phase\":" << config.repeats
        << ",\"minimum_seconds_per_phase\":" << config.minimum_seconds
        << ",\"compiler\":" << quote(__VERSION__)
        << ",\"observed_environment\":{\"OPENBLAS_NUM_THREADS\":" << environment("OPENBLAS_NUM_THREADS")
        << ",\"TS_GGML_CPU_THREADS\":" << environment("TS_GGML_CPU_THREADS")
        << ",\"GOTO_NUM_THREADS\":" << environment("GOTO_NUM_THREADS")
        << ",\"OMP_NUM_THREADS\":" << environment("OMP_NUM_THREADS") << "},\"cases\":[";
    bool comma = false;
    try {
        session cpu(config.threads);
        struct family { const char * name; int k, m; uint32_t seed; };
        // Actual classic tower: hidden8*128=1024, FFN4096; two SSCP
        // stride2 convolutions map128mel bins to32*128=4096 columns.
        // FFN down and SSCP share shape but use separate synthetic operands.
        for (const auto & f : {family{"ffn-up", 1024, 4096, 101}, {"ffn-down", 4096, 1024, 211},
                              {"qkv-attention-output", 1024, 1024, 307}, {"sscp-projection", 4096, 1024, 401}})
        for (int n : {25, 48, 96}) {
            problem p(f.k, f.m, n, f.seed + uint32_t(n));
            route baseline(cpu, p, nullptr), candidate(cpu, p, &blas);
            std::fprintf(stderr, "audio projection %s K%d M%d N%d ABBA\n", f.name, f.k, f.m, n);
            const auto baseline_a = measure(baseline, p, config);
            const auto candidate_a = measure(candidate, p, config);
            const auto candidate_b = measure(candidate, p, config);
            const auto baseline_b = measure(baseline, p, config);
            const auto counts = blas.counts();
            if (counts.shape_rejections || counts.policy_rejections || counts.active_calls || counts.gemm_calls == 0)
                throw std::runtime_error("BLAS provider rejected graph work or left active calls");
            if (comma) out << ','; comma = true;
            out << "{\"family\":" << quote(f.name) << ",\"k\":" << f.k << ",\"m\":" << f.m << ",\"n\":" << n
                << ",\"weight_bytes\":" << p.weights.elements * sizeof(float)
                << ",\"blas_partition\":\"output-channels\",\"minimum_columns\":1,\"phases\":[";
            write_timing(out, baseline_a, baseline.name, "baseline-a"); out << ',';
            write_timing(out, candidate_a, candidate.name, "blas-a"); out << ',';
            write_timing(out, candidate_b, candidate.name, "blas-b"); out << ',';
            write_timing(out, baseline_b, baseline.name, "baseline-b");
            out << "],\"blas_over_ggml_ratios\":[" << candidate_a.median / baseline_a.median << ','
                << candidate_b.median / baseline_b.median << "]}";
            out.flush();
        }
        const auto counts = blas.counts();
        out << "],\"counts\":{\"gemm_calls\":" << counts.gemm_calls << ",\"worker_mask\":" << counts.worker_mask
            << ",\"output_elements\":" << counts.output_elements << ",\"shape_rejections\":" << counts.shape_rejections
            << ",\"policy_rejections\":" << counts.policy_rejections << ",\"active_calls\":" << counts.active_calls << "}"
            << ",\"quality_bound\":\"max(2e-5,1e-6*sum(abs(original F32 products))) against double accumulation; edges, partition boundaries and deterministic sampled columns\""
            << ",\"scope\":\"Synthetic audio-shaped learned projections on actual persistent ggml CPU graph plans; unchanged upstream mul_mat PREC_F32 versus owned cpu_matmul minimum_columns=1\""
            << ",\"limitations\":[\"Single repeated matrix fits a favorable warm-cache regime; no layer weight rotation.\","
               "\"Not actual checkpoint weights or activations. Shapes are from the classic published audio helper; convolution GLU1024x2048, convolution kernels, norms, activation, relative attention, frontend and text tower are excluded.\","
               "\"Timings include CPU graph-plan submission/completion only; loading, graph construction, source uploads, output readback and oracle checks excluded.\","
               "\"No full audio tower, HTTP, cross-engine latency parity or production default promotion is inferred.\"]"
            << ",\"completed\":true,\"passed\":true}\n";
        return 0;
    } catch (const std::exception & error) {
        out << "],\"completed\":false,\"passed\":false,\"error\":" << quote(error.what()) << "}\n";
        std::fprintf(stderr, "audio projection probe failed: %s\n", error.what()); return 1;
    }
}
