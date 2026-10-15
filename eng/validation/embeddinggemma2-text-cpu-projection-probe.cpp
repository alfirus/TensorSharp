// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone text projection experiment. No production or upstream changes.
// See embeddinggemma2-text-cpu-projection-probe.md for the arithmetic contract.
#include "embedding_cpu_matmul.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "embeddinggemma2-q16-projection-probe.h"
#include <cfenv>
#include <cstring>

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
    std::string output, library, model, baseline = "blas-policy", families = "qkv-swa,gate-up,down,ple", rows = "128,1024", routes = "blas-channels,blas-rows,q8-one-pass,q8-residual-corrected,ggml-f32";
    int passes = 2;
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
        else if (name == "--model") result.model = value;
        else if (name == "--families") result.families = value;
        else if (name == "--rows") result.rows = value;
        else if (name == "--passes") result.passes = std::stoi(value);
        else if (name == "--routes") result.routes = value;
        else if (name == "--baseline") result.baseline = value;
        else if (name == "--library") result.library = value;
        else if (name == "--threads") result.threads = std::stoi(value);
        else if (name == "--warmup") result.warmup = std::stoi(value);
        else if (name == "--repeats") result.repeats = std::stoi(value);
        else if (name == "--minimum-seconds") result.minimum_seconds = std::stod(value);
        else throw std::runtime_error("Unknown option: " + name);
    }
    if (result.output.empty() || result.model.empty() || result.passes < 2 || result.passes > 3 || result.threads < 1 || result.threads > 8 ||
        (result.baseline != "blas-policy" && result.baseline != "ggml-f32") ||
        result.warmup < 1 || result.warmup > 10000 || result.repeats < 1 || result.repeats > 10000 ||
        !std::isfinite(result.minimum_seconds) || result.minimum_seconds < 0 || result.minimum_seconds > 60)
        throw std::runtime_error("Require --output, --model, passes 2 or 3, threads 1..8, warmup 1..10000, repeats 1..10000, minimum-seconds 0..60");
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

std::vector<std::string> split(const std::string & text) {
    std::vector<std::string> result;
    std::istringstream input(text); std::string item;
    while (std::getline(input, item, ',')) { if (item.empty()) throw std::runtime_error("Empty list item"); result.push_back(item); }
    if (result.empty()) throw std::runtime_error("Empty list");
    return result;
}

struct checkpoint {
    gguf_context * metadata = nullptr;
    std::ifstream file;
    explicit checkpoint(const std::string & path) : file(path, std::ios::binary) {
        metadata = gguf_init_from_file(path.c_str(), {true, nullptr});
        if (!metadata || !file) {
            if (metadata) { gguf_free(metadata); metadata = nullptr; }
            throw std::runtime_error("Cannot read GGUF metadata/model");
        }
    }
    ~checkpoint() { if (metadata) gguf_free(metadata); }
    std::vector<unsigned char> load(const std::string & name, int & k, int & m, ggml_type & type) {
        const auto id = gguf_find_tensor(metadata, name.c_str());
        if (id < 0) throw std::runtime_error("Required checkpoint matrix missing: " + name);
        type = gguf_get_tensor_type(metadata, id);
        if (type != GGML_TYPE_F32 && !ggml_get_type_traits(type)->to_float)
            throw std::runtime_error("Checkpoint type cannot be dequantized: " + name);
        const auto * shape = gguf_get_tensor_ne(metadata, id);
        if (shape[0] <= 0 || shape[0] > 65536 || shape[0] % 32 || shape[1] <= 0 || shape[1] > 65536 || shape[2] != 1 || shape[3] != 1)
            throw std::runtime_error("Unsupported experiment matrix geometry");
        k = int(shape[0]); m = int(shape[1]);
        std::vector<unsigned char> bytes(gguf_get_tensor_size(metadata, id));
        file.seekg(std::streamoff(gguf_get_data_offset(metadata) + gguf_get_tensor_offset(metadata, id)));
        if (!file.read(reinterpret_cast<char *>(bytes.data()), std::streamsize(bytes.size())))
            throw std::runtime_error("Truncated checkpoint matrix");
        return bytes;
    }
};

uint64_t fingerprint(const std::vector<unsigned char> & bytes) {
    uint64_t result = 14695981039346656037ULL;
    for (auto byte : bytes) result = (result ^ byte) * 1099511628211ULL;
    return result;
}

std::vector<std::string> tensors(const std::string & family) {
    if (family == "qkv-swa") return {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"};
    if (family == "qkv-global") return {"blk.5.attn_q.weight", "blk.5.attn_k.weight", "blk.5.attn_v.weight"};
    if (family == "gate-up") return {"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"};
    if (family == "down") return {"blk.0.ffn_down.weight"};
    if (family == "attn-swa") return {"blk.0.attn_output.weight"};
    if (family == "attn-global") return {"blk.5.attn_output.weight"};
    if (family == "inp-gate") return {"blk.0.inp_gate.weight"};
    if (family == "proj") return {"blk.0.proj.weight"};
    if (family == "ple") return {"per_layer_model_proj.weight"};
    throw std::runtime_error("Unknown text projection family: " + family);
}

struct problem {
    int k, m = 0, n;
    std::vector<unsigned char> original;
    ggml_type original_type = GGML_TYPE_COUNT;
    std::unique_ptr<guarded> weights, encoded;
    guarded inputs;
    uint64_t weight_fingerprint, original_fingerprint, encoded_fingerprint, input_fingerprint;
    problem(checkpoint & source, const std::string & family, int rows, uint32_t seed) :
        k(0), n(rows), inputs(size_t(rows) * (family == "down" ? 2048 : family == "attn-swa" ? 1024 : family == "attn-global" ? 2048 : 512)) {
        for (const auto & name : tensors(family)) {
            int inner, columns; ggml_type type; auto bytes = source.load(name, inner, columns, type);
            if (k && (k != inner || original_type != type)) throw std::runtime_error("Fused checkpoint matrix geometry/types disagree");
            k = inner; m += columns; original_type = type; original.insert(original.end(), bytes.begin(), bytes.end());
        }
        if (inputs.elements != size_t(k) * n) throw std::runtime_error("Input allocation geometry disagrees with checkpoint");
        weights = std::make_unique<guarded>(size_t(k) * m);
        encoded = std::make_unique<guarded>((original.size() + sizeof(float) - 1) / sizeof(float));
        std::memcpy(encoded->data(), original.data(), original.size());
        encoded_fingerprint = encoded->fingerprint();
        auto * traits = ggml_get_type_traits(original_type);
        if (original_type == GGML_TYPE_F32) std::memcpy(weights->data(), original.data(), original.size());
        else traits->to_float(original.data(), weights->data(), int64_t(k) * m);
        auto random = [&] { seed = seed * 1664525U + 1013904223U; return float(int(seed >> 8) - 8388608) / 8388608.0f; };
        for (size_t i = 0; i < inputs.elements; ++i) inputs.data()[i] = random();
        // One deterministic cancellation-sensitive row exercises correction;
        // remaining rows have bounded RMS-like synthetic input, not captured model states.
        for (int i = 0; i < k; ++i) inputs.data()[i] = (i % 2 ? -1.0f : 1.0f) * (1.0f + float(i % 31) * .00017f);
        weight_fingerprint = weights->fingerprint(); original_fingerprint = fingerprint(original); input_fingerprint = inputs.fingerprint();
    }
    void check_inputs() const {
        if (!weights->guards() || !encoded->guards() || encoded->fingerprint() != encoded_fingerprint || !inputs.guards() || weights->fingerprint() != weight_fingerprint ||
            fingerprint(original) != original_fingerprint || inputs.fingerprint() != input_fingerprint)
            throw std::runtime_error("Source operands or canaries changed");
    }
};

struct residual_workspace {
    std::vector<std::vector<unsigned char>> quantized;
    std::vector<std::vector<float>> decoded;
    residual_workspace(int k, int workers) : quantized(workers), decoded(workers) {
        for (int i = 0; i < workers; ++i) { quantized[i].resize(ggml_row_size(GGML_TYPE_Q8_0, k)); decoded[i].resize(k); }
    }
};

void residual(ggml_tensor * output, int ith, int nth, void * userdata) {
    auto & workspace = *static_cast<residual_workspace *>(userdata);
    const auto * x = output->src[0];
    const int k = int(x->ne[0]);
    auto * q = workspace.quantized.at(ith).data(); auto * decoded = workspace.decoded.at(ith).data();
    auto quantize = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    auto dequantize = ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float;
    for (int64_t row = ith; row < x->ne[1]; row += nth) {
        const auto * source = static_cast<const float *>(x->data) + row * k;
        auto * target = static_cast<float *>(output->data) + row * k;
        quantize(source, q, k); dequantize(q, decoded, k);
        for (int column = 0; column < k; ++column) target[column] = source[column] - decoded[column];
    }
}

struct axis_workspace { linux_blas_plan * plan; tsg_embeddings::blas_partition axis; };
void axis_matmul(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & state = *static_cast<axis_workspace *>(userdata);
    const auto * w = dst->src[0]; const auto * x = dst->src[1];
    const tsg_embeddings::blas_const_matrix weights{static_cast<const float *>(w->data), ggml_nbytes(w) / 4, w->ne[1], w->ne[0], int64_t(w->nb[1] / 4)};
    const tsg_embeddings::blas_const_matrix input{static_cast<const float *>(x->data), ggml_nbytes(x) / 4, x->ne[1], x->ne[0], int64_t(x->nb[1] / 4)};
    const tsg_embeddings::blas_matrix output{static_cast<float *>(dst->data), ggml_nbytes(dst) / 4, dst->ne[1], dst->ne[0], int64_t(dst->nb[1] / 4)};
    if (!state.plan->try_partition(weights, input, output, ith, nth, state.axis)) std::abort();
}

struct route {
    guarded values;
    ggml_context * context = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_t backend;
    ggml_backend_buffer_t weight_buffer = nullptr, input_buffer = nullptr, output_buffer = nullptr, intermediates = nullptr;
    ggml_tensor * output = nullptr;
    linux_blas_plan * blas_plan = nullptr;
    std::string name;
    residual_workspace scratch;
    axis_workspace axis;
    std::unique_ptr<tsg_validation::q16_workspace> q16;
    int passes;
    route(session & owner, problem & p, linux_blas_plan * blas, const std::string & kind, int correction_passes = 0) :
        values(size_t(p.m) * p.n), backend(owner.backend), blas_plan(kind == "ggml-f32" || kind == "q8-q16-vnni" ? nullptr : blas), name(kind), scratch(p.k, 8),
        axis{blas, kind == "blas-channels" ? tsg_embeddings::blas_partition::output_channels : tsg_embeddings::blas_partition::token_rows},
        passes(correction_passes) {
        try {
            if (passes && p.original_type != GGML_TYPE_Q8_0) throw std::runtime_error("Q8 route requires original Q8 weights; requantization is forbidden");
            if (kind == "q8-q16-vnni") {
                if (p.original_type != GGML_TYPE_Q8_0 || !tensorcpu_q16_vnni_supported() || std::fegetround() != FE_TONEAREST)
                    throw std::runtime_error("Q16 route requires original Q8 weights, qualified VNNI and nearest-even rounding");
                q16 = std::make_unique<tsg_validation::q16_workspace>(p.original, p.k, p.m, p.n);
            }
            context = ggml_init({1024 * 1024, nullptr, true});
            if (!context) throw std::runtime_error("Cannot allocate graph metadata");
            auto * w = ggml_new_tensor_2d(context, passes || q16 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32, p.k, p.m);
            auto * x = ggml_new_tensor_2d(context, GGML_TYPE_F32, p.k, p.n);
            ggml_set_input(w); ggml_set_input(x);
            auto * first_x = x;
            if (q16) {
                ggml_tensor * prepare_args[] = {x};
                auto * prepared = ggml_custom_4d(context, GGML_TYPE_F32, blas->worker_budget(), 1, 1, 1,
                    prepare_args, 1, tsg_validation::q16_prepare, blas->worker_budget(), q16.get());
                ggml_tensor * args[] = {w, x, prepared};
                output = ggml_custom_4d(context, GGML_TYPE_F32, p.m, p.n, 1, 1,
                    args, 3, tsg_validation::q16_project, blas->worker_budget(), q16.get());
            } else if (passes) {
                output = ggml_mul_mat(context, w, x);
                for (int pass = 1; pass < passes; ++pass) {
                    ggml_tensor * args[] = {x};
                    x = ggml_custom_4d(context, GGML_TYPE_F32, p.k, p.n, 1, 1, args, 1, residual, blas->worker_budget(), &scratch);
                    output = ggml_add(context, output, ggml_mul_mat(context, w, x));
                }
            } else if (kind == "blas-policy") output = tsg_embeddings::cpu_matmul(context, w, x, 1, blas);
            else if (kind == "ggml-f32") output = ggml_mul_mat(context, w, x);
            else {
                ggml_tensor * args[] = {w, x};
                output = ggml_custom_4d(context, GGML_TYPE_F32, p.m, p.n, 1, 1, args, 2, axis_matmul, blas->worker_budget(), &axis);
            }
            if (!output) throw std::runtime_error("Required route rejected this graph");
            ggml_set_output(output);
            void * weight_data = passes || q16 ? static_cast<void *>(p.encoded->data()) : p.weights->data();
            void * weight_base = passes || q16 ? static_cast<void *>(p.encoded->storage) : p.weights->storage;
            size_t weight_size = passes || q16 ? p.encoded->allocated * sizeof(float) : p.weights->allocated * sizeof(float);
            weight_buffer = ggml_backend_cpu_buffer_from_ptr(weight_base, weight_size);
            input_buffer = ggml_backend_cpu_buffer_from_ptr(p.inputs.storage, p.inputs.allocated * sizeof(float));
            output_buffer = ggml_backend_cpu_buffer_from_ptr(values.storage, values.allocated * sizeof(float));
            if (!weight_buffer || !input_buffer || !output_buffer ||
                ggml_backend_tensor_alloc(weight_buffer, w, weight_data) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(input_buffer, first_x, p.inputs.data()) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(output_buffer, output, values.data()) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Cannot bind guarded CPU graph operands");
            if (passes > 1 || q16) {
                intermediates = ggml_backend_alloc_ctx_tensors(context, backend);
                if (!intermediates) throw std::runtime_error("Cannot allocate correction intermediates");
            }
            auto * graph = ggml_new_graph_custom(context, 64, false);
            ggml_build_forward_expand(graph, output);
            if (!ggml_backend_supports_op(backend, output)) throw std::runtime_error("CPU backend rejected graph output");
            plan = ggml_backend_graph_plan_create(backend, graph);
            if (!plan) throw std::runtime_error("Cannot create persistent graph plan");
        } catch (...) { release(); throw; }
    }
    void release() {
        if (plan) { ggml_backend_graph_plan_free(backend, plan); plan = nullptr; }
        ggml_backend_buffer_free(intermediates); intermediates = nullptr;
        ggml_backend_buffer_free(output_buffer); output_buffer = nullptr;
        ggml_backend_buffer_free(input_buffer); input_buffer = nullptr;
        ggml_backend_buffer_free(weight_buffer); weight_buffer = nullptr;
        if (context) { ggml_free(context); context = nullptr; }
    }
    ~route() { release(); }
    void compute() {
        if (ggml_backend_graph_plan_compute(backend, plan) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Persistent graph computation failed: " + name);
        if (q16 && q16->error.load()) throw std::runtime_error("Q16 custom-node callback failed");
    }
};

struct quality {
    size_t sampled = 0;
    double max_absolute_error = 0, max_error_over_bound = 0;
    bool passed = true;
    bool q16_integer_oracle_passed = true;
    double q16_integer_max_error = 0;
};

quality check(const problem & p, const route & r, int threads) {
    if (!r.values.guards()) throw std::runtime_error("Output guard canary changed");
    for (size_t i = 0; i < r.values.elements; ++i)
        if (!std::isfinite(r.values.data()[i]) || r.values.data()[i] == guarded::canary)
            throw std::runtime_error("Output has a nonfinite or unwritten element");
    p.check_inputs();
    if (r.q16) r.q16->check_storage();
    std::set<int> rows{0, 1, p.n / 2, p.n - 1}, columns{0, 1, 7, 15, 16, p.m / 2, p.m - 1};
    for (int worker = 1; worker < threads; ++worker)
        for (int offset : {-1, 0, 1}) {
            columns.insert(p.m * worker / threads + offset);
            rows.insert(p.n * worker / threads + offset);
        }
    uint32_t state = 1729;
    for (int i = 0; i < 32; ++i) { state = state * 1664525U + 1013904223U; columns.insert(int(state % p.m)); }
    quality result;
    for (int row : rows) for (int column : columns) {
        if (row < 0 || row >= p.n || column < 0 || column >= p.m) continue;
        double expected = 0, absolute_products = 0;
        for (int inner = 0; inner < p.k; ++inner) {
            const double product = double(p.weights->data()[size_t(column) * p.k + inner]) * p.inputs.data()[size_t(row) * p.k + inner];
            expected += product; absolute_products += std::abs(product);
        }
        const double actual = r.values.data()[size_t(row) * p.m + column];
        const double error = std::abs(actual - expected), bound = std::max(2e-5, absolute_products * 1e-6);
        result.max_absolute_error = std::max(result.max_absolute_error, error);
        result.max_error_over_bound = std::max(result.max_error_over_bound, error / bound);
        ++result.sampled;
        if (error > bound) result.passed = false;
        if (r.q16) {
            // Independently reconstruct the original Q8 x scalar-quantized
            // Q16 product, without reading the candidate's packed weights,
            // activation integers, block scales or byte shuffle.
            float exact_q16 = 0;
            bool fallback = false;
            for (int block = 0; block < p.k / 32; ++block) {
                const auto * input = p.inputs.data() + size_t(row) * p.k + block * 32;
                const size_t encoded = (size_t(column) * (p.k / 32) + block) * 34;
                ggml_fp16_t bits; std::memcpy(&bits, p.original.data() + encoded, sizeof(bits));
                const float weight_scale = ggml_fp16_to_fp32(bits);
                float maximum = 0;
                for (int dim = 0; dim < 32; ++dim) maximum = std::max(maximum, std::abs(input[dim]));
                const float scale = maximum / 32767.f, inverse = scale == 0 ? 0 : 1.f / scale;
                fallback |= maximum > 0 && scale < 1.17549435e-38f;
                int32_t dot = 0;
                for (int dim = 0; dim < 32; ++dim) {
                    const float normalized = std::isfinite(inverse) ? input[dim] * inverse : input[dim] / scale;
                    const int quant = std::clamp(int(std::nearbyint(normalized)), -32768, 32767);
                    dot += quant * static_cast<int8_t>(p.original[encoded + 2 + dim]);
                }
                exact_q16 = std::fma(float(dot) * weight_scale, scale, exact_q16);
            }
            if (fallback) {
                exact_q16 = 0;
                for (int inner = 0; inner < p.k; ++inner)
                    exact_q16 = std::fma(p.inputs.data()[size_t(row) * p.k + inner], p.weights->data()[size_t(column) * p.k + inner], exact_q16);
            }
            const double integer_error = std::abs(double(exact_q16) - actual);
            result.q16_integer_max_error = std::max(result.q16_integer_max_error, integer_error);
            const float actual_f32 = float(actual);
            result.q16_integer_oracle_passed &= std::memcmp(&exact_q16, &actual_f32, sizeof(float)) == 0;
        }
    }
    result.passed &= result.q16_integer_oracle_passed;
    return result;
}

struct timing {
    std::vector<double> samples;
    double wall_ms = 0, median = 0, p95 = 0;
    quality checked;
    uint64_t blas_gemm_calls = 0, blas_output_elements = 0, blas_worker_mask = 0;
    uint64_t q16_prepared_rows = 0, q16_executed_groups = 0, q16_fallback_rows = 0;
};

timing measure(route & r, problem & p, const options & config) {
    const auto counts_before = r.blas_plan && !r.passes ? r.blas_plan->counts() : tsg_embeddings::linux_blas_counts{};
    const uint64_t q16_rows_before = r.q16 ? r.q16->prepared_rows.load() : 0;
    const uint64_t q16_groups_before = r.q16 ? r.q16->executed_groups.load() : 0;
    const uint64_t q16_fallback_before = r.q16 ? r.q16->fallback_rows.load() : 0;
    // Refill outside timing, so every phase independently proves its output was
    // written instead of accidentally accepting a previous phase's values.
    std::fill(r.values.data(), r.values.data() + r.values.elements, guarded::canary);
    for (int i = 0; i < config.warmup; ++i) r.compute();
    const auto warmup_quality = check(p, r, config.threads);
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
    result.checked.passed = result.checked.passed && warmup_quality.passed;
    result.checked.sampled += warmup_quality.sampled;
    result.checked.max_absolute_error = std::max(result.checked.max_absolute_error, warmup_quality.max_absolute_error);
    result.checked.max_error_over_bound = std::max(result.checked.max_error_over_bound, warmup_quality.max_error_over_bound);
    result.checked.q16_integer_oracle_passed &= warmup_quality.q16_integer_oracle_passed;
    result.checked.q16_integer_max_error = std::max(result.checked.q16_integer_max_error, warmup_quality.q16_integer_max_error);
    auto sorted = result.samples; std::sort(sorted.begin(), sorted.end());
    const size_t middle = sorted.size() / 2;
    result.median = sorted.size() % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) * .5;
    result.p95 = sorted[size_t(std::ceil(.95 * sorted.size())) - 1];
    if (r.blas_plan && !r.passes) {
        const auto after = r.blas_plan->counts();
        const uint64_t calls = uint64_t(config.warmup) + result.samples.size();
        result.blas_gemm_calls = after.gemm_calls - counts_before.gemm_calls;
        result.blas_output_elements = after.output_elements - counts_before.output_elements;
        result.blas_worker_mask = after.worker_mask;
        if (result.blas_gemm_calls != calls * uint64_t(config.threads) ||
            result.blas_output_elements != calls * p.m * p.n ||
            after.empty_partitions != counts_before.empty_partitions || after.shape_rejections != counts_before.shape_rejections ||
            after.policy_rejections != counts_before.policy_rejections || after.active_calls ||
            after.worker_mask != ((uint64_t(1) << config.threads) - 1))
            throw std::runtime_error("BLAS did not execute complete configured-worker projection partitions");
    }
    if (r.q16) {
        const uint64_t calls = uint64_t(config.warmup) + result.samples.size();
        result.q16_prepared_rows = r.q16->prepared_rows.load() - q16_rows_before;
        result.q16_executed_groups = r.q16->executed_groups.load() - q16_groups_before;
        result.q16_fallback_rows = r.q16->fallback_rows.load() - q16_fallback_before;
        if (result.q16_prepared_rows != calls * p.n || result.q16_executed_groups != calls * ((p.m + 63) / 64) * ((p.n + 15) / 16))
            throw std::runtime_error("Q16 did not prepare and project complete configured shapes");
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
        << ",\"independent_product_gate_passed\":" << (result.checked.passed ? "true" : "false")
        << ",\"blas_gemm_calls\":" << result.blas_gemm_calls
        << ",\"blas_output_elements\":" << result.blas_output_elements
        << ",\"blas_worker_mask\":" << result.blas_worker_mask
        << ",\"q16_prepared_rows\":" << result.q16_prepared_rows << ",\"q16_executed_groups\":" << result.q16_executed_groups
        << ",\"q16_f32_fallback_rows\":" << result.q16_fallback_rows
        << ",\"q16_exact_integer_oracle\":" << (route_name == "q8-q16-vnni" ? (result.checked.q16_integer_oracle_passed ? "true" : "false") : "null")
        << ",\"q16_integer_max_error\":" << result.checked.q16_integer_max_error
        << ",\"input_output_canaries_passed\":true,\"immutable_input_fingerprints_passed\":true,\"all_outputs_finite\":true,\"samples_ms\":[";
    for (size_t i = 0; i < result.samples.size(); ++i) { if (i) out << ','; out << result.samples[i]; }
    out << "]}";
}
} // namespace

int main(int argc, char ** argv) {
    options config;
    try { config = parse(argc, argv); }
    catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 2; }
    std::ostringstream out;
    out << std::setprecision(17);
    std::vector<std::string> selected_routes;
    try { selected_routes = split(config.routes); }
    catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 2; }
    linux_blas_plan blas(config.library.empty() ? nullptr : config.library.c_str(), config.threads);
    const bool requires_blas = config.baseline == "blas-policy" || std::any_of(selected_routes.begin(), selected_routes.end(), [](const std::string & route) {
        return route == "blas-channels" || route == "blas-rows";
    });
    if (!blas.available() && requires_blas) {
        out << "{\"completed\":false,\"passed\":false,\"blas_available\":false,\"reason\":" << quote(blas.reason()) << "}\n";
        std::ofstream(config.output) << out.str();
        return 2;
    }
    out << "{\"blas_available\":" << (blas.available() ? "true" : "false") << ",\"provider\":" << quote(blas.library_path())
        << ",\"provider_config\":" << quote(blas.library_config()) << ",\"actual_blas_workers\":" << (blas.available() ? "1" : "null")
        << ",\"baseline_route\":" << quote(config.baseline) << ",\"q16_vnni_available\":" << (tensorcpu_q16_vnni_supported() ? "true" : "false")
        << ",\"ggml_threads\":" << config.threads << ",\"persistent_threadpool\":true,\"persistent_graph_plans\":true"
        << ",\"warmup_per_phase\":" << config.warmup << ",\"minimum_repeats_per_phase\":" << config.repeats
        << ",\"minimum_seconds_per_phase\":" << config.minimum_seconds
        << ",\"model_path\":" << quote(config.model)
        << ",\"compiler\":" << quote(__VERSION__)
        << ",\"observed_environment\":{\"OPENBLAS_NUM_THREADS\":" << environment("OPENBLAS_NUM_THREADS")
        << ",\"TS_GGML_CPU_THREADS\":" << environment("TS_GGML_CPU_THREADS")
        << ",\"GOTO_NUM_THREADS\":" << environment("GOTO_NUM_THREADS")
        << ",\"OMP_NUM_THREADS\":" << environment("OMP_NUM_THREADS") << "},\"cases\":[";
    bool comma = false;
    try {
        session cpu(config.threads);
        checkpoint source(config.model);
        bool all_precise = true; size_t corrected_executed = 0, corrected_unavailable = 0, precise_comparisons_executed = 0;
        for (const auto & family : split(config.families)) for (const auto & row_text : split(config.rows)) {
            size_t consumed = 0; const int n = std::stoi(row_text, &consumed);
            if (consumed != row_text.size() || n < config.threads || n > 8192) throw std::runtime_error("Rows must be complete integers in threads..8192");
            problem p(source, family, n, 101 + uint32_t(n));
            route baseline(cpu, p, &blas, config.baseline);
            std::fprintf(stderr, "text projection %s K%d M%d N%d original bytes %zu\n", family.c_str(), p.k, p.m, p.n, p.original.size());
            if (comma) out << ','; comma = true;
            out << "{\"family\":" << quote(family) << ",\"k\":" << p.k << ",\"m\":" << p.m << ",\"n\":" << n
                << ",\"original_type\":" << quote(ggml_type_name(p.original_type))
                << ",\"original_bytes\":" << p.original.size() << ",\"original_fnv1a64\":" << quote(std::to_string(p.original_fingerprint))
                << ",\"f32_weight_bytes\":" << p.weights->elements * sizeof(float) << ",\"comparisons\":[";
            bool inner_comma = false;
            for (const auto & kind : selected_routes) {
                if (kind != "blas-channels" && kind != "blas-rows" && kind != "q8-one-pass" && kind != "q8-residual-corrected" && kind != "ggml-f32" && kind != "q8-q16-vnni")
                    throw std::runtime_error("Unknown route: " + kind);
                const int passes = kind == "q8-one-pass" ? 1 : kind == "q8-residual-corrected" ? config.passes : 0;
                if ((passes || kind == "q8-q16-vnni") && p.original_type != GGML_TYPE_Q8_0) {
                    if (inner_comma) out << ','; inner_comma = true;
                    out << "{\"route\":" << quote(kind) << ",\"executed\":false,\"skipped\":true,\"reason\":\"Original matrix is not Q8_0; no weight requantization is permitted.\"}";
                    if (kind == "q8-residual-corrected") ++corrected_unavailable;
                    continue;
                }
                if (kind == "q8-q16-vnni" && !tensorcpu_q16_vnni_supported()) {
                    if (inner_comma) out << ','; inner_comma = true;
                    out << "{\"route\":" << quote(kind) << ",\"executed\":false,\"skipped\":true,\"reason\":\"Owned AVX512 VNNI kernel unavailable on the current device/build.\"}";
                    continue;
                }
                if (kind == "q8-residual-corrected") ++corrected_executed;
                if (kind != "q8-one-pass") ++precise_comparisons_executed;
                route candidate(cpu, p, &blas, kind, passes);
                const auto baseline_a = measure(baseline, p, config), candidate_a = measure(candidate, p, config);
                const auto candidate_b = measure(candidate, p, config), baseline_b = measure(baseline, p, config);
                if (kind != "q8-one-pass") all_precise = all_precise && candidate_a.checked.passed && candidate_b.checked.passed;
                all_precise = all_precise && baseline_a.checked.passed && baseline_b.checked.passed;
                if (inner_comma) out << ','; inner_comma = true;
                out << "{\"route\":" << quote(kind) << ",\"executed\":true,\"skipped\":false,\"q8_passes\":" << passes << ",\"phases\":[";
                write_timing(out, baseline_a, baseline.name, "baseline-a"); out << ',';
                write_timing(out, candidate_a, candidate.name, "candidate-a"); out << ',';
                write_timing(out, candidate_b, candidate.name, "candidate-b"); out << ',';
                write_timing(out, baseline_b, baseline.name, "baseline-b");
                out << "],\"candidate_over_baseline_ratios\":[" << candidate_a.median / baseline_a.median << ',' << candidate_b.median / baseline_b.median << ']';
                if (candidate.q16) out << ",\"q16_original_weight_requantized\":false,\"q16_prepacked_weight_bytes\":" << candidate.q16->weights.bytes
                    << ",\"q16_prepacked_weight_fnv1a64\":" << quote(std::to_string(candidate.q16->weights_hash))
                    << ",\"q16_weight_sum_bytes\":" << candidate.q16->weight_sums.bytes
                    << ",\"q16_weight_scale_bytes\":" << candidate.q16->weight_scales.bytes
                    << ",\"q16_private_activation_scratch_bytes\":" << candidate.q16->activations.bytes + candidate.q16->packed.bytes + candidate.q16->scales.bytes + candidate.q16->fallback.bytes;
                out << '}';
                out.flush();
            }
            out << "]}";
        }
        const bool passed = all_precise && precise_comparisons_executed > 0;
        out << "],\"quality_bound\":\"max(2e-5,1e-6*sum(abs(original source dequantized F32 weight times original F32 activation products))) against independent double accumulation\""
            << ",\"scope\":\"Actual checkpoint weights, synthetic unquantized F32 inputs; Q8 correction only for original Q8_0 tensors, other types explicitly unavailable; persistent unchanged ggml CPU graph plans. Residual correction and output addition are explicit TensorSharp-owned graph construction.\""
            << ",\"limitations\":[\"One matrix reused in each shape; no layer rotation and favorable warm-cache regime.\",\"Synthetic activations; full-model strict quality and service performance remain unqualified.\",\"One-pass Q8 is a diagnostic and excluded from overall product-quality success.\",\"The selected baseline route is explicit; optional OpenBLAS policy includes rows below production minimum32 when selected.\",\"Graph submission, activation quantization/packing, residual construction and output addition are timed where applicable; graph construction, weight prepacking, checkpoint load and oracle checks excluded.\",\"Q16 reuses unchanged owned two-dot native group math through the existing GGML worker team; exact integer agreement does not replace the original F32-product forward-error gate.\"]"
            << ",\"precise_comparisons_executed\":" << precise_comparisons_executed
            << ",\"corrected_projection_cases_executed\":" << corrected_executed
            << ",\"corrected_projection_cases_unavailable\":" << corrected_unavailable
            << ",\"corrected_passes\":" << config.passes << ",\"completed\":true,\"passed\":" << (passed ? "true" : "false") << "}\n";
        std::ofstream file(config.output);
        if (!file || !(file << out.str())) throw std::runtime_error("Cannot write report");
        return passed ? 0 : 1;
    } catch (const std::exception & error) {
        std::ofstream(config.output) << "{\"completed\":false,\"passed\":false,\"error\":" << quote(error.what()) << "}\n";
        std::fprintf(stderr, "text projection probe failed: %s\n", error.what()); return 1;
    }
}
