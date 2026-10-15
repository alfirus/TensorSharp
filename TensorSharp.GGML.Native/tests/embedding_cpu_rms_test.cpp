// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "embedding_cpu_rms.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace {
constexpr float epsilon = 1e-6f;
void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
struct guarded {
    static constexpr size_t border = 64;
    static constexpr float canary = 9876.5f;
    float * storage;
    size_t count, allocated;
    explicit guarded(size_t n) : count(n), allocated((n + 2 * border + 15) / 16 * 16) {
#if defined(_WIN32)
        storage = static_cast<float *>(_aligned_malloc(allocated * sizeof(float), 64));
#else
        storage = static_cast<float *>(std::aligned_alloc(64, allocated * sizeof(float)));
#endif
        if (!storage) throw std::bad_alloc();
        std::fill(storage, storage + allocated, canary);
    }
    ~guarded() {
#if defined(_WIN32)
        _aligned_free(storage);
#else
        std::free(storage);
#endif
    }
    guarded(const guarded &) = delete;
    float * data() const { return storage + border; }
    bool guards() const {
        for (size_t i = 0; i < border; ++i) if (storage[i] != canary) return false;
        for (size_t i = border + count; i < allocated; ++i) if (storage[i] != canary) return false;
        return true;
    }
    uint64_t hash() const {
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
        backend = ggml_backend_cpu_init(); require(backend, "CPU backend unavailable");
        auto params = ggml_threadpool_params_default(threads);
        pool = ggml_threadpool_new(&params);
        if (!pool) { ggml_backend_free(backend); backend = nullptr; throw std::runtime_error("CPU pool unavailable"); }
        ggml_backend_cpu_set_n_threads(backend, threads);
        ggml_backend_cpu_set_threadpool(backend, pool);
        ggml_backend_cpu_set_use_ref(backend, false);
    }
    ~session() { if (backend) ggml_backend_free(backend); if (pool) ggml_threadpool_free(pool); }
};
struct fixture {
    int width, heads, tokens, head_stride, token_stride, offset;
    bool weighted;
    guarded input, weights;
    fixture(int d, int h, int n, bool w, int head_padding = 0, int token_padding = 0) :
        width(d), heads(h), tokens(n), head_stride(d + head_padding),
        token_stride((h + 3) * head_stride + token_padding), offset(head_stride), weighted(w),
        input(size_t(token_stride) * n), weights(d) {
        for (int i = 0; i < d; ++i) weights.data()[i] = .83f + .19f * std::sin(float(i) * .17f);
    }
    const float * row(int token, int head) const { return input.data() + offset + token * token_stride + head * head_stride; }
    void fill(int kind, int iteration) {
        for (size_t i = 0; i < input.count; ++i) {
            float x = std::sin(float(i * 13 % 16381) * .17f) * (iteration ? .713f : 1.13f);
            if (kind == 1) x = i % 2 ? 0.0f : -0.0f;
            if (kind == 2) x *= 1e15f;
            if (kind == 3) x *= 1e-39f;
            input.data()[i] = x;
        }
    }
};
struct route {
    ggml_backend_t backend;
    ggml_context * ctx = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_buffer_t input_buffer = nullptr, weight_buffer = nullptr, output_buffer = nullptr, intermediates = nullptr;
    guarded output_values;
    float stable_epsilon = epsilon;
    bool owned;
    route(session & s, fixture & f, bool candidate, bool expected) : backend(s.backend),
        output_values(size_t(f.width) * f.heads * f.tokens), owned(candidate) {
        try {
            ctx = ggml_init({1024 * 1024, nullptr, true}); require(ctx, "RMS metadata allocation failed");
            auto * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, f.input.count);
            auto * weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, f.width);
            input_buffer = ggml_backend_cpu_buffer_from_ptr(f.input.storage, f.input.allocated * sizeof(float));
            weight_buffer = ggml_backend_cpu_buffer_from_ptr(f.weights.storage, f.weights.allocated * sizeof(float));
            require(input_buffer && weight_buffer, "RMS host operand binding failed");
            require(ggml_backend_tensor_alloc(input_buffer, base, f.input.data()) == GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(weight_buffer, weights, f.weights.data()) == GGML_STATUS_SUCCESS, "RMS operand allocation failed");
            auto * input = ggml_view_3d(ctx, base, f.width, f.heads, f.tokens,
                size_t(f.head_stride) * sizeof(float), size_t(f.token_stride) * sizeof(float), size_t(f.offset) * sizeof(float));
            require(ggml_backend_view_init(input) == GGML_STATUS_SUCCESS, "RMS packed view initialization failed");
            ggml_set_input(base); ggml_set_input(weights);
            auto * output = candidate ? tsg_embeddings::cpu_qkv_rms(ctx, input, f.weighted ? weights : nullptr, &stable_epsilon) : nullptr;
            if (candidate) require(bool(output) == expected, "RMS factory selection differs");
            auto * rms = output ? nullptr : ggml_rms_norm(ctx, input, stable_epsilon);
            if (!output) output = f.weighted ? ggml_mul(ctx, rms, weights) : rms;
            require(ggml_is_contiguous(output) && ggml_are_same_shape(input, output), "RMS output layout differs");
            ggml_set_output(output);
            output_buffer = ggml_backend_cpu_buffer_from_ptr(output_values.storage, output_values.allocated * sizeof(float));
            require(output_buffer && ggml_backend_tensor_alloc(output_buffer, output, output_values.data()) == GGML_STATUS_SUCCESS,
                "RMS output binding failed");
            auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, output);
            if (!candidate && f.weighted) {
                bool fused = false;
                for (int i = 0; i < graph->n_nodes; ++i) if (graph->nodes[i] == rms) {
                    const ggml_op ops[] = {GGML_OP_RMS_NORM, GGML_OP_MUL};
                    fused = ggml_can_fuse(graph, i, ops, 2);
                }
                require(fused, "RMS+MUL baseline fusion unavailable");
            }
            intermediates = ggml_backend_alloc_ctx_tensors(ctx, backend);
            require(!rms || !f.weighted || intermediates, "RMS baseline intermediate allocation failed");
            plan = ggml_backend_graph_plan_create(backend, graph); require(plan, "RMS plan creation failed");
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
    void compute() {
        std::fill(output_values.data(), output_values.data() + output_values.count, guarded::canary);
        require(ggml_backend_graph_plan_compute(backend, plan) == GGML_STATUS_SUCCESS, "RMS compute failed");
        require(output_values.guards(), "RMS output canary changed");
    }
};
double maximum_error = 0;
int cases = 0;
void verify(session & s, int d, int heads, int n, bool weighted, int kind = 0, int hp = 0, int tp = 0, bool expected = true) {
    fixture f(d, heads, n, weighted, hp, tp);
    route baseline(s, f, false, false), candidate(s, f, true, expected);
    for (int iteration = 0; iteration < 2; ++iteration) {
        f.fill(kind, iteration);
        const auto input_hash = f.input.hash(), weight_hash = f.weights.hash();
        baseline.compute(); candidate.compute();
        require(f.input.guards() && f.weights.guards() && f.input.hash() == input_hash && f.weights.hash() == weight_hash,
            "RMS mutated an operand/guard");
        require(std::memcmp(baseline.output_values.data(), candidate.output_values.data(), candidate.output_values.count * sizeof(float)) == 0,
            "RMS complete baseline bit identity failed");
        for (int token = 0; token < n; ++token) for (int head = 0; head < heads; ++head) {
            const auto * x = f.row(token, head);
            double squares = 0;
            for (int i = 0; i < d; ++i) squares += double(x[i]) * double(x[i]);
            const double scale = 1.0 / std::sqrt(squares / d + double(epsilon));
            for (int i = 0; i < d; ++i) {
                const double reference = double(x[i]) * scale * (weighted ? f.weights.data()[i] : 1.0);
                const float actual = candidate.output_values.data()[(size_t(token) * heads + head) * d + i];
                const double error = std::abs(double(actual) - reference);
                require(std::isfinite(actual) && actual != guarded::canary && error <= 2e-6 + 2e-6 * std::abs(reference),
                    "RMS independent Double bound failed");
                maximum_error = std::max(maximum_error, error);
            }
        }
    }
    ++cases;
}
void rejected_shapes() {
    auto * ctx = ggml_init({1024 * 1024, nullptr, true}); require(ctx, "RMS rejection metadata allocation failed");
    const float valid_epsilon = epsilon, invalid_epsilon = NAN;
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 4, 128);
    auto * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    auto reject = [&](ggml_tensor * x, ggml_tensor * w) {
        require(tsg_embeddings::cpu_qkv_rms(ctx, x, w, &valid_epsilon) == nullptr, "Unsupported RMS shape accepted");
    };
    require(tsg_embeddings::cpu_qkv_rms(ctx, input, weight, nullptr) == nullptr &&
        tsg_embeddings::cpu_qkv_rms(ctx, input, weight, &invalid_epsilon) == nullptr, "Invalid RMS epsilon accepted");
    reject(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 256, 4, 128), nullptr);
    reject(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 128, 2), nullptr);
    reject(input, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4));
    reject(input, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512));
    const auto nb0 = input->nb[0]; input->nb[0] = 8; reject(input, weight); input->nb[0] = nb0;
    const auto nb1 = input->nb[1]; input->nb[1] = 1025; reject(input, weight); input->nb[1] = nb1;
    const auto nb2 = input->nb[2]; input->nb[2] = 4; reject(input, weight); input->nb[2] = nb2;
    input->nb[1] = std::numeric_limits<size_t>::max() - 3; reject(input, weight);
    ggml_free(ctx);
}
} // namespace

int main() try {
    require(std::getenv("GGML_CPU_DISABLE_FUSION") == nullptr, "Require unchanged default CPU fusion");
    for (int threads : {1,2,8}) {
        session s(threads);
        for (int d : {256,512}) for (int heads : {1,2,4}) for (bool weighted : {false,true}) {
            verify(s, d, heads, 128, weighted);
            verify(s, d, heads, tsg_embeddings::cpu_qkv_rms_minimum_tokens, weighted);
        }
        verify(s, 256, 2, 137, true, 0, 5, 11);
        verify(s, 512, 4, 129, false, 0, 3, 7);
        for (int kind : {1,2,3}) verify(s, 512, 1, 128, true, kind);
        // Valid unsupported geometries retain ordinary upstream arithmetic.
        verify(s, 72, 4, 128, true, 0, 0, 0, false);
        verify(s, 256, 3, 128, false, 0, 0, 0, false);
        verify(s, 256, 1, tsg_embeddings::cpu_qkv_rms_minimum_tokens - 1, true, 0, 0, 0, false);
    }
    rejected_shapes();
    std::printf("Embedding CPU RMS passed: %d cases,2 persistent-plan forwards each,threads1/2/8,complete bit identity,Double maxerr=%.12g,guards/immutability,shared-weight broadcast and fallback rejection\n", cases, maximum_error);
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "%s\n", error.what()); return 1;
}
