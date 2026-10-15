// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone exact-helper/operator/lifetime probe. No model/performance claim.
#include "ggml_ops_internal.h"
// The rejected prototype is intentionally absent from production. Link this
// probe against its frozen experiment library/source snapshot, never against
// a modified upstream checkout or a current library lacking these C++ APIs.
struct tsg_gemma4_vision_metal_mixed_stats {
    uint64_t entries, weight_bytes, workspace_bytes, projections, cache_hits,
        fallback_bound, fallback_weight, fallback_budget, invalidations;
};
extern bool tsg_gemma4_audio_metal_graph_compute(ggml_backend_t, ggml_cgraph *, ggml_status *, bool);
extern void tsg_gemma4_vision_metal_projection_bound(ggml_tensor *, const float *, float, float);
extern tsg_gemma4_vision_metal_mixed_stats tsg_gemma4_vision_metal_mixed_cache_stats();
extern void tsg_gemma4_vision_metal_mixed_invalidate(const void *);
extern void tsg_gemma4_vision_metal_mixed_clear();
#include "ggml-metal.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>
extern "C" int TSGgml_IsBackendAvailable(int);
extern "C" void TSGgml_InvalidateHostBuffer(void *);
extern "C" void TSGgml_ClearHostBufferCache();
extern "C" void TSGgml_Shutdown();
struct Weight {
    float * host = nullptr; ggml_backend_buffer_t mapping = nullptr; int inner, columns; size_t bytes;
    Weight(ggml_backend_t backend, int k, int n) : inner(k), columns(n) {
        bytes = ((static_cast<size_t>(k) * n * 4 + 16383) / 16384) * 16384;
        if (posix_memalign(reinterpret_cast<void **>(&host), 16384, bytes)) throw std::bad_alloc();
        for (int i = 0; i < k * n; ++i) host[i] = ((i * 13 + 17) % 61 - 30) / 271.f;
        mapping = ggml_backend_dev_buffer_from_host_ptr(ggml_backend_get_device(backend), host, bytes, bytes);
        if (!mapping) { std::free(host); host = nullptr; throw std::runtime_error("No original host-weight Metal mapping."); }
    }
    ~Weight() { tsg_gemma4_vision_metal_mixed_invalidate(host); ggml_backend_buffer_free(mapping); std::free(host); }
};
std::vector<float> run(ggml_backend_t backend, Weight& weight, const std::vector<float>& input, bool mixed) {
    setenv("TS_GEMMA4_VISION_METAL_MIXED_MPS", mixed ? "1" : "0", 1);
    const int rows = static_cast<int>(input.size() / weight.inner);
    ggml_context * ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    if (!ctx) throw std::bad_alloc();
    ggml_backend_buffer_t allocation = nullptr;
    try {
        auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, weight.inner, weight.columns);
        if (ggml_backend_tensor_alloc(weight.mapping, w, weight.host) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Weight mapping allocation failed.");
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, weight.inner, rows);
        auto * y = ggml_mul_mat(ctx, w, x);
        ggml_set_name(y, "tsg_gemma4_vision_f32_projection"); ggml_prec_set_acc(y, GGML_PREC_F32);
        double maximum = 0, l2 = 0;
        for (int row = 0; row < rows; ++row) {
            double square = 0;
            for (int k = 0; k < weight.inner; ++k) {
                const double value = input[row * weight.inner + k];
                maximum = std::fmax(maximum, std::fabs(value)); square += value * value;
            }
            l2 = std::fmax(l2, std::sqrt(square));
        }
        tsg_gemma4_vision_metal_projection_bound(y, weight.host,
            std::nextafter(static_cast<float>(maximum * 1.01), INFINITY),
            std::nextafter(static_cast<float>(l2 * 1.01), INFINITY));
        auto * graph = ggml_new_graph_custom(ctx, 64, false); ggml_build_forward_expand(graph, y);
        allocation = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!allocation) throw std::bad_alloc();
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * 4);
        ggml_status status = GGML_STATUS_FAILED;
        if (!tsg_gemma4_audio_metal_graph_compute(backend, graph, &status, true) || status != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Required owned helper did not execute.");
        std::vector<float> result(static_cast<size_t>(rows) * weight.columns);
        ggml_backend_tensor_get(y, result.data(), 0, result.size() * 4);
        ggml_backend_buffer_free(allocation); ggml_free(ctx); return result;
    } catch (...) { if (allocation) ggml_backend_buffer_free(allocation); ggml_free(ctx); throw; }
}
double check(const std::vector<float>& output, Weight& weight, const std::vector<float>& input, double bound) {
    double maximum = 0; const size_t rows = input.size() / weight.inner;
    for (size_t row = 0; row < rows; ++row)for (int column = 0; column < weight.columns; ++column) {
        double expected = 0;
        for (int k = 0; k < weight.inner; ++k)
            expected += static_cast<double>(input[row * weight.inner + k]) * weight.host[column * weight.inner + k];
        const double value = output[row * weight.columns + column];
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite projection.");
        maximum = std::fmax(maximum, std::fabs(value - expected));
    }
    if (maximum > bound) throw std::runtime_error("Unchanged independent double bound failed: " + std::to_string(maximum));
    return maximum;
}
int main() {
    ggml_backend_t backend = nullptr;
    try {
        backend = ggml_backend_metal_init(); if (!backend) return 2;
        if (!TSGgml_IsBackendAvailable(1)) throw std::runtime_error("Required core Metal backend unavailable.");
        tsg_gemma4_vision_metal_mixed_clear();
        setenv("TS_GEMMA4_VISION_METAL_MPS_MM", "1", 1);
        {
            Weight first(backend, 64, 19), peer(backend, 64, 19);
            std::vector<float> input(37 * 64);
            for (size_t i = 0; i < input.size(); ++i) input[i] = (static_cast<int>((i * 17) % 71) - 35) / 47.f;
            double ordinary = check(run(backend, first, input, false), first, input, 3e-5);
            auto baseline = tsg_gemma4_vision_metal_mixed_cache_stats();
            double mixed = check(run(backend, first, input, true), first, input, 3e-5);
            auto cold = tsg_gemma4_vision_metal_mixed_cache_stats();
            if (cold.projections <= baseline.projections || cold.entries != 1 || !cold.weight_bytes)
                throw std::runtime_error("Mixed prototype was a fallback-only no-op.");
            check(run(backend, first, input, true), first, input, 3e-5);
            auto warm = tsg_gemma4_vision_metal_mixed_cache_stats();
            if (warm.cache_hits <= cold.cache_hits || warm.weight_bytes != cold.weight_bytes)
                throw std::runtime_error("Immutable weight-plane reuse failed.");
            for (int i = 0; i < 64 * 19; ++i) peer.host[i] *= .73f;
            auto peer_expected = run(backend, peer, input, true); check(peer_expected, peer, input, 3e-5);
            auto both = tsg_gemma4_vision_metal_mixed_cache_stats();
            TSGgml_InvalidateHostBuffer(first.host);
            auto released = tsg_gemma4_vision_metal_mixed_cache_stats();
            if (released.entries + 1 != both.entries || released.weight_bytes >= both.weight_bytes)
                throw std::runtime_error("Public pointer invalidation did not release exactly one owned weight.");
            if (run(backend, peer, input, true) != peer_expected) throw std::runtime_error("Invalidating a peer corrupted live weights.");
            for (int i = 0; i < 64 * 19; ++i) first.host[i] += (i % 3 - 1) * .0237f;
            check(run(backend, first, input, true), first, input, 3e-5);
            TSGgml_ClearHostBufferCache();
            auto cleared = tsg_gemma4_vision_metal_mixed_cache_stats();
            if (cleared.entries || cleared.weight_bytes || cleared.workspace_bytes) throw std::runtime_error("Public cache clear retained planes/workspace.");
            std::printf("{\"case\":\"partial_unrounded_and_lifetime\",\"f32_max_error\":%.12g,\"mixed_max_error\":%.12g,\"mixed_executed\":true,\"passed\":true}\n", ordinary, mixed);
        }
        for (int mode = 0; mode < 3; ++mode) {
            Weight weight(backend, mode == 2 ? 1024 : 64, 8);
            std::vector<float> input(3 * weight.inner);
            for (size_t i = 0; i < input.size(); ++i) input[i] = (static_cast<int>(i % 17) - 8) / 11.f;
            if (mode == 0) input[0] = 80000.125f;
            if (mode == 1) weight.host[0] = 80000.125f;
            if (mode == 2) {
                for (size_t i = 0; i < input.size(); ++i) input[i] = i % 2 ? -.99951f : 1.00049f;
                for (int i = 0; i < weight.inner * weight.columns; ++i) weight.host[i] = 1.00049f;
            }
            auto before = tsg_gemma4_vision_metal_mixed_cache_stats();
            double error = check(run(backend, weight, input, true), weight, input, mode == 2 ? 3e-5 : .02);
            auto after = tsg_gemma4_vision_metal_mixed_cache_stats();
            if (after.projections != before.projections || after.fallback_bound + after.fallback_weight <= before.fallback_bound + before.fallback_weight)
                throw std::runtime_error("Large-range/cancellation guard did not select F32 fallback.");
            std::printf("{\"case\":\"%s\",\"max_error\":%.12g,\"f32_fallback\":true,\"passed\":true}\n", mode == 0 ? "input80K" : mode == 1 ? "weight80K" : "cancellation1024", error);
        }
        TSGgml_Shutdown(); auto final = tsg_gemma4_vision_metal_mixed_cache_stats();
        if (final.entries || final.weight_bytes || final.workspace_bytes) throw std::runtime_error("Shutdown retained owned model buffers.");
        ggml_backend_free(backend); std::puts("{\"completed\":true,\"cases\":4,\"passed\":true}"); return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Validation failed: %s\n", error.what());
        TSGgml_Shutdown(); if (backend) ggml_backend_free(backend); return 1;
    }
}
