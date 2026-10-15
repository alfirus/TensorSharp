// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "embedding_audio_cpu_projection.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

// The unchanged CPU host-buffer API requires a 64-byte aligned base. A
// std::vector<float> only guarantees float alignment, even with guard offsets.
struct aligned_floats {
    float * values;
    size_t count;
    aligned_floats(size_t size, float initial) :
        values(static_cast<float *>(::operator new[](size * sizeof(float), std::align_val_t(64)))), count(size) {
        std::fill(values, values + count, initial);
    }
    ~aligned_floats() { ::operator delete[](values, std::align_val_t(64)); }
    aligned_floats(const aligned_floats &) = delete;
    float * data() { return values; }
    size_t size() const { return count; }
    float * begin() { return values; }
    float * end() { return values + count; }
    float & operator[](size_t index) { return values[index]; }
};

static bool policy() {
    using tsg_embeddings::audio_cpu_projection;
    for (auto dimensions : {std::pair<int, int>{1024, 1024}, {1024, 4096}, {4096, 1024}})
        if (!audio_cpu_projection::shape(dimensions.first, dimensions.second)) return false;
    for (auto dimensions : {std::pair<int, int>{1024, 2048}, {9, 128}, {1152, 128}, {32, 64}, {0, 1024}})
        if (audio_cpu_projection::shape(dimensions.first, dimensions.second)) return false;
    for (const char * setting : {static_cast<const char *>(nullptr), "", "0", "true", "2"})
        if (audio_cpu_projection(true, true, setting, nullptr).available()) return false;
    if (audio_cpu_projection(false, true, "1", nullptr).available() ||
        audio_cpu_projection(true, false, "1", nullptr).available() ||
        audio_cpu_projection(true, true, "1", nullptr, true).available() ||
        audio_cpu_projection(true, true, "1", "/tensorsharp-missing-audio-blas-provider.so").available()) return false;
    std::printf("AUDIO_CPU_PROJECTION_POLICY optional1+CPU+precision required; upstream tiled preference; exact learned shapes; missing provider falls back\n");
    return true;
}

static bool graph(tsg_embeddings::audio_cpu_projection & projection, int k, int m, int n) {
    constexpr size_t guard = 64;
    constexpr float canary = 9876.5f;
    aligned_floats w(size_t(k) * m + 2 * guard, canary), x(size_t(k) * n + 2 * guard, canary), y(size_t(m) * n + 2 * guard, canary);
    for (size_t i = 0; i < size_t(k) * m; ++i) w[guard + i] = float(int(i * 31 % 127) - 63) * .0005f;
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * backend = ggml_backend_cpu_init();
    if (!ctx || !backend) return false;
    ggml_backend_cpu_set_n_threads(backend, 8);
    auto * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    auto * output = projection.multiply(ctx, weight, input);
    bool ok = output && output->op == GGML_OP_CUSTOM && output->type == GGML_TYPE_F32;
    auto * wb = ggml_backend_cpu_buffer_from_ptr(w.data(), w.size() * sizeof(float));
    auto * xb = ggml_backend_cpu_buffer_from_ptr(x.data(), x.size() * sizeof(float));
    auto * yb = ggml_backend_cpu_buffer_from_ptr(y.data(), y.size() * sizeof(float));
    if (ok) ok = ggml_backend_tensor_alloc(wb, weight, w.data() + guard) == GGML_STATUS_SUCCESS &&
        ggml_backend_tensor_alloc(xb, input, x.data() + guard) == GGML_STATUS_SUCCESS &&
        ggml_backend_tensor_alloc(yb, output, y.data() + guard) == GGML_STATUS_SUCCESS;
    auto * cg = ggml_new_graph_custom(ctx, 32, false);
    if (ok) { ggml_set_output(output); ggml_build_forward_expand(cg, output); }
    auto * plan = ok ? ggml_backend_graph_plan_create(backend, cg) : nullptr;
    ok = ok && plan;
    double maximum_error = 0;
    for (int repeat = 0; repeat < 3 && ok; ++repeat) {
        for (size_t i = 0; i < size_t(k) * n; ++i) x[guard + i] = float(int((i * 17 + repeat * 13) % 131) - 65) * .003f;
        const std::vector<float> source_w(w.begin(), w.end()), source_x(x.begin(), x.end());
        std::fill(y.begin() + guard, y.end() - guard, canary);
        const auto before = projection.counts();
        ok = ggml_backend_graph_plan_compute(backend, plan) == GGML_STATUS_SUCCESS;
        const auto after = projection.counts();
        ok = ok && after.gemm_calls - before.gemm_calls == 8 &&
            after.output_elements - before.output_elements == uint64_t(m) * n && after.worker_mask == 255 &&
            !after.active_calls && !after.shape_rejections && !after.policy_rejections &&
            std::equal(w.begin(), w.end(), source_w.begin()) && std::equal(x.begin(), x.end(), source_x.begin());
        for (size_t i = 0; i < guard; ++i) ok = ok && y[i] == canary && y[y.size() - 1 - i] == canary;
        for (size_t i = guard; i < y.size() - guard; ++i) ok = ok && std::isfinite(y[i]) && y[i] != canary;
        for (int row : {0, n / 2, n - 1})
        for (int column : {0, m / 8 - 1, m / 8, m / 2, m - 1}) {
            double expected = 0, magnitude = 0;
            for (int inner = 0; inner < k; ++inner) {
                const double product = double(w[guard + size_t(column) * k + inner]) * x[guard + size_t(row) * k + inner];
                expected += product; magnitude += std::abs(product);
            }
            const double error = std::abs(expected - y[guard + size_t(row) * m + column]);
            maximum_error = std::max(maximum_error, error);
            ok = ok && error <= std::max(2e-5, magnitude * 1e-6);
        }
    }
    if (plan) ggml_backend_graph_plan_free(backend, plan);
    ggml_backend_buffer_free(yb); ggml_backend_buffer_free(xb); ggml_backend_buffer_free(wb);
    ggml_free(ctx); ggml_backend_free(backend);
    std::printf("AUDIO_CPU_PROJECTION_GRAPH k=%d m=%d n=%d repeats=3 max_double_error=%.9g guards=%d\n", k, m, n, maximum_error, ok);
    return ok;
}

int main(int argc, char ** argv) {
    const bool required = argc == 2 && std::strcmp(argv[1], "--linux-blas") == 0;
    if (argc > 1 && !required) return 2;
    if (!policy()) return 1;
    if (!required) { std::printf("AUDIO_CPU_PROJECTION linux_blas_executed=0 policy_only=1\n"); return 0; }
    tsg_embeddings::audio_cpu_projection projection(true, true, "1", std::getenv("TSG_EMBEDDING_BLAS_LIBRARY"));
    if (!projection.available()) { std::fprintf(stderr, "Required qualified Linux audio BLAS provider unavailable\n"); return 2; }
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx) return 1;
    auto * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
    auto * reduced = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1024, 17);
    auto * batched = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 17, 2);
    auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1026, 17);
    auto * strided = ggml_view_2d(ctx, storage, 1024, 17, storage->nb[1], 0);
    auto * glu = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 2048);
    auto * dense = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 17);
    if (projection.multiply(ctx, weight, reduced) || projection.multiply(ctx, weight, batched) ||
        projection.multiply(ctx, weight, strided) || projection.multiply(ctx, glu, dense) || projection.nodes()) return 1;
    ggml_free(ctx);
    std::printf("AUDIO_CPU_PROJECTION_REJECTIONS F16/batched/strided/GLU keep upstream fallback\n");
    for (auto dimensions : {std::pair<int, int>{1024, 1024}, {1024, 4096}, {4096, 1024}})
        for (int n : {1, 17, 25}) if (!graph(projection, dimensions.first, dimensions.second, n)) return 1;
    if (projection.nodes() != 9) return 1;
    std::printf("AUDIO_CPU_PROJECTION linux_blas_executed=1 graphs=9 nodes=%zu\n", projection.nodes());
    return 0;
}
