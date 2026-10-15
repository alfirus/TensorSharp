// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "embedding_cpu_matmul.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool verify(int k, int m, int n, tsg_embeddings::linux_blas_plan * plan = nullptr, int threads = 4, bool close_before_compute = false) {
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_set_input(weights); ggml_set_input(input);
    auto * output = tsg_embeddings::cpu_matmul(ctx, weights, input, plan ? 1 : 32, plan);
    if (plan) {
        if (!output) return false;
    } else {
#if defined(TSG_GGML_USE_ACCELERATE)
    if ((n >= 32) != bool(output)) return false;
#else
    if (output) return false;
#endif
    }
    if (!output) output = ggml_mul_mat(ctx, weights, input);
    if (close_before_compute) plan->close();
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool success = ggml_backend_supports_op(backend, output) && ggml_gallocr_alloc_graph(allocator, graph);
    std::vector<float> w(k * m), x(k * n), result(m * n);
    for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin(float(i) * .17f) * .1f;
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(float(i) * .13f) * .2f;
    if (success) {
        ggml_backend_tensor_set(weights, w.data(), 0, w.size() * sizeof(float));
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        success = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        for (int column = 0; column < n; ++column) for (int row = 0; row < m; ++row) {
            double expected = 0;
            for (int i = 0; i < k; ++i) expected += double(w[row * k + i]) * x[column * k + i];
            success = success && std::isfinite(result[column * m + row]) &&
                std::abs(expected - result[column * m + row]) < 2e-5;
        }
    }
    ggml_gallocr_free(allocator); ggml_free(ctx); ggml_backend_free(backend);
    return success;
}

static bool verify_product(tsg_embeddings::linux_blas_plan & plan) {
    constexpr int m = 3, k = 37, n = 71, as = k + 5, bs = n + 7, ys = n + 3;
    constexpr float canary = 9876.5f;
    std::vector<float> a(1 + m * as + 5, canary), b(1 + k * bs + 5, canary), y(1 + m * ys + 5, canary);
    for (int r = 0; r < m; ++r) for (int c = 0; c < k; ++c) a[1 + r * as + c] = .1f * std::sin(float(r * 13 + c) * .17f);
    for (int r = 0; r < k; ++r) for (int c = 0; c < n; ++c) b[1 + r * bs + c] = .2f * std::cos(float(r * 7 + c) * .13f);
    const tsg_embeddings::blas_const_matrix am{a.data() + 1, a.size() - 1, m, k, as}, bm{b.data() + 1, b.size() - 1, k, n, bs};
    const tsg_embeddings::blas_matrix ym{y.data() + 1, y.size() - 1, m, n, ys};
    if (!plan.try_product(am, bm, ym, 0)) return false;
    for (int r = 0; r < m; ++r) for (int c = 0; c < ys; ++c) {
        if (c >= n) { if (y[1 + r * ys + c] != canary) return false; continue; }
        double expected = 0;
        for (int i = 0; i < k; ++i) expected += double(a[1 + r * as + i]) * b[1 + i * bs + c];
        if (!std::isfinite(y[1 + r * ys + c]) || std::abs(y[1 + r * ys + c] - expected) > 2e-5) return false;
    }
    if (y.front() != canary) return false;
    for (size_t i = 1 + m * ys; i < y.size(); ++i) if (y[i] != canary) return false;
    const auto before = y;
    auto invalid = am; invalid.elements = 1;
    if (plan.try_product(invalid, bm, ym, 0) || y != before) return false;
    invalid = am; invalid.stride = k - 1;
    if (plan.try_product(invalid, bm, ym, 0) || y != before) return false;
    // Overlap is rejected before BLAS can overwrite its own operand.
    auto alias = ym; alias.data = a.data() + 1; alias.elements = a.size() - 1; alias.columns = k; alias.stride = as;
    auto square = bm; square.columns = k;
    const auto before_a = a;
    if (plan.try_product(am, square, alias, 0) || a != before_a) return false;
    std::printf("Linux BLAS row-major product passed: padded operands/output, double oracle, bounds/stride/alias rejection\n");
    return true;
}

int main(int argc, char ** argv) {
    const bool linux_blas = argc == 2 && std::strcmp(argv[1], "--linux-blas") == 0;
    if (argc > 1 && !linux_blas) return 2;
    std::unique_ptr<tsg_embeddings::linux_blas_plan> plan;
    if (linux_blas) {
        plan = std::make_unique<tsg_embeddings::linux_blas_plan>(std::getenv("TSG_EMBEDDING_BLAS_LIBRARY"), 8);
        if (!plan->available()) {
            std::fprintf(stderr, "Required Linux BLAS unavailable: %s\n", plan->reason().c_str());
            return 2;
        }
    }
    for (const auto & shape : {std::vector<int>{37, 71, 3}, {37, 71, 32}, {513, 129, 137}, {512, 12288, 32}, {37, 8192, 137}})
        if (!verify(shape[0], shape[1], shape[2], plan.get())) {
            std::fprintf(stderr, "Embedding CPU matmul failed for %d x %d x %d\n", shape[0], shape[1], shape[2]);
            return 1;
        }
    if (plan) {
        for (int threads : {1, 2, 8})
            if (!verify(512, 257, 31, plan.get(), threads)) return 1;
        const auto counts = plan->counts();
        if (counts.gemm_calls == 0 || counts.output_elements == 0 || counts.policy_rejections || counts.shape_rejections || counts.active_calls)
            return 1;
        std::printf("Linux BLAS graph passed: calls=%llu worker_mask=%llu library=%s\n",
            (unsigned long long)counts.gemm_calls, (unsigned long long)counts.worker_mask, plan->library_path().c_str());
        if (!verify_product(*plan)) return 1;
        tsg_embeddings::linux_blas_plan closed(std::getenv("TSG_EMBEDDING_BLAS_LIBRARY"), 8);
        if (!closed.available() || !verify(37, 71, 32, &closed, 4, true) || closed.counts().policy_rejections != 4)
            return 1;
        tsg_embeddings::linux_blas_plan closed_tokens(std::getenv("TSG_EMBEDDING_BLAS_LIBRARY"), 8);
        if (!closed_tokens.available() || !verify(37, 71, 137, &closed_tokens, 4, true) || closed_tokens.counts().policy_rejections != 4)
            return 1;
    }
    return 0;
}
