// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"
#include "embedding_linux_blas.h"
#include "embedding_cpu_softmax.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace tsg_embeddings {

// The encoder serializes its graphs. One bounded scratch allocation can serve
// every layer without materializing the full heads x tokens x tokens matrix.
struct cpu_attention_workspace {
    static constexpr int query_tile = 64;
    linux_blas_plan * blas = nullptr;
    size_t worker_elements = 0;
    std::vector<float> scores;
    std::atomic<uint64_t> tiles{0}, scalar_products{0};
    bool profile = false;
    bool simd_softmax = false;
    std::atomic<uint64_t> qk_ns{0}, softmax_ns{0}, pv_ns{0};
    explicit cpu_attention_workspace(linux_blas_plan * value) : blas(value) {
        const char * setting = std::getenv("TSG_EMBEDDING_CPU_ATTENTION_PROFILE");
        profile = setting && std::strcmp(setting, "1") == 0;
        const char * simd_setting = std::getenv("TSG_EMBEDDING_CPU_ATTENTION_SIMD");
        simd_softmax = cpu_softmax_simd_supported() && !(simd_setting && std::strcmp(simd_setting, "0") == 0);
    }
    void reserve(int keys) {
        const size_t elements = size_t(query_tile) * keys;
        if (elements > worker_elements) {
            scores.resize(elements * blas->worker_budget());
            worker_elements = elements;
        }
    }
};

struct cpu_attention_node {
    cpu_attention_workspace * workspace;
    int radius; // -1: global, 0: self only; mask values are still applied.
    float scale;
};

inline size_t matrix_extent(int64_t rows, int64_t columns, int64_t stride) {
    return size_t(rows - 1) * size_t(stride) + size_t(columns);
}

inline void cpu_attention_compute(ggml_tensor * dst, int ith, int nth, void * userdata) {
    auto & config = *static_cast<cpu_attention_node *>(userdata);
    auto & workspace = *config.workspace;
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    const int dim = int(q->ne[0]), length = int(q->ne[1]), heads = int(q->ne[2]);
    const int keys = int(k->ne[1]), kv_heads = int(k->ne[2]);
    const int blocks = (length + cpu_attention_workspace::query_tile - 1) / cpu_attention_workspace::query_tile;
    const int64_t jobs = int64_t(blocks) * heads * q->ne[3];
    auto * scores = workspace.scores.data() + size_t(ith) * workspace.worker_elements;
    for (int64_t job = ith; job < jobs; job += nth) {
        const int block = int(job % blocks), head = int(job / blocks % heads), batch = int(job / (blocks * heads));
        const int first = block * cpu_attention_workspace::query_tile;
        const int rows = std::min(cpu_attention_workspace::query_tile, length - first);
        const int begin = config.radius < 0 ? 0 : std::max(0, first - config.radius);
        const int end = config.radius < 0 ? keys : std::min(keys, first + rows + config.radius);
        const int count = end - begin, kv = head / (heads / kv_heads);
        const auto * qp = reinterpret_cast<const float *>(static_cast<const char *>(q->data) +
            batch * q->nb[3] + head * q->nb[2] + first * q->nb[1]);
        const auto * kp = reinterpret_cast<const float *>(static_cast<const char *>(k->data) +
            batch * k->nb[3] + kv * k->nb[2] + begin * k->nb[1]);
        const auto * vp = reinterpret_cast<const float *>(static_cast<const char *>(v->data) +
            batch * v->nb[3] + kv * v->nb[2] + begin * v->nb[1]);
        auto * yp = static_cast<float *>(dst->data) + (size_t(batch) * length + first) * heads * dim + head * dim;
        const int64_t qs = q->nb[1] / sizeof(float), ks = k->nb[1] / sizeof(float), vs = v->nb[1] / sizeof(float);
        const blas_const_matrix qm{qp, matrix_extent(rows, dim, qs), rows, dim, qs};
        const blas_const_matrix km{kp, matrix_extent(count, dim, ks), count, dim, ks};
        const blas_matrix sm{scores, size_t(rows) * count, rows, count, count};
        using clock = std::chrono::steady_clock;
        auto before = workspace.profile ? clock::now() : clock::time_point{};
        if (!workspace.blas || !workspace.blas->try_partition(km, qm, sm, 0, 1, blas_partition::token_rows)) {
            workspace.scalar_products.fetch_add(1, std::memory_order_relaxed);
            for (int r = 0; r < rows; ++r) for (int c = 0; c < count; ++c) {
                double sum = 0;
                for (int d = 0; d < dim; ++d) sum += double(qp[r * qs + d]) * kp[c * ks + d];
                scores[r * count + c] = float(sum);
            }
        }
        if (workspace.profile) {
            const auto now = clock::now();
            workspace.qk_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(now - before).count(), std::memory_order_relaxed);
            before = now;
        }
        for (int r = 0; r < rows; ++r) {
            auto * row = scores + r * count;
            const auto * mp = mask ? static_cast<const char *>(mask->data) +
                (mask->ne[3] == 1 ? 0 : batch * mask->nb[3]) + (first + r) * mask->nb[1] : nullptr;
            cpu_softmax(row, count, mp, mask ? mask->type : GGML_TYPE_F32, begin, first + r,
                config.radius, config.scale, workspace.simd_softmax);
        }
        const blas_const_matrix probabilities{scores, size_t(rows) * count, rows, count, count};
        const blas_const_matrix values{vp, matrix_extent(count, dim, vs), count, dim, vs};
        const blas_matrix output{yp, matrix_extent(rows, dim, heads * dim), rows, dim, heads * dim};
        if (workspace.profile) {
            const auto now = clock::now();
            workspace.softmax_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(now - before).count(), std::memory_order_relaxed);
            before = now;
        }
        if (!workspace.blas || !workspace.blas->try_product(probabilities, values, output, ith)) {
            workspace.scalar_products.fetch_add(1, std::memory_order_relaxed);
            for (int r = 0; r < rows; ++r) for (int d = 0; d < dim; ++d) {
                double sum = 0;
                for (int c = 0; c < count; ++c) sum += double(scores[r * count + c]) * vp[c * vs + d];
                yp[r * heads * dim + d] = float(sum);
            }
        }
        if (workspace.profile) workspace.pv_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - before).count(), std::memory_order_relaxed);
        workspace.tiles.fetch_add(1, std::memory_order_relaxed);
    }
}

inline ggml_tensor * cpu_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
    ggml_tensor * mask, int radius, float scale, cpu_attention_workspace & workspace,
    std::vector<std::unique_ptr<cpu_attention_node>> & nodes) {
    if (!workspace.blas || !workspace.blas->available() || !workspace.blas->thread_policy_current() ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        q->nb[0] != sizeof(float) || k->nb[0] != sizeof(float) || v->nb[0] != sizeof(float) ||
        q->nb[1] < q->ne[0] * sizeof(float) || k->nb[1] < k->ne[0] * sizeof(float) || v->nb[1] < v->ne[0] * sizeof(float) ||
        q->nb[1] % sizeof(float) || k->nb[1] % sizeof(float) || v->nb[1] % sizeof(float) || k->ne[1] > 65536 ||
        q->ne[0] != k->ne[0] || k->ne[0] != v->ne[0] || k->ne[1] != v->ne[1] ||
        q->ne[1] != k->ne[1] || q->ne[2] % k->ne[2] || k->ne[2] != v->ne[2] ||
        q->ne[3] != k->ne[3] || k->ne[3] != v->ne[3] || radius < -1 || !std::isfinite(scale) ||
        (mask && ((mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) || mask->nb[0] != ggml_type_size(mask->type) ||
            mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1] || mask->ne[2] != 1 ||
            (mask->ne[3] != 1 && mask->ne[3] != q->ne[3])))) return nullptr;
    workspace.reserve(int(k->ne[1]));
    nodes.push_back(std::make_unique<cpu_attention_node>(cpu_attention_node{&workspace, radius, scale}));
    ggml_tensor * args[] = {q, k, v, mask};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, q->ne[0] * q->ne[2], q->ne[1], q->ne[3], 1,
        args, mask ? 4 : 3, cpu_attention_compute, workspace.blas->worker_budget(), nodes.back().get());
}

} // namespace tsg_embeddings
