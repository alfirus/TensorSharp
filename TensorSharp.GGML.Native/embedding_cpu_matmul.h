// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"
#include "embedding_linux_blas.h"

#if defined(TSG_GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

namespace tsg_embeddings {

inline void linux_blas_matmul(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * weights = dst->src[0];
    const auto * input = dst->src[1];
    const auto * plan = static_cast<const linux_blas_plan *>(userdata);
    const blas_const_matrix w{static_cast<const float *>(weights->data), ggml_nbytes(weights) / sizeof(float),
        weights->ne[1], weights->ne[0], int64_t(weights->nb[1] / sizeof(float))};
    const blas_const_matrix x{static_cast<const float *>(input->data), ggml_nbytes(input) / sizeof(float),
        input->ne[1], input->ne[0], int64_t(input->nb[1] / sizeof(float))};
    const blas_matrix y{static_cast<float *>(dst->data), ggml_nbytes(dst) / sizeof(float),
        dst->ne[1], dst->ne[0], int64_t(dst->nb[1] / sizeof(float))};
    // Larger batches amortize packing the complete weight matrix over token
    // tiles. The very wide PLE projection instead benefits from distributing
    // its weight rows until the token batch is large enough to amortize packing
    // that matrix on every worker. Both policies use the same F32 operands.
    const auto axis = x.rows >= 128 && !(w.rows >= 8192 && x.rows < 1024)
        ? blas_partition::token_rows : blas_partition::output_channels;
    if (plan->try_partition(w, x, y, ith, nth, axis)) return;
    // A library thread-policy change after graph construction must never leave
    // an output partition unwritten. Keep a precise, disjoint fallback here.
    const bool tokens = axis == blas_partition::token_rows;
    const int64_t row_begin = tokens ? x.rows * ith / nth : 0, row_end = tokens ? x.rows * (ith + 1) / nth : x.rows;
    const int64_t column_begin = tokens ? 0 : w.rows * ith / nth, column_end = tokens ? w.rows : w.rows * (ith + 1) / nth;
    for (int64_t row = row_begin; row < row_end; ++row) for (int64_t column = column_begin; column < column_end; ++column) {
        double sum = 0;
        for (int64_t k = 0; k < w.columns; ++k) sum += double(w.data[column * w.stride + k]) * x.data[row * x.stride + k];
        y.data[row * y.stride + column] = float(sum);
    }
}

#if defined(TSG_GGML_USE_ACCELERATE)
inline void accelerate_matmul(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) return;
    const auto * weights = dst->src[0];
    const auto * input = dst->src[1];
    // ggml matrices have contiguous columns: W[K,M], X[K,N], Y[M,N].
    // Accelerate uses its own workers, so this operation has one ggml task.
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
        weights->ne[1], input->ne[1], weights->ne[0], 1.0f,
        static_cast<const float *>(weights->data), weights->ne[0],
        static_cast<const float *>(input->data), input->ne[0], 0.0f,
        static_cast<float *>(dst->data), dst->ne[0]);
}
#endif

// Return nullptr when the native ggml CPU path is preferable or necessary.
inline ggml_tensor * cpu_matmul(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * input, int minimum_columns = 32,
    linux_blas_plan * linux_plan = nullptr) {
    if (linux_plan && linux_plan->available() && linux_plan->thread_policy_current() &&
        weights->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && input->ne[1] >= minimum_columns &&
        weights->ne[0] == input->ne[0] && weights->ne[2] == 1 && weights->ne[3] == 1 &&
        input->ne[2] == 1 && input->ne[3] == 1 && ggml_is_contiguous(weights) && ggml_is_contiguous(input)) {
        ggml_tensor * args[] = {weights, input};
        return ggml_custom_4d(ctx, GGML_TYPE_F32, weights->ne[1], input->ne[1], 1, 1,
            args, 2, linux_blas_matmul, linux_plan->worker_budget(), linux_plan);
    }
#if defined(TSG_GGML_USE_ACCELERATE)
    if (weights->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 || input->ne[1] < minimum_columns ||
        weights->ne[0] != input->ne[0] || weights->ne[2] != 1 || weights->ne[3] != 1 ||
        input->ne[2] != 1 || input->ne[3] != 1 ||
        !ggml_is_contiguous(weights) || !ggml_is_contiguous(input)) return nullptr;
    ggml_tensor * args[] = {weights, input};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, weights->ne[1], input->ne[1], 1, 1,
        args, 2, accelerate_matmul, 1, nullptr);
#else
    (void)ctx; (void)weights; (void)input; (void)minimum_columns;
    return nullptr;
#endif
}

} // namespace tsg_embeddings
