// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"
#include <cmath>
#include <cstdint>
#include <limits>

namespace tsg_embeddings {
inline constexpr int cpu_qkv_rms_minimum_tokens = 8;

// Match unchanged ggml CPU RMS arithmetic: each square is rounded in F32,
// accumulated sequentially in double, then mean/scale/products use F32.
// Only the distribution of complete logical rows across workers changes.
inline void cpu_qkv_rms_rows(ggml_tensor * dst, const ggml_tensor * input,
                            const ggml_tensor * weight, int ith, int nth, const float * epsilon) {
    const int64_t rows = ggml_nrows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        const int64_t head = row % input->ne[1], token = row / input->ne[1];
        const auto * x = reinterpret_cast<const float *>(static_cast<const char *>(input->data) +
            head * input->nb[1] + token * input->nb[2]);
        auto * y = reinterpret_cast<float *>(static_cast<char *>(dst->data) +
            head * dst->nb[1] + token * dst->nb[2]);
        double sum = 0;
        for (int64_t i = 0; i < input->ne[0]; ++i) sum += double(x[i] * x[i]);
        const float mean = sum / input->ne[0];
        const float scale = 1.0f / sqrtf(mean + *epsilon);
        GGML_ASSERT(scale > 0.0f);
        if (weight) {
            const auto * w = static_cast<const float *>(weight->data);
            for (int64_t i = 0; i < input->ne[0]; ++i) y[i] = x[i] * scale * w[i];
        } else {
            for (int64_t i = 0; i < input->ne[0]; ++i) y[i] = x[i] * scale;
        }
    }
}
inline void cpu_qkv_rms_unweighted(ggml_tensor * dst, const ggml_tensor * input,
                                   int ith, int nth, void * userdata) {
    cpu_qkv_rms_rows(dst, input, nullptr, ith, nth, static_cast<const float *>(userdata));
}
inline void cpu_qkv_rms_weighted(ggml_tensor * dst, const ggml_tensor * input, const ggml_tensor * weight,
                                 int ith, int nth, void * userdata) {
    cpu_qkv_rms_rows(dst, input, weight, ith, nth, static_cast<const float *>(userdata));
}

// Caller must select the qualified CPU model/backend. epsilon is immutable
// encoder-owned storage and must outlive this graph and all cached CPU plans.
// Return nullptr for the ordinary upstream path outside the measured 3D Q/K/V
// geometry. The optional weight is a shared 1D head vector broadcast over rows.
inline ggml_tensor * cpu_qkv_rms(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight,
                                 const float * epsilon, int minimum_tokens = cpu_qkv_rms_minimum_tokens) {
    if (!ctx || !input || !epsilon || !std::isfinite(*epsilon) || *epsilon <= 0 || minimum_tokens < 1 ||
        input->type != GGML_TYPE_F32 || (input->ne[0] != 256 && input->ne[0] != 512) ||
        (input->ne[1] != 1 && input->ne[1] != 2 && input->ne[1] != 4) || input->ne[2] < minimum_tokens ||
        input->ne[2] > 65536 || input->ne[3] != 1 || input->nb[0] != sizeof(float) ||
        input->nb[1] % sizeof(float) || input->nb[2] % sizeof(float) ||
        input->nb[1] < size_t(input->ne[0]) * sizeof(float) ||
        input->nb[1] > std::numeric_limits<size_t>::max() / size_t(input->ne[1]) ||
        input->nb[2] < size_t(input->ne[1]) * input->nb[1]) return nullptr;
    const size_t final_row_bytes = size_t(input->ne[1] - 1) * input->nb[1] + size_t(input->ne[0]) * sizeof(float);
    if (size_t(input->ne[2] - 1) > (std::numeric_limits<size_t>::max() - final_row_bytes) / input->nb[2]) return nullptr;
    if (weight && (weight->type != GGML_TYPE_F32 || weight->ne[0] != input->ne[0] ||
        weight->ne[1] != 1 || weight->ne[2] != 1 || weight->ne[3] != 1 ||
        weight->nb[0] != sizeof(float))) return nullptr;
    return weight ? ggml_map_custom2(ctx, input, weight, cpu_qkv_rms_weighted,
                                    GGML_N_TASKS_MAX, const_cast<float *>(epsilon)) :
                    ggml_map_custom1(ctx, input, cpu_qkv_rms_unweighted,
                                    GGML_N_TASKS_MAX, const_cast<float *>(epsilon));
}

} // namespace tsg_embeddings
