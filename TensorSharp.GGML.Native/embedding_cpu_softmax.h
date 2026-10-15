// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <limits>

#if defined(__linux__) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TSG_EMBEDDING_AVX512_SOFTMAX 1
#define TSG_EMBEDDING_SOFTMAX_TARGET __attribute__((target("avx512f,f16c,fma")))
#endif

namespace tsg_embeddings {

inline bool cpu_softmax_simd_supported() {
#if defined(TSG_EMBEDDING_AVX512_SOFTMAX)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

inline void cpu_softmax_scalar(float * row, int count, const void * mask, ggml_type type,
    int begin, int query, int radius, float scale) {
    float maximum = -INFINITY;
    for (int c = 0; c < count; ++c) {
        float bias = mask ? (type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(mask)[begin + c]) :
            static_cast<const float *>(mask)[begin + c]) : 0;
        if (radius >= 0 && std::abs(begin + c - query) > radius) bias = -INFINITY;
        row[c] = row[c] * scale + bias;
        maximum = std::max(maximum, row[c]);
    }
    double denominator = 0;
    for (int c = 0; c < count; ++c) {
        row[c] = std::isfinite(maximum) ? std::exp(row[c] - maximum) : 0;
        denominator += row[c];
    }
    const float inverse = denominator > 0 ? float(1.0 / denominator) : 0;
    for (int c = 0; c < count; ++c) row[c] *= inverse;
}

#if defined(TSG_EMBEDDING_AVX512_SOFTMAX)
TSG_EMBEDDING_SOFTMAX_TARGET inline __m512 cpu_softmax_exp_normal(__m512 x) {
    // Range reduction to [-log(2)/2, log(2)/2], followed by degree-7 exp.
    // Split log(2) avoids cancellation at large negative arguments. Only
    // [-80,0] enters this polynomial; masked and subnormal tails use libm.
    const auto n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto r = _mm512_fnmadd_ps(n, _mm512_set1_ps(.693145751953125f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(1.428606765330187e-6f), r);
    auto p = _mm512_set1_ps(1.0f / 5040.0f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 720.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 120.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(.5f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}

TSG_EMBEDDING_SOFTMAX_TARGET inline void cpu_softmax_avx512(float * row, int count, const void * mask, ggml_type type,
    int begin, int query, int radius, float scale) {
    auto maximum_vector = _mm512_set1_ps(-INFINITY);
    const auto negative_infinity = _mm512_set1_ps(-INFINITY);
    const auto factor = _mm512_set1_ps(scale);
    int c = 0;
    for (; c + 16 <= count; c += 16) {
        auto values = _mm512_mul_ps(_mm512_loadu_ps(row + c), factor);
        if (mask) {
            const auto bias = type == GGML_TYPE_F16
                ? _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(static_cast<const ggml_fp16_t *>(mask) + begin + c)))
                : _mm512_loadu_ps(static_cast<const float *>(mask) + begin + c);
            values = _mm512_add_ps(values, bias);
        }
        if (radius >= 0) {
            const auto keys = _mm512_add_epi32(_mm512_set1_epi32(begin + c),
                _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
            const auto valid = _mm512_cmpge_epi32_mask(keys, _mm512_set1_epi32(query - radius)) &
                _mm512_cmple_epi32_mask(keys, _mm512_set1_epi32(query + radius));
            values = _mm512_mask_mov_ps(negative_infinity, valid, values);
        }
        _mm512_storeu_ps(row + c, values);
        maximum_vector = _mm512_max_ps(maximum_vector, values);
    }
    float maximum = _mm512_reduce_max_ps(maximum_vector);
    for (; c < count; ++c) {
        float bias = mask ? (type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(mask)[begin + c]) :
            static_cast<const float *>(mask)[begin + c]) : 0;
        if (radius >= 0 && std::abs(begin + c - query) > radius) bias = -INFINITY;
        row[c] = row[c] * scale + bias;
        maximum = std::max(maximum, row[c]);
    }
    if (!std::isfinite(maximum)) { std::fill(row, row + count, 0); return; }
    auto sum_low = _mm512_setzero_pd(), sum_high = _mm512_setzero_pd();
    c = 0;
    for (; c + 16 <= count; c += 16) {
        const auto x = _mm512_sub_ps(_mm512_loadu_ps(row + c), _mm512_set1_ps(maximum));
        const auto normal = _mm512_cmp_ps_mask(x, _mm512_set1_ps(-80.0f), _CMP_GE_OQ) &
            _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OQ);
        auto values = _mm512_maskz_mov_ps(normal, cpu_softmax_exp_normal(_mm512_maskz_mov_ps(normal, x)));
        _mm512_storeu_ps(row + c, values);
        if (normal != 0xffff) {
            alignas(64) float arguments[16];
            _mm512_store_ps(arguments, x);
            for (int lane = 0; lane < 16; ++lane) if (!(normal & (1u << lane)))
                row[c + lane] = arguments[lane] == -INFINITY ? 0.0f : std::exp(arguments[lane]);
            values = _mm512_loadu_ps(row + c);
        }
        sum_low = _mm512_add_pd(sum_low, _mm512_cvtps_pd(_mm512_castps512_ps256(values)));
        const auto upper = _mm512_shuffle_f32x4(values, values, _MM_SHUFFLE(3, 2, 3, 2));
        sum_high = _mm512_add_pd(sum_high, _mm512_cvtps_pd(_mm512_castps512_ps256(upper)));
    }
    double denominator = _mm512_reduce_add_pd(_mm512_add_pd(sum_low, sum_high));
    for (; c < count; ++c) { row[c] = std::exp(row[c] - maximum); denominator += row[c]; }
    const auto inverse = _mm512_set1_ps(denominator > 0 ? float(1.0 / denominator) : 0);
    c = 0;
    for (; c + 16 <= count; c += 16) _mm512_storeu_ps(row + c, _mm512_mul_ps(_mm512_loadu_ps(row + c), inverse));
    const float tail_inverse = _mm512_cvtss_f32(inverse);
    for (; c < count; ++c) row[c] *= tail_inverse;
}
#endif

inline void cpu_softmax(float * row, int count, const void * mask, ggml_type type,
    int begin, int query, int radius, float scale, bool simd) {
#if defined(TSG_EMBEDDING_AVX512_SOFTMAX)
    if (simd) { cpu_softmax_avx512(row, count, mask, type, begin, query, radius, scale); return; }
#else
    (void)simd;
#endif
    cpu_softmax_scalar(row, count, mask, type, begin, query, radius, scale);
}

} // namespace tsg_embeddings
