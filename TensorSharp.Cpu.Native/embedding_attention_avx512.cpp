// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Bounded, stateless Float32 attention math under the caller's worker pool.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#if defined(_WIN32)
#define TS_CPU_EXPORT extern "C" __declspec(dllexport)
#else
#define TS_CPU_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TS_ATTN __attribute__((target("avx512f,fma")))
TS_CPU_EXPORT int tensorcpu_attention_avx512_supported() noexcept {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
}

// Named accumulators preserve increasing-dimension FMA order, while eight
// queries reuse each key load. Masked tails never read beyond the actual keys.
TS_ATTN static void scores8(const float* queries, const float* keys, float* output,
    int query_stride, int key_stride, int output_stride, int dimensions,
    int length, int rows, float scale) noexcept {
#define TS_Q(n) const float* q##n = queries + std::size_t(std::min(n, rows - 1)) * query_stride
    TS_Q(0); TS_Q(1); TS_Q(2); TS_Q(3); TS_Q(4); TS_Q(5); TS_Q(6); TS_Q(7);
#undef TS_Q
    const auto multiplier = _mm512_set1_ps(scale);
    for (int key = 0; key < length; key += 32) {
        const int count = std::min(32, length - key);
        const auto mask0 = __mmask16((uint32_t(1) << std::min(16, count)) - 1);
        const auto mask1 = __mmask16((uint32_t(1) << std::max(0, count - 16)) - 1);
#define TS_ZERO(n) auto c##n##0 = _mm512_setzero_ps(); auto c##n##1 = _mm512_setzero_ps()
        TS_ZERO(0); TS_ZERO(1); TS_ZERO(2); TS_ZERO(3); TS_ZERO(4); TS_ZERO(5); TS_ZERO(6); TS_ZERO(7);
#undef TS_ZERO
        for (int d = 0; d < dimensions; ++d) {
            const auto p = keys + std::size_t(d) * key_stride + key;
            const auto k0 = _mm512_maskz_loadu_ps(mask0, p);
            const auto k1 = count > 16 ? _mm512_maskz_loadu_ps(mask1, p + 16) : _mm512_setzero_ps();
#define TS_FMA(n) { const auto a = _mm512_set1_ps(q##n[d]); c##n##0 = _mm512_fmadd_ps(k0, a, c##n##0); c##n##1 = _mm512_fmadd_ps(k1, a, c##n##1); }
            TS_FMA(0); TS_FMA(1); TS_FMA(2); TS_FMA(3); TS_FMA(4); TS_FMA(5); TS_FMA(6); TS_FMA(7);
#undef TS_FMA
        }
#define TS_STORE(n) if (rows > n) { \
        auto p = output + std::size_t(n) * output_stride + key; \
        _mm512_mask_storeu_ps(p, mask0, _mm512_mul_ps(c##n##0, multiplier)); \
        if (count > 16) _mm512_mask_storeu_ps(p + 16, mask1, _mm512_mul_ps(c##n##1, multiplier)); }
        TS_STORE(0); TS_STORE(1); TS_STORE(2); TS_STORE(3); TS_STORE(4); TS_STORE(5); TS_STORE(6); TS_STORE(7);
#undef TS_STORE
    }
}

// Values use [16-channel tile,key,lane]. All eight queries share one value
// vector; every output lane retains its increasing-key FMA accumulation order.
TS_ATTN static void values8(const float* probabilities, const float* values, float* output,
    int probability_stride, int value_stride, int output_stride,
    int dimensions, int length, int rows) noexcept {
#define TS_P(n) const float* p##n = probabilities + std::size_t(std::min(n, rows - 1)) * probability_stride
    TS_P(0); TS_P(1); TS_P(2); TS_P(3); TS_P(4); TS_P(5); TS_P(6); TS_P(7);
#undef TS_P
    for (int d = 0; d < dimensions; d += 16) {
#define TS_ZERO(n) auto c##n = _mm512_setzero_ps()
        TS_ZERO(0); TS_ZERO(1); TS_ZERO(2); TS_ZERO(3); TS_ZERO(4); TS_ZERO(5); TS_ZERO(6); TS_ZERO(7);
#undef TS_ZERO
        for (int key = 0; key < length; ++key) {
            const auto v = _mm512_loadu_ps(values + std::size_t(d) * value_stride + key * 16);
#define TS_FMA(n) c##n = _mm512_fmadd_ps(v, _mm512_set1_ps(p##n[key]), c##n)
            TS_FMA(0); TS_FMA(1); TS_FMA(2); TS_FMA(3); TS_FMA(4); TS_FMA(5); TS_FMA(6); TS_FMA(7);
#undef TS_FMA
        }
#define TS_STORE(n) if (rows > n) _mm512_storeu_ps(output + std::size_t(n) * output_stride + d, c##n)
        TS_STORE(0); TS_STORE(1); TS_STORE(2); TS_STORE(3); TS_STORE(4); TS_STORE(5); TS_STORE(6); TS_STORE(7);
#undef TS_STORE
    }
}

TS_CPU_EXPORT int tensorcpu_attention_scores8(const float* queries, const float* keys, float* output,
    int query_stride, int key_stride, int output_stride, int dimensions, int length, int rows, float scale) noexcept {
    if (!queries || !keys || !output || dimensions <= 0 || dimensions > 1024 || length <= 0 || length > 16384 ||
        query_stride < dimensions || key_stride < length || output_stride < length || rows <= 0 || rows > 8 || !std::isfinite(scale)) return -1;
    if (!tensorcpu_attention_avx512_supported()) return 1;
    scores8(queries, keys, output, query_stride, key_stride, output_stride, dimensions, length, rows, scale); return 0;
}
TS_CPU_EXPORT int tensorcpu_attention_values8(const float* probabilities, const float* values, float* output,
    int probability_stride, int value_stride, int output_stride, int dimensions, int length, int rows) noexcept {
    if (!probabilities || !values || !output || dimensions <= 0 || dimensions > 1024 || dimensions % 16 != 0 ||
        length <= 0 || length > 16384 || probability_stride < length || value_stride < length || output_stride < dimensions || rows <= 0 || rows > 8) return -1;
    if (!tensorcpu_attention_avx512_supported()) return 1;
    values8(probabilities, values, output, probability_stride, value_stride, output_stride, dimensions, length, rows); return 0;
}
#else
TS_CPU_EXPORT int tensorcpu_attention_avx512_supported() noexcept { return 0; }
TS_CPU_EXPORT int tensorcpu_attention_scores8(const float*, const float*, float*, int, int, int, int, int, int, float) noexcept { return 1; }
TS_CPU_EXPORT int tensorcpu_attention_values8(const float*, const float*, float*, int, int, int, int, int, int) noexcept { return 1; }
#endif
