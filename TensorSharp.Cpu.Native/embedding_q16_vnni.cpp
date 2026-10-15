// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Stateless TensorSharp-owned CPU math. No ggml dependency, allocation,
// worker threads, device state, or model cache is used by this library.
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(_WIN32)
#define TS_CPU_EXPORT extern "C" __declspec(dllexport)
#else
#define TS_CPU_EXPORT extern "C" __attribute__((visibility("default")))
#endif

TS_CPU_EXPORT int tensorcpu_abi_version() noexcept { return 1; }

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TS_VNNI __attribute__((target("avx512f,avx512bw,avx512vnni,fma,ssse3")))

TS_CPU_EXPORT int tensorcpu_q16_vnni_supported() noexcept {
    __builtin_cpu_init();
    // The compiler's runtime checks include OS support for extended registers.
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("fma") &&
        __builtin_cpu_supports("ssse3") && __builtin_cpu_supports("avx2");
}

template<int Rows, bool Packed = false>
TS_VNNI static void tile(const int8_t* weights, const int32_t* weight_sums,
    const float* weight_scales, const int16_t* inputs, const float* input_scales,
    float* outputs, int blocks, int input_stride, int output_stride, int columns) noexcept {
    __m512 sums[Rows];
    for (int row = 0; row < Rows; ++row) sums[row] = _mm512_setzero_ps();
    const __m128i low_mask = _mm_setr_epi8(0, 2, 4, 6, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128);
    const __m128i high_mask = _mm_setr_epi8(1, 3, 5, 7, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128);
    for (int block = 0; block < blocks; ++block) {
        __m512i lows[Rows], highs[Rows];
        for (int row = 0; row < Rows; ++row) lows[row] = highs[row] = _mm512_setzero_si512();
        for (int part = 0; part < 8; ++part) {
            // 16 output lanes, each containing four original signed Q8 bytes.
            const __m512i w = _mm512_loadu_si512(weights + (std::size_t)block * 512 + part * 64);
            #pragma GCC unroll 12
            for (int row = 0; row < Rows; ++row) {
                __m512i lo, hi;
                if constexpr (Packed) {
                    const auto p = reinterpret_cast<const uint8_t*>(inputs) + (std::size_t)row * input_stride + block * 64 + part * 8;
                    int32_t low_bits, high_bits;
                    std::memcpy(&low_bits, p, 4); std::memcpy(&high_bits, p + 4, 4);
                    lo = _mm512_set1_epi32(low_bits); hi = _mm512_set1_epi32(high_bits);
                } else {
                    const auto a = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(inputs + (std::size_t)row * input_stride + block * 32 + part * 4));
                    lo = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm_shuffle_epi8(a, low_mask)));
                    hi = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm_shuffle_epi8(a, high_mask)) ^ int32_t(0x80808080u));
                }
                lows[row] = _mm512_dpbusd_epi32(lows[row], lo, w);
                highs[row] = _mm512_dpbusd_epi32(highs[row], hi, w);
            }
        }
        const auto correction = _mm512_slli_epi32(_mm512_loadu_si512(weight_sums + (std::size_t)block * 16), 15);
        const auto ws = _mm512_loadu_ps(weight_scales + (std::size_t)block * 16);
        #pragma GCC unroll 12
        for (int row = 0; row < Rows; ++row) {
            // q = low + 256*high - 32768. The exact final dot is bounded
            // by 32*32768*128 = 134217728; intermediates also fit int32.
            const auto dot = _mm512_sub_epi32(_mm512_add_epi32(lows[row], _mm512_slli_epi32(highs[row], 8)), correction);
            const auto product = _mm512_mul_ps(_mm512_cvtepi32_ps(dot), ws);
            sums[row] = _mm512_fmadd_ps(product, _mm512_set1_ps(input_scales[(std::size_t)row * blocks + block]), sums[row]);
        }
    }
    const auto mask = (__mmask16)((uint32_t(1) << columns) - 1);
    for (int row = 0; row < Rows; ++row)
        _mm512_mask_storeu_ps(outputs + (std::size_t)row * output_stride, mask, sums[row]);
}

__attribute__((target("avx2"))) static void pack(const int16_t* input, uint8_t* output, int count) noexcept {
    const auto order = _mm256_setr_epi8(0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15,
        0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15);
    const auto bias = _mm256_set1_epi64x(int64_t(0x8080808000000000ull));
    for (int i = 0; i < count; i += 16) {
        auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
        value = _mm256_xor_si256(_mm256_shuffle_epi8(value, order), bias);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + (std::size_t)i * 2), value);
    }
}

TS_CPU_EXPORT int tensorcpu_q16_vnni_pack(const int16_t* input, uint8_t* output, int count) noexcept {
    if (!input || !output || count <= 0 || count % 16 != 0) return -1;
    if (!tensorcpu_q16_vnni_supported()) return 1;
    pack(input, output, count); return 0;
}

// Named vector variables avoid materializing array accumulators on the stack;
// the generic template produced repeated block spills with validated GCC 13.
TS_VNNI static void packed_tile8(const int8_t* weights, const int32_t* weight_sums,
    const float* weight_scales, const int16_t* inputs, const float* input_scales,
    float* outputs, int blocks, int input_stride, int output_stride, int columns) noexcept {
    const auto zero = _mm512_setzero_ps();
    auto s0 = zero, s1 = zero, s2 = zero, s3 = zero;
    auto s4 = zero, s5 = zero, s6 = zero, s7 = zero;
    for (int block = 0; block < blocks; ++block) {
#define TS_INIT_ROW(n) auto l##n = _mm512_setzero_si512(); auto h##n = _mm512_setzero_si512()
        TS_INIT_ROW(0); TS_INIT_ROW(1); TS_INIT_ROW(2); TS_INIT_ROW(3);
        TS_INIT_ROW(4); TS_INIT_ROW(5); TS_INIT_ROW(6); TS_INIT_ROW(7);
#undef TS_INIT_ROW
        for (int part = 0; part < 8; ++part) {
            const auto w = _mm512_loadu_si512(weights + (std::size_t)block * 512 + part * 64);
#define TS_DOT_ROW(n) { \
            const auto p = reinterpret_cast<const uint8_t*>(inputs) + std::size_t(n) * input_stride + block * 64 + part * 8; \
            int32_t low_bits, high_bits; std::memcpy(&low_bits, p, 4); std::memcpy(&high_bits, p + 4, 4); \
            l##n = _mm512_dpbusd_epi32(l##n, _mm512_set1_epi32(low_bits), w); \
            h##n = _mm512_dpbusd_epi32(h##n, _mm512_set1_epi32(high_bits), w); }
            TS_DOT_ROW(0); TS_DOT_ROW(1); TS_DOT_ROW(2); TS_DOT_ROW(3);
            TS_DOT_ROW(4); TS_DOT_ROW(5); TS_DOT_ROW(6); TS_DOT_ROW(7);
#undef TS_DOT_ROW
        }
        const auto correction = _mm512_slli_epi32(_mm512_loadu_si512(weight_sums + (std::size_t)block * 16), 15);
        const auto ws = _mm512_loadu_ps(weight_scales + (std::size_t)block * 16);
#define TS_SCALE_ROW(n) { \
        const auto dot = _mm512_sub_epi32(_mm512_add_epi32(l##n, _mm512_slli_epi32(h##n, 8)), correction); \
        const auto product = _mm512_mul_ps(_mm512_cvtepi32_ps(dot), ws); \
        s##n = _mm512_fmadd_ps(product, _mm512_set1_ps(input_scales[std::size_t(n) * blocks + block]), s##n); }
        TS_SCALE_ROW(0); TS_SCALE_ROW(1); TS_SCALE_ROW(2); TS_SCALE_ROW(3);
        TS_SCALE_ROW(4); TS_SCALE_ROW(5); TS_SCALE_ROW(6); TS_SCALE_ROW(7);
#undef TS_SCALE_ROW
    }
    const auto mask = (__mmask16)((uint32_t(1) << columns) - 1);
#define TS_STORE_ROW(n) _mm512_mask_storeu_ps(outputs + std::size_t(n) * output_stride, mask, s##n)
    TS_STORE_ROW(0); TS_STORE_ROW(1); TS_STORE_ROW(2); TS_STORE_ROW(3);
    TS_STORE_ROW(4); TS_STORE_ROW(5); TS_STORE_ROW(6); TS_STORE_ROW(7);
#undef TS_STORE_ROW
}

TS_VNNI static void packed_dispatch(const int8_t* w, const int32_t* sums, const float* ws,
    const int16_t* a, const float* as, float* y, int blocks, int stride, int outstride, int rows, int cols) noexcept {
    switch (rows) {
#define TS_PACKED_CASE(n) case n: tile<n, true>(w, sums, ws, a, as, y, blocks, stride, outstride, cols); break
        TS_PACKED_CASE(1); TS_PACKED_CASE(2); TS_PACKED_CASE(3); TS_PACKED_CASE(4);
        TS_PACKED_CASE(5); TS_PACKED_CASE(6); TS_PACKED_CASE(7);
        case 8: packed_tile8(w, sums, ws, a, as, y, blocks, stride, outstride, cols); break;
        TS_PACKED_CASE(9); TS_PACKED_CASE(10); TS_PACKED_CASE(11); TS_PACKED_CASE(12);
#undef TS_PACKED_CASE
    }
}

TS_CPU_EXPORT int tensorcpu_q16_vnni_packed_tile(const int8_t* weights, const int32_t* weight_sums,
    const float* weight_scales, const uint8_t* inputs, const float* input_scales,
    float* outputs, int blocks, int input_stride_bytes, int output_stride, int rows, int columns) noexcept {
    if (!weights || !weight_sums || !weight_scales || !inputs || !input_scales || !outputs ||
        blocks <= 0 || input_stride_bytes <= 0 || blocks > input_stride_bytes / 64 ||
        output_stride < columns || columns <= 0 || columns > 16 || rows <= 0 || rows > 12) return -1;
    if (!tensorcpu_q16_vnni_supported()) return 1;
    packed_dispatch(weights, weight_sums, weight_scales, reinterpret_cast<const int16_t*>(inputs), input_scales,
        outputs, blocks, input_stride_bytes, output_stride, rows, columns);
    return 0;
}

// One existing CpuWorkerPool work item: at most 24 rows by 64 columns.
// This amortizes managed/native transitions without creating native workers.
TS_CPU_EXPORT int tensorcpu_q16_vnni_packed_group(const int8_t* weights, const int32_t* weight_sums,
    const float* weight_scales, const uint8_t* inputs, const float* input_scales,
    float* outputs, int blocks, int input_stride_bytes, int output_stride, int rows, int columns) noexcept {
    if (!weights || !weight_sums || !weight_scales || !inputs || !input_scales || !outputs ||
        blocks <= 0 || input_stride_bytes <= 0 || blocks > input_stride_bytes / 64 ||
        output_stride < columns || columns <= 0 || columns > 64 || rows <= 0 || rows > 24) return -1;
    if (!tensorcpu_q16_vnni_supported()) return 1;
    for (int col = 0; col < columns; col += 16)
    for (int row = 0; row < rows; row += 8) {
        const auto tile_index = std::size_t(col / 16) * blocks;
        packed_dispatch(weights + tile_index * 512, weight_sums + tile_index * 16, weight_scales + tile_index * 16,
            reinterpret_cast<const int16_t*>(inputs + (std::size_t)row * input_stride_bytes), input_scales + (std::size_t)row * blocks,
            outputs + (std::size_t)row * output_stride + col, blocks, input_stride_bytes, output_stride,
            rows - row < 8 ? rows - row : 8, columns - col < 16 ? columns - col : 16);
    }
    return 0;
}

TS_CPU_EXPORT int tensorcpu_q16_vnni_tile(const int8_t* weights, const int32_t* weight_sums,
    const float* weight_scales, const int16_t* inputs, const float* input_scales,
    float* outputs, int blocks, int input_stride, int output_stride, int rows, int columns) noexcept {
    if (!weights || !weight_sums || !weight_scales || !inputs || !input_scales || !outputs ||
        blocks <= 0 || input_stride <= 0 || blocks > input_stride / 32 ||
        output_stride < columns || columns <= 0 || columns > 16 || rows <= 0 || rows > 12) return -1;
    if (!tensorcpu_q16_vnni_supported()) return 1;
    switch (rows) {
#define TS_CASE(n) case n: tile<n>(weights, weight_sums, weight_scales, inputs, input_scales, outputs, blocks, input_stride, output_stride, columns); return 0
        TS_CASE(1); TS_CASE(2); TS_CASE(3); TS_CASE(4); TS_CASE(5); TS_CASE(6);
        TS_CASE(7); TS_CASE(8); TS_CASE(9); TS_CASE(10); TS_CASE(11); TS_CASE(12);
#undef TS_CASE
        default: return -1;
    }
}
#else
TS_CPU_EXPORT int tensorcpu_q16_vnni_supported() noexcept { return 0; }
TS_CPU_EXPORT int tensorcpu_q16_vnni_tile(const int8_t*, const int32_t*, const float*,
    const int16_t*, const float*, float*, int, int, int, int, int) noexcept { return 1; }
TS_CPU_EXPORT int tensorcpu_q16_vnni_pack(const int16_t*, uint8_t*, int) noexcept { return 1; }
TS_CPU_EXPORT int tensorcpu_q16_vnni_packed_tile(const int8_t*, const int32_t*, const float*,
    const uint8_t*, const float*, float*, int, int, int, int, int) noexcept { return 1; }
TS_CPU_EXPORT int tensorcpu_q16_vnni_packed_group(const int8_t*, const int32_t*, const float*,
    const uint8_t*, const float*, float*, int, int, int, int, int) noexcept { return 1; }
#endif
