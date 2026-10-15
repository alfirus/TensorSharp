// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Validation-only resident F32 weight layout. No production integration.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__linux__) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TSG_F32_PACKED_TARGET __attribute__((target("avx512f,fma"), noinline))
#define TSG_F32_PACKED_COMPILED 1
#else
#define TSG_F32_PACKED_COMPILED 0
#endif

namespace tsg_validation { namespace prepacked_f32 {

struct dimensions { int k, m, n; };
struct input_span { const float * data; size_t count; };
struct output_span { float * data; size_t count; };

inline bool geometry(dimensions s) noexcept {
    return s.k > 0 && s.k <= 65536 && s.m > 0 && s.m <= 65536 && s.m % 16 == 0 &&
        s.n > 0 && s.n <= 8192;
}
inline bool span(const void * p, size_t count) noexcept {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && count <= std::numeric_limits<size_t>::max() / sizeof(float) &&
        address <= std::numeric_limits<uintptr_t>::max() - count * sizeof(float);
}
inline bool disjoint(const void * a, size_t ac, const void * b, size_t bc) noexcept {
    if (!span(a, ac) || !span(b, bc)) return false;
    const auto av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
    return av + ac * sizeof(float) <= bv || bv + bc * sizeof(float) <= av;
}
inline bool admissible(dimensions s, input_span w, input_span x, output_span y) noexcept {
    if (!geometry(s) || w.count != size_t(s.k) * s.m || x.count != size_t(s.k) * s.n || y.count != size_t(s.m) * s.n) return false;
    return disjoint(w.data, w.count, x.data, x.count) && disjoint(w.data, w.count, y.data, y.count) &&
        disjoint(x.data, x.count, y.data, y.count);
}
inline size_t packed_index(int k, int channel, int inner) noexcept {
    return size_t(channel / 16) * k * 16 + size_t(inner) * 16 + channel % 16;
}
inline bool pack(dimensions s, input_span original, output_span packed) noexcept {
    if (!geometry(s) || original.count != size_t(s.k) * s.m || packed.count != original.count ||
        !disjoint(original.data, original.count, packed.data, packed.count)) return false;
    for (int channel = 0; channel < s.m; ++channel) for (int inner = 0; inner < s.k; ++inner)
        std::memcpy(packed.data + packed_index(s.k, channel, inner), original.data + size_t(channel) * s.k + inner, sizeof(float));
    return true;
}
inline bool pack_bits_equal(dimensions s, input_span original, input_span packed) noexcept {
    if (!geometry(s) || original.count != size_t(s.k) * s.m || packed.count != original.count ||
        !span(original.data, original.count) || !span(packed.data, packed.count)) return false;
    for (int channel = 0; channel < s.m; ++channel) for (int inner = 0; inner < s.k; ++inner)
        if (std::memcmp(packed.data + packed_index(s.k, channel, inner), original.data + size_t(channel) * s.k + inner, sizeof(float))) return false;
    return true;
}
inline bool supported() noexcept {
#if TSG_F32_PACKED_COMPILED
    __builtin_cpu_init();
    // These compiler queries include OS support for extended register state.
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

// A worker owns a consecutive range of complete 16-channel blocks. Keeping
// channels outside token tiles reuses one K*16 weight panel across all rows.
inline int block_begin(int blocks, int ith, int nth) noexcept { return blocks * ith / nth; }
inline int block_end(int blocks, int ith, int nth) noexcept { return blocks * (ith + 1) / nth; }

#if TSG_F32_PACKED_COMPILED
template<int Rows>
TSG_F32_PACKED_TARGET static void tile(const float * w, const float * x, float * y, int k, int m) noexcept {
    static_assert(Rows >= 1 && Rows <= 8, "One tile contains 1..8 rows");
    const auto zero = _mm512_setzero_ps();
    auto s0 = zero, s1 = zero, s2 = zero, s3 = zero;
    auto s4 = zero, s5 = zero, s6 = zero, s7 = zero;
    for (int inner = 0; inner < k; ++inner) {
        const auto weight = _mm512_loadu_ps(w + size_t(inner) * 16);
#define TSG_F32_PACKED_FMA(row) if constexpr (Rows > row) { s##row = _mm512_fmadd_ps(weight, _mm512_set1_ps(x[size_t(row) * k + inner]), s##row); }
        TSG_F32_PACKED_FMA(0); TSG_F32_PACKED_FMA(1); TSG_F32_PACKED_FMA(2); TSG_F32_PACKED_FMA(3);
        TSG_F32_PACKED_FMA(4); TSG_F32_PACKED_FMA(5); TSG_F32_PACKED_FMA(6); TSG_F32_PACKED_FMA(7);
#undef TSG_F32_PACKED_FMA
    }
#define TSG_F32_PACKED_STORE(row) if constexpr (Rows > row) { _mm512_storeu_ps(y + size_t(row) * m, s##row); }
    TSG_F32_PACKED_STORE(0); TSG_F32_PACKED_STORE(1); TSG_F32_PACKED_STORE(2); TSG_F32_PACKED_STORE(3);
    TSG_F32_PACKED_STORE(4); TSG_F32_PACKED_STORE(5); TSG_F32_PACKED_STORE(6); TSG_F32_PACKED_STORE(7);
#undef TSG_F32_PACKED_STORE
}
TSG_F32_PACKED_TARGET static void dispatch(const float * w, const float * x, float * y, int k, int m, int rows) noexcept {
    switch (rows) {
#define TSG_F32_PACKED_CASE(rows) case rows: tile<rows>(w, x, y, k, m); break
        TSG_F32_PACKED_CASE(1); TSG_F32_PACKED_CASE(2); TSG_F32_PACKED_CASE(3); TSG_F32_PACKED_CASE(4);
        TSG_F32_PACKED_CASE(5); TSG_F32_PACKED_CASE(6); TSG_F32_PACKED_CASE(7); TSG_F32_PACKED_CASE(8);
#undef TSG_F32_PACKED_CASE
    }
}
#endif

// Return false before any target instruction or output write on an unsupported
// platform, invalid geometry, overlap, span overflow or invalid partition.
inline bool project_partition(dimensions s, input_span w, input_span x, output_span y, int ith, int nth,
    uint64_t * tiles = nullptr, uint64_t * elements = nullptr) noexcept {
    if (!admissible(s, w, x, y) || nth < 1 || nth > 8 || ith < 0 || ith >= nth || !supported()) return false;
#if TSG_F32_PACKED_COMPILED
    uint64_t completed_tiles = 0, completed_elements = 0;
    const int blocks = s.m / 16;
    for (int block = block_begin(blocks, ith, nth); block < block_end(blocks, ith, nth); ++block) {
        for (int row = 0; row < s.n; row += 8) {
            const int rows = std::min(8, s.n - row);
            dispatch(w.data + size_t(block) * s.k * 16, x.data + size_t(row) * s.k, y.data + size_t(row) * s.m + block * 16,
                s.k, s.m, rows);
            // Optional untimed coverage. No shared atomic counters are updated.
            if (tiles) ++completed_tiles;
            if (elements) completed_elements += uint64_t(rows) * 16;
        }
    }
    if (tiles) *tiles = completed_tiles;
    if (elements) *elements = completed_elements;
    return true;
#else
    (void)tiles; (void)elements;
    return false;
#endif
}

}} // namespace tsg_validation::prepacked_f32
#undef TSG_F32_PACKED_TARGET
