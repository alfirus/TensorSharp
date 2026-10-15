// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Validation-only adapter. Link the unchanged owned embedding_q16_vnni.cpp.
#pragma once
#include "ggml.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

extern "C" int tensorcpu_q16_vnni_supported() noexcept;
extern "C" int tensorcpu_q16_vnni_pack(const int16_t *, uint8_t *, int) noexcept;
extern "C" int tensorcpu_q16_vnni_packed_group(const int8_t *, const int32_t *, const float *,
    const uint8_t *, const float *, float *, int, int, int, int, int) noexcept;

namespace tsg_validation {
struct guarded_bytes {
    static constexpr size_t guard = 64;
    static constexpr uint8_t canary = 0xa5;
    uint8_t * storage;
    size_t bytes, allocated;
    explicit guarded_bytes(size_t count) : bytes(count), allocated((count + 2 * guard + 63) / 64 * 64) {
        storage = static_cast<uint8_t *>(std::aligned_alloc(64, allocated));
        if (!storage) throw std::bad_alloc();
        std::memset(storage, canary, allocated);
        std::memset(data(), 0, bytes);
    }
    ~guarded_bytes() { std::free(storage); }
    guarded_bytes(const guarded_bytes &) = delete;
    guarded_bytes & operator=(const guarded_bytes &) = delete;
    uint8_t * data() { return storage + guard; }
    const uint8_t * data() const { return storage + guard; }
    bool guards() const {
        for (size_t i = 0; i < guard; ++i) if (storage[i] != canary) return false;
        for (size_t i = guard + bytes; i < allocated; ++i) if (storage[i] != canary) return false;
        return true;
    }
    uint64_t fingerprint() const {
        uint64_t value = 14695981039346656037ULL;
        for (size_t i = 0; i < allocated; ++i) value = (value ^ storage[i]) * 1099511628211ULL;
        return value;
    }
};

// Match the retained managed AVX2 quantizer's block maximum, F32 scale,
// reciprocal/division and nearest-even int16 saturation. No fast-math.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) inline int quantize_q16(const float * x, int16_t * output, float * scales, int blocks) noexcept {
    bool fallback = false;
    const auto absolute_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int block = 0; block < blocks; ++block) {
        const auto a = _mm256_loadu_ps(x + block * 32);
        const auto b = _mm256_loadu_ps(x + block * 32 + 8);
        const auto c = _mm256_loadu_ps(x + block * 32 + 16);
        const auto d = _mm256_loadu_ps(x + block * 32 + 24);
        auto maximum = _mm256_max_ps(_mm256_and_ps(a, absolute_mask), _mm256_and_ps(b, absolute_mask));
        maximum = _mm256_max_ps(maximum, _mm256_max_ps(_mm256_and_ps(c, absolute_mask), _mm256_and_ps(d, absolute_mask)));
        maximum = _mm256_max_ps(maximum, _mm256_permute2f128_ps(maximum, maximum, 1));
        maximum = _mm256_max_ps(maximum, _mm256_shuffle_ps(maximum, maximum, 0x4e));
        maximum = _mm256_max_ps(maximum, _mm256_shuffle_ps(maximum, maximum, 0xb1));
        const float amax = _mm_cvtss_f32(_mm256_castps256_ps128(maximum));
        const float scale = amax / 32767.f;
        const auto invalid = _mm256_or_ps(_mm256_cmp_ps(a, a, _CMP_UNORD_Q), _mm256_cmp_ps(b, b, _CMP_UNORD_Q));
        const auto invalid_cd = _mm256_or_ps(_mm256_cmp_ps(c, c, _CMP_UNORD_Q), _mm256_cmp_ps(d, d, _CMP_UNORD_Q));
        if (_mm256_movemask_ps(_mm256_or_ps(invalid, invalid_cd)) || !std::isfinite(scale)) return -1;
        if (amax > 0 && scale < 1.17549435e-38f) {
            fallback = true; scales[block] = 0;
            std::memset(output + block * 32, 0, 32 * sizeof(int16_t));
            continue;
        }
        scales[block] = scale;
        const float inverse = scale == 0 ? 0 : 1.f / scale;
        const auto multiplier = _mm256_set1_ps(inverse), divisor = _mm256_set1_ps(scale);
        const bool multiply = std::isfinite(inverse);
        const auto q0 = _mm256_cvtps_epi32(multiply ? _mm256_mul_ps(a, multiplier) : _mm256_div_ps(a, divisor));
        const auto q1 = _mm256_cvtps_epi32(multiply ? _mm256_mul_ps(b, multiplier) : _mm256_div_ps(b, divisor));
        const auto q2 = _mm256_cvtps_epi32(multiply ? _mm256_mul_ps(c, multiplier) : _mm256_div_ps(c, divisor));
        const auto q3 = _mm256_cvtps_epi32(multiply ? _mm256_mul_ps(d, multiplier) : _mm256_div_ps(d, divisor));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + block * 32), _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), 0xd8));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + block * 32 + 16), _mm256_permute4x64_epi64(_mm256_packs_epi32(q2, q3), 0xd8));
    }
    return fallback ? 1 : 0;
}
#else
inline int quantize_q16(const float *, int16_t *, float *, int) noexcept { return -1; }
#endif

struct q16_workspace {
    const int k, m, n, blocks, column_tiles;
    guarded_bytes weights, weight_sums, weight_scales, activations, packed, scales, fallback;
    uint64_t weights_hash, sums_hash, weight_scales_hash;
    std::atomic<int> error{0};
    std::atomic<uint64_t> prepared_rows{0}, executed_groups{0}, fallback_rows{0};
    q16_workspace(const std::vector<unsigned char> & original, int inner, int columns, int rows) :
        k(inner), m(columns), n(rows), blocks(k / 32), column_tiles((m + 15) / 16),
        weights(size_t(column_tiles) * blocks * 512), weight_sums(size_t(column_tiles) * blocks * 16 * sizeof(int32_t)),
        weight_scales(size_t(column_tiles) * blocks * 16 * sizeof(float)), activations(size_t(k) * n * sizeof(int16_t)),
        packed(size_t(k) * n * 2), scales(size_t(n) * blocks * sizeof(float)), fallback(n) {
        if (k <= 0 || k % 32 || m <= 0 || n <= 0 || original.size() != size_t(k / 32) * 34 * m)
            throw std::runtime_error("Invalid original-Q8 Q16 experiment geometry");
        auto * ws = reinterpret_cast<float *>(weight_scales.data());
        auto * sums = reinterpret_cast<int32_t *>(weight_sums.data());
        auto * w = reinterpret_cast<int8_t *>(weights.data());
        for (int col = 0; col < m; ++col) for (int block = 0; block < blocks; ++block) {
            const size_t src = (size_t(col) * blocks + block) * 34;
            const size_t tile_block = size_t(col / 16) * blocks + block;
            const int lane = col % 16;
            ggml_fp16_t bits; std::memcpy(&bits, original.data() + src, sizeof(bits));
            ws[tile_block * 16 + lane] = ggml_fp16_to_fp32(bits);
            if (!std::isfinite(ws[tile_block * 16 + lane])) throw std::runtime_error("Nonfinite Q8 weight scale");
            for (int dim = 0; dim < 32; ++dim) {
                const int8_t quant = static_cast<int8_t>(original[src + 2 + dim]);
                w[tile_block * 512 + dim / 4 * 64 + lane * 4 + dim % 4] = quant;
                sums[tile_block * 16 + lane] += quant;
            }
        }
        weights_hash = weights.fingerprint(); sums_hash = weight_sums.fingerprint(); weight_scales_hash = weight_scales.fingerprint();
    }
    void check_storage() const {
        if (!weights.guards() || !weight_sums.guards() || !weight_scales.guards() || !activations.guards() ||
            !packed.guards() || !scales.guards() || !fallback.guards() || weights.fingerprint() != weights_hash ||
            weight_sums.fingerprint() != sums_hash || weight_scales.fingerprint() != weight_scales_hash)
            throw std::runtime_error("Q16 private operand/scratch canary or immutable weight fingerprint changed");
        if (error.load()) throw std::runtime_error("Q16 native preparation/group callback failed");
    }
};

inline void q16_prepare(ggml_tensor * dst, int ith, int nth, void * userdata) noexcept {
    auto & ws = *static_cast<q16_workspace *>(userdata);
    const auto * x = static_cast<const float *>(dst->src[0]->data);
    uint64_t prepared = 0, fallback_count = 0;
    for (int row = ith; row < ws.n; row += nth) {
        const int status = quantize_q16(x + size_t(row) * ws.k, reinterpret_cast<int16_t *>(ws.activations.data()) + size_t(row) * ws.k,
            reinterpret_cast<float *>(ws.scales.data()) + size_t(row) * ws.blocks, ws.blocks);
        if (status < 0) { ws.error.store(1); continue; }
        ws.fallback.data()[row] = status != 0;
        if (status) ++fallback_count;
        if (tensorcpu_q16_vnni_pack(reinterpret_cast<const int16_t *>(ws.activations.data()) + size_t(row) * ws.k,
            ws.packed.data() + size_t(row) * ws.k * 2, ws.k) != 0) ws.error.store(2);
        ++prepared;
    }
    ws.prepared_rows.fetch_add(prepared);
    ws.fallback_rows.fetch_add(fallback_count);
    // One marker per actual worker; graph dependency completes preparation
    // before any group consumes shared packed activations.
    static_cast<float *>(dst->data)[ith] = 1;
}

inline void q16_project(ggml_tensor * dst, int ith, int nth, void * userdata) noexcept {
    auto & ws = *static_cast<q16_workspace *>(userdata);
    const auto * x = static_cast<const float *>(dst->src[1]->data);
    auto * y = static_cast<float *>(dst->data);
    const int column_groups = (ws.column_tiles + 3) / 4, row_groups = (ws.n + 15) / 16;
    uint64_t groups = 0;
    for (int work = ith; work < column_groups * row_groups; work += nth) {
        const int first_tile = work % column_groups * 4, first_row = work / column_groups * 16;
        const int rows = std::min(16, ws.n - first_row), columns = std::min(64, ws.m - first_tile * 16);
        const size_t tile_block = size_t(first_tile) * ws.blocks;
        if (tensorcpu_q16_vnni_packed_group(reinterpret_cast<const int8_t *>(ws.weights.data()) + tile_block * 512,
            reinterpret_cast<const int32_t *>(ws.weight_sums.data()) + tile_block * 16,
            reinterpret_cast<const float *>(ws.weight_scales.data()) + tile_block * 16,
            ws.packed.data() + size_t(first_row) * ws.k * 2,
            reinterpret_cast<const float *>(ws.scales.data()) + size_t(first_row) * ws.blocks,
            y + size_t(first_row) * ws.m + first_tile * 16, ws.blocks, ws.k * 2, ws.m, rows, columns) != 0) ws.error.store(3);
        // Preserve the retained quantizer's precise F32 fallback for a row
        // whose block scale is too small for representable Q16 precision.
        for (int row = first_row; row < first_row + rows; ++row) if (ws.fallback.data()[row])
            for (int col = first_tile * 16; col < first_tile * 16 + columns; ++col) {
                float sum = 0;
                for (int block = 0; block < ws.blocks; ++block) for (int dim = 0; dim < 32; ++dim) {
                    const size_t tb = size_t(col / 16) * ws.blocks + block;
                    const int quant = reinterpret_cast<const int8_t *>(ws.weights.data())[tb * 512 + dim / 4 * 64 + col % 16 * 4 + dim % 4];
                    const float scale = reinterpret_cast<const float *>(ws.weight_scales.data())[tb * 16 + col % 16];
                    sum = std::fma(x[size_t(row) * ws.k + block * 32 + dim], quant * scale, sum);
                }
                y[size_t(row) * ws.m + col] = sum;
            }
        ++groups;
    }
    ws.executed_groups.fetch_add(groups);
}
} // namespace tsg_validation
