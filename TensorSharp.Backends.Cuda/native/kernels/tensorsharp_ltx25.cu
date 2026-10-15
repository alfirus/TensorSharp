// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// TensorSharp-owned small-batch AV projections. No upstream ggml sources or
// modified dependency tree are required. K-quant byte layouts match GGUF.
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ int ltx_scale_k4(const uint8_t* s, int i) {
    return i < 4 ? s[i] & 63 : (s[i + 4] & 15) | ((s[i - 4] >> 6) << 4);
}
__device__ __forceinline__ int ltx_min_k4(const uint8_t* s, int i) {
    return i < 4 ? s[i + 4] & 63 : (s[i + 4] >> 4) | ((s[i] >> 6) << 4);
}

// Keep the multiplication/subtraction association and lane reduction equal to
// the existing scalar F32 quantized path. Only cross-row weight reuse changes.
__device__ __forceinline__ float ltx_qvalue(const uint8_t* row, int type, int col) {
    const int t = col & 255;
    if (type == 12 || type == 13) {
        const uint8_t* block = row + (col / 256) * (type == 12 ? 144 : 176);
        const float d = __half2float(*reinterpret_cast<const half*>(block));
        const float dm = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales = block + 4;
        const int sub = t / 32, pair = sub / 2, lane = t & 31;
        const uint8_t packed = block[(type == 12 ? 16 : 48) + pair * 32 + lane];
        int q = (sub & 1) ? packed >> 4 : packed & 15;
        if (type == 13) q |= ((block[16 + lane] >> sub) & 1) << 4;
        return d * (float)ltx_scale_k4(scales, sub) * (float)q - dm * (float)ltx_min_k4(scales, sub);
    }
    const uint8_t* block = row + (col / 256) * 210;
    const int h = t / 128, pos = t & 127, group = pos / 32, lane = pos & 31;
    const uint8_t* ql = block + h * 64;
    const uint8_t* qh = block + 128 + h * 32;
    const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192) + h * 8;
    const float d = __half2float(*reinterpret_cast<const half*>(block + 208));
    int q;
    if (group == 0) q = ((ql[lane] & 15) | (((qh[lane] >> 0) & 3) << 4)) - 32;
    else if (group == 1) q = ((ql[lane + 32] & 15) | (((qh[lane] >> 2) & 3) << 4)) - 32;
    else if (group == 2) q = ((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
    else q = ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
    return d * (float)scales[lane / 16 + group * 2] * (float)q;
}

template<int Tile>
__device__ __forceinline__ void ltx_quant_projection(const uint8_t* weights, const float* input, float* output,
    int type, int inner, int columns, int rows) {
    const int column = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31, start = blockIdx.y * Tile;
    if (column >= columns || start >= rows) return;
    const int row_bytes = inner / 256 * (type == 12 ? 144 : type == 13 ? 176 : 210);
    const uint8_t* weight = weights + (size_t)column * row_bytes;
    const float* x = input + (size_t)start * inner;
    const int count = min(Tile, rows - start);
    float acc[Tile];
    #pragma unroll
    for (int r = 0; r < Tile; ++r) acc[r] = 0.f;
    for (int k = lane; k < inner; k += 32) {
        const float w = ltx_qvalue(weight, type, k);
        #pragma unroll
        for (int r = 0; r < Tile; ++r)
            if (r < count) acc[r] += w * x[(size_t)r * inner + k];
    }
    #pragma unroll
    for (int r = 0; r < Tile; ++r) {
        if (r >= count) break;
        float sum = acc[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) output[(size_t)(start + r) * columns + column] = sum;
    }
}

extern "C" __global__ void ts_ltx25_quant_tile8_f32(const uint8_t* w, const float* x, float* y,
    int type, int inner, int columns, int rows) {
    ltx_quant_projection<8>(w, x, y, type, inner, columns, rows);
}
extern "C" __global__ void ts_ltx25_quant_tile16_f32(const uint8_t* w, const float* x, float* y,
    int type, int inner, int columns, int rows) {
    ltx_quant_projection<16>(w, x, y, type, inner, columns, rows);
}

// GGUF rows contain whole 256-element superblocks, so conversion can ignore
// matrix row boundaries entirely. One warp owns a superblock and emits its
// eight coalesced 32-value groups without per-element row division or the
// generic decoder's runtime format branch tree. Rounding matches the existing
// Float32 decoder followed by __float2half (round-to-nearest-even).
template<int Type, int Bytes>
__device__ __forceinline__ void ltx_dequant_k_f16(const uint8_t* weights, half* output,
    long long blocks) {
    const long long block = (long long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (block >= blocks) return;
    const int lane = threadIdx.x & 31;
    const uint8_t* source = weights + block * Bytes;
    half* target = output + block * 256;
    #pragma unroll
    for (int group = 0; group < 8; ++group) {
        const int column = group * 32 + lane;
        target[column] = __float2half(ltx_qvalue(source, Type, column));
    }
}

extern "C" __global__ void ts_ltx25_dequant_k_f16(const uint8_t* weights, half* output,
    int type, long long blocks) {
    if (type == 12) ltx_dequant_k_f16<12, 144>(weights, output, blocks);
    else if (type == 13) ltx_dequant_k_f16<13, 176>(weights, output, blocks);
    else if (type == 14) ltx_dequant_k_f16<14, 210>(weights, output, blocks);
}
