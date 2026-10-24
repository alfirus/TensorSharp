// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_q8_precision.h"
#include "ggml_ops_q8_prefill_policy.h"
#include "ggml_ops_q8_ordered_policy.h"
#include "ggml_ops_q8_hc_compact_policy.h"
#include "ggml_ops_q8_kda_compact_policy.h"
#include "ggml_ops_precision_policy.h"
#if defined(TSG_GGML_TEST_HOOKS)
#include "tests/q8_precision_test_hooks.h"
#include "tests/q8_hc_compact_test_hooks.h"
#endif
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <atomic>

namespace {
// Diagnostic opt-in while model/prefill/decode qualification is in progress.
// Fix the policy at first use: changing an environment variable must not mix
// arithmetic inside already captured CUDA graphs. The qualified default below
// retains its K-increasing sum and cross-column bitwise contract.
bool parallel_vector_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("TS_GGML_Q8_PARALLEL_VECTOR");
        const bool selected = value && std::strcmp(value, "1") == 0;
        if (selected)
            std::fprintf(stderr, "[q8-f32] Experimental parallel-K F32 vector selected (N=1).\n");
        return selected;
    }();
    return enabled;
}

bool parallel_small_batch_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("TS_GGML_Q8_PARALLEL_SMALL_BATCH");
        const bool selected = value && std::strcmp(value, "1") == 0;
        if (selected)
            std::fprintf(stderr, "[q8-f32] Experimental parallel-K F32 small batch selected (2<=N<=8).\n");
        return selected;
    }();
    return enabled;
}

bool ordered_small_batch_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("TS_GGML_Q8_ORDERED_SMALL_BATCH");
        const bool selected = value && std::strcmp(value, "1") == 0;
        if (selected)
            std::fprintf(stderr, "[q8-f32] Experimental K-ordered independent columns selected (2<=N<=8; K128/M8192 limited to N<=4).\n");
        return selected;
    }();
    return enabled;
}

bool ordered_gate_prefill_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("TS_GGML_Q8_ORDERED_GATE_PREFILL");
        const bool selected = value && std::strcmp(value, "1") == 0;
        if (selected)
            std::fprintf(stderr, "[q8-f32] Experimental K-ordered Flash gate prefill selected (K4096; M64/N9..336 or M128/N9..128).\n");
        return selected;
    }();
    return enabled;
}

bool gate_prefill_tile16_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("TS_GGML_Q8_GATE_PREFILL_TILE16");
        const bool selected = value && std::strcmp(value, "1") == 0;
        if (selected)
            std::fprintf(stderr, "[q8-f32] Experimental K-ordered Flash gate tile16 selected (K4096/M64-or128; N64..512, after ordered-column policy).\n");
        return selected;
    }();
    return enabled;
}

int prefill_tile_columns() {
    static const int columns = [] {
        const char * value = std::getenv("TS_GGML_Q8_PREFILL_TILE");
        // Automatic tiling changes reuse and launch geometry only. Every
        // output retains the same K-increasing FMA and original F32 operands.
        const int selected = !value || !*value || std::strcmp(value, "auto") == 0 ? 0
            : std::strcmp(value, "128") == 0 ? 128
            : value && std::strcmp(value, "64") == 0 ? 64 : 32;
        if (selected > 32)
            std::fprintf(stderr, "[q8-f32] Experimental K-ordered prefill column tile selected: %d.\n", selected);
        return selected;
    }();
    return columns;
}

const TsgQ8HcOption & hc_compact_option() {
    static const TsgQ8HcOption option = tsg_q8_hc_parse_option(std::getenv("TS_GGML_Q8_HC_COMPACT"));
    return option;
}

const TsgQ8KdaOption & kda_compact_option() {
    static const TsgQ8KdaOption option = tsg_q8_kda_parse_option(std::getenv("TS_GGML_Q8_KDA_COMPACT"));
    return option;
}

// Successful enqueue only, once per selected variant/tile. This is not a
// completion certificate or per-request/phase coverage. No clock or payload.
std::atomic<unsigned> & hc_reported_bits() {
    static std::atomic<unsigned> reported{0};
    return reported;
}
void report_hc_compact(int variant, int columns, int tile) {
    const unsigned bit = variant == 1 ? 1u : tile == 16 ? 2u
        : tile == 32 ? 4u : tile == 64 ? 8u : 16u;
    if ((hc_reported_bits().fetch_or(bit, std::memory_order_relaxed) & bit) != 0) return;
    std::fprintf(stderr, "[q8-hc-compact] selected=%s inner=16384 rows=24 columns=%d tile_rows=%d tile_columns=%d ordered_fma=1 extra_payload_bytes=0 scope=launch_accepted_not_completion\n",
        variant == 1 ? "row_warp" : "compact_tile", columns, variant == 1 ? 1 : 32, tile);
}

std::atomic<unsigned> & kda_reported_bits() {
    static std::atomic<unsigned> reported{0};
    return reported;
}
void report_kda_compact(int variant, int rows, int columns) {
    const unsigned bit = 1u << ((rows == 64 ? 0 : 2) + (variant == 1 ? 0 : 1));
    if ((kda_reported_bits().fetch_or(bit, std::memory_order_relaxed) & bit) != 0) return;
    std::fprintf(stderr, "[q8-kda-compact] selected=%s inner=4096 rows=%d columns=%d tile_rows=%d tile_columns=%d ordered_fma=1 extra_payload_bytes=0 scope=launch_accepted_not_completion\n",
        variant == 1 ? "row_warp" : "compact_tile", rows, columns,
        variant == 1 ? 1 : 32, variant == 1 ? 1 : 16);
}

// One full warp per output row. The decoded Q8 product and F32 activation are
// unchanged; the lane sums and five shuffle additions deliberately use a new
// reduction order. No atomics, staging allocation or input narrowing. A final
// partial CTA may contain unused whole warps, never a partial active warp.
template<bool Batched>
__global__ void q8_f32_vector_parallel(const char * weights, const char * input, float * output,
        int inner, int rows, size_t weight_stride, size_t input_inner_stride, size_t input_column_stride) {
    const int lane = int(threadIdx.x) & 31;
    const int row = int(blockIdx.x) * 4 + int(threadIdx.x) / 32;
    if (row >= rows) return;
    // Independent columns use the same lane/FMA/shuffle order as N=1. No
    // staging, duplicated weight ownership or extra device allocation.
    if constexpr (Batched) {
        input += size_t(blockIdx.y) * input_column_stride;
        output += size_t(blockIdx.y) * rows;
    }
    float sum = 0.0f;
    for (int block = 0; block < inner / 32; ++block) {
        const char * source = weights + size_t(row) * weight_stride + size_t(block) * 34;
        const float scale = __half2float(*reinterpret_cast<const half *>(source));
        const float weight = scale * float(*reinterpret_cast<const int8_t *>(source + 2 + lane));
        const float activation = *reinterpret_cast<const float *>(input + size_t(block * 32 + lane) * input_inner_stride);
        sum = fmaf(weight, activation, sum);
    }
    for (int shift = 16; shift > 0; shift /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, shift);
    if (lane == 0) output[row] = sum;
}

// Each lane owns a complete output row; independent rows provide parallelism
// without changing the reduction order. Read two adjacent signed Q8 bytes at
// once using the format's guaranteed two-byte alignment (34-byte block stride
// forbids assuming aligned 32/128-bit loads). One warp per CTA limits the active
// rows competing for L1 while the remaining bytes in each row are consumed.
// No barriers, shared/global scratch, input narrowing or parallel-K reduction.
template<bool Batched>
__global__ void q8_f32_vector(const char * weights, const char * input, float * output,
        int inner, int rows, size_t weight_stride, size_t input_inner_stride, size_t input_column_stride) {
    const int row = int(blockIdx.x) * 32 + int(threadIdx.x);
    if (row >= rows) return;
    // Each column is an independent scalar dot. Pair loads, decoded operands
    // and increasing-K FMA order are exactly the N=1 path; no shared tile pads
    // the unused columns and no synchronization or scratch is introduced.
    if constexpr (Batched) {
        input += size_t(blockIdx.y) * input_column_stride;
        output += size_t(blockIdx.y) * rows;
    }
    float sum = 0.0f;
    for (int block_index = 0; block_index < inner / 32; ++block_index) {
        const char * block = weights + size_t(row) * weight_stride + size_t(block_index) * 34;
        const float scale = __half2float(*reinterpret_cast<const half *>(block));
#pragma unroll
        for (int pair = 0; pair < 16; ++pair) {
            const uint16_t packed = *reinterpret_cast<const uint16_t *>(block + 2 + pair * 2);
            const float x0 = *reinterpret_cast<const float *>(input + size_t(block_index * 32 + pair * 2) * input_inner_stride);
            const float x1 = *reinterpret_cast<const float *>(input + size_t(block_index * 32 + pair * 2 + 1) * input_inner_stride);
            sum = fmaf(scale * float(int8_t(packed & 255)), x0, sum);
            sum = fmaf(scale * float(int8_t(packed >> 8)), x1, sum);
        }
    }
    output[row] = sum;
}

// Cooperative load only: one whole warp owns one row. ALL lanes execute both
// shuffles; only lane0 accumulates. Every original FMA remains in increasing K
// order. This is not the independent parallel-K reduction above. No shared or
// global payload scratch; tensor layout/strides and caller lifetime stay intact.
__global__ void q8_f32_hc_row_warp(const char * weights, const char * input, float * output,
        int inner, int rows, size_t weight_stride, size_t input_inner_stride) {
    const int row = int(blockIdx.x);
    if (row >= rows) return; // Uniform for the complete32-thread CTA.
    const int lane = int(threadIdx.x);
    float sum = 0.0f;
    for (int block_index = 0; block_index < inner / 32; ++block_index) {
        const char * block = weights + size_t(row) * weight_stride + size_t(block_index) * 34;
        const float scale = __half2float(*reinterpret_cast<const half *>(block));
        const float weight = scale * float(*reinterpret_cast<const int8_t *>(block + 2 + lane));
        const float activation = *reinterpret_cast<const float *>(
            input + size_t(block_index * 32 + lane) * input_inner_stride);
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            const float w = __shfl_sync(0xffffffffu, weight, k);
            const float x = __shfl_sync(0xffffffffu, activation, k);
            if (lane == 0) sum = fmaf(w, x, sum);
        }
    }
    if (lane == 0) output[row] = sum;
}

// Decode directly from the checkpoint's original 256-value/176-byte Q5_K
// block. The fifth bits and six-bit scale/minimum fields are interleaved by
// 32-value groups. Explicit F32 rounding keeps vector and tiled decoding equal
// even when the compilation enables contraction. No resident weight expansion.
__device__ __forceinline__ float q5_weight(const char * block, int element) {
    const auto * bytes = reinterpret_cast<const unsigned char *>(block);
    const int group = element / 32, lane = element % 32;
    const auto * scales = bytes + 4;
    const int scale = group < 4 ? scales[group] & 63
        : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
    const int minimum = group < 4 ? scales[group + 4] & 63
        : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    const unsigned char low = bytes[48 + (group / 2) * 32 + lane];
    const int quant = ((group & 1) ? low >> 4 : low & 15)
        | (((bytes[16 + lane] >> group) & 1) << 4);
    const float d = __half2float(*reinterpret_cast<const half *>(block));
    const float d_minimum = __half2float(*reinterpret_cast<const half *>(block + 2));
    return __fsub_rn(__fmul_rn(__fmul_rn(d, float(scale)), float(quant)),
        __fmul_rn(d_minimum, float(minimum)));
}

// A narrow lane consumes adjacent elements from the same 32-value group.
// Share that group's scale/minimum products without reassociating either
// decoded weight. Packed byte pairs require only Q5_K's two-byte alignment.
__device__ __forceinline__ void q5_narrow_weights(const unsigned char * scales,
        float d, float d_minimum, int group, uint16_t low, uint16_t high,
        float & weight0, float & weight1) {
    const int scale = group < 4 ? scales[group] & 63
        : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
    const int minimum = group < 4 ? scales[group + 4] & 63
        : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    const int low_shift = (group & 1) * 4;
    const int quant0 = ((low >> low_shift) & 15) | (((high >> group) & 1) << 4);
    const int quant1 = ((low >> (8 + low_shift)) & 15) | (((high >> (8 + group)) & 1) << 4);
    const float scaled_d = __fmul_rn(d, float(scale));
    const float scaled_minimum = __fmul_rn(d_minimum, float(minimum));
    weight0 = __fsub_rn(__fmul_rn(scaled_d, float(quant0)), scaled_minimum);
    weight1 = __fsub_rn(__fmul_rn(scaled_d, float(quant1)), scaled_minimum);
}

// One whole warp owns each narrow output. Pair adjacent products before the
// parallel-K lane reduction: summing all even and odd terms separately would
// discard small alternating residuals. The fixed lane/pair/shuffle order is
// identical at N1..N8; wide prefill retains its separate K-increasing tiles.
// No activation narrowing, atomics, split-K scratch or resident weight expansion.
__global__ void q5_f32_narrow(const char * weights, const char * input, float * output,
        int inner, int rows, size_t weight_stride, size_t input_inner_stride, size_t input_column_stride) {
    const int lane = int(threadIdx.x) & 31;
    const int row = int(blockIdx.x) * 4 + int(threadIdx.x) / 32;
    if (row >= rows) return;
    input += size_t(blockIdx.y) * input_column_stride;
    output += size_t(blockIdx.y) * rows;
    const char * row_weights = weights + size_t(row) * weight_stride;
    float sum = 0.0f;
    const int group_lane = 2 * (lane & 15);
    for (int block_index = 0; block_index < inner / 256; ++block_index) {
        const char * block = row_weights + size_t(block_index) * 176;
        const auto * bytes = reinterpret_cast<const unsigned char *>(block);
        const float d = __half2float(*reinterpret_cast<const half *>(block));
        const float d_minimum = __half2float(*reinterpret_cast<const half *>(block + 2));
        const uint16_t high = *reinterpret_cast<const uint16_t *>(bytes + 16 + group_lane);
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            // This is exactly the former k=2*lane, k+=64 traversal. Each
            // original block supplies four pairs; its half scales/high bytes
            // stay live, while each pair reuses its group's decoded metadata.
            const int group = lane / 16 + 2 * step;
            const uint16_t low = *reinterpret_cast<const uint16_t *>(bytes + 48 + 32 * step + group_lane);
            float weight0, weight1;
            q5_narrow_weights(bytes + 4, d, d_minimum, group, low, high, weight0, weight1);
            const int64_t k = int64_t(block_index) * 256 + 2 * lane + 64 * step;
            const float x0 = *reinterpret_cast<const float *>(input + size_t(k) * input_inner_stride);
            const float x1 = *reinterpret_cast<const float *>(input + size_t(k + 1) * input_inner_stride);
            const float pair = __fadd_rn(__fmul_rn(weight0, x0), __fmul_rn(weight1, x1));
            sum = __fadd_rn(sum, pair);
        }
    }
    for (int shift = 16; shift > 0; shift /= 2)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, shift));
    if (lane == 0) output[row] = sum;
}

// Each CTA owns a TileRows-row output tile. Its128 threads each accumulate
// TileRows/16 rows and Columns/8 columns. Weights widen once into shared memory
// are reused across the entire column tile, preserving quantized residency.
// Padding the shared-memory leading dimensions avoids transposed-load bank
// conflicts. Within these tiled paths every output traverses K in the same order
// for every column geometry; changing companions or moving a column to another
// tile cannot change its reduction order. Q5 narrow reduction is separate above.
// No split-K, atomics, source narrowing, or activation quants.
template<int Columns, bool Q5 = false, int TileRows = 64>
__global__ void q8_f32_tiled(const char * weights, const char * input, float * output,
        int inner, int rows, int columns, size_t weight_stride, size_t input_inner_stride,
        size_t input_column_stride) {
    static_assert(TileRows == 32 || TileRows == 64, "Unsupported Q8 row tile");
    static_assert(!Q5 || TileRows == 64, "Q5 keeps its original row tile");
    constexpr int TileInner = 32, ColumnParts = Columns / 8, RowParts = TileRows / 16;
    __shared__ float ws[TileInner][TileRows + 1];
    __shared__ float xs[TileInner][Columns + 1];
    const int row_base = int(blockIdx.x) * TileRows;
    const int column_base = int(blockIdx.y) * Columns;
    const int row_lane = threadIdx.x % 16, column_lane = threadIdx.x / 16;
    float sums[RowParts][ColumnParts] = {};
    for (int base = 0; base < inner; base += TileInner) {
        for (int index = threadIdx.x; index < TileRows * TileInner; index += 128) {
            const int row = index / TileInner, k = index % TileInner;
            float value = 0;
            if (row_base + row < rows) {
                if constexpr (Q5) {
                    const char * block = weights + size_t(row_base + row) * weight_stride + size_t(base / 256) * 176;
                    value = q5_weight(block, base % 256 + k);
                } else {
                // Q8_0 blocks contain a two-byte scale and 32 signed bytes.
                // The 34-byte block stride does not permit aligned int4 loads.
                const char * block = weights + size_t(row_base + row) * weight_stride + size_t(base / 32) * 34;
                const float scale = __half2float(*reinterpret_cast<const half *>(block));
                value = scale * float(*reinterpret_cast<const int8_t *>(block + 2 + k));
                }
            }
            ws[k][row] = value;
        }
        for (int index = threadIdx.x; index < Columns * TileInner; index += 128) {
            const int column = index / TileInner, k = index % TileInner;
            xs[k][column] = column_base + column < columns
                ? *reinterpret_cast<const float *>(input + size_t(column_base + column) * input_column_stride
                    + size_t(base + k) * input_inner_stride) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TileInner; ++k) {
            float w[RowParts], x[ColumnParts];
#pragma unroll
            for (int r = 0; r < RowParts; ++r) w[r] = ws[k][row_lane + 16 * r];
#pragma unroll
            for (int c = 0; c < ColumnParts; ++c) x[c] = xs[k][column_lane + 8 * c];
#pragma unroll
            for (int r = 0; r < RowParts; ++r)
#pragma unroll
            for (int c = 0; c < ColumnParts; ++c) sums[r][c] = fmaf(w[r], x[c], sums[r][c]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < RowParts; ++r)
#pragma unroll
    for (int c = 0; c < ColumnParts; ++c) {
        const int row = row_base + row_lane + 16 * r;
        const int column = column_base + column_lane + 8 * c;
        if (row < rows && column < columns) output[size_t(column) * rows + row] = sums[r][c];
    }
}

template<int Columns, bool Q5 = false, int TileRows = 64>
void launch(const void * weights, const void * input, float * output,
        int inner, int rows, int columns, size_t weight_stride, size_t input_inner_stride,
        size_t input_column_stride, cudaStream_t stream) {
    const dim3 grid(unsigned((int64_t(rows) + TileRows - 1) / TileRows), unsigned((columns + Columns - 1) / Columns));
    q8_f32_tiled<Columns, Q5, TileRows><<<grid, 128, 0, stream>>>(static_cast<const char *>(weights),
        static_cast<const char *>(input), output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride);
}

template<bool Q5 = false, int TileRows = 64>
int automatic_tile(int rows, int columns, int & selected) {
    if (columns < 64) { selected = 32; return int(cudaSuccess); }
    struct DevicePolicy { int device, multiprocessors, active64, active128; };
    if constexpr (TileRows == 32) {
        // One bounded last-device metadata record; changing devices re-queries
        // actual occupancy instead of retaining an unbounded heap cache.
        thread_local DevicePolicy cached{-1, 0, 0, 0};
        int device = -1;
        auto status = cudaGetDevice(&device);
        if (status != cudaSuccess) return int(status);
        if (cached.device != device) {
            DevicePolicy item{device, 0, 0, 0};
            status = cudaDeviceGetAttribute(&item.multiprocessors, cudaDevAttrMultiProcessorCount, device);
            if (status != cudaSuccess) return int(status);
            status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&item.active64, q8_f32_tiled<64, Q5, TileRows>, 128, 0);
            if (status != cudaSuccess) return int(status);
            status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&item.active128, q8_f32_tiled<128, Q5, TileRows>, 128, 0);
            if (status != cudaSuccess) return int(status);
            cached = item;
        }
        selected = tsg_q8_hc_auto_columns(rows, columns, cached.multiprocessors,
            cached.active64, cached.active128, TileRows);
        return int(cudaSuccess);
    } else {
    // Host metadata only; never retain device pointers or payload allocations.
    // Cache each device separately, without imposing a fixed maximum GPU count.
    thread_local std::vector<DevicePolicy> policies;
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return int(status);
    const DevicePolicy * policy = nullptr;
    for (const auto & item : policies) if (item.device == device) { policy = &item; break; }
    if (!policy) {
        DevicePolicy item{device, 0, 0, 0};
        status = cudaDeviceGetAttribute(&item.multiprocessors, cudaDevAttrMultiProcessorCount, device);
        if (status != cudaSuccess) return int(status);
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&item.active64, q8_f32_tiled<64, Q5, TileRows>, 128, 0);
        if (status != cudaSuccess) return int(status);
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&item.active128, q8_f32_tiled<128, Q5, TileRows>, 128, 0);
        if (status != cudaSuccess) return int(status);
        policies.push_back(item); policy = &policies.back();
        std::fprintf(stderr, "[%s-f32] Automatic K-ordered prefill tiling: device=%d SMs=%d active64=%d active128=%d; no global scratch.\n",
            Q5 ? "q5" : "q8", device, item.multiprocessors, item.active64, item.active128);
    }
    selected = tsg_q8_prefill_auto_columns(rows, columns, policy->multiprocessors, policy->active64, policy->active128);
    return int(cudaSuccess);
    }
}

int q5_cuda_launch(const void * weights, const void * input, float * output,
        int inner, int rows, int columns, size_t weight_stride, size_t input_inner_stride,
        size_t input_column_stride, cudaStream_t stream) {
    int tile = 32;
    if (columns >= 64) {
        tile = prefill_tile_columns();
        if (tile == 0) {
            const int status = automatic_tile<true>(rows, columns, tile);
            if (status != int(cudaSuccess)) return status;
        }
    }
    // The Q8-only parallel-K experiments never change this operator's
    // arithmetic. Narrow Q5 owns fixed adjacent-pair reductions; wide Q5
    // preserves its K-increasing FMA order. No cross-class bit identity promise.
    if (columns <= TSG_PRECISION_DECODE_COLUMNS) {
        const dim3 grid(unsigned((int64_t(rows) + 3) / 4), unsigned(columns));
        q5_f32_narrow<<<grid, 128, 0, stream>>>(static_cast<const char *>(weights),
            static_cast<const char *>(input), output, inner, rows, weight_stride, input_inner_stride, input_column_stride);
    }
    else if (columns <= 16) launch<16, true>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else if (columns >= 128 && tile == 128) launch<128, true>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else if (columns >= 64 && tile >= 64) launch<64, true>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else launch<32, true>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    return int(cudaGetLastError());
}
} // namespace

static int q8_cuda_launch(const void * weights, const void * input, float * output,
        int inner, int rows, int columns, size_t weight_stride, size_t input_inner_stride,
        size_t input_column_stride, void * stream_pointer, bool allow_experiments) {
    const auto stream = static_cast<cudaStream_t>(stream_pointer);
    const auto & hc = hc_compact_option();
    const auto & kda = kda_compact_option();
    if (!hc.valid || !kda.valid) return int(cudaErrorInvalidValue);
    const int hc_variant = tsg_q8_hc_selected_variant(inner, rows, columns, allow_experiments && hc.enabled);
    const int kda_variant = tsg_q8_kda_selected_variant(inner, rows, columns, allow_experiments && kda.enabled);
    const bool ordered = allow_experiments
        && ((tsg_q8_ordered_small_batch_shape(inner, rows, columns) && ordered_small_batch_enabled())
            || (tsg_q8_ordered_gate_prefill_shape(inner, rows, columns) && ordered_gate_prefill_enabled()));
    const bool gate_tile16 = allow_experiments && !ordered
        && tsg_q8_gate_prefill_tile16_shape(inner, rows, columns) && gate_prefill_tile16_enabled();
    int tile = 32;
    if ((columns >= 64 || (hc_variant == 2 && columns > 16))
            && !ordered && !gate_tile16 && kda_variant != 2) {
        tile = prefill_tile_columns();
        if (columns < 64) {
            // Previously all explicit global tiles used32 below64 columns.
            // Only the HC automatic fallback changes to16.
            tile = tile == 0 ? 16 : 32;
        } else if (tile == 0) {
            const int status = hc_variant == 2 ? automatic_tile<false, 32>(rows, columns, tile) : automatic_tile(rows, columns, tile);
            if (status != int(cudaSuccess)) return status;
        }
    }
    if (columns == 1 && parallel_vector_enabled()) {
        const unsigned blocks = unsigned((int64_t(rows) + 3) / 4);
        q8_f32_vector_parallel<false><<<blocks, 128, 0, stream>>>(static_cast<const char *>(weights),
            static_cast<const char *>(input), output, inner, rows, weight_stride, input_inner_stride, input_column_stride);
    }
    else if (hc_variant == 1 || kda_variant == 1) {
        q8_f32_hc_row_warp<<<unsigned(rows), 32, 0, stream>>>(static_cast<const char *>(weights),
            static_cast<const char *>(input), output, inner, rows, weight_stride, input_inner_stride);
        const int status = int(cudaGetLastError());
        if (status == int(cudaSuccess)) {
            if (kda_variant == 1) report_kda_compact(1, rows, columns);
            else report_hc_compact(1, columns, 1);
        }
        return status;
    }
    else if (columns == 1) {
        const unsigned blocks = unsigned((int64_t(rows) + 31) / 32);
        q8_f32_vector<false><<<blocks, 32, 0, stream>>>(static_cast<const char *>(weights),
            static_cast<const char *>(input), output, inner, rows, weight_stride, input_inner_stride, input_column_stride);
    }
    else if (columns <= 8 && parallel_small_batch_enabled()) {
        const dim3 grid(unsigned((int64_t(rows) + 3) / 4), unsigned(columns));
        q8_f32_vector_parallel<true><<<grid, 128, 0, stream>>>(static_cast<const char *>(weights),
            static_cast<const char *>(input), output, inner, rows, weight_stride, input_inner_stride, input_column_stride);
    }
    else if (ordered) {
        const dim3 grid(unsigned((int64_t(rows) + 31) / 32), unsigned(columns));
        q8_f32_vector<true><<<grid, 32, 0, stream>>>(static_cast<const char *>(weights),
            static_cast<const char *>(input), output, inner, rows, weight_stride, input_inner_stride, input_column_stride);
    }
    else if (kda_variant == 2 && !gate_tile16) {
        // Exact whitelist, fixed16 columns. No occupancy/device metadata query,
        // payload scratch or general prefill-tile policy on this selected path.
        launch<16, false, 32>(weights, input, output, inner, rows, columns,
            weight_stride, input_inner_stride, input_column_stride, stream);
        const int status = int(cudaGetLastError());
        if (status == int(cudaSuccess)) report_kda_compact(2, rows, columns);
        return status;
    }
    else if (hc_variant == 2) {
        const int selected_tile = tsg_q8_hc_compact_columns(columns, tile);
        if (selected_tile == 16) launch<16, false, 32>(weights, input, output, inner, rows, columns,
            weight_stride, input_inner_stride, input_column_stride, stream);
        else if (selected_tile == 64) launch<64, false, 32>(weights, input, output, inner, rows, columns,
            weight_stride, input_inner_stride, input_column_stride, stream);
        else if (selected_tile == 128) launch<128, false, 32>(weights, input, output, inner, rows, columns,
            weight_stride, input_inner_stride, input_column_stride, stream);
        else launch<32, false, 32>(weights, input, output, inner, rows, columns,
            weight_stride, input_inner_stride, input_column_stride, stream);
        const int status = int(cudaGetLastError());
        if (status == int(cudaSuccess)) report_hc_compact(2, columns, selected_tile);
        return status;
    }
    else if (columns <= 8) launch<8>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else if (columns <= 16 || gate_tile16) launch<16>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else if (columns >= 128 && tile == 128) launch<128>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else if (columns >= 64 && tile >= 64) launch<64>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else launch<32>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    return int(cudaGetLastError());
}

int tsg_matmul_q8_cuda_launch(const void * weights, const void * input, float * output,
        int inner, int rows, int columns, size_t weight_stride, size_t input_inner_stride,
        size_t input_column_stride, void * stream_pointer) {
    return q8_cuda_launch(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream_pointer, true);
}

#if defined(TSG_GGML_TEST_HOOKS)
int tsg_matmul_q8_hc_cuda_test_selected(int inner, int rows, int columns,
                                      int tile_columns, bool enabled) {
    (void)tile_columns;
    return tsg_q8_hc_selected_variant(inner, rows, columns, enabled);
}

unsigned tsg_matmul_q8_hc_cuda_test_reported() {
    return hc_reported_bits().load(std::memory_order_relaxed);
}

int tsg_matmul_q8_kda_cuda_test_selected(int inner, int rows, int columns, bool enabled) {
    return tsg_q8_kda_selected_variant(inner, rows, columns, enabled);
}

unsigned tsg_matmul_q8_kda_cuda_test_reported() {
    return kda_reported_bits().load(std::memory_order_relaxed);
}

int tsg_matmul_q8_hc_cuda_test_launch(TsgQ8HcTestRoute route,
        const void * weights, const void * input, float * output,
        int inner, int rows, int columns, size_t weight_stride,
        size_t input_inner_stride, size_t input_column_stride,
        int tile_columns, void * stream_pointer) {
    if (!weights || !input || !output || inner <= 0 || inner % 32 != 0 || rows <= 0 || columns <= 0
        || weight_stride < size_t(inner / 32) * 34 || weight_stride % 2 != 0
        || input_inner_stride < 4 || input_inner_stride % 4 != 0
        || input_column_stride < 4 || input_column_stride % 4 != 0) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(stream_pointer);
    if (route == TsgQ8HcTestRoute::production) return q8_cuda_launch(weights, input, output,
        inner, rows, columns, weight_stride, input_inner_stride, input_column_stride, stream_pointer, true);
    if (route == TsgQ8HcTestRoute::row_warp) {
        if (columns != 1) return int(cudaErrorInvalidValue);
        q8_f32_hc_row_warp<<<unsigned(rows), 32, 0, stream>>>(static_cast<const char *>(weights),
            static_cast<const char *>(input), output, inner, rows, weight_stride, input_inner_stride);
    } else if (route == TsgQ8HcTestRoute::control) {
        if (columns == 1) {
            const unsigned blocks = unsigned((int64_t(rows) + 31) / 32);
            q8_f32_vector<false><<<blocks, 32, 0, stream>>>(static_cast<const char *>(weights),
                static_cast<const char *>(input), output, inner, rows, weight_stride, input_inner_stride, input_column_stride);
        } else switch (tile_columns) {
            case 8: launch<8>(weights,input,output,inner,rows,columns,weight_stride,input_inner_stride,input_column_stride,stream); break;
            case 16: launch<16>(weights,input,output,inner,rows,columns,weight_stride,input_inner_stride,input_column_stride,stream); break;
            case 32: launch<32>(weights,input,output,inner,rows,columns,weight_stride,input_inner_stride,input_column_stride,stream); break;
            case 64: launch<64>(weights,input,output,inner,rows,columns,weight_stride,input_inner_stride,input_column_stride,stream); break;
            case 128: launch<128>(weights,input,output,inner,rows,columns,weight_stride,input_inner_stride,input_column_stride,stream); break;
            default: return int(cudaErrorInvalidValue);
        }
    } else if (route == TsgQ8HcTestRoute::compact_tile) {
        switch (tile_columns) {
            case 16: launch<16,false,32>(weights,input,output,inner,rows,columns,weight_stride,input_inner_stride,input_column_stride,stream); break;
            case 32: launch<32,false,32>(weights,input,output,inner,rows,columns,weight_stride,input_inner_stride,input_column_stride,stream); break;
            case 64: launch<64,false,32>(weights,input,output,inner,rows,columns,weight_stride,input_inner_stride,input_column_stride,stream); break;
            case 128: launch<128,false,32>(weights,input,output,inner,rows,columns,weight_stride,input_inner_stride,input_column_stride,stream); break;
            default: return int(cudaErrorInvalidValue);
        }
    } else return int(cudaErrorInvalidValue);
    return int(cudaGetLastError());
}

int tsg_matmul_q8_cuda_launch_test_tile(const void * weights, const void * input, float * output,
        int inner, int rows, int columns, size_t weight_stride, size_t input_inner_stride,
        size_t input_column_stride, int tile_columns, void * stream_pointer) {
    const auto stream = static_cast<cudaStream_t>(stream_pointer);
    switch (tile_columns) {
        case 16: launch<16>(weights, input, output, inner, rows, columns,
            weight_stride, input_inner_stride, input_column_stride, stream); break;
        case 32: launch<32>(weights, input, output, inner, rows, columns,
            weight_stride, input_inner_stride, input_column_stride, stream); break;
        case 64: launch<64>(weights, input, output, inner, rows, columns,
            weight_stride, input_inner_stride, input_column_stride, stream); break;
        default: return int(cudaErrorInvalidValue);
    }
    return int(cudaGetLastError());
}

int tsg_matmul_q8_cuda_launch_control(const void * weights, const void * input, float * output,
        int inner, int rows, int columns, size_t weight_stride, size_t input_inner_stride,
        size_t input_column_stride, void * stream_pointer) {
    return q8_cuda_launch(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream_pointer, false);
}

// Keep the previous implementation available only to native correctness tests;
// no runtime switch or additional public C ABI changes production dispatch.
int tsg_matmul_q8_cuda_launch_reference(const void * weights, const void * input, float * output,
        int inner, int rows, int columns, size_t weight_stride, size_t input_inner_stride,
        size_t input_column_stride, void * stream_pointer) {
    const auto stream = static_cast<cudaStream_t>(stream_pointer);
    if (columns <= 8) launch<8>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else if (columns <= 16) launch<16>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else launch<32>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    return int(cudaGetLastError());
}
#endif

void tsg_matmul_q8_cuda_compute(ggml_tensor * dst, ggml_backend_t cuda_backend) {
    GGML_ASSERT(ggml_backend_is_cuda(cuda_backend) && ggml_is_contiguous(dst));
    GGML_ASSERT(tsg_matmul_quant_f32_type_supported(dst->src[0]->type) && dst->src[1]->type == GGML_TYPE_F32);
    // CUDA y-grid is bounded even on modern devices. The encoder's maximum
    // 65536-token microbatch is well below this limit for every tile size.
    GGML_ASSERT(dst->ne[1] <= int64_t(65535) * 8);
    auto * context = static_cast<ggml_backend_cuda_context *>(cuda_backend->context);
    CUDA_CHECK(cudaSetDevice(context->device));
    const cudaStream_t stream = context->stream(context->device, 0);
    const auto * w = dst->src[0];
    const auto * x = dst->src[1];
    const int status = w->type == GGML_TYPE_Q5_K
        ? q5_cuda_launch(w->data, x->data, static_cast<float *>(dst->data),
            int(w->ne[0]), int(w->ne[1]), int(x->ne[1]), w->nb[1], x->nb[0], x->nb[1], stream)
        : tsg_matmul_q8_cuda_launch(w->data, x->data,
        static_cast<float *>(dst->data), int(w->ne[0]), int(w->ne[1]), int(x->ne[1]),
        w->nb[1], x->nb[0], x->nb[1], stream);
    CUDA_CHECK(static_cast<cudaError_t>(status));
}
