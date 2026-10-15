// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.MLX;

/// <summary>Device kernels for the LTX 2.5 diffusion video decoder: 3D
/// neighborhood attention with NATTEN window semantics, and its absolute
/// per-axis rotary embedding. Shapes and geometry that change between calls
/// travel in a small Float32 parameter tensor rather than as template values,
/// so a chunked decode reuses one compiled kernel per head size and window.</summary>
public static class MlxLtx25DiffusionVaeOps
{
    /// <summary>output, q: [TQ*H*W, heads*headDim]; k, v: [TK*H*WK, heads*headDim];
    /// parameters: {T, H, W, QT0, KT0, TQ, TK, WK, PW}. Queries are already normalized,
    /// scaled and rotated. Query frame t (global QT0 + local) attends to the
    /// window start = clamp(i - k/2, 0, L - k) on every axis, with k = min(kernel, L);
    /// key frame f is row f - KT0 of k and v. Key columns may extend the query grid by
    /// PW columns on each side (WK = W + 2*PW): column w then starts its window at
    /// clamp(w + PW - k/2, 0, WK - k). <paramref name="height"/> and <paramref name="width"/>
    /// repeat the parameters' H and W for the dispatch's tile grid.</summary>
    public static bool TryNeighborhoodAttention(Tensor output, Tensor q, Tensor k, Tensor v, Tensor parameters,
        int heads, int headDim, int kernelT, int kernelH, int kernelW, int height, int width)
    {
        if (!Same(output, q, k, v, parameters) || headDim <= 0 || headDim % 16 != 0 || heads <= 0 ||
            height <= 0 || width <= 0 || q.Sizes[0] % ((long)height * width) != 0 ||
            kernelT <= 0 || kernelH <= 0 || kernelW <= 0 || kernelT % 2 == 0 || kernelH % 2 == 0 || kernelW % 2 == 0 ||
            q.Sizes[1] != (long)heads * headDim || output.Sizes[0] != q.Sizes[0] || output.Sizes[1] != q.Sizes[1] ||
            k.Sizes[1] != q.Sizes[1] || v.Sizes[0] != k.Sizes[0] || v.Sizes[1] != k.Sizes[1] ||
            parameters.ElementCount() != 9 || output.ElementCount() > int.MaxValue || k.ElementCount() > int.MaxValue)
            return false;
        var target = (MlxStorage)output.Storage;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray qa = default, ka = default, va = default, pa = default, result = default;
            try
            {
                qa = ((MlxStorage)q.Storage).CreateArrayView(q); ka = ((MlxStorage)k.Storage).CreateArrayView(k);
                va = ((MlxStorage)v.Storage).CreateArrayView(v); pa = ((MlxStorage)parameters.Storage).CreateArrayView(parameters);
                result = MlxNative.Ltx25NeighborhoodAttention(qa, ka, va, pa, checked((int)(q.Sizes[0] / ((long)height * width))),
                    height, width, heads, headDim, kernelT, kernelH, kernelW);
                Publish(output, target, ref result);
                return true;
            }
            finally
            {
                MlxNative.FreeArray(qa); MlxNative.FreeArray(ka); MlxNative.FreeArray(va);
                MlxNative.FreeArray(pa); MlxNative.FreeArray(result);
            }
        });
    }

    /// <summary>input: [n*H*W, heads*headDim] tokens of frames T0.. on an HxW grid;
    /// output: [n*H*(W+2*PW), heads*headDim], columns edge-replicated PW wide on each
    /// side; frequencies: [headDim/2] inverse frequencies, the first splitT/2 pairs on
    /// the frame axis, the next splitH/2 on rows and the rest on columns;
    /// parameters: {H, W, T0, scale, PW}. Output column c reads input column
    /// clamp(c - PW, 0, W - 1) and rotates its interleaved (even, odd) pairs by
    /// position x frequency, with column position c - PW, then multiplies by scale.</summary>
    public static bool TryAbsoluteRope(Tensor output, Tensor input, Tensor frequencies, Tensor parameters,
        int heads, int headDim, int splitT, int splitH)
    {
        if (!Same(output, input, frequencies, parameters) || headDim <= 0 || headDim % 2 != 0 || heads <= 0 ||
            splitT < 0 || splitH < 0 || splitT % 2 != 0 || splitH % 2 != 0 || splitT + splitH > headDim ||
            input.Sizes[1] != (long)heads * headDim || output.Sizes[1] != input.Sizes[1] ||
            frequencies.ElementCount() != headDim / 2 || parameters.ElementCount() != 5 || output.ElementCount() > int.MaxValue)
            return false;
        var target = (MlxStorage)output.Storage;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray xa = default, fa = default, pa = default, result = default;
            try
            {
                xa = ((MlxStorage)input.Storage).CreateArrayView(input);
                fa = ((MlxStorage)frequencies.Storage).CreateArrayView(frequencies);
                pa = ((MlxStorage)parameters.Storage).CreateArrayView(parameters);
                result = MlxNative.Ltx25AbsoluteRope(xa, fa, pa, checked((int)output.Sizes[0]), heads, headDim, splitT, splitH);
                Publish(output, target, ref result);
                return true;
            }
            finally
            {
                MlxNative.FreeArray(xa); MlxNative.FreeArray(fa); MlxNative.FreeArray(pa); MlxNative.FreeArray(result);
            }
        });
    }

    private static bool Same(Tensor first, params Tensor[] others)
    {
        if (first?.Storage is not MlxStorage storage || first.ElementType != DType.Float32 || !first.IsContiguous() ||
            first.DimensionCount is not (1 or 2))
            return false;
        foreach (var other in others)
            if (other?.Storage is not MlxStorage source || !ReferenceEquals(source.AllocatorImpl, storage.AllocatorImpl) ||
                other.ElementType != DType.Float32 || !other.IsContiguous() || other.DimensionCount is not (1 or 2))
                return false;
        return true;
    }

    private static void Publish(Tensor output, MlxStorage target, ref MlxNative.MlxArray result)
    {
        if (output.StorageOffset == 0 && output.ElementCount() == output.Storage.ElementCount)
        { target.ReplaceDeviceArray(result); result = default; }
        else target.UpdateDeviceSlice(output, result);
    }
}

internal static partial class MlxNative
{
    private static MlxFastMetalKernel ltx25NeighborhoodAttentionKernel;
    private const int Ltx25AttentionTile = 8, Ltx25AttentionSplit = 4;
    // One threadgroup per (8x8 query tile of one frame, head); four adjacent lanes
    // share a query, each holding a quarter of the head dimension, and combine
    // their partial scores with two shuffles. NATTEN window starts are monotone in
    // the query index, so the tile's windows span at most 8 + k - 1 rows and
    // columns: each key row of that span is staged in threadgroup memory once and
    // read by every query whose window covers it, instead of every query reading
    // all of its k^3 keys from device memory. The softmax is online per key row,
    // rescaling once when a row raises the maximum.
    private const string Ltx25NeighborhoodAttentionSource = """
        constexpr int Tile = 8, Split = 4, Threads = Tile * Tile * Split;
        constexpr int C = Heads * HeadDim, V4 = HeadDim / 4, P4 = V4 / Split, SpanW = Tile + KernelW - 1;
        threadgroup float4 keys[SpanW * V4];
        threadgroup float4 values[SpanW * V4];
        const int lane = int(thread_position_in_threadgroup.x), part = lane % Split, local = lane / Split;
        const int group = int(threadgroup_position_in_grid.x);
        const int T = int(p[0]), H = int(p[1]), W = int(p[2]);
        const int qt0 = int(p[3]), kt0 = int(p[4]), WK = int(p[7]), PW = int(p[8]);
        const int tilesW = (W + Tile - 1) / Tile, tilesH = (H + Tile - 1) / Tile;
        const int head = group % Heads, tile = group / Heads;
        const int w0 = (tile % tilesW) * Tile, h0 = (tile / tilesW) % tilesH * Tile, frame = tile / (tilesW * tilesH);
        const int t = qt0 + frame, h = h0 + local / Tile, w = w0 + local % Tile;
        const bool valid = h < H && w < W;
        const int kT = min(KernelT, T), kH = min(KernelH, H), kW = min(KernelW, WK);
        const int st = clamp(t - kT / 2, 0, T - kT) - kt0;
        const int sh = clamp(h - kH / 2, 0, H - kH), sw = clamp(w + PW - kW / 2, 0, WK - kW);
        const int rowStart = clamp(h0 - kH / 2, 0, H - kH);
        const int rowEnd = clamp(min(h0 + Tile, H) - 1 - kH / 2, 0, H - kH) + kH;
        const int colStart = clamp(w0 + PW - kW / 2, 0, WK - kW);
        const int span = clamp(min(w0 + Tile, W) - 1 + PW - kW / 2, 0, WK - kW) + kW - colStart;
        const int token = (frame * H + min(h, H - 1)) * W + min(w, W - 1);
        const int offset = part * P4;
        const device float4* q4 = (const device float4*)(q + token * C + head * HeadDim) + offset;
        const device float4* k4 = (const device float4*)k;
        const device float4* v4 = (const device float4*)v;
        float4 query[P4], acc[P4];
        for (int d = 0; d < P4; ++d) { query[d] = q4[d]; acc[d] = float4(0.0f); }
        float best = -INFINITY, total = 0.0f;
        float scores[KernelW];
        for (int dt = 0; dt < kT; ++dt)
        for (int row = rowStart; row < rowEnd; ++row)
        {
            const int rowBase = ((st + dt) * H + row) * WK + colStart;
            for (int i = lane; i < span * V4; i += Threads)
            {
                const int source = ((rowBase + i / V4) * C + head * HeadDim) / 4 + i % V4;
                keys[i] = k4[source]; values[i] = v4[source];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // All four lanes of a query take this branch together, so the shuffles
            // only read active lanes.
            if (valid && row >= sh && row < sh + kH)
            {
                const int first = (sw - colStart) * V4 + offset;
                float rowBest = -INFINITY;
                for (int dw = 0; dw < kW; ++dw)
                {
                    float score = 0.0f;
                    for (int d = 0; d < P4; ++d) score += dot(query[d], keys[first + dw * V4 + d]);
                    score += simd_shuffle_xor(score, 1);
                    score += simd_shuffle_xor(score, 2);
                    scores[dw] = score; rowBest = max(rowBest, score);
                }
                if (rowBest > best)
                {
                    const float correction = metal::precise::exp(best - rowBest);
                    total *= correction;
                    for (int d = 0; d < P4; ++d) acc[d] *= correction;
                    best = rowBest;
                }
                for (int dw = 0; dw < kW; ++dw)
                {
                    const float weight = metal::precise::exp(scores[dw] - best);
                    total += weight;
                    for (int d = 0; d < P4; ++d) acc[d] += weight * values[first + dw * V4 + d];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (!valid) return;
        device float4* o4 = (device float4*)(out + token * C + head * HeadDim) + offset;
        const float inverse = 1.0f / total;
        for (int d = 0; d < P4; ++d) o4[d] = acc[d] * inverse;
        """;

    internal static MlxArray Ltx25NeighborhoodAttention(MlxArray q, MlxArray k, MlxArray v, MlxArray parameters,
        int frames, int height, int width, int heads, int headDim, int kernelT, int kernelH, int kernelW)
    {
        int tiles = checked(frames * ((height + Ltx25AttentionTile - 1) / Ltx25AttentionTile) * ((width + Ltx25AttentionTile - 1) / Ltx25AttentionTile));
        int threads = checked(tiles * heads * Ltx25AttentionTile * Ltx25AttentionTile * Ltx25AttentionSplit);
        lock (fastKernelSync)
            if (!ltx25NeighborhoodAttentionKernel.IsValid)
                ltx25NeighborhoodAttentionKernel = CreateFastMetalKernel("tensorsharp_ltx25_neighborhood_attention_f32",
                    ["q", "k", "v", "p"], ["out"], Ltx25NeighborhoodAttentionSource, string.Empty);
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        try
        {
            AddTemplateInt(config, "Heads", heads); AddTemplateInt(config, "HeadDim", headDim);
            AddTemplateInt(config, "KernelT", kernelT); AddTemplateInt(config, "KernelH", kernelH); AddTemplateInt(config, "KernelW", kernelW);
            int[] shape = [checked(frames * height * width), checked(heads * headDim)];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring LTX neighborhood attention");
            Check(mlx_fast_metal_kernel_config_set_grid(config, threads, 1, 1), "configuring LTX neighborhood attention grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, Ltx25AttentionTile * Ltx25AttentionTile * Ltx25AttentionSplit, 1, 1), "configuring LTX neighborhood attention threads");
            inputs = CreateVectorArray(q, k, v, parameters); outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, ltx25NeighborhoodAttentionKernel, inputs, config, DefaultStream()), "running LTX neighborhood attention");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading LTX neighborhood attention");
            return result;
        }
        finally
        {
            if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
            if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
            if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
        }
    }

    private static MlxFastMetalKernel ltx25AbsoluteRopeKernel;
    private const string Ltx25AbsoluteRopeSource = """
        const uint gid = thread_position_in_grid.x;
        const int H = int(p[0]), W = int(p[1]), t0 = int(p[2]), PW = int(p[4]);
        const float scale = p[3];
        const int WO = W + 2 * PW;
        constexpr int Pairs = HeadDim / 2;
        const int total = (int(x_shape[0]) / W) * WO * Heads * Pairs;
        if (int(gid) >= total) return;
        const int pair = int(gid) % Pairs;
        const int tokenHead = int(gid) / Pairs;
        const int head = tokenHead % Heads;
        const int token = tokenHead / Heads;
        const int column = token % WO, row = token / WO;
        const int source = row * W + clamp(column - PW, 0, W - 1);
        const int position = pair < SplitT / 2 ? t0 + row / H
            : pair < (SplitT + SplitH) / 2 ? row % H : column - PW;
        const float angle = float(position) * f[pair];
        const float c = metal::precise::cos(angle), s = metal::precise::sin(angle);
        const int input = (source * Heads + head) * HeadDim + 2 * pair;
        const float even = x[input], odd = x[input + 1];
        const int offset = tokenHead * HeadDim + 2 * pair;
        y[offset] = (even * c - odd * s) * scale;
        y[offset + 1] = (even * s + odd * c) * scale;
        """;

    internal static MlxArray Ltx25AbsoluteRope(MlxArray input, MlxArray frequencies, MlxArray parameters,
        int tokens, int heads, int headDim, int splitT, int splitH)
    {
        int threads = checked(tokens * heads * (headDim / 2));
        lock (fastKernelSync)
            if (!ltx25AbsoluteRopeKernel.IsValid)
                ltx25AbsoluteRopeKernel = CreateFastMetalKernel("tensorsharp_ltx25_absolute_rope_f32",
                    ["x", "f", "p"], ["y"], Ltx25AbsoluteRopeSource, string.Empty);
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        try
        {
            AddTemplateInt(config, "Heads", heads); AddTemplateInt(config, "HeadDim", headDim);
            AddTemplateInt(config, "SplitT", splitT); AddTemplateInt(config, "SplitH", splitH);
            int[] shape = [tokens, checked(heads * headDim)];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring LTX absolute RoPE");
            Check(mlx_fast_metal_kernel_config_set_grid(config, threads, 1, 1), "configuring LTX absolute RoPE grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring LTX absolute RoPE threads");
            inputs = CreateVectorArray(input, frequencies, parameters); outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, ltx25AbsoluteRopeKernel, inputs, config, DefaultStream()), "running LTX absolute RoPE");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading LTX absolute RoPE");
            return result;
        }
        finally
        {
            if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
            if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
            if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
        }
    }
}
