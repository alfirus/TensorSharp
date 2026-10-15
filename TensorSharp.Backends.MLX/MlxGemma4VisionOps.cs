// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Numerics;

namespace TensorSharp.MLX;

/// <summary>F32 Gemma 4 vision rotations using the encoder's exact X/Y angle tables.</summary>
public static class MlxGemma4VisionOps
{
    /// <summary>Evaluate the existing lazy stage without copying it to the CPU.
    /// Diagnostic callers time this explicitly; normal inference never calls it.</summary>
    public static void EvaluateForProfile(Tensor stage)
    {
        if (stage?.Storage is not MlxStorage storage)
            throw new ArgumentException("Stage profiling requires MLX storage.", nameof(stage));
        MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray array = default;
            try { array = storage.CreateArrayView(stage); MlxNative.Eval(array); }
            finally { MlxNative.FreeArray(array); }
        });
    }

    /// <summary>Add the learned X/Y lookup tables on device. Keep the legacy
    /// vector channels' (hidden + x) + y order and its scalar tail's hidden +
    /// (x + y) order. Positions are the same row-major patch grid as the cache.</summary>
    public static bool TryAddPosition2D(Tensor hidden, Tensor positions, int patchesX, int patchesY)
    {
        if (hidden?.Storage is not MlxStorage hs || positions?.Storage is not MlxStorage ps ||
            hidden.ElementType != DType.Float32 || positions.ElementType != DType.Float32 ||
            hidden.DimensionCount != 2 || positions.DimensionCount != 3 ||
            !hidden.IsContiguous() || !positions.IsContiguous() ||
            hidden.StorageOffset != 0 || hidden.Storage.ElementCount != hidden.ElementCount() ||
            ReferenceEquals(hidden.Storage, positions.Storage) || !ReferenceEquals(hs.AllocatorImpl, ps.AllocatorImpl) ||
            patchesX <= 0 || patchesY <= 0 || hidden.Sizes[1] <= 0 || hidden.Sizes[1] > int.MaxValue ||
            (long)patchesX * patchesY > int.MaxValue ||
            (long)patchesX * patchesY * hidden.Sizes[1] > int.MaxValue ||
            hidden.Sizes[0] != (long)patchesX * patchesY || positions.Sizes[0] != 2 ||
            positions.Sizes[1] < patchesX || positions.Sizes[1] < patchesY ||
            positions.Sizes[1] > int.MaxValue || positions.Sizes[2] != hidden.Sizes[1] ||
            positions.ElementCount() > int.MaxValue) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, p = default, output = default;
            try
            {
                x = hs.CreateArrayView(hidden); p = ps.CreateArrayView(positions);
                int channels = checked((int)hidden.Sizes[1]);
                output = MlxNative.Gemma4VisionPosition2D(x, p, patchesX, patchesY,
                    channels, checked((int)positions.Sizes[1]), channels / Vector<float>.Count * Vector<float>.Count);
                hs.ReplaceDeviceArray(output); output = default; return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(p); MlxNative.FreeArray(output); }
        });
    }

    /// <summary>Pool the same non-overlapping complete 3x3 patch boxes as the
    /// legacy CPU path. Accumulate each channel in ky/kx F32 order, then multiply
    /// once by the same F32 reciprocal. Incomplete right/bottom boxes are cropped.</summary>
    public static bool TryPool3x3(Tensor result, Tensor input, int patchesX, int patchesY)
    {
        if (result?.Storage is not MlxStorage rs || input?.Storage is not MlxStorage xs ||
            result.ElementType != DType.Float32 || input.ElementType != DType.Float32 ||
            result.DimensionCount != 2 || input.DimensionCount != 2 ||
            !result.IsContiguous() || !input.IsContiguous() ||
            result.StorageOffset != 0 || result.Storage.ElementCount != result.ElementCount() ||
            ReferenceEquals(result.Storage, input.Storage) || !ReferenceEquals(rs.AllocatorImpl, xs.AllocatorImpl) ||
            patchesX < 3 || patchesY < 3 || input.Sizes[1] <= 0 || input.Sizes[1] > int.MaxValue ||
            (long)patchesX * patchesY > int.MaxValue ||
            (long)patchesX * patchesY * input.Sizes[1] > int.MaxValue ||
            input.Sizes[0] != (long)patchesX * patchesY || result.Sizes[1] != input.Sizes[1] ||
            result.Sizes[0] != (long)(patchesX / 3) * (patchesY / 3)) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, output = default;
            try
            {
                x = xs.CreateArrayView(input);
                output = MlxNative.Gemma4VisionPool3x3(x, patchesX, patchesY, checked((int)input.Sizes[1]));
                rs.ReplaceDeviceArray(output); output = default; return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(output); }
        });
    }

    public static bool TryPadProjectionInput(Tensor result, Tensor input, int extraChannels)
    {
        if (result?.Storage is not MlxStorage rs || input?.Storage is not MlxStorage xs ||
            result.ElementType != DType.Float32 || input.ElementType != DType.Float32 ||
            result.DimensionCount != 2 || input.DimensionCount != 2 || !result.IsContiguous() || !input.IsContiguous() ||
            result.StorageOffset != 0 || result.Storage.ElementCount != result.ElementCount() ||
            !ReferenceEquals(rs.AllocatorImpl, xs.AllocatorImpl) || ReferenceEquals(result.Storage, input.Storage) ||
            extraChannels <= 0 || input.Sizes[0] <= 0 || input.Sizes[0] > int.MaxValue || input.Sizes[1] > int.MaxValue ||
            result.Sizes[0] != input.Sizes[0] || result.Sizes[1] != input.Sizes[1] + extraChannels) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, padding = default, output = default;
            try
            {
                x = xs.CreateArrayView(input);
                padding = MlxNative.Full([checked((int)input.Sizes[0]), extraChannels], 0, DType.Float32);
                output = MlxNative.ConcatenateAxis(x, padding, 1);
                rs.ReplaceDeviceArray(output); output = default; return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(padding); MlxNative.FreeArray(output); }
        });
    }

    /// <summary>Use tiled attention for 64-channel heads. Zero padding selects
    /// MLX's F32 SIMD kernel without changing scores or the 64 output channels.</summary>
    public static bool TrySelfAttention(Tensor result, Tensor query, Tensor key, Tensor value)
    {
        if (result?.Storage is not MlxStorage rs || query?.Storage is not MlxStorage qs ||
            key?.Storage is not MlxStorage ks || value?.Storage is not MlxStorage vs ||
            result.ElementType != DType.Float32 || query.ElementType != DType.Float32 ||
            key.ElementType != DType.Float32 || value.ElementType != DType.Float32 ||
            result.DimensionCount != 4 || query.DimensionCount != 4 || key.DimensionCount != 4 || value.DimensionCount != 4 ||
            !result.IsContiguous() || !query.IsContiguous() || !key.IsContiguous() || !value.IsContiguous() ||
            result.StorageOffset != 0 || result.Storage.ElementCount != result.ElementCount() ||
            ReferenceEquals(result.Storage, query.Storage) || ReferenceEquals(result.Storage, key.Storage) ||
            ReferenceEquals(result.Storage, value.Storage) || query.Sizes[3] != 64 ||
            !ReferenceEquals(rs.AllocatorImpl, qs.AllocatorImpl) || !ReferenceEquals(rs.AllocatorImpl, ks.AllocatorImpl) ||
            !ReferenceEquals(rs.AllocatorImpl, vs.AllocatorImpl)) return false;
        for (int d = 0; d < 4; ++d)
            if (query.Sizes[d] <= 0 || query.Sizes[d] != result.Sizes[d] ||
                query.Sizes[d] != key.Sizes[d] || query.Sizes[d] != value.Sizes[d]) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray q = default, k = default, v = default, qh = default, kh = default, vh = default,
                padding = default, qp = default, kp = default, vp = default, attention = default,
                cropped = default, restored = default, output = default;
            try
            {
                q = qs.CreateArrayView(query); k = ks.CreateArrayView(key); v = vs.CreateArrayView(value);
                qh = MlxNative.Transpose(q, [0, 2, 1, 3]); kh = MlxNative.Transpose(k, [0, 2, 1, 3]);
                vh = MlxNative.Transpose(v, [0, 2, 1, 3]);
                int batches = checked((int)query.Sizes[0]), tokens = checked((int)query.Sizes[1]), heads = checked((int)query.Sizes[2]);
                // Unchanged upstream MLX's M5 NAX path narrows F32 products to
                // TF32 for heads64/96/128/256. Its SIMD path supports head80
                // (head192 for <=8-row vector attention) with F32 operands.
                // Extra channels are zero and the scale remains exactly 1.
                int paddedHead = tokens <= 8 ? 192 : 80;
                padding = MlxNative.Full([batches, heads, tokens, paddedHead - 64], 0, DType.Float32);
                qp = MlxNative.ConcatenateAxis(qh, padding, 3); kp = MlxNative.ConcatenateAxis(kh, padding, 3);
                vp = MlxNative.ConcatenateAxis(vh, padding, 3);
                attention = MlxNative.FastScaledDotProductAttention(qp, kp, vp, 1f, string.Empty, default, forceFused: true);
                cropped = MlxNative.Slice(attention, [0, 0, 0, 0], [batches, heads, tokens, 64], [1, 1, 1, 1]);
                restored = MlxNative.Transpose(cropped, [0, 2, 1, 3]); output = MlxNative.Contiguous(restored);
                rs.ReplaceDeviceArray(output); output = default; return true;
            }
            finally
            {
                MlxNative.FreeArray(q); MlxNative.FreeArray(k); MlxNative.FreeArray(v); MlxNative.FreeArray(qh);
                MlxNative.FreeArray(kh); MlxNative.FreeArray(vh); MlxNative.FreeArray(attention);
                MlxNative.FreeArray(padding); MlxNative.FreeArray(qp); MlxNative.FreeArray(kp); MlxNative.FreeArray(vp);
                MlxNative.FreeArray(cropped); MlxNative.FreeArray(restored); MlxNative.FreeArray(output);
            }
        });
    }

    /// <summary>Each head has an X half and a Y half, with rotate-half pairs inside each half.
    /// Cosine and sine contain the X quarter followed by the Y quarter for each patch.</summary>
    public static bool TryRope2D(Tensor data, Tensor cosine, Tensor sine, int tokens, int heads, int headDim)
    {
        if (data?.Storage is not MlxStorage ds || cosine?.Storage is not MlxStorage cs ||
            sine?.Storage is not MlxStorage ss || data.ElementType != DType.Float32 ||
            cosine.ElementType != DType.Float32 || sine.ElementType != DType.Float32 ||
            data.DimensionCount != 2 || cosine.DimensionCount != 2 || sine.DimensionCount != 2 ||
            !data.IsContiguous() || !cosine.IsContiguous() || !sine.IsContiguous() ||
            data.StorageOffset != 0 || data.Storage.ElementCount != data.ElementCount() ||
            ReferenceEquals(data.Storage, cosine.Storage) || ReferenceEquals(data.Storage, sine.Storage) ||
            !ReferenceEquals(ds.AllocatorImpl, cs.AllocatorImpl) || !ReferenceEquals(ds.AllocatorImpl, ss.AllocatorImpl) ||
            tokens <= 0 || heads <= 0 || headDim <= 0 || headDim % 4 != 0 ||
            (long)heads * headDim > int.MaxValue || (long)tokens * heads * (headDim / 2) > int.MaxValue ||
            data.Sizes[0] != tokens || data.Sizes[1] != (long)heads * headDim ||
            cosine.Sizes[0] != tokens || cosine.Sizes[1] != headDim / 2 ||
            sine.Sizes[0] != tokens || sine.Sizes[1] != headDim / 2) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, c = default, s = default, result = default;
            try
            {
                x = ds.CreateArrayView(data); c = cs.CreateArrayView(cosine); s = ss.CreateArrayView(sine);
                result = MlxNative.Gemma4VisionRope2D(x, c, s, tokens, heads, headDim);
                ds.ReplaceDeviceArray(result); result = default; return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(c); MlxNative.FreeArray(s); MlxNative.FreeArray(result); }
        });
    }
}

internal static partial class MlxNative
{
    private static MlxFastMetalKernel gemma4VisionRope2DKernel;
    private static MlxFastMetalKernel gemma4VisionPosition2DKernel;
    private static MlxFastMetalKernel gemma4VisionPool3x3Kernel;

    internal static MlxArray Gemma4VisionPosition2D(MlxArray input, MlxArray positions, int patchesX, int patchesY,
        int channels, int maxPositions, int vectorChannels)
    {
        lock (fastKernelSync)
        {
            if (!gemma4VisionPosition2DKernel.IsValid)
                gemma4VisionPosition2DKernel = CreateFastMetalKernel("tensorsharp_gemma4_vision_position_2d_f32",
                    ["x", "p"], ["y"], """
                    #pragma clang fp reassociate(off)
                    #pragma clang fp contract(off)
                    const uint index = thread_position_in_grid.x;
                    if (index < Elements) {
                        const uint patch = index / Channels, channel = index % Channels;
                        const uint px = patch % PatchesX, py = patch / PatchesX;
                        const float xv = p[ulong(px) * Channels + channel];
                        const float yv = p[(ulong(MaxPositions) + py) * Channels + channel];
                        const float value = x[index];
                        if (channel < VectorChannels) {
                            const float first = value + xv;
                            y[index] = first + yv;
                        } else {
                            const float position = xv + yv;
                            y[index] = value + position;
                        }
                    }
                    """, string.Empty);
        }
        int rows = checked(patchesX * patchesY);
        return ApplyVisionSpatialKernel(gemma4VisionPosition2DKernel, [input, positions], [rows, channels],
            [("Elements", checked(rows * channels)), ("Channels", channels), ("PatchesX", patchesX),
             ("MaxPositions", maxPositions), ("VectorChannels", vectorChannels)]);
    }

    internal static MlxArray Gemma4VisionPool3x3(MlxArray input, int patchesX, int patchesY, int channels)
    {
        lock (fastKernelSync)
        {
            if (!gemma4VisionPool3x3Kernel.IsValid)
                gemma4VisionPool3x3Kernel = CreateFastMetalKernel("tensorsharp_gemma4_vision_pool_3x3_f32",
                    ["x"], ["y"], """
                    #pragma clang fp reassociate(off)
                    #pragma clang fp contract(off)
                    const uint index = thread_position_in_grid.x;
                    if (index < Elements) {
                        const uint patch = index / Channels, channel = index % Channels;
                        const uint px = (patch % MergedX) * 3, py = (patch / MergedX) * 3;
                        float sum = 0.0f;
                        #pragma clang loop unroll(disable)
                        for (uint ky = 0; ky < 3; ++ky) {
                            #pragma clang loop unroll(disable)
                            for (uint kx = 0; kx < 3; ++kx) {
                                sum = sum + x[(ulong(py + ky) * PatchesX + px + kx) * Channels + channel];
                            }
                        }
                        const float reciprocal = 1.0f / 9.0f;
                        y[index] = sum * reciprocal;
                    }
                    """, string.Empty);
        }
        int rows = checked((patchesX / 3) * (patchesY / 3));
        return ApplyVisionSpatialKernel(gemma4VisionPool3x3Kernel, [input], [rows, channels],
            [("Elements", checked(rows * channels)), ("Channels", channels), ("PatchesX", patchesX), ("MergedX", patchesX / 3)]);
    }

    private static MlxArray ApplyVisionSpatialKernel(MlxFastMetalKernel kernel, MlxArray[] arrays, int[] shape,
        (string Name, int Value)[] templates)
    {
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        try
        {
            foreach (var template in templates) AddTemplateInt(config, template.Name, template.Value);
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring vision spatial output");
            Check(mlx_fast_metal_kernel_config_set_grid(config, checked(shape[0] * shape[1]), 1, 1), "configuring vision spatial grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring vision spatial threadgroup");
            inputs = CreateVectorArray(arrays); outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running vision spatial kernel");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading vision spatial output");
            return result;
        }
        finally
        {
            if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
            if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
            if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
        }
    }

    internal static MlxArray Gemma4VisionRope2D(MlxArray input, MlxArray cosine, MlxArray sine,
        int tokens, int heads, int headDim)
    {
        int pairs = checked(tokens * heads * (headDim / 2));
        lock (fastKernelSync)
        {
            if (!gemma4VisionRope2DKernel.IsValid)
                gemma4VisionRope2DKernel = CreateFastMetalKernel("tensorsharp_gemma4_vision_rope_2d_f32",
                    ["x", "c", "s"], ["y"], """
                    const uint pair = thread_position_in_grid.x;
                    if (pair < Pairs) {
                        const uint halfWidth = HeadDim / 2, quarter = HeadDim / 4;
                        const uint token = pair / (Heads * halfWidth);
                        const uint h = (pair / halfWidth) % Heads;
                        const uint channel = pair % halfWidth;
                        const uint axis = channel / quarter;
                        const uint index = channel % quarter;
                        const ulong base = (ulong(token) * Heads + h) * HeadDim + axis * halfWidth + index;
                        const uint table = token * halfWidth + channel;
                        const float first = x[base], second = x[base + quarter];
                        const float cos = c[table], sin = s[table];
                        y[base] = first * cos - second * sin;
                        y[base + quarter] = first * sin + second * cos;
                    }
                    """, string.Empty);
        }
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        try
        {
            AddTemplateInt(config, "Pairs", pairs); AddTemplateInt(config, "Heads", heads); AddTemplateInt(config, "HeadDim", headDim);
            int[] shape = [tokens, checked(heads * headDim)];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring Gemma4 vision RoPE output");
            Check(mlx_fast_metal_kernel_config_set_grid(config, pairs, 1, 1), "configuring Gemma4 vision RoPE grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring Gemma4 vision RoPE threadgroup");
            inputs = CreateVectorArray(input, cosine, sine); outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, gemma4VisionRope2DKernel, inputs, config, DefaultStream()), "running Gemma4 vision RoPE");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading Gemma4 vision RoPE output");
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
