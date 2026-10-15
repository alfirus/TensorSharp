// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.MLX;

/// <summary>F32 device operations shared by the direct diffusion models.</summary>
public static class MlxDirectOps
{
    /// <summary>Apply pair-interleaved rotary embeddings to contiguous token-major heads.</summary>
    public static bool TryRopeInterleaved(Tensor data, Tensor cos, Tensor sin, int heads, int tokens, int headDim)
    {
        if (data?.Storage is not MlxStorage ds || cos?.Storage is not MlxStorage cs || sin?.Storage is not MlxStorage ss ||
            data.ElementType != DType.Float32 || cos.ElementType != DType.Float32 || sin.ElementType != DType.Float32 ||
            data.DimensionCount != 2 || cos.DimensionCount != 2 || sin.DimensionCount != 2 ||
            !data.IsContiguous() || !cos.IsContiguous() || !sin.IsContiguous() ||
            data.StorageOffset != 0 || data.Storage.ElementCount != data.ElementCount() ||
            heads <= 0 || tokens <= 0 || headDim <= 0 || (headDim & 1) != 0 ||
            data.Sizes[0] != tokens || data.Sizes[1] != (long)heads * headDim ||
            cos.Sizes[0] != tokens || cos.Sizes[1] != headDim || sin.Sizes[0] != tokens || sin.Sizes[1] != headDim ||
            !ReferenceEquals(ds.AllocatorImpl, cs.AllocatorImpl) || !ReferenceEquals(ds.AllocatorImpl, ss.AllocatorImpl)) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, c = default, s = default, output = default;
            try
            {
                x = ds.CreateArrayView(data); c = cs.CreateArrayView(cos); s = ss.CreateArrayView(sin);
                output = MlxNative.DirectRopeInterleaved(x, c, s, heads, tokens, headDim);
                ds.ReplaceDeviceArray(output); output = default; return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(c); MlxNative.FreeArray(s); MlxNative.FreeArray(output); }
        });
    }
}

internal static partial class MlxNative
{
    private static MlxFastMetalKernel directInterleavedRopeKernel;
    private static MlxFastMetalKernel directQ6KFloatDequantKernel;

    internal static MlxArray DirectQ6KFloatMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
    {
        if (rows <= Q6KMatvecMaxRows) return Q6KMatvec(input, rawWeight, rows, inDim, outDim);
        lock (fastKernelSync)
        {
            if (!directQ6KFloatDequantKernel.IsValid)
                directQ6KFloatDequantKernel = CreateFastMetalKernel("tensorsharp_direct_q6k_dequant_f32", ["w"], ["y"], """
                    const uint column = thread_position_in_grid.x;
                    const uint row = thread_position_in_grid.y;
                    if (column < Inner && row < Rows) {
                        const uint block = column / 256;
                        device const uchar * source = w + (ulong(row + Start) * (Inner / 256) + block) * 210;
                        y[ulong(row) * Inner + column] = tensorsharp_dequant_q6k(source, column % 256);
                    }
                    """, KQuantHelpersHeader);
        }
        int sliceRows = (int)Math.Clamp((64L * 1024 * 1024) / (sizeof(float) * (long)inDim), 1, outDim);
        var parts = new System.Collections.Generic.List<MlxArray>();
        MlxArray joined = default;
        try
        {
            for (int start = 0; start < outDim; start += sliceRows)
            {
                int count = Math.Min(sliceRows, outDim - start);
                var config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default, outputs = default;
                MlxArray weight = default, transposed = default, part = default;
                try
                {
                    AddTemplateInt(config, "Start", start); AddTemplateInt(config, "Rows", count); AddTemplateInt(config, "Inner", inDim);
                    int[] shape = [count, inDim];
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring direct F32 Q6_K weight slice");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, inDim, count, 1), "configuring direct F32 Q6_K dequantization grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring direct F32 Q6_K dequantization threadgroup");
                    inputs = CreateVectorArray(rawWeight); outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, directQ6KFloatDequantKernel, inputs, config, DefaultStream()), "dequantizing direct F32 Q6_K weight");
                    Check(mlx_vector_array_get(out weight, outputs, 0), "reading direct F32 Q6_K weight");
                    transposed = Transpose(weight, [1, 0]);
                    Check(mlx_matmul(out part, input, transposed, DefaultStream()), "running direct F32 Q6_K projection");
                    // Finish each weight slice before releasing it. The exact
                    // dense temporary remains bounded at 64 MiB on large layers.
                    Eval(part); parts.Add(part); part = default;
                }
                finally
                {
                    FreeArray(part); FreeArray(transposed); FreeArray(weight);
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            }
            if (parts.Count == 1) { var result = parts[0]; parts.Clear(); return result; }
            var arrays = CreateVectorArray(parts.ToArray());
            try { Check(mlx_concatenate_axis(out joined, arrays, 1, DefaultStream()), "joining direct F32 Q6_K projections"); }
            finally { if (arrays.IsValid) _ = mlx_vector_array_free(arrays); }
            var output = joined; joined = default; return output;
        }
        finally { foreach (var part in parts) FreeArray(part); FreeArray(joined); }
    }

    internal static MlxArray DirectRopeInterleaved(MlxArray input, MlxArray cos, MlxArray sin,
        int heads, int tokens, int headDim)
    {
        int pairs = checked(tokens * heads * (headDim / 2));
        lock (fastKernelSync)
        {
            if (!directInterleavedRopeKernel.IsValid)
                directInterleavedRopeKernel = CreateFastMetalKernel("tensorsharp_direct_interleaved_rope",
                    ["x", "c", "s"], ["y"], """
                    const uint pair = thread_position_in_grid.x;
                    if (pair < Pairs) {
                        const uint halfPairs = HeadDim / 2;
                        const uint token = pair / (Heads * halfPairs);
                        const uint position = pair % halfPairs;
                        const uint offset = pair * 2;
                        const uint table = token * HeadDim + position * 2;
                        const float even = x[offset], odd = x[offset + 1];
                        const float cosine = c[table], sine = s[table];
                        y[offset] = even * cosine - odd * sine;
                        y[offset + 1] = odd * cosine + even * sine;
                    }
                    """, string.Empty);
        }
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        try
        {
            AddTemplateInt(config, "Pairs", pairs); AddTemplateInt(config, "Heads", heads); AddTemplateInt(config, "HeadDim", headDim);
            int[] shape = [tokens, checked(heads * headDim)];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring direct interleaved RoPE output");
            Check(mlx_fast_metal_kernel_config_set_grid(config, pairs, 1, 1), "configuring direct interleaved RoPE grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring direct interleaved RoPE threadgroup");
            inputs = CreateVectorArray(input, cos, sin); outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, directInterleavedRopeKernel, inputs, config, DefaultStream()), "running direct interleaved RoPE");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading direct interleaved RoPE output");
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
