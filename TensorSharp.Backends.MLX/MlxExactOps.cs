// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.MLX;

/// <summary>F32 arithmetic for small projections where TF32 rounding affects normalization.</summary>
public static class MlxExactOps
{
    public static bool TryMatmul(Tensor result, Tensor input, Tensor weight)
    {
        if (result?.Storage is not MlxStorage rs || input?.Storage is not MlxStorage xs || weight?.Storage is not MlxStorage ws ||
            result.ElementType != DType.Float32 || input.ElementType != DType.Float32 || weight.ElementType != DType.Float32 ||
            result.DimensionCount != 2 || input.DimensionCount != 2 || weight.DimensionCount != 2 ||
            !result.IsContiguous() || !input.IsContiguous() || !weight.IsContiguous() ||
            result.StorageOffset != 0 || result.Storage.ElementCount != result.ElementCount() ||
            result.Sizes[0] != input.Sizes[0] || result.Sizes[1] != weight.Sizes[1] || input.Sizes[1] != weight.Sizes[0] ||
            !ReferenceEquals(rs.AllocatorImpl, xs.AllocatorImpl) || !ReferenceEquals(rs.AllocatorImpl, ws.AllocatorImpl)) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, w = default, y = default;
            try
            {
                x = xs.CreateArrayView(input); w = ws.CreateArrayView(weight);
                y = MlxNative.ExactFloatMatmul(x, w, checked((int)input.Sizes[0]), checked((int)input.Sizes[1]), checked((int)weight.Sizes[1]));
                rs.ReplaceDeviceArray(y); y = default; return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(w); MlxNative.FreeArray(y); }
        });
    }
}

internal static partial class MlxNative
{
    private static MlxFastMetalKernel exactFloatMatmulKernel;
    internal static MlxArray ExactFloatMatmul(MlxArray input, MlxArray weight, int rows, int inner, int columns)
    {
        if (rows <= 0 || inner <= 0 || columns <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        lock (fastKernelSync)
        {
            if (!exactFloatMatmulKernel.IsValid)
                exactFloatMatmulKernel = CreateFastMetalKernel("tensorsharp_exact_float_matmul", ["x", "w"], ["y"], """
                    const uint column = thread_position_in_grid.x;
                    const uint row = thread_position_in_grid.y;
                    if (row < Rows && column < Columns) {
                        float sum = 0.0f;
                        #pragma clang loop unroll(disable)
                        for (uint k = 0; k < Inner; ++k) {
                            sum = fma(x[row * Inner + k], w[k * Columns + column], sum);
                        }
                        y[row * Columns + column] = sum;
                    }
                    """, string.Empty);
        }
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        try
        {
            AddTemplateInt(config, "Rows", rows); AddTemplateInt(config, "Inner", inner); AddTemplateInt(config, "Columns", columns);
            int[] shape = [rows, columns];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring exact F32 matmul output");
            Check(mlx_fast_metal_kernel_config_set_grid(config, columns, rows, 1), "configuring exact F32 matmul grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1), "configuring exact F32 matmul threadgroup");
            inputs = CreateVectorArray(input, weight); outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, exactFloatMatmulKernel, inputs, config, DefaultStream()), "running exact F32 matmul");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading exact F32 matmul output");
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
