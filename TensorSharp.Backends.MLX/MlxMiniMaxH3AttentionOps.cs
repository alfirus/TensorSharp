// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.MLX;

/// <summary>Causal text attention with F32 dot products. This avoids the
/// TF32 GEMM approximation in unfused SDPA while retaining large F32 values.</summary>
public static class MlxMiniMaxH3AttentionOps
{
    public static bool TryCausalAttention(Tensor result, Tensor query, Tensor key, Tensor value,
        int heads, int kvHeads, int head, int sequence, float scale, int maximumQueryRows = 0)
    {
        if (result?.Storage is not MlxStorage rs || query?.Storage is not MlxStorage qs ||
            key?.Storage is not MlxStorage ks || value?.Storage is not MlxStorage vs ||
            result.ElementType != DType.Float32 || query.ElementType != DType.Float32 ||
            key.ElementType != DType.Float32 || value.ElementType != DType.Float32 ||
            result.DimensionCount != 2 || query.DimensionCount != 3 || key.DimensionCount != 3 || value.DimensionCount != 3 ||
            !result.IsContiguous() || !query.IsContiguous() || !key.IsContiguous() || !value.IsContiguous() ||
            !ReferenceEquals(rs.AllocatorImpl, qs.AllocatorImpl) || !ReferenceEquals(rs.AllocatorImpl, ks.AllocatorImpl) ||
            !ReferenceEquals(rs.AllocatorImpl, vs.AllocatorImpl) || heads <= 0 || kvHeads <= 0 || heads % kvHeads != 0 ||
            head <= 0 || sequence <= 0 || !float.IsFinite(scale) || scale <= 0 || maximumQueryRows < 0 ||
            result.Sizes[0] != sequence || result.Sizes[1] != checked(heads * head) ||
            query.Sizes[0] != heads || query.Sizes[1] != sequence || query.Sizes[2] != head ||
            key.Sizes[0] != kvHeads || key.Sizes[1] != sequence || key.Sizes[2] != head ||
            value.Sizes[0] != kvHeads || value.Sizes[1] != sequence || value.Sizes[2] != head) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray q = default, k = default, v = default;
            try
            {
                q = qs.CreateArrayView(query); k = ks.CreateArrayView(key); v = vs.CreateArrayView(value);
                int queryRows = (int)Math.Max(1, Math.Min(sequence, (64L << 20) / (sizeof(float) * (long)heads * sequence)));
                if (maximumQueryRows > 0) queryRows = Math.Min(queryRows, maximumQueryRows);
                for (int start = 0; start < sequence; start += queryRows)
                {
                    int rows = Math.Min(queryRows, sequence - start);
                    using var scores = new Tensor(rs.AllocatorImpl, DType.Float32, heads, rows, sequence);
                    ((MlxStorage)scores.Storage).ReplaceDeviceArray(MlxNative.H3ExactAttentionKernel(q, k, heads, kvHeads,
                        head, sequence, start, rows, score: true));
                    Ops.Mul(scores, scores, scale);
                    using var flat = scores.View(checked(heads * rows), sequence);
                    using var probabilities = Ops.Softmax(null, flat);
                    MlxNative.MlxArray p = default, output = default;
                    try
                    {
                        p = ((MlxStorage)probabilities.Storage).CreateArrayView(probabilities);
                        output = MlxNative.H3ExactAttentionKernel(p, v, heads, kvHeads, head, sequence, start, rows, score: false);
                        using var target = result.Narrow(0, start, rows);
                        rs.UpdateDeviceSlice(target, output);
                    }
                    finally { MlxNative.FreeArray(p); MlxNative.FreeArray(output); }
                }
                return true;
            }
            finally { MlxNative.FreeArray(q); MlxNative.FreeArray(k); MlxNative.FreeArray(v); }
        });
    }
}

internal static partial class MlxNative
{
    private static MlxFastMetalKernel h3ExactScoreKernel, h3ExactValueKernel;

    internal static MlxArray H3ExactAttentionKernel(MlxArray first, MlxArray second, int heads, int kvHeads,
        int head, int sequence, int start, int rows, bool score)
    {
        lock (fastKernelSync)
        {
            if (!h3ExactScoreKernel.IsValid)
                h3ExactScoreKernel = CreateFastMetalKernel("tensorsharp_h3_exact_causal_scores", ["q", "k"], ["out"], """
                    const uint key = thread_position_in_grid.x;
                    const uint row = thread_position_in_grid.y;
                    const uint h = thread_position_in_grid.z;
                    if (key < Sequence && row < Rows && h < Heads) {
                        const uint query = Start + row;
                        float sum = 0.0f;
                        if (key <= query) {
                            const uint kv = h / (Heads / KvHeads);
                            #pragma clang loop unroll(disable)
                            for (uint channel = 0; channel < Head; ++channel)
                                sum = fma(q[(h * Sequence + query) * Head + channel],
                                    k[(kv * Sequence + key) * Head + channel], sum);
                        } else sum = -INFINITY;
                        out[(h * Rows + row) * Sequence + key] = sum;
                    }
                    """, string.Empty);
            if (!h3ExactValueKernel.IsValid)
                h3ExactValueKernel = CreateFastMetalKernel("tensorsharp_h3_exact_causal_values", ["p", "v"], ["out"], """
                    const uint channel = thread_position_in_grid.x;
                    const uint row = thread_position_in_grid.y;
                    const uint h = thread_position_in_grid.z;
                    if (channel < Head && row < Rows && h < Heads) {
                        const uint kv = h / (Heads / KvHeads);
                        float sum = 0.0f;
                        #pragma clang loop unroll(disable)
                        for (uint key = 0; key <= Start + row; ++key)
                            sum = fma(p[(h * Rows + row) * Sequence + key],
                                v[(kv * Sequence + key) * Head + channel], sum);
                        out[(row * Heads + h) * Head + channel] = sum;
                    }
                    """, string.Empty);
        }
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        try
        {
            AddTemplateInt(config, "Heads", heads); AddTemplateInt(config, "KvHeads", kvHeads);
            AddTemplateInt(config, "Head", head); AddTemplateInt(config, "Sequence", sequence);
            AddTemplateInt(config, "Start", start); AddTemplateInt(config, "Rows", rows);
            int[] shape = score ? [heads, rows, sequence] : [rows, checked(heads * head)];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring H3 exact attention output");
            Check(mlx_fast_metal_kernel_config_set_grid(config, score ? sequence : head, rows, heads), "configuring H3 exact attention grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1), "configuring H3 exact attention threadgroup");
            inputs = CreateVectorArray(first, second); outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, score ? h3ExactScoreKernel : h3ExactValueKernel, inputs, config, DefaultStream()), "running H3 exact attention");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading H3 exact attention output");
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
