// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.MLX;

/// <summary>F32 sliding relative attention and causal depthwise convolution for Gemma4 audio.</summary>
public static class MlxGemma4AudioOps
{
    private static bool Compatible(Tensor result, params Tensor[] inputs)
    {
        if (result?.Storage is not MlxStorage target || result.ElementType != DType.Float32 ||
            result.DimensionCount != 2 || !result.IsContiguous() || result.StorageOffset != 0 ||
            result.Storage.ElementCount != result.ElementCount()) return false;
        foreach (var input in inputs)
            if (input?.Storage is not MlxStorage source || input.ElementType != DType.Float32 ||
                input.DimensionCount != 2 || !input.IsContiguous() || ReferenceEquals(result.Storage, input.Storage) ||
                !ReferenceEquals(target.AllocatorImpl, source.AllocatorImpl)) return false;
        return true;
    }

    public static bool TryAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor relative,
        int sequence, int heads, int head, int past, int future, float cap)
    {
        if (!Compatible(result, query, key, value, relative) || sequence <= 0 || heads <= 0 || head <= 0 ||
            past < 0 || future < 0 || past + future > 256 || !float.IsFinite(cap) || cap <= 0 ||
            (long)sequence * heads * (past + future + 1) > 16 * 1024 * 1024 ||
            query.Sizes[0] != sequence || query.Sizes[1] != (long)heads * head ||
            result.Sizes[0] != sequence || result.Sizes[1] != query.Sizes[1] ||
            key.Sizes[0] != sequence || key.Sizes[1] != query.Sizes[1] ||
            value.Sizes[0] != sequence || value.Sizes[1] != query.Sizes[1] ||
            relative.Sizes[0] != past + future + 1 || relative.Sizes[1] != query.Sizes[1]) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray q = default, k = default, v = default, r = default, output = default;
            try
            {
                q = ((MlxStorage)query.Storage).CreateArrayView(query); k = ((MlxStorage)key.Storage).CreateArrayView(key);
                v = ((MlxStorage)value.Storage).CreateArrayView(value); r = ((MlxStorage)relative.Storage).CreateArrayView(relative);
                output = MlxNative.Gemma4AudioAttention(q, k, v, r, sequence, heads, head, past, future, cap);
                ((MlxStorage)result.Storage).ReplaceDeviceArray(output); output = default; return true;
            }
            finally { MlxNative.FreeArray(q); MlxNative.FreeArray(k); MlxNative.FreeArray(v); MlxNative.FreeArray(r); MlxNative.FreeArray(output); }
        });
    }

    public static bool TryDepthwise(Tensor result, Tensor input, Tensor weight, int sequence, int channels, int kernel)
    {
        if (!Compatible(result, input, weight) || sequence <= 0 || channels <= 0 || kernel <= 0 || kernel > 256 ||
            input.Sizes[0] != sequence || input.Sizes[1] != channels || result.Sizes[0] != sequence || result.Sizes[1] != channels ||
            weight.Sizes[0] != channels || weight.Sizes[1] != kernel) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, w = default, output = default;
            try
            {
                x = ((MlxStorage)input.Storage).CreateArrayView(input); w = ((MlxStorage)weight.Storage).CreateArrayView(weight);
                output = MlxNative.Gemma4AudioKernel(2, [x, w], sequence, 1, channels, 0, 0, kernel);
                ((MlxStorage)result.Storage).ReplaceDeviceArray(output); output = default; return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(w); MlxNative.FreeArray(output); }
        });
    }
}

internal static partial class MlxNative
{
    private static MlxFastMetalKernel gemma4AudioScores, gemma4AudioValues, gemma4AudioDepthwise;

    internal static MlxArray Gemma4AudioAttention(MlxArray q, MlxArray k, MlxArray v, MlxArray relative,
        int sequence, int heads, int head, int past, int future, float cap)
    {
        MlxArray capArray = default, scores = default, probabilities = default;
        try
        {
            capArray = NewScalar(cap);
            scores = Gemma4AudioKernel(0, [q, k, relative, capArray], sequence, heads, head, past, future, 0);
            probabilities = SoftmaxLastAxis(scores);
            return Gemma4AudioKernel(1, [probabilities, v], sequence, heads, head, past, future, 0);
        }
        finally { FreeArray(capArray); FreeArray(scores); FreeArray(probabilities); }
    }

    internal static MlxArray Gemma4AudioKernel(int op, MlxArray[] operands, int sequence, int heads, int head,
        int past, int future, int kernel)
    {
        lock (fastKernelSync)
        {
            if (!gemma4AudioScores.IsValid)
                gemma4AudioScores = CreateFastMetalKernel("tensorsharp_gemma4_audio_scores_f32", ["q", "k", "r", "cap"], ["out"], """
                    const uint slot = thread_position_in_grid.x, t = thread_position_in_grid.y, h = thread_position_in_grid.z;
                    if (slot < Window && t < Sequence && h < Heads) {
                        const int key = int(t) + int(slot) - Past;
                        float cc = 0.0f, cp = 0.0f;
                        if (key >= 0 && key < Sequence) {
                            #pragma clang loop unroll(disable)
                            for (uint d = 0; d < Head; ++d) {
                                const float query = q[(t * Heads + h) * Head + d];
                                cc = fma(query, k[(uint(key) * Heads + h) * Head + d], cc);
                                cp = fma(query, r[(slot * Heads + h) * Head + d], cp);
                            }
                            out[(t * Heads + h) * Window + slot] = tanh((cc + cp) / cap) * cap;
                        } else out[(t * Heads + h) * Window + slot] = -1.0e9f;
                    }
                    """, string.Empty);
            if (!gemma4AudioValues.IsValid)
                gemma4AudioValues = CreateFastMetalKernel("tensorsharp_gemma4_audio_values_f32", ["p", "v"], ["out"], """
                    const uint d = thread_position_in_grid.x, t = thread_position_in_grid.y, h = thread_position_in_grid.z;
                    if (d < Head && t < Sequence && h < Heads) {
                        float sum = 0.0f;
                        #pragma clang loop unroll(disable)
                        for (uint slot = 0; slot < Window; ++slot) {
                            const int key = int(t) + int(slot) - Past;
                            if (key >= 0 && key < Sequence)
                                sum = fma(p[(t * Heads + h) * Window + slot], v[(uint(key) * Heads + h) * Head + d], sum);
                        }
                        out[(t * Heads + h) * Head + d] = sum;
                    }
                    """, string.Empty);
            if (!gemma4AudioDepthwise.IsValid)
                gemma4AudioDepthwise = CreateFastMetalKernel("tensorsharp_gemma4_audio_depthwise_f32", ["x", "w"], ["out"], """
                    const uint channel = thread_position_in_grid.x, t = thread_position_in_grid.y;
                    if (channel < Head && t < Sequence) {
                        float sum = 0.0f;
                        #pragma clang loop unroll(disable)
                        for (uint k = 0; k < Kernel; ++k) {
                            const int source = int(t) - (Kernel - 1 - int(k));
                            if (source >= 0) sum = fma(x[uint(source) * Head + channel], w[channel * Kernel + k], sum);
                        }
                        out[t * Head + channel] = sum;
                    }
                    """, string.Empty);
        }
        var config = mlx_fast_metal_kernel_config_new(); MlxVectorArray inputs = default, outputs = default;
        try
        {
            int window = past + future + 1;
            AddTemplateInt(config, "Sequence", sequence); AddTemplateInt(config, "Heads", heads); AddTemplateInt(config, "Head", head);
            AddTemplateInt(config, "Past", past); AddTemplateInt(config, "Window", window); AddTemplateInt(config, "Kernel", kernel);
            int[] shape = op == 0 ? [sequence, heads, window] : [sequence, checked(heads * head)];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring Gemma4 audio output");
            Check(mlx_fast_metal_kernel_config_set_grid(config, op == 0 ? window : head, sequence, heads), "configuring Gemma4 audio grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1), "configuring Gemma4 audio threadgroup");
            inputs = CreateVectorArray(operands); outputs = mlx_vector_array_new();
            var function = op == 0 ? gemma4AudioScores : op == 1 ? gemma4AudioValues : gemma4AudioDepthwise;
            Check(mlx_fast_metal_kernel_apply(ref outputs, function, inputs, config, DefaultStream()), "running Gemma4 audio kernel");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading Gemma4 audio output");
            return result;
        }
        finally { if (inputs.IsValid) _ = mlx_vector_array_free(inputs); if (outputs.IsValid) _ = mlx_vector_array_free(outputs); if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config); }
    }
}
