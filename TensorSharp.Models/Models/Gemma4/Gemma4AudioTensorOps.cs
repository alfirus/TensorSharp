// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models;

internal static class Gemma4AudioTensorOps
{
    internal static bool UsesDevicePath(IAllocator allocator) => allocator is MlxAllocator or CudaAllocator or GgmlAllocator;

    internal static bool TryAttention(Tensor result, Tensor q, Tensor k, Tensor v, Tensor relative,
        int sequence, int heads, int head, int past, int future, float cap)
        => result.Allocator switch
        {
            MlxAllocator => MlxGemma4AudioOps.TryAttention(result, q, k, v, relative, sequence, heads, head, past, future, cap),
            CudaAllocator => CudaGemma4AudioOps.TryAttention(result, q, k, v, relative, sequence, heads, head, past, future, cap),
            GgmlAllocator => GgmlGemma4AudioOps.TryAttention(result, q, k, v, relative, sequence, heads, head, past, future, cap),
            _ => false
        };

    internal static bool TryDepthwise(Tensor result, Tensor input, Tensor weight, int sequence, int channels, int kernel)
        => result.Allocator switch
        {
            MlxAllocator => MlxGemma4AudioOps.TryDepthwise(result, input, weight, sequence, channels, kernel),
            CudaAllocator => CudaGemma4AudioOps.TryDepthwise(result, input, weight, sequence, channels, kernel),
            GgmlAllocator => GgmlGemma4AudioOps.TryDepthwise(result, input, weight, sequence, channels, kernel),
            _ => false
        };
}
