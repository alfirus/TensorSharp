// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Embeddings;

/// <summary>Checks metadata that the managed and direct text graphs would otherwise ignore.</summary>
internal static class EmbeddingGemma2TextMetadata
{
    internal static void Validate(GgufFile file, int globalHead, int localHead)
    {
        const string p = "gemma-embedding2.";
        if (file.GetBool(p + "attention.causal", false) || file.GetUint32(p + "pooling_type", 1) != 1)
            throw new InvalidDataException("EmbeddingGemma 2 requires bidirectional attention and mean pooling.");
        CheckValueWidth("attention.value_length", globalHead);
        CheckValueWidth("attention.value_length_swa", localHead);
        CheckRotaryWidth("rope.dimension_count", globalHead);
        CheckRotaryWidth("rope.dimension_count_swa", localHead);
        float factor = file.GetFloat32(p + "rope.scaling.factor", 1f);
        if (!float.IsFinite(factor) || factor <= 0)
            throw new InvalidDataException("EmbeddingGemma 2 rotary scaling factor must be positive and finite.");
        if (factor != 1f)
            throw new NotSupportedException("The managed and direct EmbeddingGemma 2 text executors do not support rotary scaling; select a GGML backend.");

        void CheckValueWidth(string key, int head)
        {
            if (file.GetUint32(p + key, checked((uint)head)) != (uint)head)
                throw new InvalidDataException($"EmbeddingGemma 2 {key} must equal its key head width.");
        }
        void CheckRotaryWidth(string key, int head)
        {
            if (file.GetUint32(p + key, checked((uint)head)) != (uint)head)
                throw new NotSupportedException($"The managed and direct EmbeddingGemma 2 text executors require {key} to equal its head width; select a GGML backend for partial rotary embeddings.");
        }
    }
}
