// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;

namespace TensorSharp.MLX;

/// <summary>Pack short independent sequences into one device attention batch.</summary>
public static class MlxRaggedEmbeddingOps
{
    public static bool TryAttention(Tensor result, Tensor query, Tensor key, Tensor value, int[] lengths, float scale,
        MlxRaggedEmbeddingAttentionPlan plan, bool useFloat16Operands = false)
    {
        if (result?.Storage is not MlxStorage rs || query?.Storage is not MlxStorage qs || key?.Storage is not MlxStorage ks || value?.Storage is not MlxStorage vs ||
            result.ElementType != DType.Float32 || query.ElementType != DType.Float32 || key.ElementType != DType.Float32 || value.ElementType != DType.Float32 ||
            result.DimensionCount != 4 || query.DimensionCount != 4 || key.DimensionCount != 4 || value.DimensionCount != 4 ||
            !result.IsContiguous() || !query.IsContiguous() || !key.IsContiguous() || !value.IsContiguous() ||
            result.StorageOffset != 0 || result.Storage.ElementCount != result.ElementCount() ||
            !ReferenceEquals(rs.AllocatorImpl, qs.AllocatorImpl) || !ReferenceEquals(rs.AllocatorImpl, ks.AllocatorImpl) || !ReferenceEquals(rs.AllocatorImpl, vs.AllocatorImpl) ||
            query.Sizes[0] != 1 || query.Sizes[1] <= 0 || query.Sizes[1] > 1536 || query.Sizes[2] <= 0 || key.Sizes[2] <= 0 ||
            query.Sizes[3] is not (256 or 512) || query.Sizes[2] % key.Sizes[2] != 0 || !float.IsFinite(scale) ||
            key.Sizes[0] != 1 || key.Sizes[1] != query.Sizes[1] || key.Sizes[3] != query.Sizes[3] ||
            value.Sizes[0] != 1 || value.Sizes[1] != key.Sizes[1] || value.Sizes[2] != key.Sizes[2] || value.Sizes[3] != key.Sizes[3] ||
            lengths == null || lengths.Length < 2 || lengths.Length > 64 || plan == null) return false;
        long total = 0;
        foreach (int length in lengths) { if (length < 1 || length > 512) return false; total += length; }
        if (total != query.Sizes[1]) return false;
        for (int dimension = 0; dimension < 4; ++dimension)
            if (result.Sizes[dimension] != query.Sizes[dimension]) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray q = default, k = default, v = default, output = default;
            try
            {
                q = qs.CreateArrayView(query); k = ks.CreateArrayView(key); v = vs.CreateArrayView(value);
                output = MlxNative.RaggedEmbeddingAttention(q, k, v, plan.Geometry(lengths, useFloat16Operands),
                    checked((int)query.Sizes[2]), checked((int)key.Sizes[2]), checked((int)query.Sizes[3]), scale, useFloat16Operands);
                rs.ReplaceDeviceArray(output); output = default; return true;
            }
            finally { MlxNative.FreeArray(q); MlxNative.FreeArray(k); MlxNative.FreeArray(v); MlxNative.FreeArray(output); }
        });
    }
}

/// <summary>Immutable bounded geometry cache owned by the embedding model.</summary>
public sealed class MlxRaggedEmbeddingAttentionPlan : IDisposable
{
    internal sealed record Entry(MlxNative.MlxArray Offsets, MlxNative.MlxArray TokenSequence, MlxNative.MlxArray Mask, int Batch, int Tokens, int PaddedLength);
    private readonly Dictionary<string, Entry> _entries = new();

    internal unsafe Entry Geometry(int[] lengths, bool half)
    {
        string key = $"{half}:{string.Join(",", lengths)}";
        if (_entries.TryGetValue(key, out var cached)) return cached;
        if (_entries.Count >= 8) Clear();
        int[] offsets = new int[lengths.Length + 1];
        int maximum = 0;
        for (int i = 0; i < lengths.Length; ++i) { offsets[i + 1] = checked(offsets[i] + lengths[i]); maximum = Math.Max(maximum, lengths[i]); }
        int padded = checked((maximum + 31) / 32 * 32), tokens = offsets[^1];
        int[] sequence = new int[tokens];
        var maskValues = new float[checked(lengths.Length * padded)];
        for (int i = 0; i < lengths.Length; ++i)
        {
            Array.Fill(sequence, i, offsets[i], lengths[i]);
            Array.Fill(maskValues, float.NegativeInfinity, i * padded + lengths[i], padded - lengths[i]);
        }
        MlxNative.MlxArray o = default, s = default, m = default;
        try
        {
            fixed (int* p = offsets) o = MlxNative.NewArrayFromHost((IntPtr)p, [offsets.Length], DType.Int32);
            fixed (int* p = sequence) s = MlxNative.NewArrayFromHost((IntPtr)p, [tokens], DType.Int32);
            fixed (float* p = maskValues) m = MlxNative.NewArrayFromHost((IntPtr)p, [lengths.Length, 1, 1, padded], DType.Float32);
            if (half) { var narrowed = MlxNative.Astype(m, DType.Float16); MlxNative.FreeArray(m); m = narrowed; }
            var entry = new Entry(o, s, m, lengths.Length, tokens, padded);
            _entries.Add(key, entry); o = default; s = default; m = default; return entry;
        }
        finally { MlxNative.FreeArray(o); MlxNative.FreeArray(s); MlxNative.FreeArray(m); }
    }

    private void Clear()
    {
        foreach (var entry in _entries.Values) { MlxNative.FreeArray(entry.Offsets); MlxNative.FreeArray(entry.TokenSequence); MlxNative.FreeArray(entry.Mask); }
        _entries.Clear();
    }
    public void Dispose() => MlxWorker.Shared.Invoke(Clear);
}

internal static partial class MlxNative
{
    private static MlxFastMetalKernel raggedEmbeddingPackKernel, raggedEmbeddingUnpackKernel;

    internal static MlxArray RaggedEmbeddingAttention(MlxArray q, MlxArray k, MlxArray v,
        MlxRaggedEmbeddingAttentionPlan.Entry geometry, int heads, int kvHeads, int head, float scale, bool half)
    {
        lock (fastKernelSync)
        {
            if (!raggedEmbeddingPackKernel.IsValid)
                raggedEmbeddingPackKernel = CreateFastMetalKernel("tensorsharp_ragged_embedding_pack", ["x", "offsets"], ["y"], """
                    const uint index = thread_position_in_grid.x;
                    if (index < Elements) {
                        const uint dimension = index % Head;
                        const uint token = (index / Head) % Length;
                        const uint attentionHead = (index / (Head * Length)) % Heads;
                        const uint sequence = index / (Head * Length * Heads);
                        const uint start = offsets[sequence];
                        const uint count = offsets[sequence + 1] - start;
                        y[index] = token < count ? x[((start + token) * Heads + attentionHead) * Head + dimension] : 0.0f;
                    }
                    """, string.Empty);
            if (!raggedEmbeddingUnpackKernel.IsValid)
                raggedEmbeddingUnpackKernel = CreateFastMetalKernel("tensorsharp_ragged_embedding_unpack", ["x", "offsets", "token_sequence"], ["y"], """
                    const uint index = thread_position_in_grid.x;
                    if (index < Elements) {
                        const uint dimension = index % Head;
                        const uint attentionHead = (index / Head) % Heads;
                        const uint token = index / (Head * Heads);
                        const uint sequence = token_sequence[token];
                        const uint localToken = token - offsets[sequence];
                        y[index] = x[((sequence * Heads + attentionHead) * Length + localToken) * Head + dimension];
                    }
                    """, string.Empty);
        }
        MlxArray qp = default, kp = default, vp = default, attention = default;
        try
        {
            qp = RaggedEmbeddingKernel(q, geometry, heads, head, half, pack: true);
            kp = RaggedEmbeddingKernel(k, geometry, kvHeads, head, half, pack: true);
            vp = RaggedEmbeddingKernel(v, geometry, kvHeads, head, half, pack: true);
            attention = FastScaledDotProductAttention(qp, kp, vp, scale, "array", geometry.Mask);
            return RaggedEmbeddingKernel(attention, geometry, heads, head, half: false, pack: false);
        }
        finally { FreeArray(qp); FreeArray(kp); FreeArray(vp); FreeArray(attention); }
    }

    private static MlxArray RaggedEmbeddingKernel(MlxArray input, MlxRaggedEmbeddingAttentionPlan.Entry geometry, int heads, int head, bool half, bool pack)
    {
        int elements = checked((pack ? geometry.Batch * geometry.PaddedLength : geometry.Tokens) * heads * head);
        int[] shape = pack ? [geometry.Batch, heads, geometry.PaddedLength, head] : [1, geometry.Tokens, heads, head];
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        try
        {
            AddTemplateInt(config, "Elements", elements); AddTemplateInt(config, "Heads", heads); AddTemplateInt(config, "Head", head); AddTemplateInt(config, "Length", geometry.PaddedLength);
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(half ? DType.Float16 : DType.Float32)), "configuring ragged embedding attention layout output");
            Check(mlx_fast_metal_kernel_config_set_grid(config, elements, 1, 1), "configuring ragged embedding attention layout grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring ragged embedding attention layout threadgroup");
            inputs = pack ? CreateVectorArray(input, geometry.Offsets) : CreateVectorArray(input, geometry.Offsets, geometry.TokenSequence);
            outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, pack ? raggedEmbeddingPackKernel : raggedEmbeddingUnpackKernel, inputs, config, DefaultStream()), "running ragged embedding attention layout kernel");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading ragged embedding attention layout output");
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
