// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;

namespace TensorSharp.MLX;

/// <summary>Bounded-memory bidirectional attention on MLX for embedding inputs of at least 512 tokens.</summary>
public static class MlxEmbeddingOps
{
    /// <summary>F16 operands for statically bounded post-RMS projections; result and residual storage remain F32.
    /// Caller must bound normalized input components and all partial dots below 16384, including weight rounding.</summary>
    public static bool TryNormalizedHalfProjection(Tensor result, Tensor input, Tensor halfWeight)
    {
        if (result?.Storage is not MlxStorage rs || input?.Storage is not MlxStorage xs || halfWeight?.Storage is not MlxStorage ws ||
            result.ElementType != DType.Float32 || input.ElementType != DType.Float32 || halfWeight.ElementType != DType.Float16 ||
            result.DimensionCount != 2 || input.DimensionCount != 2 || halfWeight.DimensionCount != 2 ||
            !result.IsContiguous() || !input.IsContiguous() || !halfWeight.IsContiguous() ||
            result.StorageOffset != 0 || result.Storage.ElementCount != result.ElementCount() ||
            ReferenceEquals(result.Storage, input.Storage) || ReferenceEquals(result.Storage, halfWeight.Storage) ||
            result.Sizes[0] != input.Sizes[0] || result.Sizes[1] != halfWeight.Sizes[1] || input.Sizes[1] != halfWeight.Sizes[0] ||
            !ReferenceEquals(rs.AllocatorImpl, xs.AllocatorImpl) || !ReferenceEquals(rs.AllocatorImpl, ws.AllocatorImpl)) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, w = default, narrowed = default, product = default, widened = default;
            try
            {
                x = xs.CreateArrayView(input); w = ws.CreateArrayView(halfWeight);
                narrowed = MlxNative.Astype(x, DType.Float16);
                product = MlxNative.NormalizedHalfProjection(narrowed, w);
                widened = MlxNative.Astype(product, DType.Float32);
                rs.ReplaceDeviceArray(widened); widened = default; return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(w); MlxNative.FreeArray(narrowed); MlxNative.FreeArray(product); MlxNative.FreeArray(widened); }
        });
    }

    /// <summary>One bounded block-mask attention call; callers must prove normalized operands and score dots fit F16 before requesting it.</summary>
    public static bool TryMaskedAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor mask, float scale,
        bool useFloat16Operands = false)
    {
        if (result?.Storage is not MlxStorage rs || query?.Storage is not MlxStorage qs ||
            key?.Storage is not MlxStorage ks || value?.Storage is not MlxStorage vs || mask?.Storage is not MlxStorage ms ||
            result.ElementType != DType.Float32 || query.ElementType != DType.Float32 || key.ElementType != DType.Float32 ||
            value.ElementType != DType.Float32 || mask.ElementType != DType.Float32 ||
            result.DimensionCount != 4 || query.DimensionCount != 4 || key.DimensionCount != 4 || value.DimensionCount != 4 || mask.DimensionCount != 4 ||
            !result.IsContiguous() || !query.IsContiguous() || !key.IsContiguous() || !value.IsContiguous() || !mask.IsContiguous() ||
            result.StorageOffset != 0 || result.Storage.ElementCount != result.ElementCount() ||
            !ReferenceEquals(rs.AllocatorImpl, qs.AllocatorImpl) || !ReferenceEquals(rs.AllocatorImpl, ks.AllocatorImpl) ||
            !ReferenceEquals(rs.AllocatorImpl, vs.AllocatorImpl) || !ReferenceEquals(rs.AllocatorImpl, ms.AllocatorImpl) ||
            query.Sizes[0] <= 0 || query.Sizes[1] <= 0 || query.Sizes[1] > 1536 || query.Sizes[2] <= 0 || key.Sizes[2] <= 0 ||
            query.Sizes[3] is not (256 or 512) || query.Sizes[2] % key.Sizes[2] != 0 || !float.IsFinite(scale) ||
            query.Sizes[0] != key.Sizes[0] || query.Sizes[1] != key.Sizes[1] || query.Sizes[3] != key.Sizes[3] ||
            key.Sizes[0] != value.Sizes[0] || key.Sizes[1] != value.Sizes[1] || key.Sizes[2] != value.Sizes[2] || key.Sizes[3] != value.Sizes[3] ||
            (mask.Sizes[0] != 1 && mask.Sizes[0] != query.Sizes[0]) || (mask.Sizes[1] != 1 && mask.Sizes[1] != query.Sizes[2]) ||
            mask.Sizes[2] != query.Sizes[1] || mask.Sizes[3] != key.Sizes[1]) return false;
        for (int dimension = 0; dimension < 4; ++dimension)
            if (result.Sizes[dimension] != query.Sizes[dimension]) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray q = default, k = default, v = default, m = default, output = default;
            try
            {
                q = qs.CreateArrayView(query); k = ks.CreateArrayView(key); v = vs.CreateArrayView(value); m = ms.CreateArrayView(mask);
                output = MlxNative.MaskedEmbeddingAttention(q, k, v, m, scale, useFloat16Operands);
                rs.ReplaceDeviceArray(output); output = default; return true;
            }
            finally { MlxNative.FreeArray(q); MlxNative.FreeArray(k); MlxNative.FreeArray(v); MlxNative.FreeArray(m); MlxNative.FreeArray(output); }
        });
    }

    public static bool TryAttention(Tensor result, Tensor query, Tensor key, Tensor value, int windowRadius, float scale,
        MlxEmbeddingAttentionPlan plan = null, bool useFloat16Operands = false, int[] sequenceLengths = null)
    {
        if (result?.Storage is not MlxStorage rs || query?.Storage is not MlxStorage qs ||
            key?.Storage is not MlxStorage ks || value?.Storage is not MlxStorage vs ||
            query.DimensionCount != 4 || key.DimensionCount != 4 || value.DimensionCount != 4 ||
            result.DimensionCount != 4 || !result.IsContiguous() || !query.IsContiguous() || !key.IsContiguous() || !value.IsContiguous() ||
            result.ElementType != DType.Float32 || query.ElementType != DType.Float32 ||
            key.ElementType != DType.Float32 || value.ElementType != DType.Float32 ||
            !ReferenceEquals(rs.AllocatorImpl, qs.AllocatorImpl) || !ReferenceEquals(rs.AllocatorImpl, ks.AllocatorImpl) ||
            !ReferenceEquals(rs.AllocatorImpl, vs.AllocatorImpl) || query.Sizes[1] < 512 || query.Sizes[1] > 8192 ||
            result.StorageOffset != 0 || result.Storage.ElementCount != result.ElementCount() ||
            query.Sizes[0] <= 0 || query.Sizes[2] <= 0 || key.Sizes[2] <= 0 || query.Sizes[3] is not (256 or 512) ||
            query.Sizes[0] != key.Sizes[0] || query.Sizes[1] != key.Sizes[1] ||
            key.Sizes[0] != value.Sizes[0] || key.Sizes[1] != value.Sizes[1] || key.Sizes[2] != value.Sizes[2] ||
            query.Sizes[3] != key.Sizes[3] || query.Sizes[3] != value.Sizes[3] ||
            query.Sizes[2] % key.Sizes[2] != 0 || windowRadius < 0 || windowRadius > 8192 || !float.IsFinite(scale)) return false;
        for (int dimension = 0; dimension < 4; ++dimension)
            if (result.Sizes[dimension] != query.Sizes[dimension]) return false;
        if (sequenceLengths != null)
        {
            long total = 0;
            foreach (int length in sequenceLengths) { if (length <= 0) return false; total += length; }
            if (query.Sizes[0] != 1 || total != query.Sizes[1]) return false;
        }
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray q = default, k = default, v = default, output = default;
            try
            {
                q = qs.CreateArrayView(query); k = ks.CreateArrayView(key); v = vs.CreateArrayView(value);
                output = MlxNative.EmbeddingAttention(q, k, v, checked((int)query.Sizes[0]), checked((int)query.Sizes[1]),
                    checked((int)query.Sizes[2]), checked((int)key.Sizes[2]), checked((int)query.Sizes[3]), windowRadius, scale, plan, useFloat16Operands, sequenceLengths);
                rs.ReplaceDeviceArray(output); output = default; return true;
            }
            finally { MlxNative.FreeArray(q); MlxNative.FreeArray(k); MlxNative.FreeArray(v); MlxNative.FreeArray(output); }
        });
    }
}

/// <summary>Reusable local attention geometry, released with the embedding model.</summary>
public sealed class MlxEmbeddingAttentionPlan : IDisposable
{
    private readonly Dictionary<(int Queries, int Keys, int Offset, int Radius, bool Half), MlxNative.MlxArray> _masks = new();
    internal MlxNative.MlxArray Mask(int queries, int keys, int offset, int radius, bool half)
    {
        var key = (queries, keys, offset, radius, half);
        if (_masks.TryGetValue(key, out var mask)) return mask;
        if (_masks.Count >= 16) { foreach (var item in _masks.Values) MlxNative.FreeArray(item); _masks.Clear(); }
        mask = MlxNative.EmbeddingMask(queries, keys, offset, radius, half); _masks.Add(key, mask); return mask;
    }
    public void Dispose()
    { MlxWorker.Shared.Invoke(() => { foreach (var mask in _masks.Values) MlxNative.FreeArray(mask); _masks.Clear(); }); }
}

internal static partial class MlxNative
{
    internal static MlxArray NormalizedHalfProjection(MlxArray input, MlxArray weight)
    {
        Check(mlx_matmul(out var result, input, weight, DefaultStream()), "running bounded normalized F16 projection");
        return result;
    }

    internal static MlxArray MaskedEmbeddingAttention(MlxArray query, MlxArray key, MlxArray value, MlxArray mask, float scale, bool half)
    {
        MlxArray q = default, k = default, v = default, m = default, attention = default, seqMajor = default;
        try
        {
            q = Transpose(query, [0, 2, 1, 3]); k = Transpose(key, [0, 2, 1, 3]); v = Transpose(value, [0, 2, 1, 3]);
            if (half)
            {
                NarrowAttentionOperand(ref q); NarrowAttentionOperand(ref k); NarrowAttentionOperand(ref v);
                m = Astype(mask, DType.Float16);
            }
            attention = FastScaledDotProductAttention(q, k, v, scale, "array", half ? m : mask);
            if (half) { var widened = Astype(attention, DType.Float32); FreeArray(attention); attention = widened; }
            seqMajor = Transpose(attention, [0, 2, 1, 3]);
            return Contiguous(seqMajor);
        }
        finally { FreeArray(q); FreeArray(k); FreeArray(v); FreeArray(m); FreeArray(attention); FreeArray(seqMajor); }
    }

    internal static MlxArray EmbeddingAttention(MlxArray q, MlxArray k, MlxArray v,
        int batch, int tokens, int heads, int kvHeads, int head, int radius, float scale, MlxEmbeddingAttentionPlan plan, bool half, int[] sequenceLengths)
    {
        var chunks = new List<MlxArray>();
        MlxArray qh = default, kh = default, vh = default, joined = default, seqMajor = default;
        try
        {
            qh = Transpose(q, [0, 2, 1, 3]); kh = Transpose(k, [0, 2, 1, 3]); vh = Transpose(v, [0, 2, 1, 3]);
            if (half)
            {
                NarrowAttentionOperand(ref qh); NarrowAttentionOperand(ref kh); NarrowAttentionOperand(ref vh);
            }
            int block = radius > 0 ? 1024 : 512;
            int sequenceStart = 0;
            foreach (int length in sequenceLengths ?? [tokens])
            {
            int sequenceEnd = sequenceStart + length;
            for (int start = sequenceStart; start < sequenceEnd; start += block)
            {
                int end = Math.Min(sequenceEnd, start + block);
                int firstKey = radius > 0 ? Math.Max(sequenceStart, start - radius) : sequenceStart;
                int lastKey = radius > 0 ? Math.Min(sequenceEnd, end + radius) : sequenceEnd;
                MlxArray qc = default, kc = default, vc = default, mask = default;
                try
                {
                    qc = Slice(qh, [0, 0, start, 0], [batch, heads, end, head], [1, 1, 1, 1]);
                    kc = Slice(kh, [0, 0, firstKey, 0], [batch, kvHeads, lastKey, head], [1, 1, 1, 1]);
                    vc = Slice(vh, [0, 0, firstKey, 0], [batch, kvHeads, lastKey, head], [1, 1, 1, 1]);
                    if (radius > 0)
                    {
                        int nq = end - start, nk = lastKey - firstKey;
                        mask = plan == null ? EmbeddingMask(nq, nk, start - firstKey, radius, half) : plan.Mask(nq, nk, start - firstKey, radius, half);
                    }
                    var chunk = FastScaledDotProductAttention(qc, kc, vc, scale, radius > 0 ? "array" : "", mask);
                    chunks.Add(chunk);
                    if (half) { var widened = Astype(chunk, DType.Float32); FreeArray(chunk); chunk = widened; chunks[^1] = widened; }
                    // Only huge global hd512 scores need an evaluation barrier
                    // to bound peak memory. Smaller segments stay in one lazy
                    // graph so projection and pointwise operations can overlap.
                    if (head == 512 && lastKey - firstKey >= 4096) Eval(chunk);
                }
                finally { FreeArray(qc); FreeArray(kc); FreeArray(vc); if (plan == null) FreeArray(mask); }
            }
            sequenceStart = sequenceEnd;
            }
            MlxVectorArray arrays = CreateVectorArray(chunks.ToArray());
            try { Check(mlx_concatenate_axis(out joined, arrays, 2, DefaultStream()), "concatenating embedding attention chunks"); }
            finally { if (arrays.IsValid) _ = mlx_vector_array_free(arrays); }
            seqMajor = Transpose(joined, [0, 2, 1, 3]);
            return Contiguous(seqMajor);
        }
        finally
        {
            foreach (var chunk in chunks) FreeArray(chunk);
            FreeArray(qh); FreeArray(kh); FreeArray(vh); FreeArray(joined); FreeArray(seqMajor);
        }
    }

    private static void NarrowAttentionOperand(ref MlxArray input)
    {
        var narrowed = Astype(input, DType.Float16); FreeArray(input); input = narrowed;
    }

    internal static unsafe MlxArray EmbeddingMask(int queries, int keys, int offset, int radius, bool half)
    {
        var values = new float[checked(queries * keys)];
        for (int q = 0; q < queries; ++q)
            for (int k = 0; k < keys; ++k)
                if (Math.Abs(q + offset - k) > radius) values[q * keys + k] = float.NegativeInfinity;
        fixed (float* pointer = values)
        {
            var mask = NewArrayFromHost((IntPtr)pointer, [1, 1, queries, keys], DType.Float32);
            try { if (half) NarrowAttentionOperand(ref mask); Eval(mask); return mask; }
            catch { FreeArray(mask); throw; }
        }
    }
}
