// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace TensorSharp.Models.Embeddings;

/// <summary>Gemma attention with four F32 queries and two 16-lane key/value
/// vectors in registers. Each lane retains the original FMA reduction order.
/// K is [channel,key]; V uses 16-channel tiles [tile,key,lane].</summary>
internal static class ManagedEmbeddingAttentionAvx512
{
    public static bool IsSupported => Avx512F.IsSupported && Avx2.IsSupported && Vector512.IsHardwareAccelerated &&
        Environment.GetEnvironmentVariable("TS_CPU_DISABLE_AVX512") != "1";
    public const int ValueWidth = 16;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Scores(float* q0, float* q1, float* q2, float* q3,
        float* keys, int keyStride, int dimensions, int length, float scale, float* output)
    {
        if (!IsSupported) throw new PlatformNotSupportedException("AVX512F is required for Gemma attention.");
        int key = 0;
        for (; key + 32 <= length; key += 32)
        {
            var c00 = Vector512<float>.Zero; var c01 = Vector512<float>.Zero;
            var c10 = Vector512<float>.Zero; var c11 = Vector512<float>.Zero;
            var c20 = Vector512<float>.Zero; var c21 = Vector512<float>.Zero;
            var c30 = Vector512<float>.Zero; var c31 = Vector512<float>.Zero;
            for (int dimension = 0; dimension < dimensions; ++dimension)
            {
                var k0 = Avx512F.LoadVector512(keys + (long)dimension * keyStride + key);
                var k1 = Avx512F.LoadVector512(keys + (long)dimension * keyStride + key + 16);
                var a = Vector512.Create(q0[dimension]);
                c00 = Avx512F.FusedMultiplyAdd(k0, a, c00); c01 = Avx512F.FusedMultiplyAdd(k1, a, c01);
                a = Vector512.Create(q1[dimension]);
                c10 = Avx512F.FusedMultiplyAdd(k0, a, c10); c11 = Avx512F.FusedMultiplyAdd(k1, a, c11);
                a = Vector512.Create(q2[dimension]);
                c20 = Avx512F.FusedMultiplyAdd(k0, a, c20); c21 = Avx512F.FusedMultiplyAdd(k1, a, c21);
                a = Vector512.Create(q3[dimension]);
                c30 = Avx512F.FusedMultiplyAdd(k0, a, c30); c31 = Avx512F.FusedMultiplyAdd(k1, a, c31);
            }
            var multiplier = Vector512.Create(scale);
            Avx512F.Store(output + key, Avx512F.Multiply(c00, multiplier));
            Avx512F.Store(output + key + 16, Avx512F.Multiply(c01, multiplier));
            Avx512F.Store(output + length + key, Avx512F.Multiply(c10, multiplier));
            Avx512F.Store(output + length + key + 16, Avx512F.Multiply(c11, multiplier));
            Avx512F.Store(output + 2 * length + key, Avx512F.Multiply(c20, multiplier));
            Avx512F.Store(output + 2 * length + key + 16, Avx512F.Multiply(c21, multiplier));
            Avx512F.Store(output + 3 * length + key, Avx512F.Multiply(c30, multiplier));
            Avx512F.Store(output + 3 * length + key + 16, Avx512F.Multiply(c31, multiplier));
        }
        for (; key < length; key += 16)
            ScoresTail(q0, q1, q2, q3, keys + key, keyStride, dimensions,
                Math.Min(16, length - key), length, scale, output + key);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ScoresTail(float* q0, float* q1, float* q2, float* q3,
        float* keys, int stride, int dimensions, int count, int length, float scale, float* output)
    {
        var c0 = Vector512<float>.Zero; var c1 = Vector512<float>.Zero;
        var c2 = Vector512<float>.Zero; var c3 = Vector512<float>.Zero;
        var lowMask = Mask(Math.Min(8, count)); var highMask = Mask(Math.Max(0, count - 8));
        for (int dimension = 0; dimension < dimensions; ++dimension)
        {
            float* p = keys + (long)dimension * stride;
            // Masked AVX2 loads keep an unaligned SWA tail within its actual
            // allocation; no additional padding contract is imposed on K.
            var k = count == 16 ? Avx512F.LoadVector512(p) : Vector512.Create(
                Avx.MaskLoad(p, lowMask.AsSingle()), count > 8 ? Avx.MaskLoad(p + 8, highMask.AsSingle()) : Vector256<float>.Zero);
            c0 = Avx512F.FusedMultiplyAdd(k, Vector512.Create(q0[dimension]), c0);
            c1 = Avx512F.FusedMultiplyAdd(k, Vector512.Create(q1[dimension]), c1);
            c2 = Avx512F.FusedMultiplyAdd(k, Vector512.Create(q2[dimension]), c2);
            c3 = Avx512F.FusedMultiplyAdd(k, Vector512.Create(q3[dimension]), c3);
        }
        var multiplier = Vector512.Create(scale);
        Store(Avx512F.Multiply(c0, multiplier), output, count);
        Store(Avx512F.Multiply(c1, multiplier), output + length, count);
        Store(Avx512F.Multiply(c2, multiplier), output + 2 * length, count);
        Store(Avx512F.Multiply(c3, multiplier), output + 3 * length, count);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Values(float* probabilities, int length, float* values, int valueStride,
        int dimensions, float* output, int outputStride, int queries)
    {
        if (!IsSupported) throw new PlatformNotSupportedException("AVX512F is required for Gemma attention.");
        if (dimensions % ValueWidth != 0) throw new ArgumentException("Gemma value dimensions must be divisible by 16.");
        int dimension = 0;
        for (; dimension + 32 <= dimensions; dimension += 32)
        {
            var c00 = Vector512<float>.Zero; var c01 = Vector512<float>.Zero;
            var c10 = Vector512<float>.Zero; var c11 = Vector512<float>.Zero;
            var c20 = Vector512<float>.Zero; var c21 = Vector512<float>.Zero;
            var c30 = Vector512<float>.Zero; var c31 = Vector512<float>.Zero;
            for (int key = 0; key < length; ++key)
            {
                var v0 = Avx512F.LoadVector512(values + (long)dimension * valueStride + key * ValueWidth);
                var v1 = Avx512F.LoadVector512(values + (long)(dimension + 16) * valueStride + key * ValueWidth);
                var p = Vector512.Create(probabilities[key]);
                c00 = Avx512F.FusedMultiplyAdd(v0, p, c00); c01 = Avx512F.FusedMultiplyAdd(v1, p, c01);
                p = Vector512.Create(probabilities[length + key]);
                c10 = Avx512F.FusedMultiplyAdd(v0, p, c10); c11 = Avx512F.FusedMultiplyAdd(v1, p, c11);
                p = Vector512.Create(probabilities[2 * length + key]);
                c20 = Avx512F.FusedMultiplyAdd(v0, p, c20); c21 = Avx512F.FusedMultiplyAdd(v1, p, c21);
                p = Vector512.Create(probabilities[3 * length + key]);
                c30 = Avx512F.FusedMultiplyAdd(v0, p, c30); c31 = Avx512F.FusedMultiplyAdd(v1, p, c31);
            }
            if (queries > 0) { Avx512F.Store(output + dimension, c00); Avx512F.Store(output + dimension + 16, c01); }
            if (queries > 1) { Avx512F.Store(output + outputStride + dimension, c10); Avx512F.Store(output + outputStride + dimension + 16, c11); }
            if (queries > 2) { Avx512F.Store(output + 2 * outputStride + dimension, c20); Avx512F.Store(output + 2 * outputStride + dimension + 16, c21); }
            if (queries > 3) { Avx512F.Store(output + 3 * outputStride + dimension, c30); Avx512F.Store(output + 3 * outputStride + dimension + 16, c31); }
        }
        if (dimension < dimensions)
            ValuesTail(probabilities, length, values + (long)dimension * valueStride, dimensions - dimension,
                output + dimension, outputStride, queries);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ValuesTail(float* probabilities, int length, float* values, int count,
        float* output, int outputStride, int queries)
    {
        var c0 = Vector512<float>.Zero; var c1 = Vector512<float>.Zero;
        var c2 = Vector512<float>.Zero; var c3 = Vector512<float>.Zero;
        for (int key = 0; key < length; ++key)
        {
            var v = Avx512F.LoadVector512(values + key * ValueWidth);
            c0 = Avx512F.FusedMultiplyAdd(v, Vector512.Create(probabilities[key]), c0);
            c1 = Avx512F.FusedMultiplyAdd(v, Vector512.Create(probabilities[length + key]), c1);
            c2 = Avx512F.FusedMultiplyAdd(v, Vector512.Create(probabilities[2 * length + key]), c2);
            c3 = Avx512F.FusedMultiplyAdd(v, Vector512.Create(probabilities[3 * length + key]), c3);
        }
        if (queries > 0) Store(c0, output, count);
        if (queries > 1) Store(c1, output + outputStride, count);
        if (queries > 2) Store(c2, output + 2 * outputStride, count);
        if (queries > 3) Store(c3, output + 3 * outputStride, count);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Mask(int count) => Vector256.Create(
        count > 0 ? -1 : 0, count > 1 ? -1 : 0, count > 2 ? -1 : 0, count > 3 ? -1 : 0,
        count > 4 ? -1 : 0, count > 5 ? -1 : 0, count > 6 ? -1 : 0, count > 7 ? -1 : 0);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(Vector512<float> value, float* output, int count)
    {
        if (count == 16) Avx512F.Store(output, value);
        else for (int lane = 0; lane < count; ++lane) output[lane] = value.GetElement(lane);
    }
}
