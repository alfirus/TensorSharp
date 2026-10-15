// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers;
using System.Numerics.Tensors;
using System.Threading;

namespace TensorSharp.Models.Embeddings;

/// <summary>FFN joined [gate,up] rows to contiguous GELU(gate)*up.
/// The gate half is disposable activation storage and is mutated in place;
/// the up half stays unchanged. Arithmetic order matches the existing split
/// pipeline exactly. Buffers/worker lifetime remain with the model owner.</summary>
internal static class ManagedEmbeddingGatedGelu
{
    internal static void Apply(float[] joined, float[] output, int rows, int width, CpuWorkerPool pool,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(joined);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(pool);
        if (rows <= 0 || width <= 0 || joined.Length < checked(rows * width * 2) || output.Length < checked(rows * width))
            throw new ArgumentException("Invalid joined FFN activation shape or storage.");
        if (ReferenceEquals(joined, output))
            throw new ArgumentException("Joined and contiguous FFN output storage must be separate.");
        cancellationToken.ThrowIfCancellationRequested();
        int jobs = Math.Min(rows, Math.Max(1, pool.ThreadCount * 4));
        pool.For(jobs, job =>
        {
            var scratch = ArrayPool<float>.Shared.Rent(width);
            try
            {
                var temporary = scratch.AsSpan(0, width);
                int first = (int)((long)rows * job / jobs), end = (int)((long)rows * (job + 1) / jobs);
                for (int row = first; row < end; ++row)
                {
                    var gate = joined.AsSpan(row * width * 2, width);
                    var up = joined.AsSpan(row * width * 2 + width, width);
                    TensorPrimitives.Multiply(gate, gate, temporary);
                    TensorPrimitives.Multiply(temporary, gate, temporary);
                    TensorPrimitives.MultiplyAdd(temporary, .044715f, gate, temporary);
                    TensorPrimitives.Multiply(temporary, .7978845608028654f, temporary);
                    TensorPrimitives.Tanh(temporary, temporary);
                    TensorPrimitives.Add(temporary, 1f, temporary);
                    TensorPrimitives.Multiply(gate, temporary, gate);
                    TensorPrimitives.Multiply(gate, .5f, gate);
                    TensorPrimitives.Multiply(gate, up, output.AsSpan(row * width, width));
                }
            }
            finally { ArrayPool<float>.Shared.Return(scratch); }
        }, cancellationToken);
    }
}
