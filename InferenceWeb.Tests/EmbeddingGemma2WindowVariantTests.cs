// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Embeddings;
using TensorSharp.Runtime;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class EmbeddingGemma2WindowVariantTests(ITestOutputHelper output)
{
    private const string ModelVariable = "TENSORSHARP_EMBEDDINGGEMMA2_MODEL";

    [EmbeddingGemma2MlxFact]
    public void MlxOddWindowsAndPublishedWindowMatchIndependentManagedAttention()
    {
        MlxBackend.Register();
        using var allocator = new MlxAllocator();
        VerifyWindows(allocator);
    }

    [CudaFact(ModelVariable)]
    public void CudaOddWindowsAndPublishedWindowMatchIndependentManagedAttention()
    {
        using var allocator = new CudaAllocator();
        VerifyWindows(allocator);
    }

    private void VerifyWindows(IAllocator allocator)
    {
        using var file = new GgufFile(Environment.GetEnvironmentVariable(ModelVariable)!);
        int[] seed = EmbeddingTokenizer.Create(file).Encode("title: add.py | text: def add(a, b): return a + b").ToArray();
        int[] lengths = [2, 7, 9];
        int[] tokens = lengths.SelectMany((length, sequence) => Enumerable.Range(0, length)
            .Select(position => seed[(position * 3 + sequence * 5) % seed.Length])).ToArray();
        int dimensions = checked((int)file.GetUint32("gemma-embedding2.embedding_length_out"));
        // Width 1 is self-only; widths 3 and 9 require masks even though each
        // packed sequence is much shorter than the ragged kernel's 512 cap.
        // 1024 retains coverage of the released model's radius-512 fast path.
        foreach (uint width in new uint[] { 1, 3, 9, 1024 })
        {
            file.Metadata["gemma-embedding2.attention.sliding_window"] = width;
            using var managed = new ManagedEmbeddingGemma2Encoder(file, 2);
            using var direct = new TensorEmbeddingGemma2Encoder(file, allocator);
            var expected = new float[checked(lengths.Length * dimensions)];
            var actual = new float[expected.Length];
            managed.Encode(tokens, lengths, expected, CancellationToken.None);
            direct.Encode(tokens, lengths, actual, CancellationToken.None);
            Assert.All(actual, value => Assert.True(float.IsFinite(value)));
            int offset = 0;
            for (int sequence = 0; sequence < lengths.Length; ++sequence)
            {
                float[] expectedRow = expected.AsSpan(sequence * dimensions, dimensions).ToArray();
                float[] actualRow = actual.AsSpan(sequence * dimensions, dimensions).ToArray();
                var single = new float[dimensions];
                direct.Encode(tokens.AsSpan(offset, lengths[sequence]).ToArray(), [lengths[sequence]], single, CancellationToken.None);
                offset += lengths[sequence];
                double referenceCosine = Cosine(expectedRow, actualRow), batchCosine = Cosine(single, actualRow);
                output.WriteLine($"Full window {width}, sequence {sequence}: managed/direct cosine {referenceCosine:R}, batch/single cosine {batchCosine:R}.");
                // Cross-backend arithmetic uses the existing independent model
                // quality gate. Batch isolation keeps the tighter consistency gate.
                Assert.True(referenceCosine > .9999, $"Full window {width}, sequence {sequence}: managed/direct cosine {referenceCosine:R}.");
                Assert.True(batchCosine > .99999, $"Full window {width}, sequence {sequence}: batch/single cosine {batchCosine:R}.");
            }
        }
    }

    private static double Cosine(float[] left, float[] right)
        => left.Zip(right, (a, b) => (double)a * b).Sum()
            / Math.Sqrt(left.Sum(value => (double)value * value) * right.Sum(value => (double)value * value));
}
