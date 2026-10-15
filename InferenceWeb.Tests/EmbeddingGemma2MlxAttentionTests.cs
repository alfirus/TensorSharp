using TensorSharp;
using TensorSharp.MLX;
using TensorSharp.Models.Embeddings;

namespace InferenceWeb.Tests;

public sealed class EmbeddingGemma2MlxAttentionTests
{
    [Fact]
    public void HalfAttentionProofRejectsFiniteComponentsWhoseDotProductCanOverflow()
    {
        float[] ordinary = Enumerable.Repeat(1f, 512).ToArray();
        Assert.True(TensorEmbeddingGemma2Encoder.CanUseHalfAttention(ordinary, ordinary, 512));
        float[] unsafeDot = Enumerable.Repeat(16f, 512).ToArray();
        Assert.True(16 * MathF.Sqrt(512) < 65504); // Every component fits in F16.
        Assert.True(512 * 16 * 16 > 65504); // An aligned QK dot product does not.
        Assert.False(TensorEmbeddingGemma2Encoder.CanUseHalfAttention(unsafeDot, unsafeDot, 512));
        ordinary[0] = float.NaN;
        Assert.False(TensorEmbeddingGemma2Encoder.CanUseHalfAttention(ordinary, unsafeDot, 512));
        Assert.False(TensorEmbeddingGemma2Encoder.CanUseHalfAttention(new float[256], new float[512], 512));
    }

    [MlxTheory]
    [InlineData(256, 514, 512, 4, 2, false)]
    [InlineData(256, 1027, 512, 4, 2, false)]
    [InlineData(512, 513, 0, 4, 1, false)]
    [InlineData(256, 514, 512, 4, 2, true)]
    [InlineData(256, 1027, 512, 4, 2, true)]
    [InlineData(512, 513, 0, 4, 1, true)]
    public void ChunkedAttentionMatchesIndependentDoubleOracleAtWindowAndChunkEdges(int dimensions,
        int tokens, int window, int heads, int kvHeads, bool boundedHalfOperands)
    {
        const int batches = 2;
        using var allocator = new MlxAllocator();
        var queries = new float[batches, tokens, heads, dimensions];
        var keys = new float[batches, tokens, kvHeads, dimensions];
        var values = new float[batches, tokens, kvHeads, dimensions];
        for (int b = 0; b < batches; ++b)
        for (int t = 0; t < tokens; ++t)
        for (int d = 0; d < dimensions; ++d)
        {
            for (int h = 0; h < heads; ++h)
                queries[b, t, h, d] = ((b * 13 + t * 3 + h * 11 + d * 5) % 31 - 15) / 32f;
            for (int h = 0; h < kvHeads; ++h)
            {
                keys[b, t, h, d] = ((b * 7 + t * 5 + h * 13 + d * 3) % 29 - 14) / 32f;
                values[b, t, h, d] = ((b * 19 + t * 11 + h * 3 + d * 7) % 37 - 18) / 32f;
            }
        }
        using var q = Tensor.FromArray(allocator, queries);
        using var k = Tensor.FromArray(allocator, keys);
        using var v = Tensor.FromArray(allocator, values);
        using var output = new Tensor(allocator, DType.Float32, batches, tokens, heads, dimensions);
        using var plan = new MlxEmbeddingAttentionPlan();
        float scale = 1 / MathF.Sqrt(dimensions);
        Assert.True(MlxEmbeddingOps.TryAttention(output, q, k, v, window, scale, plan, useFloat16Operands: boundedHalfOperands));
        float[] actual = output.GetElementsAsFloat(batches * tokens * heads * dimensions);
        Assert.All(actual, value => Assert.True(float.IsFinite(value)));
        int[] positions = new[] { 0, 1, 255, 256, 511, 512, 513, 1023, 1024, 1025, tokens - 1 }.Where(t => t < tokens).Distinct().ToArray();
        for (int b = 0; b < batches; ++b)
        foreach (int position in positions)
        for (int h = 0; h < heads; ++h)
        {
            int kh = h / (heads / kvHeads);
            int first = window == 0 ? 0 : Math.Max(0, position - window);
            int last = window == 0 ? tokens : Math.Min(tokens, position + window + 1);
            var scores = new double[last - first];
            for (int t = first; t < last; ++t)
            {
                double dot = 0;
                for (int d = 0; d < dimensions; ++d)
                    dot += (double)queries[b, position, h, d] * keys[b, t, kh, d];
                scores[t - first] = dot * scale;
            }
            double maximum = scores.Max(), sum = 0;
            for (int i = 0; i < scores.Length; ++i)
            {
                scores[i] = Math.Exp(scores[i] - maximum);
                sum += scores[i];
            }
            for (int d = 0; d < dimensions; ++d)
            {
                double expected = 0;
                for (int t = first; t < last; ++t)
                    expected += scores[t - first] / sum * values[b, t, kh, d];
                float value = actual[((b * tokens + position) * heads + h) * dimensions + d];
                Assert.InRange(Math.Abs(value - expected), 0, 3e-5);
            }
        }
    }

    [MlxTheory]
    [InlineData(256, 19)]
    [InlineData(512, 17)]
    public void BoundedMemoryLongAttentionRejectsShortShapesForTheGeneralAttentionPath(int dimensions, int tokens)
    {
        using var allocator = new MlxAllocator();
        using var q = new Tensor(allocator, DType.Float32, 1, tokens, 4, dimensions);
        using var k = new Tensor(allocator, DType.Float32, 1, tokens, 1, dimensions);
        using var v = new Tensor(allocator, DType.Float32, 1, tokens, 1, dimensions);
        using var result = new Tensor(allocator, DType.Float32, 1, tokens, 4, dimensions);
        Assert.False(MlxEmbeddingOps.TryAttention(result, q, k, v, dimensions == 256 ? 512 : 0, 1 / MathF.Sqrt(dimensions)));
    }

    [MlxTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void PackedAttentionKeepsShortAndLongSequenceMeansAndGroupedHeadsIndependent(bool boundedHalfOperands)
    {
        const int tokens = 530, heads = 4, kvHeads = 2, dimensions = 256;
        int[] lengths = [17, 513];
        using var allocator = new MlxAllocator();
        using var q = Tensor.FromArray(allocator, new float[1, tokens, heads, dimensions]);
        using var k = Tensor.FromArray(allocator, new float[1, tokens, kvHeads, dimensions]);
        var values = new float[1, tokens, kvHeads, dimensions];
        int offset = 0;
        var expected = new double[2, kvHeads];
        for (int sequence = 0; sequence < lengths.Length; ++sequence)
        {
            for (int t = 0; t < lengths[sequence]; ++t)
            for (int h = 0; h < kvHeads; ++h)
            {
                float value = (sequence == h ? 1 : -1) / 32f + (t % 5 - 2) / 1024f;
                expected[sequence, h] += (double)value / lengths[sequence];
                for (int d = 0; d < dimensions; ++d) values[0, offset + t, h, d] = value;
            }
            offset += lengths[sequence];
        }
        using var v = Tensor.FromArray(allocator, values);
        using var result = new Tensor(allocator, DType.Float32, 1, tokens, heads, dimensions);
        Assert.True(MlxEmbeddingOps.TryAttention(result, q, k, v, 512, 1 / MathF.Sqrt(dimensions),
            useFloat16Operands: boundedHalfOperands, sequenceLengths: lengths));
        float[] actual = result.GetElementsAsFloat(tokens * heads * dimensions);
        offset = 0;
        for (int sequence = 0; sequence < lengths.Length; ++sequence)
        {
            foreach (int t in new[] { 0, lengths[sequence] - 1 })
            for (int h = 0; h < heads; ++h)
            for (int d = 0; d < dimensions; ++d)
                Assert.InRange(Math.Abs(actual[((offset + t) * heads + h) * dimensions + d] - expected[sequence, h / 2]), 0, 3e-5);
            offset += lengths[sequence];
        }
        Assert.False(MlxEmbeddingOps.TryAttention(result, q, k, v, 512, 1, sequenceLengths: [17, 512]));
        Assert.False(MlxEmbeddingOps.TryAttention(result, q, k, v, 512, 1, sequenceLengths: [0, 530]));
    }

    [MlxTheory]
    [InlineData(256, 2, false)]
    [InlineData(256, 2, true)]
    [InlineData(512, 1, false)]
    [InlineData(512, 1, true)]
    public void PaddedRaggedAttentionMatchesIndependentDoubleOracleAndExcludesPadding(int dimensions, int kvHeads, bool boundedHalfOperands)
    {
        const int heads = 4, tokens = 83;
        int[] lengths = [1, 17, 65];
        var queries = new float[1, tokens, heads, dimensions];
        var keys = new float[1, tokens, kvHeads, dimensions];
        var values = new float[1, tokens, kvHeads, dimensions];
        for (int t = 0; t < tokens; ++t)
        for (int d = 0; d < dimensions; ++d)
        {
            for (int h = 0; h < heads; ++h) queries[0, t, h, d] = ((t * 3 + h * 5 + d * 7) % 17 - 8) / 512f;
            for (int h = 0; h < kvHeads; ++h)
            {
                keys[0, t, h, d] = ((t * 11 + h * 7 + d * 3) % 19 - 9) / 512f;
                values[0, t, h, d] = ((t * 13 + h * 3 + d * 5) % 23 - 11) / 256f + (t < 18 ? 1 : -1) / 64f;
            }
        }
        using var allocator = new MlxAllocator();
        using var q = Tensor.FromArray(allocator, queries);
        using var k = Tensor.FromArray(allocator, keys);
        using var v = Tensor.FromArray(allocator, values);
        using var result = new Tensor(allocator, DType.Float32, 1, tokens, heads, dimensions);
        using var plan = new MlxRaggedEmbeddingAttentionPlan();
        float scale = 1 / MathF.Sqrt(dimensions);
        Assert.True(MlxRaggedEmbeddingOps.TryAttention(result, q, k, v, lengths, scale, plan, boundedHalfOperands));
        float[] actual = result.GetElementsAsFloat(tokens * heads * dimensions);
        Assert.All(actual, value => Assert.True(float.IsFinite(value)));
        int offset = 0;
        foreach (int length in lengths)
        {
            foreach (int query in new[] { 0, length / 2, length - 1 }.Distinct())
            for (int h = 0; h < heads; ++h)
            {
                int kh = h / (heads / kvHeads);
                var scores = new double[length];
                for (int t = 0; t < length; ++t)
                {
                    double dot = 0;
                    for (int d = 0; d < dimensions; ++d) dot += (double)queries[0, offset + query, h, d] * keys[0, offset + t, kh, d];
                    scores[t] = dot * scale;
                }
                double maximum = scores.Max(), sum = 0;
                for (int t = 0; t < length; ++t) { scores[t] = Math.Exp(scores[t] - maximum); sum += scores[t]; }
                for (int d = 0; d < dimensions; ++d)
                {
                    double expected = 0;
                    for (int t = 0; t < length; ++t) expected += scores[t] / sum * values[0, offset + t, kh, d];
                    Assert.InRange(Math.Abs(actual[((offset + query) * heads + h) * dimensions + d] - expected), 0, 3e-5);
                }
            }
            offset += length;
        }
        Assert.False(MlxRaggedEmbeddingOps.TryAttention(result, q, k, v, [1, 17, 64], scale, plan));
        Assert.False(MlxRaggedEmbeddingOps.TryAttention(result, q, k, v, [0, 18, 65], scale, plan));
    }
}
