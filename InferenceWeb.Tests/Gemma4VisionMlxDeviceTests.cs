using TensorSharp;
using TensorSharp.MLX;

namespace InferenceWeb.Tests;

public sealed class Gemma4VisionMlxDeviceTests
{
    [MlxTheory]
    [InlineData(1, 1, 4, 1f)]
    [InlineData(17, 3, 64, 1f)]
    [InlineData(49, 12, 64, 80000f)]
    [InlineData(13, 2, 72, 1f)]
    public void TwoDimensionalRotationMatchesIndependentScalarOracle(int tokens, int heads, int dimensions, float magnitude)
    {
        using var allocator = new MlxAllocator();
        int quarter = dimensions / 4, half = dimensions / 2;
        var values = new float[tokens, heads * dimensions];
        var cosine = new float[tokens, half]; var sine = new float[tokens, half];
        for (int t = 0; t < tokens; ++t)
        {
            for (int d = 0; d < heads * dimensions; ++d)
                values[t, d] = ((t * 7 + d * 11) % 29 - 14) * (magnitude / 16);
            for (int axis = 0; axis < 2; ++axis)
            for (int d = 0; d < quarter; ++d)
            {
                float angle = (float)((axis == 0 ? t % 7 : t / 7) * Math.Pow(100, -2.0 * d / half));
                cosine[t, axis * quarter + d] = MathF.Cos(angle);
                sine[t, axis * quarter + d] = MathF.Sin(angle);
            }
        }
        using var data = Tensor.FromArray(allocator, values);
        using var c = Tensor.FromArray(allocator, cosine); using var s = Tensor.FromArray(allocator, sine);
        Assert.True(MlxGemma4VisionOps.TryRope2D(data, c, s, tokens, heads, dimensions));
        float[] actual = data.GetElementsAsFloat(tokens * heads * dimensions);
        for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < heads; ++h)
        for (int axis = 0; axis < 2; ++axis)
        for (int d = 0; d < quarter; ++d)
        {
            int first = h * dimensions + axis * half + d, second = first + quarter;
            double cos = cosine[t, axis * quarter + d], sin = sine[t, axis * quarter + d];
            double a = values[t, first], b = values[t, second];
            Assert.InRange(Math.Abs(actual[t * heads * dimensions + first] - (a * cos - b * sin)), 0, magnitude * 3e-7);
            Assert.InRange(Math.Abs(actual[t * heads * dimensions + second] - (a * sin + b * cos)), 0, magnitude * 3e-7);
        }
        Assert.False(MlxGemma4VisionOps.TryRope2D(data, c, s, tokens, heads, dimensions - 1));
        using var backing = new Tensor(allocator, DType.Float32, tokens + 1, heads * dimensions);
        using var slice = backing.Narrow(0, 1, tokens);
        Assert.False(MlxGemma4VisionOps.TryRope2D(slice, c, s, tokens, heads, dimensions));
    }

    [MlxTheory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(19)]
    [InlineData(36)]
    [InlineData(108)]
    [InlineData(513)]
    public void Float32VisionAttentionMatchesIndependentDoubleSoftmax(int tokens)
        => VerifyAttention(tokens, false);

    [MlxTheory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(19)]
    [InlineData(36)]
    [InlineData(108)]
    [InlineData(513)]
    public void UnroundedFloat32VisionAttentionMatchesIndependentDoubleSoftmax(int tokens)
        => VerifyAttention(tokens, true);

    private static void VerifyAttention(int tokens, bool unrounded)
    {
        const int heads = 3, dimensions = 64;
        using var allocator = new MlxAllocator();
        var queries = new float[1, tokens, heads, dimensions];
        var keys = new float[1, tokens, heads, dimensions];
        var values = new float[1, tokens, heads, dimensions];
        for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < heads; ++h)
        for (int d = 0; d < dimensions; ++d)
        {
            queries[0, t, h, d] = ((t * 7 + h * 3 + d * 11) % 19 - 9) / (unrounded ? 17f : 8f);
            keys[0, t, h, d] = ((t * 5 + h * 7 + d * 3) % 23 - 11) / (unrounded ? 23f : 8f);
            values[0, t, h, d] = ((t * 3 + h * 11 + d * 7) % 29 - 14) / (unrounded ? 29f : 8f);
        }
        using var q = Tensor.FromArray(allocator, queries); using var k = Tensor.FromArray(allocator, keys);
        using var v = Tensor.FromArray(allocator, values);
        using var output = new Tensor(allocator, DType.Float32, 1, tokens, heads, dimensions);
        Assert.True(MlxGemma4VisionOps.TrySelfAttention(output, q, k, v));
        float[] actual = output.GetElementsAsFloat(tokens * heads * dimensions);
        foreach (int t in new[] { 0, tokens / 2, tokens - 1 })
        for (int h = 0; h < heads; ++h)
        {
            var scores = new double[tokens];
            for (int j = 0; j < tokens; ++j)
            for (int d = 0; d < dimensions; ++d)
                scores[j] += (double)queries[0, t, h, d] * keys[0, j, h, d];
            double max = scores.Max();
            for (int j = 0; j < tokens; ++j) scores[j] = Math.Exp(scores[j] - max);
            double sum = scores.Sum();
            for (int d = 0; d < dimensions; ++d)
            {
                double expected = 0;
                for (int j = 0; j < tokens; ++j) expected += scores[j] * values[0, j, h, d] / sum;
                Assert.InRange(Math.Abs(expected - actual[(t * heads + h) * dimensions + d]), 0, 3e-5);
            }
        }
    }
}
