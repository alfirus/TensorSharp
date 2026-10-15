using TensorSharp;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class Gemma4VisionCudaDeviceTests
{
    [CudaTheory]
    [InlineData(1, 1, 4, 1f)]
    [InlineData(17, 3, 64, 1f)]
    [InlineData(49, 12, 64, 80000f)]
    [InlineData(13, 2, 72, 1f)]
    public void AxialRotationMatchesIndependentDoublePairsAndPreservesSurroundingRows(int tokens, int heads, int dimensions, float magnitude)
    {
        using var allocator = new CudaAllocator();
        int quarter = dimensions / 4, half = dimensions / 2;
        var values = new float[tokens + 2, heads * dimensions];
        var cosine = new float[tokens + 2, half]; var sine = new float[tokens + 2, half];
        for (int t = 0; t < tokens + 2; t++)
        {
            for (int d = 0; d < heads * dimensions; d++)
                values[t, d] = ((t * 7 + d * 11) % 29 - 14) * (magnitude / 16);
            for (int axis = 0; axis < 2; axis++)
            for (int d = 0; d < quarter; d++)
            {
                float angle = (float)((axis == 0 ? t % 7 : t / 7) * Math.Pow(100, -2.0 * d / half));
                cosine[t, axis * quarter + d] = MathF.Cos(angle); sine[t, axis * quarter + d] = MathF.Sin(angle);
            }
        }
        using var backing = Tensor.FromArray(allocator, values);
        using var cBacking = Tensor.FromArray(allocator, cosine); using var sBacking = Tensor.FromArray(allocator, sine);
        using var data = backing.Narrow(0, 1, tokens); using var c = cBacking.Narrow(0, 1, tokens); using var s = sBacking.Narrow(0, 1, tokens);
        Assert.True(CudaGemma4VisionOps.TryRope2D(data, c, s, tokens, heads, dimensions));
        float[] actual = backing.GetElementsAsFloat((tokens + 2) * heads * dimensions);
        for (int t = 0; t < tokens; t++)
        for (int h = 0; h < heads; h++)
        for (int axis = 0; axis < 2; axis++)
        for (int d = 0; d < quarter; d++)
        {
            int first = h * dimensions + axis * half + d, second = first + quarter;
            double cv = cosine[t + 1, axis * quarter + d], sv = sine[t + 1, axis * quarter + d];
            double a = values[t + 1, first], b = values[t + 1, second];
            Assert.InRange(Math.Abs(actual[(t + 1) * heads * dimensions + first] - (a * cv - b * sv)), 0, magnitude * 3e-7);
            Assert.InRange(Math.Abs(actual[(t + 1) * heads * dimensions + second] - (a * sv + b * cv)), 0, magnitude * 3e-7);
        }
        for (int d = 0; d < heads * dimensions; d++)
        { Assert.Equal(values[0, d], actual[d]); Assert.Equal(values[tokens + 1, d], actual[(tokens + 1) * heads * dimensions + d]); }
        Assert.False(CudaGemma4VisionOps.TryRope2D(data, c, s, tokens, heads, dimensions - 1));
        using var strided = backing.Narrow(1, 0, heads * dimensions - 1);
        Assert.False(CudaGemma4VisionOps.TryRope2D(strided, c, s, tokens + 2, heads, dimensions));
    }

    [CudaTheory]
    [InlineData(1, 1, 1, 32)]
    [InlineData(17, 97, 31, 32)]
    [InlineData(33, 768, 32, 80000)]
    public void ProjectionPaddingCopiesExactRowsAndZerosNewColumns(int tokens, int width, int extra, int magnitude)
    {
        using var allocator = new CudaAllocator();
        var source = new float[tokens + 2, width]; var destination = new float[tokens + 2, width + extra];
        for (int t = 0; t < tokens + 2; t++)
        {
            for (int c = 0; c < width; c++) source[t, c] = ((t * 11 + c * 7) % 31 - 15) * (magnitude / 16f);
            for (int c = 0; c < width + extra; c++) destination[t, c] = 37f;
        }
        using var sourceBacking = Tensor.FromArray(allocator, source); using var targetBacking = Tensor.FromArray(allocator, destination);
        using var input = sourceBacking.Narrow(0, 1, tokens); using var target = targetBacking.Narrow(0, 1, tokens);
        Assert.True(CudaGemma4VisionOps.TryPadProjectionInput(target, input, extra));
        float[] actual = targetBacking.GetElementsAsFloat((tokens + 2) * (width + extra));
        for (int t = 0; t < tokens; t++)for (int c = 0; c < width + extra; c++)
            Assert.Equal(c < width ? source[t + 1, c] : 0, actual[(t + 1) * (width + extra) + c]);
        for (int c = 0; c < width + extra; c++)
        { Assert.Equal(37f, actual[c]); Assert.Equal(37f, actual[(tokens + 1) * (width + extra) + c]); }
        Assert.False(CudaGemma4VisionOps.TryPadProjectionInput(target, input, extra + 1));
        using var alias = target.Narrow(1, 0, width);
        Assert.False(CudaGemma4VisionOps.TryPadProjectionInput(target, alias, extra));
    }

    [CudaTheory]
    [InlineData(1, 1f)]
    [InlineData(7, 1f)]
    [InlineData(19, 1f)]
    [InlineData(36, 80000f)]
    [InlineData(108, 1f)]
    [InlineData(513, 1f)]
    public void ScaleOneFloat32AttentionMatchesIndependentDoubleSoftmaxAcrossBatches(int tokens, float valueMagnitude)
    {
        const int batches = 2, heads = 3, dimensions = 64;
        using var allocator = new CudaAllocator();
        var queries = new float[batches, tokens, heads, dimensions]; var keys = new float[batches, tokens, heads, dimensions];
        var values = new float[batches, tokens, heads, dimensions]; var destination = new float[batches + 2, tokens, heads, dimensions];
        for (int b = 0; b < batches; b++)for (int t = 0; t < tokens; t++)for (int h = 0; h < heads; h++)for (int d = 0; d < dimensions; d++)
        {
            queries[b, t, h, d] = ((b * 3 + t * 7 + h * 3 + d * 11) % 19 - 9) / 8f;
            keys[b, t, h, d] = ((b * 7 + t * 5 + h * 7 + d * 3) % 23 - 11) / 8f;
            values[b, t, h, d] = valueMagnitude * ((b * 11 + t * 3 + h * 11 + d * 7) % 29 - 14) / 8f;
        }
        for (int b = 0; b < batches + 2; b++)for (int t = 0; t < tokens; t++)for (int h = 0; h < heads; h++)for (int d = 0; d < dimensions; d++)
            destination[b, t, h, d] = 37f;
        using var q = Tensor.FromArray(allocator, queries); using var k = Tensor.FromArray(allocator, keys);
        using var v = Tensor.FromArray(allocator, values); using var targetBacking = Tensor.FromArray(allocator, destination);
        using var output = targetBacking.Narrow(0, 1, batches);
        Assert.True(CudaGemma4VisionOps.TrySelfAttention(output, q, k, v));
        float[] actual = targetBacking.GetElementsAsFloat((batches + 2) * tokens * heads * dimensions);
        foreach (int t in new[] { 0, tokens / 2, tokens - 1 })for (int b = 0; b < batches; b++)for (int h = 0; h < heads; h++)
        {
            var scores = new double[tokens];
            for (int j = 0; j < tokens; j++)for (int d = 0; d < dimensions; d++)
                scores[j] += (double)queries[b, t, h, d] * keys[b, j, h, d];
            double max = scores.Max();for (int j = 0; j < tokens; j++) scores[j] = Math.Exp(scores[j] - max);double sum = scores.Sum();
            for (int d = 0; d < dimensions; d++)
            {
                double expected = 0;for (int j = 0; j < tokens; j++) expected += scores[j] * values[b, j, h, d] / sum;
                Assert.InRange(Math.Abs(expected - actual[(((b + 1) * tokens + t) * heads + h) * dimensions + d]), 0, valueMagnitude * 3e-5);
            }
        }
        for (int i = 0; i < tokens * heads * dimensions; i++)
        { Assert.Equal(37f, actual[i]); Assert.Equal(37f, actual[(batches + 1) * tokens * heads * dimensions + i]); }
        Assert.False(CudaGemma4VisionOps.TrySelfAttention(q, q, k, v));
        using var narrow = q.Narrow(3, 0, 63);
        Assert.False(CudaGemma4VisionOps.TrySelfAttention(output, narrow, k, v));
    }
}
