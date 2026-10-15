using TensorSharp;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class CudaEmbeddingWindowMaskBudgetTests
{
    [CudaFact]
    public void FourLargestMasksStayWithin64MiBAndFifthKeyNeedsCompletedReadback()
    {
        const int tokens = 2048, heads = 2, dimensions = 16;
        using var allocator = new CudaAllocator();
        using var q = Tensor.FromArray(allocator, new float[1, tokens, heads, dimensions]);
        using var k = Tensor.FromArray(allocator, new float[1, tokens, 1, dimensions]);
        var values = new float[1, tokens, 1, dimensions];
        for (int token = 0; token < tokens; token++)for (int d = 0; d < dimensions; d++)
            values[0, token, 0, d] = ((token * 7 + d * 11) % 37 - 18) / 29f;
        using var v = Tensor.FromArray(allocator, values);
        using var y = new Tensor(allocator, DType.Float32, 1, tokens, heads, dimensions);
        using var plan = new CudaEmbeddingAttentionPlan(allocator);
        foreach (int radius in new[] { 0, 1, 512, 1024 })
            Assert.True(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, [tokens], radius));
        float[] actual = y.GetElementsAsFloat(tokens * heads * dimensions);
        Assert.Equal(4, plan.MaskEntries); Assert.Equal(64L * 1024 * 1024, plan.MaskCapacityBytes);
        Assert.Equal(4, plan.MaskUploads);
        // The stream actually completed above, but eviction stays prohibited
        // until the caller explicitly acknowledges that ordinary readback.
        Assert.False(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, [tokens], 1025));
        Assert.Equal(actual, y.GetElementsAsFloat(actual.Length)); Assert.Equal(0, plan.MaskEvictions);
        plan.NotifyReadbackCompleted();
        Assert.True(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, [tokens], 1025));
        actual = y.GetElementsAsFloat(actual.Length); plan.NotifyReadbackCompleted();
        foreach (int token in new[] { 0, 1025, 2047 })
        {
            int first = Math.Max(0, token - 1025), last = Math.Min(tokens - 1, token + 1025);
            for (int d = 0; d < dimensions; d++)
            {
                double sum = 0;for (int j = first; j <= last; j++) sum += values[0, j, 0, d];
                double expected = sum / (last - first + 1);
                for (int h = 0; h < heads; h++)
                    Assert.InRange(Math.Abs(expected - actual[(token * heads + h) * dimensions + d]), 0, 3e-5);
            }
        }
        Assert.Equal(4, plan.MaskEntries); Assert.Equal(64L * 1024 * 1024, plan.MaskCapacityBytes);
        Assert.Equal(5, plan.MaskUploads); Assert.Equal(1, plan.MaskEvictions);
        plan.Dispose(); Assert.Equal(0, plan.MaskCapacityBytes); Assert.Equal(0, plan.MaskEntries);
    }
}
