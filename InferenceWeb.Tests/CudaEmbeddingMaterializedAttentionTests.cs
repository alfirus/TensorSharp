using TensorSharp;
using TensorSharp.Cuda;
using System.Runtime.InteropServices;

namespace InferenceWeb.Tests;

public sealed class CudaEmbeddingMaterializedAttentionTests
{
    [CudaTheory]
    [InlineData(1, 1, 256, 4, 2, 1, 1f)]
    [InlineData(2, 13, 256, 4, 2, 2, 1f)]
    [InlineData(2, 29, 512, 4, 1, 1, 1f)]
    [InlineData(1, 129, 512, 4, 1, 2, 1f)]
    [InlineData(1, 513, 256, 4, 2, 2, 1f)]
    [InlineData(1, 513, 512, 4, 1, 2, 1f)]
    [InlineData(1, 19, 256, 4, 2, 2, 80000f)]
    public void EqualBatchesAndInterleavedGqaMatchUnroundedDoubleOracleAndPreserveGuardBatches(
        int batches, int tokens, int dimensions, int heads, int kvHeads, int tile, float magnitude)
    {
        using var allocator = new CudaAllocator();
        var qValues = Values(batches + 2, tokens, heads, dimensions, 17, 1);
        var kValues = Values(batches + 2, tokens, kvHeads, dimensions, 23, 1);
        var vValues = Values(batches + 2, tokens, kvHeads, dimensions, 29, magnitude);
        var destination = new float[batches + 2, tokens, heads, dimensions];
        Fill(destination, 37f);
        using var qb = Tensor.FromArray(allocator, qValues); using var kb = Tensor.FromArray(allocator, kValues);
        using var vb = Tensor.FromArray(allocator, vValues); using var yb = Tensor.FromArray(allocator, destination);
        using var q = qb.Narrow(0, 1, batches); using var k = kb.Narrow(0, 1, batches);
        using var v = vb.Narrow(0, 1, batches); using var y = yb.Narrow(0, 1, batches);
        using var plan = new CudaEmbeddingAttentionPlan(allocator, tile);
        int[] lengths = Enumerable.Repeat(tokens, batches).ToArray();
        const float scale = .7f;
        Assert.True(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, plan, lengths, scale));
        float[] actual = yb.GetElementsAsFloat(destination.Length);
        for (int b = 0; b < batches; b++)
        foreach (int token in new[] { 0, tokens / 2, tokens - 1 })
            CheckRow(actual, qValues, kValues, vValues, b + 1, token, 0, tokens, heads, kvHeads, dimensions, scale, magnitude);
        int batchElements = tokens * heads * dimensions;
        for (int i = 0; i < batchElements; i++)
        { Assert.Equal(37f, actual[i]); Assert.Equal(37f, actual[(batches + 1) * batchElements + i]); }
        Assert.Equal(69, CudaEmbeddingAttentionPlan.ComputeType);
        Assert.Equal(1, plan.SuccessfulCalls); Assert.Equal(batches, plan.SuccessfulSegments);
        Assert.Equal((long)batches * tokens * heads, plan.SuccessfulQueryRows);
        Assert.InRange(plan.ScoreCapacityBytes, 4, 512L * 1024 * 1024);
        long capacity = plan.ScoreCapacityBytes;
        Assert.True(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, plan, lengths, scale));
        Assert.Equal(actual, yb.GetElementsAsFloat(destination.Length));
        Assert.Equal(capacity, plan.ScoreCapacityBytes);
    }

    [CudaTheory]
    [InlineData(256, 2)]
    [InlineData(512, 1)]
    public void PackedUnequalLengthsRemainIsolatedAndReuseOnlyTheLargestSegmentWorkspace(int dimensions, int kvHeads)
    {
        const int heads = 4;
        int[] lengths = [1, 7, 13, 29]; int tokens = lengths.Sum();
        using var allocator = new CudaAllocator();
        var queries = Values(1, tokens + 2, heads, dimensions, 17, 1);
        var keys = Values(1, tokens + 2, kvHeads, dimensions, 23, 1);
        var values = Values(1, tokens + 2, kvHeads, dimensions, 29, 1);
        var destination = new float[1, tokens + 2, heads, dimensions]; Fill(destination, 37f);
        using var qb = Tensor.FromArray(allocator, queries); using var kb = Tensor.FromArray(allocator, keys);
        using var vb = Tensor.FromArray(allocator, values); using var yb = Tensor.FromArray(allocator, destination);
        using var q = qb.Narrow(1, 1, tokens); using var k = kb.Narrow(1, 1, tokens);
        using var v = vb.Narrow(1, 1, tokens); using var y = yb.Narrow(1, 1, tokens);
        using var plan = new CudaEmbeddingAttentionPlan(allocator);
        Assert.True(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, plan, lengths));
        float[] actual = yb.GetElementsAsFloat(destination.Length);
        int start = 1;
        foreach (int length in lengths)
        {
            foreach (int token in new[] { start, start + length / 2, start + length - 1 })
                CheckRow(actual, queries, keys, values, 0, token, start, length, heads, kvHeads, dimensions, 1, 1);
            start += length;
        }
        for (int i = 0; i < heads * dimensions; i++)
        { Assert.Equal(37f, actual[i]); Assert.Equal(37f, actual[(tokens + 1) * heads * dimensions + i]); }
        Assert.Equal(2L * 29 * 29 * sizeof(float), plan.ScoreCapacityBytes);
        Assert.Equal(4, plan.SuccessfulSegments);
        Assert.Equal((long)tokens * heads, plan.SuccessfulQueryRows);
    }

    [CudaFact]
    public void BudgetLayoutAliasAndAllocatorRejectionsLeaveTargetsUntouched()
    {
        using var allocator = new CudaAllocator(); using var peer = new CudaAllocator();
        using var q = Tensor.FromArray(allocator, Values(1, 13, 4, 256, 17, 1));
        using var k = Tensor.FromArray(allocator, Values(1, 13, 2, 256, 23, 1));
        using var v = Tensor.FromArray(allocator, Values(1, 13, 2, 256, 29, 1));
        var fill = new float[1, 13, 4, 256]; Fill(fill, 37f);
        using var y = Tensor.FromArray(allocator, fill); using var peerY = Tensor.FromArray(peer, fill);
        using var tiny = new CudaEmbeddingAttentionPlan(allocator, scoreBudgetBytes: 64);
        Assert.False(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, tiny, [13]));
        Assert.Equal(0, tiny.ScoreCapacityBytes); Assert.Equal(0, tiny.SuccessfulCalls);
        using var plan = new CudaEmbeddingAttentionPlan(allocator);
        Assert.False(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, plan, [12]));
        Assert.False(CudaEmbeddingAttentionOps.TryGlobalAttention(peerY, q, k, v, plan, [13]));
        Assert.False(CudaEmbeddingAttentionOps.TryGlobalAttention(q, q, k, v, plan, [13]));
        using var narrow = q.Narrow(3, 0, 255);
        Assert.False(CudaEmbeddingAttentionOps.TryGlobalAttention(y, narrow, k, v, plan, [13]));
        Assert.Equal(fill.Cast<float>().ToArray(), y.GetElementsAsFloat(fill.Length));
        Assert.Equal(fill.Cast<float>().ToArray(), peerY.GetElementsAsFloat(fill.Length));
    }

    [CudaFact]
    public void DisposingOnePlanKeepsPeerWorkspaceAndTensorsLive()
    {
        using var allocator = new CudaAllocator();
        using var q = Tensor.FromArray(allocator, Values(1, 13, 4, 256, 17, 1));
        using var k = Tensor.FromArray(allocator, Values(1, 13, 2, 256, 23, 1));
        using var v = Tensor.FromArray(allocator, Values(1, 13, 2, 256, 29, 1));
        using var y = new Tensor(allocator, DType.Float32, 1, 13, 4, 256);
        using var first = new CudaEmbeddingAttentionPlan(allocator); using var peer = new CudaEmbeddingAttentionPlan(allocator, 1);
        Assert.True(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, first, [13]));
        float[] expected = y.GetElementsAsFloat(13 * 4 * 256);
        Assert.True(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, peer, [13]));
        float[] peerExpected = y.GetElementsAsFloat(13 * 4 * 256);
        for (int i = 0; i < expected.Length; i++) Assert.InRange(Math.Abs(expected[i] - peerExpected[i]), 0, 3e-5);
        first.Dispose(); first.Dispose(); Assert.Equal(0, first.ScoreCapacityBytes);
        Assert.Throws<ObjectDisposedException>(() => CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, first, [13]));
        Assert.True(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, peer, [13]));
        Assert.Equal(peerExpected, y.GetElementsAsFloat(13 * 4 * 256));
        Assert.True(peer.ScoreCapacityBytes > 0);
    }

    [CudaTheory]
    [InlineData(1, 511, 256, 4, 2)]
    [InlineData(1, 512, 512, 4, 1)]
    [InlineData(1, 513, 256, 4, 2)]
    [InlineData(1, 514, 512, 4, 1)]
    [InlineData(2, 1025, 256, 4, 2)]
    [InlineData(1, 2048, 512, 4, 1)]
    public void SymmetricWindowMatchesUnroundedDoubleOracleAtBoundariesAndPreservesBatchGuards(
        int batches, int tokens, int dimensions, int heads, int kvHeads)
    {
        const int radius = 512;
        using var allocator = new CudaAllocator();
        var queries = Values(batches + 2, tokens, heads, dimensions, 17, 1);
        var keys = Values(batches + 2, tokens, kvHeads, dimensions, 23, 1);
        var values = Values(batches + 2, tokens, kvHeads, dimensions, 29, 1);
        var destination = new float[batches + 2, tokens, heads, dimensions]; Fill(destination, 37f);
        using var qb = Tensor.FromArray(allocator, queries); using var kb = Tensor.FromArray(allocator, keys);
        using var vb = Tensor.FromArray(allocator, values); using var yb = Tensor.FromArray(allocator, destination);
        using var q = qb.Narrow(0, 1, batches); using var k = kb.Narrow(0, 1, batches);
        using var v = vb.Narrow(0, 1, batches); using var y = yb.Narrow(0, 1, batches);
        using var plan = new CudaEmbeddingAttentionPlan(allocator);
        int[] lengths = Enumerable.Repeat(tokens, batches).ToArray(); const float scale = .7f;
        Assert.True(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, lengths, radius, scale));
        float[] actual = yb.GetElementsAsFloat(destination.Length); plan.NotifyReadbackCompleted();
        for (int b = 0; b < batches; b++)
        foreach (int token in new[] { 0, Math.Min(512, tokens - 1), tokens - 1 })
            CheckRow(actual, queries, keys, values, b + 1, token, 0, tokens, heads, kvHeads, dimensions, scale, 1, radius);
        int batchElements = tokens * heads * dimensions;
        for (int i = 0; i < batchElements; i++)
        { Assert.Equal(37f, actual[i]); Assert.Equal(37f, actual[(batches + 1) * batchElements + i]); }
        Assert.Equal(1, plan.SuccessfulWindowCalls);
        Assert.Equal(tokens <= radius + 1 ? 0 : 1, plan.MaskEntries);
        long bytes = tokens <= radius + 1 ? 0 : (long)tokens * tokens * sizeof(float);
        Assert.Equal(bytes, plan.MaskCapacityBytes);
        Assert.InRange(plan.MaskCapacityBytes, 0, CudaEmbeddingAttentionPlan.MaskBudgetBytes);
        long uploads = plan.MaskUploads;
        Assert.True(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, lengths, radius, scale));
        Assert.Equal(actual, yb.GetElementsAsFloat(destination.Length)); plan.NotifyReadbackCompleted();
        Assert.Equal(uploads, plan.MaskUploads);
    }

    [CudaFact]
    public void InclusiveWindowAdmitsDistance512OnBothSidesAndExcludesDistance513()
    {
        const int tokens = 1027, dimensions = 256, heads = 4, kvHeads = 2, radius = 512;
        using var allocator = new CudaAllocator();
        var queries = new float[1, tokens, heads, dimensions];
        var keys = new float[1, tokens, kvHeads, dimensions];
        var values = new float[1, tokens, kvHeads, dimensions];
        foreach (int key in new[] { 0, 1, 1024, 1025, 1026 })
        for (int kv = 0; kv < kvHeads; kv++)for (int d = 0; d < dimensions; d++)
            values[0, key, kv, d] = (key + 1) * .37f + kv * .13f + d * .0017f;
        using var q = Tensor.FromArray(allocator, queries); using var k = Tensor.FromArray(allocator, keys);
        using var v = Tensor.FromArray(allocator, values); using var y = new Tensor(allocator, DType.Float32, 1, tokens, heads, dimensions);
        using var plan = new CudaEmbeddingAttentionPlan(allocator);
        Assert.True(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, [tokens], radius));
        float[] actual = y.GetElementsAsFloat(tokens * heads * dimensions); plan.NotifyReadbackCompleted();
        foreach (int token in new[] { 511, 512, 513, 514 })
            CheckRow(actual, queries, keys, values, 0, token, 0, tokens, heads, kvHeads, dimensions, 1, 1, radius);
    }

    [CudaFact]
    public void PackedWindowSegmentsRemainIsolatedWithTokenOffsetsAndInterleavedGqa()
    {
        const int dimensions = 256, heads = 4, kvHeads = 2, radius = 512;
        int[] lengths = [513, 514, 1025]; int tokens = lengths.Sum();
        using var allocator = new CudaAllocator();
        var queries = Values(1, tokens + 2, heads, dimensions, 17, 1);
        var keys = Values(1, tokens + 2, kvHeads, dimensions, 23, 1);
        var values = Values(1, tokens + 2, kvHeads, dimensions, 29, 1);
        var destination = new float[1, tokens + 2, heads, dimensions]; Fill(destination, 37f);
        using var qb = Tensor.FromArray(allocator, queries); using var kb = Tensor.FromArray(allocator, keys);
        using var vb = Tensor.FromArray(allocator, values); using var yb = Tensor.FromArray(allocator, destination);
        using var q = qb.Narrow(1, 1, tokens); using var k = kb.Narrow(1, 1, tokens);
        using var v = vb.Narrow(1, 1, tokens); using var y = yb.Narrow(1, 1, tokens);
        using var plan = new CudaEmbeddingAttentionPlan(allocator);
        Assert.True(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, lengths, radius));
        float[] actual = yb.GetElementsAsFloat(destination.Length); plan.NotifyReadbackCompleted();
        int start = 1;
        foreach (int length in lengths)
        {
            foreach (int token in new[] { start, start + length / 2, start + length - 1 })
                CheckRow(actual, queries, keys, values, 0, token, start, length, heads, kvHeads, dimensions, 1, 1, radius);
            start += length;
        }
        for (int i = 0; i < heads * dimensions; i++)
        { Assert.Equal(37f, actual[i]); Assert.Equal(37f, actual[(tokens + 1) * heads * dimensions + i]); }
        Assert.Equal(2, plan.MaskEntries); Assert.Equal(2, plan.MaskUploads);
        Assert.Equal(2L * 1025 * 1025 * sizeof(float), plan.ScoreCapacityBytes);
        Assert.Equal((long)tokens * heads, plan.SuccessfulQueryRows);
    }

    [CudaFact]
    public void MaskEvictionWaitsForReadbackAndPeerDisposalDoesNotInvalidateLivePlan()
    {
        const int tokens = 13, dimensions = 256;
        using var allocator = new CudaAllocator();
        using var q = Tensor.FromArray(allocator, Values(1, tokens, 4, dimensions, 17, 1));
        using var k = Tensor.FromArray(allocator, Values(1, tokens, 2, dimensions, 23, 1));
        using var v = Tensor.FromArray(allocator, Values(1, tokens, 2, dimensions, 29, 1));
        using var y = new Tensor(allocator, DType.Float32, 1, tokens, 4, dimensions);
        using var plan = new CudaEmbeddingAttentionPlan(allocator); using var peer = new CudaEmbeddingAttentionPlan(allocator);
        for (int radius = 0; radius < 4; radius++)
            Assert.True(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, [tokens], radius));
        float[] expected = y.GetElementsAsFloat(tokens * 4 * dimensions);
        Assert.Equal(4, plan.MaskEntries); Assert.Equal(4L * tokens * tokens * sizeof(float), plan.MaskCapacityBytes);
        Assert.False(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, [tokens], 4));
        Assert.Equal(expected, y.GetElementsAsFloat(tokens * 4 * dimensions)); Assert.Equal(0, plan.MaskEvictions);
        plan.NotifyReadbackCompleted();
        Assert.True(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, [tokens], 4));
        expected = y.GetElementsAsFloat(tokens * 4 * dimensions); plan.NotifyReadbackCompleted();
        Assert.Equal(4, plan.MaskEntries); Assert.Equal(1, plan.MaskEvictions); Assert.Equal(5, plan.MaskUploads);
        Assert.True(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, peer, [tokens], 4));
        peer.AbortPendingWork(); peer.Dispose(); peer.Dispose();
        Assert.Equal(0, peer.MaskEntries); Assert.Equal(0, peer.MaskCapacityBytes);
        Assert.True(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, [tokens], 4));
        Assert.Equal(expected, y.GetElementsAsFloat(tokens * 4 * dimensions)); plan.NotifyReadbackCompleted();
        Assert.Equal(5, plan.MaskUploads);
    }

    [CudaFact]
    public void UnsupportedWindowLengthsMasksAndAliasesRejectBeforeWritingOutput()
    {
        const int dimensions = 8; int[] lengths = [514, 515, 516, 517, 518]; int tokens = lengths.Sum();
        using var allocator = new CudaAllocator();
        using var q = Tensor.FromArray(allocator, Values(1, tokens, 4, dimensions, 17, 1));
        using var k = Tensor.FromArray(allocator, Values(1, tokens, 2, dimensions, 23, 1));
        using var v = Tensor.FromArray(allocator, Values(1, tokens, 2, dimensions, 29, 1));
        var destination = new float[1, tokens, 4, dimensions]; Fill(destination, 37f);
        using var y = Tensor.FromArray(allocator, destination); using var plan = new CudaEmbeddingAttentionPlan(allocator);
        Assert.False(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, lengths, 512));
        Assert.False(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, [tokens], 512));
        Assert.False(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, lengths, -1));
        Assert.False(CudaEmbeddingAttentionOps.TryWindowAttention(q, q, k, v, plan, lengths, 512));
        Assert.Equal(destination.Cast<float>().ToArray(), y.GetElementsAsFloat(destination.Length));
        Assert.Equal(0, plan.SuccessfulCalls); Assert.Equal(0, plan.MaskEntries); Assert.Equal(0, plan.MaskUploads);
    }

    private static float[,,,] Values(int batches, int tokens, int heads, int dimensions, int divisor, float magnitude)
    {
        var values = new float[batches, tokens, heads, dimensions];
        for (int b = 0; b < batches; b++)for (int t = 0; t < tokens; t++)for (int h = 0; h < heads; h++)for (int d = 0; d < dimensions; d++)
            values[b, t, h, d] = ((b * 11 + t * 7 + h * 13 + d * 17) % 37 - 18) * magnitude / divisor;
        return values;
    }

    private static void Fill(float[,,,] array, float value)
        => MemoryMarshal.CreateSpan(ref array[0, 0, 0, 0], array.Length).Fill(value);

    private static void CheckRow(float[] actual, float[,,,] queries, float[,,,] keys, float[,,,] values,
        int batch, int token, int begin, int length, int heads, int kvHeads, int dimensions, float scale, float magnitude,
        int? windowRadius = null)
    {
        int tokens = queries.GetLength(1);
        for (int head = 0; head < heads; head++)
        {
            int kv = head / (heads / kvHeads); var scores = new double[length];
            for (int j = 0; j < length; j++)for (int d = 0; d < dimensions; d++)
                scores[j] += (double)queries[batch, token, head, d] * keys[batch, begin + j, kv, d];
            for (int j = 0; j < length; j++)
                scores[j] = windowRadius.HasValue && Math.Abs(begin + j - token) > windowRadius.Value ?
                    double.NegativeInfinity : scores[j] * scale;
            double maximum = scores.Max();for (int j = 0; j < length; j++) scores[j] = Math.Exp(scores[j] - maximum);
            double denominator = scores.Sum();
            for (int d = 0; d < dimensions; d++)
            {
                double expected = 0;
                for (int j = 0; j < length; j++) expected += scores[j] / denominator * values[batch, begin + j, kv, d];
                // Ordinary unrounded operands keep the fixed3e-5 operator bound.
                // The separate80K range case uses the same relative F32 bound.
                Assert.InRange(Math.Abs(expected - actual[((batch * tokens + token) * heads + head) * dimensions + d]), 0, 3e-5 * magnitude);
            }
        }
    }
}
