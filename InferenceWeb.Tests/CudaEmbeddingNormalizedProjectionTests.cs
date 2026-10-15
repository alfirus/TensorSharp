using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models.Embeddings;

namespace InferenceWeb.Tests;

public sealed class CudaEmbeddingNormalizedProjectionTests
{
    [Fact]
    public void ProofMatchesExistingPartialDotPolicyAndRejectsUnsafeGainsWeightsAndNonFiniteValues()
    {
        foreach (int inputs in new[] { 1, 65, 129, 768 })
        foreach (float scale in new[] { .00313f, 1f, 1000f })
        {
            var gains = Gains(inputs); var weights = Weights(inputs, 7, scale);
            bool expected = TensorEmbeddingGemma2Encoder.CanUseHalfProjection(gains, weights, inputs);
            Assert.Equal(expected, CudaEmbeddingProjectionProof.TryCreate(gains, weights, inputs, out var proof));
            if (expected) { Assert.InRange(proof.NormalizedComponentBound, 0, 16384); Assert.InRange(proof.MaximumPartialDotBound, 0, 16384); }
        }
        var safeGains = Gains(65); var safe = Weights(65, 19);
        safeGains[7] = 80000;
        Assert.False(CudaEmbeddingProjectionProof.TryCreate(safeGains, safe, 65, out _));
        safeGains[7] = 1; safe[9] = 80000;
        Assert.False(CudaEmbeddingProjectionProof.TryCreate(safeGains, safe, 65, out _));
        safe[9] = float.NaN;
        Assert.False(CudaEmbeddingProjectionProof.TryCreate(safeGains, safe, 65, out _));
        safe[9] = 1; safeGains[7] = float.PositiveInfinity;
        Assert.False(CudaEmbeddingProjectionProof.TryCreate(safeGains, safe, 65, out _));
        Assert.False(CudaEmbeddingProjectionProof.TryCreate([1f], [1f], 65, out _));
        // Components fit Half, but the worst complete/partial dot does not.
        Assert.False(CudaEmbeddingProjectionProof.TryCreate(Enumerable.Repeat(1f, 512).ToArray(), Enumerable.Repeat(1000f, 512).ToArray(), 512, out _));
    }

    [CudaTheory]
    [InlineData(1, 65, 19, false, 1f)]
    [InlineData(13, 129, 37, false, 1f)]
    [InlineData(29, 257, 131, false, 1f)]
    [InlineData(13, 512, 2048, false, 1f)]
    [InlineData(29, 512, 4096, false, 1f)]
    [InlineData(256, 512, 3072, false, 1f)]
    [InlineData(8192, 65, 19, false, 1f)]
    [InlineData(13, 129, 37, true, 1f)]
    [InlineData(3, 65, 19, false, 80000f)]
    public void OwnedRmsBoundaryUnroundedTailsAndOffsetsMatchIndependentNormAndRoundedOperandDoubleOracle(
        int rows, int inputs, int outputs, bool pedantic, float magnitude)
    {
        using var allocator = new CudaAllocator();
        using var plan = new CudaEmbeddingProjectionPlan(allocator, pedantic);
        var gains = Gains(inputs); var weights = Weights(inputs, outputs);
        Assert.True(plan.TryPrepareWeight(gains, weights, inputs, out var weight));
        var source = Input(rows + 2, inputs, magnitude); var destination = Fill(rows + 2, outputs, 37);
        using var xb = Tensor.FromArray(allocator, source); using var yb = Tensor.FromArray(allocator, destination);
        // This starts outside Half's range; the API must overwrite it with its
        // actual F32 RMS result instead of trusting a claimed normalized input.
        using var nb = Tensor.FromArray(allocator, Fill(rows + 2, inputs, 80000));
        using var x = xb.Narrow(0, 1, rows); using var y = yb.Narrow(0, 1, rows); using var n = nb.Narrow(0, 1, rows);
        const float epsilon = 1e-6f;
        Assert.True(plan.TryNormalizeAndProject(y, n, x, weight, epsilon));
        float[] actual = yb.GetElementsAsFloat(destination.Length), normalized = nb.GetElementsAsFloat((rows + 2) * inputs);
        plan.NotifyReadbackCompleted();
        Assert.Equal(pedantic ? 69 : 68, plan.ComputeType); Assert.Equal(1, plan.WeightUses); Assert.Equal(1, plan.SuccessfulCalls);
        Assert.Equal(rows, plan.SuccessfulRows); Assert.Equal(1, plan.PreparedWeightCount);
        Assert.InRange(plan.WeightCapacityBytes, 256, plan.WeightBudgetBytes);
        Assert.InRange(plan.WorkspaceCapacityBytes, 256, plan.WorkspaceBudgetBytes);
        foreach (int row in Enumerable.Range(0, rows))
        {
            double squares = 0;
            for (int d = 0; d < inputs; d++) squares += (double)source[row + 1, d] * source[row + 1, d];
            double reciprocal = 1 / Math.Sqrt(squares / inputs + epsilon);
            for (int d = 0; d < inputs; d++)
            {
                double expected = source[row + 1, d] * reciprocal * gains[d];
                Assert.InRange(Math.Abs(normalized[(row + 1) * inputs + d] - expected), 0, 3e-5);
            }
            // All outputs for small shapes, edge/interior outputs for actual
            // joined QKV/gate-up widths. Full model accuracy is a separate gate.
            int[] columns = outputs < 256 ? Enumerable.Range(0, outputs).ToArray() : [0, 1, outputs / 2, outputs - 2, outputs - 1];
            foreach (int output in columns)
            {
                double expected = 0;
                for (int d = 0; d < inputs; d++)
                    expected += (double)(System.Half)normalized[(row + 1) * inputs + d] * (double)(System.Half)weights[output * inputs + d];
                Assert.InRange(Math.Abs(actual[(row + 1) * outputs + output] - expected), 0, 3e-5);
            }
        }
        for (int d = 0; d < outputs; d++) { Assert.Equal(37, actual[d]); Assert.Equal(37, actual[(rows + 1) * outputs + d]); }
        for (int d = 0; d < inputs; d++) { Assert.Equal(80000, normalized[d]); Assert.Equal(80000, normalized[(rows + 1) * inputs + d]); }
        long capacity = plan.WorkspaceCapacityBytes;
        Assert.True(plan.TryNormalizeAndProject(y, n, x, weight, epsilon));
        Assert.Equal(actual, yb.GetElementsAsFloat(destination.Length)); plan.NotifyReadbackCompleted();
        Assert.Equal(capacity, plan.WorkspaceCapacityBytes); Assert.Equal(1, plan.WorkspaceReuses);
    }

    [CudaFact]
    public void Float32AccumulationKeepsSmallCancellationTermsLostByHalfReduction()
    {
        const int inputs = 1024;
        using var allocator = new CudaAllocator(); using var plan = new CudaEmbeddingProjectionPlan(allocator);
        var gains = Enumerable.Repeat(1f, inputs).ToArray(); var weights = new float[2 * inputs];
        for (int i = 0; i < inputs; i++)
        {
            weights[i] = i < 256 ? 8f : i < 768 ? 1f / 4096 : -8f;
            weights[inputs + i] = i % 2 == 0 ? 4.00313f : -4.00137f;
        }
        Assert.True(plan.TryPrepareWeight(gains, weights, inputs, out var weight));
        using var x = Tensor.FromArray(allocator, Fill(3, inputs, 1));
        using var n = new Tensor(allocator, DType.Float32, 3, inputs); using var y = new Tensor(allocator, DType.Float32, 3, 2);
        Assert.True(plan.TryNormalizeAndProject(y, n, x, weight, 1e-6f));
        var normalized = n.GetElementsAsFloat(3 * inputs); var actual = y.GetElementsAsFloat(6); plan.NotifyReadbackCompleted();
        for (int row = 0; row < 3; row++)
        for (int output = 0; output < 2; output++)
        {
            double expected = 0;
            for (int i = 0; i < inputs; i++) expected += (double)(System.Half)normalized[row * inputs + i] * (double)(System.Half)weights[output * inputs + i];
            Assert.InRange(Math.Abs(actual[row * 2 + output] - expected), 0, 3e-5);
        }
        Assert.Equal(.125f, actual[0]); // Half running reduction loses this term after the 2048 prefix.
    }

    [CudaFact]
    public void InvalidProofShapeAliasContextBudgetAndCanceledCallsLeaveBothOutputsUntouched()
    {
        using var allocator = new CudaAllocator(); using var peerAllocator = new CudaAllocator();
        using var plan = new CudaEmbeddingProjectionPlan(allocator); using var peer = new CudaEmbeddingProjectionPlan(allocator);
        var gains = Gains(65); var weights = Weights(65, 19);
        Assert.True(plan.TryPrepareWeight(gains, weights, 65, out var weight)); Assert.True(peer.TryPrepareWeight(gains, weights, 65, out var peerWeight));
        using var x = Tensor.FromArray(allocator, Input(3, 65, 1)); using var n = Tensor.FromArray(allocator, Fill(3, 65, 41));
        using var y = Tensor.FromArray(allocator, Fill(3, 19, 37)); using var py = Tensor.FromArray(peerAllocator, Fill(3, 19, 37));
        Assert.False(plan.TryNormalizeAndProject(y, n, x, null, 1e-6f));
        Assert.False(plan.TryNormalizeAndProject(y, n, x, peerWeight, 1e-6f));
        Assert.False(plan.TryNormalizeAndProject(py, n, x, weight, 1e-6f));
        Assert.False(plan.TryNormalizeAndProject(y, x, x, weight, 1e-6f));
        Assert.False(plan.TryNormalizeAndProject(n, n, x, weight, 1e-6f));
        Assert.False(plan.TryNormalizeAndProject(y, n, x, weight, float.NaN));
        using (var strided = x.Narrow(1, 0, 64)) Assert.False(plan.TryNormalizeAndProject(y, n, strided, weight, 1e-6f));
        using (var outside = new Tensor([3, 65], [65, 1], x.Storage, 65))
            Assert.False(plan.TryNormalizeAndProject(y, n, outside, weight, 1e-6f));
        using var tinyWorkspace = new CudaEmbeddingProjectionPlan(allocator, workspaceBudgetBytes: 256);
        Assert.True(tinyWorkspace.TryPrepareWeight(gains, weights, 65, out var tinyWeight));
        Assert.False(tinyWorkspace.TryNormalizeAndProject(y, n, x, tinyWeight, 1e-6f));
        using var tinyWeights = new CudaEmbeddingProjectionPlan(allocator, weightBudgetBytes: 256);
        Assert.False(tinyWeights.TryPrepareWeight(gains, weights, 65, out _)); Assert.Equal(0, tinyWeights.PreparedWeightCount);
        using var canceled = new CancellationTokenSource(); canceled.Cancel();
        Assert.Throws<OperationCanceledException>(() => plan.TryNormalizeAndProject(y, n, x, weight, 1e-6f, canceled.Token));
        Assert.Throws<OperationCanceledException>(() => plan.TryPrepareWeight(gains, weights, 65, out _, canceled.Token));
        Assert.All(y.GetElementsAsFloat(57), value => Assert.Equal(37, value));
        Assert.All(n.GetElementsAsFloat(195), value => Assert.Equal(41, value)); Assert.Equal(0, plan.SuccessfulCalls);
        gains[3] = 80000; Assert.False(plan.TryPrepareWeight(gains, weights, 65, out _));
        gains[3] = 1; weights[3] = 80000; Assert.False(plan.TryPrepareWeight(gains, weights, 65, out _));
        Assert.Equal(2, plan.ProofRejections);
    }

    [CudaFact]
    public void SnapshotReuseGrowthAbortAndDisposalPreservePeerAndCallerOwnedTensors()
    {
        using var allocator = new CudaAllocator(); using var plan = new CudaEmbeddingProjectionPlan(allocator); using var peer = new CudaEmbeddingProjectionPlan(allocator);
        var gains = Gains(65); var weights = Weights(65, 19);
        Assert.True(plan.TryPrepareWeight(gains, weights, 65, out var weight)); Assert.True(peer.TryPrepareWeight(gains, weights, 65, out var peerWeight));
        // Mutating/freeing preparation inputs cannot alter retained proof or weights.
        Array.Fill(gains, 80000); Array.Fill(weights, 80000);
        using var x = Tensor.FromArray(allocator, Input(3, 65, 1)); using var n = new Tensor(allocator, DType.Float32, 3, 65); using var y = new Tensor(allocator, DType.Float32, 3, 19);
        Assert.True(plan.TryNormalizeAndProject(y, n, x, weight, 1e-6f)); var expected = y.GetElementsAsFloat(57); plan.NotifyReadbackCompleted();
        Assert.True(peer.TryNormalizeAndProject(y, n, x, peerWeight, 1e-6f)); Assert.Equal(expected, y.GetElementsAsFloat(57)); peer.NotifyReadbackCompleted();
        using var largerX = Tensor.FromArray(allocator, Input(29, 65, 1)); using var largerN = new Tensor(allocator, DType.Float32, 29, 65); using var largerY = new Tensor(allocator, DType.Float32, 29, 19);
        Assert.True(plan.TryNormalizeAndProject(y, n, x, weight, 1e-6f));
        long old = plan.WorkspaceCapacityBytes;
        Assert.True(plan.TryNormalizeAndProject(largerY, largerN, largerX, weight, 1e-6f));
        Assert.True(plan.WorkspaceCapacityBytes > old); plan.AbortPendingWork();
        Assert.All(largerY.GetElementsAsFloat(29 * 19), value => Assert.True(float.IsFinite(value)));
        Assert.True(plan.TryNormalizeAndProject(largerY, largerN, largerX, weight, 1e-6f));
        plan.Dispose(); plan.Dispose(); Assert.Equal(0, plan.PreparedWeightCount); Assert.Equal(0, plan.WeightCapacityBytes); Assert.Equal(0, plan.WorkspaceCapacityBytes);
        Assert.Throws<ObjectDisposedException>(() => plan.TryNormalizeAndProject(y, n, x, weight, 1e-6f));
        Assert.True(peer.TryNormalizeAndProject(y, n, x, peerWeight, 1e-6f)); Assert.Equal(expected, y.GetElementsAsFloat(57)); peer.NotifyReadbackCompleted();
    }

    private static float[] Gains(int count) => Enumerable.Range(0, count).Select(i => .83137f + i % 11 * .01973f).ToArray();
    private static float[] Weights(int inputs, int outputs, float scale = 1)
        => Enumerable.Range(0, checked(inputs * outputs)).Select(i => scale * MathF.Sin(i * .13791f + .19f) * .07317f).ToArray();
    private static float[,] Input(int rows, int inputs, float magnitude)
    {
        var values = new float[rows, inputs];
        for (int r = 0; r < rows; r++) for (int i = 0; i < inputs; i++) values[r, i] = magnitude * (.07137f + MathF.Sin(r * .79173f + i * .17391f));
        return values;
    }
    private static float[,] Fill(int rows, int columns, float value)
    { var values = new float[rows, columns]; for (int r = 0; r < rows; r++) for (int c = 0; c < columns; c++) values[r, c] = value; return values; }
}
