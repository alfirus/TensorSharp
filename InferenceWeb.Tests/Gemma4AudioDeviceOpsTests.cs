using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class Gemma4AudioDeviceOpsTests
{
    [MlxFact]
    public void MlxRelativeAttentionAndCausalDepthwiseMatchIndependentDoubleOracle()
    { using var allocator = new MlxAllocator(); Verify(allocator); }

    [CudaFact]
    public void CudaRelativeAttentionAndCausalDepthwiseMatchIndependentDoubleOracle()
    { using var allocator = new CudaAllocator(); Verify(allocator); }

    [CudaFact]
    public void CudaPedanticProjectionPreservesLargeFloat32RangeAndMatchesIndependentDoubleProducts()
    {
        using var allocator = new CudaAllocator();
        foreach (int rows in new[] { 1, 3, 17 })
        foreach (int inner in new[] { 9, 1152 })
        {
            const int columns = 13;
            var inputs = new float[rows, inner]; var weights = new float[inner, columns];
            for (int row = 0; row < rows; ++row)
                for (int k = 0; k < inner; ++k)
                    inputs[row, k] = k == 0 ? 80000 + row * 64 : ((row * 3 + k) % 9 - 4) / 4f;
            for (int k = 0; k < inner; ++k)
                for (int col = 0; col < columns; ++col) weights[k, col] = ((k + col) % 7 - 3) / 4f;
            using var input = Tensor.FromArray(allocator, inputs);
            using var weight = Tensor.FromArray(allocator, weights);
            using var result = new Tensor(allocator, DType.Float32, rows, columns);
            Assert.True(CudaGemma4AudioOps.TryProjectionFloat32(result, input, weight));
            var actual = result.GetElementsAsFloat(rows * columns);
            Assert.All(actual, value => Assert.True(float.IsFinite(value)));
            for (int row = 0; row < rows; ++row)
            for (int col = 0; col < columns; ++col)
            {
                double expected = 0;
                for (int k = 0; k < inner; ++k) expected += (double)inputs[row, k] * weights[k, col];
                Assert.InRange(Math.Abs(expected - actual[row * columns + col]), 0, 3e-5);
            }
            using var halfWeight = new Tensor(allocator, DType.Float16, inner, columns);
            Assert.False(CudaGemma4AudioOps.TryProjectionFloat32(result, input, halfWeight));
        }
        using var aliasInput = new Tensor(allocator, DType.Float32, 3, 9);
        using var squareWeight = new Tensor(allocator, DType.Float32, 9, 9);
        Assert.False(CudaGemma4AudioOps.TryProjectionFloat32(aliasInput, aliasInput, squareWeight));
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuRelativeAttentionAndCausalDepthwiseMatchIndependentDoubleOracle() => VerifyGgml(GgmlBackendType.Cpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void NativeMetalRelativeAttentionAndCausalDepthwiseMatchIndependentDoubleOracle() => VerifyGgml(GgmlBackendType.Metal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void NativeCudaRelativeAttentionAndCausalDepthwiseMatchIndependentDoubleOracle() => VerifyGgml(GgmlBackendType.Cuda);

    private static void VerifyGgml(GgmlBackendType backend)
    {
        var allocator = new GgmlAllocator(new GgmlContext([0], backend), 0);
        try { Verify(allocator); }
        finally { GgmlBasicOps.ReleaseReuseComputeBuffers(); GgmlBasicOps.ClearHostBufferCache(); }
    }

    private static void Verify(IAllocator allocator)
    {
        VerifyAttention(allocator, 1, 2, 4, 12, 0, 5);
        VerifyAttention(allocator, 25, 2, 128, 12, 0, 5);
        VerifyAttention(allocator, 25, 2, 4, 3, 2, 2);
        VerifyDepthwise(allocator);
    }

    private static void VerifyAttention(IAllocator allocator, int sequence, int heads, int head, int past, int future, float cap)
    {
        int hidden = heads * head, window = past + future + 1;
        var q = new float[sequence, hidden]; var k = new float[sequence, hidden]; var v = new float[sequence, hidden];
        var relative = new float[window, hidden];
        for (int t = 0; t < sequence; ++t)
        for (int d = 0; d < hidden; ++d)
        {
            q[t, d] = ((t * 7 + d * 3) % 19 - 9) / 8f;
            k[t, d] = ((t * 11 + d * 5) % 23 - 11) / 8f;
            v[t, d] = ((t * 5 + d * 7) % 17 - 8) / 8f;
        }
        for (int p = 0; p < window; ++p)
        for (int d = 0; d < hidden; ++d) relative[p, d] = ((p * 13 + d * 7) % 29 - 14) / 8f;
        using var qt = Tensor.FromArray(allocator, q); using var kt = Tensor.FromArray(allocator, k);
        using var vt = Tensor.FromArray(allocator, v); using var rt = Tensor.FromArray(allocator, relative);
        using var result = new Tensor(allocator, DType.Float32, sequence, hidden);
        Assert.True(Gemma4AudioTensorOps.TryAttention(result, qt, kt, vt, rt, sequence, heads, head, past, future, cap));
        Assert.Equal(DType.Float32, result.ElementType);
        var actual = result.GetElementsAsFloat(sequence * hidden);
        for (int t = 0; t < sequence; ++t)
        for (int h = 0; h < heads; ++h)
        {
            int first = Math.Max(0, t - past), last = Math.Min(sequence - 1, t + future);
            var scores = new double[last - first + 1];
            for (int key = first; key <= last; ++key)
            {
                double cc = 0, cp = 0;
                for (int d = 0; d < head; ++d)
                {
                    cc += (double)q[t, h * head + d] * k[key, h * head + d];
                    cp += (double)q[t, h * head + d] * relative[key - t + past, h * head + d];
                }
                scores[key - first] = Math.Tanh((cc + cp) / cap) * cap;
            }
            double max = scores.Max(), total = scores.Sum(x => Math.Exp(x - max));
            for (int d = 0; d < head; ++d)
            {
                double expected = 0;
                for (int key = first; key <= last; ++key)
                    expected += Math.Exp(scores[key - first] - max) / total * v[key, h * head + d];
                Assert.InRange(Math.Abs(expected - actual[t * hidden + h * head + d]), 0, 3e-5);
            }
        }
        Assert.False(Gemma4AudioTensorOps.TryAttention(result, qt, kt, vt, rt, sequence, heads, head, -1, future, cap));
    }

    private static void VerifyDepthwise(IAllocator allocator)
    {
        const int sequence = 25, channels = 7, kernel = 5;
        var x = new float[sequence, channels]; var weights = new float[channels, kernel];
        for (int t = 0; t < sequence; ++t)
        for (int c = 0; c < channels; ++c) x[t, c] = (t % 2 == 0 ? 1 : -1) * (80000 + t * 13 + c * 7);
        for (int c = 0; c < channels; ++c)
        for (int k = 0; k < kernel; ++k) weights[c, k] = ((c * 3 + k * 7) % 11 - 5) / 8f;
        using var input = Tensor.FromArray(allocator, x); using var weight = Tensor.FromArray(allocator, weights);
        using var result = new Tensor(allocator, DType.Float32, sequence, channels);
        Assert.True(Gemma4AudioTensorOps.TryDepthwise(result, input, weight, sequence, channels, kernel));
        var actual = result.GetElementsAsFloat(sequence * channels);
        Assert.All(actual, value => Assert.True(float.IsFinite(value)));
        for (int t = 0; t < sequence; ++t)
        for (int c = 0; c < channels; ++c)
        {
            double expected = 0;
            for (int k = 0; k < kernel; ++k)
            {
                int source = t - (kernel - 1 - k);
                if (source >= 0) expected += (double)x[source, c] * weights[c, k];
            }
            Assert.InRange(Math.Abs(expected - actual[t * channels + c]), 0, Math.Max(1, Math.Abs(expected)) * 3e-5);
        }
    }
}
