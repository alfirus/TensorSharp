using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class EmbeddingGemma2AudioSscpTests
{
    [Theory]
    [InlineData(1, 1, 1, 3, 3, false)]
    [InlineData(5, 7, 2, 3, 3, false)]
    [InlineData(8, 5, 3, 3, 3, true)]
    [InlineData(6, 4, 2, 1, 1, true)]
    [InlineData(7, 6, 2, 3, 1, true)]
    public void ManagedSscpMatchesIndependentConvolutionNormalizationAndRelu(int width, int height,
        int channels, int kernelWidth, int kernelHeight, bool normalize)
        => Verify(new CpuAllocator(BlasEnum.DotNet), width, height, channels, kernelWidth, kernelHeight, normalize, 2e-5);

    [MlxFact]
    public void MlxSscpMatchesIndependentConvolutionNormalizationAndRelu()
    {
        using var allocator = new MlxAllocator();
        Verify(allocator, 8, 5, 3, 3, 3, true, 3e-5);
    }

    [CudaFact]
    public void CudaSscpMatchesIndependentConvolutionNormalizationAndRelu()
    {
        using var allocator = new CudaAllocator();
        Verify(allocator, 8, 5, 3, 3, 3, true, 3e-5);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuSscpMatchesIndependentConvolutionNormalizationAndRelu() => VerifyGgml(GgmlBackendType.Cpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void NativeMetalSscpMatchesIndependentConvolutionNormalizationAndRelu() => VerifyGgml(GgmlBackendType.Metal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void NativeCudaSscpMatchesIndependentConvolutionNormalizationAndRelu() => VerifyGgml(GgmlBackendType.Cuda);

    private static void VerifyGgml(GgmlBackendType backend)
    {
        var allocator = new GgmlAllocator(new GgmlContext([0], backend), 0);
        try { Verify(allocator, 8, 5, 3, 3, 3, true, 3e-4); }
        finally
        {
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }

    private static void Verify(IAllocator allocator, int width, int height, int channels,
        int kernelWidth, int kernelHeight, bool normalize, double tolerance)
    {
        const int outputs = 5;
        const float epsilon = 1e-5f;
        int columns = channels * kernelHeight * kernelWidth;
        float[] input = Enumerable.Range(0, width * height * channels).Select(i => (i % 17 - 8) / 7f).ToArray();
        var matrix = new float[columns, outputs];
        for (int row = 0; row < columns; ++row)
        for (int column = 0; column < outputs; ++column)
            matrix[row, column] = ((row * 7 + column * 3) % 19 - 9) / 13f;
        float[] normScale = [.7f, 1.1f, .9f, 1.3f, .5f], normOffset = [-.3f, .1f, -.2f, .2f, -.1f];
        using var projection = Tensor.FromArray(allocator, matrix);
        using var normWeight = normalize ? Tensor.FromArray(allocator, normScale) : null;
        using var normBias = normalize ? Tensor.FromArray(allocator, normOffset) : null;
        using var actual = Gemma4AudioEncoder.SscpConvolution(allocator, input, width, height, channels,
            kernelWidth, kernelHeight, projection, normWeight!, normBias!, epsilon);
        int outWidth = (width + 2 - kernelWidth) / 2 + 1, outHeight = (height + 2 - kernelHeight) / 2 + 1;
        Assert.Equal(new long[] { outWidth * outHeight, outputs }, actual.Sizes.ToArray());
        float[] values = actual.GetElementsAsFloat(outWidth * outHeight * outputs);
        // This reference works in output/channel space, without im2col or tensor
        // operators. It checks both planar input indexing and the projection's
        // (input channel, kernel y, kernel x) axis ordering.
        for (int oy = 0; oy < outHeight; ++oy)
        for (int ox = 0; ox < outWidth; ++ox)
        {
            var reference = new double[outputs];
            for (int oc = 0; oc < outputs; ++oc)
            for (int ic = 0; ic < channels; ++ic)
            for (int ky = 0; ky < kernelHeight; ++ky)
            for (int kx = 0; kx < kernelWidth; ++kx)
            {
                int iy = oy * 2 - 1 + ky, ix = ox * 2 - 1 + kx;
                if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                    reference[oc] += (double)input[(ic * height + iy) * width + ix]
                        * matrix[(ic * kernelHeight + ky) * kernelWidth + kx, oc];
            }
            if (normalize)
            {
                double mean = reference.Average();
                double variance = reference.Sum(value => (value - mean) * (value - mean)) / outputs;
                for (int oc = 0; oc < outputs; ++oc)
                    reference[oc] = (reference[oc] - mean) / Math.Sqrt(variance + epsilon) * normScale[oc] + normOffset[oc];
            }
            for (int oc = 0; oc < outputs; ++oc)
            {
                double expected = Math.Max(0, reference[oc]);
                float value = values[(oy * outWidth + ox) * outputs + oc];
                Assert.True(float.IsFinite(value));
                Assert.InRange(Math.Abs(value - expected), 0, Math.Max(1, Math.Abs(expected)) * tolerance);
            }
        }
    }
}
