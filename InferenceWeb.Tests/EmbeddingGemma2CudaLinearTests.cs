using TensorSharp;
using TensorSharp.Cuda;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
internal sealed class EmbeddingGemma2Tf32TheoryAttribute : TheoryAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda";

    public EmbeddingGemma2Tf32TheoryAttribute()
    {
        Skip = TestGates.CudaSkip;
        if (Skip != null) return;
        try
        {
            if (CudaDevice.GetDevice(0).ComputeCapabilityMajor < 8)
                Skip = "Requires CUDA tensor cores with compute capability 8.0 or newer.";
        }
        catch (Exception exception)
        {
            Skip = $"Cannot query the CUDA TF32 capability: {exception.Message}";
        }
    }
}

public sealed class EmbeddingGemma2CudaLinearTests
{
    [EmbeddingGemma2Tf32Theory]
    [InlineData(1, 64, 8, 1f)]
    [InlineData(3, 97, 13, 1f)]
    [InlineData(33, 512, 128, 1f)]
    [InlineData(7, 64, 9, 80000f)]
    public void Tf32ProjectionPreservesShapesOffsetsAndFiniteFloat32Range(int rows, int inputs, int outputs, float magnitude)
    {
        // Input/output views deliberately have nonzero storage offsets. Guard
        // rows verify that writing a device view retains the surrounding data.
        var source = new float[rows + 2, inputs];
        var weights = new float[inputs, outputs];
        var destination = new float[rows + 2, outputs];
        for (int row = 0; row < rows + 2; ++row)
        for (int column = 0; column < inputs; ++column)
            source[row, column] = magnitude * (float)Math.Sin((row + 1) * 0.71 + (column + 1) * 0.17);
        for (int column = 0; column < inputs; ++column)
        for (int output = 0; output < outputs; ++output)
            weights[column, output] = (float)Math.Cos((column + 3) * 0.29 + (output + 1) * 0.43) * 0.125f;
        for (int row = 0; row < rows + 2; ++row)
        for (int output = 0; output < outputs; ++output)
            destination[row, output] = 37f;

        using var allocator = new CudaAllocator();
        using var sourceTensor = Tensor.FromArray(allocator, source);
        using var weightTensor = Tensor.FromArray(allocator, weights);
        using var destinationTensor = Tensor.FromArray(allocator, destination);
        using var input = sourceTensor.Narrow(0, 1, rows);
        using var result = destinationTensor.Narrow(0, 1, rows);
        Assert.True(CudaEmbeddingOps.TryLinearTf32(result, input, weightTensor));

        float[] actual = destinationTensor.GetElementsAsFloat((rows + 2) * outputs);
        double errorSquared = 0, expectedSquared = 0;
        for (int row = 0; row < rows; ++row)
        for (int output = 0; output < outputs; ++output)
        {
            double expected = 0, absoluteSum = 0;
            for (int inputIndex = 0; inputIndex < inputs; ++inputIndex)
            {
                double product = (double)source[row + 1, inputIndex] * weights[inputIndex, output];
                expected += product; absoluteSum += Math.Abs(product);
            }
            float value = actual[(row + 1) * outputs + output];
            Assert.True(float.IsFinite(value));
            Assert.InRange(Math.Abs(value - expected), 0, Math.Max(1, absoluteSum) * 1e-3);
            errorSquared += (value - expected) * (value - expected);
            expectedSquared += expected * expected;
        }
        Assert.InRange(Math.Sqrt(errorSquared / expectedSquared), 0, 1e-3);
        for (int output = 0; output < outputs; ++output)
        {
            Assert.Equal(37f, actual[output]);
            Assert.Equal(37f, actual[(rows + 1) * outputs + output]);
        }
    }
}
