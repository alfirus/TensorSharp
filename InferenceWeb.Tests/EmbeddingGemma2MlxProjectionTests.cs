using System.Buffers.Binary;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.MLX;
using TensorSharp.Models.Embeddings;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class EmbeddingGemma2MlxProjectionTests
{
    [Fact]
    public void NormalizedProjectionRangeProofRejectsUnsafePartialDotsAndNonFiniteWeights()
    {
        float[] gains = Enumerable.Repeat(1f, 512).ToArray();
        float[] safe = Enumerable.Repeat(2f, 512).ToArray();
        Assert.True(TensorEmbeddingGemma2Encoder.CanUseHalfProjection(gains, safe, 512));
        // Every input and weight component fits F16, but the dot can exceed its range.
        float[] unsafeWeights = Enumerable.Repeat(1000f, 512).ToArray();
        Assert.False(TensorEmbeddingGemma2Encoder.CanUseHalfProjection(gains, unsafeWeights, 512));
        safe[10] = float.NaN;
        Assert.False(TensorEmbeddingGemma2Encoder.CanUseHalfProjection(gains, safe, 512));
        safe[10] = 2;
        gains[10] = float.PositiveInfinity;
        Assert.False(TensorEmbeddingGemma2Encoder.CanUseHalfProjection(gains, safe, 512));
        Assert.False(TensorEmbeddingGemma2Encoder.CanUseHalfProjection([1f], [1f], 512));
    }

    [MlxFact]
    public void BoundedNormalizedProjectionUsesFiniteF32StorageAndExactBinaryProducts()
    {
        const int rows = 3, inputs = 64, outputs = 8;
        var source = new float[rows, inputs];
        var weights = new float[inputs, outputs];
        for (int row = 0; row < rows; ++row)
        for (int column = 0; column < inputs; ++column)
            source[row, column] = ((row * 7 + column * 3) % 9 - 4) / 8f;
        for (int column = 0; column < inputs; ++column)
        for (int output = 0; output < outputs; ++output)
            weights[column, output] = ((column * 5 + output * 7) % 7 - 3) / 8f;
        using var allocator = new MlxAllocator();
        using var input = Tensor.FromArray(allocator, source);
        using var originalWeight = Tensor.FromArray(allocator, weights);
        using var halfWeight = new Tensor(allocator, DType.Float16, inputs, outputs);
        Ops.Copy(halfWeight, originalWeight);
        using var result = new Tensor(allocator, DType.Float32, rows, outputs);
        Assert.True(MlxEmbeddingOps.TryNormalizedHalfProjection(result, input, halfWeight));
        Assert.Equal(DType.Float32, result.ElementType);
        var actual = result.GetElementsAsFloat(rows * outputs);
        for (int row = 0; row < rows; ++row)
        for (int output = 0; output < outputs; ++output)
        {
            double expected = 0;
            for (int column = 0; column < inputs; ++column) expected += (double)source[row, column] * weights[column, output];
            Assert.Equal((float)expected, actual[row * outputs + output]);
        }
    }

    [MlxTheory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(17)]
    public void QuantizedProjectionPreservesFiniteActivationsBeyondFloat16Range(int rows)
    {
        const int inputs = 64, outputs = 8, blockSize = 32, blockBytes = 34;
        byte[] weights = new byte[outputs * (inputs / blockSize) * blockBytes];
        var dense = new float[outputs, inputs];
        for (int output = 0; output < outputs; ++output)
        for (int block = 0; block < inputs / blockSize; ++block)
        {
            int offset = (output * (inputs / blockSize) + block) * blockBytes;
            float scale = (output + block + 1) / 256f;
            BinaryPrimitives.WriteUInt16LittleEndian(weights.AsSpan(offset), BitConverter.HalfToUInt16Bits((System.Half)scale));
            for (int column = 0; column < blockSize; ++column)
            {
                sbyte value = (sbyte)(((output + 3) * (column + block * blockSize + 7)) % 127 - 63);
                weights[offset + 2 + column] = unchecked((byte)value);
                dense[output, block * blockSize + column] = scale * value;
            }
        }
        var source = new float[rows, inputs];
        for (int row = 0; row < rows; ++row)
        for (int input = 0; input < inputs; ++input)
            source[row, input] = (input % 2 == 0 ? 1 : -1) * (80000f + row * 257 + input * 113);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        using var allocator = new MlxAllocator();
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var input = Tensor.FromArray(allocator, source);
            using var result = new Tensor(allocator, DType.Float32, rows, outputs);
            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(result, input, host, host,
                (int)GgmlTensorType.Q8_0, inputs, outputs, weights.Length, preserveFloat32: true));
            float[] actual = result.GetElementsAsFloat(rows * outputs);
            Assert.All(actual, value => Assert.True(float.IsFinite(value)));
            for (int row = 0; row < rows; ++row)
            for (int output = 0; output < outputs; ++output)
            {
                double expected = 0;
                for (int inputIndex = 0; inputIndex < inputs; ++inputIndex)
                    expected += (double)source[row, inputIndex] * dense[output, inputIndex];
                Assert.InRange(Math.Abs(actual[row * outputs + output] - expected), 0, Math.Max(1, Math.Abs(expected)) * 2e-5);
            }
        }
        finally
        {
            MlxQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            Marshal.FreeHGlobal(host);
        }
    }
}
