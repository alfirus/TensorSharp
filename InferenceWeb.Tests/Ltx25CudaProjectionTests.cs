using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>Exact F32 LTX projection against independently dequantized GGML
/// weights. CUDA absence is a visible skip; a missing owned module on an
/// available CUDA device is an explicit failure, never a fallback pass.</summary>
public sealed class Ltx25CudaProjectionTests
{
    public static IEnumerable<object[]> Cases
    {
        get
        {
            foreach (int type in new[] { 12, 13, 14 })
            {
                foreach (int rows in new[] { 2, 7, 8, 12, 16, 17, 31 })
                    yield return new object[] { type, rows, rows % 2 == 0 ? 256 : 512, rows % 2 == 0 ? 19 : 67, 1f };
                yield return new object[] { type, 7, 512, 67, 80000f };
            }
        }
    }

    [CudaTheory]
    [MemberData(nameof(Cases))]
    public void QuantizedBlocksRaggedTilesAndUnroundedActivationsMatchDoubleDotOracle(
        int type, int rows, int inner, int columns, float magnitude)
    {
        byte[] packed = Weights(type, columns, inner);
        using var allocator = new CudaAllocator();
        IntPtr host = Marshal.AllocHGlobal(packed.Length);
        try
        {
            Marshal.Copy(packed, 0, host, packed.Length);
            using var session = new CudaLtx25ProjectionSession(allocator, requireNative: true);
            Assert.True(session.IsAvailable);
            var inputValues = new float[rows + 2, inner];
            var initialOutput = new float[rows + 2, columns];
            for (int row = 0; row < rows + 2; ++row)
            {
                for (int column = 0; column < inner; ++column)
                    inputValues[row, column] = MathF.Sin((row + 1) * .17f + column * .131f) * magnitude + (column % 11 - 5) * .03125f;
                for (int column = 0; column < columns; ++column)
                    initialOutput[row, column] = 37f;
            }
            using var inputBase = Tensor.FromArray(allocator, inputValues);
            using var outputBase = Tensor.FromArray(allocator, initialOutput);
            using var input = inputBase.Narrow(0, 1, rows);
            using var output = outputBase.Narrow(0, 1, rows);
            Assert.True(session.TryLinear(output, input, host, type, inner, columns, packed.Length));
            float[] actual = outputBase.GetElementsAsFloat(initialOutput.Length);
            Assert.Equal(1, session.ExecutedProjections);
            var weights = new float[inner];
            int rowBytes = checked((int)GgmlGgufTensorDequant.GetRowSizeBytes(type, inner));
            for (int column = 0; column < columns; ++column)
            {
                GgmlGgufTensorDequant.DequantizeToFloat32(type, packed, column * rowBytes, weights, 0, inner);
                for (int row = 0; row < rows; ++row)
                {
                    double expected = 0, absoluteProducts = 0;
                    for (int k = 0; k < inner; ++k)
                    {
                        double product = (double)inputValues[row + 1, k] * weights[k];
                        expected += product;
                        absoluteProducts += Math.Abs(product);
                    }
                    float value = actual[(row + 1) * columns + column];
                    Assert.True(float.IsFinite(value));
                    // A reduction-order bound, scaled by its actual operands.
                    // Half or q8 activation narrowing exceeds this F32 bound.
                    double tolerance = 1e-5 + absoluteProducts * 3e-6;
                    Assert.True(Math.Abs(value - expected) <= tolerance,
                        $"type={type}, row={row}, column={column}: expected {expected:R}, actual {value:R}, tolerance {tolerance:R}");
                }
            }
            for (int column = 0; column < columns; ++column)
            {
                Assert.Equal(37f, actual[column]);
                Assert.Equal(37f, actual[(rows + 1) * columns + column]);
            }
            // Reuse the same host identity and module; a second native execution
            // must overwrite the target consistently and increment the counter.
            Assert.True(session.TryLinear(output, input, host, type, inner, columns, packed.Length));
            Assert.Equal(actual, outputBase.GetElementsAsFloat(initialOutput.Length));
            Assert.Equal(2, session.ExecutedProjections);
        }
        finally { CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host); Marshal.FreeHGlobal(host); }
    }

    [CudaFact]
    public void InvalidStorageAndDimensionsRefuseBeforeWritingTheTarget()
    {
        const int inner = 256, columns = 19;
        byte[] packed = Weights(12, columns, inner);
        using var allocator = new CudaAllocator();
        IntPtr host = Marshal.AllocHGlobal(packed.Length);
        try
        {
            Marshal.Copy(packed, 0, host, packed.Length);
            using var session = new CudaLtx25ProjectionSession(allocator, requireNative: true);
            using var input = new Tensor(allocator, DType.Float32, 2, inner);
            using var output = new Tensor(allocator, DType.Float32, 2, columns);
            Ops.Fill(input, 1f); Ops.Fill(output, 37f);
            Assert.False(session.TryLinear(output, input, IntPtr.Zero, 12, inner, columns, packed.Length));
            Assert.False(session.TryLinear(output, input, host, 0, inner, columns, packed.Length));
            Assert.False(session.TryLinear(output, input, host, 12, inner - 1, columns, packed.Length));
            Assert.False(session.TryLinear(output, input, host, 12, inner, columns, packed.Length - 1));
            foreach (int rows in new[] { 1, 32, 33 })
            {
                using var invalidInput = new Tensor(allocator, DType.Float32, rows, inner);
                using var invalidOutput = new Tensor(allocator, DType.Float32, rows, columns);
                Ops.Fill(invalidInput, 1f); Ops.Fill(invalidOutput, 37f);
                Assert.False(session.TryLinear(invalidOutput, invalidInput, host, 12, inner, columns, packed.Length));
                Assert.All(invalidOutput.GetElementsAsFloat(rows * columns), value => Assert.Equal(37f, value));
            }
            Assert.Equal(0, session.ExecutedProjections);
            Assert.All(output.GetElementsAsFloat(2 * columns), value => Assert.Equal(37f, value));
        }
        finally { CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host); Marshal.FreeHGlobal(host); }
    }

    [CudaFact]
    public void AliasedProjectionCannotOverwriteInputsDuringParallelReduction()
    {
        const int inner = 256;
        byte[] packed = Weights(12, inner, inner);
        using var allocator = new CudaAllocator();
        IntPtr host = Marshal.AllocHGlobal(packed.Length);
        try
        {
            Marshal.Copy(packed, 0, host, packed.Length);
            using var session = new CudaLtx25ProjectionSession(allocator, requireNative: true);
            using var storage = new Tensor(allocator, DType.Float32, 4, inner);
            Ops.Fill(storage, 37f);
            using var input = storage.Narrow(0, 0, 2);
            using var exactAlias = storage.Narrow(0, 0, 2);
            using var shiftedAlias = storage.Narrow(0, 1, 2);
            Assert.False(session.TryLinear(exactAlias, input, host, 12, inner, inner, packed.Length));
            Assert.False(session.TryLinear(shiftedAlias, input, host, 12, inner, inner, packed.Length));
            Assert.Equal(0, session.ExecutedProjections);
            Assert.All(storage.GetElementsAsFloat(4 * inner), value => Assert.Equal(37f, value));
        }
        finally { CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host); Marshal.FreeHGlobal(host); }
    }

    [CudaFact]
    public void ConfiguredF16ThresholdKeepsItsProjectionRouteAndTargetUnchanged()
    {
        const int inner = 256, columns = 19;
        byte[] packed = Weights(12, columns, inner);
        // The production threshold is process-static. Run this fact alone in a
        // fresh process with MIN_ROWS=8 to validate the configured boundary as
        // well as the default boundary32; never mutate process-global settings.
        long threshold = CudaQuantizedOps.F16GemmMinRows;
        int before = (int)Math.Clamp(threshold - 1, 2L, 31L);
        int at = (int)Math.Clamp(threshold, 2L, 32L);
        using var allocator = new CudaAllocator();
        IntPtr host = Marshal.AllocHGlobal(packed.Length);
        try
        {
            Marshal.Copy(packed, 0, host, packed.Length);
            using var session = new CudaLtx25ProjectionSession(allocator, requireNative: true);
            long expectedExecutions = 0;
            foreach (int rows in new[] { before, at }.Distinct())
            {
                using var input = new Tensor(allocator, DType.Float32, rows, inner);
                using var output = new Tensor(allocator, DType.Float32, rows, columns);
                Ops.Fill(input, 1f); Ops.Fill(output, 37f);
                bool f16Eligible = rows >= threshold && 2L * inner * columns <= CudaQuantizedOps.F16GemmMaxWeightBytes;
                bool expectedNative = rows <= 31 && !f16Eligible;
                Assert.Equal(expectedNative, session.TryLinear(output, input, host, 12, inner, columns, packed.Length));
                if (expectedNative) ++expectedExecutions;
                else Assert.All(output.GetElementsAsFloat(rows * columns), value => Assert.Equal(37f, value));
                Assert.Equal(expectedExecutions, session.ExecutedProjections);
            }
        }
        finally { CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host); Marshal.FreeHGlobal(host); }
    }

    private static byte[] Weights(int type, int columns, int inner)
    {
        int bytesPerBlock = type switch { 12 => 144, 13 => 176, 14 => 210, _ => throw new ArgumentOutOfRangeException(nameof(type)) };
        int blocks = inner / 256;
        var bytes = new byte[checked(columns * blocks * bytesPerBlock)];
        new Random(913 + type).NextBytes(bytes);
        for (int column = 0; column < columns; ++column)
        for (int block = 0; block < blocks; ++block)
        {
            int offset = (column * blocks + block) * bytesPerBlock;
            if (type == 14)
                WriteHalf(bytes, offset + 208, .00390625f * (1 + (column + block) % 3));
            else
            {
                WriteHalf(bytes, offset, .0009765625f * (1 + (column + block) % 7));
                WriteHalf(bytes, offset + 2, .00048828125f * (1 + column % 3));
            }
        }
        return bytes;
    }

    private static void WriteHalf(byte[] bytes, int offset, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((System.Half)value);
        bytes[offset] = (byte)bits;
        bytes[offset + 1] = (byte)(bits >> 8);
    }
}
