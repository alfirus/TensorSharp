using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class Ltx25CudaPrefillProjectionTests
{
    [CudaTheory]
    [InlineData(12)]
    [InlineData(13)]
    [InlineData(14)]
    public void OwnedConverterKeepsExistingHalfGemmBitsAndRebuildsItsWorkspace(int type)
    {
        using var allocator = new CudaAllocator();
        using var session = new CudaLtx25ProjectionSession(allocator, requireNative: true, enableF16: true);
        var shapes = new[] { (32, 256, 19), (35, 512, 67), (32, 256, 19), (35, 512, 67) };
        long calls = 0;
        foreach (var (rows, inner, columns) in shapes)
        {
            if (calls == 3)
            {
                Assert.True(session.F16ScratchBytes > 0);
                session.ReleaseWorkspace();
                Assert.Equal(0, session.F16ScratchBytes);
            }
            byte[] packed = Weights(type, inner, columns);
            IntPtr host = Marshal.AllocHGlobal(packed.Length);
            try
            {
                Marshal.Copy(packed, 0, host, packed.Length);
                var values = new float[rows + 2, inner];
                for (int row = 0; row < rows + 2; ++row)
                for (int k = 0; k < inner; ++k)
                    values[row, k] = MathF.Sin(row * .117f + k * .0331f) + (k % 13 - 6) * .015625f;
                using var inputBase = Tensor.FromArray(allocator, values);
                using var outputBase = new Tensor(allocator, DType.Float32, rows + 2, columns);
                using var baseline = new Tensor(allocator, DType.Float32, rows, columns);
                Ops.Fill(outputBase, 37f);
                using var input = inputBase.Narrow(0, 1, rows);
                using var output = outputBase.Narrow(0, 1, rows);
                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(baseline, input,
                    host, host, type, inner, columns, packed.Length));
                Assert.True(session.TryPrefillLinear(output, input, host, type, inner, columns, packed.Length));
                ++calls;
                Assert.Equal(calls, session.ExecutedF16Projections);
                Assert.Equal(calls, session.ExecutedF16Dequantizations);
                Assert.Equal(baseline.GetElementsAsFloat(rows * columns).Select(BitConverter.SingleToInt32Bits),
                    output.GetElementsAsFloat(rows * columns).Select(BitConverter.SingleToInt32Bits));
                float[] guards = outputBase.GetElementsAsFloat((rows + 2) * columns);
                for (int c = 0; c < columns; ++c)
                {
                    Assert.Equal(37f, guards[c]);
                    Assert.Equal(37f, guards[(rows + 1) * columns + c]);
                }
            }
            finally { CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host); Marshal.FreeHGlobal(host); }
        }
        session.Dispose();
        Assert.Equal(0, session.F16ScratchBytes);
    }

    [CudaFact]
    public void IneligiblePrefillAndAliasedStorageRefuseBeforeDispatch()
    {
        const int inner = 256;
        using var allocator = new CudaAllocator();
        using var session = new CudaLtx25ProjectionSession(allocator, requireNative: true, enableF16: true);
        byte[] packed = Weights(12, inner, inner);
        IntPtr host = Marshal.AllocHGlobal(packed.Length);
        try
        {
            Marshal.Copy(packed, 0, host, packed.Length);
            using var input = new Tensor(allocator, DType.Float32, 32, inner);
            using var output = new Tensor(allocator, DType.Float32, 32, inner);
            using var smallInput = input.Narrow(0, 0, 2);
            using var smallOutput = output.Narrow(0, 0, 2);
            Ops.Fill(input, 1f); Ops.Fill(output, 37f);
            Assert.False(session.TryPrefillLinear(smallOutput, smallInput, host, 12, inner, inner, packed.Length));
            Assert.False(session.TryPrefillLinear(output, input, host, 12, inner, inner, packed.Length - 1));
            Assert.False(session.TryPrefillLinear(input, input, host, 12, inner, inner, packed.Length));
            Assert.Equal(0, session.ExecutedF16Projections);
            Assert.Equal(0, session.ExecutedF16Dequantizations);
            Assert.Equal(0, session.F16ScratchBytes);
            Assert.All(output.GetElementsAsFloat(32 * inner), value => Assert.Equal(37f, value));
        }
        finally { CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host); Marshal.FreeHGlobal(host); }
    }

    private static byte[] Weights(int type, int inner, int columns)
    {
        int size = type switch { 12 => 144, 13 => 176, 14 => 210, _ => throw new ArgumentOutOfRangeException(nameof(type)) };
        var packed = new byte[inner / 256 * columns * size];
        new Random(1231 + type).NextBytes(packed);
        for (int block = 0; block < inner / 256 * columns; ++block)
        {
            int offset = block * size;
            WriteHalf(packed, offset + (type == 14 ? 208 : 0), .0009765625f * (1 + block % 7));
            if (type != 14) WriteHalf(packed, offset + 2, .00048828125f);
        }
        return packed;
    }

    private static void WriteHalf(byte[] bytes, int offset, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((System.Half)value);
        bytes[offset] = (byte)bits; bytes[offset + 1] = (byte)(bits >> 8);
    }
}
