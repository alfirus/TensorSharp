using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>Exact half-bit checks for the optional owned LTX converter.
/// CUDA absence is a visible skip; missing converter exports on CUDA fail.</summary>
public sealed class Ltx25CudaF16DequantTests
{
    public static IEnumerable<object[]> Cases
    {
        get
        {
            foreach (int type in new[] { 12, 13, 14 })
            foreach (int inner in new[] { 256, 512 })
            foreach (int columns in new[] { 19, 67 })
            foreach (bool edgeScales in new[] { false, true })
                yield return new object[] { type, inner, columns, edgeScales };
        }
    }

    [CudaTheory]
    [MemberData(nameof(Cases))]
    public void BlockwiseConverterMatchesIndependentHalfBitsAndPreservesViewGuards(
        int type, int inner, int columns, bool edgeScales)
    {
        byte[] packed = Weights(type, columns, inner, edgeScales);
        var decoded = new float[checked(inner * columns)];
        GgmlGgufTensorDequant.DequantizeToFloat32(type, packed, 0, decoded, 0, decoded.Length);
        using var allocator = new CudaAllocator();
        IntPtr host = Marshal.AllocHGlobal(packed.Length);
        try
        {
            Marshal.Copy(packed, 0, host, packed.Length);
            using var session = new CudaLtx25ProjectionSession(allocator, requireNative: true, enableF16: true);
            Assert.True(session.F16DequantAvailable);
            using var storage = new Tensor(allocator, DType.Float16, columns + 2, inner);
            Ops.Fill(storage, 37f);
            using var output = storage.Narrow(0, 1, columns);
            Assert.True(session.TryDequantizeF16(output, host, type, inner, columns, packed.Length));
            Assert.Equal(1, session.ExecutedF16Dequantizations);
            ushort[] actual = ReadHalfBits(storage);
            for (int index = 0; index < decoded.Length; ++index)
            {
                ushort expectedBits = BitConverter.HalfToUInt16Bits((System.Half)decoded[index]);
                ushort actualBits = actual[inner + index];
                Assert.True(expectedBits == actualBits,
                    $"type={type},inner={inner},columns={columns},edge={edgeScales},index={index}: expected0x{expectedBits:x4},actual0x{actualBits:x4}");
            }
            for (int i = 0; i < inner; ++i)
            {
                Assert.Equal(BitConverter.HalfToUInt16Bits((System.Half)37f), actual[i]);
                Assert.Equal(BitConverter.HalfToUInt16Bits((System.Half)37f), actual[(columns + 1) * inner + i]);
            }
            Assert.True(session.TryDequantizeF16(output, host, type, inner, columns, packed.Length));
            Assert.Equal(2, session.ExecutedF16Dequantizations);
            Assert.Equal(actual, ReadHalfBits(storage));
        }
        finally { CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host); Marshal.FreeHGlobal(host); }
    }

    [CudaFact]
    public void InvalidConverterInputsRefuseWithoutWritingOrCountingExecution()
    {
        const int inner = 256, columns = 19;
        byte[] packed = Weights(12, columns, inner, false);
        using var allocator = new CudaAllocator();
        IntPtr host = Marshal.AllocHGlobal(packed.Length);
        try
        {
            Marshal.Copy(packed, 0, host, packed.Length);
            using var session = new CudaLtx25ProjectionSession(allocator, requireNative: true, enableF16: true);
            using var output = new Tensor(allocator, DType.Float16, columns, inner);
            using var floatOutput = new Tensor(allocator, DType.Float32, columns, inner);
            using var flatOutput = new Tensor(allocator, DType.Float16, columns * inner);
            using var wideOutput = new Tensor(allocator, DType.Float16, columns, inner * 2);
            using var noncontiguous = wideOutput.Narrow(1, 0, inner);
            Ops.Fill(output, 37f); Ops.Fill(floatOutput, 37f); Ops.Fill(flatOutput, 37f); Ops.Fill(wideOutput, 37f);
            Assert.False(session.TryDequantizeF16(output, IntPtr.Zero, 12, inner, columns, packed.Length));
            Assert.False(session.TryDequantizeF16(output, host, 0, inner, columns, packed.Length));
            Assert.False(session.TryDequantizeF16(output, host, 12, inner - 1, columns, packed.Length));
            Assert.False(session.TryDequantizeF16(output, host, 12, inner, columns + 1, packed.Length));
            Assert.False(session.TryDequantizeF16(output, host, 12, inner, columns, packed.Length - 1));
            Assert.False(session.TryDequantizeF16(floatOutput, host, 12, inner, columns, packed.Length));
            Assert.False(session.TryDequantizeF16(flatOutput, host, 12, inner, columns, packed.Length));
            Assert.False(session.TryDequantizeF16(noncontiguous, host, 12, inner, columns, packed.Length));
            Assert.Equal(0, session.ExecutedF16Dequantizations);
            foreach (Tensor tensor in new[] { output, floatOutput, flatOutput, wideOutput })
                for (long index = 0; index < tensor.ElementCount(); ++index)
                    Assert.Equal(37f, tensor.Storage.GetElementAsFloat(tensor.StorageOffset + index));
            session.Dispose();
            Assert.Throws<ObjectDisposedException>(() => session.TryDequantizeF16(output, host, 12, inner, columns, packed.Length));
        }
        finally { CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host); Marshal.FreeHGlobal(host); }
    }

    private static ushort[] ReadHalfBits(Tensor tensor)
    {
        var bits = new ushort[checked((int)tensor.ElementCount())];
        var pinned = GCHandle.Alloc(bits, GCHandleType.Pinned);
        try { tensor.Storage.CopyFromStorage(pinned.AddrOfPinnedObject(), tensor.StorageOffset, bits.LongLength * sizeof(ushort)); }
        finally { pinned.Free(); }
        return bits;
    }

    private static byte[] Weights(int type, int columns, int inner, bool edgeScales)
    {
        int bytesPerBlock = type switch { 12 => 144, 13 => 176, 14 => 210, _ => throw new ArgumentOutOfRangeException(nameof(type)) };
        int blocks = inner / 256;
        var bytes = new byte[checked(columns * blocks * bytesPerBlock)];
        new Random(1729 + type + inner + columns).NextBytes(bytes);
        for (int column = 0; column < columns; ++column)
        for (int block = 0; block < blocks; ++block)
        {
            int offset = (column * blocks + block) * bytesPerBlock;
            ushort scale = edgeScales ? ((column + block) % 4) switch
            {
                0 => (ushort)0, 1 => (ushort)1, 2 => (ushort)0x8001, _ => BitConverter.HalfToUInt16Bits((System.Half).00390625f),
            } : BitConverter.HalfToUInt16Bits((System.Half)(.0009765625f * (1 + (column + block) % 7)));
            WriteBits(bytes, type == 14 ? offset + 208 : offset, scale);
            if (type != 14) WriteBits(bytes, offset + 2, edgeScales ? scale : BitConverter.HalfToUInt16Bits((System.Half).00048828125f));
        }
        return bytes;
    }

    private static void WriteBits(byte[] bytes, int offset, ushort bits)
    {
        bytes[offset] = (byte)bits;
        bytes[offset + 1] = (byte)(bits >> 8);
    }
}
