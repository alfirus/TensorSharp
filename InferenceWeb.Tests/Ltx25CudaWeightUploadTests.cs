using System.Runtime.InteropServices;
using System.Security.Cryptography;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Models.Direct;

namespace InferenceWeb.Tests;

public sealed class Ltx25CudaWeightUploadTests
{
    [CudaTheory]
    [InlineData((1 << 20) - 4)]
    [InlineData((3 << 20) + 148)]
    [InlineData((16 << 20) + 124)]
    public void ExactSourceBytesCrossPinnedSlotsAndRaggedFinalChunk(int bytes)
    {
        using var allocator = new CudaAllocator();
        using var uploader = new CudaLtx25WeightUploadSession(allocator, 1 << 20);
        var source = new BytesSource(Pattern(bytes + 62));
        IntPtr key = new(0x1234000);
        try
        {
            Assert.True(uploader.Preload(key, source, 31, 0, 1, bytes / 4, bytes));
            Assert.Equal(bytes, uploader.UploadedBytes);
            Assert.Equal(1, uploader.UploadedWeights);
            Assert.Equal(2L << 20, uploader.PinnedBytes);
            Assert.Equal(bytes, source.ReadBytes);
            Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, key, out var device, out int type));
            Assert.Equal(0, type);
            byte[] actual = ReadDevice(device, bytes);
            Assert.Equal(source.Data.AsSpan(31, bytes).ToArray(), actual);
            Assert.Equal(SHA256.HashData(source.Data.AsSpan(31, bytes)), SHA256.HashData(actual));
            long reads = source.ReadBytes;
            Assert.False(uploader.Preload(key, source, 31, 0, 1, bytes / 4, bytes));
            Assert.Equal(reads, source.ReadBytes);
            Assert.Equal(1, uploader.UploadedWeights);
        }
        finally { CudaQuantizedOps.ReleaseQuantizedWeight(allocator, key); }
    }

    [CudaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void IoFailureOrCancellationPublishesNoPartialWeightAndRetryIsExact(bool cancel)
    {
        using var allocator = new CudaAllocator();
        using var uploader = new CudaLtx25WeightUploadSession(allocator, 1 << 20);
        using var cancellation = new CancellationTokenSource();
        var source = new BytesSource(Pattern((3 << 20) + 124));
        source.AfterRead = reads =>
        {
            if (reads != 2) return;
            if (cancel) cancellation.Cancel();
            else throw new EndOfStreamException("Synthetic source ended after one in-flight DMA.");
        };
        IntPtr key = new(0x1235000);
        try
        {
            if (cancel)
                Assert.Throws<OperationCanceledException>(() => uploader.Preload(key, source, 0, 0, 1, source.Data.Length / 4,
                    source.Data.Length, cancellation.Token));
            else
                Assert.Throws<EndOfStreamException>(() => uploader.Preload(key, source, 0, 0, 1, source.Data.Length / 4, source.Data.Length));
            Assert.False(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, key, out _, out _));
            Assert.Equal(0, uploader.UploadedWeights);
            Assert.Equal(0, uploader.UploadedBytes);
            source.AfterRead = null;
            Assert.True(uploader.Preload(key, source, 0, 0, 1, source.Data.Length / 4, source.Data.Length));
            Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, key, out var device, out _));
            Assert.Equal(source.Data, ReadDevice(device, source.Data.Length));
        }
        finally { CudaQuantizedOps.ReleaseQuantizedWeight(allocator, key); }
    }

    [CudaFact]
    public void InvalidSourceRangesCancellationAndCacheConflictsDoNotRead()
    {
        using var allocator = new CudaAllocator();
        using var uploader = new CudaLtx25WeightUploadSession(allocator, 1 << 20);
        var source = new BytesSource(Pattern(4096));
        IntPtr key = new(0x1236000);
        try
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => uploader.Preload(key, source, long.MaxValue - 1, 0, 1, 1024, 4096));
            Assert.Throws<ArgumentOutOfRangeException>(() => uploader.Preload(IntPtr.Zero, source, 0, 0, 1, 1024, 4096));
            Assert.Throws<ArgumentException>(() => uploader.Preload(key, source, 0, 0, 1, 1024, 4095));
            Assert.Throws<ArgumentException>(() => uploader.Preload(key, source, 0, 12, 255, 1, 144));
            using var canceled = new CancellationTokenSource(); canceled.Cancel();
            Assert.Throws<OperationCanceledException>(() => uploader.Preload(key, source, 0, 0, 1, 1024, 4096, canceled.Token));
            Assert.Equal(0, source.ReadBytes);
            Assert.Equal(0, uploader.PinnedBytes);
            Assert.True(uploader.Preload(key, source, 0, 0, 1, 1024, 4096));
            long read = source.ReadBytes;
            Assert.Throws<InvalidOperationException>(() => uploader.Preload(key, source, 0, 0, 2, 512, 4096));
            Assert.Equal(read, source.ReadBytes);
        }
        finally { CudaQuantizedOps.ReleaseQuantizedWeight(allocator, key); }
    }

    [CudaFact]
    public void SessionDisposalReleasesPinnedMemoryWithoutEvictingLeaseOwnedWeight()
    {
        using var allocator = new CudaAllocator();
        var uploader = new CudaLtx25WeightUploadSession(allocator, 1 << 20);
        var source = new BytesSource(Pattern(4096));
        IntPtr key = new(0x1237000);
        try
        {
            Assert.True(uploader.Preload(key, source, 0, 0, 1, 1024, 4096));
            uploader.Dispose(); uploader.Dispose();
            Assert.Equal(0, uploader.PinnedBytes);
            Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, key, out var device, out _));
            Assert.Equal(source.Data, ReadDevice(device, 4096));
            Assert.Throws<ObjectDisposedException>(() => uploader.Preload(key, source, 0, 0, 1, 1024, 4096));
        }
        finally { uploader.Dispose(); CudaQuantizedOps.ReleaseQuantizedWeight(allocator, key); }
    }

    [CudaFact]
    public void PreloadedWeightUsesExistingSharedLeasesAndMatchesMappedProjectionBitwise()
    {
        const int inner = 512, columns = 1025;
        var weights = Enumerable.Range(0, inner * columns).Select(i => MathF.Sin(i * .113f) * .25f).ToArray();
        byte[] bytes = new byte[weights.Length * sizeof(float)]; Buffer.BlockCopy(weights, 0, bytes, 0, bytes.Length);
        IntPtr host = Marshal.AllocHGlobal(bytes.Length);
        using var allocator = new CudaAllocator();
        using var first = new DirectContext(allocator);
        using var second = new DirectContext(allocator);
        using var uploader = new CudaLtx25WeightUploadSession(allocator, 1 << 20);
        try
        {
            Marshal.Copy(bytes, 0, host, bytes.Length);
            var a = DirectLinear.FromMappedH3Dense(first, host, 0, inner, columns, bytes.Length, null);
            var b = DirectLinear.FromMappedH3Dense(second, host, 0, inner, columns, bytes.Length, null);
            var inputValues = new float[1, inner];
            for (int i = 0; i < inner; ++i) inputValues[0, i] = MathF.Cos(i * .41f);
            using var input = Tensor.FromArray(allocator, inputValues);
            using var mapped = a.Forward(input);
            float[] expected = mapped.GetElementsAsFloat(columns);
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            var source = new BytesSource(bytes);
            Assert.True(uploader.Preload(host, source, 0, 0, inner, columns, bytes.Length));
            using var loaded = a.Forward(input);
            Assert.Equal(expected, loaded.GetElementsAsFloat(columns));
            first.Dispose();
            Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, host, out _, out _));
            using var peer = b.Forward(input);
            Assert.Equal(expected, peer.GetElementsAsFloat(columns));
            second.ReleaseDeviceResidency();
            Assert.False(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, host, out _, out _));
            Assert.True(uploader.Preload(host, source, 0, 0, inner, columns, bytes.Length));
            using var reloaded = b.Forward(input);
            Assert.Equal(expected, reloaded.GetElementsAsFloat(columns));
            second.Dispose();
            Assert.False(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, host, out _, out _));
        }
        finally { first.Dispose(); second.Dispose(); Marshal.FreeHGlobal(host); }
    }

    private static byte[] Pattern(int bytes) => Enumerable.Range(0, bytes).Select(i => (byte)((i * 43 + i / 257) & 255)).ToArray();
    private static unsafe byte[] ReadDevice(IntPtr pointer, int bytes)
    {
        byte[] result = new byte[bytes];
        fixed (byte* output = result)
            Assert.Equal(0, CudaDriverApi.cuMemcpyDtoH((IntPtr)output, pointer, new UIntPtr((ulong)bytes)));
        return result;
    }

    private sealed class BytesSource(byte[] data) : ICudaWeightSource
    {
        internal byte[] Data { get; } = data;
        internal long ReadBytes { get; private set; }
        internal int Reads { get; private set; }
        internal Action<int> AfterRead { get; set; }
        public void Read(long offset, IntPtr destination, long bytes)
        {
            if (offset < 0 || bytes < 0 || offset > Data.LongLength - bytes) throw new EndOfStreamException();
            Marshal.Copy(Data, checked((int)offset), destination, checked((int)bytes));
            ReadBytes += bytes;
            AfterRead?.Invoke(++Reads);
        }
    }
}
