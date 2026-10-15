using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models.Direct;
using System.Runtime.InteropServices;
using Half = System.Half;
using Xunit.Abstractions;
using TensorSharp.MLX;
using TensorSharp.Runtime;
using System.Reflection;
using System.Collections;

namespace InferenceWeb.Tests;

public sealed class DirectContextLifetimeTests
{
    private readonly ITestOutputHelper _output;
    public DirectContextLifetimeTests(ITestOutputHelper output) => _output = output;
    [CudaFact]
    public void SharedMappedWeightSurvivesPeerEvictionAndDisposalThenReloads()
    {
        using var allocator = new CudaAllocator(0);
        using var first = new DirectContext(allocator);
        using var second = new DirectContext(allocator);
        Half[] weights = Array.ConvertAll(new float[] { 2, -1, 4, 3, 1, -2, 3, -4 }, value => (Half)value);
        IntPtr host = Marshal.AllocHGlobal(weights.Length * sizeof(ushort));
        try
        {
            for (int i = 0; i < weights.Length; ++i)
                Marshal.WriteInt16(host, i * sizeof(ushort), unchecked((short)BitConverter.HalfToUInt16Bits(weights[i])));
            var firstLinear = DirectLinear.FromMappedH3Dense(first, host, 1, 4, 2, weights.Length * sizeof(ushort), null);
            var secondLinear = DirectLinear.FromMappedH3Dense(second, host, 1, 4, 2, weights.Length * sizeof(ushort), null);
            using var input = Tensor.FromArray(allocator, new float[,] { { 120000, -90000, 80000, -70000 } });
            using var warm = secondLinear.Forward(input);
            float[] expected = warm.GetElementsAsFloat(2);
            Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, host, out var sharedPointer, out _));
            first.ReleaseDeviceResidency();
            first.Dispose();
            Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, host, out var afterPeerDisposal, out _));
            Assert.Equal(sharedPointer, afterPeerDisposal);
            _output.WriteLine($"Shared CUDA key retained across peer eviction/disposal: pointer=0x{sharedPointer.ToInt64():x}");
            using var afterPeer = secondLinear.Forward(input);
            Assert.Equal(expected, afterPeer.GetElementsAsFloat(2));
            second.ReleaseDeviceResidency();
            Assert.False(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, host, out _, out _));
            _output.WriteLine("Final owner's residency release removed the shared key; reload follows.");
            using var reloaded = secondLinear.Forward(input);
            Assert.Equal(expected, reloaded.GetElementsAsFloat(2));
            Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, host, out _, out _));
            second.Dispose();
            Assert.False(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, host, out _, out _));
            Assert.Throws<ObjectDisposedException>(() => secondLinear.Forward(input));
        }
        finally
        {
            first.Dispose(); second.Dispose();
            Marshal.FreeHGlobal(host);
        }
    }

    [CudaFact]
    public void RepeatedContextDisposalReclaimsActualWeightArenaMemoryOnSharedAllocator()
    {
        using var allocator = new CudaAllocator(0);
        float[] weights = Enumerable.Range(0, 32 * 16).Select(index => (index % 11 - 5) / 4f).ToArray();
        float[,] inputs = { { 1, 2, 3, 4, -1, -2, -3, -4, 2, 3, 4, 5, -2, -3, -4, -5,
            3, 4, 5, 6, -3, -4, -5, -6, 4, 5, 6, 7, -4, -5, -6, -7 } };
        void RunContext()
        {
            using var context = new DirectContext(allocator);
            var linear = DirectLinear.FromFloats(context, weights, 32, 16, null);
            using var input = Tensor.FromArray(allocator, inputs);
            using var output = linear.Forward(input);
            float[] actual = output.GetElementsAsFloat(16);
            for (int column = 0; column < 16; ++column)
            {
                double expected = 0;
                for (int dimension = 0; dimension < 32; ++dimension)
                    expected += inputs[0, dimension] * weights[column * 32 + dimension];
                Assert.Equal((float)expected, actual[column]);
            }
        }
        // Initialize CUDA modules/cuBLAS/workspace before taking the driver
        // baseline; metadata disappearance alone cannot prove VRAM reclamation.
        RunContext();
        // Clear the warmup's arena once, including on the pre-fix implementation.
        // Each subsequent disposal must reclaim its own slab without this reset.
        CudaQuantizedOps.ReleaseArena(allocator);
        long baseline = allocator.GetMemoryInfo().freeBytes;
        _output.WriteLine($"CUDA driver free-memory baseline after warmup: {baseline} bytes");
        for (int iteration = 0; iteration < 8; ++iteration)
        {
            RunContext();
            long free = allocator.GetMemoryInfo().freeBytes;
            _output.WriteLine($"Context reload {iteration + 1}/8: free={free} bytes, baseline-minus-free={baseline - free} bytes");
            Assert.True(free >= baseline - (64L << 20),
                $"After context {iteration + 1}, CUDA free memory dropped by {baseline - free} bytes; released weight slabs must be reclaimed.");
        }
    }

    [CudaFact]
    public void DisposingMappedH3ContextPreservesOtherContextsResidentHalfWeights()
    {
        using var allocator=new CudaAllocator(0);
        using var first=new DirectContext(allocator);
        using var second=new DirectContext(allocator);
        Half[] firstWeights=Array.ConvertAll(new float[]{1,2,3,4,-4,-3,-2,-1},value => (Half)value);
        Half[] secondWeights=Array.ConvertAll(new float[]{2,-1,4,3,1,-2,3,-4},value => (Half)value);
        IntPtr firstHost=Marshal.AllocHGlobal(firstWeights.Length*sizeof(ushort));
        IntPtr secondHost=Marshal.AllocHGlobal(secondWeights.Length*sizeof(ushort));
        try
        {
            for(int i=0;i<firstWeights.Length;i++)
            {
                Marshal.WriteInt16(firstHost,i*sizeof(ushort),unchecked((short)BitConverter.HalfToUInt16Bits(firstWeights[i])));
                Marshal.WriteInt16(secondHost,i*sizeof(ushort),unchecked((short)BitConverter.HalfToUInt16Bits(secondWeights[i])));
            }
            var firstLinear=DirectLinear.FromMappedH3Dense(first,firstHost,1,4,2,firstWeights.Length*sizeof(ushort),null);
            var secondLinear=DirectLinear.FromMappedH3Dense(second,secondHost,1,4,2,secondWeights.Length*sizeof(ushort),null);
            using var input=Tensor.FromArray(allocator,new float[,]{{120000,-90000,80000,-70000},{-80000,70000,-60000,50000}});
            using var firstResult=firstLinear.Forward(input);
            using var warmResult=secondLinear.Forward(input);
            float[] expected=warmResult.GetElementsAsFloat(4);
            Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator,firstHost,out _,out int firstType));
            Assert.Equal(1,firstType);
            Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator,secondHost,out var residentBefore,out int secondType));
            Assert.Equal(1,secondType);
            first.Dispose();
            Assert.False(CudaQuantizedOps.TryGetResidentDevicePtr(allocator,firstHost,out _,out _));
            Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator,secondHost,out var residentAfter,out _));
            Assert.Equal(residentBefore,residentAfter);
            using var resultAfter=secondLinear.Forward(input);
            Assert.Equal(expected,resultAfter.GetElementsAsFloat(4));
            second.Dispose();
            Assert.False(CudaQuantizedOps.TryGetResidentDevicePtr(allocator,secondHost,out _,out _));
        }
        finally
        {
            first.Dispose(); second.Dispose();
            Marshal.FreeHGlobal(firstHost); Marshal.FreeHGlobal(secondHost);
        }
    }

    [CudaFact]
    public void DisposingOneContextPreservesOtherContextsResidentWeightsAndWarmResults()
    {
        using var allocator = new CudaAllocator(0);
        using var first = new DirectContext(allocator);
        using var second = new DirectContext(allocator);
        var firstLinear = DirectLinear.FromFloats(first, [1, 2, 3, 4, -4, -3, -2, -1], 4, 2, null);
        var secondLinear = DirectLinear.FromFloats(second, [2, -1, 4, 3, 1, -2, 3, -4], 4, 2, null);
        using var input = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3, 4 }, { -4, 3, -2, 1 } });
        using var firstResult = firstLinear.Forward(input);
        using var warmResult = secondLinear.Forward(input);
        float[] expected = warmResult.GetElementsAsFloat(4);
        Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, firstLinear.CacheKey, out _, out _));
        Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, secondLinear.CacheKey, out var residentBefore, out _));
        first.Dispose();
        Assert.False(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, firstLinear.CacheKey, out _, out _));
        Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, secondLinear.CacheKey, out var residentAfter, out _));
        Assert.Equal(residentBefore, residentAfter);
        using var resultAfter = secondLinear.Forward(input);
        Assert.Equal(expected, resultAfter.GetElementsAsFloat(4));
        second.Dispose();
        Assert.False(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, secondLinear.CacheKey, out _, out _));
    }
}

public sealed class DirectContextMlxLifetimeTests
{
    [MlxFact]
    public void DeviceGlobalMappedCacheSurvivesPeerContextOnAnotherAllocatorAndReloads()
    {
        string path = Path.Combine(Path.GetTempPath(), "ts-lifetime-" + Guid.NewGuid().ToString("N") + ".gguf");
        // Two exact Q8_0 rows exercise the compressed MLX cache instead of the
        // separately owned dense-tensor fallback. Both contexts map the same key.
        using (var writer = new BinaryWriter(File.Create(path)))
        {
            writer.Write(0x46554747U); writer.Write(3U); writer.Write(1UL); writer.Write(0UL);
            writer.Write(6UL); writer.Write(System.Text.Encoding.UTF8.GetBytes("weight"));
            writer.Write(2U); writer.Write(32UL); writer.Write(2UL);
            writer.Write((uint)GgmlTensorType.Q8_0); writer.Write(0UL);
            while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
            for (int row = 0; row < 2; ++row)
            {
                writer.Write(BitConverter.HalfToUInt16Bits((Half)0.5));
                for (int column = 0; column < 32; ++column)
                    writer.Write(unchecked((byte)(sbyte)(column % 9 - 4 + row)));
            }
        }
        try
        {
            using var gguf = new GgufFile(path);
            using var firstAllocator = new MlxAllocator();
            using var secondAllocator = new MlxAllocator();
            using var first = new DirectContext(firstAllocator);
            using var second = new DirectContext(secondAllocator);
            // Keep the mapped byte count/key unchanged but reject a per-row
            // Q8 layout during preload. A failed descriptor must not acquire an
            // owner lease that prevents the later final eviction.
            GgufTensorInfo info = gguf.Tensors["weight"];
            ulong[] originalShape = info.Shape;
            info.Shape = [16UL, 4UL];
            try { Assert.Throws<NotSupportedException>(() => DirectLinear.FromGguf(first, gguf, "weight", null, false)); }
            finally { info.Shape = originalShape; }
            var firstLinear = DirectLinear.FromGguf(first, gguf, "weight", null, false);
            var secondLinear = DirectLinear.FromGguf(second, gguf, "weight", null, false);
            Assert.Equal(firstLinear.CacheKey, secondLinear.CacheKey);
            float[,] inputs = { { 1, 2, 3, 4, -1, -2, -3, -4, 2, 3, 4, 5, -2, -3, -4, -5,
                3, 4, 5, 6, -3, -4, -5, -6, 4, 5, 6, 7, -4, -5, -6, -6 } };
            using var input = Tensor.FromArray(secondAllocator, inputs);
            using var warm = secondLinear.Forward(input);
            float[] expected = warm.GetElementsAsFloat(2);
            for (int row = 0; row < 2; ++row)
            {
                float scalar = 0;
                for (int column = 0; column < 32; ++column)
                    scalar += inputs[0, column] * 0.5f * (column % 9 - 4 + row);
                Assert.Equal(scalar, expected[row]);
            }
            IntPtr cached = CachedMlxWeight(secondAllocator.DeviceId, secondLinear.CacheKey);
            Assert.NotEqual(IntPtr.Zero, cached);
            first.ReleaseDeviceResidency(); first.Dispose();
            Assert.Equal(cached, CachedMlxWeight(secondAllocator.DeviceId, secondLinear.CacheKey));
            using var afterPeer = secondLinear.Forward(input);
            Assert.Equal(expected, afterPeer.GetElementsAsFloat(2));
            second.ReleaseDeviceResidency();
            Assert.Equal(IntPtr.Zero, CachedMlxWeight(secondAllocator.DeviceId, secondLinear.CacheKey));
            using var reload = secondLinear.Forward(input);
            Assert.Equal(expected, reload.GetElementsAsFloat(2));
            Assert.NotEqual(IntPtr.Zero, CachedMlxWeight(secondAllocator.DeviceId, secondLinear.CacheKey));
            second.Dispose();
            Assert.Equal(IntPtr.Zero, CachedMlxWeight(secondAllocator.DeviceId, secondLinear.CacheKey));
            Assert.Throws<ObjectDisposedException>(() => secondLinear.Forward(input));
        }
        finally { File.Delete(path); }
    }

    private static IntPtr CachedMlxWeight(int device, IntPtr host)
    {
        // Inspect real cache identity without adding a production-only test API.
        const BindingFlags flags = BindingFlags.Static | BindingFlags.NonPublic;
        Type type = typeof(MlxQuantizedOps);
        object sync = type.GetField("Sync", flags)!.GetValue(null)!;
        var cache = (IDictionary)type.GetField("Cache", flags)!.GetValue(null)!;
        Type keyType = type.GetNestedType("CacheKey", BindingFlags.NonPublic)!;
        object key = Activator.CreateInstance(keyType, device, host)!;
        lock (sync)
        {
            if (!cache.Contains(key)) return IntPtr.Zero;
            object node = cache[key]!;
            object entry = node.GetType().GetProperty("Value")!.GetValue(node)!;
            return ((MlxNative.MlxArray)entry.GetType().GetField("Weight")!.GetValue(entry)!).Ctx;
        }
    }
}
