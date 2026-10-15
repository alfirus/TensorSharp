using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class GgmlClampTests
{
    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuClampSupportsInPlaceStorageViewsAndLargeFloat32Values() => Verify(GgmlBackendType.Cpu);
    [GgmlFact(BackendType.GgmlMetal)]
    public void MetalClampSupportsInPlaceStorageViewsAndLargeFloat32Values() => Verify(GgmlBackendType.Metal);
    [GgmlFact(BackendType.GgmlCuda)]
    public void CudaClampSupportsInPlaceStorageViewsAndLargeFloat32Values() => Verify(GgmlBackendType.Cuda);

    private static void Verify(GgmlBackendType backend)
    {
        var allocator = new GgmlAllocator(new GgmlContext([0], backend), 0);
        try
        {
            var data = new float[,] {{37,37,37,37,37}, {-80000,-.125f,0,.125f,80000}, {37,37,37,37,37}};
            using var storage = Tensor.FromArray(allocator, data);
            using var row = storage.Narrow(0, 1, 1);
            Ops.Clamp(row, row, -10000, 10000);
            Assert.Equal(new float[] {37,37,37,37,37,-10000,-.125f,0,.125f,10000,37,37,37,37,37},
                storage.GetElementsAsFloat(15));
            using var separate = Ops.Clamp(null, row, -1, 1);
            Assert.Equal(new float[] {-1,-.125f,0,.125f,1}, separate.GetElementsAsFloat(5));
            Assert.Equal(new float[] {-10000,-.125f,0,.125f,10000}, row.GetElementsAsFloat(5));
        }
        finally
        {
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }
}
