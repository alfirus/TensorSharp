using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.MLX;

namespace InferenceWeb.Tests;

public sealed class MlxNativeCallbackTests
{
    [Fact]
    public void NativeCallbackAbiCanBeCompiledAheadOfTime()
    {
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<MlxNative.MlxVectorArray>());
        Assert.Equal(IntPtr.Size, Unsafe.SizeOf<MlxNative.MlxVectorArray>());
        foreach (string name in new[] { "ClosureCallback", "ClosureDestructor", "NoOpDeleter", "CaptureError" })
        {
            MethodInfo method = typeof(MlxNative).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)!;
            var attribute = Assert.IsType<UnmanagedCallersOnlyAttribute>(
                method.GetCustomAttribute<UnmanagedCallersOnlyAttribute>());
            Assert.Equal(new[] { typeof(CallConvCdecl) }, attribute.CallConvs);
        }
    }

    [MlxFact]
    public unsafe void NativeNoCopyDeleterKeepsTheCallersBufferAlive()
    {
        using var allocator = new MlxAllocator();
        float* buffer = (float*)NativeMemory.AlignedAlloc(4096, 4096);
        Assert.True(buffer != null);
        try
        {
            buffer[0] = 1; buffer[1] = -2; buffer[2] = 3; buffer[3] = 4;
            var array = MlxNative.NewArrayFromHostNoCopy((IntPtr)buffer, [4], DType.Float32);
            try { Assert.Equal(new[] { 1f, -2f, 3f, 4f }, Read(array, 4)); }
            finally { MlxNative.FreeArray(array); }
            MlxWorker.Shared.Invoke(() => 0); // Drain the native reference release.
            Assert.Equal(3f, buffer[2]);
            buffer[2] = 9;
            Assert.Equal(9f, buffer[2]);
        }
        finally { NativeMemory.AlignedFree(buffer); }
    }

    [MlxFact]
    public void NativeCompiledClosureSurvivesGcAndReleasesItsPayloadOnce()
    {
        using var allocator = new MlxAllocator();
        var closure = MlxNative.NewClosure(inputs =>
            [MlxNative.Binary(MlxNative.MlxBinaryOp.Add, inputs[0], inputs[0])], shapeless: false);
        var input = NewArray([1f, -2f, 3f]);
        try
        {
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
                var output = MlxNative.ApplyClosure1(closure, input);
                try { Assert.Equal(new[] { 2f, -4f, 6f }, Read(output, 3)); }
                finally { MlxNative.FreeArray(output); }
            }
        }
        finally
        {
            MlxNative.FreeArray(input);
            MlxNative.FreeCompiledClosure(closure);
            MlxNative.FreeCompiledClosure(closure);
            MlxWorker.Shared.Invoke(() => 0);
        }
        Assert.True(closure.Disposed);
        Assert.False(closure.TraceHandle.IsAllocated);
        Assert.Null(closure.Trace);
        Assert.Throws<ObjectDisposedException>(() => MlxNative.ApplyClosure(closure, []));
    }

    [MlxFact]
    public void NativeErrorCallbackPreservesTheDiagnostic()
    {
        using var allocator = new MlxAllocator();
        var input = NewArray([1f, 2f, 3f]);
        try
        {
            var error = Assert.Throws<InvalidOperationException>(() => MlxNative.Reshape(input, [4]));
            Assert.Contains("size", error.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("3", error.Message);
        }
        finally { MlxNative.FreeArray(input); }
    }

    private static unsafe MlxNative.MlxArray NewArray(float[] values)
    {
        fixed (float* pointer = values)
            return MlxNative.NewArrayFromHost((IntPtr)pointer, [values.Length], DType.Float32);
    }

    private static unsafe float[] Read(MlxNative.MlxArray array, int length)
    {
        var result = new float[length];
        fixed (float* pointer = result)
            MlxNative.CopyArrayToHost(array, DType.Float32, (IntPtr)pointer, length * sizeof(float));
        return result;
    }
}
