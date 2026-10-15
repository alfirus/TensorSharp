using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using TensorSharp.Cuda;
using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

public sealed class Ltx25CudaFilePrefaultTests
{
    [Theory]
    [InlineData(1)] [InlineData(8)]
    public void DirectSourceReadsDisjointPinnedSlicesAndRaggedTailExactly(int workers)
    {
        string path = Path.GetTempFileName();
        const int offset = 31, length = (8 << 20) + 123;
        byte[] expected = new byte[offset + length + 17];
        for (int i = 0; i < expected.Length; ++i) expected[i] = (byte)(i * 43 + i / 127);
        IntPtr destination = Marshal.AllocHGlobal(length);
        try
        {
            File.WriteAllBytes(path, expected);
            using var source = new Ltx25CudaFilePrefault(path, checkResident: false, workers: workers);
            ((ICudaWeightSource)source).Read(offset, destination, length);
            byte[] actual = new byte[length]; Marshal.Copy(destination, actual, 0, length);
            Assert.Equal(expected.AsSpan(offset, length).ToArray(), actual);
            Assert.Equal(length, source.ReadBytes);
            Assert.Null(source.Failure);
        }
        finally { Marshal.FreeHGlobal(destination); File.Delete(path); }
    }

    [Theory]
    [InlineData(1)] [InlineData(8)]
    public void DirectSourceEofJoinsAllReadersAndPreservesSuccessfulPartialCounts(int workers)
    {
        string path = Path.GetTempFileName();
        const int available = (8 << 20) + 4, requested = (9 << 20) + 16;
        IntPtr destination = Marshal.AllocHGlobal(requested);
        try
        {
            using (var file = File.OpenWrite(path)) file.SetLength(available);
            using var source = new Ltx25CudaFilePrefault(path, checkResident: false, workers: workers);
            if (workers == 1)
                Assert.Throws<EndOfStreamException>(() => ((ICudaWeightSource)source).Read(0, destination, requested));
            else
            {
                var error = Assert.Throws<AggregateException>(() => ((ICudaWeightSource)source).Read(0, destination, requested));
                Assert.All(error.Flatten().InnerExceptions, inner => Assert.IsType<EndOfStreamException>(inner));
            }
            // Parallel.For may stop scheduling the final partial slice after a
            // sibling reaches EOF. Completed reads must still be counted, and
            // every started worker must have joined before Read returns.
            if (workers == 1) Assert.Equal(available, source.ReadBytes);
            else Assert.InRange(source.ReadBytes, 8 << 20, available);
            long joinedCount = source.ReadBytes;
            source.Dispose();
            Assert.Equal(joinedCount, source.ReadBytes);
            Assert.Throws<ObjectDisposedException>(() => ((ICudaWeightSource)source).Read(0, destination, 16));
        }
        finally { Marshal.FreeHGlobal(destination); File.Delete(path); }
    }

    [Theory]
    [InlineData(1)] [InlineData(8)]
    public unsafe void SelectedUnalignedRangeCrossesBufferBoundaryWithoutReadingTheWholeFile(int workers)
    {
        string path = Path.GetTempFileName();
        try
        {
            using (var stream = File.OpenWrite(path)) stream.SetLength(12 << 20);
            using var file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            using var view = file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* pointer = null; view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            try
            {
                using var prefetch = new Ltx25CudaFilePrefault(path, checkResident: false, workers: workers);
                const int offset = 31, length = (8 << 20) + 123;
                prefetch.Warm((IntPtr)(pointer + offset), offset, length);
                Assert.Null(prefetch.Failure);
                Assert.Equal(length, prefetch.ReadBytes);
                Assert.Equal(0, prefetch.ResidentBytes);
                Assert.Equal(1, prefetch.Ranges);
                Assert.Equal(workers, prefetch.Workers);
                Assert.True(prefetch.Milliseconds >= 0);
            }
            finally { view.SafeMemoryMappedViewHandle.ReleasePointer(); }
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData(1)] [InlineData(8)]
    public unsafe void TruncatedRangeRecordsPartialReadAndDisablesOptionalWarming(int workers)
    {
        string path = Path.GetTempFileName();
        try
        {
            int length = (workers == 8 ? 2 << 20 : 8 << 20) + 4;
            using (var stream = File.OpenWrite(path)) stream.SetLength(length);
            using var file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            using var view = file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* pointer = null; view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            try
            {
                using var prefetch = new Ltx25CudaFilePrefault(path, checkResident: false, workers: workers);
                prefetch.Warm((IntPtr)pointer, 0, length + 12);
                Assert.StartsWith(workers == 8 ? nameof(AggregateException) : nameof(EndOfStreamException), prefetch.Failure);
                if (workers == 8) Assert.InRange(prefetch.ReadBytes, 4, length);
                else Assert.Equal(length, prefetch.ReadBytes);
                Assert.Equal(0, prefetch.Ranges);
                long read = prefetch.ReadBytes;
                prefetch.Warm((IntPtr)pointer, 0, 16);
                Assert.Equal(read, prefetch.ReadBytes);
            }
            finally { view.SafeMemoryMappedViewHandle.ReleasePointer(); }
        }
        finally { File.Delete(path); }
    }

}
