using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;
using TensorSharp.Cuda;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>Bounded file-cache reads before CUDA uploads of mapped weights.
/// One handle stays open so warming never reopens the model between matrices.</summary>
internal sealed partial class Ltx25CudaFilePrefault : IDisposable, ICudaWeightSource
{
    private const int BufferBytes = 8 << 20;
    private readonly SafeFileHandle _handle;
    private delegate int ReadAtOffset(long offset, Span<byte> destination);
    private ReadAtOffset _read;
    private bool _disposed;
    private readonly byte[] _buffer;
    private readonly byte[] _residency;
    private readonly bool _checkResident;
    private readonly ParallelOptions _parallel;
    private long _readBytes;
    private CudaLtx25WeightUploadSession _uploader;
    internal long ReadBytes => Interlocked.Read(ref _readBytes);
    internal int Workers { get; }
    internal long ResidentBytes { get; private set; }
    internal long Ranges { get; private set; }
    internal double Milliseconds { get; private set; }
    internal string Failure { get; private set; }
    internal long SourceUploadedBytes => _uploader?.UploadedBytes ?? 0;
    internal long SourceUploadedWeights => _uploader?.UploadedWeights ?? 0;
    internal long SourcePinnedBytes => _uploader?.PinnedBytes ?? 0;
    internal double SourceUploadMilliseconds => _uploader?.UploadMilliseconds ?? 0;

    internal Ltx25CudaFilePrefault(GgufFile file) : this((string)null) => _read = file.ReadMappedBytes;
    internal Ltx25CudaFilePrefault(SafetensorsFile file) : this((string)null) => _read = file.ReadMappedBytes;

    internal Ltx25CudaFilePrefault(string path, bool checkResident = true, int? workers = null)
    {
        int requested = workers ?? (int.TryParse(Environment.GetEnvironmentVariable("TS_LTX25_CUDA_PREFAULT_WORKERS"), out int configured)
            ? configured : OperatingSystem.IsLinux() ? 8 : 1);
        Workers = Math.Clamp(requested, 1, 8);
        _parallel = new ParallelOptions { MaxDegreeOfParallelism = Workers };
        _checkResident = checkResident && OperatingSystem.IsLinux();
        try
        {
            if (path != null)
            {
                _handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.SequentialScan);
                _read = (offset, destination) => RandomAccess.Read(_handle, destination, offset);
            }
            _buffer = new byte[BufferBytes];
            _residency = new byte[BufferBytes / Environment.SystemPageSize + 2];
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        { Failure = error.GetType().Name + ": " + error.Message; }
    }

    internal void Prepare(CudaAllocator allocator, IntPtr mappedPointer, long fileOffset, long length,
        int type, long inner, long columns, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        // Keep the established mapped fast path for resident pages, small
        // tensors, other platforms and the default policy until the controlled
        // source-upload candidate has passed full-process benchmarks.
        if (_read == null || Failure != null || !_checkResident || length < (1 << 20) ||
            Environment.GetEnvironmentVariable("TS_LTX25_CUDA_SOURCE_UPLOAD") != "1")
        { Warm(mappedPointer, fileOffset, length); return; }
        if (mappedPointer == IntPtr.Zero || fileOffset < 0 || length < 0 || fileOffset > long.MaxValue - length)
            throw new ArgumentOutOfRangeException(nameof(length));
        cancellationToken.ThrowIfCancellationRequested();
        if (CudaQuantizedOps.TryGetResidentDevicePtr(allocator, mappedPointer, out _, out _)) return;
        long started = Stopwatch.GetTimestamp();
        bool? resident = true;
        for (long offset = 0; offset < length;)
        {
            int count = (int)Math.Min(BufferBytes, length - offset);
            resident = ResidentState(new IntPtr(checked(mappedPointer.ToInt64() + offset)), count);
            if (resident != true) break;
            offset += count;
        }
        if (resident == null) { Warm(mappedPointer, fileOffset, length); return; }
        try
        {
            if (resident == true) ResidentBytes += length;
            else
            {
                _uploader ??= new CudaLtx25WeightUploadSession(allocator);
                _uploader.Preload(mappedPointer, this, fileOffset, type, inner, columns, length, cancellationToken);
            }
            ++Ranges;
        }
        finally { Milliseconds += Stopwatch.GetElapsedTime(started).TotalMilliseconds; }
    }

    public void Read(long offset, IntPtr destination, long bytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (offset < 0 || bytes < 0 || offset > long.MaxValue - bytes || destination == IntPtr.Zero)
            throw new ArgumentOutOfRangeException(nameof(bytes));
        for (long done = 0; done < bytes;)
        {
            int count = (int)Math.Min(BufferBytes, bytes - done);
            var pointer = new IntPtr(checked(destination.ToInt64() + done));
            long position = checked(offset + done);
            if (Workers == 1 || count <= (1 << 20))
                ReadDestination(position, pointer, count);
            else
            {
                // Match the established NFS warming concurrency, but write
                // directly to disjoint slices of this pinned slot. Join before
                // the uploader can submit any DMA reading the slot.
                Parallel.For(0, (count + (1 << 20) - 1) >> 20, _parallel, slice =>
                {
                    int begin = slice << 20, length = Math.Min(1 << 20, count - begin);
                    ReadDestination(checked(position + begin), new IntPtr(checked(pointer.ToInt64() + begin)), length);
                });
            }
            done += count;
        }
    }

    private unsafe void ReadDestination(long offset, IntPtr destination, int bytes)
    {
        for (int done = 0; done < bytes;)
        {
            var span = new Span<byte>((void*)new IntPtr(checked(destination.ToInt64() + done)), bytes - done);
            int read = _read(checked(offset + done), span);
            if (read == 0) throw new EndOfStreamException("LTX checkpoint changed during source-aware weight upload.");
            done += read;
            Interlocked.Add(ref _readBytes, read);
        }
    }

    internal void Warm(IntPtr mappedPointer, long fileOffset, long length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_read == null || Failure != null || length == 0) return;
        if (mappedPointer == IntPtr.Zero || fileOffset < 0 || length < 0)
            throw new ArgumentOutOfRangeException(nameof(length));
        long started = Stopwatch.GetTimestamp();
        try
        {
            for (long offset = 0; offset < length;)
            {
                int count = (int)Math.Min(BufferBytes, length - offset);
                var pointer = new IntPtr(checked(mappedPointer.ToInt64() + offset));
                if (_checkResident && ResidentState(pointer, count) == true)
                    ResidentBytes += count;
                else
                {
                    if (Workers == 1 || count <= (1 << 20))
                        ReadSlice(0, count, checked(fileOffset + offset));
                    else
                    {
                        // Every worker owns a disjoint 1 MiB slice of the same
                        // 8 MiB buffer. Join before that buffer is reused.
                        long position = checked(fileOffset + offset);
                        Parallel.For(0, (count + (1 << 20) - 1) >> 20, _parallel, slice =>
                        {
                            int begin = slice << 20, bytes = Math.Min(1 << 20, count - begin);
                            ReadSlice(begin, bytes, checked(position + begin));
                        });
                    }
                }
                offset += count;
            }
            ++Ranges;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException ||
            error is AggregateException aggregate && aggregate.Flatten().InnerExceptions.All(inner => inner is IOException or UnauthorizedAccessException))
        {
            // Warming is optional. Actual tensor reads and uploads retain their
            // strict validation; no inference operation falls back to the CPU.
            Failure = error.GetType().Name + ": " + error.Message;
        }
        finally { Milliseconds += Stopwatch.GetElapsedTime(started).TotalMilliseconds; }
    }

    private void ReadSlice(int bufferOffset, int length, long fileOffset)
    {
        int done = 0;
        while (done < length)
        {
            int read = _read(checked(fileOffset + done), _buffer.AsSpan(bufferOffset + done, length - done));
            if (read == 0) throw new EndOfStreamException("LTX checkpoint changed during mapped-weight warmup.");
            done += read;
            // Preserve successful partial reads even when a different worker
            // reaches EOF or fails. Parallel.For joins all started readers.
            Interlocked.Add(ref _readBytes, read);
        }
    }

    private unsafe bool? ResidentState(IntPtr pointer, int length)
    {
        long address = pointer.ToInt64(), page = Environment.SystemPageSize;
        long aligned = address - address % page, bytes = address - aligned + length;
        int pages = checked((int)((bytes + page - 1) / page));
        try
        {
            fixed (byte* vector = _residency)
            {
                if (mincore((void*)aligned, (nuint)bytes, vector) != 0) return null;
                for (int i = 0; i < pages; ++i) if ((vector[i] & 1) == 0) return false;
            }
            return true;
        }
        catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException) { return null; }
    }

    [LibraryImport("libc", EntryPoint = "mincore", SetLastError = true)]
    private static unsafe partial int mincore(void* address, nuint length, byte* vector);

    public void Dispose()
    {
        if (_disposed) return;
        _uploader?.Dispose();
        _handle?.Dispose();
        _read = null;
        _disposed = true;
    }
}
