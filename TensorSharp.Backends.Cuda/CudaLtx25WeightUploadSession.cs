using System;
using System.Diagnostics;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Bounded exact-byte file uploads for LTX weights whose mapped file
/// pages are absent. The existing pointer-keyed weight cache owns device storage.</summary>
public sealed class CudaLtx25WeightUploadSession : IDisposable
{
    public const int DefaultChunkBytes = 8 << 20;
    private readonly CudaAllocator _allocator;
    private readonly object _sync = new();
    private readonly IntPtr[] _buffers = new IntPtr[2], _events = new IntPtr[2];
    private IntPtr _stream;
    private bool _disposed;
    public int ChunkBytes { get; }
    public long PinnedBytes => _buffers[0] == IntPtr.Zero ? 0 : 2L * ChunkBytes;
    public long UploadedBytes { get; private set; }
    public long UploadedWeights { get; private set; }
    public double UploadMilliseconds { get; private set; }

    public CudaLtx25WeightUploadSession(CudaAllocator allocator, int chunkBytes = DefaultChunkBytes)
    {
        ArgumentNullException.ThrowIfNull(allocator);
        if (chunkBytes is < (1 << 20) or > DefaultChunkBytes)
            throw new ArgumentOutOfRangeException(nameof(chunkBytes));
        _allocator = allocator;
        ChunkBytes = chunkBytes;
    }

    /// <summary>Returns false for a cache hit, without reading the source. On a
    /// miss, uploads exact source bytes and completes DMA before publishing the
    /// cache entry. Cancellation or I/O failure publishes no partial weight.</summary>
    public bool Preload(IntPtr cacheKey, ICudaWeightSource source, long sourceOffset,
        int ggmlType, long inner, long columns, long bytes, CancellationToken cancellationToken = default)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentNullException.ThrowIfNull(source);
            if (cacheKey == IntPtr.Zero || sourceOffset < 0 || bytes < 1 || sourceOffset > long.MaxValue - bytes ||
                inner is < 1 or > int.MaxValue || columns is < 1 or > int.MaxValue || !CudaQuantizedOps.SupportsMatmulType(ggmlType))
                throw new ArgumentOutOfRangeException(nameof(bytes));
            if (ExpectedBytes(ggmlType, inner, columns) != bytes)
                throw new ArgumentException("Source upload byte count does not match its GGML matrix type and dimensions.", nameof(bytes));
            cancellationToken.ThrowIfCancellationRequested();
            bool uploaded = false;
            CudaQuantizedOps.PreloadQuantizedWeightFromSource(_allocator, cacheKey, ggmlType, inner, columns, bytes, destination =>
            {
                long started = Stopwatch.GetTimestamp();
                Upload(destination, source, sourceOffset, bytes, cancellationToken);
                UploadMilliseconds += Stopwatch.GetElapsedTime(started).TotalMilliseconds;
                uploaded = true;
            });
            if (uploaded) { UploadedBytes += bytes; ++UploadedWeights; }
            return uploaded;
        }
    }

    private static long ExpectedBytes(int type, long inner, long columns)
    {
        (int block, int bytes) = type switch
        {
            0 => (1, 4), 1 or 30 => (1, 2), 2 or 20 => (32, 18), 3 => (32, 20),
            6 => (32, 22), 7 => (32, 24), 8 => (32, 34), 9 => (32, 36),
            10 => (256, 84), 11 => (256, 110), 12 => (256, 144), 13 => (256, 176),
            14 => (256, 210), 16 => (256, 66), 18 => (256, 98), 21 => (256, 110),
            22 => (256, 82), 23 => (256, 136), 39 => (32, 17), 40 => (64, 36),
            _ => throw new ArgumentOutOfRangeException(nameof(type)),
        };
        if (inner % block != 0) throw new ArgumentException("Source matrix rows must contain complete GGML quantization blocks.", nameof(inner));
        return checked(inner / block * bytes * columns);
    }

    private void EnsureBuffers()
    {
        if (_stream != IntPtr.Zero) return;
        try
        {
            CudaDriverApi.cuStreamCreate(out _stream, 1 /* NON_BLOCKING */).ThrowOnError();
            for (int i = 0; i < 2; ++i)
            {
                CudaDriverApi.cuMemHostAlloc(out _buffers[i], new UIntPtr((ulong)ChunkBytes), 0).ThrowOnError();
                CudaDriverApi.cuEventCreate(out _events[i], 2 /* DISABLE_TIMING */).ThrowOnError();
            }
        }
        catch { FreeBuffers(); throw; }
    }

    private void Upload(IntPtr destination, ICudaWeightSource source, long offset, long bytes, CancellationToken token)
    {
        _allocator.Context.MakeCurrent();
        EnsureBuffers();
        var pending = new bool[2];
        try
        {
            int slot = 0;
            for (long done = 0; done < bytes;)
            {
                token.ThrowIfCancellationRequested();
                if (pending[slot])
                    CudaDriverApi.cuEventSynchronize(_events[slot]).ThrowOnError();
                long count = Math.Min(ChunkBytes, bytes - done);
                source.Read(checked(offset + done), _buffers[slot], count);
                token.ThrowIfCancellationRequested();
                CudaDriverApi.cuMemcpyHtoDAsync(new IntPtr(checked(destination.ToInt64() + done)), _buffers[slot],
                    new UIntPtr((ulong)count), _stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(_events[slot], _stream).ThrowOnError();
                pending[slot] = true;
                done += count;
                slot ^= 1;
            }
        }
        finally
        {
            // Also required on EOF/cancellation: EnsureWeight may recycle the
            // device allocation and the next call may reuse either pinned slot.
            CudaDriverApi.cuStreamSynchronize(_stream).ThrowOnError();
        }
    }

    private void FreeBuffers()
    {
        for (int i = 0; i < 2; ++i)
        {
            if (_events[i] != IntPtr.Zero) { CudaDriverApi.cuEventDestroy(_events[i]); _events[i] = IntPtr.Zero; }
            if (_buffers[i] != IntPtr.Zero) { CudaDriverApi.cuMemFreeHost(_buffers[i]); _buffers[i] = IntPtr.Zero; }
        }
        if (_stream != IntPtr.Zero) { CudaDriverApi.cuStreamDestroy(_stream); _stream = IntPtr.Zero; }
    }

    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed) return;
            _allocator.Context.MakeCurrent();
            if (_stream != IntPtr.Zero) CudaDriverApi.cuStreamSynchronize(_stream).ThrowOnError();
            FreeBuffers();
            _disposed = true;
        }
    }
}
