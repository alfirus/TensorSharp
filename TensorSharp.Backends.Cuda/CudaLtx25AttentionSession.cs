// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Model-owned, bounded score workspace for unbiased LTX attention.
/// Both products use pedantic Float32 cuBLAS arithmetic. Token-major head views
/// are passed directly to cuBLAS without copies or activation narrowing.</summary>
public sealed class CudaLtx25AttentionSession : IDisposable
{
    private readonly CudaAllocator _allocator;
    private readonly object _sync = new();
    private Tensor _scores;
    private bool _disposed;
    public const int ComputeType = 69; // CUBLAS_COMPUTE_32F_PEDANTIC
    // This bounds the live score tensor; allocator pooling may retain backing
    // allocations independently after a tensor is released.
    public long ScoreBudgetBytes { get; }
    public long ScoreCapacityBytes => (_scores?.ElementCount() ?? 0) * sizeof(float);
    public long ExecutedAttentions { get; private set; }
    public int LastQueryTileRows { get; private set; }
    public int LastHeadTileCount { get; private set; }

    public CudaLtx25AttentionSession(CudaAllocator allocator, long scoreBudgetBytes = 256L << 20)
    {
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        if (scoreBudgetBytes < sizeof(float) || scoreBudgetBytes > (256L << 20))
            throw new ArgumentOutOfRangeException(nameof(scoreBudgetBytes));
        ScoreBudgetBytes = scoreBudgetBytes;
    }

    /// <summary>Returns false before dispatch for unsupported shapes, aliases,
    /// devices or workspace requirements. A successful call increments the
    /// execution counter; no fallback is reported as native execution.</summary>
    public bool TryAttention(Tensor output, Tensor query, Tensor key, Tensor value,
        int heads, int headDim, float scale)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (heads is < 1 or > 128 || headDim is < 1 or > 1024 || !float.IsFinite(scale) ||
                output == null || query == null || key == null || value == null ||
                output.DimensionCount != 2 || query.DimensionCount != 2 || key.DimensionCount != 2 || value.DimensionCount != 2 ||
                !CudaKernelOps.TryGetContiguousFloat(output, out var ys, out _, out _) ||
                !CudaKernelOps.TryGetContiguousFloat(query, out var qs, out _, out _) ||
                !CudaKernelOps.TryGetContiguousFloat(key, out var ks, out _, out _) ||
                !CudaKernelOps.TryGetContiguousFloat(value, out var vs, out _, out _) ||
                !ReferenceEquals(ys.AllocatorImpl, _allocator) || !ReferenceEquals(qs.AllocatorImpl, _allocator) ||
                !ReferenceEquals(ks.AllocatorImpl, _allocator) || !ReferenceEquals(vs.AllocatorImpl, _allocator) ||
                ReferenceEquals(ys, qs) || ReferenceEquals(ys, ks) || ReferenceEquals(ys, vs)) return false;
            long tokensLong = query.Sizes[0], keysLong = key.Sizes[0], widthLong = (long)heads * headDim;
            if (tokensLong is < 1 or > int.MaxValue || keysLong is < 1 or > int.MaxValue || query.Sizes[1] != widthLong ||
                value.Sizes[0] != keysLong || output.Sizes[0] != tokensLong ||
                key.Sizes[1] != widthLong || value.Sizes[1] != widthLong || output.Sizes[1] != widthLong) return false;
            long budgetElements = ScoreBudgetBytes / sizeof(float);
            int queryTile = (int)Math.Min(tokensLong, budgetElements / keysLong);
            if (queryTile < 1) return false; // Even one complete-key softmax row must fit.
            // Preserve the established route and arithmetic for fitting heads.
            // Larger reference sequences tile queries, never keys: each softmax
            // still sees every key. A modest query tile also batches more heads.
            if (tokensLong * keysLong > budgetElements) queryTile = Math.Min(queryTile, 2048);
            long tileElements = queryTile * keysLong;
            int tileHeads = (int)Math.Min(heads, budgetElements / tileElements);
            if (tileHeads < 1) return false;
            _allocator.Context.MakeCurrent();
            if (!CudaKernelOps.TryGetKernels(_allocator, out _)) return false;
            long elements = tileElements * tileHeads;
            if (_scores == null || _scores.ElementCount() < elements)
            {
                var larger = new Tensor(_allocator, DType.Float32, elements);
                _scores?.Dispose(); _scores = larger;
            }
            qs.EnsureDeviceCurrent(); ks.EnsureDeviceCurrent(); vs.EnsureDeviceCurrent();
            // Keep guards outside a contiguous row view intact.
            if (output.StorageOffset != 0 || output.ElementCount() != output.Storage.ElementCount)
                ys.EnsureDeviceCurrent();
            _allocator.Blas.SetStream(_allocator.Stream.Handle);
            int tokens = (int)tokensLong, keys = (int)keysLong, width = (int)widthLong;
            var ss = (CudaStorage)_scores.Storage;
            IntPtr scorePointer = ss.DevicePtrAtElement(_scores.StorageOffset);
            for (int first = 0; first < heads; first += tileHeads)
            {
                int count = Math.Min(tileHeads, heads - first);
                IntPtr k = ks.DevicePtrAtElement(key.StorageOffset + first * headDim);
                IntPtr v = vs.DevicePtrAtElement(value.StorageOffset + first * headDim);
                for (int queryStart = 0; queryStart < tokens;)
                {
                    int queryCount = Math.Min(queryTile, tokens - queryStart);
                    long headElements = (long)queryCount * keys;
                    IntPtr q = qs.DevicePtrAtElement(query.StorageOffset + (long)queryStart * width + first * headDim);
                    IntPtr y = ys.DevicePtrAtElement(output.StorageOffset + (long)queryStart * width + first * headDim);
                    float alpha = scale, beta = 0;
                    // Column-major K[D,key]^T Q[D,query] produces contiguous
                    // row-major scores[query,key], one matrix per head.
                    CublasApi.cublasGemmStridedBatchedEx(_allocator.Blas.Handle,
                        CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                        keys, queryCount, headDim, ref alpha,
                        k, CublasApi.CUDA_R_32F, width, headDim,
                        q, CublasApi.CUDA_R_32F, width, headDim, ref beta,
                        scorePointer, CublasApi.CUDA_R_32F, keys, headElements, count,
                        ComputeType, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                    ss.MarkDeviceModified();
                    using var active = _scores.Narrow(0, 0, count * headElements);
                    using var rows = active.View((long)count * queryCount, keys);
                    if (!CudaKernelOps.TrySoftmax(rows, rows))
                        throw new NotSupportedException("LTX attention requires CUDA Float32 softmax.");
                    alpha = 1;
                    // V[D,key] P[key,query] writes disjoint head columns directly
                    // into token-major output rows (leading dimension = width).
                    CublasApi.cublasGemmStridedBatchedEx(_allocator.Blas.Handle,
                        CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                        headDim, queryCount, keys, ref alpha,
                        v, CublasApi.CUDA_R_32F, width, headDim,
                        scorePointer, CublasApi.CUDA_R_32F, keys, headElements, ref beta,
                        y, CublasApi.CUDA_R_32F, width, headDim, count,
                        ComputeType, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                    queryStart += queryCount;
                }
            }
            LastQueryTileRows = queryTile; LastHeadTileCount = tileHeads;
            ys.MarkDeviceModified(); ExecutedAttentions++;
            return true;
        }
    }

    public void ReleaseWorkspace()
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ReleaseScores();
        }
    }

    private void ReleaseScores()
    {
        if (_scores == null) return;
        _allocator.Context.MakeCurrent(); _allocator.Stream.Synchronize();
        _scores.Dispose(); _scores = null;
    }

    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed) return;
            _disposed = true;
            ReleaseScores();
        }
    }
}
