// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>Separate F32 Ingredients factors must retain their small updates.
/// Pedantic compute is per call; the base model's cuBLAS policy is unchanged.</summary>
public static class CudaLtx25LoraOps
{
    public static bool TryLinear(Tensor output, Tensor input, IntPtr host, long inner, long columns, long bytes)
    {
        if (host == IntPtr.Zero || inner is < 1 or > int.MaxValue || columns is < 1 or > int.MaxValue ||
            bytes != checked(inner * columns * sizeof(float)) || input == null || output == null ||
            input.DimensionCount != 2 || output.DimensionCount != 2 || input.Sizes[0] is < 1 or > int.MaxValue ||
            input.Sizes[1] != inner || output.Sizes[0] != input.Sizes[0] || output.Sizes[1] != columns ||
            !CudaKernelOps.TryGetContiguousFloat(input, out var xs, out var x, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(output, out var ys, out var y, out _) ||
            !ReferenceEquals(xs.AllocatorImpl, ys.AllocatorImpl) || ReferenceEquals(xs, ys)) return false;
        var allocator = ys.AllocatorImpl;
        allocator.Context.MakeCurrent();
        CudaQuantizedOps.PreloadQuantizedWeight(allocator, host, host, 0, inner, columns, bytes);
        if (!CudaQuantizedOps.TryGetResidentDevicePtr(allocator, host, out var weight, out int type) || type != 0) return false;
        xs.EnsureDeviceCurrent();
        if (output.StorageOffset != 0 || output.ElementCount() != output.Storage.ElementCount) ys.EnsureDeviceCurrent();
        allocator.Blas.SetStream(allocator.Stream.Handle);
        float alpha = 1, beta = 0;
        CublasApi.cublasGemmEx(allocator.Blas.Handle, CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
            checked((int)columns), checked((int)input.Sizes[0]), checked((int)inner), ref alpha,
            weight, CublasApi.CUDA_R_32F, checked((int)inner), x, CublasApi.CUDA_R_32F, checked((int)inner),
            ref beta, y, CublasApi.CUDA_R_32F, checked((int)columns),
            69 /* CUBLAS_COMPUTE_32F_PEDANTIC */, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        ys.MarkDeviceModified(); return true;
    }
}
