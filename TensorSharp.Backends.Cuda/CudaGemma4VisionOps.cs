using System;

namespace TensorSharp.Cuda;

/// <summary>F32 Gemma4 vision rotations and scale-one attention. Inputs remain
/// on the device; these operations do not narrow to F16 or TF32.</summary>
public static class CudaGemma4VisionOps
{
    /// <summary>Learned F32 projection with per-call pedantic accumulation.
    /// Weight layout is [in,out], matching the existing dense encoder storage.</summary>
    public static bool TryProjectionFloat32(Tensor result, Tensor input, Tensor weight)
        => CudaGemma4AudioOps.TryProjectionFloat32(result, input, weight);

    public static bool TryPadProjectionInput(Tensor result, Tensor input, int extraChannels)
    {
        if (extraChannels <= 0 || result?.DimensionCount != 2 || input?.DimensionCount != 2 ||
            input.Sizes[0] <= 0 || input.Sizes[1] <= 0 || input.Sizes[0] > int.MaxValue ||
            input.Sizes[1] > int.MaxValue || result.Sizes[0] != input.Sizes[0] ||
            result.Sizes[1] != input.Sizes[1] + (long)extraChannels || result.Sizes[1] > int.MaxValue ||
            !CudaKernelOps.TryGetContiguousFloat(result, out var ys, out var y, out int count) ||
            !CudaKernelOps.TryGetContiguousFloat(input, out var xs, out var x, out _) ||
            !ReferenceEquals(ys.AllocatorImpl, xs.AllocatorImpl) || ReferenceEquals(ys, xs)) return false;
        var allocator = ys.AllocatorImpl;
        if (!CudaKernelOps.TryGetKernels(allocator, out var kernels) || !kernels.HasGemma4VisionKernels) return false;
        allocator.Context.MakeCurrent(); xs.EnsureDeviceCurrent(); ys.EnsureDeviceCurrent();
        kernels.LaunchGemma4VisionPad(x, y, count, checked((int)input.Sizes[1]), checked((int)result.Sizes[1]), allocator.Stream.Handle);
        ys.MarkDeviceModified(); return true;
    }

    /// <summary>Unmasked bidirectional attention on [batch,tokens,heads,64].
    /// Reuses the existing F32 streaming kernel with exactly scale1. Scores,
    /// softmax and value accumulation are F32; no tensor-core mantissa rounding.</summary>
    public static bool TrySelfAttention(Tensor result, Tensor query, Tensor key, Tensor value)
    {
        if (result?.DimensionCount != 4 || query?.DimensionCount != 4 || key?.DimensionCount != 4 || value?.DimensionCount != 4 ||
            query.Sizes[3] != 64 || !CudaKernelOps.TryGetContiguousFloat(result, out var ys, out var y, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(query, out var qs, out var q, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(key, out var ks, out var k, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(value, out var vs, out var v, out _) ||
            !ReferenceEquals(ys.AllocatorImpl, qs.AllocatorImpl) || !ReferenceEquals(ys.AllocatorImpl, ks.AllocatorImpl) ||
            !ReferenceEquals(ys.AllocatorImpl, vs.AllocatorImpl) || ReferenceEquals(ys, qs) || ReferenceEquals(ys, ks) || ReferenceEquals(ys, vs)) return false;
        for (int d = 0; d < 4; d++)
            if (query.Sizes[d] <= 0 || query.Sizes[d] > int.MaxValue || query.Sizes[d] != result.Sizes[d] ||
                query.Sizes[d] != key.Sizes[d] || query.Sizes[d] != value.Sizes[d]) return false;
        var allocator = ys.AllocatorImpl;
        if (!CudaKernelOps.TryGetKernels(allocator, out var kernels) || !kernels.HasVisionAttentionF32(64)) return false;
        allocator.Context.MakeCurrent(); qs.EnsureDeviceCurrent(); ks.EnsureDeviceCurrent(); vs.EnsureDeviceCurrent(); ys.EnsureDeviceCurrent();
        int tokens = checked((int)query.Sizes[1]), heads = checked((int)query.Sizes[2]);
        long batchBytes = checked((long)tokens * heads * 64 * sizeof(float));
        for (int batch = 0; batch < query.Sizes[0]; batch++)
        {
            long offset = checked(batch * batchBytes);
            kernels.LaunchVisionAttentionF32(new IntPtr(q.ToInt64() + offset), new IntPtr(k.ToInt64() + offset),
                new IntPtr(v.ToInt64() + offset), new IntPtr(y.ToInt64() + offset), heads, tokens, 64, 1f, allocator.Stream.Handle);
        }
        ys.MarkDeviceModified(); return true;
    }

    /// <summary>Separate rotate-half pairs in each head's X and Y halves.
    /// Tables are [tokens,headDim/2]: the X quarter followed by the Y quarter.</summary>
    public static bool TryRope2D(Tensor data, Tensor cosine, Tensor sine, int tokens, int heads, int headDim)
    {
        if (tokens <= 0 || heads <= 0 || headDim <= 0 || headDim % 4 != 0 ||
            (long)tokens * heads * (headDim / 2) > int.MaxValue ||
            data?.DimensionCount != 2 || cosine?.DimensionCount != 2 || sine?.DimensionCount != 2 ||
            data.Sizes[0] != tokens || data.Sizes[1] != (long)heads * headDim ||
            cosine.Sizes[0] != tokens || cosine.Sizes[1] != headDim / 2 ||
            sine.Sizes[0] != tokens || sine.Sizes[1] != headDim / 2 ||
            !CudaKernelOps.TryGetContiguousFloat(data, out var xs, out var x, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(cosine, out var cs, out var c, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(sine, out var ss, out var s, out _) ||
            ReferenceEquals(xs, cs) || ReferenceEquals(xs, ss) || !ReferenceEquals(xs.AllocatorImpl, cs.AllocatorImpl) ||
            !ReferenceEquals(xs.AllocatorImpl, ss.AllocatorImpl)) return false;
        var allocator = xs.AllocatorImpl;
        if (!CudaKernelOps.TryGetKernels(allocator, out var kernels) || !kernels.HasGemma4VisionKernels) return false;
        allocator.Context.MakeCurrent(); xs.EnsureDeviceCurrent(); cs.EnsureDeviceCurrent(); ss.EnsureDeviceCurrent();
        kernels.LaunchGemma4VisionRope2D(x, c, s, checked(tokens * heads * (headDim / 2)), heads, headDim, allocator.Stream.Handle);
        xs.MarkDeviceModified(); return true;
    }
}
