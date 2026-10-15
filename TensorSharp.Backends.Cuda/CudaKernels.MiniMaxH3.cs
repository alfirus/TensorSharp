using System;

namespace TensorSharp.Cuda;

internal sealed unsafe partial class CudaKernels
{
    private IntPtr h3Transform, h3GroupNorm, h3CopyStrided, h3Clamp, h3WeightF16F32, h3WeightBf16F32;
    private IntPtr h3WeightQuantF32;
    internal bool HasH3QuantBandKernel
    {
        get
        {
            if (h3WeightQuantF32 != IntPtr.Zero) return true;
            try { h3WeightQuantF32 = module.GetFunction("ts_h3_weight_quant_f32"); return true; }
            catch { return false; }
        }
    }
    internal bool HasH3Kernels
    {
        get
        {
            if (h3Transform != IntPtr.Zero && h3GroupNorm != IntPtr.Zero && h3CopyStrided != IntPtr.Zero && h3Clamp != IntPtr.Zero && h3WeightF16F32 != IntPtr.Zero && h3WeightBf16F32 != IntPtr.Zero) return true;
            try
            {
                h3Transform = module.GetFunction("ts_h3_transform_f32");
                h3GroupNorm = module.GetFunction("ts_h3_groupnorm_silu_f32");
                h3CopyStrided = module.GetFunction("ts_h3_copy_strided_f32");
                h3Clamp = module.GetFunction("ts_h3_clamp_f32");
                h3WeightF16F32 = module.GetFunction("ts_h3_weight_f16_f32");
                h3WeightBf16F32 = module.GetFunction("ts_h3_weight_bf16_f32");
                return true;
            }
            catch { return false; }
        }
    }

    internal void LaunchH3WeightF16F32(IntPtr x,IntPtr y,int count,IntPtr stream)
    {
        void** args=stackalloc void*[3] { &x,&y,&count };
        Launch(h3WeightF16F32,Grid(count),1,1,BlockSize,1,1,0,stream,args);
    }

    internal void LaunchH3WeightBf16F32(IntPtr x,IntPtr y,int count,IntPtr stream)
    {
        void** args=stackalloc void*[3] { &x,&y,&count };
        Launch(h3WeightBf16F32,Grid(count),1,1,BlockSize,1,1,0,stream,args);
    }

    internal void LaunchH3WeightQuantF32(IntPtr x,IntPtr y,int type,int inner,int count,IntPtr stream)
    {
        void** args=stackalloc void*[5] { &x,&y,&type,&inner,&count };
        Launch(h3WeightQuantF32,Grid(count),1,1,BlockSize,1,1,0,stream,args);
    }

    internal void LaunchH3CopyStrided(IntPtr x,IntPtr y,int count,long n0,long n1,long n2,long n3,
        long xs0,long xs1,long xs2,long xs3,long ys0,long ys1,long ys2,long ys3,IntPtr stream)
    {
        void** args=stackalloc void*[15] { &x,&y,&count,&n0,&n1,&n2,&n3,&xs0,&xs1,&xs2,&xs3,&ys0,&ys1,&ys2,&ys3 };
        Launch(h3CopyStrided,Grid(count),1,1,BlockSize,1,1,0,stream,args);
    }

    internal void LaunchH3Clamp(IntPtr x,IntPtr y,int count,float lower,float upper,IntPtr stream)
    {
        void** args=stackalloc void*[5] { &x,&y,&count,&lower,&upper };
        Launch(h3Clamp,Grid(count),1,1,BlockSize,1,1,0,stream,args);
    }

    internal void LaunchH3Transform(IntPtr x, IntPtr a, IntPtr b, IntPtr y, int count, int op,
        int p0, int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8, int p9, IntPtr stream)
    {
        void** args = stackalloc void*[16] { &x, &a, &b, &y, &count, &op, &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9 };
        Launch(h3Transform, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
    }

    internal void LaunchH3GroupNorm(IntPtr x, IntPtr gain, IntPtr bias, IntPtr y,
        int frames, int rows, int channels, int groups, float eps, IntPtr stream)
    {
        void** args = stackalloc void*[8] { &x, &gain, &bias, &y, &rows, &channels, &groups, &eps };
        Launch(h3GroupNorm, checked((uint)(frames*groups)), 1, 1, 256, 1, 1, 0, stream, args);
    }
}
