using System;

namespace TensorSharp.Cuda;

internal sealed unsafe partial class CudaKernels
{
    private IntPtr gemma4VisionRope2DF32, gemma4VisionPadF32;
    internal bool HasGemma4VisionKernels
    {
        get
        {
            if (gemma4VisionRope2DF32 != IntPtr.Zero && gemma4VisionPadF32 != IntPtr.Zero) return true;
            try
            {
                gemma4VisionRope2DF32 = module.GetFunction("ts_gemma4_vision_rope2d_f32");
                gemma4VisionPadF32 = module.GetFunction("ts_gemma4_vision_pad_f32");
                return true;
            }
            catch { return false; }
        }
    }
    internal void LaunchGemma4VisionRope2D(IntPtr data, IntPtr cosine, IntPtr sine, int pairs, int heads, int headDim, IntPtr stream)
    {
        void** args = stackalloc void*[6] { &data, &cosine, &sine, &pairs, &heads, &headDim };
        Launch(gemma4VisionRope2DF32, Grid(pairs), 1, 1, BlockSize, 1, 1, 0, stream, args);
    }
    internal void LaunchGemma4VisionPad(IntPtr input, IntPtr output, int count, int inputColumns, int outputColumns, IntPtr stream)
    {
        void** args = stackalloc void*[5] { &input, &output, &count, &inputColumns, &outputColumns };
        Launch(gemma4VisionPadF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
    }
}
