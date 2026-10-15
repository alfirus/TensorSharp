using System;

namespace TensorSharp.Cuda;

internal sealed unsafe partial class CudaKernels
{
    private IntPtr gemma4AudioAttention, gemma4AudioDepthwise;
    internal bool HasGemma4AudioKernels
    {
        get
        {
            if (gemma4AudioAttention != IntPtr.Zero && gemma4AudioDepthwise != IntPtr.Zero) return true;
            try
            {
                gemma4AudioAttention = module.GetFunction("ts_gemma4_audio_attention_f32");
                gemma4AudioDepthwise = module.GetFunction("ts_gemma4_audio_depthwise_f32");
                return true;
            }
            catch { return false; }
        }
    }

    internal void LaunchGemma4AudioAttention(IntPtr q, IntPtr k, IntPtr v, IntPtr r, IntPtr y,
        int sequence, int heads, int head, int past, int future, float cap, IntPtr stream)
    {
        void** args = stackalloc void*[11] { &q, &k, &v, &r, &y, &sequence, &heads, &head, &past, &future, &cap };
        Launch(gemma4AudioAttention, (uint)sequence, (uint)heads, 1, 128, 1, 1, 0, stream, args);
    }

    internal void LaunchGemma4AudioDepthwise(IntPtr x, IntPtr w, IntPtr y, int sequence, int channels, int kernel, IntPtr stream)
    {
        void** args = stackalloc void*[6] { &x, &w, &y, &sequence, &channels, &kernel };
        Launch(gemma4AudioDepthwise, Grid(checked(sequence * channels)), 1, 1, BlockSize, 1, 1, 0, stream, args);
    }
}
