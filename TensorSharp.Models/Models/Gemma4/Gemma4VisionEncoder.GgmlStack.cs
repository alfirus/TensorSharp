// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class Gemma4VisionEncoder
{
    // Only the validated embedding CUDA constructor reaches this candidate.
    // A synchronous owner lock also prevents its Dispose from freeing borrowed
    // weight pointers during the longer transient native call.
    private readonly object _ggmlVisionStackSync = new();
    private bool _ggmlVisionStackDisposed, _lastGgmlVisionStackUsed;

    private bool TryGgmlVisionStack(ref Tensor hidden, int numPatches, Rope2DCache rope)
    {
        _lastGgmlVisionStackUsed=false;
        bool Diagnostic(string name) => Environment.GetEnvironmentVariable(name) is string value && value.Length>0 && value[0]!='0';
        if (!_requirePreciseGgmlCudaVision || _hostModel != null || VisionTraceEnabled || VisionStageEnabled ||
            Diagnostic("TS_VISION_STAGE") || Diagnostic("TS_VISION_TRACE") || Diagnostic("TS_VISION_PROF") ||
            _allocator is not GgmlAllocator allocator || allocator.Context.Degree != 1) return false;
        var blocks=new GgmlGemma4VisionBlock[16];
        for (int b=0;b<blocks.Length;++b)
        {
            string p=$"v.blk.{b}";
            if (_weights.ContainsKey($"{p}.out_scale.weight")) return false;
            var matrices=new Tensor[7]; var norms=new Tensor[6];
            string[] matrixNames={"attn_q","attn_k","attn_v","attn_out","ffn_gate","ffn_up","ffn_down"};
            string[] normNames={"ln1","attn_q_norm","attn_k_norm","attn_post_norm","ln2","ffn_post_norm"};
            for (int m=0;m<7;++m) if (!_weights.TryGetValue($"{p}.{matrixNames[m]}.weight",out matrices[m])) return false;
            for (int n=0;n<6;++n) if (!_weights.TryGetValue($"{p}.{normNames[n]}.weight",out norms[n])) return false;
            blocks[b]=new GgmlGemma4VisionBlock(matrices,norms,BuildClampArray(p));
        }
        var result=new Tensor(_allocator,DType.Float32,numPatches,_hiddenSize);
        try
        {
            if (!GgmlBasicOps.TryGemma4VisionStack(result,hidden,blocks,rope.PosX,rope.PosY,_eps,_ropeTheta))
            { result.Dispose(); return false; }
            hidden.Dispose(); hidden=result;
            _lastGgmlVisionStackUsed=true;
            return true;
        }
        catch { result.Dispose(); hidden.Dispose(); throw; }
    }
}
