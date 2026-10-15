// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;

namespace TensorSharp.Models.MiniMaxH3;

/// <summary>Audio tensors are token-major [time, channel] throughout the device path.
/// Convolution weights and activation vectors are uploaded once; only the initial
/// waveform/latent and the final result cross the device boundary.</summary>
internal sealed class MiniMaxH3AudioTensorOps : IDisposable
{
    internal readonly DirectContext Context;
    private readonly Dictionary<object, Tensor> _weights = new();
    private readonly MiniMaxH3TensorOps _attention;
    private const long ColumnBytes = 32L << 20;
    internal MiniMaxH3AudioTensorOps(IAllocator allocator)
    { Context = new DirectContext(allocator); _attention = new MiniMaxH3TensorOps(Context); }
    internal bool IsAccelerated => Context.IsAccelerated;

    internal Tensor Vector(float[] values)
    {
        if (values == null) return null;
        if (_weights.TryGetValue(values, out var tensor)) return tensor;
        tensor = Context.Own(Context.FromFloats(values, values.Length));
        _weights.Add(values, tensor); return tensor;
    }

    internal Tensor Input(float[] data, int channels, int time)
    {
        using var planar = Context.FromFloats(data, channels, time);
        using var transposed = planar.Transpose();
        return Ops.NewContiguous(transposed);
    }
    internal float[] Output(Tensor value)
    {
        using var transposed = value.Transpose();
        using var planar = Ops.NewContiguous(transposed);
        return planar.GetElementsAsFloat(checked((int)planar.ElementCount()));
    }

    internal Tensor Conv(Tensor x, MiniMaxH3DirectAudioVae.Conv conv, int transposeStride = 0)
    {
        int time = checked((int)x.Sizes[0]), channels = checked((int)x.Sizes[1]);
        bool transpose = transposeStride > 0;
        int outputChannels = transpose ? conv.Ic : conv.Oc, inputChannels = transpose ? conv.Oc : conv.Ic;
        if (channels != inputChannels) throw new InvalidOperationException("H3 audio convolution channel mismatch.");
        int stride = transpose ? transposeStride : conv.Stride;
        int outputTime = transpose ? checked((time-1)*stride+conv.K)
            : (time+2*conv.Pad-conv.Dilation*(conv.K-1)-1)/stride+1;
        if (outputTime <= 0) throw new InvalidOperationException("H3 audio convolution has no output.");
        if (!_weights.TryGetValue(conv, out var weight))
        {
            float[] values = conv.W;
            if (transpose)
            {
                values = new float[conv.W.Length];
                for (int i=0;i<channels;i++) for(int o=0;o<outputChannels;o++)
                    Array.Copy(conv.W, (i*outputChannels+o)*conv.K, values, (o*channels+i)*conv.K, conv.K);
            }
            weight = Context.Own(Context.FromFloats(values,outputChannels,checked(channels*conv.K)));
            _weights.Add(conv,weight);
        }
        var output = Context.NewF32(outputTime,outputChannels);
        try
        {
            long bands = Math.Max(1, Math.Min(outputTime,ColumnBytes/(sizeof(float)*(long)channels*conv.K)));
            for(int start=0;start<outputTime;start+=checked((int)bands))
            {
                int rows=(int)Math.Min(bands,outputTime-start);
                using var target=output.Narrow(0,start,rows);
                if (!transpose && conv.K == 1 && stride == 1 && conv.Pad == 0)
                {
                    using var source=x.Narrow(0,start,rows);
                    Gemm(target,source,weight);
                }
                else
                {
                    using var columns=Context.NewF32(rows,checked(channels*conv.K));
                    Transform(columns,x,null,null,1,channels,time,conv.K,stride,conv.Pad,conv.Dilation,start,transpose ? 1 : 0);
                    Gemm(target,columns,weight);
                }
            }
            // A transposed convolution's bias is applied after its caller trims.
            if (!transpose && conv.B != null) DirectOps.AddBiasRows(Context,output,Vector(conv.B));
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    internal Tensor Activation(Tensor x, MiniMaxH3DirectAudioVae.Act act)
    {
        int channels=checked((int)x.Sizes[1]), time=checked((int)x.Sizes[0]), kernel=act.Kernel;
        int padding=kernel/2-1;
        int padLeft=padding*2+(kernel-2)/2, padRight=padding*2+(kernel-1)/2;
        int upTime=(time+2*padding-1)*2+kernel-padLeft-padRight;
        using var up=Context.NewF32(upTime,channels);
        Transform(up,x,Vector(act.UpFilter),null,2,channels,time,kernel,1,kernel-1,1,padding,padLeft);
        Ops.Mul(up,up,2);
        float[] reciprocal;
        if (!_reciprocal.TryGetValue(act.Beta,out reciprocal))
        {
            reciprocal=Array.ConvertAll(act.Beta,value => 1f/value); _reciprocal.Add(act.Beta,reciprocal);
        }
        Snake(up,up,act.Alpha,reciprocal);
        int downLeft=kernel/2-(kernel%2 == 0 ? 1 : 0), downRight=kernel/2;
        int outputTime=(upTime+downLeft+downRight-kernel)/2+1;
        var output=Context.NewF32(outputTime,channels);
        try { Transform(output,up,Vector(act.DownFilter),null,2,channels,upTime,kernel,2,downLeft); return output; }
        catch { output.Dispose(); throw; }
    }
    private readonly Dictionary<float[],float[]> _reciprocal=new();
    internal void Snake(Tensor output, Tensor input, float[] alpha, float[] inverse)
        => Transform(output,input,Vector(alpha),Vector(inverse),3,checked((int)input.Sizes[1]));

    internal Tensor Linear(Tensor x, MiniMaxH3DirectAudioVaeEncoder.Lin layer, float[] bias)
    {
        if (!_weights.TryGetValue(layer,out var weight))
        {
            weight=Context.Own(Context.FromFloats(layer.W,layer.Ne1,layer.Ne0)); _weights.Add(layer,weight);
        }
        var output=Context.NewF32(x.Sizes[0],layer.Ne1);
        try
        {
            Gemm(output,x,weight);
            if(bias != null) DirectOps.AddBiasRows(Context,output,Vector(bias));
            return output;
        }
        catch { output.Dispose(); throw; }
    }
    internal Tensor Norm(Tensor input,float[] gain,float[] bias,float eps)
        => DirectOps.LayerNorm(Context,input,Vector(gain),Vector(bias),eps);
    internal Tensor AttendAndPool(Tensor qkv,int heads,int head,int latent)
    {
        int seq=checked((int)qkv.Sizes[0]),trunk=checked(heads*head);
        using var qview=qkv.Narrow(1,0,trunk); using var kview=qkv.Narrow(1,trunk,trunk); using var vview=qkv.Narrow(1,2*trunk,trunk);
        using var q=Ops.NewContiguous(qview); using var k=Ops.NewContiguous(kview); using var v=Ops.NewContiguous(vview);
        using var attended=_attention.CausalGqaAttention(q,k,v,seq,heads,heads,head);
        var pooled=Context.NewF32(seq,latent);
        try { Transform(pooled,attended,null,null,4,heads,head,latent); return pooled; }
        catch { pooled.Dispose(); throw; }
    }
    private void Gemm(Tensor output,Tensor input,Tensor weight)
    {
        if(Context.IsCuda)
        {
            if(!CudaMiniMaxH3Ops.TryMatmul(output,input,weight))
                throw new NotSupportedException("The selected H3 CUDA backend cannot execute F32 audio projection.");
        }
        else
        {
            using var transpose=weight.Transpose();
            if(input.Sizes[0]*input.Sizes[1]*weight.Sizes[0] <= 16_000_000L)
            {
                using var contiguous=Ops.NewContiguous(transpose);
                using var exact=Context.NewF32(output.Sizes);
                if(!MlxExactOps.TryMatmul(exact,input,contiguous))
                    throw new NotSupportedException("The selected H3 MLX backend cannot execute exact small F32 audio projection.");
                Ops.Copy(output,exact);
            }
            else Ops.Addmm(output,0,output,1,input,transpose);
        }
    }
    internal void Transform(Tensor output,Tensor input,Tensor a,Tensor b,int op,int p0,int p1=0,int p2=0,int p3=0,
        int p4=0,int p5=0,int p6=0,int p7=0,int p8=0,int p9=0)
    {
        bool success=Context.IsCuda
            ? CudaMiniMaxH3Ops.TryTransform(output,input,a,b,op,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9)
            : MlxMiniMaxH3Ops.TryTransform(output,input,a,b,op,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
        if(!success) throw new NotSupportedException("The selected H3 backend cannot execute its F32 audio VAE transform.");
    }
    internal static void Replace(ref Tensor slot,Tensor next) { var old=slot; slot=next; old.Dispose(); }
    public void Dispose() { _attention.Dispose(); Context.Dispose(); _weights.Clear(); _reciprocal.Clear(); }
}
