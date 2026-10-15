// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.Cuda;
using TensorSharp.Models.Direct;
using TensorSharp.Models.MiniMaxH3;

namespace TensorSharp.Models.Ltx25;

/// <summary>Float32 audio operations; spatial activations use [time,frequency,channel].</summary>
internal sealed class Ltx25AudioTensorOps : IDisposable
{
    internal readonly DirectContext Context;
    private readonly CudaLtx25AudioSession _spectral;
    private readonly MiniMaxH3AudioTensorOps _accelerated;
    private readonly Dictionary<object, Tensor> _weights = new();
    internal Ltx25AudioTensorOps(IAllocator allocator)
    {
        Context = new DirectContext(allocator);
        if(Context.IsCuda) _spectral=new CudaLtx25AudioSession(Context.CudaAllocator);
        if (Context.IsAccelerated) _accelerated = new MiniMaxH3AudioTensorOps(allocator);
    }
    internal Tensor Vector(float[] values)
    {
        if (_weights.TryGetValue(values, out var t)) return t;
        t = Context.Own(Context.FromFloats(values, values.Length)); _weights.Add(values, t); return t;
    }
    internal void Gemm(Tensor output, Tensor input, Tensor weight)
    {
        if (Context.IsCuda)
        {
            if (!CudaMiniMaxH3Ops.TryMatmul(output,input,weight))
                throw new NotSupportedException("LTX audio requires Float32 CUDA projection support.");
        }
        else { using var wt = weight.Transpose(); Ops.Addmm(output,0,output,1,input,wt); }
    }
    internal Tensor Linear(Tensor x, float[] values, int outputs, float[] bias = null)
    {
        if (!_weights.TryGetValue(values,out var w))
        { w=Context.Own(Context.FromFloats(values,outputs,x.Sizes[1])); _weights.Add(values,w); }
        var y=Context.NewF32(x.Sizes[0],outputs);
        try { Gemm(y,x,w); if(bias!=null) DirectOps.AddBiasRows(Context,y,Vector(bias)); return y; }
        catch { y.Dispose(); throw; }
    }
    internal Tensor Conv2D(Tensor x,float[] values,float[] bias,int outputs,int kernel)
    {
        int time=(int)x.Sizes[0],freq=(int)x.Sizes[1],channels=(int)x.Sizes[2];
        if (!_weights.TryGetValue(values,out var w))
        {
            var reordered=new float[values.Length];
            for(int o=0;o<outputs;o++) for(int c=0;c<channels;c++) for(int h=0;h<kernel;h++) for(int f=0;f<kernel;f++)
                reordered[((o*kernel+h)*kernel+f)*channels+c]=values[((o*channels+c)*kernel+h)*kernel+f];
            w=Context.Own(Context.FromFloats(reordered,outputs,kernel*kernel*channels)); _weights.Add(values,w);
        }
        var y=Context.NewF32(time,freq,outputs);
        try
        {
            int band=Math.Max(1,(32<<20)/(4*freq*channels*kernel*kernel));
            for(int start=0;start<time;start+=band)
            {
                int count=Math.Min(band,time-start);
                using var columns=Context.NewF32(count,freq,kernel,kernel,channels); Ops.Fill(columns,0);
                for(int h=0;h<kernel;h++) for(int f=0;f<kernel;f++)
                {
                    int shiftT=h-(kernel-1),shiftF=f-kernel/2;
                    int t0=Math.Max(start,-shiftT),t1=Math.Min(start+count,time-shiftT);
                    int f0=Math.Max(0,-shiftF),f1=Math.Min(freq,freq-shiftF);
                    if(t1<=t0||f1<=f0) continue;
                    using var st=x.Narrow(0,t0+shiftT,t1-t0); using var src=st.Narrow(1,f0+shiftF,f1-f0);
                    using var dt=columns.Narrow(0,t0-start,t1-t0); using var df=dt.Narrow(1,f0,f1-f0);
                    using var dh=df.Select(2,h); using var dst=dh.Select(2,f); Ops.Copy(dst,src);
                }
                using var flat=columns.View(count*freq,kernel*kernel*channels);
                using var target3=y.Narrow(0,start,count); using var target=target3.View(count*freq,outputs);
                Gemm(target,flat,w); if(bias!=null) DirectOps.AddBiasRows(Context,target,Vector(bias));
            }
            return y;
        }
        catch {y.Dispose();throw;}
    }
    internal Tensor PixelSilu(Tensor x)
    {
        using var rows=x.View(x.Sizes[0]*x.Sizes[1],x.Sizes[2]);
        using var norm=DirectOps.RmsNorm(Context,rows,Context.Ones(x.Sizes[2]),1e-6f);
        Ops.SiLU(norm,norm); using var shaped=norm.View(x.Sizes); return Ops.NewContiguous(shaped);
    }
    internal Tensor Upsample2D(Tensor x)
    {
        using var t=Context.NewF32(x.Sizes[0]*2,x.Sizes[1],x.Sizes[2]); Ops.RepeatInterleave(t,x,2,0);
        var y=Context.NewF32(t.Sizes[0],t.Sizes[1]*2,t.Sizes[2]);
        try {Ops.RepeatInterleave(y,t,2,1);return y;} catch {y.Dispose();throw;}
    }
    internal Tensor Conv1D(Tensor x,MiniMaxH3DirectAudioVae.Conv c,int transposeStride=0)
    {
        if(_accelerated!=null) return _accelerated.Conv(x,c,transposeStride);
        bool trans=transposeStride>0; int time=(int)x.Sizes[0],channels=(int)x.Sizes[1];
        int outputChannels=trans?c.Ic:c.Oc,stride=trans?transposeStride:c.Stride;
        if(channels!=(trans?c.Oc:c.Ic))throw new InvalidOperationException("LTX audio convolution channel mismatch.");
        int length=trans?(time-1)*stride+c.K:(time+2*c.Pad-c.Dilation*(c.K-1)-1)/stride+1;
        if(length<=0) throw new InvalidOperationException("Empty LTX audio convolution.");
        if(!_weights.TryGetValue(c,out var weight))
        {
            var values=c.W;
            if(trans)
            {
                values=new float[c.W.Length];
                for(int i=0;i<channels;i++)for(int o=0;o<outputChannels;o++)
                    Array.Copy(c.W,(i*outputChannels+o)*c.K,values,(o*channels+i)*c.K,c.K);
            }
            weight=Context.Own(Context.FromFloats(values,outputChannels,channels*c.K));_weights.Add(c,weight);
        }
        var input=x.GetElementsAsFloat((int)x.ElementCount()); var result=Context.NewF32(length,outputChannels);
        try
        {
            int band=Math.Max(1,(32<<20)/(4*channels*c.K));
            for(int start=0;start<length;start+=band)
            {
                int count=Math.Min(band,length-start); var col=new float[count*channels*c.K];
                for(int row=0;row<count;row++)for(int ch=0;ch<channels;ch++)for(int k=0;k<c.K;k++)
                {
                    int index=(start+row)*stride-c.Pad+k*c.Dilation;
                    if(trans){int n=start+row-k;index=n>=0&&n%stride==0?n/stride:-1;}
                    if((uint)index<(uint)time) col[(row*channels+ch)*c.K+k]=input[index*channels+ch];
                }
                using var cols=Context.FromFloats(col,count,channels*c.K);using var target=result.Narrow(0,start,count);Gemm(target,cols,weight);
            }
            if(!trans&&c.B!=null)DirectOps.AddBiasRows(Context,result,Vector(c.B));return result;
        }
        catch {result.Dispose();throw;}
    }
    internal Tensor Activation(Tensor x,MiniMaxH3DirectAudioVae.Act a)
    {
        if(_accelerated!=null)return _accelerated.Activation(x,a);
        int time=(int)x.Sizes[0],channels=(int)x.Sizes[1],kernel=a.Kernel,pad=kernel/2-1,crop=pad*2+(kernel-2)/2;
        float[] input=x.GetElementsAsFloat((int)x.ElementCount()),up=new float[time*2*channels],output=new float[time*channels];
        for(int t=0;t<2*time;t++)for(int c=0;c<channels;c++)
        {
            float value=0;
            for(int k=0;k<kernel;k++)
            {
                int n=t+crop-(kernel-1-k); if(n%2!=0)continue;
                int source=n/2-pad;if(source< -pad||source>=time+pad)continue;source=Math.Clamp(source,0,time-1);
                value+=input[source*channels+c]*a.UpFilter[k];
            }
            value*=2;float s=MathF.Sin(value*a.Alpha[c]);up[t*channels+c]=value+s*s/a.Beta[c];
        }
        for(int t=0;t<time;t++)for(int c=0;c<channels;c++)
        {float sum=0;for(int k=0;k<kernel;k++)sum+=up[Math.Clamp(t*2-(kernel/2-1)+k,0,2*time-1)*channels+c]*a.DownFilter[k];output[t*channels+c]=sum;}
        return Context.FromFloats(output,time,channels);
    }
    internal bool TryCudaLogClamp(Tensor value)=>_spectral?.TryLogClamp(value,value)==true;
    internal bool TryCudaSqrt(Tensor value)=>_spectral?.TrySqrt(value,value)==true;
    public void Dispose(){_spectral?.Dispose();_accelerated?.Dispose();Context.Dispose();_weights.Clear();}
}
