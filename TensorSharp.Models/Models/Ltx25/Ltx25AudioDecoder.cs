// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using TensorSharp.Cpu;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;
using TensorSharp.Runtime;
using Conv = TensorSharp.Models.MiniMaxH3.MiniMaxH3DirectAudioVae.Conv;
using Act = TensorSharp.Models.MiniMaxH3.MiniMaxH3DirectAudioVae.Act;

namespace TensorSharp.Models.Ltx25;

/// <summary>Native LTX 2.5 causal audio VAE, alias-free BigVGAN and learned bandwidth
/// extension. BF16 checkpoint weights are widened to Float32, including spectral buffers.
/// The supplied allocator is borrowed and remains owned by the caller.</summary>
public sealed class Ltx25AudioDecoder : IDisposable
{
    private readonly SafetensorsFile _file;
    private readonly Ltx25AudioTensorOps _ops;
    private readonly IDisposable _ownedAllocator;
    private readonly JsonDocument _metadata;
    private readonly JsonElement _config, _decoder;
    private readonly Dictionary<string,float[]> _values=new(StringComparer.Ordinal);
    private readonly Dictionary<string,Conv> _convs=new(StringComparer.Ordinal);
    private readonly Dictionary<string,Act> _acts=new(StringComparer.Ordinal);
    private readonly object _gate=new();
    private bool _disposed;
    public const int LatentChannels = 8;
    public const int LatentFrequencyBins = 16;
    public const int SampleRate = 48000;

    public Ltx25AudioDecoder(string weightsPath,IAllocator allocator=null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(weightsPath);
        _file=new SafetensorsFile(weightsPath);
        if(allocator==null) {var cpu=new CpuAllocator(BlasEnum.DotNet);allocator=cpu;_ownedAllocator=cpu as IDisposable;}
        try
        {
            _ops=new Ltx25AudioTensorOps(allocator);
            if(!_file.Metadata.TryGetValue("config",out string json))throw new InvalidDataException("LTX audio checkpoint has no config metadata.");
            _metadata=JsonDocument.Parse(json);_config=_metadata.RootElement;
            _decoder=_config.GetProperty("audio_vae").GetProperty("model").GetProperty("params").GetProperty("ddconfig");
            ValidateConfig();
        }
        catch {_metadata?.Dispose();_file.Dispose();_ops?.Dispose();_ownedAllocator?.Dispose();throw;}
    }
    private static int Int(JsonElement value,string key)=>value.GetProperty(key).GetInt32();
    private static int[] Ints(JsonElement value,string key)=>value.GetProperty(key).EnumerateArray().Select(v=>v.GetInt32()).ToArray();
    private void ValidateConfig()
    {
        if(Int(_decoder,"z_channels")!=8||Int(_decoder,"mel_bins")!=64||Int(_decoder,"out_ch")!=2
            ||_decoder.GetProperty("norm_type").GetString()!="pixel"||_decoder.GetProperty("causality_axis").GetString()!="height"
            ||_decoder.GetProperty("mid_block_add_attention").GetBoolean()||_decoder.GetProperty("attn_resolutions").GetArrayLength()!=0
            ||!Ints(_decoder,"ch_mult").SequenceEqual(new[]{1,2,4})||Int(_decoder,"num_res_blocks")!=2)
            throw new NotSupportedException("LTX audio decoder requires the 2.5 stereo pixel-normalized causal VAE architecture.");
        var audio=_config.GetProperty("audio_vae");
        var stft=audio.GetProperty("preprocessing").GetProperty("stft");
        if(Int(audio.GetProperty("model").GetProperty("params"),"sampling_rate")!=16000
            ||Int(stft,"hop_length")!=160||!stft.GetProperty("causal").GetBoolean())
            throw new NotSupportedException("LTX audio requires the causal 16 kHz VAE with a 160-sample mel hop.");
        var v=_config.GetProperty("vocoder");var bwe=v.GetProperty("bwe");
        if(Int(bwe,"input_sampling_rate")!=16000||Int(bwe,"output_sampling_rate")!=48000||Int(bwe,"hop_length")!=80
            ||Int(bwe,"n_fft")!=512||Int(bwe,"num_mels")!=64)throw new NotSupportedException("Unsupported LTX audio bandwidth extension configuration.");
        foreach(string name in new[]{"vocoder","bwe"})
        {
            var c=v.GetProperty(name);
            if(c.GetProperty("resblock").GetString()!="AMP1"||c.GetProperty("activation").GetString()!="snakebeta"
                ||!c.GetProperty("stereo").GetBoolean()||c.GetProperty("use_tanh_at_final").GetBoolean()
                ||c.GetProperty("use_bias_at_final").GetBoolean())throw new NotSupportedException("Unsupported LTX audio vocoder configuration.");
            int[] rates=Ints(c,"upsample_rates"),kernels=Ints(c,"upsample_kernel_sizes");
            if(rates.Length!=kernels.Length||rates.Any(n=>n<=0)||kernels.Where((n,i)=>n<rates[i]||(n-rates[i])%2!=0).Any()
                ||rates.Aggregate(1,(a,b)=>checked(a*b))!=(name=="vocoder"?160:240)
                ||!Ints(c,"resblock_kernel_sizes").SequenceEqual(new[]{3,7,11}))throw new NotSupportedException("Unsupported LTX audio upsampling geometry.");
            foreach(var dil in c.GetProperty("resblock_dilation_sizes").EnumerateArray())
                if(!dil.EnumerateArray().Select(x=>x.GetInt32()).SequenceEqual(new[]{1,3,5}))throw new NotSupportedException("Unsupported LTX AMP dilation.");
        }
        Require("audio_vae.per_channel_statistics.mean-of-means",128);
        Require("audio_vae.per_channel_statistics.std-of-means",128);
        Require("vocoder.mel_stft.stft_fn.forward_basis",514,1,512);
        Require("vocoder.mel_stft.mel_basis",64,257);
    }
    private void Require(string name,params long[] shape)
    {
        if(!_file.TensorShape(name).SequenceEqual(shape))throw new InvalidDataException($"LTX audio tensor '{name}' must have shape [{string.Join(',',shape)}].");
    }
    private float[] Value(string name)
    {
        if(!_values.TryGetValue(name,out var v)){v=_file.ReadFloat32(name);_values.Add(name,v);}return v;
    }
    private Tensor Conv2D(Tensor x,string name)
    {
        string w=name+".weight",b=name+".bias";var s=_file.TensorShape(w);
        if(s.Length!=4||s[1]!=x.Sizes[2]||s[2]!=s[3]||s[2] is not (1 or 3))throw new InvalidDataException($"Invalid LTX audio convolution '{name}'.");
        Require(b,s[0]);return _ops.Conv2D(x,Value(w),Value(b),(int)s[0],(int)s[2]);
    }
    private Tensor Residual(Tensor x,string name)
    {
        using var n1=_ops.PixelSilu(x);using var c1=Conv2D(n1,name+".conv1.conv");
        using var n2=_ops.PixelSilu(c1);var y=Conv2D(n2,name+".conv2.conv");
        try
        {
            if(x.Sizes[2]==y.Sizes[2])Ops.Add(y,y,x);
            else {using var skip=Conv2D(x,name+".nin_shortcut.conv");Ops.Add(y,y,skip);}return y;
        }
        catch {y.Dispose();throw;}
    }
    private static void Replace(ref Tensor slot,Tensor next){var old=slot;slot=next;old.Dispose();}
    /// <summary>Decode packed [audioTokens,128] latents (channel then frequency within
    /// each token) to planar stereo PCM. sampleCount only trims; short authentic
    /// decoded audio is never padded. Trace snapshots use token-major layouts.</summary>
    public GeneratedVideoAudio Decode(float[] tokenMajorLatents,int audioTokens,int? sampleCount=null,
        CancellationToken cancellationToken=default,Action<string,float[],int[]> trace=null)
    {
        lock(_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed,this);
            ArgumentNullException.ThrowIfNull(tokenMajorLatents);
            if(audioTokens<=0||tokenMajorLatents.Length!=checked(audioTokens*128))throw new ArgumentException("LTX audio latents must have shape [audioTokens,128].",nameof(tokenMajorLatents));
            if(sampleCount is <=0)throw new ArgumentOutOfRangeException(nameof(sampleCount));
            if(tokenMajorLatents.Any(v=>!float.IsFinite(v)))throw new ArgumentException("LTX audio latents must be finite.",nameof(tokenMajorLatents));
            cancellationToken.ThrowIfCancellationRequested();
            float[] mean=Value("audio_vae.per_channel_statistics.mean-of-means"),std=Value("audio_vae.per_channel_statistics.std-of-means");
            var denorm=new float[tokenMajorLatents.Length];
            for(int t=0;t<audioTokens;t++)for(int f=0;f<16;f++)for(int c=0;c<8;c++)
            {int feature=c*16+f;denorm[(t*16+f)*8+c]=tokenMajorLatents[t*128+feature]*std[feature]+mean[feature];}
            Tensor x=_ops.Context.FromFloats(denorm,audioTokens,16,8);
            try
            {
                Replace(ref x,Conv2D(x,"audio_vae.decoder.conv_in.conv"));
                Replace(ref x,Residual(x,"audio_vae.decoder.mid.block_1"));Replace(ref x,Residual(x,"audio_vae.decoder.mid.block_2"));
                for(int level=2;level>=0;level--)
                {
                    for(int block=0;block<3;block++){cancellationToken.ThrowIfCancellationRequested();Replace(ref x,Residual(x,$"audio_vae.decoder.up.{level}.block.{block}"));}
                    if(level>0)
                    {
                        using var up=_ops.Upsample2D(x);using var convolved=Conv2D(up,$"audio_vae.decoder.up.{level}.upsample.conv.conv");
                        using var cropped=convolved.Narrow(0,1,convolved.Sizes[0]-1);Replace(ref x,Ops.NewContiguous(cropped));
                    }
                }
                using(var normalized=_ops.PixelSilu(x))Replace(ref x,Conv2D(normalized,"audio_vae.decoder.conv_out.conv"));
                Snapshot("mel",x,trace);
                int melFrames=checked((int)x.Sizes[0]);
                // Merge stereo and frequency in [left64,right64] order.
                using var transposed=x.Transpose(1,2);using var contiguous=Ops.NewContiguous(transposed);using var mel=contiguous.View(melFrames,128);
                using var native=Vocoder(mel,"vocoder.vocoder",_config.GetProperty("vocoder").GetProperty("vocoder"),true,cancellationToken);
                Snapshot("native16k",native,trace);
                using var bweMel=Mel(native);Snapshot("bweMel",bweMel,trace);
                using var residual=Vocoder(bweMel,"vocoder.bwe_generator",_config.GetProperty("vocoder").GetProperty("bwe"),false,cancellationToken);
                Snapshot("bweResidual",residual,trace);
                using var skip=Resample3(native);Ops.Add(residual,residual,skip);Ops.Clamp(residual,residual,-1,1);
                int count=Math.Min(checked((int)native.Sizes[0]*3),sampleCount??int.MaxValue);
                using var crop=residual.Narrow(0,0,count);using var final=Ops.NewContiguous(crop);Snapshot("pcm48k",final,trace);
                float[] interleaved=final.GetElementsAsFloat(count*2);float[][] channels={new float[count],new float[count]};
                for(int i=0;i<count;i++)for(int c=0;c<2;c++)
                {float v=interleaved[i*2+c];if(!float.IsFinite(v))throw new InvalidOperationException("LTX audio decoder produced non-finite PCM.");channels[c][i]=v;}
                return new GeneratedVideoAudio{Channels=channels,SampleRate=SampleRate};
            }
            finally{x.Dispose();}
        }
    }
    private static void Snapshot(string name,Tensor tensor,Action<string,float[],int[]> trace)
    {if(trace!=null)trace(name,tensor.GetElementsAsFloat((int)tensor.ElementCount()),tensor.Sizes.ToArray().Select(n=>(int)n).ToArray());}
    private Conv Conv1D(string name,int stride=1,int pad=0,int dilation=1,bool transpose=false)
    {
        if(_convs.TryGetValue(name,out var c))return c;var s=_file.TensorShape(name+".weight");
        if(s.Length!=3)throw new InvalidDataException($"Invalid LTX vocoder convolution '{name}'.");
        string bias=name+".bias";
        c=new Conv{W=Value(name+".weight"),B=_file.HasTensor(bias)?Value(bias):null,Oc=(int)s[0],Ic=(int)s[1],K=(int)s[2],Stride=stride,Pad=pad,Dilation=dilation,Transposed=transpose};
        if(c.B!=null&&c.B.Length!=(transpose?c.Ic:c.Oc))throw new InvalidDataException($"Invalid LTX vocoder bias '{bias}'.");
        _convs.Add(name,c);return c;
    }
    private Act Activation(string name,int channels)
    {
        if(_acts.TryGetValue(name,out var a))return a;
        Require(name+".act.alpha",channels);Require(name+".act.beta",channels);
        Require(name+".upsample.filter",1,1,12);Require(name+".downsample.lowpass.filter",1,1,12);
        float[] up=(float[])Value(name+".upsample.filter").Clone();Array.Reverse(up);
        a=new Act{Alpha=Array.ConvertAll(Value(name+".act.alpha"),MathF.Exp),Beta=Array.ConvertAll(Value(name+".act.beta"),v=>MathF.Exp(v)+1e-9f),UpFilter=up,DownFilter=Value(name+".downsample.lowpass.filter"),Channels=channels,Kernel=12};
        _acts.Add(name,a);return a;
    }
    private Tensor Amp(Tensor input,string prefix,int kernel)
    {
        var x=Ops.NewContiguous(input);
        try
        {
            int[] dilations={1,3,5};
            for(int j=0;j<3;j++)
            {
                using var a=_ops.Activation(x,Activation($"{prefix}.acts1.{j}",(int)x.Sizes[1]));
                using var c=_ops.Conv1D(a,Conv1D($"{prefix}.convs1.{j}",pad:(kernel-1)*dilations[j]/2,dilation:dilations[j]));
                using var b=_ops.Activation(c,Activation($"{prefix}.acts2.{j}",(int)x.Sizes[1]));
                using var d=_ops.Conv1D(b,Conv1D($"{prefix}.convs2.{j}",pad:(kernel-1)/2));Ops.Add(x,x,d);
            }
            return x;
        }
        catch{x.Dispose();throw;}
    }
    private Tensor Vocoder(Tensor mel,string prefix,JsonElement config,bool clamp,CancellationToken token)
    {
        var x=_ops.Conv1D(mel,Conv1D(prefix+".conv_pre",pad:3));
        try
        {
            int[] rates=Ints(config,"upsample_rates"),kernels=Ints(config,"upsample_kernel_sizes"),blocks=Ints(config,"resblock_kernel_sizes");
            for(int i=0;i<rates.Length;i++)
            {
                token.ThrowIfCancellationRequested();var c=Conv1D($"{prefix}.ups.{i}",transpose:true);
                using var up=_ops.Conv1D(x,c,rates[i]);int pad=(kernels[i]-rates[i])/2;
                using var trimmed=up.Narrow(0,pad,up.Sizes[0]-2*pad);Replace(ref x,Ops.NewContiguous(trimmed));
                if(c.B!=null)DirectOps.AddBiasRows(_ops.Context,x,_ops.Vector(c.B));
                Tensor sum=null;
                try
                {
                    for(int b=0;b<blocks.Length;b++)
                    {var branch=Amp(x,$"{prefix}.resblocks.{i*blocks.Length+b}",blocks[b]);if(sum==null)sum=branch;else{using(branch)Ops.Add(sum,sum,branch);}}
                    Ops.Div(sum,sum,blocks.Length);Replace(ref x,sum);sum=null;
                }
                finally{sum?.Dispose();}
            }
            using var post=_ops.Activation(x,Activation(prefix+".act_post",(int)x.Sizes[1]));
            Replace(ref x,_ops.Conv1D(post,Conv1D(prefix+".conv_post",pad:3)));if(clamp)Ops.Clamp(x,x,-1,1);return x;
        }
        catch{x.Dispose();throw;}
    }
    private Tensor Mel(Tensor waveform)
    {
        int length=(int)waveform.Sizes[0],frames=(length+79)/80;var output=_ops.Context.NewF32(frames,128);
        try
        {
            for(int channel=0;channel<2;channel++)
            {
                using var padded=_ops.Context.NewF32(frames*80+432,1);Ops.Fill(padded,0);
                using var source=waveform.Narrow(1,channel,1);using var target=padded.Narrow(0,432,length);Ops.Copy(target,source);
                const string basis="vocoder.mel_stft.stft_fn.forward_basis";
                if(!_convs.TryGetValue(basis,out var conv))
                {conv=new Conv{W=Value(basis),Oc=514,Ic=1,K=512,Stride=80,Dilation=1};_convs.Add(basis,conv);}
                // STFT bases are stored directly as [514,1,512], without a .weight suffix.
                using var spectrum=_ops.Conv1D(padded,conv);
                using var realView=spectrum.Narrow(1,0,257);using var imagView=spectrum.Narrow(1,257,257);
                using var real=Ops.NewContiguous(realView);using var imag=Ops.NewContiguous(imagView);
                using var mag=_ops.Context.NewF32(frames,257);using var imag2=_ops.Context.NewF32(frames,257);
                Ops.Mul(mag,real,real);Ops.Mul(imag2,imag,imag);Ops.Add(mag,mag,imag2);
                if(_ops.Context.IsCuda)
                {if(!_ops.TryCudaSqrt(mag))throw new NotSupportedException("LTX audio requires Float32 CUDA spectral magnitude support.");}
                else if(!_ops.Context.IsMlx) Ops.Pow(mag,mag,.5f);
                else if(!MlxLtx25AudioOps.TrySqrt(mag,mag)) throw new NotSupportedException("LTX audio requires Float32 MLX spectral magnitude support.");
                using var mel=_ops.Linear(mag,Value("vocoder.mel_stft.mel_basis"),64);
                if(_ops.Context.IsCuda)
                {if(!_ops.TryCudaLogClamp(mel))throw new NotSupportedException("LTX audio requires Float32 CUDA log-mel support.");}
                else {Ops.Clamp(mel,mel,1e-5f,float.MaxValue);Ops.Log(mel,mel);}
                using var slice=output.Narrow(1,channel*64,64);Ops.Copy(slice,mel);
            }
            return output;
        }
        catch{output.Dispose();throw;}
    }
    private Tensor Resample3(Tensor waveform)
    {
        int length=(int)waveform.Sizes[0];float[] source=waveform.GetElementsAsFloat(length*2),result=new float[length*6];
        const int width=7,kernelSize=43,padLeft=42;var kernel=new float[kernelSize];
        for(int k=0;k<kernelSize;k++)
        {
            float t=(k/3f-width)*.99f,c=Math.Clamp(t,-6,6);float window=MathF.Cos(c*MathF.PI/12);window*=window;
            float sinc=t==0?1:MathF.Sin(MathF.PI*t)/(MathF.PI*t);kernel[k]=sinc*window*.99f/3;
        }
        for(int t=0;t<length*3;t++)for(int c=0;c<2;c++)
        {
            float sum=0;for(int k=0;k<kernelSize;k++)
            {
                int n=t+padLeft-k;if(n%3!=0)continue;int sourceIndex=n/3-width;
                if(sourceIndex< -width||sourceIndex>=length+width)continue;sum+=source[Math.Clamp(sourceIndex,0,length-1)*2+c]*kernel[k];
            }
            result[t*2+c]=sum*3;
        }
        return _ops.Context.FromFloats(result,length*3,2);
    }
    public void Dispose()
    {
        lock(_gate){if(_disposed)return;_disposed=true;_ops.Dispose();_file.Dispose();_metadata?.Dispose();_values.Clear();_convs.Clear();_acts.Clear();_ownedAllocator?.Dispose();}
    }
}
