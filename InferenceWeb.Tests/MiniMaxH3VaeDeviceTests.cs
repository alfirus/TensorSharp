using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Models;
using TensorSharp.Models.MiniMaxH3;
using System.Runtime.InteropServices;
using Half = System.Half;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3VaeDeviceTests
{
    [MlxFact] public void MlxReflectColumnsMatchScalarAtCornersAndBandOffset() => Mlx(ReflectColumns);
    [CudaFact] public void CudaReflectColumnsMatchScalarAtCornersAndBandOffset() => Cuda(ReflectColumns);
    [MlxFact] public void MlxAudioConvolutionsMatchScalarIncludingTransposedOverlap() => Mlx(AudioConvolution);
    [CudaFact] public void CudaAudioConvolutionsMatchScalarIncludingTransposedOverlap() => Cuda(AudioConvolution);
    [MlxFact] public void MlxAliasFreeActivationMatchesIndependentScalar() => Mlx(AudioActivation);
    [CudaFact] public void CudaAliasFreeActivationMatchesIndependentScalar() => Cuda(AudioActivation);
    [MlxFact] public void MlxFrameGroupNormMatchesCenteredDoubleOracleInPlace() => Mlx(GroupNorm);
    [CudaFact] public void CudaFrameGroupNormMatchesCenteredDoubleOracleInPlace() => Cuda(GroupNorm);
    [CudaFact]
    public void CudaTransposeCopyPreservesEveryElementOnDevice() => Cuda(allocator =>
    {
        float[] data=Enumerable.Range(0,3*4*2*5).Select(i => i*.25f-7).ToArray();
        using var input=new Tensor(allocator,DType.Float32,3,4,2,5);
        input.SetElementsAsFloat(data);
        using var view=input.Permute(2,0,3,1);
        using var output=new Tensor(allocator,DType.Float32,2,3,5,4);
        Assert.True(CudaMiniMaxH3Ops.TryCopyStrided(output,view));
        float[] actual=output.GetElementsAsFloat(data.Length);
        for(int z=0;z<2;z++)for(int x=0;x<3;x++)for(int w=0;w<5;w++)for(int y=0;y<4;y++)
            Assert.Equal(data[((x*4+y)*2+z)*5+w],actual[((z*3+x)*5+w)*4+y]);
    });
    [CudaFact]
    public void CudaWaveformClampPreservesFloat32StorageAndInPlaceViews() => Cuda(allocator =>
    {
        float[,] data={{37,37,37,37,37},{-80000,-.125f,0,.125f,80000},{37,37,37,37,37}};
        using var storage=Tensor.FromArray(allocator,data);
        using var row=storage.Narrow(0,1,1);
        Ops.Clamp(row,row,-10000,10000);
        Assert.Equal(new float[]{37,37,37,37,37,-10000,-.125f,0,.125f,10000,37,37,37,37,37},storage.GetElementsAsFloat(15));
        using var separate=Ops.Clamp(null,row,-1,1);
        Assert.Equal(new float[]{-1,-.125f,0,.125f,1},separate.GetElementsAsFloat(5));
    });
    [MlxFact]
    public void MlxWaveformClampPreservesFloat32RangeViewsAndStrideGuards() => Mlx(allocator =>
    {
        float[,] data={{37,37,37,37,37},{float.NegativeInfinity,-80000,-.125f,80000,float.PositiveInfinity},{37,37,37,37,37}};
        using var storage=Tensor.FromArray(allocator,data);
        using var row=storage.Narrow(0,1,1);
        Ops.Clamp(row,row,-100000,100000);
        Assert.Equal(new float[]{37,37,37,37,37,-100000,-80000,-.125f,80000,100000,37,37,37,37,37},storage.GetElementsAsFloat(15));
        using var column=storage.Narrow(1,1,1);
        using var output=new Tensor(allocator,DType.Float32,3,1);
        Assert.True(MlxMiniMaxH3Ops.TryClamp(output,column,-1,1));
        Assert.Equal(new float[]{1,-1,1},output.GetElementsAsFloat(3));
        Assert.False(MlxMiniMaxH3Ops.TryClamp(column,output,-1,1));
        Assert.False(MlxMiniMaxH3Ops.TryClamp(output,column,2,1));
        Assert.Equal(new float[]{37,37,37,37,37,-100000,-80000,-.125f,80000,100000,37,37,37,37,37},storage.GetElementsAsFloat(15));
    });
    [CudaFact]
    public void CudaMappedHalfWeightsPreserveLargeFloat32InputsAcrossColumnBands() => Cuda(allocator => Mapped16BitProjections(allocator,1));
    [CudaFact]
    public void CudaMappedBFloat16WeightsPreserveLargeFloat32InputsAcrossColumnBands() => Cuda(allocator => Mapped16BitProjections(allocator,30));

    [CudaFact]
    public void CudaH3QuantizedBandsPreserveRangeAcrossLargeRowsAndPartialColumns() => Cuda(allocator =>
    {
        const int inner=1024,columns=129,maximumRows=256,budget=65*inner*sizeof(float);
        foreach(int type in new[]{2,8,12,14})
        {
            byte[] bytes=QuantBandWeights(type,inner,columns);
            float[] weights=new float[inner*columns];
            ManagedQuantizedOps.DequantizeToFloat32(type,bytes,0,weights,0,weights.Length);
            double[] expected=new double[maximumRows*columns];
            for(int row=0;row<maximumRows;row++)for(int column=0;column<columns;column++)
                for(int k=0;k<inner;k++)expected[row*columns+column]+=((row*11+k*13)%31-15)*(double)weights[column*inner+k];
            IntPtr host=Marshal.AllocHGlobal(bytes.Length);
            try
            {
                Marshal.Copy(bytes,0,host,bytes.Length);
                using var context=new DirectContext(allocator);
                foreach(int rows in new[]{1,16,33,256})foreach(float amplitude in new[]{1f/65536,1f/64,65536f})
                {
                    float[] values=Enumerable.Range(0,rows*inner).Select(i=>((i/inner*11+i%inner*13)%31-15)*amplitude).ToArray();
                    using var input=context.FromFloats(values,rows,inner);
                    using var output=context.NewF32(rows,columns);
                    Assert.True(CudaMiniMaxH3Ops.TryQuantizedBands(output,input,host,type,inner,columns,bytes.Length,budget));
                    float[] actual=output.GetElementsAsFloat(rows*columns);
                    for(int i=0;i<actual.Length;i++)
                    {
                        Assert.True(float.IsFinite(actual[i]));
                        Assert.Equal((float)(expected[i]*amplitude),actual[i]);
                    }
                    Assert.False(CudaMiniMaxH3Ops.TryQuantizedBands(output,input,host,type,inner,columns,bytes.Length-1,budget));
                    Assert.False(CudaMiniMaxH3Ops.TryQuantizedBands(output,input,host,type,inner,columns,bytes.Length,inner*sizeof(float)-1));
                }
            }
            finally { CudaQuantizedOps.ReleaseQuantizedWeight((CudaAllocator)allocator,host); Marshal.FreeHGlobal(host); }
        }
    });

    [CudaFact]
    public void CudaH3QuantizedBandsPedanticRetainsLargeNearCancellationResiduals() => Cuda(allocator =>
    {
        const int inner=1024,columns=129,rows=256;
        byte[] bytes=QuantBandWeights(8,inner,columns,paired:true);
        float[] weights=new float[inner*columns];
        ManagedQuantizedOps.DequantizeToFloat32(8,bytes,0,weights,0,weights.Length);
        float[] values=Enumerable.Range(0,rows*inner).Select(i=>i%2==0 ? 131104f : -131072f).ToArray();
        IntPtr host=Marshal.AllocHGlobal(bytes.Length);
        try
        {
            Marshal.Copy(bytes,0,host,bytes.Length);
            using var context=new DirectContext(allocator);
            using var input=context.FromFloats(values,rows,inner);
            using var output=context.NewF32(rows,columns);
            Assert.True(CudaMiniMaxH3Ops.TryQuantizedBands(output,input,host,8,inner,columns,bytes.Length,forcePedantic:true));
            float[] actual=output.GetElementsAsFloat(rows*columns);
            using var dispatched=context.NewF32(rows,columns);
            Assert.True(CudaMiniMaxH3Ops.TryLinear(dispatched,input,host,8,inner,columns,bytes.Length));
            float[] defaultActual=dispatched.GetElementsAsFloat(rows*columns);
            for(int column=0;column<columns;column++)
            {
                double expected=0;
                for(int k=0;k<inner;k++)expected+=(double)values[k]*weights[column*inner+k];
                Assert.NotEqual(0,expected);
                for(int row=0;row<rows;row++)
                {
                    Assert.Equal((float)expected,actual[row*columns+column]);
                    Assert.Equal((float)expected,defaultActual[row*columns+column]);
                }
            }
        }
        finally { CudaQuantizedOps.ReleaseQuantizedWeight((CudaAllocator)allocator,host); Marshal.FreeHGlobal(host); }
    });

    private static byte[] QuantBandWeights(int type,int inner,int columns,bool paired=false)
    {
        int block=type is 2 or 8 ? 32 : 256;
        int size=type switch {2=>18,8=>34,12=>144,14=>210,_=>throw new ArgumentOutOfRangeException(nameof(type))};
        byte[] bytes=new byte[columns*(inner/block)*size];
        for(int column=0;column<columns;column++)for(int b=0;b<inner/block;b++)
        {
            int start=(column*(inner/block)+b)*size;
            for(int i=0;i<size;i++)bytes[start+i]=(byte)((column*17+b*13+i*7)%256);
            int scaleOffset=type==14 ? 208 : 0;
            BitConverter.TryWriteBytes(bytes.AsSpan(start+scaleOffset),BitConverter.HalfToUInt16Bits((Half)(type==8 ? .125f : 1f/64)));
            if(type==12)
            {
                BitConverter.TryWriteBytes(bytes.AsSpan(start+2),BitConverter.HalfToUInt16Bits((Half)(1f/128)));
                for(int i=0;i<12;i++)bytes[start+4+i]=(byte)(1+(column+b+i)%7);
            }
            if(type==14)for(int i=0;i<16;i++)bytes[start+192+i]=unchecked((byte)(sbyte)((i%2==0 ? 1 : -1)*(1+(column+b+i)%7)));
            if(type==8)for(int i=0;i<32;i++)bytes[start+2+i]=unchecked((byte)(sbyte)(paired ? 1+(column+b*16+i/2)%7 : (column*17+b*13+i*7)%63-31));
        }
        return bytes;
    }

    private static void Mapped16BitProjections(IAllocator allocator,int type)
    {
        foreach(var geometry in new[]{(Inner:8,Columns:19,Rows:3,Budget:64),(Inner:1024,Columns:513,Rows:32,Budget:64<<20)})
            Mapped16BitProjection(allocator,type,geometry.Inner,geometry.Columns,geometry.Rows,geometry.Budget);
    }

    [CudaFact]
    public void CudaH3Q8ProjectionPreservesTinyAndLargeFloat32InputsAtEveryDispatchBoundary() => Cuda(allocator =>
    {
        const int inner=64,columns=11,blockBytes=34;
        var bytes=new byte[columns*(inner/32)*blockBytes];
        var weights=new float[inner*columns];
        for(int column=0;column<columns;column++)for(int block=0;block<inner/32;block++)
        {
            int offset=(column*(inner/32)+block)*blockBytes;
            Half scale=(Half)(block==0 ? .125f : .0625f);
            BitConverter.TryWriteBytes(bytes.AsSpan(offset),BitConverter.HalfToUInt16Bits(scale));
            for(int k=0;k<32;k++)
            {
                sbyte quant=(sbyte)(((column*17+block*13+k*7)%63)-31);
                bytes[offset+2+k]=unchecked((byte)quant);
                weights[column*inner+block*32+k]=(float)scale*quant;
            }
        }
        IntPtr host=Marshal.AllocHGlobal(bytes.Length);
        try
        {
            Marshal.Copy(bytes,0,host,bytes.Length);
            using var context=new DirectContext(allocator);
            // Binary amplitudes keep every product and 64-term sum exactly
            // representable, so cancellation cannot hide behind relative error.
            foreach(int rows in new[]{1,16,22,33,118})foreach(float amplitude in new[]{1f/65536,1f/64,65536f})
            {
                float[] values=Enumerable.Range(0,rows*inner).Select(i=>((i*13)%31-15)*amplitude).ToArray();
                using var input=context.FromFloats(values,rows,inner);
                using var output=context.NewF32(rows,columns);
                Assert.True(CudaMiniMaxH3Ops.TryLinear(output,input,host,8,inner,columns,bytes.Length));
                float[] actual=output.GetElementsAsFloat(rows*columns);
                for(int row=0;row<rows;row++)for(int column=0;column<columns;column++)
                {
                    double expected=0;
                    for(int k=0;k<inner;k++)expected+=(double)values[row*inner+k]*weights[column*inner+k];
                    Assert.True(float.IsFinite(actual[row*columns+column]));
                    Assert.Equal((float)expected,actual[row*columns+column]);
                }
            }
        }
        finally
        {
            CudaQuantizedOps.ReleaseQuantizedWeight((CudaAllocator)allocator,host);
            Marshal.FreeHGlobal(host);
        }
    });

    private static void Mapped16BitProjection(IAllocator allocator,int type,int inner,int columns,int rows,int budget)
    {
        var weight=new Half[inner*columns];
        for(int i=0;i<weight.Length;i++)weight[i]=(Half)(((i*7)%17-8)/8f);
        var bytes=new byte[weight.Length*sizeof(ushort)];
        for(int i=0;i<weight.Length;i++)BitConverter.TryWriteBytes(bytes.AsSpan(i*sizeof(ushort)),type == 1
            ? BitConverter.HalfToUInt16Bits(weight[i]) : (ushort)(BitConverter.SingleToUInt32Bits((float)weight[i])>>16));
        IntPtr host=Marshal.AllocHGlobal(bytes.Length);
        try
        {
            Marshal.Copy(bytes,0,host,bytes.Length);
            using var context=new DirectContext(allocator);
            float[] values=Enumerable.Range(0,rows*inner).Select(i => ((i*13)%31-15)*8000f).ToArray();
            using var input=context.FromFloats(values,rows,inner);
            using var output=context.NewF32(rows,columns);
            Assert.True(CudaMiniMaxH3Ops.TryMappedDense(output,input,host,type,inner,columns,bytes.Length,maximumConvertedWeightBytes:budget));
            float[] actual=output.GetElementsAsFloat(rows*columns);
            using var linear=DirectLinear.FromMappedH3Dense(context,host,type,inner,columns,bytes.Length,null);
            using var factoryResult=linear.Forward(input);
            Assert.Equal(actual,factoryResult.GetElementsAsFloat(actual.Length));
            for(int row=0;row<rows;row++)for(int column=0;column<columns;column++)
            {
                double expected=0;
                for(int k=0;k<inner;k++)expected+=(double)values[row*inner+k]*(float)weight[column*inner+k];
                Assert.True(float.IsFinite(actual[row*columns+column]));
                Assert.Equal((float)expected,actual[row*columns+column]);
            }
        }
        finally { Marshal.FreeHGlobal(host); }
    }

    private static void Mlx(Action<IAllocator> run) => MlxWorker.Shared.Invoke(() =>
    { using var allocator = new MlxAllocator(); run(allocator); });
    private static void Cuda(Action<IAllocator> run)
    { using var allocator = new CudaAllocator(); run(allocator); }

    private static void ReflectColumns(IAllocator allocator)
    {
        using var operations = new MiniMaxH3AudioTensorOps(allocator);
        const int width = 5, height = 4, channels = 3, kernel = 3, bandStart = 2, bandRows = 2;
        float[] values = Enumerable.Range(0,width*height*channels).Select(i => (float)i).ToArray();
        using var input = operations.Context.FromFloats(values,height,width,channels);
        using var backing = operations.Context.NewF32((bandRows+1)*width,channels*kernel*kernel);
        using var output = backing.Narrow(0,width,bandRows*width);
        operations.Transform(output,input,null,null,0,channels,height,width,kernel,width,1,1,1,bandStart);
        float[] actual = output.GetElementsAsFloat(checked((int)output.ElementCount()));
        for (int row=0;row<bandRows*width;row++) for(int ch=0;ch<channels;ch++)
        for(int ky=0;ky<kernel;ky++) for(int kx=0;kx<kernel;kx++)
        {
            int y=bandStart+row/width-1+ky, x=row%width-1+kx;
            if(y<0)y=-y; if(y>=height)y=2*(height-1)-y;
            if(x<0)x=-x; if(x>=width)x=2*(width-1)-x;
            Assert.Equal(values[(y*width+x)*channels+ch],actual[row*channels*kernel*kernel+(ch*kernel+ky)*kernel+kx]);
        }
    }

    private static void AudioConvolution(IAllocator allocator)
    {
        using var operations = new MiniMaxH3AudioTensorOps(allocator);
        const int time=11, channels=3, outputChannels=4, kernel=5;
        float[] values=Enumerable.Range(0,time*channels).Select(i => MathF.Sin(i*.37f)).ToArray();
        using var input=operations.Context.FromFloats(values,time,channels);
        foreach(bool transpose in new[]{false,true})
        {
            var convolution=new MiniMaxH3DirectAudioVae.Conv
            {
                W=Enumerable.Range(0,channels*outputChannels*kernel).Select(i => MathF.Cos(i*.29f)/17).ToArray(),
                B=transpose ? null : [.1f,-.2f,.3f,-.4f],
                K=kernel, Ic=transpose ? outputChannels : channels, Oc=transpose ? channels : outputChannels,
                Stride=2, Pad=3, Dilation=2, Transposed=transpose,
            };
            using var result=operations.Conv(input,convolution,transpose ? 3 : 0);
            float[] actual=result.GetElementsAsFloat(checked((int)result.ElementCount()));
            int length=checked((int)result.Sizes[0]);
            for(int t=0;t<length;t++) for(int o=0;o<outputChannels;o++)
            {
                double expected=0;
                for(int c=0;c<channels;c++) for(int k=0;k<kernel;k++)
                {
                    int source=transpose ? t-k : t*2-3+k*2;
                    if(transpose && source%3 != 0)continue;
                    if(transpose)source/=3;
                    if((uint)source >= time)continue;
                    int weight=transpose ? (c*outputChannels+o)*kernel+k : (o*channels+c)*kernel+k;
                    expected+=(double)values[source*channels+c]*convolution.W[weight];
                }
                if(convolution.B != null)expected+=convolution.B[o];
                Assert.InRange(Math.Abs(expected-actual[t*outputChannels+o]),0,2e-6);
            }
        }
    }

    private static void AudioActivation(IAllocator allocator)
    {
        using var operations=new MiniMaxH3AudioTensorOps(allocator);
        const int time=9, channels=2, kernel=12;
        float[] values=Enumerable.Range(0,time*channels).Select(i => MathF.Sin(i*.47f)*2).ToArray();
        var activation=new MiniMaxH3DirectAudioVae.Act
        {
            Alpha=[.7f,1.3f], Beta=[1.1f,.8f], Channels=channels, Kernel=kernel,
            UpFilter=Enumerable.Range(0,kernel).Select(i => MathF.Sin(i*.31f)/8).ToArray(),
            DownFilter=Enumerable.Range(0,kernel).Select(i => MathF.Cos(i*.27f)/9).ToArray(),
        };
        using var input=operations.Context.FromFloats(values,time,channels);
        using var result=operations.Activation(input,activation);
        float[] actual=result.GetElementsAsFloat(checked((int)result.ElementCount()));
        const int pad=kernel/2-1, left=pad*2+(kernel-2)/2, right=pad*2+(kernel-1)/2;
        int midTime=(time+2*pad-1)*2+kernel-left-right;
        var mid=new double[midTime*channels];
        for(int t=0;t<midTime;t++) for(int c=0;c<channels;c++)
        {
            double sum=0;
            for(int k=0;k<kernel;k++)
            {
                int stuffed=t+left-(kernel-1)+k;
                if(stuffed<0 || stuffed >= (time+2*pad)*2 || stuffed%2 != 0)continue;
                int source=Math.Clamp(stuffed/2-pad,0,time-1);
                sum+=(double)activation.UpFilter[k]*values[source*channels+c];
            }
            double value=sum*2, sine=Math.Sin(value*activation.Alpha[c]);
            mid[t*channels+c]=value+sine*sine/activation.Beta[c];
        }
        Assert.Equal(time,result.Sizes[0]);
        for(int t=0;t<time;t++) for(int c=0;c<channels;c++)
        {
            double expected=0;
            for(int k=0;k<kernel;k++) expected+=activation.DownFilter[k]*mid[Math.Clamp(t*2-(kernel/2-1)+k,0,midTime-1)*channels+c];
            Assert.InRange(Math.Abs(expected-actual[t*channels+c]),0,5e-6);
        }
    }

    private static void GroupNorm(IAllocator allocator)
    {
        using var context=new DirectContext(allocator);
        const int frames=2, height=19, width=17, channels=8, groups=4, rows=height*width, groupWidth=channels/groups;
        float[] values=Enumerable.Range(0,frames*rows*channels).Select(i => MathF.Sin(i*.73f)*80000+(i/(rows*channels))*12000).ToArray();
        float[] gains=Enumerable.Range(0,channels).Select(i => .6f+i*.03f).ToArray();
        float[] biases=Enumerable.Range(0,channels).Select(i => -.2f+i*.07f).ToArray();
        using var input=context.FromFloats(values,frames,height,width,channels);
        using var gain=context.FromFloats(gains,channels); using var bias=context.FromFloats(biases,channels);
        bool success=context.IsCuda ? CudaMiniMaxH3Ops.TryGroupNormSilu(input,input,gain,bias,groups,1e-6f)
            : MlxMiniMaxH3Ops.TryGroupNormSilu(input,input,gain,bias,groups,1e-6f);
        Assert.True(success);
        float[] actual=input.GetElementsAsFloat(values.Length);
        for(int f=0;f<frames;f++) for(int g=0;g<groups;g++)
        {
            double sum=0;
            for(int row=0;row<rows;row++) for(int c=0;c<groupWidth;c++) sum+=values[(f*rows+row)*channels+g*groupWidth+c];
            float mean=(float)(sum/(rows*groupWidth));
            double squares=0;
            for(int row=0;row<rows;row++) for(int c=0;c<groupWidth;c++)
            { float v=values[(f*rows+row)*channels+g*groupWidth+c]-mean; squares+=(double)v*v; }
            double inverse=1/Math.Sqrt(squares/(rows*groupWidth)+1e-6);
            for(int row=0;row<rows;row++) for(int c=0;c<groupWidth;c++)
            {
                int ch=g*groupWidth+c,index=(f*rows+row)*channels+ch;
                double v=(values[index]-mean)*inverse*gains[ch]+biases[ch],expected=v/(1+Math.Exp(-v));
                Assert.InRange(Math.Abs(expected-actual[index]),0,2e-5);
            }
        }
    }
}
