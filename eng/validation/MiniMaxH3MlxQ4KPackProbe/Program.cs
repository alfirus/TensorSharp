using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.MLX;
using Half=System.Half;

int iterations=args.Length > 0 ? int.Parse(args[0]) : 9;
if(iterations < 1) throw new ArgumentOutOfRangeException(nameof(iterations));
unsafe
{
    foreach(var shape in new[]{(Inner:5120,Rows:8192),(Inner:5120,Rows:25600),(Inner:25600,Rows:5120)})
    {
        int blocks=shape.Inner/256;
        long rawBytes=(long)shape.Rows*blocks*144,packedBytes=(long)shape.Rows*shape.Inner/2,parameters=(long)shape.Rows*blocks*8;
        IntPtr source=Alloc(rawBytes),baseline=Alloc(packedBytes),optimized=Alloc(packedBytes);
        IntPtr oldScales=Alloc(parameters*2),oldBiases=Alloc(parameters*2),newScales=Alloc(parameters*2),newBiases=Alloc(parameters*2);
        try
        {
            byte* raw=(byte*)source.ToPointer();
            for(long block=0;block<(long)shape.Rows*blocks;++block)
            {
                byte* row=raw+block*144;
                for(int i=0;i<144;++i) row[i]=(byte)((block*13+i*17)%256);
                ((Half*)row)[0]=(Half).125f; ((Half*)row)[1]=(Half).0625f;
            }
            for(int i=0;i<2;++i) foreach(int mode in new[]{0,1,2}) Run(mode);
            AssertSame(baseline,optimized,packedBytes); AssertSame(oldScales,newScales,parameters*2); AssertSame(oldBiases,newBiases,parameters*2);
            var times=new[]{new List<double>(),new List<double>(),new List<double>()};
            for(int i=0;i<iterations;++i)
                foreach(int mode in i%2 == 0 ? new[]{0,1,2} : new[]{2,1,0})
                { var watch=Stopwatch.StartNew(); Run(mode); watch.Stop(); times[mode].Add(watch.Elapsed.TotalMilliseconds); }
            foreach(var values in times) values.Sort();
            double oldMs=times[0][iterations/2],serialMs=times[1][iterations/2],parallelMs=times[2][iterations/2];
            Console.WriteLine(JsonSerializer.Serialize(new{shape=new[]{shape.Rows,shape.Inner},rawBytes,iterations,
                processorCount=Environment.ProcessorCount,baselineMedianMs=oldMs,fastSerialMedianMs=serialMs,parallelMedianMs=parallelMs,
                serialSpeedup=oldMs/serialMs,parallelSpeedup=oldMs/parallelMs,packedAndAffineParametersBitExact=true,
                scope="CPU repacking plus baseline required zeroing; excludes allocation, file paging, upload, F32 parameter preparation and inference"}));
            void Run(int mode)
            {
                if(mode == 0) Baseline(source,baseline,oldScales,oldBiases,shape.Inner,shape.Rows);
                else MlxQ4KRepack.Pack(source,optimized,newScales,newBiases,shape.Inner,shape.Rows,parallel:mode == 2);
            }
        }
        finally
        {
            foreach(IntPtr pointer in new[]{source,baseline,optimized,oldScales,oldBiases,newScales,newBiases}) NativeMemory.AlignedFree(pointer.ToPointer());
        }
    }
}

static unsafe IntPtr Alloc(long count) => (IntPtr)NativeMemory.AlignedAlloc((nuint)count,64);
static unsafe void AssertSame(IntPtr first,IntPtr second,long count)
{
    byte* a=(byte*)first.ToPointer(),b=(byte*)second.ToPointer();
    for(long i=0;i<count;++i) if(a[i]!=b[i]) throw new InvalidOperationException($"Packing mismatch at {i}");
}

// Original scalar row/block/group extraction followed by generic bit packing.
static unsafe void Baseline(IntPtr source,IntPtr packed,IntPtr scales,IntPtr biases,int inner,int rows)
{
    int blocks=inner/256,groups=blocks*8;
    byte* raw=(byte*)source.ToPointer(),destination=(byte*)packed.ToPointer();
    Half* scale=(Half*)scales.ToPointer(),bias=(Half*)biases.ToPointer();
    NativeMemory.Clear(destination,(nuint)((long)rows*inner/2));
    byte[] values=new byte[32];
    for(int row=0;row<rows;++row) for(int block=0;block<blocks;++block)
    {
        byte* input=raw+((long)row*blocks+block)*144;
        float d=(float)BitConverter.UInt16BitsToHalf((ushort)(input[0]|input[1]<<8));
        float min=(float)BitConverter.UInt16BitsToHalf((ushort)(input[2]|input[3]<<8));
        byte* p=input+4,q=input+16;
        for(int group=0;group<8;++group)
        {
            int s=group<4 ? p[group]&63 : (p[group+4]&15)|((p[group-4]>>6)<<4);
            int m=group<4 ? p[group+4]&63 : (p[group+4]>>4)|((p[group]>>6)<<4);
            byte* qGroup=q+(group/2)*32;
            for(int i=0;i<32;++i) values[i]=(byte)(group%2 != 0 ? qGroup[i]>>4 : qGroup[i]&15);
            scale[row*groups+block*8+group]=(Half)(d*s); bias[row*groups+block*8+group]=(Half)(-min*m);
            PackUnsignedBits(values,4,destination,(long)row*(inner/2)+block*128+group*16);
        }
    }
}

static unsafe void PackUnsignedBits(byte[] values,int bits,byte* destination,long offset)
{
    int position=0,mask=(1<<bits)-1;
    foreach(byte value in values)
    {
        int index=(int)(offset+position/8),shift=position&7;
        uint shifted=(uint)((value&mask)<<shift);
        destination[index]|=(byte)shifted;
        if(shifted>255) destination[index+1]|=(byte)(shifted>>8);
        if(shifted>65535) destination[index+2]|=(byte)(shifted>>16);
        position+=bits;
    }
}
