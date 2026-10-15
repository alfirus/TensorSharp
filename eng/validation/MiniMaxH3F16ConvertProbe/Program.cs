using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.Runtime;

int iterations=args.Length>0 ? int.Parse(args[0]) : 9;
if(iterations<1) throw new ArgumentOutOfRangeException(nameof(iterations));
unsafe
{
    foreach(int count in new[]{1<<20,16<<20})
    {
        ushort* input=(ushort*)NativeMemory.AlignedAlloc((nuint)count*sizeof(ushort),64);
        float* scalar=(float*)NativeMemory.AlignedAlloc((nuint)count*sizeof(float),64);
        float* vector=(float*)NativeMemory.AlignedAlloc((nuint)count*sizeof(float),64);
        try
        {
            for(int i=0;i<count;++i) input[i]=(ushort)(((i*73)%0x7C00)|((i%2)<<15));
            for(int i=0;i<2;++i) { Scalar(input,scalar,count); SafetensorsKernels.F16ToF32(input,vector,count); }
            for(int i=0;i<count;++i)
                if(BitConverter.SingleToInt32Bits(scalar[i])!=BitConverter.SingleToInt32Bits(vector[i]))
                    throw new InvalidOperationException($"F16 widening mismatch at {i}");
            var scalarTimes=new List<double>(); var vectorTimes=new List<double>();
            for(int i=0;i<iterations;++i)
            {
                if(i%2 == 0) { Measure(false,scalarTimes); Measure(true,vectorTimes); }
                else { Measure(true,vectorTimes); Measure(false,scalarTimes); }
            }
            scalarTimes.Sort();vectorTimes.Sort();
            double scalarMs=scalarTimes[iterations/2],vectorMs=vectorTimes[iterations/2];
            Console.WriteLine(JsonSerializer.Serialize(new{count,iterations,scalarMedianMs=scalarMs,vectorMedianMs=vectorMs,
                speedup=scalarMs/vectorMs,finiteInputOutputBitsIdentical=true,
                scope="resident synthetic CPU conversion only; excludes file paging, allocation, storage copy, device upload and inference"}));
            void Measure(bool optimized,List<double> times)
            {
                var watch=Stopwatch.StartNew();
                if(optimized) SafetensorsKernels.F16ToF32(input,vector,count);
                else Scalar(input,scalar,count);
                watch.Stop();times.Add(watch.Elapsed.TotalMilliseconds);
            }
        }
        finally { NativeMemory.AlignedFree(input);NativeMemory.AlignedFree(scalar);NativeMemory.AlignedFree(vector); }
    }
}

static unsafe void Scalar(ushort* input,float* output,int count)
{
    for(int i=0;i<count;++i) output[i]=(float)BitConverter.UInt16BitsToHalf(input[i]);
}
