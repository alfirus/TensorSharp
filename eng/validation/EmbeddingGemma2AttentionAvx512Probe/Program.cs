using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models.Embeddings;

string output = args.Length == 2 && args[0] == "--output" ? Path.GetFullPath(args[1])
    : throw new ArgumentException("Pass --output <ignored evidence JSON path>.");
if (!ManagedEmbeddingAttentionAvx512.IsSupported)
    throw new PlatformNotSupportedException("This probe requires actual AVX512F and AVX2 hardware; no scenario is counted as passing.");
var reports = new List<object>();
var shapes = new (int Dimensions, int Length, int Offset, int Queries, int Window)[]
{
    (16,1,0,1,0), (256,7,3,3,0), (512,13,1,4,12), (256,31,7,4,12),
    (512,32,0,4,0), (256,33,5,3,12), (256,511,7,4,512),
    (512,513,3,4,512), (256,1025,11,4,512), (512,1025,5,4,0), (512,8192,0,4,0)
};
foreach (var shape in shapes) reports.Add(Validate(shape.Dimensions,shape.Length,shape.Offset,shape.Queries,shape.Window));
var timings = new[] { Time(256,511), Time(512,1025), Time(512,8192) };
var report = new
{
    completed = true, cpu_count = Environment.ProcessorCount,
    avx512f = Avx512F.IsSupported, avx2 = Avx2.IsSupported, fma = Fma.IsSupported,
    runtime = Environment.Version.ToString(), architecture = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(),
    binary_sha256 = Hash(typeof(Program).Assembly.Location), cases = reports, timings,
    limitations = "Isolated four-query arithmetic and warmed single-thread kernel timing only. Packing, worker scheduling, softmax, model inference and HTTP are excluded."
};
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
File.WriteAllText(output, JsonSerializer.Serialize(report,new JsonSerializerOptions{WriteIndented=true})+"\n");
Console.WriteLine(JsonSerializer.Serialize(new { completed = true, cases = reports.Count, timings }));

static string Hash(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
}

static unsafe object Validate(int dimensions,int length,int offset,int queries,int window)
{
    int stride = length + offset;
    var q = new float[dimensions*4]; var key = new float[dimensions*stride];
    var value = new float[length*dimensions]; var packed = new float[dimensions*stride];
    var scores = new float[length*4+2]; var probabilities = new float[length*4];
    const int guard = 3;
    int outStride = dimensions+13;
    var result = new float[guard+outStride*4+2]; Array.Fill(result,37); Array.Fill(scores,37);
    for(int row=0;row<4;++row) for(int d=0;d<dimensions;++d)
        q[row*dimensions+d]=(float)Math.Sin((row+1)*.71+(d+1)*.013);
    for(int k=0;k<length;++k) for(int d=0;d<dimensions;++d)
    {
        key[d*stride+offset+k]=(float)Math.Cos((k+1)*.23+(d+1)*.053);
        value[k*dimensions+d]=(float)Math.Sin((k+7)*.17+(d+2)*.031);
        packed[(d/16*16)*stride+(offset+k)*16+d%16]=value[k*dimensions+d];
    }
    for(int row=0;row<4;++row)
    {
        int position=row==0?0:row==3?length-1:length/2+row-1;
        double sum=0;
        for(int k=0;k<length;++k)
        {
            float p=window>0&&Math.Abs(k-position)>window?0:(float)Math.Exp(Math.Sin(k*.17+row*.31));
            probabilities[row*length+k]=p; sum+=p;
        }
        for(int k=0;k<length;++k) probabilities[row*length+k]/=(float)sum;
    }
    fixed(float* qp=q,kp=key,vp=packed,sp=scores,pp=probabilities,op=result)
    {
        ManagedEmbeddingAttentionAvx512.Scores(qp,qp+dimensions,qp+2*dimensions,qp+3*dimensions,
            kp+offset,stride,dimensions,length,.125f,sp+1);
        ManagedEmbeddingAttentionAvx512.Values(pp,length,vp+offset*16,stride,dimensions,op+guard,outStride,queries);
    }
    double scoreError=0,valueError=0;
    for(int row=0;row<4;++row) for(int k=0;k<length;++k)
    {
        float scalar=0; double expected=0,absSum=0;
        for(int d=0;d<dimensions;++d)
        {
            float a=q[row*dimensions+d],b=key[d*stride+offset+k];
            scalar=MathF.FusedMultiplyAdd(a,b,scalar); expected+=(double)a*b; absSum+=Math.Abs((double)a*b);
        }
        scalar*=.125f; expected*=.125; absSum*=.125;
        float actual=scores[1+row*length+k];
        if(BitConverter.SingleToInt32Bits(actual)!=BitConverter.SingleToInt32Bits(scalar)) throw new Exception("Scores changed F32 FMA order.");
        double error=Math.Abs(actual-expected); scoreError=Math.Max(scoreError,error);
        if(error>Math.Max(1,absSum)*2e-5) throw new Exception("Scores differ from the double oracle.");
    }
    for(int row=0;row<queries;++row) for(int d=0;d<dimensions;++d)
    {
        float scalar=0; double expected=0,absSum=0;
        for(int k=0;k<length;++k)
        {
            float a=probabilities[row*length+k],b=value[k*dimensions+d];
            scalar=MathF.FusedMultiplyAdd(a,b,scalar); expected+=(double)a*b; absSum+=Math.Abs((double)a*b);
        }
        float actual=result[guard+row*outStride+d];
        if(BitConverter.SingleToInt32Bits(actual)!=BitConverter.SingleToInt32Bits(scalar)) throw new Exception("Values changed F32 FMA order.");
        double error=Math.Abs(actual-expected); valueError=Math.Max(valueError,error);
        if(error>Math.Max(1,absSum)*2e-5) throw new Exception("Values differ from the double oracle.");
    }
    for(int index=0;index<result.Length;++index)
    {
        int p=index-guard;
        if(p<0 || p/outStride>=queries || p%outStride>=dimensions)
            if(result[index]!=37) throw new Exception("Values overwrote a guard or inactive query.");
    }
    if(scores[0]!=37 || scores[^1]!=37) throw new Exception("Scores overwrote a guard.");
    Console.WriteLine($"PASS D={dimensions} keys={length} offset={offset} queries={queries} window={window} scores={scoreError:R} values={valueError:R}");
    return new { dimensions,length,offset,queries,window,score_max_absolute_error=scoreError,value_max_absolute_error=valueError,
        f32_fma_bit_exact=true,passed=true };
}

static unsafe object Time(int dimensions,int length)
{
    int stride=(length+31)/16*16;
    var q=new float[dimensions*4]; var key=new float[dimensions*stride];
    var v8=new float[dimensions*stride]; var v16=new float[dimensions*stride];
    var scores=new float[length*4]; var p=new float[length*4]; var result=new float[dimensions*4];
    for(int i=0;i<q.Length;++i)q[i]=(float)Math.Sin(i*.017);
    for(int d=0;d<dimensions;++d) for(int k=0;k<length;++k)
    {
        key[d*stride+k]=(float)Math.Cos(k*.13+d*.011);
        float value=(float)Math.Sin(k*.19+d*.021);
        v8[d/8*8*stride+k*8+d%8]=value;
        v16[d/16*16*stride+k*16+d%16]=value;
    }
    for(int i=0;i<p.Length;++i)p[i]=1f/length;
    fixed(float* qp=q,kp=key,v8p=v8,v16p=v16,sp=scores,pp=p,op=result)
    {
        nint qa=(nint)qp,ka=(nint)kp,sa=(nint)sp,pa=(nint)pp,oa=(nint)op,va8=(nint)v8p,va16=(nint)v16p;
        Action s512=()=>ManagedEmbeddingAttentionAvx512.Scores((float*)qa,(float*)qa+dimensions,(float*)qa+2*dimensions,(float*)qa+3*dimensions,(float*)ka,stride,dimensions,length,1,(float*)sa);
        Action v512=()=>ManagedEmbeddingAttentionAvx512.Values((float*)pa,length,(float*)va16,stride,dimensions,(float*)oa,dimensions,4);
        Action s256=()=>Scores256((float*)qa,(float*)ka,stride,dimensions,length,(float*)sa);
        Action v256=()=>Values256((float*)pa,length,(float*)va8,stride,dimensions,(float*)oa);
        double scores512=Median(s512),scores256=Median(s256),values512=Median(v512),values256=Median(v256);
        return new{dimensions,length,scores_avx512_us=scores512,scores_avx2_us=scores256,scores_ratio=scores512/scores256,
            values_avx512_us=values512,values_avx2_us=values256,values_ratio=values512/values256};
    }
}

static double Median(Action action)
{
    for(int i=0;i<20;++i)action();
    var times=new List<double>();
    for(int sample=0;sample<9;++sample)
    {
        long start=Stopwatch.GetTimestamp();
        for(int i=0;i<20;++i)action();
        times.Add(Stopwatch.GetElapsedTime(start).TotalMicroseconds/20);
    }
    times.Sort();return times[times.Count/2];
}

static unsafe void Scores256(float* q,float* keys,int stride,int dimensions,int length,float* output)
{
    for(int key=0;key<length;key+=8)
    {
        var s0=Vector256<float>.Zero;var s1=Vector256<float>.Zero;var s2=Vector256<float>.Zero;var s3=Vector256<float>.Zero;
        for(int d=0;d<dimensions;++d)
        {
            var k=Avx.LoadVector256(keys+(long)d*stride+key);
            s0=Fma.MultiplyAdd(k,Vector256.Create(q[d]),s0);s1=Fma.MultiplyAdd(k,Vector256.Create(q[dimensions+d]),s1);
            s2=Fma.MultiplyAdd(k,Vector256.Create(q[2*dimensions+d]),s2);s3=Fma.MultiplyAdd(k,Vector256.Create(q[3*dimensions+d]),s3);
        }
        int count=Math.Min(8,length-key);
        if(count==8)
        {Avx.Store(output+key,s0);Avx.Store(output+length+key,s1);Avx.Store(output+2*length+key,s2);Avx.Store(output+3*length+key,s3);}
        else for(int lane=0;lane<count;++lane)
        {output[key+lane]=s0.GetElement(lane);output[length+key+lane]=s1.GetElement(lane);output[2*length+key+lane]=s2.GetElement(lane);output[3*length+key+lane]=s3.GetElement(lane);}
    }
}

static unsafe void Values256(float* p,int length,float* values,int stride,int dimensions,float* output)
{
    for(int d=0;d<dimensions;d+=8)
    {
        var s0=Vector256<float>.Zero;var s1=Vector256<float>.Zero;var s2=Vector256<float>.Zero;var s3=Vector256<float>.Zero;
        for(int key=0;key<length;++key)
        {
            var v=Avx.LoadVector256(values+(long)d*stride+key*8);
            s0=Fma.MultiplyAdd(v,Vector256.Create(p[key]),s0);s1=Fma.MultiplyAdd(v,Vector256.Create(p[length+key]),s1);
            s2=Fma.MultiplyAdd(v,Vector256.Create(p[2*length+key]),s2);s3=Fma.MultiplyAdd(v,Vector256.Create(p[3*length+key]),s3);
        }
        Avx.Store(output+d,s0);Avx.Store(output+dimensions+d,s1);Avx.Store(output+2*dimensions+d,s2);Avx.Store(output+3*dimensions+d,s3);
    }
}
