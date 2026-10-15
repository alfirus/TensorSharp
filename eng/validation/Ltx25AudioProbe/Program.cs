using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;

string Arg(string name,string fallback="") {int i=Array.IndexOf(args,name);return i>=0?args[i+1]:fallback;}
string checkpoint=Arg("--checkpoint"),backend=Arg("--backend","cpu"),oraclePath=Arg("--oracle"),outputPath=Arg("--output");
int repeats=int.Parse(Arg("--repeats","1"));
using var oracle=JsonDocument.Parse(File.ReadAllText(oraclePath));
IAllocator allocator=backend switch{"cpu"=>new CpuAllocator(BlasEnum.DotNet),"cuda"=>new CudaAllocator(0),"mlx"=>new MlxAllocator(0),_=>throw new ArgumentException("Unknown backend")};
try
{
    var load=Stopwatch.StartNew();using var decoder=new Ltx25AudioDecoder(checkpoint,allocator);load.Stop();
    var cases=new List<object>();
    foreach(var item in oracle.RootElement.GetProperty("cases").EnumerateArray())
    {
        int tokens=item.GetProperty("tokens").GetInt32();float[] input=item.GetProperty("latents").EnumerateArray().Select(v=>v.GetSingle()).ToArray();
        var trace=new Dictionary<string,float[]>();var timer=Stopwatch.StartNew();var audio=decoder.Decode(input,tokens,trace:(n,v,s)=>trace[n]=v);timer.Stop();
        var errors=new Dictionary<string,object>();
        foreach(var pair in trace)
        {
            float[] expected=item.GetProperty(pair.Key).EnumerateArray().Select(v=>v.GetSingle()).ToArray();
            if(expected.Length!=pair.Value.Length)throw new InvalidOperationException(pair.Key+" shape mismatch");
            double squared=0,norm=0,max=0;for(int i=0;i<expected.Length;i++){double d=expected[i]-pair.Value[i];squared+=d*d;norm+=(double)expected[i]*expected[i];max=Math.Max(max,Math.Abs(d));}
            errors[pair.Key]=new{relative_l2=Math.Sqrt(squared/Math.Max(norm,1e-30)),max_absolute=max,sha256=Hash(pair.Value),elements=expected.Length};
        }
        var samples=new List<double>();for(int repeat=0;repeat<repeats;repeat++){var sw=Stopwatch.StartNew();decoder.Decode(input,tokens);sw.Stop();samples.Add(sw.Elapsed.TotalMilliseconds);}
        cases.Add(new{tokens,input_sha256=Hash(input),sample_rate=audio.SampleRate,samples=audio.SampleCount,first_decode_ms=timer.Elapsed.TotalMilliseconds,warm_decode_ms=samples,errors});
    }
    var report=new{backend,checkpoint=Path.GetFullPath(checkpoint),checkpoint_sha256=Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(checkpoint))),oracle_sha256=Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(oraclePath))),load_ms=load.Elapsed.TotalMilliseconds,cases};
    File.WriteAllText(outputPath,JsonSerializer.Serialize(report,new JsonSerializerOptions{WriteIndented=true})+"\n");Console.WriteLine(outputPath);
}
finally{(allocator as IDisposable)?.Dispose();}
static string Hash(float[] values){byte[] bytes=new byte[values.Length*4];Buffer.BlockCopy(values,0,bytes,0,bytes.Length);return Convert.ToHexStringLower(SHA256.HashData(bytes));}
