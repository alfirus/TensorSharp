using System.Diagnostics;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;

string? Value(string name) => args.SkipWhile(value => value != name).Skip(1).FirstOrDefault();
string models=Value("--models") ?? throw new ArgumentException("--models is required");
string output=Value("--output") ?? throw new ArgumentException("--output is required");
string arm=Value("--arm") ?? "mapped";
if (arm is not ("mapped" or "expanded")) throw new ArgumentException("--arm must be mapped or expanded");
int warmRuns=int.Parse(Value("--warm-runs") ?? "2");
Environment.SetEnvironmentVariable("TS_H3_CUDA_MAPPED_VAE_WEIGHTS",arm == "mapped" ? "1" : "0");
float[] Read(string name)
{
    byte[] bytes=File.ReadAllBytes(Path.Combine(models,"fixtures",name));
    float[] data=new float[bytes.Length/sizeof(float)]; Buffer.BlockCopy(bytes,0,data,0,bytes.Length); return data;
}
int[] shape=File.ReadAllText(Path.Combine(models,"fixtures","h3_video_latent_in.bin.shape")).Split(',').Select(int.Parse).ToArray();
int channels=shape[1], time=shape[2], height=shape[3], width=shape[4];
float[] chw=Read("h3_video_latent_in.bin"), latent=new float[chw.Length], expected=Read("h3_video_pixels_out.bin");
for(int c=0;c<channels;c++)for(int t=0;t<time;t++)for(int h=0;h<height;h++)for(int w=0;w<width;w++)
    latent[((t*height+h)*width+w)*channels+c]=chw[((c*time+t)*height+h)*width+w];
using var allocator=new CudaAllocator(0);
var constructor=Stopwatch.StartNew();
using var model=new MiniMaxH3VideoVae(Path.Combine(models,"minimax_h3_video_vae_fp16.safetensors"),BackendType.Cuda,allocator);
constructor.Stop();
var records=new List<object>();
float[]? first=null;
bool passed=true;
for(int run=0;run<=warmRuns;run++)
{
    var timer=Stopwatch.StartNew();
    float[] tokens=model.DecodeTokens(latent,time,height,width);
    timer.Stop();
    float[] pixels=MiniMaxH3VideoVae.UnpackPatches(tokens,time,height,width);
    double dot=0, squareReference=0, squareActual=0, maximum=0;
    if(pixels.Length != expected.Length) throw new InvalidOperationException("Oracle output length mismatch");
    for(int i=0;i<pixels.Length;i++)
    {
        if(!float.IsFinite(pixels[i])) throw new InvalidOperationException("Nonfinite pixel output");
        dot+=(double)pixels[i]*expected[i]; squareReference+=(double)expected[i]*expected[i]; squareActual+=(double)pixels[i]*pixels[i];
        maximum=Math.Max(maximum,Math.Abs(pixels[i]-expected[i]));
    }
    double cosine=dot/(Math.Sqrt(squareReference)*Math.Sqrt(squareActual)+1e-30);
    bool repeat=first == null || first.AsSpan().SequenceEqual(tokens);
    passed &= cosine > .999 && repeat;
    records.Add(new { run, cacheScope=run == 0 ? "first decode including deferred weight load and cold resident weights" : "same loaded model, resident weights", seconds=timer.Elapsed.TotalSeconds,
                      cosine, maxAbs=maximum, identicalToCold=repeat });
    first ??= tokens;
}
var report=new { arm, backend="cuda", latentShape=new[]{time,height,width,channels}, constructorSeconds=constructor.Elapsed.TotalSeconds,
                 warmRuns, oracleBound="existing independent video oracle cosine > 0.999; finite outputs; exact cold/warm repeat", passed, records };
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
string json=JsonSerializer.Serialize(report,new JsonSerializerOptions{WriteIndented=true});
File.WriteAllText(output,json+"\n"); Console.WriteLine(json);
return passed ? 0 : 1;
