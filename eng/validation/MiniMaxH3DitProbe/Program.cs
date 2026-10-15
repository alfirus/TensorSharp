using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;

string? Value(string name) => args.SkipWhile(value=>value!=name).Skip(1).FirstOrDefault();
if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_H3_DIT_LAYERS")))
    throw new ArgumentException("Full-depth validation rejects TS_H3_DIT_LAYERS; clear the inherited layer override.");
string model=Value("--model") ?? throw new ArgumentException("--model is required");
string reference=Value("--reference") ?? throw new ArgumentException("--reference is required");
string output=Path.GetFullPath(Value("--output") ?? throw new ArgumentException("--output is required"));
float sigma=float.Parse(Value("--sigma") ?? "1",System.Globalization.CultureInfo.InvariantCulture);
bool traceEnabled=!args.Contains("--no-trace");
int warmup=int.Parse(Value("--warmup") ?? "0"),repeats=int.Parse(Value("--repeats") ?? "1");
if(warmup<0 || repeats<1 || (traceEnabled && warmup+repeats!=1)) throw new ArgumentException("Repeated timings require --no-trace");
string trace=Path.GetFullPath(Value("--trace-dir") ?? Path.Combine(Path.GetDirectoryName(output)!,"trace"));
if(File.Exists(output) || File.Exists(Path.Combine(trace,"packed-in.f32")) ||
   File.Exists(Path.Combine(trace,"carrier-audio-velocity.f32"))) throw new InvalidOperationException("Refusing existing trace output");
Directory.CreateDirectory(trace);
Environment.SetEnvironmentVariable("TS_H3_DIT_TRACE_DIR",traceEnabled ? trace : null);
Environment.SetEnvironmentVariable("TS_H3_LINEAR_TRACE_FILE",traceEnabled ? Path.Combine(trace,"linear-calls.jsonl") : null);
var metadata=Directory.GetFiles(reference,"*.f32.json")
    .Select(path=>JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(path))).ToArray();
float[] Read(string file)
{
    byte[] bytes=File.ReadAllBytes(Path.Combine(reference,file));
    if(bytes.Length%4!=0) throw new InvalidDataException("F32 file is not aligned");
    var values=new float[bytes.Length/4]; Buffer.BlockCopy(bytes,0,values,0,bytes.Length);
    if(values.Any(value=>!float.IsFinite(value))) throw new InvalidDataException("Nonfinite reference input");
    return values;
}
JsonElement ProjectionInput(string suffix)
{
    var projection=metadata.Single(item=>item.GetProperty("file").GetString()!.StartsWith("projection-") &&
        item.GetProperty("sources").EnumerateArray().Any(source=>source.GetProperty("name").GetString()!.Contains(suffix)));
    int id=projection.GetProperty("sources")[1].GetProperty("id").GetInt32();
    return metadata.Single(item=>item.GetProperty("id").GetInt32()==id);
}
var vInput=ProjectionInput("video_patch_proj.weight");
var aInput=ProjectionInput("audio_patch_proj.weight");
var tInput=ProjectionInput("condition_proj.weight");
var videoLeaf=metadata.Single(item=>item.GetProperty("file").GetString()!.StartsWith("input-") &&
    item.GetProperty("ne")[3].GetInt32()==24);
int latentW=videoLeaf.GetProperty("ne")[0].GetInt32(), latentH=videoLeaf.GetProperty("ne")[1].GetInt32(),
    latentT=videoLeaf.GetProperty("ne")[2].GetInt32();
int audioCount=aInput.GetProperty("ne")[1].GetInt32(), textCount=tInput.GetProperty("ne")[1].GetInt32();
var shape=new MiniMaxH3Shape {Width=latentW*16,Height=latentH*16,
    Frames=MiniMaxH3Geometry.LatentFramesToVideoFrames(latentT),Fps=24,
    LatentWidth=latentW,LatentHeight=latentH,LatentFrames=latentT,AudioLatentFrames=audioCount/2};
float[] video=Read(vInput.GetProperty("file").GetString()!), audio=Read(aInput.GetProperty("file").GetString()!),
    text=Read(tInput.GetProperty("file").GetString()!);
void Write(string name,float[] values)
{
    if(values.Any(value=>!float.IsFinite(value))) throw new InvalidDataException("Nonfinite output");
    using var writer=new BinaryWriter(File.Create(Path.Combine(trace,name+".f32")));
    foreach(float value in values)writer.Write(value);
}
double diagnosticSeconds;
var forwardSeconds=new List<double>();
string? velocitySha256=null;
int blocks;
using(var allocator=new CudaAllocator(0))
using(var dit=new MiniMaxH3DiT(model,BackendType.Cuda,allocator))
{
    var layout=MiniMaxH3Layout.Build(textCount,shape,sigma);
    for(int iteration=0;iteration<warmup+repeats;iteration++)
    {
        var timer=Stopwatch.StartNew();
        var(vVelocity,aVelocity)=dit.Forward(video,shape.VideoTokenCount,audio,shape.AudioTokenCount,text,textCount,layout,sigma);
        timer.Stop();
        forwardSeconds.Add(timer.Elapsed.TotalSeconds);
        MiniMaxH3Scheduler.ConvertAudioVelocityToCarrier(audio,aVelocity,sigma);
        byte[] values=new byte[(vVelocity.Length+aVelocity.Length)*sizeof(float)];
        Buffer.BlockCopy(vVelocity,0,values,0,vVelocity.Length*sizeof(float));
        Buffer.BlockCopy(aVelocity,0,values,vVelocity.Length*sizeof(float),aVelocity.Length*sizeof(float));
        string current=Convert.ToHexStringLower(SHA256.HashData(values));
        if(velocitySha256!=null && velocitySha256!=current) throw new InvalidDataException("Repeated identical forward changed velocity");
        velocitySha256=current;
        if(iteration==warmup+repeats-1) { Write("carrier-audio-velocity",aVelocity); Write("video-velocity",vVelocity); }
    }
    diagnosticSeconds=forwardSeconds[0];
    blocks=dit.Config.NumLayers;
}
string Sha(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
var inputs=new[]{vInput,aInput,tInput}.Select(item=>new{file=item.GetProperty("file").GetString(),
    sha256=Sha(Path.Combine(reference,item.GetProperty("file").GetString()!))});
var result=new {completed=true,backend="cuda",sigma,geometry=new[]{shape.Width,shape.Height,shape.Frames},
    textCount,audioCount,videoCount=shape.VideoTokenCount,blocks,inputs,
    bf16Pedantic=Environment.GetEnvironmentVariable("TS_H3_CUDA_BF16_PEDANTIC") == "1",trace,
    quantBands=Environment.GetEnvironmentVariable("TS_H3_CUDA_F32_BANDS") != "0",
    quantBandMode=Environment.GetEnvironmentVariable("TS_H3_CUDA_F32_BANDS") ?? "pedantic (default)",traceEnabled,warmup,repeats,forwardSeconds,
    diagnosticSeconds,velocitySha256,
    timingScope=traceEnabled ? "Synchronized diagnostic boundaries, excluded from performance" : "Full50 Forward including host outputs; first includes device weight upload, later same-model forwards; OS page cache not controlled",
    limitation="Exact SD callback projection inputs; excludes text encoding, sampler loop, VAE and output codec"};
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
File.WriteAllText(output,JsonSerializer.Serialize(result,new JsonSerializerOptions{WriteIndented=true})+"\n");
Console.WriteLine(JsonSerializer.Serialize(result));
return 0;
