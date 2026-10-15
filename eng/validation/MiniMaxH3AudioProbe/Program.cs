using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;

string? Value(string name) => args.SkipWhile(value=>value!=name).Skip(1).FirstOrDefault();
string model=Value("--vae") ?? throw new ArgumentException("--vae is required");
string input=Value("--tokens") ?? throw new ArgumentException("--tokens is required");
string output=Path.GetFullPath(Value("--output") ?? throw new ArgumentException("--output is required"));
int frames=int.Parse(Value("--frames") ?? "8");
byte[] raw=File.ReadAllBytes(input);
if(raw.Length!=2*frames*32*sizeof(float)) throw new InvalidDataException("Expected token-major [2,T,32] F32 endpoint");
var tokens=new float[raw.Length/4];Buffer.BlockCopy(raw,0,tokens,0,raw.Length);
if(tokens.Any(value=>!float.IsFinite(value))) throw new InvalidDataException("Nonfinite endpoint input");
float[] waveform;
double seconds;
using(var allocator=new CudaAllocator(0))
using(var vae=new MiniMaxH3AudioVae(model,BackendType.Cuda,true,allocator))
{
    var timer=Stopwatch.StartNew();
    var audio=vae.DecodeStereo(tokens,frames);
    timer.Stop();seconds=timer.Elapsed.TotalSeconds;
    if(audio.ChannelCount!=2 || audio.SampleRate!=32000) throw new InvalidDataException("Expected 32kHz stereo output");
    waveform=new float[audio.SampleCount*2];
    for(int i=0;i<audio.SampleCount;i++)for(int channel=0;channel<2;channel++)waveform[i*2+channel]=audio.Channels[channel][i];
}
if(waveform.Any(value=>!float.IsFinite(value))) throw new InvalidDataException("Nonfinite waveform");
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
using(var writer=new BinaryWriter(File.Create(output)))foreach(float value in waveform)writer.Write(value);
var result=new{completed=true,backend="cuda",inputSha256=Convert.ToHexStringLower(SHA256.HashData(raw)),frames,
    samplesPerChannel=waveform.Length/2,channels=2,sampleRate=32000,
    peak=waveform.Max(value=>Math.Abs(value)),rms=Math.Sqrt(waveform.Sum(value=>(double)value*value)/waveform.Length),
    diagnosticDecodeSeconds=seconds,scope="Saved physical audio endpoint through public DecodeStereo including normalization and plane packing; caller records endpoint provenance"};
File.WriteAllText(output+".json",JsonSerializer.Serialize(result,new JsonSerializerOptions{WriteIndented=true})+"\n");
Console.WriteLine(JsonSerializer.Serialize(result));
return 0;
