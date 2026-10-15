using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;

string? Value(string name) => args.SkipWhile(value => value != name).Skip(1).FirstOrDefault();
string models=Path.GetFullPath(Value("--models") ?? throw new ArgumentException("--models required"));
string output=Path.GetFullPath(Value("--output") ?? throw new ArgumentException("--output required"));
string prompt=Value("--prompt") ?? "A red fox walks through a quiet forest, soft natural light, camera held steady. Forest birds chirp.";
int repeats=int.Parse(Value("--repeats") ?? "3");
if(repeats < 2 || File.Exists(output) || File.Exists(output+".f32"))
    throw new ArgumentException("Require at least two forwards and fresh output paths");
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
var constructor=Stopwatch.StartNew();
using var encoder=new MiniMaxH3TextEncoder(Path.Combine(models,"qwen3vl_32b_minimax_h3-Q4_K_M.gguf"),models,BackendType.GgmlCuda,null);
constructor.Stop();
var tokens=encoder.Tokenize(prompt);
var times=new List<double>();
string? digest=null;
float[]? last=null;
for(int iteration=0;iteration<repeats;++iteration)
{
    var timer=Stopwatch.StartNew();
    last=encoder.Encode(tokens);
    timer.Stop();
    if(last.Any(value => !float.IsFinite(value))) throw new ArithmeticException("Nonfinite full-depth text output");
    byte[] bytes=new byte[checked(last.Length*sizeof(float))];
    Buffer.BlockCopy(last,0,bytes,0,bytes.Length);
    string current=Convert.ToHexStringLower(SHA256.HashData(bytes));
    if(digest != null && current != digest) throw new ArithmeticException("Repeated full-depth outputs differ");
    digest=current;
    times.Add(timer.Elapsed.TotalSeconds);
    if(iteration==repeats-1) File.WriteAllBytes(output+".f32",bytes);
}
var result=new { completed=true,backend="ggml_cuda",prompt,tokens,blocks=encoder.Config.NumLayers,
    constructorSeconds=constructor.Elapsed.TotalSeconds,forwardSeconds=times,outputSha256=digest,
    uploadPrefault=Environment.GetEnvironmentVariable("TS_H3_UPLOAD_PREFAULT") ?? "auto",
    uploadWorkers=Environment.GetEnvironmentVariable("TS_H3_UPLOAD_WORKERS") ?? "8",
    uploadPrefaultPolicy="Auto for CUDA large-weight ranges within half effective physical/cgroup memory capacity; 0 disables, 1 forces",
    uploadProfile=Environment.GetEnvironmentVariable("TS_H3_UPLOAD_PROFILE") == "1",
    scope="Full text encoder; first includes device weight upload, later same-model repeats; file cache uncontrolled" };
File.WriteAllText(output,JsonSerializer.Serialize(result,new JsonSerializerOptions {WriteIndented=true})+"\n");
Console.WriteLine(JsonSerializer.Serialize(result));
