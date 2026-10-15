using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;
using TensorSharp.Runtime;

// Runs TensorSharp's x2 spatial latent upsampler on channel-major [128,T,H,W]
// Float32 latents (shape in <latents>.json) and writes <output>.f32 plus a report.
// Compare with ltx25-latent-upsampler-reference.py on the same input.
var options = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 == args.Length || !args[i].StartsWith("--") || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Expected unique --option value pairs.");
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
string upsamplerPath = Path.GetFullPath(Required("upsampler")), vaePath = Path.GetFullPath(Required("vae"));
string latentsPath = Path.GetFullPath(Required("latents")), outputPath = Path.GetFullPath(Required("output"));
string backend = options.GetValueOrDefault("backend", "cpu").ToLowerInvariant();
if (File.Exists(outputPath + ".f32") || File.Exists(outputPath + ".json"))
    throw new IOException("Refusing to replace an existing probe output.");
string Hash(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(stream));
}
using var shapeDocument = JsonDocument.Parse(File.ReadAllText(latentsPath + ".json"));
int[] shape = shapeDocument.RootElement.GetProperty("shape").EnumerateArray().Select(value => value.GetInt32()).ToArray();
if (shape.Length != 4 || shape[0] != 128) throw new InvalidDataException("Latents must be channel-major [128,T,H,W].");
byte[] bytes = File.ReadAllBytes(latentsPath);
var latents = new float[bytes.Length / 4];
Buffer.BlockCopy(bytes, 0, latents, 0, bytes.Length);
float[] mean, std;
using (var vae = new SafetensorsFile(vaePath))
{
    mean = vae.ReadFloat32("per_channel_statistics.mean-of-means");
    std = vae.ReadFloat32("per_channel_statistics.std-of-means");
}
IAllocator allocator = backend switch
{
    "cpu" => new CpuAllocator(BlasEnum.DotNet),
    "cuda" => new CudaAllocator(0),
    "mlx" => new MlxAllocator(),
    _ => throw new ArgumentException("Supported backends: cpu, cuda, mlx."),
};
using var upsampler = new Ltx25LatentUpsampler(upsamplerPath, allocator);
var timer = Stopwatch.StartNew();
float[] result = upsampler.Upsample(latents, shape[1], shape[2], shape[3], mean, std);
double milliseconds = timer.Elapsed.TotalMilliseconds;
var output = new byte[checked(result.Length * 4)];
Buffer.BlockCopy(result, 0, output, 0, output.Length);
File.WriteAllBytes(outputPath + ".f32", output);
var report = new
{
    scope = "TensorSharp x2 spatial latent upsampler only (un-normalize, LatentUpsampler, normalize). Not generation quality evidence.",
    backend, upsampler = upsamplerPath, upsampler_sha256 = Hash(upsamplerPath), vae = vaePath,
    latents = latentsPath, latents_sha256 = Hash(latentsPath),
    input_shape = shape, output_shape = new[] { 128, shape[1], shape[2] * 2, shape[3] * 2 },
    first_call_milliseconds = milliseconds, output_sha256 = Hash(outputPath + ".f32"),
    mid_channels = upsampler.MidChannels, blocks_per_stage = upsampler.BlocksPerStage,
};
File.WriteAllText(outputPath + ".json", JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"Upsampled [128,{shape[1]},{shape[2]},{shape[3]}] in {milliseconds:F1} ms");
