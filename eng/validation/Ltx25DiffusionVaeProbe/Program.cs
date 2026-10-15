using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;

// Decodes channel-major [128,T,H,W] latents (shape in <latents>.json) with
// TensorSharp's LTX 2.5 diffusion video decoder. With --noise (a Float32 [3,F,H,W]
// canvas, shape in <noise>.json) it writes the raw [-1,1] DecodeRaw output for
// ltx25-diffusion-vae-reference.py to compare; otherwise it decodes with --seed
// and writes frame-major RGB [F,H,W,3] in [0,1] (and an MP4 when --mp4 is given).
var options = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 == args.Length || !args[i].StartsWith("--") || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Expected unique --option value pairs.");
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
string vaePath = Path.GetFullPath(Required("vae")), latentsPath = Path.GetFullPath(Required("latents"));
string outputPath = Path.GetFullPath(Required("output"));
string backend = options.GetValueOrDefault("backend", "cpu").ToLowerInvariant();
if (File.Exists(outputPath + ".f32") || File.Exists(outputPath + ".json"))
    throw new IOException("Refusing to replace an existing probe output.");
string Hash(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(stream));
}
float[] ReadFloats(string path)
{
    byte[] bytes = File.ReadAllBytes(path);
    var values = new float[bytes.Length / 4];
    Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
    return values;
}
int[] Shape(string path)
{
    using var document = JsonDocument.Parse(File.ReadAllText(path + ".json"));
    return document.RootElement.GetProperty("shape").EnumerateArray().Select(value => value.GetInt32()).ToArray();
}
int[] shape = Shape(latentsPath);
float[] latents = ReadFloats(latentsPath);
IAllocator allocator = backend switch
{
    "cpu" => new CpuAllocator(BlasEnum.DotNet),
    "mlx" => new MlxAllocator(),
    _ => throw new ArgumentException("Supported backends: cpu, mlx."),
};
using var decoder = new Ltx25DiffusionVideoDecoder(vaePath, allocator);
var timer = Stopwatch.StartNew();
float[] result;
string kind;
if (options.TryGetValue("noise", out string? noisePath))
{
    noisePath = Path.GetFullPath(noisePath);
    if (!Shape(noisePath).SequenceEqual(decoder.NoiseShape(shape[1], shape[2], shape[3])))
        throw new InvalidDataException("The noise does not cover the decoder's pixel canvas.");
    result = decoder.DecodeRaw(latents, shape[1], shape[2], shape[3], ReadFloats(noisePath));
    kind = "raw channel-major [3,F,H,W] in [-1,1]";
}
else
{
    result = decoder.Decode(latents, shape[1], shape[2], shape[3], long.Parse(options.GetValueOrDefault("seed", "42")));
    kind = "frame-major RGB [F,H,W,3] in [0,1]";
}
double milliseconds = timer.Elapsed.TotalMilliseconds;
var bytes = new byte[checked(result.Length * 4)];
Buffer.BlockCopy(result, 0, bytes, 0, bytes.Length);
File.WriteAllBytes(outputPath + ".f32", bytes);
int frames = (shape[1] - 1) * Ltx25DiffusionVideoDecoder.TemporalScale + 1;
int height = shape[2] * Ltx25DiffusionVideoDecoder.SpatialScale, width = shape[3] * Ltx25DiffusionVideoDecoder.SpatialScale;
if (options.TryGetValue("mp4", out string? mp4) && noisePath == null)
{
    var rgbFrames = Enumerable.Range(0, frames).Select(frame =>
        new TensorSharp.Models.QwenImage.RgbImage(width, height, result.AsSpan(frame * height * width * 3, height * width * 3).ToArray())).ToArray();
    TensorSharp.Models.WanVideo.VideoIO.SaveMp4(Path.GetFullPath(mp4), rgbFrames, 24);
}
var report = new
{
    scope = "TensorSharp LTX 2.5 diffusion video decoder only. Not end-to-end generation evidence.",
    backend, vae = vaePath, vae_sha256 = Hash(vaePath), latents = latentsPath, latents_sha256 = Hash(latentsPath),
    noise = noisePath, latent_shape = shape, output_shape = new[] { frames, height, width }, output_kind = kind,
    decode_milliseconds = milliseconds, output_sha256 = Hash(outputPath + ".f32"),
    mlx_memory = backend == "mlx" ? (object)MlxBackend.GetMemorySnapshot() : null,
};
File.WriteAllText(outputPath + ".json", JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"Decoded [128,{shape[1]},{shape[2]},{shape[3]}] in {milliseconds:F1} ms");
