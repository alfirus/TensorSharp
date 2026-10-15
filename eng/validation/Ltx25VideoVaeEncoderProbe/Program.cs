using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.QwenImage;

var options = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 == args.Length || !args[i].StartsWith("--") || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Expected unique --option value pairs.");
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
int Number(string name, int fallback) => int.Parse(options.GetValueOrDefault(name, fallback.ToString()), System.Globalization.CultureInfo.InvariantCulture);
string modelPath = Path.GetFullPath(Required("model")), outputPath = Path.GetFullPath(Required("output"));
string backend = options.GetValueOrDefault("backend", "cpu").ToLowerInvariant();
int frames = Number("frames", 1), height = Number("height", 512), width = Number("width", 512);
int warmup = Number("warmup", 0), repeats = Number("repeats", 1);
bool traceEnabled = Number("trace", 1) == 1;
bool staticVideo = Number("static", 0) == 1;
if (frames <= 0 || height <= 0 || width <= 0 || warmup < 0 || repeats <= 0) throw new ArgumentException("Invalid geometry or repetition count.");
if (Directory.Exists(outputPath) && Directory.EnumerateFileSystemEntries(outputPath).Any()) throw new IOException("Refusing to replace an existing output directory.");
Directory.CreateDirectory(outputPath);
string Hash(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(stream));
}
void SaveFloats(string path, float[] data)
{
    byte[] bytes = new byte[checked(data.Length * 4)];
    Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length); File.WriteAllBytes(path, bytes);
}
var report = new Dictionary<string, object?>
{
    ["completed"] = false, ["created_utc"] = DateTimeOffset.UtcNow,
    ["scope"] = "TensorSharp trained causal video VAE encoder only. Float32 activations and widened checkpoint weights; no sampling, perceptual-quality or end-to-end performance coverage. Traces add device readbacks.",
    ["backend"] = backend, ["model"] = modelPath, ["checkpoint_sha256"] = Hash(modelPath),
    ["input_shape"] = new[] { frames, height, width, 3 }, ["warmup"] = warmup, ["repeats"] = repeats,
    ["static_video_optimized"] = staticVideo,
    ["environment"] = new[] { "MLX_ENABLE_TF32", "TENSORSHARP_MLX_LIBRARY", "TENSORSHARP_MLX_LIBRARY_DIR" }.ToDictionary(name => name, Environment.GetEnvironmentVariable),
    ["managed_binaries"] = new[] { Assembly.GetExecutingAssembly(), typeof(Ltx25VideoVaeEncoder).Assembly, typeof(CudaAllocator).Assembly,
        typeof(MlxAllocator).Assembly, typeof(CpuAllocator).Assembly }.Distinct().Select(assembly => new { path = assembly.Location, sha256 = Hash(assembly.Location) }).ToArray(),
};
void Save() => File.WriteAllText(Path.Combine(outputPath, "probe.json"), JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Save();
try
{
    IAllocator allocator = backend switch
    {
        "cpu" => new CpuAllocator(BlasEnum.DotNet), "cuda" => new CudaAllocator(), "mlx" => new MlxAllocator(),
        _ => throw new ArgumentException("Unknown backend."),
    };
    try
    {
        using var encoder = new Ltx25VideoVaeEncoder(modelPath, allocator);
        int count = checked((staticVideo ? 1 : frames) * height * width * 3);
        float[] input = new float[count];
        if (options.TryGetValue("image", out var imagePath))
        {
            if (frames != 1 && !staticVideo) throw new ArgumentException("--image requires one frame or --static 1.");
            input = ImageIO.ResizeCover(ImageIO.Load(imagePath), width, height).Pixels;
            report["image"] = new { path = Path.GetFullPath(imagePath), sha256 = Hash(imagePath), preprocessing = "EXIF-upright RGB center-crop and resize through shared ImageIO.ResizeCover" };
        }
        else if (options.TryGetValue("input", out var inputPath))
        {
            var bytes = File.ReadAllBytes(inputPath);
            if (bytes.Length != checked(count * 4)) throw new ArgumentException("Wrong RGB input byte count.");
            Buffer.BlockCopy(bytes, 0, input, 0, bytes.Length);
        }
        else
            for (int i = 0; i < input.Length; ++i) input[i] = .5f + MathF.Sin(i * .029f) * .3f + MathF.Cos(i * .013f) * .15f;
        var inputFile = Path.Combine(outputPath, "input.f32"); SaveFloats(inputFile, input);
        report["input_sha256"] = Hash(inputFile);
        report["output_shape"] = new[] { encoder.LatentChannels, 1 + (frames - 1) / encoder.TemporalScale, height / encoder.SpatialScale, width / encoder.SpatialScale };
        float[] Encode(Action<string, float[], int[]>? snapshot = null) => staticVideo
            ? encoder.EncodeStaticVideo(input, frames, height, width, trace: snapshot)
            : encoder.Encode(input, frames, height, width, trace: snapshot);
        for (int run = 0; run < warmup; ++run) Encode();
        var stages = new Dictionary<string, object>();
        Action<string, float[], int[]> trace = (name, values, shape) =>
        {
            string file = name.Replace('.', '_') + ".f32"; var path = Path.Combine(outputPath, file);
            SaveFloats(path, values); stages[name] = new { shape, file, sha256 = Hash(path) };
        };
        var timings = new List<double>();
        float[]? previous = null; double repeatDifference = 0;
        for (int run = 0; run < repeats; ++run)
        {
            var timer = Stopwatch.StartNew();
            var result = Encode(traceEnabled && run == 0 ? trace : null);
            timer.Stop(); timings.Add(timer.Elapsed.TotalMilliseconds);
            if (result.Any(value => !float.IsFinite(value))) throw new InvalidOperationException("Non-finite encoder output.");
            if (previous != null) for (int i = 0; i < result.Length; ++i) repeatDifference = Math.Max(repeatDifference, Math.Abs(result[i] - previous[i]));
            previous = result;
        }
        var outputFile = Path.Combine(outputPath, "output.f32"); SaveFloats(outputFile, previous!);
        report["stages"] = stages; report["output_sha256"] = Hash(outputFile); report["times_ms"] = timings;
        report["timing_scope"] = traceEnabled ? "diagnostic with first-run stage readbacks" : "encode plus synchronized final normalized latent readback";
        report["repeat_max_absolute_difference"] = repeatDifference;
        if (allocator is MlxAllocator mlx) report["mlx_memory_after_runs"] = mlx.GetMemorySnapshot();
        report["completed"] = true; Save();
        Console.WriteLine(JsonSerializer.Serialize(new { completed = true, backend, outputPath, timings, repeatDifference }));
    }
    finally { (allocator as IDisposable)?.Dispose(); }
}
catch (Exception error) { report["error"] = error.ToString(); Save(); throw; }
