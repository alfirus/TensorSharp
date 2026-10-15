using System.Security.Cryptography;
using System.Text.Json;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Cuda;
using TensorSharp.Models;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
    if (!args[i].StartsWith("--") || i + 1 >= args.Length || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Pass --projector --pixels --width --height --backend --output --report as value pairs.");
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
string projector = Path.GetFullPath(Required("projector")), pixelsPath = Path.GetFullPath(Required("pixels"));
string outputPath = Path.GetFullPath(Required("output")), reportPath = Path.GetFullPath(Required("report"));
string backend = Required("backend").ToUpperInvariant();
bool requireMlxDefault = bool.Parse(options.GetValueOrDefault("require-mlx-default", "false"));
bool requireEmbeddingDefault = bool.Parse(options.GetValueOrDefault("require-embedding-default", "false"));
bool requireGgmlCudaStack = bool.Parse(options.GetValueOrDefault("require-ggml-cuda-stack", "false"));
if (requireGgmlCudaStack && backend != "GGML_CUDA") throw new ArgumentException("The required CUDA stack gate needs GGML_CUDA.");
string[] visionOverrides = ["TS_GEMMA4_VISION_DEVICE", "TS_GEMMA4_VISION_CUDA_DEVICE", "TS_GEMMA4_VISION_CPU_Q16",
    "TS_GEMMA4_VISION_METAL_PRECISE", "TS_GEMMA4_VISION_METAL_MPS_MM",
    "TS_GEMMA4_VISION_METAL_SPLIT_MM", "TS_GEMMA4_VISION_METAL_MIXED_MPS", "TS_GEMMA4_VISION_SPATIAL_DEVICE",
    "TS_GEMMA4_VISION_SPLIT_PROJECTION", "TS_GEMMA4_VISION_SIMD_PROJECTION", "TS_GEMMA4_VISION_SPLIT_ATTENTION",
    "TSG_EMBEDDING_LINUX_BLAS", "TSG_EMBEDDING_BLAS_LIBRARY", "TSG_EMBEDDING_CPU_ATTENTION",
    "TSG_EMBEDDING_CPU_PROFILE", "TSG_EMBEDDING_CPU_ATTENTION_PROFILE", "TSG_EMBEDDING_CPU_ATTENTION_SIMD",
    "TS_VISION_STAGE", "TS_VISION_TRACE", "TS_VISION_PROF", "TSG_CUPTI_ACTIVITY_OUTPUT"];
if (requireEmbeddingDefault && visionOverrides.Any(name => Environment.GetEnvironmentVariable(name) != null))
    throw new ArgumentException("The embedding default gate requires all vision override variables unset.");
if (requireMlxDefault && (backend != "MLX" || visionOverrides.Any(name => Environment.GetEnvironmentVariable(name) != null)))
    throw new ArgumentException("The MLX default gate requires backend MLX with vision override variables unset.");
int width = int.Parse(Required("width")), height = int.Parse(Required("height"));
int warmup = int.Parse(options.GetValueOrDefault("benchmark-warmup", "3"));
int repeats = int.Parse(options.GetValueOrDefault("benchmark-repeats", "0"));
double minimumSeconds = double.Parse(options.GetValueOrDefault("benchmark-minimum-seconds", "1"),
    System.Globalization.CultureInfo.InvariantCulture);
if (warmup < 0 || repeats < 0 || !double.IsFinite(minimumSeconds) || minimumSeconds < 0)
    throw new ArgumentException("Invalid benchmark counts/duration.");
if (width <= 0 || height <= 0 || new FileInfo(pixelsPath).Length != checked(3L * width * height * sizeof(float)))
    throw new InvalidDataException("Expected [3,height,width] F32 pixels.");
float[] pixels = new float[checked(3 * width * height)];
using (var reader = new BinaryReader(File.OpenRead(pixelsPath)))
    for (int i = 0; i < pixels.Length; ++i) pixels[i] = reader.ReadSingle();
if (pixels.Any(value => !float.IsFinite(value) || value < 0 || value > 1)) throw new InvalidDataException("Pixels must be finite [0,1].");
GgmlContext? context = null;
IAllocator allocator = backend switch
{
    "CPU" => new CpuAllocator(BlasEnum.DotNet), "MLX" => new MlxAllocator(), "CUDA" => new CudaAllocator(),
    "GGML_CPU" or "GGML_METAL" or "GGML_CUDA" => new GgmlAllocator(context = new GgmlContext([0],
        backend == "GGML_METAL" ? GgmlBackendType.Metal : backend == "GGML_CUDA" ? GgmlBackendType.Cuda : GgmlBackendType.Cpu), 0),
    _ => throw new ArgumentException("Unsupported vision probe backend " + backend),
};
string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
var binaries = Directory.EnumerateFiles(AppContext.BaseDirectory, "*", SearchOption.AllDirectories).Where(path =>
    (Path.GetFileName(path).StartsWith("TensorSharp", StringComparison.Ordinal) && Path.GetExtension(path) == ".dll")
    || Path.GetExtension(path) is ".dylib" or ".so" or ".metallib")
    .Select(path => new { path, sha256 = Hash(path) }).ToArray();
int rows = 0, dimensions = 0, stackEncodes = 0, blockLoopEncodes = 0;
var stackObservation = typeof(Gemma4VisionEncoder).GetField("_lastGgmlVisionStackUsed",
    System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
var samples = new List<double>();
try
{
    using var encoder = new Gemma4VisionEncoder(projector, allocator, inputPixelsInZeroToOneRange: true,
        requireValidatedFloat32Precision: true);
    float[] Encode()
    {
        using var output = encoder.Encode(pixels, width, height);
        bool usedStack = stackObservation?.GetValue(encoder) is true;
        if (usedStack) ++stackEncodes; else ++blockLoopEncodes;
        if (requireGgmlCudaStack && !usedStack)
            throw new InvalidOperationException("The required precise CUDA stack did not execute; no fallback counts as this gate.");
        rows = checked((int)output.Sizes[0]); dimensions = checked((int)output.Sizes[1]);
        return output.GetElementsAsFloat(checked(rows * dimensions));
    }
    float[] values;
    if (repeats == 0) values = Encode();
    else
    {
        for (int i = 0; i < warmup; ++i) Encode();
        var total = Stopwatch.StartNew();
        do
        {
            var sample = Stopwatch.StartNew(); values = Encode();
            samples.Add(sample.Elapsed.TotalMilliseconds);
        } while (samples.Count < repeats || total.Elapsed.TotalSeconds < minimumSeconds);
    }
    if (values.Any(value => !float.IsFinite(value))) throw new InvalidDataException("Vision output must be finite.");
    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
    using var writer = new BinaryWriter(File.Create(outputPath));
    foreach (float value in values) writer.Write(value);
}
finally
{
    if (context != null)
    {
        GgmlBasicOps.ReleaseReuseComputeBuffers(); GgmlBasicOps.ClearHostBufferCache(); context.ReleasePoolForDisposal();
        GgmlBasicOps.Shutdown();
    }
    (allocator as IDisposable)?.Dispose();
}
Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
File.WriteAllText(reportPath, JsonSerializer.Serialize(new
{
    completed = true, backend, allocator = allocator.GetType().FullName, projector, projector_sha256 = Hash(projector),
    pixels = pixelsPath, pixels_sha256 = Hash(pixelsPath), width, height, rows, dimensions,
    output = outputPath, output_sha256 = Hash(outputPath), vision_device = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_DEVICE"),
    vision_split_projection = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_SPLIT_PROJECTION"),
    vision_simd_projection = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_SIMD_PROJECTION"),
    vision_split_attention = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_SPLIT_ATTENTION"),
    vision_cuda_device = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_CUDA_DEVICE"),
    vision_cpu_q16 = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_CPU_Q16"),
    vision_metal_precise = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_METAL_PRECISE"),
    vision_metal_mps_mm = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_METAL_MPS_MM"),
    vision_metal_split_mm = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_METAL_SPLIT_MM"),
    vision_metal_mixed_mps = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_METAL_MIXED_MPS"),
    vision_spatial_device = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_SPATIAL_DEVICE"),
    ggml_cpu_threads = Environment.GetEnvironmentVariable("TS_GGML_CPU_THREADS"),
    embedding_linux_blas = Environment.GetEnvironmentVariable("TSG_EMBEDDING_LINUX_BLAS"),
    embedding_blas_library = Environment.GetEnvironmentVariable("TSG_EMBEDDING_BLAS_LIBRARY"),
    embedding_cpu_attention = Environment.GetEnvironmentVariable("TSG_EMBEDDING_CPU_ATTENTION"),
    embedding_cpu_profile = Environment.GetEnvironmentVariable("TSG_EMBEDDING_CPU_PROFILE"),
    embedding_cpu_attention_profile = Environment.GetEnvironmentVariable("TSG_EMBEDDING_CPU_ATTENTION_PROFILE"),
    embedding_cpu_attention_simd = Environment.GetEnvironmentVariable("TSG_EMBEDDING_CPU_ATTENTION_SIMD"),
    vision_stage = Environment.GetEnvironmentVariable("TS_VISION_STAGE"),
    vision_trace = Environment.GetEnvironmentVariable("TS_VISION_TRACE"),
    vision_profile = Environment.GetEnvironmentVariable("TS_VISION_PROF"),
    cupti_activity_output = Environment.GetEnvironmentVariable("TSG_CUPTI_ACTIVITY_OUTPUT"),
    embedding_default_guard_version = 4,
    require_embedding_default = requireEmbeddingDefault,
    require_validated_float32_precision = true,
    require_ggml_cuda_stack = requireGgmlCudaStack,
    ggml_cuda_stack_observation_available = stackObservation != null,
    ggml_cuda_stack_encode_count = stackEncodes,
    retained_block_loop_encode_count = blockLoopEncodes,
    vision_constructor = "validated-embedding-four-argument",
    raw_tower_benchmark = samples.Count == 0 ? null : new
    {
        warmup, minimum_repeats = repeats, minimum_seconds = minimumSeconds,
        samples_ms = samples, sample_count = samples.Count,
        median_ms = (samples.Order().ElementAt((samples.Count - 1) / 2) + samples.Order().ElementAt(samples.Count / 2)) / 2,
        p95_ms = samples.Order().ElementAt(Math.Min(samples.Count - 1, (int)Math.Ceiling(.95 * samples.Count) - 1)),
        scope = "Warm raw vision tower encode and final readback/disposal; excludes model load, image decoding/resize, text tower, and HTTP. GPU completion is included in final readback.",
    },
    require_mlx_default = requireMlxDefault,
    binary_artifacts = binaries, created_utc = DateTimeOffset.UtcNow,
    limitations = "Raw unnormalized vision soft rows from exact pixels; no text pooling, resize, protocol or performance claim.",
}, new JsonSerializerOptions { WriteIndented = true }) + "\n");
