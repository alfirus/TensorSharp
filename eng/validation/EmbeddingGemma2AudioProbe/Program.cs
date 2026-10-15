using System.Security.Cryptography;
using System.Diagnostics;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Models;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
{
    if (!args[i].StartsWith("--") || i + 1 >= args.Length || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Pass --projector <gguf> --mel <little-endian.f32> --frames <count> --backend <CPU|MLX|CUDA|GGML_CPU|GGML_METAL|GGML_CUDA> --output <f32> --report <json>.");
}
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
string projector = Path.GetFullPath(Required("projector")), melPath = Path.GetFullPath(Required("mel"));
string outputPath = Path.GetFullPath(Required("output")), reportPath = Path.GetFullPath(Required("report"));
string backend = Required("backend").ToUpperInvariant();
bool requireEmbeddingDefault = bool.Parse(options.GetValueOrDefault("require-embedding-default", "false"));
bool validatedPrecision = bool.Parse(options.GetValueOrDefault("validated-precision", requireEmbeddingDefault ? "true" : "false"));
if (requireEmbeddingDefault)
{
    if (!validatedPrecision) throw new ArgumentException("The embedding-default gate requires --validated-precision true.");
    string[] overrides = ["TS_GEMMA4_AUDIO_SSCP_GRAPH", "TS_GEMMA4_AUDIO_BLOCK_GRAPH", "TS_GEMMA4_AUDIO_PRECISE_ROWS",
        "TS_GEMMA4_AUDIO_METAL_F32_MM", "TS_GEMMA4_AUDIO_METAL_MPS_MM", "TS_GEMMA4_AUDIO_METAL_FUSED_OPS",
        "TS_GEMMA4_AUDIO_METAL_SIMD_MM", "TS_GEMMA4_AUDIO_CUDA_PRECISE_SSCP", "TS_GEMMA4_AUDIO_CUDA_PRECISE_LINEAR", "TS_GEMMA4_AUDIO_GGML_CUDA_PEDANTIC",
        "TSG_EMBEDDING_LINUX_BLAS", "TSG_EMBEDDING_BLAS_LIBRARY", "TSG_EMBEDDING_CPU_PROFILE", "TSG_CUPTI_ACTIVITY_OUTPUT"];
    foreach (string name in overrides)
        if (Environment.GetEnvironmentVariable(name) != null)
            throw new ArgumentException("The embedding-default gate forbids process override " + name + ".");
}
string? traceDirectory = options.TryGetValue("trace-directory", out string? traceOption) ? Path.GetFullPath(traceOption) : null;
bool profilePhases = bool.Parse(options.GetValueOrDefault("profile-phases", "false"));
List<object> traces = [];
List<object> phaseTimings = [];
long phaseStarted = 0;
double previousPhaseMs = 0;
int phaseIteration = 0;
int frames = int.Parse(Required("frames"));
if (frames <= 0) throw new ArgumentOutOfRangeException("frames");
int warmup = int.Parse(options.GetValueOrDefault("warmup", "0"));
int repeats = int.Parse(options.GetValueOrDefault("repeats", "1"));
double minimumDurationSeconds = double.Parse(options.GetValueOrDefault("minimum-duration-seconds", "0"), System.Globalization.CultureInfo.InvariantCulture);
if (warmup < 0 || warmup > 10000 || repeats <= 0 || repeats > 10000
    || !double.IsFinite(minimumDurationSeconds) || minimumDurationSeconds < 0 || minimumDurationSeconds > 60)
    throw new ArgumentOutOfRangeException("Timing counts and duration must be bounded and nonnegative; repeats must be positive.");
if (new FileInfo(melPath).Length != checked(frames * 128L * sizeof(float))) throw new InvalidDataException("Expected [frames,128] F32 mel values.");
float[] mel = new float[checked(frames * 128)];
using (var reader = new BinaryReader(File.OpenRead(melPath)))
    for (int i = 0; i < mel.Length; ++i) mel[i] = reader.ReadSingle();
if (mel.Any(value => !float.IsFinite(value))) throw new InvalidDataException("Mel values must be finite.");
GgmlContext? context = null;
IAllocator allocator = backend switch
{
    "CPU" => new CpuAllocator(BlasEnum.DotNet),
    "MLX" => new MlxAllocator(),
    "CUDA" => new CudaAllocator(),
    "GGML_CPU" or "GGML_METAL" or "GGML_CUDA" => new GgmlAllocator(context = new GgmlContext([0], backend switch
    {
        "GGML_METAL" => GgmlBackendType.Metal,
        "GGML_CUDA" => GgmlBackendType.Cuda,
        _ => GgmlBackendType.Cpu,
    }), 0),
    _ => throw new ArgumentException("Unsupported backend " + backend),
};
string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
var binaries = Directory.EnumerateFiles(AppContext.BaseDirectory, "*", SearchOption.AllDirectories).Where(path =>
    (Path.GetFileName(path).StartsWith("TensorSharp", StringComparison.Ordinal) && Path.GetExtension(path) == ".dll")
    || Path.GetExtension(path) is ".dylib" or ".so" or ".ptx" or ".metallib")
    .Select(path => new { path, sha256 = Hash(path) }).ToArray();
int rows = 0, dimensions = 0;
List<double> samples = [];
try
{
    using var encoder = validatedPrecision
        ? new Gemma4AudioEncoder(projector, allocator, requireValidatedFloat32Precision: true)
        : new Gemma4AudioEncoder(projector, allocator);
    if (traceDirectory != null || profilePhases)
    {
        if (traceDirectory != null) Directory.CreateDirectory(traceDirectory);
        Action<string, float[], int, int> observer = (name, values, stageRows, stageColumns) =>
        {
            double elapsed = Stopwatch.GetElapsedTime(phaseStarted).TotalMilliseconds;
            if (profilePhases) phaseTimings.Add(new { iteration = phaseIteration, name, elapsed_ms = elapsed, stage_ms = elapsed - previousPhaseMs });
            previousPhaseMs = elapsed;
            if (traceDirectory == null) return;
            string path = Path.Combine(traceDirectory, name + ".f32");
            using (var writer = new BinaryWriter(File.Create(path)))
                foreach (float value in values) writer.Write(value);
            traces.Add(new { name, rows = stageRows, columns = stageColumns, path, sha256 = Hash(path) });
        };
        var property = typeof(Gemma4AudioEncoder).GetProperty("ValidationTrace", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("This Models binary does not include the requested stage diagnostics.");
        property.SetValue(encoder, observer);
    }
    float[] Run()
    {
        phaseStarted = Stopwatch.GetTimestamp(); previousPhaseMs = 0; ++phaseIteration;
        using var output = encoder.Encode(mel, frames);
        rows = checked((int)output.Sizes[0]); dimensions = checked((int)output.Sizes[1]);
        return output.GetElementsAsFloat(checked(rows * dimensions));
    }
    for (int i = 0; i < warmup; ++i) Run();
    float[] values = [];
    double elapsedSeconds = 0;
    do
    {
        long start = Stopwatch.GetTimestamp();
        values = Run();
        double milliseconds = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        samples.Add(milliseconds);
        elapsedSeconds += milliseconds / 1000;
        if (samples.Count == 10000 && (samples.Count < repeats || elapsedSeconds < minimumDurationSeconds))
            throw new InvalidOperationException("Minimum timing duration exceeds the bounded iteration budget.");
    }
    while (samples.Count < repeats || elapsedSeconds < minimumDurationSeconds);
    if (values.Any(value => !float.IsFinite(value))) throw new InvalidDataException("Audio output must be finite.");
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
double[] sortedSamples = samples.Order().ToArray();
double median = sortedSamples.Length % 2 == 0
    ? (sortedSamples[sortedSamples.Length / 2 - 1] + sortedSamples[sortedSamples.Length / 2]) / 2
    : sortedSamples[sortedSamples.Length / 2];
File.WriteAllText(reportPath, JsonSerializer.Serialize(new
{
    completed = true, backend, require_embedding_default = requireEmbeddingDefault, validated_precision = validatedPrecision,
    constructor = validatedPrecision ? "validated-three-argument" : "chat-two-argument",
    allocator = allocator.GetType().FullName, projector, projector_sha256 = Hash(projector),
    mel = melPath, mel_sha256 = Hash(melPath), frames, rows, dimensions, output = outputPath, output_sha256 = Hash(outputPath),
    binary_artifacts = binaries, created_utc = DateTimeOffset.UtcNow,
    trace_directory = traceDirectory, traces, profile_phases = profilePhases, phase_timings = phaseTimings,
    timing = new { warmup, requested_repeats = repeats, minimum_duration_seconds = minimumDurationSeconds,
        actual_repeats = samples.Count, median_ms = median, mean_ms = samples.Average(), min_ms = samples.Min(),
        max_ms = samples.Max(), samples_ms = samples,
        scope = "Audio tower Encode, device completion/F32 readback, and output tensor disposal; loading and precomputed mel excluded." },
    environment = new { sscp_graph = Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_SSCP_GRAPH"),
        conformer_block_graph = Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_BLOCK_GRAPH"),
        precise_rows = Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_PRECISE_ROWS"),
        metal_f32_mm = Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_METAL_F32_MM"),
        metal_mps_mm = Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_METAL_MPS_MM"),
        metal_fused_ops = Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_METAL_FUSED_OPS"),
        metal_simd_mm = Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_METAL_SIMD_MM"),
        cuda_precise_sscp = Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_CUDA_PRECISE_SSCP"),
        cuda_precise_linear = Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_CUDA_PRECISE_LINEAR"),
        ggml_cuda_pedantic = Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_GGML_CUDA_PEDANTIC"),
        embedding_linux_blas = Environment.GetEnvironmentVariable("TSG_EMBEDDING_LINUX_BLAS"),
        embedding_blas_library = Environment.GetEnvironmentVariable("TSG_EMBEDDING_BLAS_LIBRARY"),
        embedding_cpu_profile = Environment.GetEnvironmentVariable("TSG_EMBEDDING_CPU_PROFILE"),
        cupti_activity_output = Environment.GetEnvironmentVariable("TSG_CUPTI_ACTIVITY_OUTPUT"),
        nvidia_tf32_override = Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE") },
    limitations = "Raw unnormalized audio soft rows from precomputed independent mel; timings exclude media decoding/frontend, text pooling, HTTP, and model loading. Optional stage tracing/phase profiling adds diagnostic readbacks and optional file I/O and must not be used for performance claims. No end-to-end performance parity claim.",
}, new JsonSerializerOptions { WriteIndented = true }) + "\n");
