using System.Diagnostics;
using System.Security.Cryptography;
using System.Buffers.Binary;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Models.Embeddings;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int index = 0; index < args.Length; index += 2)
{
    if (!args[index].StartsWith("--") || index + 1 >= args.Length)
        throw new ArgumentException("Pass --model <gguf> --cases <json> --output <json> [--backend <GGML_CPU|GGML_METAL>] [--mmproj <gguf>] [--warmup <count>] [--repeats <count>].");
    if (!options.TryAdd(args[index][2..], args[index + 1])) throw new ArgumentException("Duplicate option " + args[index]);
}
string Required(string name) => options.TryGetValue(name, out var value) ? value : throw new ArgumentException("Missing --" + name);
string modelPath = Path.GetFullPath(Required("model"));
string casePath = Path.GetFullPath(Required("cases"));
string outputPath = Path.GetFullPath(Required("output"));
string backend = options.GetValueOrDefault("backend", "GGML_CPU");
int warmup = int.Parse(options.GetValueOrDefault("warmup", "1"));
int repeats = int.Parse(options.GetValueOrDefault("repeats", "3"));
int threads = int.Parse(options.GetValueOrDefault("threads", "8"));
bool shutdownGgml = bool.Parse(options.GetValueOrDefault("shutdown-ggml", "true"));
var audioLogPolicy = options.GetValueOrDefault("audio-log-policy", "huggingface_add") switch
{
    "huggingface_add" => Gemma4AudioLogPolicy.HuggingFaceAddFloor,
    "llama_clamp" => Gemma4AudioLogPolicy.LlamaClampFloor,
    _ => throw new ArgumentException("Audio log policy must be huggingface_add or llama_clamp."),
};
if (warmup < 0 || repeats < 1 || threads < 0) throw new ArgumentOutOfRangeException("Warmup and threads must be nonnegative; repeats must be positive.");
using var cases = JsonDocument.Parse(File.ReadAllText(casePath));
var modelOptions = new EmbeddingModelOptions { Backend = backend, Threads = threads };
using var multimodal = options.TryGetValue("mmproj", out var projectorPath)
    ? EmbeddingGemma2MultimodalEncoder.Load(modelPath, Path.GetFullPath(projectorPath), modelOptions,
        new() { VideoPrefix = options.GetValueOrDefault("video-prefix", ""), AudioLogPolicy = audioLogPolicy })
    : null;
using var ownedTextModel = multimodal == null ? EmbeddingModel.Load(modelPath, modelOptions) : null;
var textModel = multimodal?.TextModel ?? ownedTextModel!;
var results = new List<object>();
var coverage = new HashSet<string>();
string HashFile(string path)
{
    using var file = File.OpenRead(path);
    return Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant();
}
string modelHash = HashFile(modelPath);
string? projectorHash = projectorPath == null ? null : HashFile(projectorPath);
string manifestHash = HashFile(casePath);
var binaries = new[] { typeof(EmbeddingModel).Assembly.Location,
    Path.Combine(AppContext.BaseDirectory, OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" : OperatingSystem.IsWindows() ? "GgmlOps.dll" : "libGgmlOps.so"),
    Environment.GetEnvironmentVariable("TENSORSHARP_CPU_NATIVE_LIBRARY") ?? string.Empty,
    Environment.GetEnvironmentVariable("TSG_EMBEDDING_BLAS_LIBRARY") ?? string.Empty,
    Path.Combine(AppContext.BaseDirectory, OperatingSystem.IsMacOS() ? "libTensorSharpCpu.dylib" : OperatingSystem.IsWindows() ? "TensorSharpCpu.dll" : "libTensorSharpCpu.so") }
    .Where(File.Exists).Select(path => new { path, sha256 = HashFile(path) }).ToArray();
double Cosine(float[] left, float[] right)
    => left.Zip(right, (a, b) => (double)a * b).Sum()
        / Math.Sqrt(left.Sum(value => (double)value * value) * right.Sum(value => (double)value * value));
void WriteReport(bool completed)
{
    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
    File.WriteAllText(outputPath, JsonSerializer.Serialize(new
    {
        completed, created_utc = DateTimeOffset.UtcNow,
        model = modelPath, model_sha256 = modelHash,
        projector = projectorPath, projector_sha256 = projectorHash, binary_artifacts = binaries,
        backend = textModel.Backend, backend_requested = backend, is_managed = textModel.IsManaged,
        cpu_native_attention_requested = Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_NATIVE_ATTENTION") == "1",
        cpu_native_q16_requested = Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_NATIVE_Q16") == "1",
        cpu_linux_blas_requested = Environment.GetEnvironmentVariable("TSG_EMBEDDING_LINUX_BLAS") == "1",
        cpu_tiled_attention_requested = Environment.GetEnvironmentVariable("TSG_EMBEDDING_CPU_ATTENTION") == "1",
        cpu_attention_simd_override = Environment.GetEnvironmentVariable("TSG_EMBEDDING_CPU_ATTENTION_SIMD"),
        cpu_split_graph_profile_requested = Environment.GetEnvironmentVariable("TSG_EMBEDDING_CPU_PROFILE") == "1",
        cpu_attention_profile_requested = Environment.GetEnvironmentVariable("TSG_EMBEDDING_CPU_ATTENTION_PROFILE") == "1",
        timings_are_diagnostic = new[] { "TSG_EMBEDDING_CPU_PROFILE", "TSG_EMBEDDING_CPU_ATTENTION_PROFILE",
            "TENSORSHARP_EMBEDDING_CPU_PROFILE", "TS_EMBEDDING_CUDA_PHASES", "TS_VISION_PROF", "TS_GEMMA4_AUDIO_METAL_PROFILE" }
            .Any(name => Environment.GetEnvironmentVariable(name) == "1") ||
            Environment.GetEnvironmentVariable("TSG_CUPTI_ACTIVITY_OUTPUT") != null,
        openblas_num_threads = Environment.GetEnvironmentVariable("OPENBLAS_NUM_THREADS"),
        ggml_media_thread_override = Environment.GetEnvironmentVariable("TS_GGML_CPU_THREADS"),
        cuda_materialized_attention_requested = Environment.GetEnvironmentVariable("TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION") == "1",
        cuda_normalized_half_requested = Environment.GetEnvironmentVariable("TS_EMBEDDING_CUDA_NORMALIZED_HALF") == "1",
        cuda_materialized_attention_tf32_requested = Environment.GetEnvironmentVariable("TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION_TF32") == "1",
        audio_log_policy = multimodal?.AudioLogPolicy.ToString(),
        shutdown_ggml_requested = shutdownGgml, threads,
        threads_scope = "Text encoder. Per-op GGML media uses its separate CPU pool; TS_GGML_CPU_THREADS records an explicit process-start override.",
        warmup, repeats, hidden_size = textModel.HiddenSize, dimensions = textModel.Dimensions,
        case_manifest = casePath, case_manifest_sha256 = manifestHash,
        modalities_tested = coverage, cases = results,
        limitations = new[] { "Model loading is excluded; media decoding and preprocessing are included in EmbedAsync timings.",
            "is_managed describes graph ownership. Optional owned CPU math is evidenced by the EMBEDDING_CPU_MATH disposal log, including actual successful tiles and library hash.",
            "Synthetic media and the small text corpus are regression checks, not retrieval leaderboard evaluations.",
            "Only recorded model, projector, device, and cases were tested; unavailable modalities are not counted as passing." },
    }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
}

bool completed = false;
try
{
    foreach (var item in cases.RootElement.GetProperty("cases").EnumerateArray())
    {
        string name = item.GetProperty("name").GetString()!;
        var parts = new List<EmbeddingGemma2InputPart>();
        bool hasParts = item.TryGetProperty("parts", out var partArray);
        int[]? tokenOverride = item.TryGetProperty("tokens", out var tokenArray)
            ? tokenArray.EnumerateArray().Select(token => token.GetInt32()).ToArray() : null;
        if (!hasParts && tokenOverride == null) throw new ArgumentException(name + " requires parts or token IDs.");
        foreach (var part in hasParts ? partArray.EnumerateArray() : Enumerable.Empty<JsonElement>())
        {
            string kind = part.GetProperty("kind").GetString()!;
            string value = part.GetProperty("value").GetString()!;
            coverage.Add(kind);
            parts.Add(kind switch
            {
                "text" => EmbeddingGemma2InputPart.Text(value),
                "image" => EmbeddingGemma2InputPart.Image(Path.GetFullPath(value, Path.GetDirectoryName(casePath)!)),
                "audio" => EmbeddingGemma2InputPart.Audio(Path.GetFullPath(value, Path.GetDirectoryName(casePath)!)),
                "video" => EmbeddingGemma2InputPart.Video(Path.GetFullPath(value, Path.GetDirectoryName(casePath)!)),
                _ => throw new ArgumentException("Unknown input kind " + kind),
            });
        }
        bool textOnly = parts.All(part => part.Type == "text");
        if (tokenOverride != null && !textOnly) throw new ArgumentException(name + " cannot override token IDs in a sequence containing media.");
        int[]? tokens = tokenOverride ?? (textOnly ? textModel.Tokenize(string.Concat(parts.Select(part => part.Value))) : null);
        if (tokens != null && tokens.Length == 0) throw new ArgumentException(name + " token input must not be empty.");
        Task<EmbeddingBatchResult> Embed() => tokenOverride != null ? textModel.EmbedTokensAsync([tokenOverride]) : multimodal != null
            ? multimodal.EmbedAsync(parts)
            : textOnly ? textModel.EmbedTokensAsync([tokens!])
            : throw new ArgumentException("Media cases require --mmproj.");
        for (int index = 0; index < warmup; ++index) await Embed();
        var samples = new List<double>();
        EmbeddingBatchResult result = null!;
        for (int index = 0; index < repeats; ++index)
        {
            long started = Stopwatch.GetTimestamp();
            result = await Embed();
            samples.Add(Stopwatch.GetElapsedTime(started).TotalMilliseconds);
        }
        if (result.Embeddings.Length != 1) throw new InvalidDataException(name + " must return one embedding.");
        float[] vector = result.Embeddings[0];
        double norm = Math.Sqrt(vector.Sum(value => (double)value * value));
        if (vector.Length != 768 || vector.Any(value => !float.IsFinite(value)) || Math.Abs(norm - 1) > 1e-5)
            throw new InvalidDataException(name + " returned invalid normalized 768-dimensional embeddings.");
        double? rawCosine = null;
        if (tokens != null)
        {
            float[] raw = textModel.GetTokenEmbeddings(tokens);
            var fromRaw = await textModel.EmbedInputsAsync(raw, tokens.Length);
            rawCosine = Cosine(vector, fromRaw.Embeddings[0]);
            if (!(rawCosine > .99999)) throw new InvalidDataException($"{name} raw/token cosine {rawCosine:R}.");
        }
        var sorted = samples.Order().ToArray();
        double median = sorted.Length % 2 == 0 ? (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2 : sorted[sorted.Length / 2];
        byte[]? tokenBytes = tokens == null ? null : new byte[tokens.Length * sizeof(int)];
        if (tokens != null)
            for (int index = 0; index < tokens.Length; ++index)
                BinaryPrimitives.WriteInt32LittleEndian(tokenBytes!.AsSpan(index * sizeof(int)), tokens[index]);
        var mediaFiles = parts.Where(part => part.Type != "text").Select(part => new { kind = part.Type,
            path = part.Value, sha256 = HashFile(part.Value) }).ToArray();
        results.Add(new { name, prompt_tokens = result.PromptTokens, norm, raw_token_cosine = rawCosine,
            input_tokens = tokens, input_token_sha256 = tokenBytes == null ? null : Convert.ToHexString(SHA256.HashData(tokenBytes)).ToLowerInvariant(),
            media_files = mediaFiles,
            embedding = vector, samples_ms = samples, median_ms = median,
            p95_ms = sorted[Math.Min(sorted.Length - 1, (int)Math.Ceiling(sorted.Length * .95) - 1)] });
        Console.WriteLine($"{name}: {result.PromptTokens} tokens, {median:F3} ms, norm {norm:R}");
        WriteReport(false);
    }
    completed = true;
}
catch
{
    WriteReport(false);
    throw;
}
finally
{
    // This standalone process owns every model and the global GGML operator
    // registry. Release those resources before declaring end-to-end success.
    // A shared application must keep Shutdown outside an individual model.
    multimodal?.Dispose();
    ownedTextModel?.Dispose();
    if (shutdownGgml && (backend.StartsWith("GGML", StringComparison.OrdinalIgnoreCase) || multimodal != null && backend.Equals("CPU", StringComparison.OrdinalIgnoreCase)))
        TensorSharp.GGML.GgmlBasicOps.Shutdown();
    WriteReport(completed);
}
