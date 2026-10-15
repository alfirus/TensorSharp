using System.Buffers.Binary;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Models.Embeddings;

// A bounded API diagnostic using the exact inputs of a preserved HTTP report.
// It does not replace paired HTTP measurements or independent model goldens.
var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int index = 0; index < args.Length; index += 2)
{
    if (!args[index].StartsWith("--") || index + 1 >= args.Length || !options.TryAdd(args[index][2..], args[index + 1]))
        throw new ArgumentException("Pass --model <gguf> --http-report <json> --output <json> [--threads <n>].");
}
string Required(string key) => options.TryGetValue(key, out var value) ? Path.GetFullPath(value) : throw new ArgumentException("Missing --" + key);
string modelPath = Required("model"), inputPath = Required("http-report"), outputPath = Required("output");
int threads = int.Parse(options.GetValueOrDefault("threads", "8"));
if (threads < 1 || threads > 512) throw new ArgumentOutOfRangeException(nameof(threads));
if (Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_CPU_PROFILE") != "1")
    throw new ArgumentException("Enable the buffered CPU profiler before loading the model.");
string HashFile(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
}
using var manifest = JsonDocument.Parse(File.ReadAllText(inputPath));
string[] names = ["single_short", "single_medium", "single_long", "batch8_short", "batch8_medium", "batch32_short", "batch_mixed"];
var inputs = manifest.RootElement.GetProperty("benchmark_inputs");
if (inputs.EnumerateObject().Count() != names.Length || names.Any(name => !inputs.TryGetProperty(name, out _)))
    throw new InvalidDataException("Require exactly the seven preserved standard text scenarios.");
string modelHash = HashFile(modelPath);
if (modelHash != manifest.RootElement.GetProperty("model_sha256").GetString())
    throw new InvalidDataException("The diagnostic model differs from the preserved HTTP measurement.");
var artifacts = new[] { typeof(EmbeddingModel).Assembly.Location,
    Environment.GetEnvironmentVariable("TENSORSHARP_CPU_NATIVE_LIBRARY") ?? string.Empty }
    .Where(File.Exists).Select(path => new { path = Path.GetFullPath(path), sha256 = HashFile(path) }).ToArray();
var results = new List<object>();
bool completed = false;
double apiTotal = 0, tokenTotal = 0, tokenizeTotal = 0, warmTotal = 0;
long apiCalls = 0, tokenCalls = 0, warmCalls = 0, computeCallsExpected = 0, computeRowsExpected = 0;
int CompactBatches(int[][] tokens)
{
    var lengths = tokens.Select(row => row.Length).Order().ToArray();
    int calls = 0;
    for (int first = 0; first < lengths.Length; ++calls)
    {
        int next = first + 1, count = lengths[first];
        while (next < lengths.Length && next - first < 64 && count + lengths[next] <= 4096)
            count += lengths[next++];
        first = next;
    }
    return calls;
}
void CheckResult(EmbeddingBatchResult result, int[][] tokens, int dimensions)
{
    if (result.Embeddings.Length != tokens.Length || result.PromptTokens != tokens.Sum(row => row.Length))
        throw new InvalidDataException("Embedding result count/token count differs from the exact inputs.");
    foreach (var vector in result.Embeddings)
    {
        if (vector.Length != dimensions || vector.Any(value => !float.IsFinite(value)))
            throw new InvalidDataException("Invalid embedding dimension or finite-value invariant.");
        double norm = Math.Sqrt(vector.Sum(value => (double)value * value));
        if (Math.Abs(norm - 1) > 1e-5) throw new InvalidDataException("Embedding normalization invariant failed.");
    }
}
void RequireIdentical(EmbeddingBatchResult left, EmbeddingBatchResult right)
{
    if (left.PromptTokens != right.PromptTokens || left.Embeddings.Length != right.Embeddings.Length)
        throw new InvalidDataException("String/direct-token results differ.");
    for (int row = 0; row < left.Embeddings.Length; ++row)
        for (int index = 0; index < left.Embeddings[row].Length; ++index)
            if (BitConverter.SingleToInt32Bits(left.Embeddings[row][index]) != BitConverter.SingleToInt32Bits(right.Embeddings[row][index]))
                throw new InvalidDataException($"String/direct-token bit identity failed at row {row}, column {index}.");
}
void WriteReport()
{
    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
    File.WriteAllText(outputPath, JsonSerializer.Serialize(new
    {
        completed, created_utc = DateTimeOffset.UtcNow, backend = "CPU", graph = "managed",
        model = modelPath, model_sha256 = modelHash, threads,
        http_report = inputPath, http_report_sha256 = HashFile(inputPath), binary_artifacts = artifacts,
        flags = new[] { "TENSORSHARP_EMBEDDING_CPU_PROFILE", "TENSORSHARP_EMBEDDING_Q16_AVX512",
            "TENSORSHARP_EMBEDDING_NATIVE_Q16", "TENSORSHARP_EMBEDDING_NATIVE_ATTENTION", "TENSORSHARP_EMBEDDING_FUSED_GATE_GELU", "TS_CPU_DISABLE_AVX512" }
            .ToDictionary(key => key, Environment.GetEnvironmentVariable),
        measured_string_api_ms = apiTotal, measured_direct_token_api_ms = tokenTotal,
        separate_tokenization_ms = tokenizeTotal, warmup_api_ms = warmTotal,
        measured_string_calls = apiCalls, measured_direct_token_calls = tokenCalls, warmup_calls = warmCalls,
        expected_successful_compute_calls = computeCallsExpected, expected_compute_rows = computeRowsExpected,
        total_enclosing_embedding_api_ms = apiTotal + tokenTotal + warmTotal,
        cases = results,
        limitations = new[] {
            "Bounded diagnostic: one direct-token warmup and two AB/BA pairs per scenario, includes profiling overhead and task scheduling; no formal performance claim.",
            "Separate tokenization is measured before embedding. Its elapsed time is not part of the direct-token API; it overlaps the corresponding work already included in string API time.",
            "The disposal profile includes every warmup and measured embedding call. Compute is nested inside the recorded total enclosing API time; never add the two totals.",
            "The remaining API duration also includes input preparation, cloning, sorting, batch assembly, scheduling, output allocation, profiling and timing overhead. It cannot be called HTTP overhead.",
            "This tool does not perform HTTP requests. The preserved HTTP report identifies inputs and prior measurements, not an additional controlled comparison.",
            "Bit identity proves string/direct-token API equivalence for these inputs; independent NumPy quality qualification remains a separate required gate." },
    }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
}
EmbeddingModel? model = null;
try
{
    model = EmbeddingModel.Load(modelPath, new EmbeddingModelOptions { Backend = "CPU", Threads = threads });
    if (!model.IsManaged || model.Architecture != "gemma-embedding2") throw new InvalidDataException("Require the managed EmbeddingGemma 2 CPU graph.");
    foreach (string name in names)
    {
        string[] text = inputs.GetProperty(name).EnumerateArray().Select(item => item.GetString() ?? throw new InvalidDataException("Null input text.")).ToArray();
        if (text.Length == 0) throw new InvalidDataException("Empty scenario.");
        var tokenSamples = new List<double>();
        int[][] tokens = [];
        for (int iteration = 0; iteration < 2; ++iteration)
        {
            long started = Stopwatch.GetTimestamp();
            var current = text.Select(value => model.Tokenize(value)).ToArray();
            double ms = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
            if (iteration > 0 && (current.Length != tokens.Length || current.Where((row, index) => !row.SequenceEqual(tokens[index])).Any()))
                throw new InvalidDataException("Tokenization is not deterministic.");
            tokens = current; tokenSamples.Add(ms); tokenizeTotal += ms;
        }
        int batches = CompactBatches(tokens), rows = tokens.Sum(row => row.Length);
        var httpCase = manifest.RootElement.GetProperty("engines").GetProperty("tensorsharp").GetProperty("benchmarks").GetProperty(name);
        if (httpCase.GetProperty("prompt_tokens").GetInt32() != rows || httpCase.GetProperty("texts").GetInt32() != text.Length)
            throw new InvalidDataException("Token/text counts differ from the preserved actual HTTP response.");
        long warmStart = Stopwatch.GetTimestamp();
        var warm = await model.EmbedTokensAsync(tokens);
        double warmMs = Stopwatch.GetElapsedTime(warmStart).TotalMilliseconds;
        warmTotal += warmMs; ++warmCalls; computeCallsExpected += batches; computeRowsExpected += rows;
        CheckResult(warm, tokens, model.Dimensions);
        var apiSamples = new List<double>(); var directSamples = new List<double>();
        for (int order = 0; order < 2; ++order)
        {
            EmbeddingBatchResult? api = null, direct = null;
            for (int slot = 0; slot < 2; ++slot)
            {
                bool stringApi = (order + slot) % 2 == 0;
                long started = Stopwatch.GetTimestamp();
                var result = stringApi ? await model.EmbedAsync(text) : await model.EmbedTokensAsync(tokens);
                double ms = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
                CheckResult(result, tokens, model.Dimensions); RequireIdentical(warm, result);
                if (stringApi) { api = result; apiSamples.Add(ms); apiTotal += ms; ++apiCalls; }
                else { direct = result; directSamples.Add(ms); tokenTotal += ms; ++tokenCalls; }
                computeCallsExpected += batches; computeRowsExpected += rows;
            }
            RequireIdentical(api!, direct!);
        }
        byte[] tokenBytes = new byte[checked(rows * sizeof(int))];
        int offset = 0;
        foreach (var row in tokens)
            foreach (int token in row) { BinaryPrimitives.WriteInt32LittleEndian(tokenBytes.AsSpan(offset, sizeof(int)), token); offset += sizeof(int); }
        results.Add(new { name, sequence_lengths = tokens.Select(row => row.Length).ToArray(), tokens,
            tokens_sha256 = Convert.ToHexString(SHA256.HashData(tokenBytes)).ToLowerInvariant(),
            text_utf8_sha256 = text.Select(value => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(value))).ToLowerInvariant()).ToArray(),
            compact_compute_calls_per_api = batches, prompt_tokens = rows, api_direct_bit_identical = true,
            separate_tokenization_samples_ms = tokenSamples, warmup_ms = warmMs,
            string_api_samples_ms = apiSamples, direct_token_api_samples_ms = directSamples });
    }
    model.Dispose(); model = null;
    completed = true;
}
finally
{
    try { model?.Dispose(); }
    finally { WriteReport(); }
}
