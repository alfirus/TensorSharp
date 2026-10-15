using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Models.Embeddings;
using TensorSharp.Runtime;

var options = BatchOptions.Parse(args);
string output = Path.GetFullPath(options.Output);
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
using var destination = new FileStream(output, FileMode.CreateNew, FileAccess.Write, FileShare.None);
string Hash(string path) { using var file = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant(); }
string ByteHash(byte[] value) => Convert.ToHexString(SHA256.HashData(value)).ToLowerInvariant();
void Require(bool condition, string reason) { if (!condition) throw new InvalidDataException(reason); }
var report = new Dictionary<string, object?> { ["completed"] = false, ["passed"] = false, ["available"] = false,
    ["scope"] = "Frozen native GGML_CPU raw-token batching quality; explicit gated native wrappers versus public singleton/API outputs. No timing or inferred native dispatch counters.",
    ["workers"] = options.Workers, ["policy"] = options.Policy, ["diagnostic"] = options.Diagnostic,
    ["include_long"] = options.IncludeLong, ["case_selection"] = options.Cases, ["route_set"] = options.RouteSet,
    ["quality_qualified"] = false, ["full_core_qualified"] = false, ["long_branch_qualified"] = false,
    ["performance_qualified"] = false, ["native_admission_observed"] = false,
    ["batch_single_cosine_minimum"] = .99999, ["comparison_max_absolute_error"] = .0005 };
var before = new Dictionary<string, string>(StringComparer.Ordinal);
report["before_hashes"] = before;
var results = new List<object>(); report["cases"] = results;
EmbeddingModel? model = null;
bool passed = true;
int publicTokenCalls = 0, publicInputCalls = 0, nativeTokenCalls = 0, nativeInputCalls = 0, tokenRowCalls = 0;
try
{
    Require(OperatingSystem.IsLinux() && RuntimeInformation.ProcessArchitecture == Architecture.X64, "Actual CPU candidate qualification requires Linux x64; local ARM builds/pure controls are not model qualification.");
    string snapshot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(typeof(Program).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>().Single(x => x.Key == "TensorSharpBatchQualitySnapshot").Value!));
    report["snapshot"] = snapshot;
    string modelPath = Path.GetFullPath(options.Model), native = Path.Combine(AppContext.BaseDirectory, "libGgmlOps.so");
    string models = typeof(EmbeddingModel).Assembly.Location, runtime = typeof(ITokenizer).Assembly.Location;
    string server = Path.Combine(AppContext.BaseDirectory, "TensorSharp.Server.dll");
    foreach (var (path, expected) in new[] { (modelPath, options.ModelSha), (native, options.NativeSha), (models, options.ModelsSha), (runtime, options.RuntimeSha), (server, options.ServerSha) })
    {
        Require(File.Exists(path), "Required model/runtime unavailable: " + path);
        Require(Hash(path) == expected, "Expected input identity differs: " + path); before[path] = expected;
    }
    foreach (string frozen in Directory.EnumerateFiles(snapshot, "*", SearchOption.AllDirectories)
        .Where(path => path.Contains(Path.DirectorySeparatorChar + "runtimes" + Path.DirectorySeparatorChar, StringComparison.Ordinal) ||
            Path.GetDirectoryName(path) == snapshot && (Path.GetExtension(path) is ".dll" or ".so" or ".dylib" || Path.GetFileName(path).Contains(".so.", StringComparison.Ordinal)))
        .Where(path => Path.GetFileName(path) is not "TensorSharp.Server.Host.dll" and not "GgmlOps.dll"))
    {
        string copied = Path.Combine(AppContext.BaseDirectory, Path.GetRelativePath(snapshot, frozen)), digest = Hash(frozen);
        Require(File.Exists(copied) && Hash(copied) == digest, "Copied frozen dependency differs: " + copied);
        Require(!before.TryGetValue(copied, out var previous) || previous == digest, "Validated dependency changed during snapshot binding.");
        before[frozen] = digest; before[copied] = digest;
    }
    before[typeof(Program).Assembly.Location] = Hash(typeof(Program).Assembly.Location);
    var environment = DefaultHttpSource.RequiredOptimizationFlags.ToDictionary(key => key, Environment.GetEnvironmentVariable);
    var optional = new HashSet<string>(["TSG_EMBEDDING_LINUX_BLAS", "TSG_EMBEDDING_CPU_ATTENTION", "TSG_EMBEDDING_BLAS_LIBRARY"]);
    var diagnostics = new HashSet<string>(["TSG_EMBEDDING_CPU_PROFILE", "TSG_EMBEDDING_CPU_ATTENTION_PROFILE"]);
    foreach (var flag in environment)
        if (!(options.Policy == "optional" && optional.Contains(flag.Key)) && !(options.Diagnostic && diagnostics.Contains(flag.Key)))
            Require(flag.Value == null, "Unqualified compute or instrumentation override: " + flag.Key);
    if (options.Policy == "optional")
    {
        Require(Environment.GetEnvironmentVariable("TSG_EMBEDDING_LINUX_BLAS") == "1" && Environment.GetEnvironmentVariable("TSG_EMBEDDING_CPU_ATTENTION") == "1" && Environment.GetEnvironmentVariable("OPENBLAS_NUM_THREADS") == "1", "Optional policy requires explicit BLAS/attention1 and one OpenBLAS worker.");
        string provider = Environment.GetEnvironmentVariable("TSG_EMBEDDING_BLAS_LIBRARY") ?? throw new InvalidDataException("Explicit qualified BLAS provider required.");
        Require(File.Exists(provider), "Optional provider unavailable."); before[Path.GetFullPath(provider)] = Hash(provider);
    }
    string? threadEnvironment = Environment.GetEnvironmentVariable("TS_GGML_CPU_THREADS");
    Require(threadEnvironment == null || threadEnvironment == options.Workers.ToString(), "Conflicting process CPU thread override.");
    report["optimization_flags"] = environment;
    Require(Environment.GetEnvironmentVariable("GGML_CPU_DISABLE_FUSION") == null, "Unqualified upstream fusion override.");
    report["thread_environment"] = new[] { "TS_GGML_CPU_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME", "DOTNET_PROCESSOR_COUNT", "GGML_CPU_DISABLE_FUSION" }.ToDictionary(key => key, Environment.GetEnvironmentVariable);
    int[]? longTokens = null;
    if (options.IncludeLong)
    {
        string fixturePath = Path.GetFullPath(options.LongFixture!); byte[] bytes = File.ReadAllBytes(fixturePath);
        Require(ByteHash(bytes) == options.LongFixtureSha, "Original long fixture SHA differs."); before[fixturePath] = options.LongFixtureSha!;
        using var fixture = JsonDocument.Parse(bytes);
        if (fixture.RootElement.TryGetProperty("tokens", out var raw)) longTokens = raw.EnumerateArray().Select(x => x.GetInt32()).ToArray();
        else
        {
            var matching = fixture.RootElement.GetProperty("cases").EnumerateArray().Where(x => x.TryGetProperty("input_tokens", out var tokens) && tokens.ValueKind == JsonValueKind.Array && tokens.GetArrayLength() == 8192).ToArray();
            Require(matching.Length == 1, "Require exactly one original8192 input_tokens case.");
            longTokens = matching[0].GetProperty("input_tokens").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        }
        Require(longTokens.Length == 8192, "Long fixture length differs.");
        report["long_fixture"] = new { path = fixturePath, sha256 = options.LongFixtureSha, original8192_token_sha256_le = BatchMath.TokenHash(longTokens), shorter8191 = "drop original penultimate token; retain original EOS" };
    }
    Require(before.All(pair => Hash(pair.Key) == pair.Value), "Bound inputs changed before model load.");
    model = EmbeddingModel.Load(modelPath, new() { Backend = "GGML_CPU", Threads = options.Workers });
    Require(model.Backend == "GGML_CPU" && !model.IsManaged && model.Architecture == "gemma-embedding2" && model.HiddenSize == 512 && model.Dimensions == 768 && model.MaxTokens == 8192, "Required qualified native model geometry unavailable.");
    report["available"] = true;
    int[] seed = model.Tokenize("title: add.py | text: def add(a, b): return a + b");
    var cases = BatchFixtures.Select(BatchFixtures.Create(seed, longTokens), options.Cases);
    bool nativeOnly = options.RouteSet == "native-token-only";
    Require(cases.SelectMany(c => c.Tokens).SelectMany(row => row).All(id => id >= 0 && id < model.VocabularySize), "Fixture token outside vocabulary.");
    report["fixture_manifest"] = cases.Select(c => new { plan = c.Plan(nativeOnly), tokens = c.Tokens, token_sha256_le = c.Tokens.Select(BatchMath.TokenHash).ToArray() }).ToArray();
    report["planned_learned_forwards"] = cases.Sum(c => c.LearnedForwards(nativeOnly));
    report["planned_actual_learned_token_rows"] = cases.Sum(c => c.LearnedTokenRows(nativeOnly));
    report["planned_token_row_graphs"] = nativeOnly ? 0 : cases.Sum(c => c.Unique);
    report["planned_quality_route_set"] = !nativeOnly;
    report["forward_budget_scope"] = "Public internal microbatch forward counts are source-derived from the bound frozen API policy; wrapper/API call attempts are observed separately. Token-row graphs are separate from learned forward counts.";
    report["long_case_omitted"] = !options.IncludeLong;
    var handleField = typeof(EmbeddingModel).GetField("_handle", BindingFlags.Instance | BindingFlags.NonPublic)!;
    var modelGate = (SemaphoreSlim)typeof(EmbeddingModel).GetField("_gate", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
    Require(handleField.FieldType == typeof(IntPtr), "Frozen native handle contract differs.");
    IntPtr handle = (IntPtr)handleField.GetValue(model)!; Require(handle != IntPtr.Zero, "Native model handle unavailable.");
    var resize = Assembly.LoadFrom(server).GetType("TensorSharp.Server.ProtocolAdapters.EmbeddingAdapter", true)!
        .GetMethod("ResizeAndNormalize", BindingFlags.Static | BindingFlags.NonPublic) ?? throw new MissingMethodException("Frozen MRL helper unavailable.");
    float[][] Native(int[][] tokens, float[][]? inputs)
    {
        int[] lengths = tokens.Select(row => row.Length).ToArray();
        Require(lengths.All(length => length is > 0 and <= 8192) && lengths.Sum(length => (length + 15) / 16 * 16) <= 65536, "Native padded cap exceeded.");
        int[] ownedTokens = tokens.SelectMany(row => row).ToArray();
        float[]? ownedInputs = inputs?.SelectMany(row => row).ToArray();
        Require(inputs == null || inputs.Length == lengths.Length && inputs.Zip(lengths).All(pair => pair.First.Length == pair.Second * 512 && pair.First.All(float.IsFinite)), "Invalid complete external rows.");
        float[] vectors = new float[checked(tokens.Length * 768)];
        modelGate.Wait();
        try
        {
            if (ownedInputs == null) { ++nativeTokenCalls; GgmlEmbeddingNative.Encode(handle, ownedTokens, lengths, vectors); }
            else { ++nativeInputCalls; GgmlEmbeddingNative.EncodeInputs(handle, ownedInputs, lengths, vectors); }
        }
        finally { modelGate.Release(); GC.KeepAlive(model); }
        Require(lengths.SequenceEqual(tokens.Select(row => row.Length)), "Native wrapper changed input lengths.");
        Require(BatchMath.TokenHash(ownedTokens) == BatchMath.TokenHash(tokens.SelectMany(row => row).ToArray()), "Native wrapper changed token input.");
        if (ownedInputs != null) Require(BatchMath.FloatHash(ownedInputs) == BatchMath.FloatHash(inputs!.SelectMany(row => row).ToArray()), "Native wrapper changed external input.");
        return Enumerable.Range(0, tokens.Length).Select(i => vectors.AsSpan(i * 768, 768).ToArray()).ToArray();
    }
    foreach (var c in cases)
    {
        Console.WriteLine("BATCH_QUALITY_CASE " + c.Name);
        var singletonEvidence = new List<object>();
        var observations = new List<object>();
        var caseEvidence = new Dictionary<string, object?> { ["name"] = c.Name, ["plan"] = c.Plan(nativeOnly), ["tokens"] = c.Tokens,
            ["singleton_evidence"] = singletonEvidence, ["routes"] = observations, ["completed"] = false,
            ["quality_route_set_executed"] = !nativeOnly, ["phase"] = "start" };
        results.Add(caseEvidence);
        string[] tokenBefore = c.Tokens.Select(BatchMath.TokenHash).ToArray();
        var groups = c.Tokens.GroupBy(BatchMath.TokenHash).ToArray();
        var singles = new Dictionary<string, float[]>(); var external = new Dictionary<string, float[]>();
        foreach (var group in nativeOnly ? [] : groups)
        {
            int[] tokens = group.First();
            caseEvidence["phase"] = "public-singleton:" + group.Key;
            ++publicTokenCalls; var single = model.EmbedTokensAsync([tokens]).GetAwaiter().GetResult();
            Require(single.PromptTokens == tokens.Length && single.Embeddings.Length == 1, "Public singleton result contract differs.");
            BatchMath.Normalized(single.Embeddings[0], 768); singles[group.Key] = single.Embeddings[0];
            caseEvidence["phase"] = "token-rows:" + group.Key;
            ++tokenRowCalls; float[] rows = model.GetTokenEmbeddings(tokens);
            Require(rows.Length == tokens.Length * 512 && rows.All(float.IsFinite), "Token rows invalid."); external[group.Key] = rows;
            float[]? inputVector = null; VectorGate? rawGate = null;
            if (!c.Long)
            {
                caseEvidence["phase"] = "external-singleton:" + group.Key;
                string rowHash = BatchMath.FloatHash(rows);
                ++publicInputCalls; var fromInputs = model.EmbedInputsAsync(rows, tokens.Length).GetAwaiter().GetResult();
                Require(fromInputs.PromptTokens == tokens.Length && fromInputs.Embeddings.Length == 1, "Public external singleton result contract differs.");
                Require(BatchMath.FloatHash(rows) == rowHash, "Public singleton changed source external rows.");
                BatchMath.Normalized(fromInputs.Embeddings[0], 768);
                inputVector = fromInputs.Embeddings[0]; rawGate = BatchMath.Compare(single.Embeddings[0], inputVector); passed &= rawGate.Passed;
            }
            singletonEvidence.Add(new { token_sha256_le = group.Key, tokens, token_rows_sha256_le = BatchMath.FloatHash(rows),
                token_embedding = single.Embeddings[0], token_embedding_sha256_le = BatchMath.FloatHash(single.Embeddings[0]),
                external_embedding = inputVector, external_embedding_sha256_le = inputVector == null ? null : BatchMath.FloatHash(inputVector), raw_token_external_gate = rawGate });
        }
        var externalBefore = external.ToDictionary(pair => pair.Key, pair => BatchMath.FloatHash(pair.Value));
        void Observe(string name, Func<float[][]> execute, bool reversed = false)
        {
            caseEvidence["phase"] = name;
            var routeEvidence = new Dictionary<string, object?> { ["route"] = name, ["completed"] = false,
                ["returned_order_reversed_for_comparison"] = reversed, ["batch_single_comparison_executed"] = !nativeOnly };
            observations.Add(routeEvidence);
            var returned = execute();
            Require(returned.Length == c.Tokens.Length, "Whole batch output count differs.");
            var vectors = reversed ? returned.Reverse().ToArray() : returned;
            // Reject nonfinite data before JSON serialization; the phase/error remains preserved.
            Require(vectors.All(vector => vector.Length == 768 && vector.All(float.IsFinite)), "Returned complete vectors invalid/nonfinite.");
            routeEvidence["vectors_in_original_order"] = vectors;
            routeEvidence["vector_sha256_le"] = vectors.Select(BatchMath.FloatHash).ToArray();
            var gates = new List<object>();
            routeEvidence["singleton_and_mrl_gates"] = gates;
            for (int i = 0; i < vectors.Length; ++i)
            {
                float[]? expected = nativeOnly ? null : singles[BatchMath.TokenHash(c.Tokens[i])];
                BatchMath.Normalized(vectors[i], 768); var full = expected == null ? null : BatchMath.Compare(expected, vectors[i]);
                if (full != null) passed &= full.Passed;
                var prefixes = new List<object>();
                foreach (int width in new[] { 128,256,512 })
                {
                    var expectedPrefix = expected == null ? null : (float[])resize.Invoke(null, [expected, width])!;
                    var actualPrefix = (float[])resize.Invoke(null, [vectors[i], width])!;
                    var reference = BatchMath.Prefix(vectors[i], width);
                    BatchMath.Normalized(actualPrefix, width); var prefixGate = expectedPrefix == null ? null : BatchMath.Compare(expectedPrefix, actualPrefix);
                    Require(reference.Zip(actualPrefix).All(pair => Math.Abs((double)pair.First - pair.Second) <= 1e-6), "Actual MRL helper differs from independently normalized prefix.");
                    if (prefixGate != null) passed &= prefixGate.Passed;
                    prefixes.Add(new { dimensions = width, vector = actualPrefix, sha256_le = BatchMath.FloatHash(actualPrefix), singleton_gate = prefixGate });
                }
                gates.Add(new { index = i, full_gate = full, mrl = prefixes });
            }
            var duplicateGates = new List<VectorGate>();
            foreach (var group in Enumerable.Range(0,c.Tokens.Length).GroupBy(i => BatchMath.TokenHash(c.Tokens[i])))
                foreach (int index in group.Skip(1)) duplicateGates.Add(BatchMath.Compare(vectors[group.First()], vectors[index]));
            passed &= duplicateGates.All(gate => gate.Passed);
            routeEvidence["duplicate_gates"] = duplicateGates;
            Require(c.Tokens.Select(BatchMath.TokenHash).SequenceEqual(tokenBefore), "Source token fixture mutated.");
            Require(external.All(pair => BatchMath.FloatHash(pair.Value) == externalBefore[pair.Key]), "Source external rows mutated.");
            routeEvidence["completed"] = true;
        }
        Observe("native-whole-token-batch", () => Native(c.Tokens, null));
        if (!nativeOnly)
        {
            Observe("native-whole-token-reversed", () => Native(c.Tokens.Reverse().ToArray(), null), true);
            float[][] inputRows = c.Tokens.Select(row => external[BatchMath.TokenHash(row)]).ToArray();
            Observe("native-whole-external-batch", () => Native(c.Tokens, inputRows));
            if (!c.Long)
            {
                Observe("native-whole-external-reversed", () => Native(c.Tokens.Reverse().ToArray(), inputRows.Reverse().ToArray()), true);
                Observe("public-sorted-microbatch", () =>
                {
                    ++publicTokenCalls; var publicBatch = model.EmbedTokensAsync(c.Tokens).GetAwaiter().GetResult();
                    Require(publicBatch.PromptTokens == c.Lengths.Sum() && publicBatch.Embeddings.Length == c.Tokens.Length, "Public batch result contract differs.");
                    return publicBatch.Embeddings;
                });
            }
        }
        caseEvidence["phase"] = "complete"; caseEvidence["completed"] = true;
    }
    int expectedTokenCalls = nativeOnly ? 0 : cases.Sum(c => c.Unique + (c.Long ? 0 : 1));
    Require(publicTokenCalls == expectedTokenCalls && publicInputCalls == (nativeOnly ? 0 : cases.Where(c => !c.Long).Sum(c => c.Unique)) &&
        nativeTokenCalls == cases.Length * (nativeOnly ? 1 : 2) && nativeInputCalls == (nativeOnly ? 0 : cases.Sum(c => c.Long ? 1 : 2)) && tokenRowCalls == (nativeOnly ? 0 : cases.Sum(c => c.Unique)), "Observed API/wrapper call budget differs.");
    using var process = Process.GetCurrentProcess();
    var modules = process.Modules.Cast<ProcessModule>().Where(module => Path.GetFileName(module.FileName).StartsWith("libGgmlOps.so", StringComparison.Ordinal)).ToArray();
    Require(modules.Length == 1 && Hash(modules[0].FileName) == options.NativeSha, "Actual loaded native identity differs.");
    report["loaded_native"] = new { path = modules[0].FileName, sha256 = Hash(modules[0].FileName) };
    report["completed"] = true; report["passed"] = passed;
    report["quality_qualified"] = passed && !nativeOnly && !options.Diagnostic;
    report["full_core_qualified"] = report["quality_qualified"] is true && BatchFixtures.Names[..^1].All(name => cases.Any(c => c.Name == name));
    report["long_branch_qualified"] = report["quality_qualified"] is true && cases.Any(c => c.Long);
}
catch (Exception error) { report["error"] = error.ToString(); report["completed"] = false; report["passed"] = false; }
finally
{
    try { model?.Dispose(); report["model_disposed"] = model != null; }
    catch (Exception error) { report["dispose_error"] = error.ToString(); report["completed"] = report["passed"] = report["quality_qualified"] = report["full_core_qualified"] = report["long_branch_qualified"] = false; }
    report["observed_probe_call_attempts"] = new { public_token_api = publicTokenCalls, public_external_singleton_api = publicInputCalls, native_token_wrappers = nativeTokenCalls, native_external_wrappers = nativeInputCalls, token_row_api = tokenRowCalls,
        scope = "Probe-observed calls only; expected internal public microbatch graph count is source-derived and separately named. No fake native kernel counters." };
    var after = new Dictionary<string, string?>();
    foreach (var pair in before)
        try { after[pair.Key] = File.Exists(pair.Key) ? Hash(pair.Key) : null; }
        catch (Exception error) { after[pair.Key] = null; report["hash_guard_error"] = error.ToString(); }
    bool unchanged = before.Count > 0 && before.All(pair => after[pair.Key] == pair.Value); report["after_hashes"] = after; report["input_binary_hashes_unchanged"] = unchanged;
    if (!unchanged) report["completed"] = report["passed"] = report["quality_qualified"] = report["full_core_qualified"] = report["long_branch_qualified"] = false;
    report["limitations"] = new[] { "Quality-only frozen native wrapper/public API comparison, not timing or independent NumPy model arithmetic.",
        "Existing4-worker facts do not exercise candidate7/8-worker admission. Actual native admission/profile remains separately required.",
        "Optional BLAS/attention selectors and bound provider are recorded; successful optional operator execution requires separately bound native disposal/profile logs, not inferred counters.",
        "Explicit case subsets qualify only those cases. Diagnostic native-token-only executes one whole batch per selected case, omits all singleton comparisons/token-row graphs and cannot qualify batch/single quality.",
        "Longcase uses original8192-token fixture and8191withEOS retained. Public batch would split; long whole-batch checks deliberately use the existing gated native wrapper. Long external singleton/reversal omitted to bound resources, explicitly null/unexecuted.",
        "MRL invokes actual frozen server helper and independently checks prefix normalization; no HTTP protocol qualification.",
        "All full vectors/token IDs are saved for independent baseline/candidate comparison; failed gates retained, unavailable model/device never passing." };
    JsonSerializer.Serialize(destination, report, new JsonSerializerOptions { WriteIndented = true });
}
return report["completed"] is true && report["passed"] is true ? 0 : 1;
