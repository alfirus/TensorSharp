using System.Diagnostics;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Models.Embeddings;
using TensorSharp.Runtime;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
string[] required = ["--model", "--projector", "--wav", "--oracle", "--oracle-sha256", "--audio-oracle", "--policy", "--output",
    "--expected-models-sha256", "--expected-runtime-sha256", "--expected-native-sha256"];
string[] optional = ["--mode", "--prewarm-image"];
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 >= args.Length || !required.Contains(args[i]) && !optional.Contains(args[i]) || !options.TryAdd(args[i], args[i + 1])) throw new ArgumentException("Pass every documented unique option.");
if (required.Any(x => !options.ContainsKey(x))) throw new ArgumentException("Missing required scheduling-probe option.");
var mode = SchedulingModes.Parse(options.GetValueOrDefault("--mode"), options.GetValueOrDefault("--prewarm-image"));
bool wholeCallbackMode = mode == SchedulingMode.ThreadPoolResident;
bool publicResidentMode = mode == SchedulingMode.PublicResident;
string? prewarmImage = options.TryGetValue("--prewarm-image", out string? image) ? Path.GetFullPath(image) : null;
string output = Path.GetFullPath(options["--output"]);
if (File.Exists(output)) throw new IOException("Preserve prior evidence; output already exists.");
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
string ByteHash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
string FloatHash(float[] values) => Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan()))).ToLowerInvariant();
void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
var report = new Dictionary<string, object?> { ["completed"] = false, ["passed"] = false, ["http_qualified"] = false,
    ["workers_joined_no_active_callbacks"] = false,
    ["scope"] = "Exclusive frozen CPU audio Prepare plus exact public pinned native EncodeInputs wrapper on persistent same/split master threads. No production async scheduling or HTTP latency claim.",
    ["created_utc"] = DateTimeOffset.UtcNow, ["warmup_per_phase"] = 3, ["minimum_repeats"] = 20, ["minimum_timed_seconds"] = 1 };
if (wholeCallbackMode)
{
    report["mode"] = "threadpool-resident";
    report["scope"] = "Exclusive frozen CPU identical synchronous Prepare/clone/Encode callback via natural Task.Run or one resident caller; no production scheduling or HTTP latency claim.";
    report["equal_input_clone_inside_callback"] = true;
    report["prewarm_image"] = null;
}
if (publicResidentMode)
{
    report["mode"] = "public-resident";
    report["scope"] = "Exclusive frozen restored public EmbedAsync API versus an internal resident Prepare/clone/Encode callback. Caller wall time includes all route work; the routes are not identical delegates. No HTTP or production scheduling qualification.";
    report["public_api_internals_observed"] = false;
    report["equal_input_clone_inside_callback"] = null;
    report["prewarm_image"] = null;
}
PersistentWorker? media = null, text = null;
EmbeddingGemma2MultimodalEncoder? model = null;
var before = new Dictionary<string, string>(StringComparer.Ordinal);
bool passed = true;
try
{
    Require(OperatingSystem.IsLinux(), "Real scheduling measurements require Linux/OpenMP; local builds are source checks only.");
    string modelPath = Path.GetFullPath(options["--model"]), projectorPath = Path.GetFullPath(options["--projector"]), wav = Path.GetFullPath(options["--wav"]);
    string oraclePath = Path.GetFullPath(options["--oracle"]), audioPath = Path.GetFullPath(options["--audio-oracle"]);
    string policy = options["--policy"];
    Require(policy is "hf" or "llama_clamp", "Policy must be hf or llama_clamp.");
    Require(Environment.GetEnvironmentVariable("TS_GGML_CPU_THREADS") == "8", "Explicit TS_GGML_CPU_THREADS=8 is required for the audio graph as well as text options.");
    Require(Environment.GetEnvironmentVariable("TS_EMBEDDING_AUDIO_LOG_POLICY") == null, "Use explicit options; audio environment override must be absent.");
    foreach (string flag in new[] { "TSG_EMBEDDING_CPU_PROFILE", "TSG_EMBEDDING_CPU_ATTENTION_PROFILE", "TSG_CUPTI_ACTIVITY_OUTPUT", "TENSORSHARP_EMBEDDING_CPU_PROFILE" })
        Require(Environment.GetEnvironmentVariable(flag) == null, "Profiling/instrumentation must be absent for the scheduling screen.");
    string snapshot = typeof(Program).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>().Single(x => x.Key == "TensorSharpSchedulingProbeSnapshot").Value!;
    string modelsAssembly = typeof(EmbeddingGemma2MultimodalEncoder).Assembly.Location, runtimeAssembly = typeof(ITokenizer).Assembly.Location;
    string native = Path.Combine(AppContext.BaseDirectory, "libGgmlOps.so");
    Require(Hash(modelsAssembly) == options["--expected-models-sha256"] && Hash(runtimeAssembly) == options["--expected-runtime-sha256"], "Loaded frozen Models/Runtime mismatch.");
    Require(Hash(native) == options["--expected-native-sha256"] && Hash(Path.Combine(snapshot, "libGgmlOps.so")) == options["--expected-native-sha256"], "Frozen native library mismatch.");
    byte[] oracleBytes = File.ReadAllBytes(oraclePath), audioBytes = File.ReadAllBytes(audioPath);
    string oracleHash = ByteHash(oracleBytes), audioHash = ByteHash(audioBytes);
    Require(oracleHash == options["--oracle-sha256"], "Independent pooled oracle hash mismatch.");
    using var oracleDocument = JsonDocument.Parse(oracleBytes); var oracle = oracleDocument.RootElement;
    using var audioDocument = JsonDocument.Parse(audioBytes); var audio = audioDocument.RootElement;
    string rawHash = oracle.GetProperty("audio_oracle").ValueKind == JsonValueKind.Object ? oracle.GetProperty("audio_oracle").GetProperty("sha256").GetString()! : oracle.GetProperty("audio_oracle_sha256").GetString()!;
    Require(audioHash == rawHash, "Independent learned audio rows differ.");
    string modelHash = Hash(modelPath), projectorHash = Hash(projectorPath), waveHash = Hash(wav);
    Require(oracle.GetProperty("model_sha256").GetString() == modelHash && oracle.GetProperty("projector_sha256").GetString() == projectorHash && oracle.GetProperty("waveform_sha256").GetString() == waveHash, "Oracle model/projector/waveform differs.");
    Require(audio.GetProperty("projector_sha256").GetString() == projectorHash && audio.GetProperty("waveform_sha256").GetString() == waveHash && audio.GetProperty("audio_log_policy").GetString() == (policy == "hf" ? "huggingface_add" : "llama_clamp"), "Independent frontend policy differs.");
    Require(!oracle.TryGetProperty("prefix_text", out var prefix) || prefix.GetString() == "", "This probe is audio-only without text prefix.");
    int oracleTokens = oracle.GetProperty("total_input_rows").GetInt32();
    double[] expected = oracle.GetProperty("embedding").EnumerateArray().Select(x => x.GetDouble()).ToArray();
    Require(oracleTokens == 29 && expected.Length == 768 && expected.All(double.IsFinite) && Math.Abs(Math.Sqrt(expected.Sum(x => x * x)) - 1) <= 1e-5, "Independent complete pooled vector invalid.");
    before[modelPath] = modelHash; before[projectorPath] = projectorHash; before[wav] = waveHash;
    if (prewarmImage != null) before[prewarmImage] = Hash(prewarmImage);
    before[oraclePath] = oracleHash; before[audioPath] = audioHash;
    before[native] = options["--expected-native-sha256"]; before[Path.Combine(snapshot, "libGgmlOps.so")] = options["--expected-native-sha256"];
    before[modelsAssembly] = options["--expected-models-sha256"]; before[runtimeAssembly] = options["--expected-runtime-sha256"];
    before[typeof(Program).Assembly.Location] = Hash(typeof(Program).Assembly.Location);
    // Bind every copied top-level frozen binary, including the native pinning wrapper.
    foreach (string frozen in Directory.GetFiles(snapshot)
        .Where(x => Path.GetExtension(x) is ".dll" or ".so" or ".dylib" || Path.GetFileName(x).Contains(".so.", StringComparison.Ordinal))
        .Where(x => Path.GetFileName(x) is not "TensorSharp.Server.Host.dll" and not "GgmlOps.dll"))
    {
        string copied = Path.Combine(AppContext.BaseDirectory, Path.GetFileName(frozen));
        string digest = Hash(frozen);
        Require(File.Exists(copied) && Hash(copied) == digest, "A copied frozen dependency differs: " + Path.GetFileName(frozen));
        foreach (string file in new[] { frozen, copied })
        {
            Require(!before.TryGetValue(file, out var validated) || validated == digest, "A validated dependency changed during snapshot binding.");
            before[file] = digest;
        }
    }
    // Native codec/runtime dependencies are copied by the frozen-only project.
    string runtimeRoot = Path.Combine(snapshot, "runtimes");
    foreach (string frozen in Directory.Exists(runtimeRoot) ? Directory.EnumerateFiles(runtimeRoot, "*", SearchOption.AllDirectories) : [])
    {
        string copied = Path.Combine(AppContext.BaseDirectory, "runtimes", Path.GetRelativePath(runtimeRoot, frozen));
        string digest = Hash(frozen);
        Require(File.Exists(copied) && Hash(copied) == digest, "A copied frozen runtime dependency differs.");
        before[frozen] = digest; before[copied] = digest;
    }
    Require(before.All(x => Hash(x.Key) == x.Value), "Validated inputs or frozen dependencies changed before model loading.");
    report["before_hashes"] = before; report["frozen_snapshot"] = snapshot; report["model_sha256"] = modelHash; report["projector_sha256"] = projectorHash; report["waveform_sha256"] = waveHash;
    report["oracle_sha256"] = oracleHash; report["audio_oracle_sha256"] = rawHash; report["audio_log_policy"] = policy; report["backend"] = "GGML_CPU"; report["text_threads"] = 8; report["audio_cpu_threads_environment"] = 8;
    report["optimization_flags"] = DefaultHttpSource.RequiredOptimizationFlags.ToDictionary(x => x, Environment.GetEnvironmentVariable);
    string[] threadFlags = ["TS_GGML_CPU_THREADS", "TS_CPU_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS", "DOTNET_PROCESSOR_COUNT", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME"];
    report["thread_policy_environment"] = threadFlags.ToDictionary(x => x, Environment.GetEnvironmentVariable);
    media = new PersistentWorker("embedding-audio-master0");
    if (!wholeCallbackMode && !publicResidentMode) text = new PersistentWorker("embedding-text-master1");
    model = media.Invoke(() => EmbeddingGemma2MultimodalEncoder.Load(modelPath, projectorPath,
        new EmbeddingModelOptions { Backend = "GGML_CPU", Threads = 8 }, new EmbeddingGemma2MultimodalOptions
        { LoadVision = prewarmImage != null, LoadAudio = true, AudioLogPolicy = policy == "hf" ? Gemma4AudioLogPolicy.HuggingFaceAddFloor : Gemma4AudioLogPolicy.LlamaClampFloor }));
    Require(model.SupportsAudio && model.SupportsImages == (prewarmImage != null) && model.TextModel.Backend == "GGML_CPU" && !model.TextModel.IsManaged && model.TextModel.HiddenSize == 512 && model.Dimensions == 768, "Require exclusive native audio model with only explicitly requested prewarm vision.");
    var prepare = typeof(EmbeddingGemma2MultimodalEncoder).GetMethod("Prepare", BindingFlags.NonPublic | BindingFlags.Instance, [typeof(EmbeddingGemma2InputPart[]), typeof(CancellationToken)]) ?? throw new MissingMethodException("Frozen Prepare signature unavailable.");
    var handleField = typeof(EmbeddingModel).GetField("_handle", BindingFlags.NonPublic | BindingFlags.Instance) ?? throw new MissingFieldException("Native handle unavailable.");
    Require(handleField.FieldType == typeof(IntPtr), "Unexpected native handle type.");
    IntPtr handle = (IntPtr)handleField.GetValue(model.TextModel)!; Require(handle != IntPtr.Zero, "Native handle is empty.");
    var modelGate = (SemaphoreSlim)typeof(EmbeddingGemma2MultimodalEncoder).GetField("_gate", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(model)!;
    var textGate = (SemaphoreSlim)typeof(EmbeddingModel).GetField("_gate", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(model.TextModel)!;
    var parts = new[] { EmbeddingGemma2InputPart.Audio(wav) };
    float[] Prepare()
    {
        try { return (float[])prepare.Invoke(model, [parts, CancellationToken.None])!; }
        catch (TargetInvocationException error) when (error.InnerException != null) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
    }
    float[] Encode(float[] inputs)
    {
        int tokens = inputs.Length / 512;
        Require(tokens == oracleTokens && inputs.Length == tokens * 512 && inputs.All(float.IsFinite), "Prepared complete input rows invalid.");
        var vector = new float[768];
        textGate.Wait();
        try { GgmlEmbeddingNative.EncodeInputs(handle, inputs, [tokens], vector); }
        finally { textGate.Release(); GC.KeepAlive(model); }
        return vector;
    }
    Measurement Run(bool split)
    {
        modelGate.Wait();
        try
        {
            long start = Stopwatch.GetTimestamp();
            if (!split)
            {
                var completed = media.Invoke(() =>
                {
                    long p0 = Stopwatch.GetTimestamp(); var inputs = Prepare(); double prep = Stopwatch.GetElapsedTime(p0).TotalMilliseconds;
                    long t0 = Stopwatch.GetTimestamp(); var vector = Encode(inputs); double enc = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                    return (Input: inputs, Vector: vector, PrepareMs: prep, TextMs: enc);
                });
                return new Measurement(completed.Input, completed.Vector, Stopwatch.GetElapsedTime(start).TotalMilliseconds, completed.PrepareMs, completed.TextMs);
            }
            var prepared = media.Invoke(() => { long p0 = Stopwatch.GetTimestamp(); var input = Prepare(); return (Input: input, Ms: Stopwatch.GetElapsedTime(p0).TotalMilliseconds); });
            var encoded = text!.Invoke(() => { long t0 = Stopwatch.GetTimestamp(); var vector = Encode(prepared.Input); return (Vector: vector, Ms: Stopwatch.GetElapsedTime(t0).TotalMilliseconds); });
            return new Measurement(prepared.Input, encoded.Vector, Stopwatch.GetElapsedTime(start).TotalMilliseconds, prepared.Ms, encoded.Ms);
        }
        finally { modelGate.Release(); }
    }
    Measurement RunWhole(bool threadPool)
    {
        modelGate.Wait();
        try
        {
            // This single delegate is identical across both dispatch routes.
            Measurement Callback()
            {
                var caller = CallerIdentity.Capture();
                Require(caller.IsThreadPoolThread == threadPool, "Actual caller dispatch kind differs.");
                if (!threadPool) Require(caller.ManagedThreadId == media.ManagedThreadId && caller.NativeThreadId == media.NativeThreadId, "Resident caller identity changed.");
                long p0 = Stopwatch.GetTimestamp(); var inputs = Prepare(); double prep = Stopwatch.GetElapsedTime(p0).TotalMilliseconds;
                long c0 = Stopwatch.GetTimestamp(); var owned = (float[])inputs.Clone(); double clone = Stopwatch.GetElapsedTime(c0).TotalMilliseconds;
                long t0 = Stopwatch.GetTimestamp(); var vector = Encode(owned); double enc = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                Require(CallerIdentity.Capture() == caller, "Synchronous callback changed OS caller identity.");
                return new Measurement(owned, vector, 0, prep, enc) { Caller = caller, CloneMs = clone };
            }
            long start = Stopwatch.GetTimestamp();
            var completed = SchedulingDispatch.Run(threadPool, media, Callback);
            return completed with { TotalMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds };
        }
        finally { modelGate.Release(); }
    }
    Measurement RunPublicResident(bool publicApi)
    {
        // Start before any route gates or dispatch. The public API acquires its
        // own gates; holding the MM gate across it would deadlock. The resident
        // route acquires the same gates in RunWhole, without concurrent calls.
        long start = Stopwatch.GetTimestamp();
        if (!publicApi)
        {
            var completed = RunWhole(false);
            return completed with { TotalMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds,
                PromptTokens = completed.Inputs!.Length / 512 };
        }
        var result = model.EmbedAsync(parts, CancellationToken.None).GetAwaiter().GetResult();
        double elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        Require(result.PromptTokens == oracleTokens && result.Embeddings.Length == 1, "Public API prompt tokens or batch count differs.");
        return new Measurement(null, result.Embeddings[0], elapsed, 0, 0)
        { PublicApi = true, PromptTokens = result.PromptTokens };
    }
    Measurement RunPhase(bool route) => publicResidentMode ? RunPublicResident(route) : wholeCallbackMode ? RunWhole(route) : Run(route);
    var baseline = RunPhase(false);
    Quality Check(Measurement actual)
    {
        Require(actual.Vector.Length == 768 && actual.Vector.All(float.IsFinite), "Nonfinite complete pooled vector.");
        double norm = Math.Sqrt(actual.Vector.Sum(x => (double)x * x)), expectedNorm = Math.Sqrt(expected.Sum(x => x * x));
        Require(norm > 0 && Math.Abs(norm - 1) <= 1e-5, "Complete pooled output is not normalized.");
        double cosine = expected.Zip(actual.Vector, (x, y) => x * y).Sum() / (norm * expectedNorm);
        double error = expected.Zip(actual.Vector, (x, y) => Math.Abs(x - y)).Max();
        bool? inputBits = actual.PublicApi ? null : actual.Inputs != null &&
            MemoryMarshal.AsBytes(actual.Inputs.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(baseline.Inputs!.AsSpan()));
        bool outputBits = MemoryMarshal.AsBytes(actual.Vector.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(baseline.Vector.AsSpan()));
        Require(!actual.PublicApi || publicResidentMode && actual.Inputs == null && actual.Caller == null && actual.PromptTokens == 29,
            "Public result observations or prompt tokens invalid.");
        return new Quality(cosine >= .9999 && error <= .002 && (actual.PublicApi || inputBits == true) && outputBits, cosine, error, inputBits, outputBits);
    }
    passed &= Check(baseline).Passed;
    report["baseline_embedding"] = baseline.Vector; report["baseline_input_rows_sha256"] = FloatHash(baseline.Inputs!); report["baseline_embedding_sha256"] = FloatHash(baseline.Vector);
    if (publicResidentMode) report["baseline_route"] = "resident internal synchronous callback; complete prepared rows observed";
    var phases = new List<object>(); report["phases"] = phases;
    var prewarms = new List<object>();
    if (prewarmImage != null)
    {
        report["prewarm_image"] = new { path = prewarmImage, sha256 = before[prewarmImage], route = "same restored public EmbedAsync before every phase; outside audio timing", calls_per_phase = 1 };
        report["prewarm_calls"] = prewarms;
    }
    var phaseDefinitions = SchedulingModes.Phases(mode);
    foreach (var phase in phaseDefinitions)
    {
        if (prewarmImage != null)
        {
            // Common history: use the exact restored public async API, not a
            // route-specific resident prewarm. Never hold the MM gate here.
            long start = Stopwatch.GetTimestamp();
            var warmed = model.EmbedAsync([EmbeddingGemma2InputPart.Image(prewarmImage)], CancellationToken.None).GetAwaiter().GetResult();
            Require(warmed.Embeddings.Length == 1 && warmed.Embeddings[0].Length == 768 && warmed.Embeddings[0].All(float.IsFinite), "Invalid image prewarm output.");
            Require(Math.Abs(Math.Sqrt(warmed.Embeddings[0].Sum(x => (double)x * x)) - 1) <= 1e-5, "Image prewarm is not normalized.");
            prewarms.Add(new { phase = phase.Item1, elapsed_ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds,
                prompt_tokens = warmed.PromptTokens, embedding_sha256 = FloatHash(warmed.Embeddings[0]), finite_and_normalized = true,
                independent_image_quality_qualified = false });
        }
        var gates = new List<Quality>(); var samples = new List<double>(); var prepSamples = new List<double>(); var textSamples = new List<double>();
        var cloneSamples = new List<double>(); var callers = new List<object>(); var identities = new List<CallerIdentity>();
        var promptTokens = new List<int>();
        Measurement last = baseline;
        for (int i = 0; i < 3; ++i)
        {
            last = RunPhase(phase.Item2); gates.Add(Check(last));
            if (publicResidentMode) promptTokens.Add(last.PromptTokens!.Value);
            if (wholeCallbackMode || publicResidentMode && !last.PublicApi) { callers.Add(new { kind = "warmup", index = i, identity = last.Caller }); identities.Add(last.Caller!); }
        }
        do
        {
            last = RunPhase(phase.Item2); samples.Add(last.TotalMs); gates.Add(Check(last));
            if (!last.PublicApi) { prepSamples.Add(last.PrepareMs); textSamples.Add(last.TextMs); }
            if (publicResidentMode) promptTokens.Add(last.PromptTokens!.Value);
            if (wholeCallbackMode || publicResidentMode && !last.PublicApi) { cloneSamples.Add(last.CloneMs); callers.Add(new { kind = "timed", index = samples.Count - 1, identity = last.Caller }); identities.Add(last.Caller!); }
        }
        while (samples.Count < 20 || samples.Sum() < 1000);
        bool gate = gates.All(x => x.Passed); passed &= gate;
        if (publicResidentMode)
        {
            bool publicApi = phase.Item2;
            var evidence = PublicResidentReport.Observations(publicApi, prepSamples, textSamples, cloneSamples, callers, identities,
                publicApi ? null : gates.All(x => x.InputBits == true));
            evidence["name"] = phase.Item1; evidence["public_api"] = publicApi;
            evidence["samples_ms"] = samples; evidence["median_ms"] = Median(samples); evidence["actual_repeats"] = samples.Count;
            evidence["checked_complete_vectors"] = gates.Count; evidence["quality_passed"] = gate;
            evidence["min_cosine"] = gates.Min(x => x.Cosine); evidence["max_absolute_error"] = gates.Max(x => x.MaxError);
            evidence["prompt_tokens"] = promptTokens.Distinct().Single(); evidence["prompt_tokens_per_checked_call"] = promptTokens;
            evidence["pooled_bit_identity"] = gates.All(x => x.OutputBits);
            evidence["embedding"] = last.Vector; evidence["embedding_sha256"] = FloatHash(last.Vector);
            phases.Add(evidence);
        }
        else if (wholeCallbackMode) phases.Add(new { name = phase.Item1, whole_callback_threadpool = phase.Item2,
            samples_ms = samples, prepare_samples_ms = prepSamples, text_samples_ms = textSamples, clone_samples_ms = cloneSamples, caller_observations = callers,
            distinct_managed_callers = identities.Select(x => x.ManagedThreadId).Distinct().Count(), distinct_native_callers = identities.Select(x => x.NativeThreadId).Distinct().Count(),
            median_ms = Median(samples), actual_repeats = samples.Count, checked_complete_vectors = gates.Count, quality_passed = gate,
            min_cosine = gates.Min(x => x.Cosine), max_absolute_error = gates.Max(x => x.MaxError), prepared_bit_identity = gates.All(x => x.InputBits == true),
            pooled_bit_identity = gates.All(x => x.OutputBits), embedding = last.Vector, embedding_sha256 = FloatHash(last.Vector) });
        else phases.Add(new { name = phase.Item1, split_master_threads = phase.Item2, samples_ms = samples, prepare_samples_ms = prepSamples, text_samples_ms = textSamples,
            median_ms = Median(samples), actual_repeats = samples.Count, checked_complete_vectors = gates.Count, quality_passed = gate,
            min_cosine = gates.Min(x => x.Cosine), max_absolute_error = gates.Max(x => x.MaxError), prepared_bit_identity = gates.All(x => x.InputBits == true),
            pooled_bit_identity = gates.All(x => x.OutputBits), embedding = last.Vector, embedding_sha256 = FloatHash(last.Vector) });
    }
    media.Invoke(() => true);
    if (wholeCallbackMode || publicResidentMode)
    {
        Require(media.NativeThreadId > 0, "Resident native master identity invalid.");
        report["workers"] = new[] { new { role = "resident", media.ManagedThreadId, media.NativeThreadId } };
    }
    else
    {
        text!.Invoke(() => true);
        Require(media.ManagedThreadId != text.ManagedThreadId && media.NativeThreadId > 0 && text.NativeThreadId > 0 && media.NativeThreadId != text.NativeThreadId, "Persistent native master identities invalid.");
        report["workers"] = new[] { new { role = "media/same", media.ManagedThreadId, media.NativeThreadId }, new { role = "split-text", text.ManagedThreadId, text.NativeThreadId } };
    }
    using var process = Process.GetCurrentProcess();
    var modules = process.Modules.Cast<ProcessModule>().Where(x => Path.GetFileName(x.FileName).StartsWith("libGgmlOps.so", StringComparison.Ordinal)).ToArray();
    Require(modules.Length == 1 && Hash(modules[0].FileName) == options["--expected-native-sha256"], "Actual loaded native identity differs.");
    report["loaded_native"] = new { path = modules[0].FileName, sha256 = Hash(modules[0].FileName) };
    report["completed"] = true; report["passed"] = passed;
}
catch (Exception error) { report["error"] = error.ToString(); report["completed"] = false; report["passed"] = false; }
finally
{
    try { if (model != null) { media!.Invoke(() => { model.Dispose(); return true; }); report["model_disposed_on_owner_thread"] = true; } }
    catch (Exception error) { report["dispose_error"] = error.ToString(); report["completed"] = false; report["passed"] = false; }
    try { text?.Dispose(); media?.Dispose(); report["workers_joined_no_active_callbacks"] = (text?.Joined ?? true) && (media?.Joined ?? true); }
    catch (Exception error) { report["worker_dispose_error"] = error.ToString(); report["completed"] = false; report["passed"] = false; }
    var after = before.ToDictionary(x => x.Key, x => File.Exists(x.Key) ? Hash(x.Key) : null);
    bool unchanged = before.All(x => after[x.Key] == x.Value);
    report["after_hashes"] = after; report["input_binary_hashes_unchanged"] = unchanged;
    if (!unchanged) { report["completed"] = false; report["passed"] = false; }
    report["limitations"] = new[] { "Warm synthetic one-waveform scheduling screen, not HTTP or production Task.Run latency.", "Persistent same/split masters can expose per-caller OpenMP team behavior; they do not establish a dedicated owner is faster than one whole-MM ThreadPool callback.", "No global OpenMP/environment policy is changed. Existing nonsecret selectors and thread policy are recorded.", "Direct native calls omit production async clone/scheduling overhead equally in both routes; the exclusively owned model remains alive and gated.", "Independent pooled thresholds remain .9999/.002; complete prepared/output bit identity to the initial same-thread route is additional, not a relaxed quality substitute." };
    if (wholeCallbackMode) report["limitations"] = new[] { "One-waveform whole-callback scheduling screen, not HTTP or production scheduling qualification.",
        "Identical synchronous preparation/input clone/finite validation/native encode callbacks are dispatched through Task.Run or one resident caller; public async gate/continuation overhead is not measured.",
        "Natural managed/native caller identities are recorded; no ThreadPool churn is forced and no pool/OpenMP policy is changed. Multiple caller identities do not establish actual OpenMP team creation or worker reuse.",
        "Optional public image prewarm is identical before each phase and excluded from audio timing; finite/normalized image output is not independent image quality qualification.",
        "Independent pooled .9999/.002 and complete input/output bit identity remain mandatory for initial/warm/every timed vector; all callbacks complete before model disposal and owner join." };
    if (publicResidentMode) report["limitations"] = new[] { "One-waveform restored public API versus internal resident callback screen; no HTTP or production scheduling qualification.",
        "Caller timing includes each route's complete gate/dispatch/API work. The public API and synchronous resident path are different delegates with real validation, cloning, gate and continuation cost differences; no contention is introduced.",
        "Public native caller identities, prepared input bytes and preparation/clone/text timing breakdown are unobserved and null. Only resident internals are observed; no ThreadPool churn or OpenMP policy change is forced.",
        "Every public call must report29 prompt tokens and one complete finite normalized768-vector, pass the independent pooled .9999/.002 gates and retain bit identity to the resident baseline. Resident calls additionally retain prepared-input bit identity.",
        "Optional common restored public image prewarm occurs before every phase outside audio timing. Its finite/normalized output does not qualify independent image quality. All calls complete before owner-thread disposal and join." };
    File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
}
Console.WriteLine(JsonSerializer.Serialize(new { completed = report["completed"], passed = report["passed"], model_disposed = report.GetValueOrDefault("model_disposed_on_owner_thread"), workers_joined = report["workers_joined_no_active_callbacks"] }));
return report["completed"] is true && report["passed"] is true ? 0 : 1;

static double Median(List<double> input) { var sorted = input.Order().ToArray(); return (sorted[(sorted.Length - 1) / 2] + sorted[sorted.Length / 2]) / 2; }
sealed record Measurement(float[]? Inputs, float[] Vector, double TotalMs, double PrepareMs, double TextMs)
{
    public CallerIdentity? Caller { get; init; }
    public double CloneMs { get; init; }
    public bool PublicApi { get; init; }
    public int? PromptTokens { get; init; }
}
sealed record Quality(bool Passed, double Cosine, double MaxError, bool? InputBits, bool OutputBits);
