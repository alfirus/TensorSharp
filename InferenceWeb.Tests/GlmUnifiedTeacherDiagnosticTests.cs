// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Buffers.Binary;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
public sealed class GlmUnifiedTeacherDiagnosticFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda,Models";
    public GlmUnifiedTeacherDiagnosticFactAttribute() => Skip =
        string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_GLM_UNIFIED_TEACHER_MODEL"))
            ? "Requires explicit TS_TEST_GLM_UNIFIED_TEACHER_MODEL; each profile runs in a fresh process."
            : TestGates.CudaSkip;
}
[CollectionDefinition("GLM unified teacher diagnostic", DisableParallelization = true)]
public sealed class GlmUnifiedTeacherDiagnosticCollection { }

/// <summary>Four bounded teacher continuations through the production scheduler.
/// The full-logit metrics are diagnostic, not a quality acceptance threshold.
/// Native stderr must separately certify every actual chunk and source call.</summary>
[Collection("GLM unified teacher diagnostic")]
public sealed class GlmUnifiedTeacherDiagnosticTests
{
    private const int PromptLength = 1872, MaximumTeachers = 32;
    private const long GiB = 1L << 30;
    private const string Scope = "bounded_teacher_numerical_diagnostic_not_general_quality_or_timing";
    private const string PaddingSentence = "The history of computing includes counting tools, algorithms, logic and modern electronic computers. ";
    private const BindingFlags PrivateInstance = BindingFlags.NonPublic | BindingFlags.Instance;
    private static ActualOwner? s_retainedCleanupOwner;
    private static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, WriteIndented = true };
    // Prompts are the existing BSD-authored DirectCudaModelProbe quality cases.
    // Answers are fixed before either numerical result, not sampled references.
    private static readonly (string Name, string Prompt, string Teacher)[] Cases = [
        ("math", "Compute 17 + 25. Reply with only the integer.", "42"),
        ("extraction", "Inventory: cedar=3, maple=7, pine=2. Return only the names whose quantity is at least 3, comma-separated in the original order.", "cedar,maple"),
        ("squares", "Return the squares of the integers 1 through 20, in order, as comma-separated integers. Return only the list.", "1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,400"),
        ("code", "Write a Python function square(x) that returns x*x. Return only the function, without Markdown.", "def square(x):\n    return x*x")
    ];

    [GlmUnifiedTeacherDiagnosticFact]
    public void FixedTeachersUseUnchangedFullLogitsAndActualSchedulerHeads()
    {
        string modelPath = Path.GetFullPath(Required("MODEL"));
        string profile = Required("PROFILE");
        Assert.Contains(profile, new[] { "pp512", "adaptive1024" });
        string output = Path.GetFullPath(Required("OUTPUT"));
        string inputsPath = Path.GetFullPath(Required("INPUTS"));
        Assert.True(File.Exists(modelPath));
        Assert.False(Directory.Exists(output), "Fresh per-profile destination required.");
        Directory.CreateDirectory(output);
        Assert.Null(s_retainedCleanupOwner);
        using var speculation = new EnvScope(); speculation.ClearSpeculationVars();
        using var env = ConfigureEnvironment(profile);
        ActualOwner? owner = null; Exception? original = null;
        var results = new List<CaseResult>();
        InputSet? inputs = null; string? inputFileHash = null;
        string? mappedNative = null;
        string modelsMvid = typeof(GlmDsaModel).Assembly.ManifestModule.ModuleVersionId.ToString("D");
        string testsMvid = typeof(GlmUnifiedTeacherDiagnosticTests).Assembly.ManifestModule.ModuleVersionId.ToString("D");
        string? featureReport = null; int resolvedPrefill = 0, resolvedForward = 0;
        try {
            GgmlGlmNative.RequireCheckedFreeSupport();
            owner = new ActualOwner(GlmUnifiedMemoryModelSession.Create(modelPath, null, BackendType.GgmlCuda, Options(profile)));
            owner.NativeExecutor.RequireCheckedDisposal();
            mappedNative = TestGates.MappedNativeGgmlOpsPath();
            var model = owner.Model;
            Assert.False(model.UnifiedHostTokenEmbeddingCertified);
            Assert.False(model.HasDraftHead); Assert.Equal(3, model.UnifiedExpertGroups);
            Assert.Equal(2048, owner.Session.ContextTokens); Assert.Equal(2048, model.MaxContextLength);
            // All special modes stay OFF. This production session uses its
            // ordinary budget callback attachment, not checked ACK recovery.
            Assert.False(owner.Native.IncludesCheckedRelease); Assert.Null(owner.Native.CallbackError);
            resolvedPrefill = owner.Session.PrefillChunkTokens;
            resolvedForward = ReadPrivate<int>(owner.Session, "_forwardTokens");
            Assert.InRange(resolvedPrefill, 1, profile == "pp512" ? 512 : 1024);
            Assert.True(resolvedForward >= PromptLength, "The production admission must permit one layer-major prompt submission.");
            featureReport = owner.Session.FeatureReport;
            var generated = BuildInputs(model, owner.Session, modelPath);
            if (File.Exists(inputsPath)) {
                inputs = JsonSerializer.Deserialize<InputSet>(File.ReadAllText(inputsPath), Json)!;
                Assert.NotNull(inputs);
                Assert.Equal(JsonSerializer.Serialize(generated, Json), JsonSerializer.Serialize(inputs, Json));
            } else {
                Assert.Equal("pp512", profile); // Freeze teachers before the first arm's first Forward.
                WriteNew(inputsPath, generated); inputs = generated;
            }
            inputFileHash = FileHash(inputsPath);
            foreach (var input in inputs.Cases) {
                Marker("CASE-BEGIN", $"case={input.Name} profile={profile} prompt_sha={input.PromptSha256} teacher_sha={input.TeacherSha256} inputs_sha={inputFileHash}");
                owner.Prepare(input.PromptIds);
                var submissions = new List<int>();
                Marker("PP-BEGIN", $"case={input.Name} rows={PromptLength}");
                lock (model.GpuComputeLock) {
                    var schedule = owner.Scheduler!.Schedule(); var work = Assert.Single(schedule.ScheduledWork);
                    Assert.Same(owner.Sequence, work.Sequence); Assert.True(work.IsPrefill);
                    Assert.Equal(PromptLength, work.NumScheduledTokens); Assert.NotNull(owner.Sequence!.MemoryEnvelope);
                    var step = Assert.Single(owner.Executor!.ExecuteStep(schedule));
                    Assert.Null(step.Error); Assert.False(step.SlotUnavailable); Assert.True(step.IsPrefill);
                    Assert.Equal(PromptLength, step.TokensForwarded); submissions.Add(step.TokensForwarded);
                    Assert.Equal(PromptLength, owner.Sequence.NumComputedTokens);
                    Assert.Equal(PromptLength, GgmlGlmNative.NPast(owner.Handle));
                    Assert.Empty(owner.Sequence.OutputTokens);
                    Assert.NotNull(owner.Sequence.LastLogits); Assert.Equal(154880, owner.Sequence.LastLogits.Length);
                    Assert.NotNull(owner.Scheduler.Config.MemoryAdmission!.AfterPrefill);
                    owner.Scheduler.Config.MemoryAdmission.AfterPrefill(owner.Sequence);
                    owner.Scheduler.NotifyStepCompleted(schedule);
                }
                Marker("PP-END", $"case={input.Name} head={PromptLength}");
                var rows = new List<TeacherRow>();
                for (int index = 0; index < input.TeacherIds.Length; ++index) {
                    int target = input.TeacherIds[index], position = PromptLength + index;
                    lock (model.GpuComputeLock) {
                        var seq = owner.Sequence!;
                        Assert.Equal(position, seq.NumComputedTokens); Assert.Equal(position, GgmlGlmNative.NPast(owner.Handle));
                        Assert.Equal(index, seq.OutputTokens.Count); Assert.Null(seq.PendingDeviceToken);
                        Assert.Equal(input.TeacherIds.Take(index), seq.OutputTokens);
                        float[] logits = seq.LastLogits!; Assert.Equal(154880, logits.Length);
                        string history = TokenHash(input.PromptIds.Concat(input.TeacherIds.Take(index)));
                        var score = GlmTeacherForcingNumerics.Score(logits, target);
                        Assert.Equal(score.LogitsSha256, GlmTeacherForcingNumerics.Score(logits, target).LogitsSha256);
                        // This stash is consumed by the unmodified executor. It appends
                        // exactly one input; the fixture never appends teacher IDs itself.
                        seq.PendingDeviceToken = target; seq.PendingDevicePosition = position;
                        Marker("N1-BEGIN", $"case={input.Name} index={index} p0={position} target={target} history_sha={history} logits_sha={score.LogitsSha256}");
                        var schedule = owner.Scheduler!.Schedule(); var work = Assert.Single(schedule.ScheduledWork);
                        Assert.Same(seq, work.Sequence); Assert.False(work.IsPrefill); Assert.Equal(1, work.NumScheduledTokens);
                        Assert.NotNull(seq.MemoryEnvelope);
                        var step = Assert.Single(owner.Executor!.ExecuteStep(schedule));
                        Assert.Null(step.Error); Assert.False(step.SlotUnavailable); Assert.False(step.IsPrefill);
                        Assert.Equal(1, step.TokensForwarded); Assert.Equal(target, step.SampledToken);
                        Assert.Null(seq.PendingDeviceToken); Assert.Equal(index + 1, seq.OutputTokens.Count);
                        Assert.Equal(input.TeacherIds.Take(index + 1), seq.OutputTokens);
                        Assert.Equal(position + 1, seq.NumComputedTokens); Assert.Equal(position + 1, GgmlGlmNative.NPast(owner.Handle));
                        // Fused execution owns a separate next-logit array; scoring did not
                        // modify the borrowed previous distribution.
                        Assert.Equal(score.LogitsSha256, GlmTeacherForcingNumerics.Score(logits, target).LogitsSha256);
                        owner.Scheduler.NotifyStepCompleted(schedule);
                        rows.Add(new(index, position, target, history, TokenHash(input.PromptIds.Concat(seq.OutputTokens)), score, step.SampledToken, step.TokensForwarded, seq.NumComputedTokens));
                        Marker("N1-END", $"case={input.Name} index={index} p0={position} target={target} head={position + 1}");
                    }
                }
                owner.Release(recover: true);
                Assert.Null(owner.Sequence!.MemoryEnvelope); Assert.Equal(0, owner.Scheduler!.RunningCount);
                Assert.Equal(owner.Pool!.NumBlocks, owner.Pool.NumFreeBlocks); Assert.Equal(0, owner.Pool.Storage.AllocatedBytes);
                results.Add(new(input.Name, submissions.ToArray(), rows.ToArray(), rows.Sum(x => x.Score.Nll), rows.Average(x => x.Score.Nll)));
                Marker("CASE-END", $"case={input.Name} profile={profile} tokens={rows.Count} outcome=completed");
            }
            Assert.Equal(inputFileHash, FileHash(inputsPath));
            var executor = owner.NativeExecutor; var native = owner.Native;
            owner.Dispose();
            Assert.Equal(IntPtr.Zero, ReadPrivate<IntPtr>(executor, "_handle")); Assert.Null(owner.Session.Model);
            Assert.Null(native.CallbackError); Assert.Equal(0, native.ActiveAllocations); Assert.Empty(native.AllocationUsage);
            Assert.All(owner.Session.Budget.Snapshot(), p => { Assert.Equal(0, p.Reserved); Assert.Equal(0, p.Committed); });
            owner.Session.Dispose();
            var report = new {
                schema = 1, status = "numerical_diagnostic_unqualified_until_log_validation", scope = Scope,
                profile, model_path = modelPath, mapped_native_path = mappedNative, models_mvid = modelsMvid, tests_mvid = testsMvid,
                inputs_path = inputsPath, inputs_sha256 = inputFileHash,
                requested_context = 2048, requested_prefill = 512, requested_adaptive_max = profile == "pp512" ? 512 : 1024,
                scheduler_chunk_env = profile == "pp512" ? "512" : null,
                resolved_context = owner.Session.ContextTokens, resolved_prefill = resolvedPrefill, resolved_forward_tokens = resolvedForward,
                host_embedding = 0, max_running = 1, feature_report = featureReport,
                environment = EnvironmentValues(profile), cases = results,
                teacher_tokens = results.Sum(x => x.Rows.Length), total_nll = results.Sum(x => x.TotalNll),
                mean_nll = results.Sum(x => x.TotalNll) / results.Sum(x => x.Rows.Length),
                checked_model_free_and_ordinary_scope_drained = true, checked_cache_release_attachment = false,
                native_handle_zero = true, active_allocations = 0, ledger_reserved = 0, ledger_committed = 0,
                general_quality_qualified = false, strict_canonical_api_qualified = false, performance_qualified = false,
                source_file_offsets_independently_captured = false, binary_identity_requires_external_launcher_pins = true
            };
            WriteNew(Path.Combine(output, "report.json"), report);
            Marker("DONE", $"profile={profile} cases=4 checked_model_free=1 scope={Scope}");
        } catch (UnifiedMemoryModelLoadException error) {
            owner = error.UnreleasedSession == null ? null : new ActualOwner(error.UnreleasedSession); original = error; throw;
        } catch (Exception error) { original = error; throw; }
        finally {
            if (owner != null) try { owner.Dispose(); }
            catch (Exception cleanup) { s_retainedCleanupOwner = owner; throw original == null
                ? new AggregateException("Checked diagnostic cleanup failed; full owner retained.", cleanup)
                : new AggregateException("Diagnostic and checked cleanup failed; full owner retained.", original, cleanup); }
        }
    }

    private static InputSet BuildInputs(GlmDsaModel model, IUnifiedMemoryModelSession session, string modelPath)
    {
        var geometry = ReadPrivate<GlmUnifiedMemoryGeometry>(session, "_geometry");
        Assert.Equal(45, geometry.Layers); Assert.Equal(4096, geometry.Hidden); Assert.Equal(288, geometry.ExpertCount);
        Assert.Equal(8, geometry.UsedExperts); Assert.Equal(154880, geometry.Vocab);
        var catalog = geometry.ExpertLayerLayouts.Select((layout, layer) => (layout, layer)).Where(x => x.layout != null)
            .Select(x => new SourceLayout(x.layer, (int)x.layout!.Gate.Type, (int)x.layout.Up.Type, (int)x.layout.Down.Type,
                x.layout.Gate.BytesPerExpert, x.layout.Up.BytesPerExpert, x.layout.Down.BytesPerExpert)).ToArray();
        Assert.Equal(Enumerable.Range(3, 42), catalog.Select(x => x.Layer));
        string template = model.Config.ChatTemplate; Assert.False(string.IsNullOrEmpty(template));
        var cases = Cases.Select(test => BuildCase(model, template, test.Name, test.Prompt, test.Teacher)).ToArray();
        var file = new FileInfo(modelPath);
        return new(1, modelPath, file.Length, file.LastWriteTimeUtc.Ticks, model.Config.Architecture,
            model.Tokenizer.GetType().FullName!, TextHash(template), TextHash(JsonSerializer.Serialize(catalog,
                new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower })), catalog,
            "eng/validation/DirectCudaModelProbe/QualityOutput.cs; BSD-3-Clause; math/extraction/squares/code",
            "eng/validation/DirectCudaModelProbe/Program.cs; existing history-of-computing padding sentence",
            "fixed-answers-before-first-forward; teacher suffix is full rendered-plus-teacher encoding minus exact prompt prefix; first32 suffix tokens",
            cases);
    }
    private static TeacherCase BuildCase(GlmDsaModel model, string template, string name, string prompt, string teacher)
    {
        string filler = string.Concat(Enumerable.Repeat(PaddingSentence, 256));
        (string Text, int[] Ids) Candidate(int length) {
            string user = "Background context (the final instruction is the task):\n" + filler[..length] + "\n\n" + prompt;
            string rendered = new GgufPromptRenderer().Render(template,
                [new() { Role = "system", Content = "You are a helpful assistant." }, new() { Role = "user", Content = user }],
                addGenerationPrompt: true, architecture: model.Config.Architecture, tools: null, enableThinking: false, reasoningEffort: null);
            return (rendered, model.Tokenizer.Encode(rendered, addSpecial: true).ToArray());
        }
        int lo = 0, hi = filler.Length;
        while (lo < hi) { int mid = lo + (hi - lo) / 2; if (Candidate(mid).Ids.Length < PromptLength) lo = mid + 1; else hi = mid; }
        // Token counts need not be strictly monotonic at BPE boundaries. Search
        // a fixed local character neighborhood and refuse if no exact length.
        (string Text, int[] Ids)? exact = null;
        for (int length = Math.Max(0, lo - 512); length <= Math.Min(filler.Length, lo + 512); ++length) {
            var candidate = Candidate(length); if (candidate.Ids.Length == PromptLength) { exact = candidate; break; }
        }
        Assert.True(exact.HasValue, "No exact1872 naturally rendered prompt in the fixed bounded padding search.");
        var selected = exact!.Value;
        int[] combined = model.Tokenizer.Encode(selected.Text + teacher, addSpecial: true).ToArray();
        Assert.True(combined.Length > PromptLength); Assert.Equal(selected.Ids, combined.Take(PromptLength));
        int[] allTeacher = combined.Skip(PromptLength).ToArray(), ids = allTeacher.Take(MaximumTeachers).ToArray();
        Assert.InRange(ids.Length, 1, MaximumTeachers);
        Assert.All(combined, token => Assert.InRange(token, 0, 154879));
        Assert.DoesNotContain(ids, token => model.Tokenizer.IsEos(token));
        return new(name, prompt, teacher, selected.Text, selected.Ids, ids, allTeacher.Length, ids.Length < allTeacher.Length,
            TextHash(selected.Text), TokenHash(selected.Ids), TokenHash(ids), TokenHash(selected.Ids.Concat(ids)));
    }
    private static UnifiedMemoryModelOptions Options(string profile) => new() {
        ContextTokens = 0, PrefillChunkTokens = 512, AdaptiveGlmPrefillChunkTokens = profile == "pp512" ? 512 : 1024,
        LayerMajorPrefillTokens = 8192, MaximumRunningRequests = 1, RetainedConversationCount = 0, PrefixCheckpointCount = 0,
        MaximumHostBytes = 28 * GiB, MaximumDeviceBytes = 14 * GiB, MaximumHostExpertCacheBytes = 0,
        EnableGlmSelectedExperts = true, EnableAdaptiveGlmPrefill = true, EnableExpertCacheCapacityGrowth = true,
        EnableAsyncExpertPrefill = false, EnableGlmFlashMtp = false
    };
    private static Dictionary<string, string?> EnvironmentValues(string profile)
    {
        var values = new Dictionary<string, string?>();
        foreach (string name in new[] { "TS_GLM_TRACE", "TS_GLM_TRACE_DUMP_DIR", "TS_GLM_NODES_PER_LAYER", "TS_GLM_MOE_MMAP", "TS_GLM_TP_SHARD", "TS_GLM_TP_OVERSUBSCRIBE" }) values[name] = null;
        foreach (string name in new[] { "TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", "TS_GLM_FLASH_MTP", "TS_GLM_FLASH_MTP_SHAPE_PLAN", "TS_GLM_FLASH_MTP_PREFIX_COMMIT", "TS_GLM_DECODE_GRAPH_RESERVE", "TS_GLM_OP_OFFLOAD", "TS_GGML_STRICT_RECURRENT_PROJECTIONS", "TS_GGML_Q8_ORDERED_SMALL_BATCH", "TS_GGML_Q8_ORDERED_GATE_PREFILL", "TS_GGML_Q8_GATE_PREFILL_TILE16", "TS_GLM_SELECTED_EXPERT_TIMING", "TS_GLM_GRAPH_MEMORY_DIAGNOSTICS", "TS_DSV4_PERF", "TS_HOST_MOE_RAM_CACHE_MB", "TS_HOST_MOE_RAM_CACHE_MAPPED", "TS_HOST_MOE_RAM_CACHE_PINNED", "TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD", "TS_HOST_MOE_RAM_CACHE_VICTIM", "TS_HOST_MOE_RAM_CACHE_FREQUENCY", "TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY", "TS_HOST_MOE_LAYER_PREFETCH", "TS_HOST_MOE_DECODE_PREFETCH", "TS_HOST_MOE_PREFILL_RING_COALESCE16", "TS_HOST_MOE_PREFILL_DOWN_OVERLAP", "TS_HOST_MOE_PHASE_COLD_GROWTH", "TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER", "TS_HOST_MOE_EXPERT_CACHE_PREFETCH", "TS_HOST_MOE_EXPERT_CACHE_LFU", "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS", "TS_HOST_MOE_FILE_READ_WORKERS", "TS_HOST_MOE_FILE_RANDOM_ACCESS", "TS_HOST_MOE_FILE_READ_SORT", "TS_HOST_MOE_ROUTE_TRACE", "TS_HOST_MOE_FILE_SOURCE_VIEW", "TS_HOST_MOE_FILE_SOURCE_VIEW_PREFETCH", "TS_HOST_MOE_VMM_WEIGHTS", "TS_GLM_HOST_TOKEN_EMBEDDING", "TS_GLM_LAYER_MAJOR_DEVICE_BATON", "TS_HOST_MOE_CAUSAL_ROUTE_NEIGHBOR" }) values[name] = "0";
        foreach (string name in new[] { "TS_GLM_NATIVE", "TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS", "TS_HOST_MOE_EXPERT_CACHE_TIMING", "TS_GLM_FLASH_MTP_FORWARD_TIMING", "TS_HOST_MOE_DECODE_REBALANCE", "TS_HOST_MOE_DECODE_STAGING_72MB", "TS_HOST_MOE_PREFILL_RING_72MB" }) values[name] = "1";
        values["TS_HOST_MOE_EXPERT_CACHE_RING_READS"] = "2"; values["MAX_CONTEXT"] = "2048";
        values["TS_SCHED_PREFILL_CHUNK"] = profile == "pp512" ? "512" : null;
        values["TS_UNIFIED_LAYER_MAJOR_TOKENS"] = "8192";
        return values;
    }
    private static NativeEnvScope ConfigureEnvironment(string profile) {
        var env = new NativeEnvScope(); foreach (var pair in EnvironmentValues(profile)) env.Set(pair.Key, pair.Value); return env;
    }
    private static void Marker(string kind, string values) { Console.Error.WriteLine($"[GLM-TEACHER-{kind}] {values}"); Console.Error.Flush(); }
    private static string Required(string suffix) => Environment.GetEnvironmentVariable("TS_TEST_GLM_UNIFIED_TEACHER_" + suffix)
        ?? throw new InvalidOperationException("Missing explicit teacher diagnostic " + suffix);
    private static string TextHash(string text) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();
    private static string FileHash(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    private static string TokenHash(IEnumerable<int> tokens) {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256); Span<byte> bytes = stackalloc byte[4];
        foreach (int token in tokens) { BinaryPrimitives.WriteInt32LittleEndian(bytes, token); hash.AppendData(bytes); }
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }
    private static void WriteNew(string path, object value) {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!); using var file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
        JsonSerializer.Serialize(file, value, Json);
    }
    private static T ReadPrivate<T>(object value, string name) => Assert.IsType<T>(value.GetType().GetField(name, PrivateInstance)!.GetValue(value));
    internal sealed record SourceLayout(int Layer, int GateType, int UpType, int DownType, long GateBytes, long UpBytes, long DownBytes);
    internal sealed record TeacherCase(string Name, string Instruction, string TeacherText, string RenderedPrompt, int[] PromptIds, int[] TeacherIds,
        int FullTeacherTokens, bool TeacherTruncated, string RenderedSha256, string PromptSha256, string TeacherSha256, string FullHistorySha256);
    internal sealed record InputSet(int Schema, string ModelPath, long ModelLength, long ModelModifiedUtcTicks, string Architecture, string TokenizerType,
        string TemplateSha256, string SourceCatalogSha256, SourceLayout[] SourceCatalog, string CaseProvenance, string PaddingProvenance, string Selection, TeacherCase[] Cases);
    internal sealed record TeacherRow(int Index, int Position, int Target, string HistorySha256, string ForwardedHistorySha256,
        GlmTeacherForcingScore Score, int SampledToken, int TokensForwarded, int HeadAfter);
    internal sealed record CaseResult(string Name, int[] PrefillSubmissions, TeacherRow[] Rows, double TotalNll, double MeanNll);

    private sealed class ActualOwner(IUnifiedMemoryModelSession session) : IDisposable
    {
        internal IUnifiedMemoryModelSession Session { get; } = session;
        internal GlmDsaModel Model => Assert.IsType<GlmDsaModel>(Session.Model);
        internal NativeGlmExecutor NativeExecutor => ReadPrivate<NativeGlmExecutor>(Model, "_exec");
        internal IntPtr Handle => ReadPrivate<IntPtr>(NativeExecutor, "_handle");
        internal GgmlCacheBudgetScope Native => ReadPrivate<GgmlCacheBudgetScope>(Session, "_native");
        internal BlockPool? Pool; internal ContinuousBatchScheduler? Scheduler; internal BatchExecutor? Executor; internal SequenceState? Sequence;
        private bool _released = true, _disposed;
        internal void Prepare(int[] prompt) {
            Assert.True(_released); Pool?.Storage.Dispose();
            var model = Model; lock (model.GpuComputeLock) model.ResetKVCache();
            Assert.Equal(0, GgmlGlmNative.NPast(Handle));
            var config = Session.ConfigureScheduler(new SchedulerConfig { EnablePrefixCaching = false });
            Assert.False(config.EnablePrefixCaching); Assert.Equal(1, config.MaxNumRunningSequences);
            Assert.NotNull(config.MemoryAdmission); Assert.True(config.MemoryAdmission.UsePerSequenceHolders);
            Assert.Same(Session.Budget, config.MemoryAdmission.Budget);
            Pool = new BlockPool(config.NumBlocks, config.BlockSize, model.ComputeKVBlockByteSize(config.BlockSize));
            Scheduler = new ContinuousBatchScheduler(config, Pool, NullLogger.Instance, supportsCrossSequenceKvReuse: false);
            Executor = new BatchExecutor(model, Pool, Scheduler, NullLogger.Instance);
            var sampling = new SamplingConfig { Temperature = 0, TopK = 0, TopP = 1, RepetitionPenalty = 1, PenaltyLastN = 0,
                PresencePenalty = 0, FrequencyPenalty = 0, ThinkingBudget = null, MessageThinkingBudget = null, FirstTokenAllowList = null,
                Grammar = null, StopRepetition = false, StopSequences = null, Seed = 0 };
            Sequence = new SequenceState("teacher-" + Guid.NewGuid().ToString("N"), prompt, 64, config.BlockSize, sampling);
            Scheduler.Submit(Sequence); _released = false;
        }
        internal void Release(bool recover) {
            if (_released || Scheduler == null || Sequence == null) return;
            lock (Model.GpuComputeLock) {
                Scheduler.Abort(Sequence.RequestId); Executor?.DiscardReleasedFusedCacheBookkeeping(Sequence.RequestId);
                Executor?.ReleaseLinearSpeculativeContext(Sequence.RequestId); Model.OnSequenceReleased(Sequence.RequestId);
                Executor?.RadixCache?.Drain(); Scheduler.NotifyMemoryReleased(Sequence.RequestId); _released = true;
                if (recover) Scheduler.Config.MemoryAdmission?.AfterSequenceMemoryRelease?.Invoke(Scheduler.GetRunningSequencesSnapshot());
            }
        }
        public void Dispose() {
            if (_disposed) return;
            if (Session.Model is GlmDsaModel model) lock (model.GpuComputeLock) model.PrepareUnifiedNativeDispose();
            Release(recover: false); Pool?.Storage.Dispose(); Session.Dispose(); _disposed = true;
        }
    }
}
