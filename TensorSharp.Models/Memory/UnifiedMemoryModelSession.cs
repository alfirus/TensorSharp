// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models;

/// <summary>Operator ceilings. Automatic plans additionally preserve physical
/// RAM/commit/VRAM headroom; these values are not OS or driver hard limits.</summary>
public sealed record UnifiedMemoryModelOptions
{
    public int ContextTokens { get; init; }
    public int PrefillChunkTokens { get; init; } = 512;
    // The expert cache executes rows separately. Keep small requests on that
    // reuse path, but let longer prefill tails use a compact CUDA batch.
    public int ExpertCachePrefillRows { get; init; } = 32;
    public long MaximumHostBytes { get; init; } = long.MaxValue;
    public long MaximumDeviceBytes { get; init; } = long.MaxValue;
    public long MaximumExpertCacheBytes { get; init; } = long.MaxValue;
}

/// <summary>One Qwen4Exp GGML CUDA model, including its optional image tower.
/// Owns the actual allocation scopes and expert-cache policy for the complete
/// load/serve/unload lifetime. Instrumented allocations are quota enforced;
/// mapped pages, GC and vendor/driver memory remain observed with headroom.
/// Concurrent callers queue; one request executes at a time without KV swaps.</summary>
public sealed class UnifiedMemoryModelSession : IDisposable
{
    public const string HostPool = "unified/ram";
    public const string DevicePool = "unified/cuda0";
    private readonly UnifiedMemoryModelOptions _options;
    private readonly Qwen4ExpUnifiedGeometry _geometry;
    private readonly Qwen35VisionMemoryGeometry? _vision;
    private readonly long _hostHeadroom, _deviceHeadroom;
    private readonly long _hostCeiling, _deviceCeiling;
    private readonly int _contextTokens, _prefillTokens;
    private GgmlCacheBudgetScope? _native;
    private HostAllocationBudgetScope? _host;
    private GgmlHostMoeExpertCacheScope? _experts;
    private Qwen4ExpModel? _model;
    private bool _disposed;

    private UnifiedMemoryModelSession(UnifiedMemoryModelOptions options, Qwen4ExpUnifiedGeometry geometry,
        Qwen35VisionMemoryGeometry? vision, InferenceHardwareMemory hardware, int contextTokens)
    {
        _options = options; _geometry = geometry; _vision = vision; _contextTokens = contextTokens;
        _prefillTokens = Math.Min(options.PrefillChunkTokens, contextTokens);
        var hostCapacity = hardware.HostAllocationCapacity();
        _hostHeadroom = Math.Max(512L << 20, Math.Min(hostCapacity.Total, options.MaximumHostBytes) / 16);
        _deviceHeadroom = Math.Max(768L << 20, Math.Min(hardware.DeviceTotal, options.MaximumDeviceBytes) / 16);
        _hostCeiling = Math.Min(options.MaximumHostBytes, Math.Max(0, hostCapacity.Available - _hostHeadroom));
        _deviceCeiling = Math.Min(options.MaximumDeviceBytes, Math.Max(0, hardware.DeviceAvailable - _deviceHeadroom));
        if (_hostCeiling <= 0 || _deviceCeiling <= 0)
            throw new MemoryPressureException("No usable RAM/commit or CUDA capacity remains after unified-memory headroom.");
        Budget = new MemoryBudget([new(HostPool, _hostCeiling), new(DevicePool, _deviceCeiling)]);
    }

    /// <summary>Null only on a failed construction retained for cleanup retry.</summary>
    public ModelBase? Model => _model;
    public MemoryBudget Budget { get; }
    public int ContextTokens => _contextTokens;
    public int PrefillChunkTokens => _prefillTokens;
    public string FeatureReport =>
        $"[unified-memory] enabled: qwen4exp, one GGML CUDA device; context={_contextTokens}, prefill={_prefillTokens}, active-requests=1\n" +
        $"[unified-memory] allocation ceilings: RAM={_hostCeiling / 1048576.0:F0} MiB, CUDA={_deviceCeiling / 1048576.0:F0} MiB; " +
        $"expert-cache={(_experts?.BudgetBytes ?? 0) / 1048576.0:F0} MiB, expert-prefill-rows={_experts?.MaxPrefillRows ?? 0}\n" +
        $"[unified-memory] on: shared native/host payload accounting, hardware/commit headroom, request admission, " +
        $"bounded expert reuse and pressure trimming, file staging for CUDA/CPU expert execution, fused model kernels{(_vision != null ? ", image preparation admission" : "")}\n" +
        "[unified-memory] off: host KV paging (hybrid state has no complete block snapshot adapter); prefix retention, " +
        "batched concurrent execution, MTP/TP (not qualified with this request envelope); video/audio (no admitted preparation adapter)\n" +
        "[unified-memory] limits: mapped/page-cache residency, GC, vendor/driver pools are observed with margins, " +
        "not fully attributed or OS-enforced RAM/VRAM hard caps.";

    public static UnifiedMemoryModelSession Create(string path, string? mmprojPath, BackendType backend)
        => Create(path, mmprojPath, backend, new UnifiedMemoryModelOptions());

    public static UnifiedMemoryModelSession Create(string path, string? mmprojPath, BackendType backend,
        UnifiedMemoryModelOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ValidateOptions(options);
        if (backend != BackendType.GgmlCuda)
            throw new NotSupportedException("--unified-memory currently requires qwen4exp on ggml_cuda.");
        if (ParallelDegree("TENSORSHARP_TP_DEGREE") != 1 || ParallelDegree("TENSORSHARP_LAYER_SPLIT_DEGREE") != 1
            || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_TP_NODE_ID"))
            || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_TP_PEERS")))
            throw new NotSupportedException("--unified-memory has no qualified tensor/layer-parallel adapter.");
        if (SpeculationOptions.FromEnvironment().Enabled
            || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TS_SPEC_DRAFT_MODEL")))
            throw new NotSupportedException("--unified-memory currently requires speculative execution and draft loading to be disabled.");
        if (MoeCpuOffloadConfig.IsExplicitlySet)
            throw new NotSupportedException("--unified-memory selects expert placement from its shared budget; remove explicit CPU MoE placement overrides.");
        string? fileReadMode = Environment.GetEnvironmentVariable("TS_HOST_MOE_FILE_READ");
        if (fileReadMode != null && fileReadMode != "1")
            throw new NotSupportedException("--unified-memory requires bounded expert file staging, including its CPU fallback; remove TS_HOST_MOE_FILE_READ or set it to 1.");
        Qwen4ExpUnifiedGeometry geometry;
        using (var file = GgufFile.OpenForMemoryPlanning(path)) geometry = Qwen4ExpUnifiedGeometry.Read(file);
        Qwen35VisionMemoryGeometry? vision = string.IsNullOrWhiteSpace(mmprojPath) || mmprojPath == "none"
            ? null : Qwen35VisionMemoryGeometry.Read(mmprojPath);
        int context = options.ContextTokens;
        if (context == 0)
        {
            string? configured = Environment.GetEnvironmentVariable("MAX_CONTEXT");
            if (!string.IsNullOrWhiteSpace(configured))
            {
                if (!int.TryParse(configured, NumberStyles.None, CultureInfo.InvariantCulture, out context) || context <= 0)
                    throw new ArgumentException("MAX_CONTEXT must be a positive token count.");
            }
            else context = Math.Min(32768, geometry.DeclaredContextTokens);
        }
        if (context <= 0 || context > geometry.DeclaredContextTokens)
            throw new ArgumentOutOfRangeException(nameof(options), "Unified context exceeds the model's declared context.");
        var session = new UnifiedMemoryModelSession(options, geometry, vision, InferenceHardwareMemory.CaptureCuda(), context);
        try
        {
            var warmShape = geometry.RequestPeak(Math.Min(context, 8192), session._prefillTokens);
            var decodeShape = geometry.RequestPeak(Math.Min(context, 8192), 1);
            long denseDevice = checked(geometry.LoadingDeviceBoundBytes + (vision?.DeviceWeightBytes ?? 0));
            long loadHost = checked(geometry.ResidentHostPayloadBytes + geometry.LoadingHostTemporaryBytes
                + (vision?.ResidentHostBytes ?? 0) + (vision?.LoadingTemporaryBytes ?? 0) + (128L << 20));
            if (loadHost > session._hostCeiling || denseDevice + warmShape.Device > session._deviceCeiling)
                throw new MemoryPressureException("Unified load forecast cannot fit dense/vision weights and execution workspace within available RAM/commit/VRAM.");
            // This is an optional maximum, not a resident allocation. Reserve
            // wide-prefill workspace when that request is admitted; otherwise
            // the largest supported prefill would permanently suppress reuse
            // even during a smaller request. PrepareCapacity trims before work.
            long cache = SelectExpertCacheBytes(session._deviceCeiling, denseDevice, decodeShape.Device,
                geometry.ExpertSourceBytes, geometry.AllLayerExpertCacheMinimumBytes, options.MaximumExpertCacheBytes);
            Console.WriteLine($"[unified-memory] load forecast: dense+vision CUDA={denseDevice / 1048576.0:F0} MiB, " +
                $"prefill forecast={warmShape.Device / 1048576.0:F0} MiB, decode reserve={decodeShape.Device / 1048576.0:F0} MiB, " +
                $"host load={loadHost / 1048576.0:F0} MiB; " +
                $"automatic expert-cache ceiling={cache / 1048576.0:F0} MiB");
            session._native = new(session.Budget, [[DevicePool]], includeGraphBuffers: true, hostPools: [HostPool]);
            session._host = new(session.Budget, [HostPool]);
            session._experts = new(cache, Math.Min(options.ExpertCachePrefillRows, session._prefillTokens), geometry.Layers);
            var policy = new ModelMemoryPolicy(context, session._prefillTokens)
            {
                OmitEmbeddedDraftWeights = true, ExpertCacheBytes = cache,
                ExpertCachePrefillRows = session._experts.MaxPrefillRows,
                DeviceExecutionReserveBytes = checked(warmShape.Device + (vision?.DeviceWeightBytes ?? 0))
            };
            session._model = (Qwen4ExpModel)ModelBase.Create(path, backend, 1, null!, null!, 1, null!, policy);
            if (vision != null) session._model.LoadVisionEncoder(mmprojPath!);
            session._model.ConfigureUnifiedVisionAdmission(session.EnterVisionPreparation);
            Console.WriteLine(session.FeatureReport);
            return session;
        }
        catch (Exception creation)
        {
            if (creation is Qwen4ExpConstructionCleanupException partial) session._model = partial.UnreleasedModel;
            try { session.Dispose(); }
            catch (Exception cleanup) { throw new UnifiedMemoryModelLoadException(session, new AggregateException(creation, cleanup)); }
            throw;
        }
    }

    internal static long SelectExpertCacheBytes(long available, long dense, long execution, long source,
        long minimum, long ceiling)
    {
        if (new[] { available, dense, execution, source, minimum, ceiling }.Any(x => x < 0))
            throw new ArgumentOutOfRangeException(nameof(available));
        long spare = (long)Math.Max(0, (decimal)available - dense - execution);
        long result = Math.Min(ceiling, Math.Min(source, spare)) / (1L << 20) * (1L << 20);
        return minimum > 0 && result >= minimum ? result : 0;
    }

    public SchedulerConfig ConfigureScheduler(SchedulerConfig requested)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(requested);
        if (requested.Speculation.Enabled || requested.KvSnapshots != null || requested.MemoryAdmission != null)
            throw new NotSupportedException("Unified Qwen execution cannot combine with independent KV snapshots, speculation or another admission adapter.");
        int block = requested.BlockSize > 0 ? requested.BlockSize : SchedulerConfig.DefaultBlockSize;
        return new SchedulerConfig
        {
            BlockSize = block, NumBlocks = Math.Max(1, checked((_contextTokens + block - 1) / block)),
            MaxNumRunningSequences = 1, MaxNumBatchedTokens = _prefillTokens,
            MaxPrefillChunkSize = _prefillTokens, SoloPrefillChunkSize = _prefillTokens,
            PrefillChunkTokenLimit = _prefillTokens, EnablePrefixCaching = false,
            StopRepetition = requested.StopRepetition, DecodeQuantumTokens = requested.DecodeQuantumTokens,
            Speculation = SpeculationOptions.Disabled,
            ExecutionOptionsOverride = new ExecutionOptions
            {
                BatchedPathDisabled = true, PerSeqFusedEnabled = false, BatchedFusedDecodeEnabled = false,
                RetainedFusedCacheBudget = 0, PrefixCheckpointBudget = 0, KvHolderPoolMax = 0
            },
            MemoryAdmission = new RequestMemoryAdmission(Budget, EstimateRequest)
            {
                ReestimateAfterPhysicalCheck = true,
                BeforeAdmission = seq => PrepareCapacity(RequestPeak(seq)),
                EnterSerialExecution = seq => EnterAllocations(seq.MemoryEnvelope
                    ?? throw new InvalidOperationException("Unified execution requires an admitted envelope."))
            }
        };
    }

    private InferenceMemoryBytes RequestPeak(SequenceState seq)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _model!.GetUnifiedMemoryRequestPeak(checked(seq.PromptTokens.Count + seq.MaxNewTokens),
            Math.Min(_prefillTokens, Math.Max(1, seq.PromptTokens.Count)));
    }

    private IReadOnlyList<MemoryCharge> EstimateRequest(SequenceState seq) => IncrementalPeak(RequestPeak(seq));

    private MemoryCharge[] IncrementalPeak(InferenceMemoryBytes peak)
    {
        long expert = _experts!.CachedBytes;
        // This exclusive, trunk-only adapter preloads non-expert quantized
        // weights only. The explicit preload registry excludes both lazy KV
        // copies and the selected-expert cache (which also uses charge kind 1).
        long boundWeights = GgmlBasicOps.TryGetCacheMemoryUsage(0, out var cache)
            ? cache.PreloadCommittedBytes : 0;
        long dense = checked(_geometry.LoadingDeviceBoundBytes + (_vision?.DeviceWeightBytes ?? 0));
        // Opaque host payloads and graph arenas are not reusable credit:
        // text/image shapes can differ and old/new arenas can coexist.
        long host = checked(peak.Host + (64L << 20));
        long device = checked(Math.Max(0, dense - boundWeights)
            + peak.Device
            + Math.Max(0, _experts.BudgetBytes - expert));
        return [new(HostPool, host), new(DevicePool, device)];
    }

    /// <summary>Called only under the compute lock before a new request/encoder,
    /// never once per token. Re-sample after trimming; accounted cache allowance
    /// is deliberately not assumed to equal physically reclaimable VRAM.</summary>
    private IReadOnlyList<MemoryCharge> PrepareCapacity(InferenceMemoryBytes peak, bool recoverExpertCache = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_native!.CallbackError != null) throw new InvalidOperationException("Unified allocation accounting failed.", _native.CallbackError);
        // Wide CPU prefill reuses one charged file-read arena across layers.
        // Release that idle arena before forecasting a new request instead of
        // reserving a second copy. Keep all accounted CUDA expert entries warm.
        _experts!.Trim(_experts.CachedBytes);
        var limits = PhysicalConstraints();
        var needed = IncrementalPeak(peak);
        long previousCacheBudget = _experts!.BudgetBytes;
        if (!Fits(needed, limits))
        {
            // Keep the largest optional cache that leaves the measured deficit.
            var ledger = Budget.Snapshot().ToDictionary(p => p.Pool);
            long free = Math.Max(0, limits.Single(c => c.Pool == DevicePool).Bytes
                - ledger[DevicePool].Committed - ledger[DevicePool].Reserved);
            long deficit = Math.Max(0, needed.Single(c => c.Pool == DevicePool).Bytes - free);
            if (deficit > 0)
            {
                // Leave a little resize hysteresis for lazily materialized
                // driver pools. Repeated tiny per-layer reductions would throw
                // away every warm entry even for the same request geometry.
                long reduced = Math.Max(0, _experts.BudgetBytes - deficit - (128L << 20));
                _experts.UpdateBudget(reduced >= _geometry.AllLayerExpertCacheMinimumBytes ? reduced : 0);
            }
            _model!.TrimIdleMemory();
            limits = PhysicalConstraints(); needed = IncrementalPeak(peak);
            var afterTrim = Budget.Snapshot().Single(p => p.Pool == DevicePool);
            bool deviceStillFull = (decimal)needed.Single(c => c.Pool == DevicePool).Bytes
                + afterTrim.Committed + afterTrim.Reserved > limits.Single(c => c.Pool == DevicePool).Bytes;
            if (deviceStillFull && _experts!.BudgetBytes > 0)
            {
                _experts.UpdateBudget(0);
                limits = PhysicalConstraints(); needed = IncrementalPeak(peak);
            }
            if (!Fits(needed, limits))
                throw new MemoryPressureException("Unified request peak exceeds currently available RAM/commit/VRAM after optional cache reclamation. " +
                    string.Join("; ", needed.Select(c => $"{c.Pool}: additional={c.Bytes / 1048576.0:F0} MiB, " +
                        $"constraint={limits.Single(p => p.Pool == c.Pool).Bytes / 1048576.0:F0} MiB")));
        }
        // Recover optional reuse at request boundaries when a smaller shape or
        // reduced external pressure leaves room. Never use accounted bytes as a
        // substitute for the physical re-sample, and avoid tiny resize churn.
        var current = Budget.Snapshot().Single(p => p.Pool == DevicePool);
        long slack = Math.Max(0, limits.Single(c => c.Pool == DevicePool).Bytes
            - current.Committed - current.Reserved - needed.Single(c => c.Pool == DevicePool).Bytes);
        long recovery = Math.Min(Math.Max(0, slack - (128L << 20)),
            _experts.MaximumBudgetBytes - _experts.BudgetBytes) / (1L << 20) * (1L << 20);
        long recoveredBudget = checked(_experts.BudgetBytes + recovery);
        if (recoverExpertCache && recovery >= 512L << 20
            && recoveredBudget >= _geometry.AllLayerExpertCacheMinimumBytes)
            _experts.UpdateBudget(recoveredBudget);
        if (previousCacheBudget != _experts.BudgetBytes)
            Console.WriteLine($"[unified-memory] request-boundary expert cache: {previousCacheBudget / 1048576.0:F0} -> " +
                $"{_experts.BudgetBytes / 1048576.0:F0} MiB (physical capacity and request peak)");
        return limits;
    }

    private bool Fits(IReadOnlyList<MemoryCharge> peak, IReadOnlyList<MemoryCharge> limits)
    {
        var owned = Budget.Snapshot().ToDictionary(p => p.Pool);
        return peak.All(c => (decimal)c.Bytes + owned[c.Pool].Committed + owned[c.Pool].Reserved
            <= limits.Single(p => p.Pool == c.Pool).Bytes);
    }

    private MemoryCharge[] PhysicalConstraints()
    {
        var hardware = InferenceHardwareMemory.CaptureCuda();
        var host = hardware.HostAllocationCapacity();
        var ledger = Budget.Snapshot().ToDictionary(p => p.Pool);
        // Express additional physical allowance as an absolute ledger constraint.
        // Existing reserved credit still consumes it; no mapped/driver owner is
        // invented, and the configured allocation ceiling never increases.
        long hostLimit = (long)Math.Min(_hostCeiling, (decimal)ledger[HostPool].Committed + Math.Max(0, host.Available - _hostHeadroom));
        long deviceLimit = (long)Math.Min(_deviceCeiling, (decimal)ledger[DevicePool].Committed + Math.Max(0, hardware.DeviceAvailable - _deviceHeadroom));
        return [new(HostPool, hostLimit), new(DevicePool, deviceLimit)];
    }

    private IDisposable EnterVisionPreparation(QwenVisionPreparationPlan plan)
    {
        if (plan.ExpandedPromptTokens > _contextTokens) throw new MemoryPressureException("Image-expanded prompt exceeds the unified context.");
        // The media plan includes cold vision bindings; IncrementalPeak accounts
        // for all not-yet-bound shared weights once, independently of workspace.
        var peak = new InferenceMemoryBytes(plan.HostPeakBytes,
            Math.Max(0, plan.DevicePeakBytes - (_vision?.DeviceWeightBytes ?? 0)));
        // Image encoding is followed immediately by text execution. Do not
        // regrow the expert cache for this temporarily smaller workspace only
        // to evict its warm entries again at text admission.
        var constraints = PrepareCapacity(peak, recoverExpertCache: false);
        var owned = new OwnedExecution();
        var envelope = Budget.TryReserve(IncrementalPeak(peak), constraints)
            ?? throw new MemoryPressureException("Image preparation could not reserve its unified peak.");
        owned.Envelope = envelope;
        try { owned.Execution = EnterAllocations(envelope); return owned; }
        catch { owned.Dispose(); throw; }
    }

    private IDisposable EnterAllocations(BudgetReservation envelope)
    {
        var owned = new AllocationExecution();
        try
        {
            owned.Native = _native!.EnterExecution(envelope);
            owned.Host = _host!.EnterExecution(envelope);
            return owned;
        }
        catch { owned.Dispose(); throw; }
    }

    private sealed class AllocationExecution : IDisposable
    {
        internal IDisposable? Native, Host;
        public void Dispose() { try { Host?.Dispose(); } finally { Native?.Dispose(); } }
    }
    private sealed class OwnedExecution : IDisposable
    {
        internal IDisposable? Execution, Envelope;
        public void Dispose() { try { Execution?.Dispose(); } finally { Envelope?.Dispose(); } }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _model?.Dispose(); _model = null;
        _experts?.Dispose(); _experts = null;
        _host?.Dispose(); _host = null;
        _native?.Dispose(); _native = null;
        _disposed = true;
    }

    private static int ParallelDegree(string name)
    {
        string? value = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(value)) return 1;
        return int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int degree) && degree > 0
            ? degree : throw new ArgumentException(name + " must be a positive degree.");
    }

    private static void ValidateOptions(UnifiedMemoryModelOptions options)
    {
        if (options.ContextTokens < 0 || options.PrefillChunkTokens is < 1 or > 512
            || options.ExpertCachePrefillRows is < 1 or > 128
            || options.MaximumHostBytes <= 0 || options.MaximumDeviceBytes <= 0 || options.MaximumExpertCacheBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(options));
    }
}

/// <summary>Cleanup failed during construction. Keep this owner and retry its
/// Dispose before loading any replacement; allocation callbacks stay attached.</summary>
public sealed class UnifiedMemoryModelLoadException : InvalidOperationException
{
    public UnifiedMemoryModelLoadException(UnifiedMemoryModelSession session, Exception inner)
        : base("Unified model construction failed with unreleased owners; cleanup must be retried.", inner)
        => UnreleasedSession = session;
    public UnifiedMemoryModelSession UnreleasedSession { get; }
}
