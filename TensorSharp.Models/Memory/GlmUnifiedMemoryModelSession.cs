// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models;

/// <summary>Single-device native GLM5Next shared-budget owner. Native weights,
/// complete MLA/indexer/KDA slots and cached scheduler payloads use the same
/// ledger as managed tensors and upload/media staging. Source mappings, GC and
/// vendor pools are observed with headroom rather than claimed as hard-capped.</summary>
public sealed class GlmUnifiedMemoryModelSession : IUnifiedMemoryModelSession
{
    private const string HostPool = UnifiedMemoryModelSession.HostPool;
    private const string DevicePool = UnifiedMemoryModelSession.DevicePool;
    private const int GraphEntriesWithTransient = GlmDsaModel.UnifiedGraphCacheCapacity + 1;
    private readonly UnifiedMemoryModelOptions _options;
    private readonly GlmUnifiedMemoryGeometry _geometry;
    private readonly GlmUnifiedVisionMemoryGeometry? _vision;
    private readonly long _hostCeiling, _deviceCeiling, _hostHeadroom, _deviceHeadroom;
    private readonly int _context, _prefill;
    private GgmlCacheBudgetScope? _native;
    private HostAllocationBudgetScope? _host;
    private BudgetReservation? _joint;
    private GgmlHostMoeExpertCacheScope? _experts;
    private BudgetReservation? _expertEnvelope;
    private long _expertCacheBytes, _expertMinimumBytes, _expertMinimumForecastBytes;
    private readonly long _executionReserveBytes;
    private int _expertGroups;
    private GlmDsaModel? _model;
    private bool _disposed;
    private int _cpuLayers;
    private GlmUnifiedGraphShape? _lastGraphShape;
    private BudgetReservation? _executingRequest;

    private GlmUnifiedMemoryModelSession(UnifiedMemoryModelOptions options, GlmUnifiedMemoryGeometry geometry,
        GlmUnifiedVisionMemoryGeometry? vision, InferenceHardwareMemory hardware, int context)
    {
        _options = options; _geometry = geometry; _vision = vision; _context = context;
        var host = hardware.HostAllocationCapacity();
        _hostHeadroom = Math.Max(512L << 20, Math.Min(host.Total, options.MaximumHostBytes) / 16);
        _deviceHeadroom = Math.Max(768L << 20, Math.Min(hardware.DeviceTotal, options.MaximumDeviceBytes) / 16);
        _hostCeiling = Math.Min(options.MaximumHostBytes, Math.Max(0, host.Available - _hostHeadroom));
        _deviceCeiling = Math.Min(options.MaximumDeviceBytes, Math.Max(0, hardware.DeviceAvailable - _deviceHeadroom));
        if (_hostCeiling <= 0 || _deviceCeiling <= 0) throw new MemoryPressureException("No GLM RAM/commit/CUDA capacity remains after headroom.");
        var plan = GlmSelectedExpertMemoryPlan.Select(geometry, options, context, _deviceCeiling, vision?.DeviceWeightBytes ?? 0);
        _prefill = plan.PrefillRows;
        _cpuLayers = plan.CpuExpertLayers;
        _expertCacheBytes = plan.CacheCapacityBytes;
        _expertMinimumForecastBytes = plan.CacheMinimumForecastBytes;
        _executionReserveBytes = plan.ExecutionReserveBytes;
        Budget = new MemoryBudget([new(HostPool, _hostCeiling), new(DevicePool, _deviceCeiling)]);
    }
    public ModelBase? Model => _model;
    public MemoryBudget Budget { get; }
    public int ContextTokens => _context;
    public int PrefillChunkTokens => _prefill;
    public string FeatureReport =>
        $"[unified-memory] enabled: {_geometry.Architecture}, native GGML CUDA; context={_context}, prefill={_prefill}, active-request-limit={_options.MaximumRunningRequests}\n" +
        $"[unified-memory] allocation ceilings: RAM={_hostCeiling / 1048576.0:F0} MiB, CUDA={_deviceCeiling / 1048576.0:F0} MiB; CPU-expert-leading-layers={_cpuLayers}/{_geometry.Layers}\n" +
        "[unified-memory] on: native weight/constant/slot/graph accounting, host tensor/upload accounting, physical RAM/commit/VRAM headroom, request admission, native joint decode, bounded donation of exact terminal-state conversation prefixes, admitted still-image preparation\n" +
        (_experts == null
            ? (_options.EnableGlmSelectedExperts && _cpuLayers == 0
                ? "[unified-memory] GLM experts: all routed weights are resident on CUDA; a compact selected-expert tier is unnecessary for this capacity plan\n"
                : "[unified-memory] GLM experts: capacity-placed resident CUDA layers and pageable mapped CPU layers; selected-expert CUDA reuse is off pending full-model qualification\n")
            : $"[unified-memory] GLM experts: experimental pooled CUDA reuse; groups={_expertGroups}, budget={_expertCacheBytes / 1048576.0:F0} MiB, metadata forecast={_expertMinimumForecastBytes / 1048576.0:F0} MiB, exact minimum={_expertMinimumBytes / 1048576.0:F0} MiB, planned decode widths=1..{_options.MaximumRunningRequests}, prefill rows up to {_prefill} (requested {_options.PrefillChunkTokens}); wide prefill protects its entire routed union; bounded source-file staging; full-model performance remains unqualified\n") +
        "[unified-memory] asynchronous file/transfer/compute overlap is off pending GLM qualification\n" +
        "[unified-memory] off: host KV paging, MTP/draft and TP (complete shared-budget adapters remain unqualified), video/audio\n" +
        "[unified-memory] limits: mapping/page-cache residency, GC, native metadata and vendor/driver pools retain observed headroom; no complete OS RAM/VRAM hard-cap claim.";

    public static GlmUnifiedMemoryModelSession Create(string path, string? mmprojPath, BackendType backend,
        UnifiedMemoryModelOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (backend != BackendType.GgmlCuda || options.MaximumRunningRequests is < 1 or > 8
            || options.PrefillChunkTokens is < 1 or > 512 || options.ContextTokens < 0
            || options.MaximumHostBytes <= 0 || options.MaximumDeviceBytes <= 0 || options.MaximumExpertCacheBytes < 0
            || options.RetainedConversationCount is < 0 or > 8)
            throw new NotSupportedException("Unified GLM requires GGML CUDA and valid single-device request geometry.");
        if (Environment.GetEnvironmentVariable("TS_GLM_NATIVE") == "0" || SpeculationOptions.FromEnvironment().Enabled)
            throw new NotSupportedException("Unified GLM requires its native, non-speculative executor.");
        if (options.EnableAsyncExpertPrefill || (options.EnableGlmSelectedExperts && options.MaximumRunningRequests > 3))
            throw new NotSupportedException("Experimental unified GLM supports at most three selected-expert decode rows and no asynchronous expert pipeline.");
        GgmlGlmNative.RequireUnifiedBudgetSupport();
        if (options.EnableGlmSelectedExperts) GgmlGlmNative.RequireSelectedExpertPlanningSupport();
        GlmUnifiedMemoryGeometry geometry;
        using (var file = GgufFile.OpenForMemoryPlanning(path)) geometry = GlmUnifiedMemoryGeometry.Read(file);
        var vision = string.IsNullOrWhiteSpace(mmprojPath) || mmprojPath == "none" ? null : GlmUnifiedVisionMemoryGeometry.Read(mmprojPath);
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
        if (context <= 0 || context > geometry.DeclaredContextTokens) throw new ArgumentOutOfRangeException(nameof(options));
        var owner = new GlmUnifiedMemoryModelSession(options, geometry, vision, InferenceHardwareMemory.CaptureCuda(), context);
        try
        {
            owner._native = new(owner.Budget, [[DevicePool], [HostPool]], includeGraphBuffers: true, hostPools: [HostPool]);
            owner._host = new(owner.Budget, [HostPool]);
            owner._joint = owner.Budget.Reserve([new(HostPool, 0), new(DevicePool, 0)]);
            owner._native.SetSharedJointEnvelope(owner._joint);
            if (owner._expertCacheBytes > 0)
            {
                owner._expertEnvelope = owner.Budget.Reserve([new(DevicePool, owner._expertCacheBytes)]);
                owner._native.SetSharedExpertEnvelope(owner._expertEnvelope);
            }
            var policy = new ModelMemoryPolicy(context, owner._prefill)
            {
                OmitEmbeddedDraftWeights = true, NativeMoeCpuLayers = owner._cpuLayers,
                DeviceExecutionReserveBytes = owner._executionReserveBytes
            };
            owner._model = (GlmDsaModel)ModelBase.Create(path, backend, 1, null!, null!, 1, null!, policy);
            if (owner._model.MaxContextLength != context) throw new InvalidOperationException("Native GLM changed an explicit unified context.");
            owner._model.ConfigureUnifiedMemory(options.RetainedConversationCount);
            if (owner._expertCacheBytes > 0)
            {
                owner._expertGroups = owner._model.UnifiedExpertGroups;
                if (owner._expertGroups <= 0) throw new NotSupportedException("No complete mapped GLM expert layout groups were found.");
                owner._experts = new(owner._expertCacheBytes, owner._prefill, owner._expertGroups, options.MaximumRunningRequests);
                owner._expertMinimumBytes = owner._model.PlanUnifiedExpertMinimum(options.MaximumRunningRequests);
                if (owner._expertCacheBytes < owner._expertMinimumBytes)
                    throw new MemoryPressureException($"GLM pooled expert budget {owner._expertCacheBytes} bytes cannot protect its complete admitted union/graph minimum {owner._expertMinimumBytes} bytes.");
                owner._model.ConfigureUnifiedSelectedExperts();
                owner._model.TrimUnifiedMappedResidency(); // Advisory only; no imputed released-byte credit.
            }
            owner._model.ConfigureUnifiedGraphAdmission(owner.PrepareGraph);
            if (vision != null) owner._model.LoadVisionEncoder(mmprojPath!);
            owner._model.ConfigureUnifiedVisionAdmission(owner.EnterVisionPreparation);
            Console.WriteLine(owner.FeatureReport);
            return owner;
        }
        catch (Exception creation)
        {
            if (creation is GlmConstructionCleanupException partial) owner._model = partial.UnreleasedModel;
            try { owner.Dispose(); }
            catch (Exception cleanup) { throw new UnifiedMemoryModelLoadException(owner, new AggregateException(creation, cleanup)); }
            throw;
        }
    }

    public SchedulerConfig ConfigureScheduler(SchedulerConfig requested)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (requested.Speculation.Enabled || requested.KvSnapshots != null || requested.MemoryAdmission != null)
            throw new NotSupportedException("Unified GLM cannot combine with independent speculation or snapshot/admission adapters.");
        int block = requested.BlockSize > 0 ? requested.BlockSize : SchedulerConfig.DefaultBlockSize;
        return new SchedulerConfig
        {
            BlockSize = block, NumBlocks = checked((_context + block - 1) / block *
                (_options.MaximumRunningRequests + _options.RetainedConversationCount + 1)),
            MaxNumRunningSequences = _options.MaximumRunningRequests, MaxNumBatchedTokens = _prefill,
            MaxPrefillChunkSize = _prefill, SoloPrefillChunkSize = _prefill, PrefillChunkTokenLimit = _prefill,
            EnablePrefixCaching = requested.EnablePrefixCaching, StopRepetition = requested.StopRepetition,
            DecodeQuantumTokens = requested.DecodeQuantumTokens, Speculation = SpeculationOptions.Disabled,
            ExecutionOptionsOverride = new ExecutionOptions
            {
                BatchedPathDisabled = true, PerSeqFusedEnabled = true, BatchedFusedDecodeEnabled = true,
                RetainedFusedCacheBudget = requested.EnablePrefixCaching ? _options.RetainedConversationCount : 0,
                PrefixCheckpointBudget = 0, KvHolderPoolMax = 0
            },
            MemoryAdmission = new RequestMemoryAdmission(Budget, EstimateRequest)
            {
                UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
                EstimateMaterializationPeak = EstimateMaterialization,
                BeforeAdmissionWithRunning = PrepareAdmission,
                AfterPrefixMaterialization = CompletePrefix, AfterPrefill = CompletePrefill,
                EnterSerialExecution = EnterSerial,
                EnterBatchedExecution = _ => EnterAllocations(_joint
                    ?? throw new InvalidOperationException("GLM joint execution requires its shared graph envelope."))
            }
        };
    }
    private InferenceMemoryBytes RequestPeak(SequenceState seq, bool materializing = false)
    {
        int tokens = checked(seq.PromptTokens.Count + seq.MaxNewTokens);
        if (tokens > _context) throw new MemoryPressureException("GLM request exceeds its unified context.");
        int rows = Math.Min(_prefill, Math.Max(1, seq.PromptTokens.Count - seq.NumComputedTokens));
        var staging = _geometry.Workspace(Math.Max(1, seq.PromptTokens.Count), rows);
        // A Submit estimate always includes a new full slot. At the fenced
        // materialization boundary the model can reuse an actual idle primary
        // or donated holder. Existing child ownership is never revoked.
        long slot = materializing && _model!.CanReuseUnifiedSlot(seq.RequestId) ? 0 : _geometry.SlotBytes(_context);
        return new(staging.Host, slot);
    }
    private IReadOnlyList<MemoryCharge> EstimateRequest(SequenceState seq) => Charges(RequestPeak(seq));
    private IReadOnlyList<MemoryCharge> EstimateMaterialization(SequenceState seq) => Charges(RequestPeak(seq, true));
    private static MemoryCharge[] Charges(InferenceMemoryBytes bytes) => [new(HostPool, bytes.Host), new(DevicePool, bytes.Device)];
    private IReadOnlyList<MemoryCharge> PrepareAdmission(SequenceState seq, IReadOnlyList<SequenceState> running)
    {
        if (_native!.CallbackError != null) throw new InvalidOperationException("GLM allocation accounting failed.", _native.CallbackError);
        var limits = PhysicalConstraints();
        int context = Math.Max(1, seq.PromptTokens.Count);
        foreach (var live in running) context = Math.Max(context, checked(live.NumComputedTokens + 1));
        context = PaddedContext(context);
        int rows = Math.Min(_prefill, Math.Max(1, seq.PromptTokens.Count - seq.NumComputedTokens));
        var prefill = _geometry.GraphPeak(context, rows, false, GraphEntriesWithTransient);
        var joint = _geometry.GraphPeak(context, running.Count + 1, running.Count > 0, GraphEntriesWithTransient);
        var target = PreserveReplacement(new(Math.Max(prefill.Host, joint.Host), Math.Max(prefill.Device, joint.Device)));
        var needed = EstimateMaterialization(seq);
        bool FitsCombined()
        {
            var owned = _joint!.OwnedCharges.ToDictionary(c => c.Pool, c => c.Bytes);
            var growth = Charges(target).Select(c => new MemoryCharge(c.Pool,
                checked(c.Bytes - owned[c.Pool] + needed.Single(n => n.Pool == c.Pool).Bytes))).ToArray();
            return Fits(growth, limits);
        }
        if (!FitsCombined())
        {
            _model!.TrimUnifiedGraphs(); _model.TrimUnifiedRetainedSlots(); limits = PhysicalConstraints();
            _lastGraphShape = null;
            target = PreserveReplacement(new(Math.Max(prefill.Host, joint.Host), Math.Max(prefill.Device, joint.Device)));
            needed = EstimateMaterialization(seq);
            if (!FitsCombined()) throw new MemoryPressureException("GLM slot and shared graph peak exceed current RAM/commit/VRAM after idle graph/prefix reclamation.");
        }
        // Combined capacity is checked before consuming any shared credit. The
        // scheduler then acquires the separate materialization envelope under
        // its compute lock; live shared children preserve their physical floor.
        if (!_joint!.TryResizeTotal(Charges(target), limits)) throw new MemoryPressureException("GLM shared graph admission could not reserve its replacement peak.");
        return limits;
    }
    private void CompletePrefix(SequenceState seq)
    {
        if (!seq.MemoryEnvelope!.TryResizeTotal(EstimateMaterialization(seq))) throw new MemoryPressureException("GLM prefix transition cannot preserve its request state/staging peak.");
    }
    private void CompletePrefill(SequenceState seq)
    {
        _model!.TrimUnifiedGraphs(onlyPrefill: true);
        _lastGraphShape = null;
        int context = PaddedContext(Math.Max(1, checked(seq.NumComputedTokens + 1)));
        var graph = PreserveReplacement(_geometry.GraphPeak(context, 1, false, GraphEntriesWithTransient));
        if (!Budget.TryResizeTotals([(_joint!, Charges(graph)), (seq.MemoryEnvelope!, EstimateMaterialization(seq))]))
            throw new MemoryPressureException("GLM prefill transition cannot preserve its shared graph and request state peak.");
    }

    private int PaddedContext(int tokens)
        => Math.Min(_context, checked((Math.Max(1, tokens) + 255) / 256 * 256));
    private InferenceMemoryBytes PreserveReplacement(InferenceMemoryBytes twoGraphs)
    {
        var children = _joint!.LiveChildCharges.ToDictionary(c => c.Pool, c => c.Bytes);
        // The native cache retains one graph while allocating its replacement.
        // An earlier wider PP graph can be larger than this next decode graph.
        return new(Math.Max(twoGraphs.Host, checked(children[HostPool] + twoGraphs.Host / GraphEntriesWithTransient)),
            Math.Max(twoGraphs.Device, checked(children[DevicePool] + twoGraphs.Device / GraphEntriesWithTransient)));
    }
    private void PrepareGraph(GlmUnifiedGraphShape shape)
    {
        int rowLimit = shape.Batched ? _options.MaximumRunningRequests : _prefill;
        if (shape.Rows <= 0 || shape.Rows > rowLimit || shape.ContextTokens <= 0 || shape.ContextTokens > _context)
            throw new MemoryPressureException("GLM graph shape exceeds the admitted execution geometry.");
        shape = shape with { ContextTokens = PaddedContext(shape.ContextTokens) };
        if (_lastGraphShape == shape) return;
        var target = PreserveReplacement(_geometry.GraphPeak(shape.ContextTokens, shape.Rows, shape.Batched, GraphEntriesWithTransient));
        bool Resize(IReadOnlyList<MemoryCharge>? limits = null) => _executingRequest is { } request
            ? Budget.TryResizeTotals([(_joint!, Charges(target)), (request, request.OwnedCharges)], limits)
            : _joint!.TryResizeTotal(Charges(target), limits);
        var owned = _joint!.OwnedCharges.ToDictionary(c => c.Pool, c => c.Bytes);
        bool grows = target.Host > owned[HostPool] || target.Device > owned[DevicePool];
        if (!Resize(grows ? PhysicalConstraints() : null))
        {
            _model!.TrimUnifiedGraphs(); _model.TrimUnifiedRetainedSlots(); _lastGraphShape = null;
            target = PreserveReplacement(_geometry.GraphPeak(shape.ContextTokens, shape.Rows, shape.Batched, GraphEntriesWithTransient));
            var limits = PhysicalConstraints();
            if (!Resize(limits))
            {
                // File mappings have no anonymous payload ticket. Their pages
                // remain reclaimable, but a very large CPU expert scan can
                // consume currently observed RAM before its next graph grows.
                // Ask this quiescent model to release only those read-only
                // mapping pages, then authorize growth from a fresh observation.
                bool hinted = _model.TrimUnifiedMappedResidency();
                limits = PhysicalConstraints();
                if (!Resize(limits))
                {
                    string Describe(MemoryCharge charge) => $"{charge.Pool}={charge.Bytes}";
                    string ledger = string.Join(", ", Budget.Snapshot().Select(p =>
                        $"{p.Pool}: reserved={p.Reserved}, committed={p.Committed}, capacity={p.Capacity}"));
                    throw new MemoryPressureException($"GLM next graph cannot reserve its replacement peak before dispatch: " +
                        $"shape={shape}; target=[{string.Join(", ", Charges(target).Select(Describe))}]; " +
                        $"limits=[{string.Join(", ", limits.Select(Describe))}]; {ledger}; mapped-trim-hint={hinted}.");
                }
            }
        }
        _lastGraphShape = shape;
    }
    private IDisposable EnterSerial(SequenceState seq)
    {
        var request = seq.MemoryEnvelope ?? throw new InvalidOperationException("GLM execution requires an admitted envelope.");
        if (_executingRequest != null) throw new InvalidOperationException("GLM request allocation scopes cannot nest.");
        var scope = EnterAllocations(request);
        try
        {
            var execution = new SerialExecution(this, scope);
            _executingRequest = request;
            return execution;
        }
        catch { scope.Dispose(); throw; }
    }
    private sealed class SerialExecution(GlmUnifiedMemoryModelSession owner, IDisposable scope) : IDisposable
    {
        public void Dispose() { try { scope.Dispose(); } finally { owner._executingRequest = null; } }
    }
    private bool Fits(IReadOnlyList<MemoryCharge> bytes, IReadOnlyList<MemoryCharge> limits)
    {
        var owned = Budget.Snapshot().ToDictionary(p => p.Pool);
        return bytes.All(c => (decimal)c.Bytes + owned[c.Pool].Reserved + owned[c.Pool].Committed <= limits.Single(p => p.Pool == c.Pool).Bytes);
    }
    private MemoryCharge[] PhysicalConstraints()
    {
        var hardware = InferenceHardwareMemory.CaptureCuda();
        var owned = Budget.Snapshot().ToDictionary(p => p.Pool);
        long host = (long)Math.Min(_hostCeiling, (decimal)owned[HostPool].Committed
            + Math.Max(0, hardware.HostAllocationCapacity().Available - _hostHeadroom));
        long device = (long)Math.Min(_deviceCeiling, (decimal)owned[DevicePool].Committed
            + Math.Max(0, hardware.DeviceAvailable - _deviceHeadroom));
        return [new(HostPool, host), new(DevicePool, device)];
    }
    private IDisposable EnterVisionPreparation(GlmVisionPreparationPlan plan)
    {
        var limits = PhysicalConstraints();
        var needed = Charges(new(plan.HostPeakBytes, plan.DevicePeakBytes));
        if (!Fits(needed, limits)) { _model!.TrimUnifiedGraphs(); _model.TrimUnifiedRetainedSlots(); limits = PhysicalConstraints(); }
        var envelope = Budget.TryReserve(needed, limits) ?? throw new MemoryPressureException("GLM image preparation exceeds the shared budget.");
        try { return new OwnedExecution(EnterAllocations(envelope), envelope); }
        catch { envelope.Dispose(); throw; }
    }
    private IDisposable EnterAllocations(BudgetReservation envelope)
    {
        IDisposable native = _native!.EnterExecution(envelope);
        IDisposable? host = null;
        try { host = _host!.EnterExecution(envelope); return new OwnedExecution(host, native); }
        catch { try { host?.Dispose(); } finally { native.Dispose(); } throw; }
    }
    private sealed class OwnedExecution(IDisposable first, IDisposable second) : IDisposable
    {
        public void Dispose() { try { first.Dispose(); } finally { second.Dispose(); } }
    }
    public void Dispose()
    {
        if (_disposed) return;
        _model?.Dispose(); _model = null;
        _experts?.Dispose(); _experts = null;
        _native?.SetSharedExpertEnvelope(null);
        _expertEnvelope?.Dispose(); _expertEnvelope = null;
        _native?.SetSharedJointEnvelope(null);
        _joint?.Dispose(); _joint = null;
        _host?.Dispose(); _host = null;
        _native?.Dispose(); _native = null;
        _disposed = true;
    }
}
