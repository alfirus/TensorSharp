// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
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
public sealed partial class GlmUnifiedMemoryModelSession : IUnifiedMemoryModelSession
{
    private const string HostPool = UnifiedMemoryModelSession.HostPool;
    private const string DevicePool = UnifiedMemoryModelSession.DevicePool;
    private const int GraphEntriesWithTransient = GlmDsaModel.UnifiedGraphCacheCapacity + 1;
    private readonly UnifiedMemoryModelOptions _options;
    private readonly GlmUnifiedMemoryGeometry _geometry;
    private readonly GlmUnifiedVisionMemoryGeometry? _vision;
    private readonly long _hostCeiling, _deviceCeiling, _hostHeadroom, _deviceHeadroom;
    private readonly int _context, _prefill;
    private readonly bool _decodeGraphReserveRequested;
    private readonly long _layerMajorGraphCapBytes;
    private readonly bool _vmmWeightsRequested;
    private readonly long _vmmMetadataBytes;
    private readonly bool _fileSourceViewRequested;
    private readonly GgmlHostMoeFileSourceViewPlan _fileSourceViewPlan;
    private readonly bool _causalRouteNeighborRequested;
    private readonly long _causalRouteNeighborHostBytes;
    // Prompt tokens per Forward; above _prefill only with layer-major prefill.
    private readonly int _forwardTokens;
    private readonly int _prefixCopies;
    private int _effectivePrefixCopies;
    private int _effectiveRetainedConversations;
    private GgmlCacheBudgetScope? _native;
    private HostAllocationBudgetScope? _host;
    private BudgetReservation? _joint;
    private BudgetReservation? _speculativeEnvelope;
    private GgmlHostMoeExpertCacheScope? _experts;
    private BudgetReservation? _expertEnvelope;
    private BudgetReservation? _vmmMetadataEnvelope;
    private BudgetReservation? _fileSourceViewEnvelope;
    private BudgetReservation? _causalRouteNeighborEnvelope;
    private GgmlHostMoeRamCacheScope? _hostExperts;
    private BudgetReservation? _hostExpertEnvelope;
    private long _expertCacheBytes, _expertMinimumBytes, _expertMinimumForecastBytes;
    private readonly long _expertMaximumBytes;
    private readonly long _executionReserveBytes;
    private int _expertGroups;
    private GlmDsaModel? _model;
    private bool _disposed;
    private int _cpuLayers;
    private GlmUnifiedGraphShape? _lastGraphShape;
    private bool _lastGraphUsedBoundedDecode;
    private bool _lastGraphUsedStrictLayerMajor;
    private BudgetReservation? _executingRequest;
    private SequenceState? _executingSequence;
    private int _prefixCaptureDiagnostics;
    private long _batonObservedCompletedCall;
    private long SpeculativeSnapshotBytes => _options.EnableGlmFlashMtp
        ? GlmUnifiedMtpDraftPlan.SnapshotForecast(_geometry) : 0;
    private int ExpertDecodeRows => _options.EnableGlmFlashMtp ? 3 : _options.MaximumRunningRequests;
    private sealed class PrefillState(int cap)
    {
        internal int Cap = cap;
        internal InferenceMemoryBytes Peak;
        internal WeakReference<SequenceState>[] Siblings = [];
        internal bool DecodeGraphRecoveryAttempted;
        internal bool StrictGraphRecoveryAttempted;
    }
    private readonly ConditionalWeakTable<SequenceState, PrefillState> _requestPrefill = new();
    private readonly GlmUnifiedPrimarySlotClaim _primarySlotClaim = new();

    private GlmUnifiedMemoryModelSession(UnifiedMemoryModelOptions options, GlmUnifiedMemoryGeometry geometry,
        GlmUnifiedVisionMemoryGeometry? vision, InferenceHardwareMemory hardware, int context,
        long layerMajorGraphCapBytes = 0, bool vmmWeightsRequested = false, long vmmMetadataBytes = 0,
        bool fileSourceViewRequested = false, GgmlHostMoeFileSourceViewPlan fileSourceViewPlan = default,
        bool causalRouteNeighborRequested = false, long causalRouteNeighborHostBytes = 0)
    {
        _options = options; _geometry = geometry; _vision = vision; _context = context;
        _layerMajorGraphCapBytes = layerMajorGraphCapBytes;
        _vmmWeightsRequested = vmmWeightsRequested; _vmmMetadataBytes = vmmMetadataBytes;
        _fileSourceViewRequested = fileSourceViewRequested; _fileSourceViewPlan = fileSourceViewPlan;
        _causalRouteNeighborRequested = causalRouteNeighborRequested;
        _causalRouteNeighborHostBytes = causalRouteNeighborHostBytes;
        _effectiveRetainedConversations = options.RetainedConversationCount;
        _decodeGraphReserveRequested = GlmUnifiedDecodeGraphPlan.Enabled(
            Environment.GetEnvironmentVariable("TS_GLM_DECODE_GRAPH_RESERVE"));
        var host = hardware.HostAllocationCapacity();
        _hostHeadroom = Math.Max(512L << 20, Math.Min(host.Total, options.MaximumHostBytes) / 16);
        _deviceHeadroom = Math.Max(768L << 20, Math.Min(hardware.DeviceTotal, options.MaximumDeviceBytes) / 16);
        _hostCeiling = Math.Min(options.MaximumHostBytes, Math.Max(0, host.Available - _hostHeadroom));
        _deviceCeiling = Math.Min(options.MaximumDeviceBytes, Math.Max(0, hardware.DeviceAvailable - _deviceHeadroom));
        if (_hostCeiling <= 0 || _deviceCeiling <= 0) throw new MemoryPressureException("No GLM RAM/commit/CUDA capacity remains after headroom.");
        var prefix = UnifiedMemoryPrefixCapacity.Select(GgmlGlmNative.SupportsSlotCopy()
            ? options.PrefixCheckpointCount : 0, geometry.SlotBytes(context), _deviceCeiling);
        long visionPlacement = VisionPlacementBytes(geometry, vision, options, context);
        GlmSelectedExpertMemoryPlan plan;
        while (true)
        {
            try
            {
                plan = GlmSelectedExpertMemoryPlan.SelectForSession(geometry, options, context, _deviceCeiling,
                    checked(visionPlacement + prefix.PlacementReserveBytes
                        // Fresh clone destinations coexist with the idle primary.
                        // Count it rather than assuming a donation consumes it.
                        + (prefix.RetainedCopies > 0 ? geometry.SlotBytes(context) : 0)));
                break;
            }
            catch (MemoryPressureException) when (prefix.RetainedCopies > 0)
            {
                int copies = prefix.RetainedCopies - 1;
                prefix = new(copies, copies > 0 ? checked((long)copies * geometry.SlotBytes(context)) : 0);
            }
        }
        _prefixCopies = prefix.RetainedCopies;
        _prefill = plan.PrefillRows;
        _forwardTokens = Math.Max(_prefill, Math.Min(options.LayerMajorPrefillTokens, context));
        _cpuLayers = plan.CpuExpertLayers;
        _expertCacheBytes = plan.CacheCapacityBytes;
        _expertMaximumBytes = GlmUnifiedExpertLending.MaximumBudget(_expertCacheBytes,
            _deviceCeiling, geometry.DenseDeviceBytes, options.MaximumExpertCacheBytes,
            options.EnableAdaptiveGlmPrefill && options.EnableExpertCacheCapacityGrowth);
        _expertMinimumForecastBytes = plan.CacheMinimumForecastBytes;
        _executionReserveBytes = plan.ExecutionReserveBytes;
        Budget = new MemoryBudget([new(HostPool, _hostCeiling), new(DevicePool, _deviceCeiling)]);
    }
    /// <summary>The streamed encoder holds no vision bytes between images, so
    /// the adaptive plan withholds only what its idle execution reservation
    /// cannot hold for one minimum image. Withholding the whole F32 tower
    /// (2.37 GB for GLM-5.3's F16 projector) collapsed text prefill on a 16 GB
    /// GPU from 784 to 10 rows.</summary>
    internal static long VisionPlacementBytes(GlmUnifiedMemoryGeometry geometry, GlmUnifiedVisionMemoryGeometry? vision,
        UnifiedMemoryModelOptions options, int context)
        => vision == null ? 0 : options.EnableAdaptiveGlmPrefill
            ? GlmUnifiedVisionMemoryGeometry.PlacementBytes(vision,
                geometry.AdaptiveExecutionPlacementReserve(context, options.MaximumRunningRequests))
            : vision.DeviceWeightBytes;
    public ModelBase? Model => _model;
    public MemoryBudget Budget { get; }
    public int ContextTokens => _context;
    public int PrefillChunkTokens => _prefill;
    public string FeatureReport =>
        $"[unified-memory] enabled: {_geometry.Architecture}, native GGML CUDA; context={_context}, prefill={_prefill}, active-request-limit={_options.MaximumRunningRequests}\n" +
        $"[unified-memory] allocation ceilings: RAM={_hostCeiling / 1048576.0:F0} MiB, CUDA={_deviceCeiling / 1048576.0:F0} MiB; CPU-expert-leading-layers={_cpuLayers}/{_geometry.Layers}\n" +
        "[unified-memory] on: native weight/constant/slot/graph accounting, host tensor/upload accounting, physical RAM/commit/VRAM headroom, request admission" +
        (_options.EnableGlmFlashMtp ? "; one primary speculative executor\n"
            : ", native joint decode, bounded donation of exact terminal-state conversation prefixes, admitted still-image preparation\n") +
        $"[unified-memory] public prefix capacity: planned={_prefixCopies}, effective={_effectivePrefixCopies}; retained complete-state copies are placed, optional capture-before-eviction requires live spare capacity, no recurrent rewind or disk persistence\n" +
        (_experts == null
            ? (_options.EnableGlmSelectedExperts && _cpuLayers == 0
                ? "[unified-memory] GLM experts: all routed weights are resident on CUDA; a compact selected-expert tier is unnecessary for this capacity plan\n"
                : "[unified-memory] GLM experts: capacity-placed resident CUDA layers and pageable mapped CPU layers; selected-expert CUDA reuse is off pending full-model qualification\n")
            : $"[unified-memory] GLM experts: experimental pooled CUDA reuse; groups={_expertGroups}, budget={_expertCacheBytes / 1048576.0:F0} MiB, metadata forecast={_expertMinimumForecastBytes / 1048576.0:F0} MiB, exact minimum={_expertMinimumBytes / 1048576.0:F0} MiB, planned decode widths=1..{ExpertDecodeRows}, prefill rows up to {_prefill} (requested {_options.ResolveGlmPrefillChunkTokens()}); wide prefill protects its entire routed union; bounded source-file staging; full-model performance remains unqualified\n") +
        (_options.EnableAdaptiveGlmPrefill
            ? "[unified-memory] adaptive GLM prefill: actual prompt/joint shapes, full physical KV slots, immutable expert phase floor, fenced request-boundary cache reclaim; widths may decrease before scheduling under pressure\n" : "") +
        (_decodeGraphReserveRequested
            ? "[unified-memory] experimental decode graph reserve requested: eligible single-primary text decode keeps live device graphs plus one replacement forecast; host and physical admission remain enforced\n" : "") +
        (_layerMajorGraphCapBytes > 0
            ? $"[unified-memory] experimental strict layer-major graph payload cap: {_layerMajorGraphCapBytes >> 20} MiB per native generation plus separate 64 MiB forecast allowance; profile 1, context 2048/chunk 512, no ordinary fallback after strict admission; graph fit and performance require qualification\n" : "") +
        (_geometry.HostTokenEmbedding
            ? "[unified-memory] experimental host token embedding: exact Q4_K raw rows, original CUDA GET_ROWS; 340.3125 MiB input table excluded from CUDA weights, separate output retained; pageable immutable source, one accounted 1.125 MiB host pack and charged compact CUDA graph inputs; no resident-RAM or performance claim\n" : "") +
        (_vmmWeightsRequested
            ? $"[unified-memory] experimental VMM expert weights: profile 1; durable Host metadata {_vmmMetadataBytes} bytes, independent rounded Gate/Up/Down tickets; checked retirement before model cleanup; no performance or total allocator-memory claim\n" : "") +
        (_fileSourceViewRequested
            ? $"[unified-memory] experimental unlocked expert source views: profile 1; {_fileSourceViewPlan.Sources} exact registered sources; durable Host promise {_fileSourceViewPlan.TotalHostPromise} bytes = control {_fileSourceViewPlan.OwnedControlBytes} + OS mapping forecast {_fileSourceViewPlan.PteForecastBytes}; copies into existing staging, no pinning/private payload/residency or performance claim\n" : "") +
        (_causalRouteNeighborRequested
            ? $"[unified-memory] experimental causal route-neighbor: profile 1; one durable {_causalRouteNeighborHostBytes}-byte Host slab, completed 64-route history, horizon 8; unchanged expert union/math, zero additional Device payload; runtime and physical-read benefits require measurement\n" : "") +
        "[unified-memory] selected-expert file staging: bounded native read/upload overlap follows the phase policy; " +
        (Environment.GetEnvironmentVariable("TS_HOST_MOE_LAYER_PREFETCH") == "1"
            ? "experimental whole-layer prefetch requested; eligible layer-major calls require token and budget checks\n"
            : "whole-layer file/compute prefetch remains off\n") +
        (_geometry.ExpandedDecodeStaging
            ? "[unified-memory] experimental scalar GLM file staging requested: 72 MiB retained host arena for eligible top-8 misses; additional 40 MiB forecast, 32 MiB fallback on preallocation refusal\n" : "") +
        (_geometry.ExpandedPrefillRingStaging
            ? "[unified-memory] experimental wide GLM file ring requested: two 36 MiB slots in the shared 72 MiB host arena; one shared 40 MiB expansion forecast, 32 MiB ring on preallocation refusal\n" : "") +
        (_hostExperts != null
            ? $"[unified-memory] optional RAM expert cache: {_hostExperts.CapacityBytes / 1048576.0:F0} MiB admitted; lazy exact source payload, persistent host ownership, reclaimable at execution boundaries\n" : "") +
        (_options.EnableGlmFlashMtp
            ? "[unified-memory] experimental Flash MTP: one primary text request, context <=2048, at most two drafts, exact target verification and KDA rollback; persistent draft/snapshot ownership; numerical and performance qualification required\n"
            : "[unified-memory] MTP/draft: off\n") +
        "[unified-memory] off: host KV paging, TP, video/audio\n" +
        "[unified-memory] limits: mapping/page-cache residency, GC, native metadata and vendor/driver pools retain observed headroom; no complete OS RAM/VRAM hard-cap claim.";

    public static GlmUnifiedMemoryModelSession Create(string path, string? mmprojPath, BackendType backend,
        UnifiedMemoryModelOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        bool causalRouteNeighbor = GlmUnifiedCausalRouteNeighborPlan.Resolve(
            Environment.GetEnvironmentVariable(GlmUnifiedCausalRouteNeighborPlan.EnvironmentVariable));
        if (causalRouteNeighbor)
            options = GlmUnifiedCausalRouteNeighborPlan.ResolveContextOptions(options,
                Environment.GetEnvironmentVariable("MAX_CONTEXT"));
        bool deviceBaton = GlmUnifiedLayerMajorDeviceBatonPlan.Resolve(
            Environment.GetEnvironmentVariable(GlmUnifiedLayerMajorDeviceBatonPlan.EnvironmentVariable));
        if (deviceBaton)
            options = GlmUnifiedLayerMajorDeviceBatonPlan.ResolveContextOptions(options,
                Environment.GetEnvironmentVariable("MAX_CONTEXT"));
        bool vmmWeights = GgmlHostMoeExpertCacheScope.ResolveVmmWeightsMode(
            Environment.GetEnvironmentVariable(GgmlHostMoeExpertCacheScope.VmmWeightsEnvironmentVariable));
        bool fileSourceView = GgmlHostMoeExpertCacheScope.ResolveFileSourceViewMode(
            Environment.GetEnvironmentVariable(GgmlHostMoeExpertCacheScope.FileSourceViewEnvironmentVariable));
        bool hostEmbedding = GlmUnifiedHostTokenEmbeddingPlan.Resolve(
            Environment.GetEnvironmentVariable(GlmUnifiedHostTokenEmbeddingPlan.EnvironmentVariable));
        if (hostEmbedding)
        {
            if (!GlmUnifiedHostTokenEmbeddingPlan.ProfileOptions(options,
                !string.IsNullOrWhiteSpace(mmprojPath) && mmprojPath != "none"))
                throw new NotSupportedException("Host token embedding requires one text request, chunk at most 512, no MTP, retained or prefix copies.");
            GlmUnifiedHostTokenEmbeddingPlan.RequireSupport(true, GgmlGlmNative.HostTokenEmbeddingVersion);
        }
        long layerMajorGraphCap = GlmUnifiedLayerMajorGraphCapPlan.Resolve(
            Environment.GetEnvironmentVariable(GlmUnifiedLayerMajorGraphCapPlan.EnvironmentVariable));
        if (vmmWeights) ValidateVmmWeightsOptions(options, backend, hostEmbedding, layerMajorGraphCap,
            !string.IsNullOrWhiteSpace(mmprojPath) && mmprojPath != "none", Environment.GetEnvironmentVariable);
        if (fileSourceView) ValidateFileSourceViewOptions(options, backend, hostEmbedding, layerMajorGraphCap,
            !string.IsNullOrWhiteSpace(mmprojPath) && mmprojPath != "none", Environment.GetEnvironmentVariable);
        if (deviceBaton) {
            ValidateDeviceBatonOptions(options,backend,hostEmbedding,layerMajorGraphCap,
                !string.IsNullOrWhiteSpace(mmprojPath) && mmprojPath != "none",Environment.GetEnvironmentVariable);
            GgmlGlmNative.RequireLayerMajorDeviceBatonSupport();
        }
        if (causalRouteNeighbor)
            ValidateCausalRouteNeighborOptions(options,backend,hostEmbedding,layerMajorGraphCap,
                !string.IsNullOrWhiteSpace(mmprojPath) && mmprojPath != "none",Environment.GetEnvironmentVariable);
        long causalRouteNeighborHostBytes = causalRouteNeighbor
            ? GlmUnifiedCausalRouteNeighborPlan.ValidateHostPlan(GgmlGlmNative.CausalRouteNeighborHostPlan()) : 0;
        if (fileSourceView) GgmlGlmNative.RequireCheckedMappedResidencySupport();
        long vmmMetadataBytes = vmmWeights ? GgmlHostMoeExpertCacheScope.PlanVmmWeightsHostBytes() : 0;
        if (vmmWeights || fileSourceView || deviceBaton || causalRouteNeighbor) GgmlGlmNative.RequireCheckedFreeSupport();
        if (layerMajorGraphCap > 0)
        {
            if (!GlmUnifiedLayerMajorGraphCapPlan.ProfileOptions(options,
                !string.IsNullOrWhiteSpace(mmprojPath) && mmprojPath != "none"))
                throw new NotSupportedException("Strict layer-major graph caps require one text request, selected experts, chunk 512, layer-major execution, and no draft, retained or prefix copies.");
            GlmUnifiedLayerMajorGraphCapPlan.RequireCertificate(GgmlGlmNative.LayerMajorGraphCapVersion(), true);
        }
        if (backend != BackendType.GgmlCuda || options.MaximumRunningRequests is < 1 or > 8
            || options.PrefillChunkTokens is < 1 or > 512 || options.ContextTokens < 0
            || options.MaximumHostBytes <= 0 || options.MaximumDeviceBytes <= 0 || options.MaximumExpertCacheBytes < 0
            || options.MaximumHostExpertCacheBytes < 0
            || options.RetainedConversationCount is < 0 or > 8 || options.PrefixCheckpointCount is < 0 or > 8)
            throw new NotSupportedException("Unified GLM requires GGML CUDA and valid single-device request geometry.");
        if (Environment.GetEnvironmentVariable("TS_GLM_NATIVE") == "0"
            || (SpeculationOptions.FromEnvironment().Enabled && !options.EnableGlmFlashMtp))
            throw new NotSupportedException("Unified GLM requires its native, non-speculative executor.");
        if (options.EnableGlmFlashMtp)
        {
            ValidateFlashMtpLoadOptions(options, SpeculationOptions.FromEnvironment(), mmprojPath,
                Environment.GetEnvironmentVariable("TS_GLM_FLASH_MTP"));
            GgmlGlmNative.RequireFlashMtpSupport();
        }
        if (options.EnableAsyncExpertPrefill || (options.EnableGlmSelectedExperts && options.MaximumRunningRequests > 3))
            throw new NotSupportedException("Experimental unified GLM supports at most three selected-expert decode rows; the whole-layer asynchronous expert prefill option is not supported.");
        if (options.EnableAdaptiveGlmPrefill && !options.EnableGlmSelectedExperts)
            throw new NotSupportedException("Adaptive GLM prefill requires its complete selected-expert planning policy.");
        if (options.MaximumHostExpertCacheBytes > 0 && !options.EnableGlmSelectedExperts)
            throw new NotSupportedException("The optional GLM RAM expert cache requires selected-expert execution.");
        _ = options.ResolveGlmPrefillChunkTokens(); // Validate the separate GLM bound before native/model allocation.
        GgmlGlmNative.RequireUnifiedBudgetSupport();
        if (options.EnableGlmSelectedExperts) GgmlGlmNative.RequireSelectedExpertPlanningSupport();
        string? stagingOption = Environment.GetEnvironmentVariable(GlmUnifiedFileStagingPlan.EnvironmentVariable);
        bool expandedStaging = GlmUnifiedFileStagingPlan.Resolve(stagingOption,
            GgmlBasicOps.SupportsEnabledHostMoeDecodeStaging());
        bool expandedPrefillRing = GlmUnifiedFileStagingPlan.ResolvePrefillRing(
            Environment.GetEnvironmentVariable(GlmUnifiedFileStagingPlan.PrefillRingEnvironmentVariable),
            GgmlBasicOps.SupportsEnabledHostMoePrefillRingStaging());
        GlmUnifiedMemoryGeometry geometry;
        string? shapePlanOption = options.EnableGlmFlashMtp ? Environment.GetEnvironmentVariable("TS_GLM_FLASH_MTP_SHAPE_PLAN") : null;
        bool shapePlan = GlmUnifiedMtpDraftPlan.ResolveShapePlan(shapePlanOption,
            shapePlanOption == "1" && GgmlGlmNative.SupportsFlashMtpShapePlan());
        string? prefixCommitOption = options.EnableGlmFlashMtp ? Environment.GetEnvironmentVariable("TS_GLM_FLASH_MTP_PREFIX_COMMIT") : null;
        bool prefixCommit = GlmUnifiedMtpDraftPlan.ResolvePrefixCommit(prefixCommitOption,
            prefixCommitOption == "1" && GgmlGlmNative.SupportsFlashMtpPrefixCommit());
        using (var file = GgufFile.OpenForMemoryPlanning(path))
            geometry = GlmUnifiedMemoryGeometry.Read(file, options.EnableGlmFlashMtp, hostEmbedding);
        if (deviceBaton) geometry = geometry with { LayerMajorDeviceBaton = true };
        if (expandedStaging) geometry = geometry with { ExpandedDecodeStaging = true };
        if (expandedPrefillRing) geometry = geometry with { ExpandedPrefillRingStaging = true };
        if (shapePlan) geometry = geometry with { Mtp = geometry.Mtp! with { ShapeSpecificPlan = true } };
        if (prefixCommit) geometry = geometry with { Mtp = geometry.Mtp! with { PrefixStateCommit = true } };
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
        if (layerMajorGraphCap > 0 && !GlmUnifiedLayerMajorGraphCapPlan.ProfileGeometry(geometry, context))
            throw new NotSupportedException("Strict layer-major graph caps require the exact GLM5Next profile-1 geometry and context 2048.");
        if (vmmWeights && !GlmUnifiedLayerMajorGraphCapPlan.ProfileGeometry(geometry, context))
            throw new NotSupportedException("VMM expert weights require the exact GLM5Next profile-1 geometry and context 2048.");
        if (deviceBaton && !GlmUnifiedLayerMajorGraphCapPlan.ProfileGeometry(geometry, context))
            throw new NotSupportedException("Device baton requires the exact GLM5Next profile-1 geometry/context2048.");
        if (fileSourceView && !GlmUnifiedLayerMajorGraphCapPlan.ProfileGeometry(geometry, context))
            throw new NotSupportedException("Source views require the exact GLM5Next profile-1 geometry and context 2048.");
        if (causalRouteNeighbor && !GlmUnifiedLayerMajorGraphCapPlan.ProfileGeometry(geometry, context))
            throw new NotSupportedException("Causal route-neighbor requires the exact GLM5Next profile-1 geometry and context 2048.");
        var fileSourceViewPlan = fileSourceView ? GgmlHostMoeExpertCacheScope.PlanFileSourceViews(
            checked(3 * geometry.ExpertLayerLayouts.OfType<GlmUnifiedExpertLayout>().Count()), geometry.RawExpertSourceBytes) : default;
        if (options.EnableGlmFlashMtp && context > geometry.Mtp!.MaximumContext)
            throw new NotSupportedException("Experimental Flash MTP requires an explicit context at most " + geometry.Mtp.MaximumContext + ".");
        var owner = new GlmUnifiedMemoryModelSession(options, geometry, vision, InferenceHardwareMemory.CaptureCuda(), context,
            layerMajorGraphCap, vmmWeights, vmmMetadataBytes, fileSourceView, fileSourceViewPlan,
            causalRouteNeighbor, causalRouteNeighborHostBytes);
        try
        {
            if (options.EnableGlmFlashMtp && owner._expertCacheBytes <= 0)
                throw new NotSupportedException("Experimental Flash MTP requires the selected-expert cache for its host-routed draft block.");
            if (vmmWeights && (owner._prefill != 512 || owner._expertCacheBytes <= 0))
                throw new MemoryPressureException("VMM expert weights require an admitted 512-row selected-expert capacity plan.");
            if (fileSourceView && (owner._prefill != 512 || owner._expertCacheBytes <= 0))
                throw new MemoryPressureException("Source views require an admitted 512-row selected-expert capacity plan.");
            if (causalRouteNeighbor && (owner._prefill != 512 || owner._expertCacheBytes <= 0))
                throw new MemoryPressureException("Causal route-neighbor requires an admitted 512-row selected-expert capacity plan.");
            if (deviceBaton && (owner._prefill != 512 || owner._expertCacheBytes <= 0))
                throw new MemoryPressureException("Device baton requires admitted chunk512 and selected-expert capacity.");
            owner._native = new(owner.Budget, [[DevicePool], [HostPool]], includeGraphBuffers: true, hostPools: [HostPool],
                requireCheckedRelease: vmmWeights || fileSourceView || deviceBaton || causalRouteNeighbor);
            owner._host = new(owner.Budget, [HostPool]);
            if (vmmWeights)
                owner._vmmMetadataEnvelope = owner.Budget.Reserve([new(HostPool, vmmMetadataBytes)]);
            if (fileSourceView)
                owner._fileSourceViewEnvelope = owner.Budget.Reserve([new(HostPool, fileSourceViewPlan.TotalHostPromise)]);
            if (causalRouteNeighbor)
                owner._causalRouteNeighborEnvelope = owner.Budget.Reserve([new(HostPool, causalRouteNeighborHostBytes)]);
            owner._joint = owner.Budget.Reserve([new(HostPool, 0), new(DevicePool, 0)]);
            owner._native.SetSharedJointEnvelope(owner._joint);
            if (options.EnableGlmFlashMtp)
            {
                owner._speculativeEnvelope = owner.Budget.Reserve(Charges(new(0, owner.SpeculativeSnapshotBytes)));
                owner._native.SetSharedSpeculativeEnvelope(owner._speculativeEnvelope);
            }
            if (owner._expertCacheBytes > 0)
            {
                owner._expertEnvelope = owner.Budget.Reserve([new(DevicePool, owner._expertCacheBytes)]);
                owner._native.SetSharedExpertEnvelope(owner._expertEnvelope);
            }
            var policy = new ModelMemoryPolicy(context, owner._prefill)
            {
                OmitEmbeddedDraftWeights = !options.EnableGlmFlashMtp, EnableGlmFlashMtp = options.EnableGlmFlashMtp,
                EnableGlmHostTokenEmbedding = hostEmbedding, EnableGlmLayerMajorDeviceBaton = deviceBaton,
                EnableGlmCausalRouteNeighbor = causalRouteNeighbor,
                NativeMoeCpuLayers = owner._cpuLayers,
                DeviceExecutionReserveBytes = owner._executionReserveBytes
            };
            owner._model = (GlmDsaModel)ModelBase.Create(path, backend, 1, null!, null!, 1, null!, policy);
            if (vmmWeights || fileSourceView || deviceBaton || causalRouteNeighbor)
                owner._model.ConfigureUnifiedBeforeDispose(owner.PrepareCheckedExpertsForModelDispose);
            if (owner._model.UnifiedHostTokenEmbeddingCertified != geometry.HostTokenEmbedding)
                throw new NotSupportedException("The native GLM handle did not adopt the planned host token embedding mode; ordinary fallback is not authorized.");
            if (options.EnableGlmFlashMtp && !owner._model.HasDraftHead)
                throw new NotSupportedException("Experimental Flash MTP was requested but the native draft did not load; plain fallback is not authorized.");
            if (shapePlan && !owner._model.UnifiedMtpShapePlanEnabled)
                throw new NotSupportedException("The native Flash MTP model did not adopt the requested shape-specific plan.");
            owner._model.ConfigureUnifiedMtpPrefixCommit(prefixCommit);
            if (owner._model.MaxContextLength != context) throw new InvalidOperationException("Native GLM changed an explicit unified context.");
            owner._model.ConfigureUnifiedMemory(checked(options.RetainedConversationCount + owner._prefixCopies));
            if (owner._prefixCopies > 0)
                owner._model.ConfigureUnifiedPrefixCopies(geometry.SlotBytes(context), owner.EnterPrefixCapture,
                    owner.PrefixCloneSpareBytes);
            if (owner._expertCacheBytes > 0)
            {
                owner._expertGroups = owner._model.UnifiedExpertGroups;
                if (owner._expertGroups <= 0) throw new NotSupportedException("No complete mapped GLM expert layout groups were found.");
                // MaximumBudgetBytes permits later lending; it owns no ledger
                // credit. Keep the original startup envelope/current budget
                // before planning or allocating any selected-expert payload.
                // Assign first so failed initialization still disposes its scope.
                owner._experts = new(owner._expertMaximumBytes, owner._prefill, owner._expertGroups,
                    owner.ExpertDecodeRows, options.EnableAdaptiveGlmPrefill && options.EnableExpertCacheCapacityGrowth);
                if (owner._expertMaximumBytes != owner._expertCacheBytes)
                    owner._experts.UpdateBudget(owner._expertCacheBytes);
                if (vmmWeights)
                {
                    using (owner._native.EnterAllocationOwner(owner._vmmMetadataEnvelope!))
                        owner._experts.ConfigureVmmWeights(vmmMetadataBytes);
                }
                owner._expertMinimumBytes = owner._model.PlanUnifiedExpertMinimum(owner.ExpertDecodeRows);
                if (owner._expertCacheBytes < owner._expertMinimumBytes)
                    throw new MemoryPressureException($"GLM pooled expert budget {owner._expertCacheBytes} bytes cannot protect its complete admitted union/graph minimum {owner._expertMinimumBytes} bytes.");
                owner._model.ConfigureUnifiedSelectedExperts();
                if (causalRouteNeighbor)
                {
                    using (owner._native.EnterAllocationOwner(owner._causalRouteNeighborEnvelope!))
                        owner._model.ConfigureUnifiedCausalRouteNeighbor(owner._experts,causalRouteNeighborHostBytes);
                }
                if (fileSourceView)
                {
                    using (owner._native.EnterAllocationOwner(owner._fileSourceViewEnvelope!))
                        owner._experts.ConfigureFileSourceViews(fileSourceViewPlan);
                }
                if (options.EnableAdaptiveGlmPrefill && options.EnableExpertCacheCapacityGrowth)
                    owner._model.ConfigureUnifiedFreshPrimaryAdmission(owner.EnterFreshPrimaryAllocation);
                owner._model.TrimUnifiedMappedResidency(); // Advisory only; no imputed released-byte credit.
            }
            if (layerMajorGraphCap > 0)
            {
                if (owner._experts == null)
                    throw new NotSupportedException("Strict layer-major graph caps require the configured native selected-expert seam.");
                owner._model.ConfigureUnifiedLayerMajorGraphCap(layerMajorGraphCap);
            }
            owner._model.ConfigureUnifiedGraphAdmission(owner.PrepareGraph, owner._decodeGraphReserveRequested);
            if (!options.EnableGlmFlashMtp && Environment.GetEnvironmentVariable("TS_GLM_GRAPH_MEMORY_DIAGNOSTICS") == "1")
            {
                owner._native.GraphMemoryDiagnosticWriter = Console.Error.WriteLine;
                owner._model.UnifiedGraphMemoryDiagnosticContext = owner._native.SetGraphMemoryDiagnosticContext;
            }
            if (options.EnableGlmFlashMtp) owner._model.ConfigureUnifiedDraftAdmission(owner.PrepareDraftGraph);
            owner._model.ConfigureUnifiedPrimarySlotReuse(requestId => owner._primarySlotClaim.AllowsBinding(requestId, owner.HasNativeSlot));
            if (vision != null) owner._model.LoadVisionEncoder(mmprojPath!);
            owner._model.ConfigureUnifiedVisionAdmission(owner.EnterVisionPreparation);
            if (options.MaximumHostExpertCacheBytes > 0 && owner._experts != null)
            {
                owner._hostExpertEnvelope = owner.Budget.Reserve([new(HostPool, 0)]);
                owner._native.SetSharedHostExpertEnvelope(owner._hostExpertEnvelope);
                owner._hostExperts = new();
                owner.RecoverHostExpertCapacity();
            }
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

    internal static void ValidateCausalRouteNeighborOptions(UnifiedMemoryModelOptions options, BackendType backend,
        bool hostEmbedding, long graphCap, bool media, Func<string,string?> environment)
    {
        if (backend != BackendType.GgmlCuda || !hostEmbedding || graphCap != 0
            || !GlmUnifiedCausalRouteNeighborPlan.ProfileOptions(options,media))
            throw new NotSupportedException("Causal route-neighbor requires Host1, one text request, context2048/chunk512, selected experts, no retained/prefix/draft/RAM tier.");
        foreach (string flag in VmmWeightsDisabledFlags.Concat(["TS_HOST_MOE_VMM_WEIGHTS",
            "TS_GLM_LAYER_MAJOR_DEVICE_BATON", "TS_HOST_MOE_FILE_READ_SORT", "TS_HOST_MOE_FILE_SOURCE_VIEW_PREFETCH",
            "TS_GLM_GRAPH_MEMORY_DIAGNOSTICS", "TS_HOST_MOE_ROUTE_TRACE", "TS_DSV4_PERF"]))
            if (environment(flag) is not (null or "0"))
                throw new NotSupportedException("Causal route-neighbor requires " + flag + " disabled.");
        if (environment("TS_HOST_MOE_EXPERT_CACHE_LFU") != "0")
            throw new NotSupportedException("Causal route-neighbor requires explicit TS_HOST_MOE_EXPERT_CACHE_LFU=0.");
        if (!string.IsNullOrEmpty(environment("TS_GLM_TRACE")) || !string.IsNullOrEmpty(environment("TS_GLM_TRACE_DUMP_DIR")))
            throw new NotSupportedException("Causal route-neighbor requires graph tracing disabled.");
    }
    private void ValidateCausalRouteNeighborMode()
    {
        if (!_causalRouteNeighborRequested) return;
        GlmUnifiedCausalRouteNeighborPlan.ValidateMode(true,
            Environment.GetEnvironmentVariable(GlmUnifiedCausalRouteNeighborPlan.EnvironmentVariable));
        ValidateCausalRouteNeighborOptions(_options,BackendType.GgmlCuda,_geometry.HostTokenEmbedding,
            _layerMajorGraphCapBytes,_vision != null,Environment.GetEnvironmentVariable);
    }

    internal static void ValidateDeviceBatonOptions(UnifiedMemoryModelOptions options, BackendType backend,
        bool hostEmbedding, long graphCap, bool media, Func<string,string?> environment)
    {
        if (backend != BackendType.GgmlCuda || !hostEmbedding || graphCap != 0
            || !GlmUnifiedLayerMajorDeviceBatonPlan.ProfileOptions(options,media))
            throw new NotSupportedException("Device baton requires Host1, one text request, context2048/chunk512, selected experts, no retained/prefix/draft/RAM tier.");
        foreach (string flag in VmmWeightsDisabledFlags.Concat(["TS_HOST_MOE_VMM_WEIGHTS",
            "TS_HOST_MOE_FILE_READ_SORT", "TS_HOST_MOE_FILE_SOURCE_VIEW_PREFETCH"]))
            if (environment(flag) is not (null or "0"))
                throw new NotSupportedException("Device baton requires " + flag + " disabled.");
        if (!string.IsNullOrEmpty(environment("TS_GLM_TRACE")) || !string.IsNullOrEmpty(environment("TS_GLM_TRACE_DUMP_DIR")))
            throw new NotSupportedException("Device baton requires tracing disabled.");
    }
    private void ValidateDeviceBatonMode()
    {
        if (!_geometry.LayerMajorDeviceBaton) return;
        GlmUnifiedLayerMajorDeviceBatonPlan.ValidateMode(true,
            Environment.GetEnvironmentVariable(GlmUnifiedLayerMajorDeviceBatonPlan.EnvironmentVariable));
        ValidateDeviceBatonOptions(_options,BackendType.GgmlCuda,_geometry.HostTokenEmbedding,
            _layerMajorGraphCapBytes,_vision != null,Environment.GetEnvironmentVariable);
    }

    internal static void ValidateVmmWeightsOptions(UnifiedMemoryModelOptions options, BackendType backend,
        bool hostEmbedding, long graphCap, bool media, Func<string, string?> environment)
    {
        if (backend != BackendType.GgmlCuda || !hostEmbedding || graphCap != 0 || media
            || !options.EnableGlmSelectedExperts || !options.EnableAdaptiveGlmPrefill || options.EnableGlmFlashMtp
            || options.EnableAsyncExpertPrefill || options.MaximumRunningRequests != 1
            || options.RetainedConversationCount != 0 || options.PrefixCheckpointCount != 0
            || options.MaximumHostExpertCacheBytes != 0 || options.ResolveGlmPrefillChunkTokens() != 512
            || options.LayerMajorPrefillTokens <= 512)
            throw new NotSupportedException("VMM expert weights require Host token embedding, one selected-expert text request, chunk 512, and no graph cap, draft, prefix, retained or RAM expert tier.");
        if (environment("TS_HOST_MOE_DECODE_REBALANCE") != "1")
            throw new NotSupportedException("VMM expert weights require TS_HOST_MOE_DECODE_REBALANCE=1.");
        foreach (string flag in VmmWeightsDisabledFlags)
            if (environment(flag) is not (null or "0"))
                throw new NotSupportedException("VMM expert weights require " + flag + " disabled.");
        if (!string.IsNullOrEmpty(environment("TS_GLM_TRACE"))
            || !string.IsNullOrEmpty(environment("TS_GLM_TRACE_DUMP_DIR")))
            throw new NotSupportedException("VMM expert weights require graph tracing disabled.");
    }
    private static readonly string[] VmmWeightsDisabledFlags = [
        "TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER", "TS_HOST_MOE_PHASE_COLD_GROWTH",
        "TS_HOST_MOE_PREFILL_DOWN_OVERLAP", "TS_HOST_MOE_PREFILL_RING_COALESCE16",
        "TS_HOST_MOE_EXPERT_CACHE_PREFETCH", "TS_HOST_MOE_DECODE_PREFETCH", "TS_HOST_MOE_LAYER_PREFETCH",
        "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS", "TS_HOST_MOE_EXPERT_CACHE_LFU",
        "TS_HOST_MOE_RAM_CACHE_MB", "TS_HOST_MOE_RAM_CACHE_MAPPED", "TS_HOST_MOE_RAM_CACHE_VICTIM",
        "TS_HOST_MOE_RAM_CACHE_PINNED", "TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD", "TS_HOST_MOE_RAM_CACHE_FREQUENCY",
        "TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY", "TS_GLM_DECODE_GRAPH_RESERVE",
        "TS_HOST_MOE_FILE_READ_WORKERS", "TS_HOST_MOE_FILE_RANDOM_ACCESS", "TS_HOST_MOE_FILE_SOURCE_VIEW"
    ];
    private void ValidateVmmWeightsMode()
    {
        if (!_vmmWeightsRequested) return;
        if (!GgmlHostMoeExpertCacheScope.ResolveVmmWeightsMode(
                Environment.GetEnvironmentVariable(GgmlHostMoeExpertCacheScope.VmmWeightsEnvironmentVariable)))
            throw new InvalidOperationException("VMM weight mode changed after immutable session startup.");
        ValidateVmmWeightsOptions(_options, BackendType.GgmlCuda, _geometry.HostTokenEmbedding,
            _layerMajorGraphCapBytes, _vision != null, Environment.GetEnvironmentVariable);
        _experts!.ValidateVmmWeightsForExecution();
    }
    internal static void ValidateFileSourceViewOptions(UnifiedMemoryModelOptions options, BackendType backend,
        bool hostEmbedding, long graphCap, bool media, Func<string, string?> environment)
    {
        if (backend != BackendType.GgmlCuda || !hostEmbedding || graphCap != 0 || media
            || !options.EnableGlmSelectedExperts || !options.EnableAdaptiveGlmPrefill || options.EnableGlmFlashMtp
            || options.EnableAsyncExpertPrefill || options.MaximumRunningRequests != 1
            || options.RetainedConversationCount != 0 || options.PrefixCheckpointCount != 0
            || options.MaximumHostExpertCacheBytes != 0 || options.ResolveGlmPrefillChunkTokens() != 512
            || options.LayerMajorPrefillTokens <= 512)
            throw new NotSupportedException("Source views require Host token embedding, one selected-expert text request, chunk 512, and no graph cap, draft, prefix, retained or RAM expert tier.");
        if (environment("TS_HOST_MOE_DECODE_REBALANCE") != "1")
            throw new NotSupportedException("Source views require TS_HOST_MOE_DECODE_REBALANCE=1.");
        // Legacy cache LFU defaults ON when absent; this experiment requires
        // the original LRU policy explicitly, before any model allocation.
        if (environment("TS_HOST_MOE_EXPERT_CACHE_LFU") != "0")
            throw new NotSupportedException("Source views require explicit TS_HOST_MOE_EXPERT_CACHE_LFU=0.");
        foreach (string flag in VmmWeightsDisabledFlags.Where(f => f != GgmlHostMoeExpertCacheScope.FileSourceViewEnvironmentVariable)
            .Concat([GgmlHostMoeExpertCacheScope.VmmWeightsEnvironmentVariable, "TS_HOST_MOE_FILE_READ_SORT"]))
            if (environment(flag) is not (null or "0"))
                throw new NotSupportedException("Source views require " + flag + " disabled.");
        if (!string.IsNullOrEmpty(environment("TS_GLM_TRACE")) || !string.IsNullOrEmpty(environment("TS_GLM_TRACE_DUMP_DIR")))
            throw new NotSupportedException("Source views require graph tracing disabled.");
    }
    private void ValidateFileSourceViewMode()
    {
        if (!_fileSourceViewRequested) return;
        if (!GgmlHostMoeExpertCacheScope.ResolveFileSourceViewMode(
                Environment.GetEnvironmentVariable(GgmlHostMoeExpertCacheScope.FileSourceViewEnvironmentVariable)))
            throw new InvalidOperationException("Source-view mode changed after immutable session startup.");
        ValidateFileSourceViewOptions(_options, BackendType.GgmlCuda, _geometry.HostTokenEmbedding,
            _layerMajorGraphCapBytes, _vision != null, Environment.GetEnvironmentVariable);
        _experts!.ValidateFileSourceViewsForExecution();
    }
    private void PrepareCheckedExpertsForModelDispose()
    {
        // A failed drain/ACK must stop BEFORE an expert graph/source owner can
        // retire. The native FreeChecked export independently repeats this guard.
        if (_geometry.LayerMajorDeviceBaton) _model?.PrepareUnifiedDeviceBatonDispose();
        if (_causalRouteNeighborRequested) _model?.PrepareUnifiedCausalRouteNeighborDispose();
        _experts?.PrepareForModelDispose();
    }

    public SchedulerConfig ConfigureScheduler(SchedulerConfig requested)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((requested.Speculation.Enabled && !_options.EnableGlmFlashMtp) || requested.KvSnapshots != null || requested.MemoryAdmission != null)
            throw new NotSupportedException("Unified GLM cannot combine with independent speculation or snapshot/admission adapters.");
        _effectiveRetainedConversations = requested.EnablePrefixCaching ? _options.RetainedConversationCount : 0;
        if (_options.EnableGlmFlashMtp) return ConfigureFlashMtpScheduler(requested);
        int block = requested.BlockSize > 0 ? requested.BlockSize : SchedulerConfig.DefaultBlockSize;
        _effectivePrefixCopies = requested.EnablePrefixCaching
            ? Math.Min(_prefixCopies, (requested.ExecutionOptionsOverride ?? ExecutionOptions.FromEnvironment()).PrefixCheckpointBudget) : 0;
        return new SchedulerConfig
        {
            BlockSize = block, NumBlocks = checked((_context + block - 1) / block *
                (_options.MaximumRunningRequests + _options.RetainedConversationCount + _effectivePrefixCopies + 1)),
            MaxNumRunningSequences = _options.MaximumRunningRequests, MaxNumBatchedTokens = _forwardTokens,
            MaxPrefillChunkSize = _forwardTokens, SoloPrefillChunkSize = _forwardTokens, PrefillChunkTokenLimit = _forwardTokens,
            EnablePrefixCaching = requested.EnablePrefixCaching, StopRepetition = requested.StopRepetition,
            DecodeQuantumTokens = requested.DecodeQuantumTokens, Speculation = SpeculationOptions.Disabled,
            ExecutionOptionsOverride = new ExecutionOptions
            {
                BatchedPathDisabled = true, PerSeqFusedEnabled = true, BatchedFusedDecodeEnabled = true,
                RetainedFusedCacheBudget = requested.EnablePrefixCaching ? _options.RetainedConversationCount : 0,
                PrefixCheckpointBudget = _effectivePrefixCopies, KvHolderPoolMax = 0
            },
            MemoryAdmission = new RequestMemoryAdmission(Budget, EstimateRequest)
            {
                UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
                EstimateMaterializationPeak = EstimateMaterialization,
                BeforeAdmissionWithRunning = PrepareAdmission,
                AfterPrefixMaterialization = CompletePrefix, AfterPrefill = CompletePrefill,
                PrefillTokenLimit = ForwardPrefillLimit,
                AfterSequenceMemoryRelease = RecoverReleasedCapacity,
                EnterSerialExecution = EnterSerial,
                EnterBatchedExecution = _ => EnterAllocations(_joint
                    ?? throw new InvalidOperationException("GLM joint execution requires its shared graph envelope."),
                    clearGraphDiagnostics: true)
            }
        };
    }
    private InferenceMemoryBytes RequestPeak(SequenceState seq, bool materializing = false)
        => RequestPeakForRows(seq, AdmittedPrefillLimit(seq), materializing);
    private int AdmittedPrefillLimit(SequenceState seq)
        => _requestPrefill.TryGetValue(seq, out var state) ? state.Cap : _prefill;
    /// <summary>Tokens one prefill Forward may carry. A layer-major forward runs
    /// every chunk at the admitted width, one trunk layer at a time, so device
    /// peaks stay per chunk; only host residual streams widen. Media prompts and
    /// narrow admissions keep ordinary chunk order.</summary>
    private int ForwardPrefillLimit(SequenceState seq)
    {
        int rows = AdmittedPrefillLimit(seq);
        int limit = LayerMajor(seq, rows) ? _forwardTokens : rows;
        return _geometry.LayerMajorDeviceBaton ? Math.Min(limit,Math.Max(1,seq.PromptTokens.Count - seq.NumComputedTokens)) : limit;
    }
    // LayerMajorPrefillTokens = 0 disables it outright: an admission narrower
    // than the load-time width must then keep that narrower Forward.
    private bool LayerMajor(SequenceState seq, int rows) => _options.LayerMajorPrefillTokens > 0
        && seq.MediaSpans.Count == 0 && _forwardTokens > rows && rows >= Qwen4ExpModel.LayerMajorMinimumChunkTokens;
    private InferenceMemoryBytes RequestPeakForRows(SequenceState seq, int maximumRows, bool materializing)
    {
        if (_options.EnableGlmFlashMtp && seq.MediaSpans.Count != 0)
            throw new NotSupportedException("Experimental unified Flash MTP accepts text-only requests.");
        int tokens = checked(seq.PromptTokens.Count + seq.MaxNewTokens);
        if (tokens > _context) throw new MemoryPressureException("GLM request exceeds its unified context.");
        int rows = Math.Min(maximumRows, Math.Max(1, seq.PromptTokens.Count - seq.NumComputedTokens));
        var staging = _geometry.Workspace(Math.Max(1, seq.PromptTokens.Count), rows, outputRows: 1);
        // A Submit estimate always includes a new full slot. At the fenced
        // materialization boundary the model can reuse an actual idle primary
        // or donated holder. Existing child ownership is never revoked.
        // Copy materialization initially uses a new independent slot. Unlike a
        // donation it must not consume/reset a primary still advertised by the
        // radix tree. Its complete allocation is admitted before adoption.
        bool reusable = _effectivePrefixCopies > 0 ? HasNativeSlot(seq) : CanReuseSlotFor(seq);
        long slot = _options.EnableGlmFlashMtp || (materializing && reusable) ? 0 : _geometry.SlotBytes(_context);
        // Residual streams of the largest layer-major Forward still to run.
        int remaining = seq.PromptTokens.Count - seq.NumComputedTokens;
        long residual = remaining > rows && LayerMajor(seq, maximumRows)
            ? _geometry.LayerMajorHostBytes(Math.Min(_forwardTokens, remaining)) : 0;
        long speculationHost = _options.EnableGlmFlashMtp
            ? checked((32L << 20) + 4L * (8L * _geometry.Vocab + (2L * _forwardTokens + 16) * _geometry.Hidden)) : 0;
        return new(checked(staging.Host + residual + speculationHost), slot);
    }
    private bool HasNativeSlot(SequenceState seq) => _model!.CanReuseUnifiedSlot(seq.RequestId, includeIdlePrimary: false);
    private bool CanReuseSlotFor(SequenceState seq) => _primarySlotClaim.CanReuse(seq, HasNativeSlot(seq),
        _model!.CanReuseUnifiedSlot(seq.RequestId), HasNativeSlot);
    private void PublishPrimarySlotClaim(WeakReference<SequenceState> prepared)
    {
        if (!prepared.TryGetTarget(out var seq)) throw new InvalidOperationException("GLM materialization lost its live request.");
        if (!CanReuseSlotFor(seq)) return; // This request owns its complete new-slot allowance instead.
        _primarySlotClaim.PublishAfterMaterialization(prepared, HasNativeSlot(seq),
            _model!.CanReuseUnifiedSlot(seq.RequestId), HasNativeSlot);
    }
    private IReadOnlyList<MemoryCharge> EstimateRequest(SequenceState seq) => Charges(RequestPeak(seq));
    private IReadOnlyList<MemoryCharge> EstimateMaterialization(SequenceState seq) => Charges(RequestPeak(seq, true));
    private static MemoryCharge[] Charges(InferenceMemoryBytes bytes) => [new(HostPool, bytes.Host), new(DevicePool, bytes.Device)];
    private IReadOnlyList<MemoryCharge> PrepareAdmission(SequenceState seq, IReadOnlyList<SequenceState> running)
    {
        if (_options.EnableAdaptiveGlmPrefill) return PrepareAdaptiveAdmission(seq, running);
        if (_native!.CallbackError != null) throw new InvalidOperationException("GLM allocation accounting failed.", _native.CallbackError);
        var physical = ObservePhysicalConstraints();
        var limits = physical.Limits;
        var graphs = GlmUnifiedAdmissionGraphPlan.Create(_geometry, _context, _prefill,
            seq.PromptTokens.Count, seq.NumComputedTokens, running.Select(live => live.NumComputedTokens).ToArray(),
            GraphEntriesWithTransient);
        var target = PreserveReplacement(graphs.MaximumPeak);
        var needed = EstimateMaterialization(seq);
        MemoryCharge[] growth = [];
        bool? mappedTrimHint = null;
        bool FitsCombined()
        {
            var owned = _joint!.OwnedCharges.ToDictionary(c => c.Pool, c => c.Bytes);
            growth = Charges(target).Select(c => new MemoryCharge(c.Pool,
                checked(c.Bytes - owned[c.Pool] + needed.Single(n => n.Pool == c.Pool).Bytes))).ToArray();
            physical = physical with { Ledger = Budget.Snapshot() };
            physical = RepayHostExpertCapacity(physical, growth);
            return physical.Fits(growth);
        }
        MemoryPressureException Refusal(string message)
        {
            string Describe(IEnumerable<MemoryCharge> charges) => string.Join(", ", charges.Select(c =>
                FormattableString.Invariant($"{c.Pool}={c.Bytes}")));
            return new MemoryPressureException(message + " " + FormattableString.Invariant(
                $"request={seq.RequestId}; prompt={seq.PromptTokens.Count}; computed={seq.NumComputedTokens}; max-new={seq.MaxNewTokens}; logical-context={_context}; serial-graph-context={graphs.SerialContext}; joint-graph-context={graphs.JointContext}; prefill-rows={graphs.PrefillRows}; joint-rows={graphs.JointRows}; active-requests={running.Count}; joint-target=[{Describe(Charges(target))}]; joint-owned=[{Describe(_joint!.OwnedCharges)}]; joint-children=[{Describe(_joint.LiveChildCharges)}]; materialization=[{Describe(needed)}]; mapped-trim-hint={mappedTrimHint?.ToString() ?? "not-attempted"}; ")
                + physical.Describe(growth));
        }
        if (!FitsCombined())
        {
            _model!.TrimUnifiedGraphs(); _model.TrimUnifiedRetainedSlots();
            physical = ObservePhysicalConstraints(); limits = physical.Limits;
            _lastGraphShape = null;
            target = PreserveReplacement(graphs.MaximumPeak);
            needed = EstimateMaterialization(seq);
            if (!FitsCombined())
            {
                // Pageable expert scans can consume observed RAM without an
                // anonymous Host ticket. At this quiescent boundary hint only
                // those mappings, and accept no imputed released-byte credit.
                if (physical.TryRefreshMappedHostPressure(growth, _model.TrimUnifiedMappedResidency,
                    ObservePhysicalConstraints, out var refreshed, out mappedTrimHint))
                {
                    physical = refreshed; limits = physical.Limits;
                }
                if (!FitsCombined()) throw Refusal("GLM slot and shared graph peak exceed current RAM/commit/VRAM after idle graph/prefix reclamation.");
            }
        }
        // Combined capacity is checked before consuming any shared credit. The
        // scheduler then acquires the separate materialization envelope under
        // its compute lock; live shared children preserve their physical floor.
        if (!_joint!.TryResizeTotal(Charges(target), limits))
        {
            FitsCombined(); // Refresh the observed ledger if another owner changed it before the atomic reserve.
            throw Refusal("GLM shared graph admission could not reserve its replacement peak.");
        }
        return limits;
    }
    private void CompletePrefix(SequenceState seq)
    {
        // Allocate claim bookkeeping before native/ledger mutations. Admission
        // estimates do not claim anything: several waiting requests may sample
        // an idle primary, but only a successfully materialized first request
        // may promise its reuse before BindSequenceCache actually checks it out.
        var primaryClaim = GlmUnifiedPrimarySlotClaim.Prepare(seq);
        if (_options.EnableAdaptiveGlmPrefill)
        {
            var state = _requestPrefill.GetValue(seq, _ => new(_prefill));
            var running = state.Siblings.Select(reference => reference.TryGetTarget(out var live) ? live : null)
                .Where(live => live != null && !live.Status.IsFinished()).Cast<SequenceState>().ToArray();
            PrepareAdaptiveAdmission(seq, running, materialized: true);
            PublishPrimarySlotClaim(primaryClaim);
            RecoverExpertCapacity();
            return;
        }
        if (!seq.MemoryEnvelope!.TryResizeTotal(EstimateMaterialization(seq))) throw new MemoryPressureException("GLM prefix transition cannot preserve its request state/staging peak.");
        PublishPrimarySlotClaim(primaryClaim);
    }
    private void CompletePrefill(SequenceState seq)
    {
        // Only an acknowledged physical retirement may shorten the optional
        // recovery threshold. Keep the certificate unconsumed if admission fails.
        long observedCompletedCall = _batonObservedCompletedCall;
        long checkedFreedBank = _geometry.LayerMajorDeviceBaton
            ? GlmUnifiedLayerMajorDeviceBatonPlan.CompletedRelease(_model!.UnifiedDeviceBatonInfo,ref observedCompletedCall) : 0;
        long beforeJointDevice = Bytes(_joint!.OwnedCharges).Device;
        _model!.TrimUnifiedGraphs(onlyPrefill: true);
        _experts?.Trim(_experts.CachedBytes); // Return charged idle file staging; retain CUDA weights.
        _lastGraphShape = null;
        int context = PaddedContext(Math.Max(1, checked(seq.NumComputedTokens + 1)));
        var graph = WithDeviceBatonPromise(PreserveReplacement(WithLivePrefillPromises(
            _geometry.GraphPeak(context, 1, false, GraphEntriesWithTransient), except: seq)),except: seq);
        if (!Budget.TryResizeTotals([(_joint!, Charges(graph)), (seq.MemoryEnvelope!, EstimateMaterialization(seq))]))
            throw new MemoryPressureException("GLM prefill transition cannot preserve its shared graph and request state peak.");
        _batonObservedCompletedCall = observedCompletedCall;
        if (_options.EnableGlmFlashMtp) PrepareDraftGraph(context, 1);
        long releasedBankCredit = Math.Min(checkedFreedBank,Math.Max(0,beforeJointDevice - graph.Device));
        RecoverExpertCapacity(GlmUnifiedLayerMajorDeviceBatonPlan.RecoveryMinimum(releasedBankCredit,checkedFreedBank));
        if (checkedFreedBank > 0) Console.WriteLine(FormattableString.Invariant(
            $"[GLM-DEVICE-BATON-RECOVERY] checked_freed_bank_bytes={checkedFreedBank} released_owned_device_credit={releasedBankCredit} minimum_growth_bytes={GlmUnifiedLayerMajorDeviceBatonPlan.RecoveryMinimum(releasedBankCredit,checkedFreedBank)} expert_budget={_experts?.BudgetBytes ?? 0} scope=checked_payload_retired_fresh_physical_admission_required"));
        LogMtpMemory("prefill-complete", context, 1, 1, force: true, requestId: seq.RequestId);
    }

    private static InferenceMemoryBytes Bytes(IReadOnlyList<MemoryCharge> charges)
        => new(charges.Single(c => c.Pool == HostPool).Bytes, charges.Single(c => c.Pool == DevicePool).Bytes);
    private InferenceMemoryBytes WithLivePrefillPromises(InferenceMemoryBytes graph, SequenceState? except = null,
        bool replaceExecutingDevicePromise = false)
    {
        if (!_options.EnableAdaptiveGlmPrefill) return graph;
        foreach (var entry in _requestPrefill)
        {
            var sequence = entry.Key;
            if (ReferenceEquals(sequence, except) || sequence.MemoryEnvelope == null
                || sequence.Status.IsFinished() || sequence.NumComputedTokens >= sequence.PromptTokens.Count) continue;
            graph = GlmUnifiedLayerMajorGraphCapPlan.PreservePromise(graph, entry.Value.Peak,
                ReferenceEquals(sequence, _executingSequence), replaceExecutingDevicePromise);
        }
        return graph;
    }

    private long DeviceBatonPromise(SequenceState? proposed = null, int proposedRows = 0, SequenceState? except = null)
    {
        if (!_geometry.LayerMajorDeviceBaton) return 0;
        long result = 0;
        long For(SequenceState seq, int rows) => GlmUnifiedLayerMajorDeviceBatonPlan.Bytes(_geometry,
            Math.Min(_forwardTokens,Math.Max(0,seq.PromptTokens.Count - seq.NumComputedTokens)),rows,LayerMajor(seq,rows));
        foreach (var entry in _requestPrefill) {
            var seq = entry.Key;
            if (ReferenceEquals(seq,except) || ReferenceEquals(seq,proposed) || seq.MemoryEnvelope == null
                || seq.Status.IsFinished() || seq.NumComputedTokens >= seq.PromptTokens.Count) continue;
            result = Math.Max(result,For(seq,entry.Value.Cap));
        }
        if (proposed != null) result = Math.Max(result,For(proposed,proposedRows));
        return result; // N1 serial calls cannot own two banks concurrently.
    }
    private InferenceMemoryBytes WithDeviceBatonPromise(InferenceMemoryBytes graph, long explicitBytes = 0,
        SequenceState? proposed = null, int proposedRows = 0, SequenceState? except = null)
        => GlmUnifiedLayerMajorDeviceBatonPlan.AddOnce(graph,
            Math.Max(explicitBytes,DeviceBatonPromise(proposed,proposedRows,except)));

    private bool HasOutstandingPrefill()
    {
        if (_executingSequence is { } active && active.NumComputedTokens < active.PromptTokens.Count) return true;
        foreach (var entry in _requestPrefill)
        {
            var sequence = entry.Key;
            if (sequence.MemoryEnvelope != null && !sequence.Status.IsFinished()
                && sequence.NumComputedTokens < sequence.PromptTokens.Count) return true;
        }
        return false;
    }

    private IReadOnlyList<MemoryCharge> PrepareAdaptiveAdmission(SequenceState seq,
        IReadOnlyList<SequenceState> running, bool materialized = false)
    {
        if (_native!.CallbackError is { } callback) throw new InvalidOperationException("GLM allocation accounting failed.", callback);
        // Allocate all publication bookkeeping before native or ledger mutation.
        var state = _requestPrefill.GetValue(seq, _ => new(_prefill));
        var siblings = running.Select(live => new WeakReference<SequenceState>(live)).ToArray();
        var heads = running.Select(live => live.NumComputedTokens).ToArray();
        int maximum = Math.Min(_prefill, Math.Max(1, seq.PromptTokens.Count - seq.NumComputedTokens));
        var physical = ObservePhysicalConstraints();
        bool mayReclaimExperts = true;
        bool sourceViewPressureAttempted = false;
        GlmUnifiedPrefillCandidate selected = default;
        (InferenceMemoryBytes Joint, InferenceMemoryBytes Request) Estimate(int rows)
        {
            var graphs = GlmUnifiedAdmissionGraphPlan.Create(_geometry, _context, rows,
                seq.PromptTokens.Count, seq.NumComputedTokens, heads, GraphEntriesWithTransient);
            var target = WithDeviceBatonPromise(PreserveReplacement(WithLivePrefillPromises(graphs.MaximumPeak, except: seq)),
                proposed: seq,proposedRows: rows);
            var request = RequestPeakForRows(seq, rows, materializing: true);
            if (materialized) request = GlmUnifiedPrefillAdmission.Floor(request, Bytes(seq.MemoryEnvelope!.LiveChildCharges));
            return (target, request);
        }
        bool Select()
        {
            physical = physical with { Ledger = Budget.Snapshot() };
            if (_hostExperts != null)
            {
                var widest = Estimate(maximum);
                var ownedJoint = Bytes(_joint!.OwnedCharges);
                var ownedRequest = materialized ? Bytes(seq.MemoryEnvelope!.OwnedCharges) : default;
                physical = RepayHostExpertCapacity(physical, Charges(new(
                    checked(widest.Joint.Host - ownedJoint.Host + widest.Request.Host - ownedRequest.Host),
                    checked(widest.Joint.Device - ownedJoint.Device + widest.Request.Device - ownedRequest.Device))));
            }
            bool TryRows() => GlmUnifiedPrefillAdmission.TrySelect(physical.Ledger, physical.Limits,
                Bytes(_joint!.OwnedCharges), materialized ? Bytes(seq.MemoryEnvelope!.OwnedCharges) : default,
                _experts?.BudgetBytes ?? 0,
                mayReclaimExperts ? _expertMinimumBytes : _experts?.BudgetBytes ?? 0, maximum, Estimate, out selected,
                _geometry.LayerMajorDeviceBaton && seq.PromptTokens.Count - seq.NumComputedTokens > 512 ? 512 : 1);
            if (!TryRows()) return false;
            if (_fileSourceViewRequested && selected.Rows < maximum && !sourceViewPressureAttempted)
            {
                // A narrow candidate already fits, so the all-width refusal
                // fallback below would never hint mapped pages. Before keeping
                // that width, test the exact widest replacement/request delta.
                var widest = Estimate(maximum);
                var ownedJoint = Bytes(_joint!.OwnedCharges);
                var ownedRequest = materialized ? Bytes(seq.MemoryEnvelope!.OwnedCharges) : default;
                var growth = Charges(new(checked(widest.Joint.Host - ownedJoint.Host + widest.Request.Host - ownedRequest.Host),
                    checked(widest.Joint.Device - ownedJoint.Device + widest.Request.Device - ownedRequest.Device)));
                long reclaimable = mayReclaimExperts ? (_experts?.BudgetBytes ?? 0) - _expertMinimumBytes : 0;
                if (physical.TryRefreshNarrowSourceViewPrefill(true, selected.Rows, maximum, reclaimable,
                    growth, ref sourceViewPressureAttempted, _model!.TrimUnifiedMappedResidency,
                    ObservePhysicalConstraints, out var refreshed))
                {
                    physical = refreshed;
                    return TryRows(); // No inferred Host credit or changed width/floor policy.
                }
            }
            return true;
        }
        if (!Select())
        {
            _model!.TrimUnifiedGraphs(); _model.TrimUnifiedRetainedSlots();
            _experts?.Trim(_experts.CachedBytes);
            _lastGraphShape = null; physical = ObservePhysicalConstraints();
            if (!Select())
            {
                // This advisory hint has no imputed refund. Re-sample tracked
                // physical capacity before retrying the complete candidate set.
                var smallest = Estimate(1);
                var ownedJoint = Bytes(_joint!.OwnedCharges);
                var ownedRequest = materialized ? Bytes(seq.MemoryEnvelope!.OwnedCharges) : default;
                var minimumGrowth = new InferenceMemoryBytes(
                    checked(smallest.Joint.Host - ownedJoint.Host + smallest.Request.Host - ownedRequest.Host),
                    checked(smallest.Joint.Device - ownedJoint.Device + smallest.Request.Device - ownedRequest.Device));
                var charges = Charges(minimumGrowth);
                long reclaimableExperts = (_experts?.BudgetBytes ?? 0) - _expertMinimumBytes;
                if (physical.Shortfall(HostPool, charges) > 0 && physical.Shortfall(DevicePool, charges) <= reclaimableExperts
                    && (!_fileSourceViewRequested || !sourceViewPressureAttempted))
                {
                    if (_fileSourceViewRequested) sourceViewPressureAttempted = true;
                    _model.TrimUnifiedMappedResidency(); physical = ObservePhysicalConstraints();
                }
                if (!Select()) throw new MemoryPressureException(
                    $"GLM adaptive prefill cannot preserve its complete request, graph and expert phase floors: request={seq.RequestId}; prompt={seq.PromptTokens.Count}; logical-context={_context}; maximum-rows={maximum}; "
                    + physical.Describe(charges));
            }
        }
        if (GlmUnifiedPrefillAdmission.ShouldReclaimRetained(materialized, seq.NumComputedTokens,
            seq.PrefixCacheReusedTokens, selected.Rows, maximum))
        {
            // The completed lookup confirmed no usable state for this request.
            // Only idle terminal holders are eligible; active slots and every
            // pending prefill promise remain charged throughout the retry.
            _model!.TrimUnifiedRetainedSlots(terminalOnly: true); _lastGraphShape = null;
            physical = ObservePhysicalConstraints();
            if (!Select()) throw new MemoryPressureException("GLM adaptive prefill failed its fresh physical check after idle terminal-state reclamation.");
        }
        if (_experts != null && selected.ExpertTargetBytes < _experts.BudgetBytes)
        {
            long previous = _experts.BudgetBytes;
            UnifiedMemoryExpertCapacity.Resize(_experts, _expertEnvelope!, DevicePool, selected.ExpertTargetBytes,
                _expertMinimumBytes, physical.Limits);
            // Native release does not prove equivalent physical availability.
            physical = ObservePhysicalConstraints();
            mayReclaimExperts = false;
            if (!Select()) throw new MemoryPressureException("GLM adaptive prefill failed its fresh physical check after fenced expert reclamation.");
            Console.WriteLine($"[unified-memory] request {seq.RequestId}: fenced expert cache {previous / 1048576.0:F0} -> {_experts.BudgetBytes / 1048576.0:F0} MiB for actual prefill shape");
        }
        var admittedGraph = GlmUnifiedAdmissionGraphPlan.Create(_geometry, _context, selected.Rows,
            seq.PromptTokens.Count, seq.NumComputedTokens, heads, GraphEntriesWithTransient);
        var targets = new List<(BudgetReservation Envelope, IReadOnlyList<MemoryCharge> TotalTarget)>
            { (_joint!, Charges(selected.JointTarget)) };
        if (materialized) targets.Add((seq.MemoryEnvelope!, Charges(selected.RequestTarget)));
        // A new request's separate materialization envelope is acquired next by
        // the scheduler. Include its whole growth in this pre-transaction check.
        if (!materialized)
        {
            var owned = Bytes(_joint!.OwnedCharges);
            var growth = new InferenceMemoryBytes(checked(selected.JointTarget.Host - owned.Host + selected.RequestTarget.Host),
                checked(selected.JointTarget.Device - owned.Device + selected.RequestTarget.Device));
            physical = physical with { Ledger = Budget.Snapshot() };
            if (!physical.Fits(Charges(growth))) throw new MemoryPressureException("GLM adaptive prefill combined materialization peak changed before reservation.");
        }
        if (!Budget.TryResizeTotals(targets, physical.Limits, requirePhysicalFit: true))
            throw new MemoryPressureException("GLM adaptive prefill could not atomically admit its actual graph/request targets.");
        state.Cap = selected.Rows; state.Siblings = siblings;
        state.Peak = admittedGraph.PrefillPeak;
        if (selected.Rows < maximum)
            Console.WriteLine($"[unified-memory] request {seq.RequestId}: prefill {maximum} -> {selected.Rows} rows for actual prompt/joint shapes and immutable expert floor");
        return physical.Limits;
    }

    private void RecoverExpertCapacity(long minimumGrowthBytes = 512L << 20, bool reportDecision = false)
    {
        RecoverHostExpertCapacity();
        if (!_options.EnableAdaptiveGlmPrefill || _experts == null) return;
        if (_native!.CallbackError is { } callback) throw new InvalidOperationException("GLM optional expert recovery is disabled after an accounting failure.", callback);
        var physical = ObservePhysicalConstraints();
        long previous = _experts.BudgetBytes;
        long target = GlmUnifiedExpertLending.RecoveryTarget(physical, previous, _experts.MaximumBudgetBytes, minimumGrowthBytes);
        void Report(string reason)
        {
            if (!reportDecision) return;
            var device = physical.Ledger.Single(p => p.Pool == DevicePool);
            long effectiveLimit = Math.Min(device.Capacity, physical.Limits.Single(c => c.Pool == DevicePool).Bytes);
            long candidate = GlmUnifiedExpertLending.RecoveryTarget(physical, previous, _experts.MaximumBudgetBytes, 1);
            var usage = _experts.ReadUsage();
            Console.WriteLine(FormattableString.Invariant(
                $"[GLM-EXPERT-RECOVERY] request={_executingSequence?.RequestId ?? "-"} reason={reason} ceiling_before={previous} ceiling_candidate={candidate} ceiling_target={target} ceiling_after={_experts.BudgetBytes} ceiling_max={_experts.MaximumBudgetBytes} native_accounted={usage.AccountedBytes} native_budget={usage.BudgetBytes} physical_available={physical.Hardware.DeviceAvailable} physical_headroom={physical.DeviceHeadroom} ledger_committed={device.Committed} ledger_reserved={device.Reserved} ledger_capacity={device.Capacity} effective_limit={effectiveLimit} minimum_growth={minimumGrowthBytes} recovery_margin={128L << 20} sample=before_recovery unit=bytes scope=owned_credit_not_physical_free"));
        }
        if (target == previous)
        {
            if (reportDecision)
                Report(previous == _experts.MaximumBudgetBytes ? "already_max"
                    : GlmUnifiedExpertLending.RecoveryTarget(physical, previous, _experts.MaximumBudgetBytes, 1) == previous
                        ? "physical_or_ledger_slack" : "below_threshold");
            return;
        }
        bool resized = UnifiedMemoryExpertCapacity.Resize(_experts, _expertEnvelope!, DevicePool, target, _expertMinimumBytes, physical.Limits);
        if (_fileSourceViewRequested && !resized)
        {
            // The Device proposal can fit while the unchanged global Host
            // promise fails its all-pool physical check. Hint only that actual
            // refusal, once, without dropping the Host constraint or its owner.
            physical = physical with { Ledger = Budget.Snapshot() };
            long refusedTarget = target;
            decimal hostBefore = physical.Shortfall(HostPool, Charges(new(0, checked(target - previous))));
            if (GlmUnifiedExpertLending.TryRefreshSourceViewRecovery(true, resized, physical, previous, target,
                _experts.MaximumBudgetBytes, minimumGrowthBytes, _model!.TrimUnifiedMappedResidency,
                ObservePhysicalConstraints, out var refreshed, out var freshTarget, out var hinted))
            {
                physical = refreshed; target = freshTarget;
                if (target > previous)
                    resized = UnifiedMemoryExpertCapacity.Resize(_experts, _expertEnvelope!, DevicePool, target,
                        _expertMinimumBytes, physical.Limits);
                Console.WriteLine(FormattableString.Invariant(
                    $"[GLM-SOURCE-VIEW-EXPERT-RECOVERY] version=1 ceiling_before={previous} target_before={refusedTarget} target_after={target} ceiling_after={_experts.BudgetBytes} hint={(hinted == true ? 1 : 0)} resized={(resized ? 1 : 0)} host_shortfall_before={hostBefore} host_shortfall_after={physical.Shortfall(HostPool, Charges(new(0, checked(target - previous))))} fresh_host_available={physical.Hardware.HostAvailable} fresh_host_allocation_available={physical.Hardware.HostAllocationCapacity().Available} fresh_host_physical_limit={physical.Limits.Single(c => c.Pool == HostPool).Bytes} fresh_device_available={physical.Hardware.DeviceAvailable} fresh_device_physical_limit={physical.Limits.Single(c => c.Pool == DevicePool).Bytes} scope=checked_hint_fresh_physics_not_reclaimed_byte_credit"));
            }
        }
        if (resized)
            Console.WriteLine($"[unified-memory] released request workspace: expert-cache ceiling {previous / 1048576.0:F0} -> {target / 1048576.0:F0} MiB within live reservations; retained tables may grow inside the admitted capacity");
        Report(resized ? "increased" : "envelope_refused");
    }

    private void RecoverHostExpertCapacity()
    {
        if (_hostExperts == null) return;
        if (_native!.CallbackError is { } callback)
            throw new InvalidOperationException("RAM expert recovery is disabled after an accounting failure.", callback);
        var physical = ObservePhysicalConstraints();
        long previous = _hostExperts.CapacityBytes;
        long target = GlmUnifiedHostExpertCapacity.RecoveryTarget(physical, previous,
            Math.Min(_hostCeiling, _options.MaximumHostExpertCacheBytes));
        if (target != previous && GlmUnifiedHostExpertCapacity.Resize(_hostExperts, _hostExpertEnvelope!, target, physical.Limits))
            Console.WriteLine($"[unified-memory] RAM expert-cache capacity {previous / 1048576.0:F0} -> {target / 1048576.0:F0} MiB from observed unused host credit");
    }

    private GlmUnifiedPhysicalObservation RepayHostExpertCapacity(GlmUnifiedPhysicalObservation physical,
        IReadOnlyList<MemoryCharge> growth)
    {
        if (_hostExperts == null) return physical;
        if (_native!.CallbackError is { } callback)
            throw new InvalidOperationException("RAM expert repayment is disabled after an accounting failure.", callback);
        long current = _hostExperts.CapacityBytes;
        long target = GlmUnifiedHostExpertCapacity.RepaymentTarget(physical, growth, current);
        if (target == current) return physical;
        GlmUnifiedHostExpertCapacity.Resize(_hostExperts, _hostExpertEnvelope!, target, physical.Limits);
        return ObservePhysicalConstraints(); // Native refunds alone are not observed RAM availability.
    }

    private GlmUnifiedPhysicalObservation RepayExpertCapacity(GlmUnifiedPhysicalObservation physical,
        IReadOnlyList<MemoryCharge> growth, long minimum)
    {
        if (!_options.EnableAdaptiveGlmPrefill || !_options.EnableExpertCacheCapacityGrowth || _experts == null) return physical;
        if (_native!.CallbackError is { } callback)
            throw new InvalidOperationException("GLM expert repayment is disabled after an accounting failure.", callback);
        long current = _experts.BudgetBytes;
        long target = GlmUnifiedExpertLending.RepaymentTarget(physical, growth, current,
            Math.Min(current, Math.Max(_expertMinimumBytes, minimum)));
        if (target == current) return physical;
        UnifiedMemoryExpertCapacity.Resize(_experts, _expertEnvelope!, DevicePool, target,
            _expertMinimumBytes, physical.Limits);
        // Freed native tickets are not equivalent to available physical bytes.
        return ObservePhysicalConstraints();
    }

    private void RecoverReleasedCapacity(IReadOnlyList<SequenceState> running)
    {
        if (!_options.EnableAdaptiveGlmPrefill) { RecoverHostExpertCapacity(); LogHostExpertUsage(); return; }
        if (_native!.CallbackError is { } callback)
            throw new InvalidOperationException("GLM released-capacity recovery is disabled after an accounting failure.", callback);
        // The engine has fenced model release and disposed this request's root.
        // Invalidate only managed shape bookkeeping: native decode graphs may
        // remain reusable, but their unused large-prefill credit must not stay
        // hidden until a 256-token graph boundary happens to change the shape.
        _lastGraphShape = null;
        InferenceMemoryBytes PrefillPeak(SequenceState sequence)
            => _requestPrefill.TryGetValue(sequence, out var state) ? state.Peak : default;
        var plan = GlmUnifiedReleaseGraphPlan.Create(_geometry, _context, running, PrefillPeak,
            default, GraphEntriesWithTransient);
        if (plan.RunningRows == 0)
            _model!.TrimUnifiedGraphs(); // All graphs are idle; terminal KV slots stay live.
        else if (!plan.HasRunningPrefill)
            _model!.TrimUnifiedGraphs(onlyPrefill: true); // No surviving request needs a PP graph.
        plan = GlmUnifiedReleaseGraphPlan.Create(_geometry, _context, running, PrefillPeak,
            default, GraphEntriesWithTransient);
        var target = plan.RunningRows == 0
            ? GlmUnifiedPrefillAdmission.Floor(default, Bytes(_joint!.LiveChildCharges))
            : WithDeviceBatonPromise(PreserveReplacement(WithLivePrefillPromises(plan.Target)));
        if (!_joint!.TryResizeTotal(Charges(target)))
            throw new MemoryPressureException("GLM sequence release could not preserve the remaining graph and physical-child floors.");
        if (_speculativeEnvelope != null && plan.RunningRows == 0)
        {
            var children = Bytes(_speculativeEnvelope.LiveChildCharges);
            if (!_speculativeEnvelope.TryResizeTotal(Charges(new(children.Host,
                Math.Max(children.Device, SpeculativeSnapshotBytes)))))
                throw new MemoryPressureException("Flash MTP release could not preserve its persistent state.");
        }
        // Cache recovery sees the corrected shared ownership and a fresh
        // physical sample only after graph retirement/credit shrink complete.
        RecoverExpertCapacity();
        LogHostExpertUsage();
    }

    private void LogHostExpertUsage()
    {
        if (_hostExperts == null || Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS") != "1") return;
        var usage = _hostExperts.ReadUsage();
        if (_hostExperts.IsMapped) _ = _hostExperts.ReadMappedUsage();
        Console.WriteLine(FormattableString.Invariant(
            $"[HOSTMOE-RAM-CACHE] boundary=request-release capacity={usage.CapacityBytes} resident={usage.ResidentBytes} reserved={usage.ReservedBytes} hits={usage.Hits} misses={usage.Misses} hit_bytes={usage.HitBytes} load_bytes={usage.LoadBytes} bypass_bytes={usage.BypassBytes} evictions={usage.Evictions} scope=logical-source-payload"));
    }

    private int PaddedContext(int tokens)
        => Math.Min(_context, checked((Math.Max(1, tokens) + 255) / 256 * 256));
    private InferenceMemoryBytes PreserveReplacement(InferenceMemoryBytes twoGraphs)
    {
        // The native cache retains one graph while allocating its replacement.
        // An earlier wider PP graph can be larger than this next decode graph.
        return GlmUnifiedPrefillAdmission.PreserveReplacement(twoGraphs,
            Bytes(_joint!.LiveChildCharges), GraphEntriesWithTransient);
    }
    private void PrepareGraph(GlmUnifiedGraphShape shape)
    {
        ValidateVmmWeightsMode();
        ValidateFileSourceViewMode();
        ValidateDeviceBatonMode();
        ValidateCausalRouteNeighborMode();
        int rowLimit = shape.Batched ? _options.MaximumRunningRequests
            : _executingSequence is { } sequence ? AdmittedPrefillLimit(sequence) : _prefill;
        if (_options.EnableGlmFlashMtp && shape.OutputRows > 1)
            rowLimit = Math.Max(rowLimit, ExpertDecodeRows);
        if (shape.Rows <= 0 || shape.Rows > rowLimit || shape.ContextTokens <= 0 || shape.ContextTokens > _context)
            throw new MemoryPressureException("GLM graph shape exceeds the admitted execution geometry.");
        bool strictLayerMajor = GlmUnifiedLayerMajorGraphCapPlan.Eligible(shape, _layerMajorGraphCapBytes, _context,
            _options.MaximumRunningRequests == 1 && _effectiveRetainedConversations == 0 && _prefixCopies == 0
                && _vision == null && !_options.EnableGlmFlashMtp && _executingRequest != null
                && _executingSequence is { MediaSpans.Count: 0 });
        if (shape.StrictLayerMajor && !strictLayerMajor)
            throw new MemoryPressureException("GLM strict layer-major dispatch exceeds its certified execution profile.");
        shape = shape with { ContextTokens = PaddedContext(shape.ContextTokens) };
        bool boundedDecode = _decodeGraphReserveRequested && GlmUnifiedDecodeGraphPlan.Eligible(shape,
            _prefill, ExpertDecodeRows,
            _options.MaximumRunningRequests == 1 && _effectiveRetainedConversations == 0
                && _effectivePrefixCopies == 0 && _vision == null
                && _executingRequest != null && _executingSequence is { MediaSpans.Count: 0 },
            HasOutstandingPrefill());
        if ((boundedDecode || strictLayerMajor) && _native!.CallbackError is { } callback)
            throw new InvalidOperationException("GLM bounded decode admission is disabled after an allocation accounting failure.", callback);
        if (!GlmUnifiedDecodeGraphPlan.MustPrepare(shape, _lastGraphShape, boundedDecode, _lastGraphUsedBoundedDecode)
            && !GlmUnifiedLayerMajorGraphCapPlan.MustPrepare(shape, _lastGraphShape, strictLayerMajor,
                _lastGraphUsedStrictLayerMajor)) return;
        InferenceMemoryBytes Target()
        {
            var peak = _geometry.GraphPeak(shape.ContextTokens, shape.Rows, shape.Batched, GraphEntriesWithTransient,
                shape.OutputRows > 0 ? shape.OutputRows : null);
            var broad = WithDeviceBatonPromise(PreserveReplacement(WithLivePrefillPromises(peak)),
                GlmUnifiedLayerMajorDeviceBatonPlan.Bytes(_geometry,shape.SubmittedTokens,shape.Rows,shape.LayerMajor));
            if (strictLayerMajor)
            {
                // Keep every Host promise and every other request's Device
                // promise. Only this actual certified dispatch substitutes its
                // own broad prefill Device promise. Native children include
                // pending allocations and retain a full replacement allowance.
                var capped = GlmUnifiedLayerMajorGraphCapPlan.Target(broad, Bytes(_joint!.LiveChildCharges),
                    _layerMajorGraphCapBytes);
                return WithLivePrefillPromises(capped, replaceExecutingDevicePromise: true);
            }
            if (!boundedDecode) return broad;
            return GlmUnifiedDecodeGraphPlan.Target(broad, Bytes(_joint!.LiveChildCharges),
                _geometry.Workspace(shape.ContextTokens, shape.Rows, shape.OutputRows > 0 ? shape.OutputRows : 1), true);
        }
        // Re-evaluate live children even when the managed shape repeats. Native
        // keys also contain hidden-output and per-row KV extents, and a miss may
        // replace a graph after the previous admission used the same shape.
        var target = Target();
        bool Resize(IReadOnlyList<MemoryCharge>? limits = null) => _executingRequest is { } request
            ? Budget.TryResizeTotals([(_joint!, Charges(target)), (request, request.OwnedCharges)], limits)
            : _joint!.TryResizeTotal(Charges(target), limits);
        var owned = _joint!.OwnedCharges.ToDictionary(c => c.Pool, c => c.Bytes);
        bool grows = target.Host > owned[HostPool] || target.Device > owned[DevicePool];
        MemoryCharge[]? limits = grows ? PhysicalConstraints() : null;
        bool AdmitGraph()
        {
            var physical = ObservePhysicalConstraints();
            var current = Bytes(_joint.OwnedCharges);
            var growth = Charges(new(checked(target.Host - current.Host), checked(target.Device - current.Device)));
            physical = RepayHostExpertCapacity(physical, growth);
            physical = RepayExpertCapacity(physical, growth, _expertMinimumBytes);
            limits = physical.Limits;
            return Resize(limits);
        }
        if (!Resize(limits) && !AdmitGraph())
        {
            _model!.TrimUnifiedGraphs(); _model.TrimUnifiedRetainedSlots(); _lastGraphShape = null;
            target = Target();
            if (!AdmitGraph())
            {
                // File mappings have no anonymous payload ticket. Their pages
                // remain reclaimable, but a very large CPU expert scan can
                // consume currently observed RAM before its next graph grows.
                // Ask this quiescent model to release only those read-only
                // mapping pages, then authorize growth from a fresh observation.
                bool hinted = _model.TrimUnifiedMappedResidency();
                if (!AdmitGraph())
                {
                    string Describe(MemoryCharge charge) => $"{charge.Pool}={charge.Bytes}";
                    string ledger = string.Join(", ", Budget.Snapshot().Select(p =>
                        $"{p.Pool}: reserved={p.Reserved}, committed={p.Committed}, capacity={p.Capacity}"));
                    throw new MemoryPressureException($"GLM next graph cannot reserve its replacement peak before dispatch: " +
                        $"shape={shape}; target=[{string.Join(", ", Charges(target).Select(Describe))}]; " +
                        $"limits=[{string.Join(", ", limits!.Select(Describe))}]; {ledger}; mapped-trim-hint={hinted}.");
                }
            }
        }
        _lastGraphShape = shape;
        _lastGraphUsedBoundedDecode = boundedDecode;
        _lastGraphUsedStrictLayerMajor = strictLayerMajor;
        if (strictLayerMajor)
        {
            var state = _requestPrefill.GetValue(_executingSequence!, _ => new(_prefill));
            bool first = !state.StrictGraphRecoveryAttempted;
            state.StrictGraphRecoveryAttempted = true;
            long released = Math.Max(0, owned[DevicePool] - target.Device);
            if (released > 0)
                RecoverExpertCapacity(GlmUnifiedDecodeGraphPlan.RecoveryMinimum(released, first));
            Console.WriteLine(FormattableString.Invariant(
                $"[GLM-LAYER-MAJOR-GRAPH-CAP] profile=1 cap_bytes={_layerMajorGraphCapBytes} allowance_bytes={GlmUnifiedLayerMajorGraphCapPlan.GenerationAllowance} rows={shape.Rows} submitted={shape.SubmittedTokens} context={shape.ContextTokens} owned_before={owned[DevicePool]} owned_after={target.Device} live_device={Bytes(_joint!.LiveChildCharges).Device} released={released} expert_budget={_experts?.BudgetBytes ?? 0} scope=owned_device_credit_not_physical_free"));
        }
        if (boundedDecode)
        {
            var state = _requestPrefill.GetValue(_executingSequence!, _ => new(_prefill));
            bool first = !state.DecodeGraphRecoveryAttempted;
            state.DecodeGraphRecoveryAttempted = true;
            long released = Math.Max(0, owned[DevicePool] - target.Device);
            // This is a quiescent boundary before native dispatch. A smaller
            // recovery threshold is allowed only once per request, and only
            // after at least 64 MiB of graph credit was actually released.
            // Future growth still repays this optional loan before allocation.
            if (released > 0)
                RecoverExpertCapacity(GlmUnifiedDecodeGraphPlan.RecoveryMinimum(released, first),
                    first && Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS") == "1");
            if (first)
                Console.WriteLine(FormattableString.Invariant(
                    $"[GLM-DECODE-GRAPH-RESERVE] rows={shape.Rows} output_rows={shape.OutputRows} context={shape.ContextTokens} owned_before={owned[DevicePool]} owned_after={target.Device} live_device={Bytes(_joint!.LiveChildCharges).Device} released={released} expert_budget={_experts?.BudgetBytes ?? 0} scope=owned_device_credit_not_physical_free"));
        }
        LogMtpMemory("target-admitted", shape.ContextTokens, shape.Rows, shape.OutputRows);
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
            _executingSequence = seq;
            int rows = AdmittedPrefillLimit(seq);
            _model!.LayerMajorPrefillChunkTokens = LayerMajor(seq, rows) ? rows : 0;
            return execution;
        }
        catch { scope.Dispose(); throw; }
    }
    private sealed class SerialExecution(GlmUnifiedMemoryModelSession owner, IDisposable scope) : IDisposable
    {
        public void Dispose()
        {
            try { scope.Dispose(); }
            finally
            {
                owner._executingRequest = null; owner._executingSequence = null;
                owner._native?.ClearGraphMemoryDiagnosticContext();
                if (owner._model != null) owner._model.LayerMajorPrefillChunkTokens = 0;
            }
        }
    }
    private GlmUnifiedPhysicalObservation ObservePhysicalConstraints()
        => GlmUnifiedPhysicalObservation.Create(InferenceHardwareMemory.CaptureCuda(), Budget.Snapshot(),
            _hostCeiling, _deviceCeiling, _hostHeadroom, _deviceHeadroom);
    private MemoryCharge[] PhysicalConstraints() => ObservePhysicalConstraints().Limits;
    private long PrefixCloneSpareBytes()
        => _executingRequest?.Charges.Single(c => c.Pool == DevicePool).Bytes
            // Outside a materializing request this is a cache-policy hint.
            // Repayable loan credit must not make existing prefixes appear
            // unavailable. Every actual clone still needs its admitted owner.
            ?? checked(Budget.Snapshot().Single(p => p.Pool == DevicePool).Available
                + Math.Max(0, (_experts?.BudgetBytes ?? 0) - _expertCacheBytes));
    private IDisposable? EnterFreshPrimaryAllocation()
    {
        // Called under the model's native lock, after terminal-retention
        // prechecks and before the only new physical allocation. The completed
        // slot keeps its original owner. Do not trim retained slots or alter
        // request/graph promises while this ownership move is in progress.
        var charges = Charges(new(0, _geometry.SlotBytes(_context)));
        var physical = RepayExpertCapacity(ObservePhysicalConstraints(), charges, _expertCacheBytes);
        var envelope = OwnedAllocationScope.TryReserveOptional(Budget, charges, physical.Limits);
        if (envelope == null) return null;
        return OwnedAllocationScope.Enter(envelope, () => _native!.EnterAllocationOwner(envelope));
    }
    private IDisposable? EnterPrefixCapture(int tokens)
    {
        var sequence = _executingSequence;
        if (_effectivePrefixCopies <= 0 || sequence == null || tokens <= 0
            || !sequence.PublicCheckpointBoundaries.Contains(tokens)
            || _model!.RetainedCopiedSlotCount >= _effectivePrefixCopies + 1)
            return null;
        var request = _executingRequest ?? throw new InvalidOperationException("GLM prefix capture requires its executing request envelope.");
        // Prefix materialization runs before binding the actual native slot.
        // An idle primary may therefore have satisfied the slot forecast without
        // consuming this request's allowance. Settle that unused Device credit
        // now, preserving physical children and all Host/staging and joint-graph
        // promises. This is a ledger adjustment, not a physical memory refund.
        if (!HasNativeSlot(sequence)) return null;
        long previousDevice = request.OwnedCharges.Single(c => c.Pool == DevicePool).Bytes;
        bool settled = OwnedAllocationScope.ReleaseUnusedPoolReservation(request, DevicePool);
        long releasedDevice = previousDevice - request.OwnedCharges.Single(c => c.Pool == DevicePool).Bytes;
        if (settled && _options.EnableAdaptiveGlmPrefill && tokens < sequence.PromptTokens.Count)
        {
            // The completed forward has advanced the sequence before this
            // checkpoint. Its old wide-prefill promise no longer describes the
            // remaining tail. Reuse the ordinary atomic admission transaction,
            // retaining physical children, replacement graphs and every other
            // pending prefill promise. Snapshot the current admitted cohort:
            // siblings recorded at this request's start can omit newer decoders.
            if (sequence.NumComputedTokens != tokens)
                throw new InvalidOperationException("GLM checkpoint state does not match its admitted request head.");
            var siblings = UnifiedMemoryAdmissionCohort.SnapshotOtherAdmitted(
                _requestPrefill.Select(entry => entry.Key), sequence);
            try { _ = PrepareAdaptiveAdmission(sequence, siblings, materialized: true); }
            catch (MemoryPressureException) { return null; } // A discretionary cache must not reject this request.
        }
        var charges = Charges(new(0, _geometry.SlotBytes(_context)));
        var physical = ObservePhysicalConstraints();
        if (settled) physical = RepayExpertCapacity(physical, charges, _expertCacheBytes);
        var envelope = settled ? OwnedAllocationScope.TryReserveOptional(Budget, charges, physical.Limits) : null;
        try
        {
            if (System.Threading.Interlocked.Increment(ref _prefixCaptureDiagnostics) <= 8)
            {
                Console.WriteLine($"[unified-memory] public prefix capture: boundary={tokens}, prompt={sequence.PromptTokens.Count}, remaining={sequence.PromptTokens.Count - tokens}, " +
                    $"reservation={(envelope != null ? "admitted" : settled ? "unavailable" : "unsettled")}, unused-device-credit-released={releasedDevice}, " +
                    $"request-owned={Bytes(request.OwnedCharges)}, request-children={Bytes(request.LiveChildCharges)}, " +
                    $"joint-owned={Bytes(_joint!.OwnedCharges)}, joint-children={Bytes(_joint.LiveChildCharges)}, " +
                    $"expert-ceiling={_experts?.BudgetBytes ?? 0}; before-optional-admission: {physical.Describe(charges)}");
            }
            if (envelope == null) return null; // Optional cache misses must not reject the user request.
            // Closing the root refunds slack, while each actual native child
            // keeps its checkpoint charged until its physical buffer is freed.
            return OwnedAllocationScope.Enter(envelope, () => _native!.EnterAllocationOwner(envelope));
        }
        catch { envelope?.Dispose(); throw; }
    }
    private IDisposable EnterVisionPreparation(GlmVisionPreparationPlan plan)
    {
        var needed = Charges(new(plan.HostPeakBytes, plan.DevicePeakBytes));
        var physical = RepayHostExpertCapacity(ObservePhysicalConstraints(), needed);
        physical = RepayExpertCapacity(physical, needed, _expertCacheBytes);
        if (!physical.Fits(needed))
        {
            _model!.TrimUnifiedGraphs(); _model.TrimUnifiedRetainedSlots();
            physical = RepayHostExpertCapacity(ObservePhysicalConstraints(), needed);
            physical = RepayExpertCapacity(physical, needed, _expertCacheBytes);
        }
        var envelope = Budget.TryReserve(needed, physical.Limits) ?? throw new MemoryPressureException("GLM image preparation exceeds the shared budget.");
        try { return OwnedAllocationScope.Enter(envelope, () => EnterAllocations(envelope)); }
        catch { envelope.Dispose(); throw; }
    }
    private IDisposable EnterAllocations(BudgetReservation envelope, bool clearGraphDiagnostics = false)
    {
        IDisposable native = _native!.EnterExecution(envelope);
        IDisposable? host = null;
        try
        {
            host = _host!.EnterExecution(envelope);
            return new OwnedExecution(host, native, clearGraphDiagnostics ? _native : null);
        }
        catch { try { host?.Dispose(); } finally { native.Dispose(); } throw; }
    }
    private sealed class OwnedExecution(IDisposable first, IDisposable second,
        GgmlCacheBudgetScope? graphDiagnostics = null) : IDisposable
    {
        public void Dispose()
        {
            try { try { first.Dispose(); } finally { second.Dispose(); } }
            finally { graphDiagnostics?.ClearGraphMemoryDiagnosticContext(); }
        }
    }
    public void Dispose()
    {
        if (_disposed) return;
        if (_vmmWeightsRequested || _fileSourceViewRequested || _geometry.LayerMajorDeviceBaton || _causalRouteNeighborRequested)
        {
            if (_model != null) _model.PrepareUnifiedNativeDispose();
            else PrepareCheckedExpertsForModelDispose();
        }
        _model?.Dispose(); _model = null;
        _experts?.Dispose(); _experts = null;
        _hostExperts?.Dispose(); _hostExperts = null;
        _native?.SetSharedHostExpertEnvelope(null);
        _hostExpertEnvelope?.Dispose(); _hostExpertEnvelope = null;
        _native?.SetSharedExpertEnvelope(null);
        _native?.SetSharedSpeculativeEnvelope(null);
        _speculativeEnvelope?.Dispose(); _speculativeEnvelope = null;
        _expertEnvelope?.Dispose(); _expertEnvelope = null;
        _native?.SetSharedJointEnvelope(null);
        _joint?.Dispose(); _joint = null;
        _host?.Dispose(); _host = null;
        _vmmMetadataEnvelope?.Dispose(); _vmmMetadataEnvelope = null;
        _fileSourceViewEnvelope?.Dispose(); _fileSourceViewEnvelope = null;
        _causalRouteNeighborEnvelope?.Dispose(); _causalRouteNeighborEnvelope = null;
        _native?.Dispose(); _native = null;
        _disposed = true;
    }
}
