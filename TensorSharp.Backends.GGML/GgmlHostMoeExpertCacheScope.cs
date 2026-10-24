// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

internal static partial class GgmlNative
{
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeExpertCacheBudgetKind")]
    internal static partial int HostMoeExpertCacheBudgetKind();
    [LibraryImport(DllName, EntryPoint = "TSGgml_BeginHostMoeExpertCacheScope")]
    internal static partial int BeginHostMoeExpertCacheScope(long bytes, int maxRows, int layers, out ulong token);
    [LibraryImport(DllName, EntryPoint = "TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows")]
    internal static partial int BeginHostMoeExpertCacheScopeWithDecodeRows(long bytes, int maxRows, int layers,
        int maxDecodeRows, out ulong token);
    [LibraryImport(DllName, EntryPoint = "TSGgml_BeginHostMoeExpertCacheScopeWithCapacityGrowth")]
    internal static partial int BeginHostMoeExpertCacheScopeWithCapacityGrowth(long bytes, int maxRows, int layers,
        int maxDecodeRows, int enableCapacityGrowth, out ulong token);
    [LibraryImport(DllName, EntryPoint = "TSGgml_UpdateHostMoeExpertCacheScope")]
    internal static partial int UpdateHostMoeExpertCacheScope(ulong token, long bytes, out long released);
    [LibraryImport(DllName, EntryPoint = "TSGgml_TrimHostMoeExpertCacheScope")]
    internal static partial int TrimHostMoeExpertCacheScope(ulong token, long bytes, out long released);
    [LibraryImport(DllName, EntryPoint = "TSGgml_EndHostMoeExpertCacheScope")]
    internal static partial int EndHostMoeExpertCacheScope(ulong token, out long released);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeExpertCacheStats")]
    internal static partial int HostMoeExpertCacheStats(out long accounted, out long budget,
        out long hits, out long misses, out long calls);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeExpertCacheGrowthStats")]
    internal static unsafe partial int HostMoeExpertCacheGrowthStats(long* values, int length);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeExpertCacheEntryStats")]
    internal static unsafe partial int HostMoeExpertCacheEntryStats(long* values, int rowCapacity, out int count);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeVmmWeightsVersion")]
    internal static partial int HostMoeVmmWeightsVersion();
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeVmmWeightsHostPlan")]
    internal static partial int HostMoeVmmWeightsHostPlan(int profile, int cudaDevice, int rank, long maxChunks, out long hostBytes);
    [LibraryImport(DllName, EntryPoint = "TSGgml_ConfigureHostMoeVmmWeights")]
    internal static partial int ConfigureHostMoeVmmWeights(ulong token, int profile, int cudaDevice, int rank, long maxChunks);
    [LibraryImport(DllName, EntryPoint = "TSGgml_CloseHostMoeVmmWeights")]
    internal static partial int CloseHostMoeVmmWeights(ulong token);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeVmmWeightsStats")]
    internal static unsafe partial int HostMoeVmmWeightsStats(long* values, int length);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeFileSourceViewVersion")]
    internal static partial int HostMoeFileSourceViewVersion();
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeFileSourceViewEnabled")]
    internal static partial int HostMoeFileSourceViewEnabled();
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeFileSourceViewHostPlan")]
    internal static unsafe partial int HostMoeFileSourceViewHostPlan(int profile, int sources, long sourceBytes,
        long* values, int length);
    [LibraryImport(DllName, EntryPoint = "TSGgml_ConfigureHostMoeFileSourceView")]
    internal static partial int ConfigureHostMoeFileSourceView(ulong token, int profile, int sources,
        long sourceBytes, long totalHostPromise);
    [LibraryImport(DllName, EntryPoint = "TSGgml_CloseHostMoeFileSourceView")]
    internal static partial int CloseHostMoeFileSourceView(ulong token);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeFileSourceViewStats")]
    internal static unsafe partial int HostMoeFileSourceViewStats(ulong token, long* values, int length);
}

/// <summary>Process-wide compact expert-cache counters. AccountedBytes includes
/// its graph payload and fixed workspace allowance; it is not physical VRAM.
/// Host file-read staging and vendor/driver pools are excluded. Counters are
/// cumulative until the existing full cache teardown resets them.</summary>
public readonly record struct GgmlHostMoeExpertCacheUsage(long AccountedBytes, long BudgetBytes,
    long Hits, long Misses, long Calls);

/// <summary>One retained cache table. AccountedBytes includes the fixed
/// allowance. ResidentExperts counts valid payload slots, not all allocated
/// capacity; ExpertBytes is the three original quantized projections per slot.</summary>
public readonly record struct GgmlHostMoeExpertCacheEntryUsage(int Layer, int Capacity, int ResidentExperts,
    int PlannedDecodeRows, long AccountedBytes, int LocalExperts, int PooledExpertExtent, long ExpertBytes);

/// <summary>Cumulative growth diagnostics until full cache teardown. Peak bytes
/// count simultaneous old and new payload ownership plus the fixed allowance;
/// vendor/driver storage and whole-board VRAM are not included.</summary>
public readonly record struct GgmlHostMoeExpertCacheGrowthUsage(long Attempts, long Successes, long Refusals,
    long Failures, long UnchangedPlans, long RetainedHotSlots, long CopiedBytes, long PeakCoexistingAccountedBytes);

/// <summary>Native-held VMM owners and cumulative reshape counters. These are
/// not a managed ledger snapshot: failed checked acknowledgment may follow an
/// actual refund while the original native ticket remains held for retry.</summary>
public readonly record struct GgmlHostMoeVmmWeightsUsage(long Version, long Configured, long Closed,
    long DeclaredHostBytes, long HostTicketOwners, long LiveTables, long OwnedDeviceBytes,
    long PendingTicketBytes, long QuarantinedBytes, long ReshapeAttempts, long ReshapeCompleted, long ReshapeFailures);

/// <summary>Durable Host promise for unlocked source views. Only OwnedControlBytes
/// is a declared native payload; the PTE allowance is a conservative planning
/// forecast, not a measured OS bound, resident file payload or free-memory credit.</summary>
public readonly record struct GgmlHostMoeFileSourceViewPlan(long Version, long TotalHostPromise,
    long OwnedControlBytes, long PteForecastBytes, long Sources, long SourceBytes, long PageBytes, long MapGranularity);

/// <summary>Native view/control ownership. HeldControlTicketBytes can remain
/// after physical cleanup or even ledger refund until checked acknowledgment.</summary>
public readonly record struct GgmlHostMoeFileSourceViewUsage(long Version, long Configured, long Closing, long Closed,
    long OwnedRecords, long LiveViews, long ActiveReaders, long MappedVaBytes, long OwnedControlBytes,
    long HeldControlTicketBytes, long PendingRecords, long QuarantinedRecords, long DisabledRecords,
    long MapCalls, long ViewCloseCalls, long TotalHostPromise);

public partial class GgmlBasicOps
{
    public static void RequireSharedExpertCacheBudgetKind()
    {
        try
        {
            if (GgmlNative.HostMoeExpertCacheBudgetKind() != 4)
                throw new NotSupportedException("Unified concurrent execution requires dedicated selected-expert budget ownership (kind 4).");
        }
        catch (EntryPointNotFoundException error)
        {
            throw new NotSupportedException("Unified concurrent execution requires the matching GgmlOps native library with shared expert-cache ownership.", error);
        }
    }
    public static GgmlHostMoeExpertCacheUsage GetHostMoeExpertCacheUsage()
    {
        if (GgmlNative.HostMoeExpertCacheStats(out long accounted, out long budget,
                out long hits, out long misses, out long calls) != 1)
            throw new InvalidOperationException("GGML expert-cache usage read failed: " + GgmlNative.LastNativeError());
        return new(accounted, budget, hits, misses, calls);
    }

    public static unsafe GgmlHostMoeExpertCacheGrowthUsage GetHostMoeExpertCacheGrowthUsage()
    {
        Span<long> values = stackalloc long[8];
        try
        {
            fixed (long* pointer = values)
                if (GgmlNative.HostMoeExpertCacheGrowthStats(pointer, values.Length) != 1)
                    throw new InvalidOperationException("Expert-cache growth usage read failed: " + GgmlNative.LastNativeError());
        }
        catch (EntryPointNotFoundException error)
        { throw new NotSupportedException("Expert-cache growth diagnostics require the matching native library.", error); }
        return new(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
    }

    /// <summary>Diagnostic snapshot, independent of retained weight ownership.
    /// Prefer a quiescent boundary. A concurrent entry-count change retries at
    /// most three times; these reads are not enabled in normal token execution.</summary>
    public static unsafe GgmlHostMoeExpertCacheEntryUsage[] GetHostMoeExpertCacheEntries()
    {
        try
        {
            if (GgmlNative.HostMoeExpertCacheEntryStats(null, 0, out int count) != 1)
                throw new InvalidOperationException("Expert-cache entry count read failed: " + GgmlNative.LastNativeError());
            for (int attempt = 0; attempt < 3; ++attempt)
            {
                if (count == 0) return [];
                if (count is < 0 or > 1048576) throw new InvalidOperationException("Expert-cache entry metadata exceeds its diagnostic bound.");
                int capacity = count;
                long[] values = new long[checked(capacity * 8)];
                int status;
                fixed (long* pointer = values)
                    status = GgmlNative.HostMoeExpertCacheEntryStats(pointer, capacity, out count);
                if (status == 2) continue;
                if (status != 1 || count < 0 || count > capacity)
                    throw new InvalidOperationException("Expert-cache entry read failed: " + GgmlNative.LastNativeError());
                var entries = new GgmlHostMoeExpertCacheEntryUsage[count];
                for (int row = 0; row < entries.Length; ++row)
                {
                    int offset = row * 8;
                    entries[row] = new(checked((int)values[offset]), checked((int)values[offset + 1]),
                        checked((int)values[offset + 2]), checked((int)values[offset + 3]), values[offset + 4],
                        checked((int)values[offset + 5]), checked((int)values[offset + 6]), values[offset + 7]);
                }
                return entries;
            }
            throw new InvalidOperationException("Expert-cache entry count kept changing during the diagnostic snapshot.");
        }
        catch (EntryPointNotFoundException error)
        { throw new NotSupportedException("Expert-cache entry diagnostics require the matching native library.", error); }
    }
}

internal interface IHostMoeExpertCacheScopeNative
{
    ulong Begin(long bytes, int maxRows, int layers, int maxDecodeRows);
    ulong Begin(long bytes, int maxRows, int layers, int maxDecodeRows, bool enableCapacityGrowth)
    {
        if (enableCapacityGrowth) throw new NotSupportedException("The native expert-cache scope does not support capacity growth.");
        return Begin(bytes, maxRows, layers, maxDecodeRows);
    }
    long Update(ulong token, long bytes);
    long Trim(ulong token, long bytes);
    long End(ulong token);
    GgmlHostMoeExpertCacheUsage ReadUsage();
    int VmmWeightsVersion() => throw new NotSupportedException("The native scope does not support VMM weights.");
    long PlanVmmHostBytes() => throw new NotSupportedException("The native scope does not support VMM weights.");
    void ConfigureVmmWeights(ulong token) => throw new NotSupportedException("The native scope does not support VMM weights.");
    void CloseVmmWeights(ulong token) => throw new NotSupportedException("The native scope does not support VMM weights.");
    GgmlHostMoeVmmWeightsUsage ReadVmmUsage() => throw new NotSupportedException("The native scope does not support VMM weights.");
    int FileSourceViewVersion() => throw new NotSupportedException("The native scope does not support source views.");
    int FileSourceViewEnabled() => throw new NotSupportedException("The native scope does not support source views.");
    GgmlHostMoeFileSourceViewPlan PlanFileSourceViews(int sources, long sourceBytes)
        => throw new NotSupportedException("The native scope does not support source views.");
    void ConfigureFileSourceViews(ulong token, int sources, long sourceBytes, long totalHostPromise)
        => throw new NotSupportedException("The native scope does not support source views.");
    void CloseFileSourceViews(ulong token) => throw new NotSupportedException("The native scope does not support source views.");
    GgmlHostMoeFileSourceViewUsage ReadFileSourceViewUsage(ulong token)
        => throw new NotSupportedException("The native scope does not support source views.");
}

/// <summary>Exclusive, process-wide configuration of the optional Qwen4Exp/GLM CUDA
/// compact expert cache. This changes no environment variable or model placement.
/// Install the shared budget first and create this scope before loading the model,
/// with the expert cache empty. Quiesce all model work before changing or releasing
/// this scope. Dispose after model cleanup and before detaching its shared budget.
/// The cache's existing allocation tickets remain active; this is a capacity
/// setting, not a complete VRAM/RAM or driver-memory cap. No finalizer performs GPU
/// work. Failed disposal retains the native token and may be retried.</summary>
public sealed class GgmlHostMoeExpertCacheScope : IDisposable
{
    private readonly object _gate = new();
    private readonly IHostMoeExpertCacheScopeNative _native;
    private readonly ulong _token;
    private long _budgetBytes;
    private bool _disposed;
    private bool _vmmAttempted, _vmmConfigured, _vmmClosed;
    private bool _fileSourceViewAttempted, _fileSourceViewConfigured, _fileSourceViewClosing, _fileSourceViewClosed;
    public const string VmmWeightsEnvironmentVariable = "TS_HOST_MOE_VMM_WEIGHTS";
    public const string FileSourceViewEnvironmentVariable = "TS_HOST_MOE_FILE_SOURCE_VIEW";
    private const int VmmProfile = 1;
    private const long VmmMaximumChunks = 8192;

    /// <summary>Bind a model-owned causal metadata slab under its durable Host
    /// allocation owner. The exclusive native scope token remains private.</summary>
    public void ConfigureCausalRouteNeighbor(IntPtr model, long expectedHostBytes)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (model == IntPtr.Zero || expectedHostBytes <= 0 || expectedHostBytes > (1L << 20)
                || MaxPrefillRows != 512 || MaxDecodeRows != 1 || LayerCount != 3
                || _vmmAttempted || _fileSourceViewAttempted)
                throw new NotSupportedException("Causal route-neighbor requires the standalone scalar GLM profile and its admitted Host slab.");
            GgmlGlmNative.ConfigureCausalRouteNeighbor(model, _token, expectedHostBytes);
        }
    }

    public static bool ResolveVmmWeightsMode(string? value) => value switch
    {
        null or "0" => false,
        "1" => true,
        _ => throw new ArgumentException(VmmWeightsEnvironmentVariable + " must be unset, 0 or 1.")
    };

    public static bool ResolveFileSourceViewMode(string? value) => value switch
    {
        null or "0" => false,
        "1" => true,
        _ => throw new ArgumentException(FileSourceViewEnvironmentVariable + " must be unset, 0 or 1.")
    };

    public static GgmlHostMoeFileSourceViewPlan PlanFileSourceViews(int sources, long sourceBytes)
        => PlanFileSourceViews(NativeApi.Instance, sources, sourceBytes);
    /// <summary>Mandatory even for a direct native model load: an older DLL
    /// must not silently ignore the requested new source service.</summary>
    public static void RequireFileSourceViewSupport() => RequireFileSourceViewSupport(NativeApi.Instance);
    internal static void RequireFileSourceViewSupport(IHostMoeExpertCacheScopeNative native)
    {
        ArgumentNullException.ThrowIfNull(native);
        if (native.FileSourceViewVersion() != 1 || native.FileSourceViewEnabled() != 1)
            throw new NotSupportedException("Source views require the requested native profile-1 service.");
    }
    internal static GgmlHostMoeFileSourceViewPlan PlanFileSourceViews(IHostMoeExpertCacheScopeNative native,
        int sources, long sourceBytes)
    {
        ArgumentNullException.ThrowIfNull(native);
        if (sources != 126 || sourceBytes <= 0)
            throw new NotSupportedException("Source views require the exact 126-source GLM profile.");
        RequireFileSourceViewSupport(native);
        var plan = native.PlanFileSourceViews(sources, sourceBytes);
        if (plan.Version != 1 || plan.Sources != sources || plan.SourceBytes != sourceBytes
            || plan.OwnedControlBytes <= 0 || plan.PteForecastBytes <= 0
            || plan.OwnedControlBytes > long.MaxValue - plan.PteForecastBytes
            || plan.TotalHostPromise != plan.OwnedControlBytes + plan.PteForecastBytes
            || plan.PageBytes <= 0 || (plan.PageBytes & (plan.PageBytes - 1)) != 0
            || plan.MapGranularity < plan.PageBytes || (plan.MapGranularity & (plan.MapGranularity - 1)) != 0
            || plan.MapGranularity % plan.PageBytes != 0)
            throw new InvalidOperationException("Native source-view Host plan is invalid.");
        return plan;
    }

    /// <summary>Invoke under a dedicated session-lifetime Host owner after the
    /// model registered its exact selected sources and before any compute.</summary>
    public void ConfigureFileSourceViews(GgmlHostMoeFileSourceViewPlan plan)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_fileSourceViewAttempted) throw new InvalidOperationException("Source-view configuration was already attempted.");
            if (MaxPrefillRows != 512 || MaxDecodeRows != 1 || LayerCount != 3 || _vmmAttempted
                || PlanFileSourceViews(_native, checked((int)plan.Sources), plan.SourceBytes) != plan)
                throw new NotSupportedException("Source views require the exact standalone profile and admitted Host plan.");
            _fileSourceViewAttempted = true; // A failed native call can retain owners.
            _native.ConfigureFileSourceViews(_token, checked((int)plan.Sources), plan.SourceBytes, plan.TotalHostPromise);
            var usage = ReadFileSourceViewUsageCore();
            if (usage.Configured != 1 || usage.Closing != 0 || usage.Closed != 0
                || usage.OwnedRecords != plan.Sources || usage.LiveViews != plan.Sources
                || usage.MappedVaBytes < plan.SourceBytes || usage.MapCalls < plan.Sources
                || usage.ActiveReaders != 0 || usage.OwnedControlBytes != plan.OwnedControlBytes
                || usage.HeldControlTicketBytes != plan.OwnedControlBytes || usage.PendingRecords != 0
                || usage.QuarantinedRecords != 0 || usage.DisabledRecords != 0
                || usage.TotalHostPromise != plan.TotalHostPromise)
                throw new InvalidOperationException("Native source views did not certify complete quiescent ownership.");
            _fileSourceViewConfigured = true;
        }
    }

    public void ValidateFileSourceViewsForExecution()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_fileSourceViewConfigured || _fileSourceViewClosing || _fileSourceViewClosed)
                throw new InvalidOperationException("Source views are not configured for execution.");
        }
    }

    public GgmlHostMoeFileSourceViewUsage ReadFileSourceViewUsage()
    {
        lock (_gate) { ObjectDisposedException.ThrowIf(_disposed, this); return ReadFileSourceViewUsageCore(); }
    }
    private GgmlHostMoeFileSourceViewUsage ReadFileSourceViewUsageCore()
    {
        var v = _native.ReadFileSourceViewUsage(_token);
        // The bounded cleanup shell survives the record slab. Its one retained
        // Host owner can represent failed slab cleanup or an ACK-only ticket.
        bool shellQuarantine = v.QuarantinedRecords == 1 && v.OwnedRecords == 0
            && v.Closing == 1 && v.Closed == 0
            && (v.OwnedControlBytes > 0 || v.HeldControlTicketBytes > 0);
        if (v.Version != 1 || v.Configured is < 0 or > 1 || v.Closing is < 0 or > 1 || v.Closed is < 0 or > 1
            || v.OwnedRecords is < 0 or > 128 || v.LiveViews < 0 || v.LiveViews > v.OwnedRecords
            || v.ActiveReaders < 0 || v.MappedVaBytes < 0 || v.OwnedControlBytes < 0 || v.HeldControlTicketBytes < 0
            || v.PendingRecords < 0 || v.PendingRecords > v.OwnedRecords
            || v.QuarantinedRecords < 0 || (v.QuarantinedRecords > v.OwnedRecords && !shellQuarantine)
            || v.DisabledRecords < 0 || v.DisabledRecords > v.OwnedRecords
            || v.MapCalls < 0 || v.ViewCloseCalls < 0 || v.TotalHostPromise < 0)
            throw new InvalidOperationException("Native source-view ownership snapshot is invalid.");
        return v;
    }

    private void CloseFileSourceViewsCore()
    {
        if (!_fileSourceViewAttempted || _fileSourceViewClosed) return;
        _fileSourceViewClosing = true; // A failed partial close must never resume execution.
        _native.CloseFileSourceViews(_token);
        var v = ReadFileSourceViewUsageCore();
        if ((v.Configured == 1 && v.Closed != 1) || v.OwnedRecords != 0 || v.LiveViews != 0 || v.ActiveReaders != 0
            || v.MappedVaBytes != 0 || v.OwnedControlBytes != 0 || v.HeldControlTicketBytes != 0
            || v.PendingRecords != 0 || v.QuarantinedRecords != 0 || v.DisabledRecords != 0)
            throw new InvalidOperationException("Native source views still own resources or unacknowledged tickets.");
        _fileSourceViewClosed = true;
    }

    /// <summary>Read the combined adapter/cache durable Host allowance before
    /// admitting its root envelope. This reserves no payload or cache scope.</summary>
    public static long PlanVmmWeightsHostBytes() => PlanVmmWeightsHostBytes(NativeApi.Instance);
    internal static long PlanVmmWeightsHostBytes(IHostMoeExpertCacheScopeNative native)
    {
        ArgumentNullException.ThrowIfNull(native);
        if (native.VmmWeightsVersion() != VmmProfile)
            throw new NotSupportedException("VMM expert weights require native profile 1.");
        long bytes = native.PlanVmmHostBytes();
        if (bytes <= 0) throw new InvalidOperationException("The native VMM durable Host plan is invalid.");
        return bytes;
    }

    /// <summary>Caller holds a dedicated session-lifetime Host allocation owner,
    /// with checked-release accounting attached. Assign this scope before calling:
    /// even failed configuration can own quarantine that requires checked close.</summary>
    public void ConfigureVmmWeights(long declaredHostBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(declaredHostBytes);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_vmmAttempted) throw new InvalidOperationException("VMM weight configuration was already attempted.");
            if (MaxPrefillRows != 512 || MaxDecodeRows != 1 || LayerCount != 3 || _fileSourceViewAttempted
                || PlanVmmWeightsHostBytes(_native) != declaredHostBytes)
                throw new NotSupportedException("VMM weights require the exact single-device profile and durable Host plan.");
            _vmmAttempted = true; // Retain this token before any fallible native mutation.
            _native.ConfigureVmmWeights(_token);
            var usage = ReadVmmUsageCore();
            if (usage.Configured != 1 || usage.Closed != 0 || usage.DeclaredHostBytes != declaredHostBytes
                || usage.HostTicketOwners == 0 || usage.LiveTables != 0 || usage.OwnedDeviceBytes != 0
                || usage.PendingTicketBytes != 0 || usage.QuarantinedBytes != 0)
                throw new InvalidOperationException("Native VMM weights did not certify the admitted empty arena.");
            _vmmConfigured = true;
        }
    }

    /// <summary>Check the already-certified immutable mode without collecting
    /// process-wide diagnostic ownership snapshots on every decode step.</summary>
    public void ValidateVmmWeightsForExecution()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_vmmConfigured || _vmmClosed)
                throw new InvalidOperationException("VMM expert weights are not configured for execution.");
        }
    }

    public GgmlHostMoeVmmWeightsUsage ReadVmmUsage()
    {
        lock (_gate) { ObjectDisposedException.ThrowIf(_disposed, this); return ReadVmmUsageCore(); }
    }
    private GgmlHostMoeVmmWeightsUsage ReadVmmUsageCore()
    {
        var value = _native.ReadVmmUsage();
        if (value.Version != VmmProfile || value.Configured is < 0 or > 1 || value.Closed is < 0 or > 1
            || value.DeclaredHostBytes < 0 || value.HostTicketOwners is < 0 or > 2 || value.LiveTables is < 0 or > 3
            || value.OwnedDeviceBytes < 0 || value.PendingTicketBytes < 0 || value.QuarantinedBytes < 0
            || value.ReshapeAttempts < 0 || value.ReshapeCompleted < 0 || value.ReshapeFailures < 0
            || value.ReshapeCompleted > value.ReshapeAttempts || value.ReshapeFailures > value.ReshapeAttempts
            || value.ReshapeCompleted > value.ReshapeAttempts - value.ReshapeFailures)
            throw new InvalidOperationException("Native VMM weight ownership snapshot is invalid.");
        return value;
    }

    /// <summary>Final checked retirement before model/backend disposal. Pressure
    /// Trim(0) alone keeps reusable durable metadata. A failure leaves this scope
    /// and every native owner retryable; no downstream disposal may continue.</summary>
    public void PrepareForModelDispose()
    {
        lock (_gate) { ObjectDisposedException.ThrowIf(_disposed, this); CloseFileSourceViewsCore(); CloseVmmCore(); }
    }
    private void CloseVmmCore()
    {
        if (!_vmmAttempted || _vmmClosed) return;
        _native.Trim(_token, 0);
        _native.CloseVmmWeights(_token);
        var usage = ReadVmmUsageCore();
        if ((usage.Configured == 1 && usage.Closed != 1) || usage.HostTicketOwners != 0 || usage.LiveTables != 0
            || usage.OwnedDeviceBytes != 0 || usage.PendingTicketBytes != 0 || usage.QuarantinedBytes != 0)
            throw new InvalidOperationException("Native VMM weights still own resources or unacknowledged tickets.");
        _vmmClosed = true;
    }

    public GgmlHostMoeExpertCacheScope(long capacityBytes, int maxPrefillRows, int layerCount)
        : this(capacityBytes, maxPrefillRows, layerCount, NativeApi.Instance, 1) { }

    /// <summary>Plan exact-width decode variants up to the session's actual
    /// supported concurrency. Variants share the same resident expert weights;
    /// their measured allocation remains inside the cache capacity. GLM's
    /// explicitly installed phase plan permits prefill widths through 1024;
    /// ordinary Qwen compact-prefill policy retains its separate 128-row cap.</summary>
    public GgmlHostMoeExpertCacheScope(long capacityBytes, int maxPrefillRows, int layerCount, int maxDecodeRows)
        : this(capacityBytes, maxPrefillRows, layerCount, NativeApi.Instance, maxDecodeRows) { }

    /// <summary>Optionally grow retained tables after a larger scoped budget is
    /// admitted. Expansion preserves hot payloads and charges simultaneous old
    /// and new storage; a normal admission refusal keeps the old table. Disabled
    /// by default pending model and performance qualification.</summary>
    public GgmlHostMoeExpertCacheScope(long capacityBytes, int maxPrefillRows, int layerCount, int maxDecodeRows,
        bool enableCapacityGrowth)
        : this(capacityBytes, maxPrefillRows, layerCount, NativeApi.Instance, maxDecodeRows, enableCapacityGrowth) { }

    internal GgmlHostMoeExpertCacheScope(long capacityBytes, int maxPrefillRows, int layerCount,
        IHostMoeExpertCacheScopeNative native, int maxDecodeRows = 1, bool enableCapacityGrowth = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(capacityBytes);
        if (maxPrefillRows is < 1 or > 1024) throw new ArgumentOutOfRangeException(nameof(maxPrefillRows));
        if (maxDecodeRows is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(maxDecodeRows));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);
        ArgumentNullException.ThrowIfNull(native);
        _native = native;
        MaximumBudgetBytes = capacityBytes;
        MaxPrefillRows = maxPrefillRows;
        MaxDecodeRows = maxDecodeRows;
        LayerCount = layerCount;
        CapacityGrowthEnabled = enableCapacityGrowth;
        // All managed allocations/validation precede the native mutation.
        // There is no managed allocation after a successful Begin.
        _token = native.Begin(capacityBytes, maxPrefillRows, layerCount, maxDecodeRows, enableCapacityGrowth);
        _budgetBytes = capacityBytes;
    }

    /// <summary>The initial ceiling; pressure recovery cannot exceed this value.</summary>
    public long MaximumBudgetBytes { get; }
    public long BudgetBytes { get { lock (_gate) return _budgetBytes; } }
    public int MaxPrefillRows { get; }
    public int MaxDecodeRows { get; }
    public int LayerCount { get; }
    public bool CapacityGrowthEnabled { get; }
    /// <summary>Accounted device-cache bytes, including the fixed allowance;
    /// excludes the separately charged host file-read workspace.</summary>
    public long CachedBytes => ReadUsage().AccountedBytes;
    public GgmlHostMoeExpertCacheUsage ReadUsage()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _native.ReadUsage();
        }
    }

    /// <summary>Set the current ceiling and synchronously evict entries exceeding
    /// the total or per-layer allowance. Zero disables this tier. Later growth is
    /// limited to MaximumBudgetBytes. Returns released accounted device bytes,
    /// not measured driver free memory. On failure the old ceiling remains active,
    /// although successfully retired entries need not be recreated.</summary>
    public long UpdateBudget(long capacityBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(capacityBytes);
        if (capacityBytes > MaximumBudgetBytes) throw new ArgumentOutOfRangeException(nameof(capacityBytes));
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_vmmClosed || _fileSourceViewClosing || _fileSourceViewClosed) throw new InvalidOperationException("Checked expert ownership is finally closed.");
            long released = _native.Update(_token, capacityBytes);
            _budgetBytes = capacityBytes;
            return released;
        }
    }

    /// <summary>Retire LRU entries down to targetBytes without changing the ceiling.
    /// Always releases reusable CPU prefill weight/work staging and its dedicated
    /// CUDA file-stream graph arena at this quiescent
    /// boundary, even when targetBytes leaves every device entry resident.
    /// Zero also releases the host file-read workspace once entries are empty.
    /// Return value counts device cache tickets, excluding released host workspace.</summary>
    public long Trim(long targetBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(targetBytes);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _native.Trim(_token, targetBytes);
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            CloseFileSourceViewsCore();
            CloseVmmCore();
            _native.End(_token); // Frees physical storage before restoring old settings.
            _disposed = true; // Do not lose ownership if native cleanup fails.
        }
    }

    private sealed class NativeApi : IHostMoeExpertCacheScopeNative
    {
        public static readonly NativeApi Instance = new();
        private static void Require(int status)
        {
            if (status != 1) throw new InvalidOperationException("GGML expert-cache scope failed: " + GgmlNative.LastNativeError());
        }
        public ulong Begin(long bytes, int maxRows, int layers, int maxDecodeRows)
        {
            try
            {
                ulong token;
                // The scalar API continues to work with older native libraries.
                // A wider plan requires the matching ABI and must never silently
                // return a scope that would rebuild weights on first admission.
                Require(maxDecodeRows == 1
                    ? GgmlNative.BeginHostMoeExpertCacheScope(bytes, maxRows, layers, out token)
                    : GgmlNative.BeginHostMoeExpertCacheScopeWithDecodeRows(bytes, maxRows, layers, maxDecodeRows, out token));
                return token;
            }
            catch (EntryPointNotFoundException error)
            {
                throw new NotSupportedException("Expert-cache configuration scopes require the matching updated GgmlOps native library.", error);
            }
        }
        public ulong Begin(long bytes, int maxRows, int layers, int maxDecodeRows, bool enableCapacityGrowth)
        {
            if (!enableCapacityGrowth) return Begin(bytes, maxRows, layers, maxDecodeRows);
            try
            {
                Require(GgmlNative.BeginHostMoeExpertCacheScopeWithCapacityGrowth(bytes, maxRows, layers,
                    maxDecodeRows, 1, out ulong token));
                return token;
            }
            catch (EntryPointNotFoundException error)
            {
                throw new NotSupportedException("Capacity growth requires the matching updated GgmlOps native library.", error);
            }
        }
        public long Update(ulong token, long bytes)
        { Require(GgmlNative.UpdateHostMoeExpertCacheScope(token, bytes, out long released)); return released; }
        public long Trim(ulong token, long bytes)
        { Require(GgmlNative.TrimHostMoeExpertCacheScope(token, bytes, out long released)); return released; }
        public long End(ulong token)
        { Require(GgmlNative.EndHostMoeExpertCacheScope(token, out long released)); return released; }
        public GgmlHostMoeExpertCacheUsage ReadUsage() => GgmlBasicOps.GetHostMoeExpertCacheUsage();
        public int VmmWeightsVersion()
        {
            try { return GgmlNative.HostMoeVmmWeightsVersion(); }
            catch (EntryPointNotFoundException error) { throw new NotSupportedException("VMM weights require the matching native library.", error); }
        }
        public long PlanVmmHostBytes()
        { Require(GgmlNative.HostMoeVmmWeightsHostPlan(VmmProfile, 0, 0, VmmMaximumChunks, out long bytes)); return bytes; }
        public void ConfigureVmmWeights(ulong token)
            => Require(GgmlNative.ConfigureHostMoeVmmWeights(token, VmmProfile, 0, 0, VmmMaximumChunks));
        public void CloseVmmWeights(ulong token) => Require(GgmlNative.CloseHostMoeVmmWeights(token));
        public unsafe GgmlHostMoeVmmWeightsUsage ReadVmmUsage()
        {
            Span<long> values = stackalloc long[12];
            fixed (long* pointer = values) Require(GgmlNative.HostMoeVmmWeightsStats(pointer, values.Length));
            return new(values[0], values[1], values[2], values[3], values[4], values[5],
                values[6], values[7], values[8], values[9], values[10], values[11]);
        }
        public int FileSourceViewVersion()
        {
            try { return GgmlNative.HostMoeFileSourceViewVersion(); }
            catch (EntryPointNotFoundException error) { throw new NotSupportedException("Source views require the matching native library.", error); }
        }
        public int FileSourceViewEnabled()
        {
            try { return GgmlNative.HostMoeFileSourceViewEnabled(); }
            catch (EntryPointNotFoundException error) { throw new NotSupportedException("Source views require the matching native library.", error); }
        }
        public unsafe GgmlHostMoeFileSourceViewPlan PlanFileSourceViews(int sources, long sourceBytes)
        {
            Span<long> v = stackalloc long[8];
            fixed (long* pointer = v) Require(GgmlNative.HostMoeFileSourceViewHostPlan(1, sources, sourceBytes, pointer, v.Length));
            return new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
        }
        public void ConfigureFileSourceViews(ulong token, int sources, long sourceBytes, long totalHostPromise)
            => Require(GgmlNative.ConfigureHostMoeFileSourceView(token, 1, sources, sourceBytes, totalHostPromise));
        public void CloseFileSourceViews(ulong token) => Require(GgmlNative.CloseHostMoeFileSourceView(token));
        public unsafe GgmlHostMoeFileSourceViewUsage ReadFileSourceViewUsage(ulong token)
        {
            Span<long> v = stackalloc long[16];
            fixed (long* pointer = v) Require(GgmlNative.HostMoeFileSourceViewStats(token, pointer, v.Length));
            return new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
        }
    }
}
