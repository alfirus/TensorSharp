// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Globalization;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

internal static partial class GgmlNative
{
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeRamCacheConfigure")]
    internal static partial int HostMoeRamCacheConfigure(long bytes);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeRamCacheStats")]
    internal static unsafe partial int HostMoeRamCacheStats(long* values, int length);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeMappedRamVersion")]
    internal static partial int HostMoeMappedRamVersion();
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeMappedRamEnabled")]
    internal static partial int HostMoeMappedRamEnabled();
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeMappedRamStats")]
    internal static unsafe partial int HostMoeMappedRamStats(long* values, int length);
}

/// <summary>Logical RAM payload and source-read counters; not physical disk I/O
/// or total process memory. Native allocation tickets include resident payload.</summary>
public readonly record struct GgmlHostMoeRamCacheUsage(long CapacityBytes, long ResidentBytes,
    long ReservedBytes, long Hits, long Misses, long HitBytes, long LoadBytes,
    long BypassBytes, long Evictions);

/// <summary>Separate mapped-view accounting. Locked and metadata bytes belong
/// to charged owners; logical bytes and copy counters do not measure disk I/O
/// or unique physical pages shared with other mappings.</summary>
public readonly record struct GgmlHostMoeMappedRamCacheUsage(long CapacityChargeBytes,
    long LogicalResidentBytes, long OwnedChargeBytes, long LockedPageBytes, long MetadataChargeBytes,
    long ResidentEntries, long ActiveLeases, long QuarantineChargeBytes, long Hits, long Misses,
    long HitCopyBytes, long AdmittedLogicalBytes, long BypassLogicalBytes, long Evictions,
    long PendingAdmissions, bool QuotaRestorePending)
{
    internal bool IsClean => CapacityChargeBytes == 0 && LogicalResidentBytes == 0 && OwnedChargeBytes == 0
        && LockedPageBytes == 0 && MetadataChargeBytes == 0 && ResidentEntries == 0 && ActiveLeases == 0
        && QuarantineChargeBytes == 0 && PendingAdmissions == 0 && !QuotaRestorePending;
}

internal interface IHostMoeRamCacheNative
{
    void Configure(long bytes);
    GgmlHostMoeRamCacheUsage ReadUsage();
    int MappedVersion() => 0;
    int MappedEnabled() => 0;
    GgmlHostMoeMappedRamCacheUsage ReadMappedUsage()
        => throw new NotSupportedException("The native RAM expert cache has no mapped-view capability.");
}

/// <summary>One session's process-wide RAM expert cache. The caller owns the
/// exclusive selected-expert session and admits a dedicated host reservation
/// before enabling this lazy cache. Configure/Dispose require quiescent model
/// execution and finish native eviction before returning. Dispose before the
/// host reservation or shared budget; failed disposal remains retryable.</summary>
public sealed class GgmlHostMoeRamCacheScope : IDisposable
{
    private static readonly object Gate = new();
    private static GgmlHostMoeRamCacheScope? Active;
    private readonly IHostMoeRamCacheNative _native;
    private readonly Func<string, string?> _environment;
    private readonly MappedMode _mappedMode;
    private bool _disposed;
    public long CapacityBytes { get; private set; }
    public bool IsMapped => _mappedMode.Enabled;
    public bool UsesExclusiveMappedQuota => _mappedMode.ExclusiveQuota;
    public long MappedQuotaCeilingBytes => _mappedMode.QuotaCeilingBytes;
    public GgmlHostMoeRamCacheScope() : this(NativeApi.Instance) { }
    internal GgmlHostMoeRamCacheScope(IHostMoeRamCacheNative native, Func<string, string?>? environment = null)
    {
        _native = native ?? throw new ArgumentNullException(nameof(native));
        _environment = environment ?? Environment.GetEnvironmentVariable;
        lock (Gate)
        {
            if (Active != null) throw new InvalidOperationException("A RAM expert-cache scope is already active.");
            Active = this;
            try
            {
                _mappedMode = ReadMappedMode();
                if (IsMapped) ValidateMappedCapability();
                _native.Configure(0); // Verify/clean the selected ABI before lending any host credit.
                EnsureCanReleaseHostCredit(0);
            }
            catch { Active = null; throw; }
        }
    }
    public void Configure(long bytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        lock (Gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            // Zero is always a cleanup request, including after environment
            // drift. Positive budgets must match both managed and native latches.
            if (bytes != 0)
            {
                if (ReadMappedMode() != _mappedMode)
                    throw new InvalidOperationException("RAM expert-cache mode or quota settings changed after scope construction.");
                if (IsMapped) ValidateMappedCapability();
            }
            _native.Configure(bytes);
            if (IsMapped)
            {
                var usage = ReadMappedUsageCore();
                if (usage.CapacityChargeBytes != bytes)
                    throw new InvalidOperationException("Mapped RAM expert cache did not apply its requested charged-byte budget.");
                EnsureCanReleaseHostCredit(bytes, usage);
            }
            CapacityBytes = bytes;
        }
    }
    public GgmlHostMoeMappedRamCacheUsage ReadMappedUsage()
    {
        lock (Gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!IsMapped) throw new InvalidOperationException("This RAM expert-cache scope does not use mapped views.");
            return ReadMappedUsageCore();
        }
    }
    /// <summary>Checks the native ownership floor before the caller lends
    /// Host credit. The caller must keep model execution quiescent through
    /// the corresponding reservation change.</summary>
    public void EnsureCanReleaseHostCredit(long target)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(target);
        lock (Gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (IsMapped) EnsureCanReleaseHostCredit(target, ReadMappedUsageCore());
        }
    }
    private static void EnsureCanReleaseHostCredit(long target, GgmlHostMoeMappedRamCacheUsage usage)
    {
        if (usage.OwnedChargeBytes > target || usage.PendingAdmissions != 0 || (target == 0 && !usage.IsClean))
            throw new InvalidOperationException("Mapped RAM expert cache still owns locked views, leases, pending admissions, quarantine, or quota restoration.");
    }
    private GgmlHostMoeMappedRamCacheUsage ReadMappedUsageCore()
    {
        var usage = _native.ReadMappedUsage();
        if (usage.CapacityChargeBytes < 0 || usage.LogicalResidentBytes < 0 || usage.OwnedChargeBytes < 0
            || usage.LockedPageBytes < 0 || usage.MetadataChargeBytes < 0 || usage.ResidentEntries < 0
            || usage.ActiveLeases < 0 || usage.QuarantineChargeBytes < 0 || usage.Hits < 0 || usage.Misses < 0
            || usage.HitCopyBytes < 0 || usage.AdmittedLogicalBytes < 0 || usage.BypassLogicalBytes < 0
            || usage.Evictions < 0 || usage.PendingAdmissions < 0
            || usage.LogicalResidentBytes > usage.OwnedChargeBytes
            || usage.LockedPageBytes > usage.OwnedChargeBytes
            || usage.MetadataChargeBytes > usage.OwnedChargeBytes - usage.LockedPageBytes
            || usage.QuarantineChargeBytes > usage.OwnedChargeBytes)
            throw new InvalidOperationException("Mapped RAM expert-cache accounting is invalid.");
        return usage;
    }
    private void ValidateMappedCapability()
    {
        try
        {
            if (_native.MappedVersion() != 1)
                throw new NotSupportedException("Mapped RAM expert caching requires native capability version 1.");
            if (_native.MappedEnabled() != 1)
                throw new InvalidOperationException("Mapped RAM expert-cache request differs from the native mode latch.");
        }
        catch (Exception error) when (error is EntryPointNotFoundException or DllNotFoundException)
        { throw new NotSupportedException("Mapped RAM expert caching requires the matching GgmlOps native library.", error); }
    }
    private readonly record struct MappedMode(bool Enabled, bool ExclusiveQuota, long QuotaCeilingBytes);
    private MappedMode ReadMappedMode()
    {
        if (_environment("TS_HOST_MOE_RAM_CACHE_MAPPED") != "1") return default;
        foreach (string name in new[] { "TS_HOST_MOE_RAM_CACHE_PINNED", "TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD",
            "TS_HOST_MOE_RAM_CACHE_VICTIM", "TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY", "TS_HOST_MOE_RAM_CACHE_FREQUENCY",
            "TS_HOST_MOE_LAYER_PREFETCH", "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "TS_HOST_MOE_EXPERT_CACHE_PREFETCH",
            "TS_HOST_MOE_DECODE_PREFETCH" })
            if (_environment(name) == "1")
                throw new ArgumentException("Mapped RAM expert caching conflicts with " + name + ".");
        bool exclusive = _environment("TS_HOST_MOE_RAM_CACHE_MAPPED_EXCLUSIVE_QUOTA") == "1";
        string? configured = _environment("TS_HOST_MOE_RAM_CACHE_MAPPED_QUOTA_MB");
        if (!exclusive)
        {
            if (!string.IsNullOrEmpty(configured) && configured != "0")
                throw new ArgumentException("Mapped RAM quota expansion requires TS_HOST_MOE_RAM_CACHE_MAPPED_EXCLUSIVE_QUOTA=1.");
            return new(true, false, 0);
        }
        if (!long.TryParse(configured, NumberStyles.None, CultureInfo.InvariantCulture, out long mib)
            || mib <= 0 || mib > long.MaxValue / (1L << 20))
            throw new ArgumentException("TS_HOST_MOE_RAM_CACHE_MAPPED_QUOTA_MB must be a positive, representable MiB ceiling.");
        return new(true, true, checked(mib * (1L << 20)));
    }
    public GgmlHostMoeRamCacheUsage ReadUsage()
    {
        lock (Gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _native.ReadUsage();
        }
    }
    public void Dispose()
    {
        lock (Gate)
        {
            if (_disposed) return;
            _native.Configure(0);
            EnsureCanReleaseHostCredit(0);
            CapacityBytes = 0;
            _disposed = true;
            Active = null;
        }
    }
    private sealed class NativeApi : IHostMoeRamCacheNative
    {
        internal static readonly NativeApi Instance = new();
        public void Configure(long bytes)
        {
            try
            {
                if (GgmlNative.HostMoeRamCacheConfigure(bytes) != 1)
                    throw new InvalidOperationException("RAM expert-cache configuration failed: " + GgmlNative.LastNativeError());
            }
            catch (EntryPointNotFoundException error)
            { throw new NotSupportedException("RAM expert caching requires the matching GgmlOps native library.", error); }
        }
        public unsafe GgmlHostMoeRamCacheUsage ReadUsage()
        {
            Span<long> values = stackalloc long[9];
            fixed (long* pointer = values)
                if (GgmlNative.HostMoeRamCacheStats(pointer, values.Length) != 1)
                    throw new InvalidOperationException("RAM expert-cache usage read failed: " + GgmlNative.LastNativeError());
            return new(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]);
        }
        public int MappedVersion() => GgmlNative.HostMoeMappedRamVersion();
        public int MappedEnabled() => GgmlNative.HostMoeMappedRamEnabled();
        public unsafe GgmlHostMoeMappedRamCacheUsage ReadMappedUsage()
        {
            Span<long> values = stackalloc long[16];
            try
            {
                fixed (long* pointer = values)
                    if (GgmlNative.HostMoeMappedRamStats(pointer, values.Length) != 1)
                        throw new InvalidOperationException("Mapped RAM expert-cache usage read failed: " + GgmlNative.LastNativeError());
            }
            catch (EntryPointNotFoundException error)
            { throw new NotSupportedException("Mapped RAM expert-cache usage requires the matching GgmlOps native library.", error); }
            if (values[15] is not (0 or 1)) throw new InvalidOperationException("Mapped RAM quota-restoration state is invalid.");
            return new(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
                values[8], values[9], values[10], values[11], values[12], values[13], values[14], values[15] == 1);
        }
    }
}
