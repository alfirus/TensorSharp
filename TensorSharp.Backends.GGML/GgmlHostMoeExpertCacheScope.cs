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
    [LibraryImport(DllName, EntryPoint = "TSGgml_UpdateHostMoeExpertCacheScope")]
    internal static partial int UpdateHostMoeExpertCacheScope(ulong token, long bytes, out long released);
    [LibraryImport(DllName, EntryPoint = "TSGgml_TrimHostMoeExpertCacheScope")]
    internal static partial int TrimHostMoeExpertCacheScope(ulong token, long bytes, out long released);
    [LibraryImport(DllName, EntryPoint = "TSGgml_EndHostMoeExpertCacheScope")]
    internal static partial int EndHostMoeExpertCacheScope(ulong token, out long released);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeExpertCacheStats")]
    internal static partial int HostMoeExpertCacheStats(out long accounted, out long budget,
        out long hits, out long misses, out long calls);
}

/// <summary>Process-wide compact expert-cache counters. AccountedBytes includes
/// its graph payload and fixed workspace allowance; it is not physical VRAM.
/// Host file-read staging and vendor/driver pools are excluded. Counters are
/// cumulative until the existing full cache teardown resets them.</summary>
public readonly record struct GgmlHostMoeExpertCacheUsage(long AccountedBytes, long BudgetBytes,
    long Hits, long Misses, long Calls);

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
}

internal interface IHostMoeExpertCacheScopeNative
{
    ulong Begin(long bytes, int maxRows, int layers);
    long Update(ulong token, long bytes);
    long Trim(ulong token, long bytes);
    long End(ulong token);
    GgmlHostMoeExpertCacheUsage ReadUsage();
}

/// <summary>Exclusive, process-wide configuration of the optional Qwen4Exp CUDA
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

    public GgmlHostMoeExpertCacheScope(long capacityBytes, int maxPrefillRows, int layerCount)
        : this(capacityBytes, maxPrefillRows, layerCount, NativeApi.Instance) { }

    internal GgmlHostMoeExpertCacheScope(long capacityBytes, int maxPrefillRows, int layerCount,
        IHostMoeExpertCacheScopeNative native)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(capacityBytes);
        if (maxPrefillRows is < 1 or > 128) throw new ArgumentOutOfRangeException(nameof(maxPrefillRows));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);
        ArgumentNullException.ThrowIfNull(native);
        _native = native;
        MaximumBudgetBytes = capacityBytes;
        MaxPrefillRows = maxPrefillRows;
        LayerCount = layerCount;
        // All managed allocations/validation precede the native mutation.
        // There is no managed allocation after a successful Begin.
        _token = native.Begin(capacityBytes, maxPrefillRows, layerCount);
        _budgetBytes = capacityBytes;
    }

    /// <summary>The initial ceiling; pressure recovery cannot exceed this value.</summary>
    public long MaximumBudgetBytes { get; }
    public long BudgetBytes { get { lock (_gate) return _budgetBytes; } }
    public int MaxPrefillRows { get; }
    public int LayerCount { get; }
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
        public ulong Begin(long bytes, int maxRows, int layers)
        {
            try
            {
                Require(GgmlNative.BeginHostMoeExpertCacheScope(bytes, maxRows, layers, out ulong token));
                return token;
            }
            catch (EntryPointNotFoundException error)
            {
                throw new NotSupportedException("Expert-cache configuration scopes require the matching updated GgmlOps native library.", error);
            }
        }
        public long Update(ulong token, long bytes)
        { Require(GgmlNative.UpdateHostMoeExpertCacheScope(token, bytes, out long released)); return released; }
        public long Trim(ulong token, long bytes)
        { Require(GgmlNative.TrimHostMoeExpertCacheScope(token, bytes, out long released)); return released; }
        public long End(ulong token)
        { Require(GgmlNative.EndHostMoeExpertCacheScope(token, out long released)); return released; }
        public GgmlHostMoeExpertCacheUsage ReadUsage() => GgmlBasicOps.GetHostMoeExpertCacheUsage();
    }
}
