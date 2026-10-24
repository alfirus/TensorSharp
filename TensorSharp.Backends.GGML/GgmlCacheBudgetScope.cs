// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.Memory;

namespace TensorSharp.GGML;

/// <summary>Instrumented payload ownership, counted once even with multiple
/// constraint pools. Pending bytes are reservations, not proven allocations.</summary>
public sealed record GgmlAllocationUsage(int Rank, int Kind, int Allocations, long PendingBytes, long CommittedBytes);

/// <summary>A refused allocation is ordinary pressure, separate from a callback
/// error. RemainingEnvelope is null outside a request. Observations are diagnostic,
/// not reusable capacity; other owners can change immediately after the callback.</summary>
public sealed record GgmlAllocationRefusal(int Rank, int Kind, long Bytes,
    IReadOnlyList<MemoryCharge>? RemainingEnvelope, IReadOnlyList<MemoryPoolSnapshot> Budget);

/// <summary>Opt-in, process-wide budget ownership for new GGML lazy device-copy
/// and explicit-preload cache allocations. Install before those allocations exist.
/// Each rank maps to one or more existing budget pools; listing RAM and GPU for
/// UMA applies two constraints to one allocation, not two physical copies.
/// With includeGraphBuffers=true, additionally charge the explicitly routed
/// TensorSharp-owned context buffers and shared reuse graph allocators, including
/// Qwen4Exp graph arenas, recurrent state and device state snapshots, Qwen-Image
/// retained graphs/prefix storage, and MiniMax-H3 execution buffers. Buffers
/// retain their original native backend interfaces. Other allocator paths, live KV
/// outside these buffers, host-pointer wrappers, backend pools, allocator rounding
/// and driver overhead are not covered; this is not a whole-model memory cap.
/// Cache-only refusal can use unbudgeted per-graph streaming. Dispose only after
/// model work has stopped and all covered caches/graphs have been released.
/// Supplying hostPools additionally accounts for the compact expert file-read
/// arena, DeepSeek demand-read staging and GLM uploader/TP-strip scratch payload
/// against the same budget, independently
/// of rank/device pool mappings.
/// Failed disposal keeps callbacks rooted and accounting active for a retry.</summary>
public sealed class GgmlCacheBudgetScope : IDisposable
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate ulong ReserveCallback(IntPtr context, int rank, int kind, long bytes);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int CommitCallback(IntPtr context, ulong token);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void ReleaseCallback(IntPtr context, ulong token);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int CheckedReleaseCallback(IntPtr context, ulong token);
    private static readonly ReserveCallback ReserveRoot = ReserveNative;
    private static readonly CommitCallback CommitRoot = CommitNative;
    private static readonly ReleaseCallback ReleaseRoot = ReleaseNative;
    private static readonly CheckedReleaseCallback CheckedReleaseRoot = ReleaseNativeChecked;
    private readonly object _gate = new();
    private readonly object _lifecycleGate = new();
    private readonly MemoryBudget _budget;
    private readonly string[][] _rankPools;
    private readonly string[] _hostPools;
    private sealed class Allocation(BudgetReservation reservation, int rank, int kind, long bytes)
    {
        public readonly BudgetReservation Reservation = reservation;
        public readonly int Rank = rank, Kind = kind;
        public readonly long Bytes = bytes;
        public bool Committed;
        public long GraphDiagnosticOriginCall;
    }
    private readonly Dictionary<ulong, Allocation> _allocations = new();
    private GCHandle _handle;
    private ulong _nextToken;
    private Exception? _callbackError;
    private bool _disposed;
    private bool _disposing;
    private BudgetReservation? _executionEnvelope;
    private AllocationOwner? _allocationOwner;
    private BudgetReservation? _sharedExpertEnvelope;
    private BudgetReservation? _sharedHostExpertEnvelope;
    private BudgetReservation? _sharedJointEnvelope;
    private BudgetReservation? _sharedSpeculativeEnvelope;
    private long _allocationRefusalCount;
    private GgmlAllocationRefusal? _firstAllocationRefusal, _lastAllocationRefusal;
    private const string NoGraphDiagnosticContext = "call=0 context_tokens=0 admitted_rows=0 output_rows=0 submitted_tokens=0 layer_major=0";
    private string _graphDiagnosticContext = NoGraphDiagnosticContext;
    private long _graphDiagnosticCall;
    private long _graphDiagnosticNextCall;
    private Dictionary<int, long>? _graphDiagnosticPeaks;
    /// <summary>Optional kind-5 payload-charge diagnostics. The writer runs under
    /// the accounting gate and must not call native APIs. Writer failures are
    /// ignored and never turn a successful allocation into a refusal.</summary>
    public Action<string>? GraphMemoryDiagnosticWriter { get; set; }
    internal Action? BeforeExecutionOwnerAllocation { get; set; }
    internal Action? BeforeAllocationOwnerAllocation { get; set; }

    /// <summary>Label the next synchronous managed forward, including its native
    /// worker callbacks. Rows is its admitted maximum, not an observed inner
    /// layer/chunk width. This changes no reservation or execution geometry.</summary>
    public void SetGraphMemoryDiagnosticContext(int contextTokens, int rows, int outputRows,
        int submittedTokens, bool layerMajor)
    {
        if (GraphMemoryDiagnosticWriter == null) return;
        lock (_gate)
        {
            try
            {
                _graphDiagnosticCall = ++_graphDiagnosticNextCall;
                _graphDiagnosticContext = FormattableString.Invariant($"call={_graphDiagnosticCall} context_tokens={contextTokens} admitted_rows={rows} output_rows={outputRows} submitted_tokens={submittedTokens} layer_major={(layerMajor ? 1 : 0)}");
                WriteGraphMemoryDiagnostic("call-begin", 0, 0, 0, _graphDiagnosticCall);
            }
            catch { } // Diagnostics must not change model/accounting behavior.
        }
    }

    /// <summary>End diagnostic attribution when the synchronous execution scope
    /// leaves; later cache teardown is labelled call=0.</summary>
    public void ClearGraphMemoryDiagnosticContext()
    {
        if (GraphMemoryDiagnosticWriter == null) return;
        lock (_gate)
        {
            WriteGraphMemoryDiagnostic("call-end", 0, 0, 0, _graphDiagnosticCall);
            _graphDiagnosticContext = NoGraphDiagnosticContext;
            _graphDiagnosticCall = 0;
        }
    }

    private void WriteGraphMemoryDiagnostic(string action, ulong token, int rank, long bytes, long originCall)
    {
        var writer = GraphMemoryDiagnosticWriter;
        if (writer == null) return;
        try
        {
            long pending = 0, committed = 0;
            int allocations = 0;
            foreach (var allocation in _allocations.Values)
            {
                if (allocation.Kind != 5 || allocation.Rank != rank) continue;
                ++allocations;
                if (allocation.Committed) committed = checked(committed + allocation.Bytes);
                else pending = checked(pending + allocation.Bytes);
            }
            long owned = checked(pending + committed);
            var peaks = _graphDiagnosticPeaks ??= new();
            peaks.TryGetValue(rank, out long peak);
            peaks[rank] = peak = Math.Max(peak, owned);
            writer(FormattableString.Invariant($"[GLM-GRAPH-MEMORY] event={action} {_graphDiagnosticContext ?? NoGraphDiagnosticContext} origin_call={originCall} rank={rank} kind=5 token={token} bytes={bytes} allocations={allocations} pending_bytes={pending} committed_bytes={committed} owned_bytes={owned} peak_owned_bytes={peak} scope=instrumented_payload_charges_pending_is_not_physical_excludes_driver_pools"));
        }
        catch { } // Never let telemetry change callback success or ownership.
    }

    /// <summary>Bounded failure telemetry: no growing event log or successful-allocation sampling.</summary>
    public (long Count, GgmlAllocationRefusal? First, GgmlAllocationRefusal? Last) AllocationRefusals
    { get { lock (_gate) return (_allocationRefusalCount, _firstAllocationRefusal, _lastAllocationRefusal); } }

    public GgmlCacheBudgetScope(MemoryBudget budget, IEnumerable<IEnumerable<string>> rankPools)
        : this(budget, rankPools, includeGraphBuffers: false) { }

    public GgmlCacheBudgetScope(MemoryBudget budget, IEnumerable<IEnumerable<string>> rankPools, bool includeGraphBuffers)
        : this(budget, rankPools, includeGraphBuffers, hostPools: null) { }

    /// <summary>Also charge explicitly routed native host staging to the supplied
    /// host pools. Null excludes these buffers, preserving the legacy contract.
    /// This mapping is process-wide, independent of device-rank pools. It covers
    /// the compact expert file arena, DeepSeek demand-read staging and GLM upload scratch,
    /// not mmap residency or all host allocations.</summary>
    public GgmlCacheBudgetScope(MemoryBudget budget, IEnumerable<IEnumerable<string>> rankPools,
        bool includeGraphBuffers, IEnumerable<string>? hostPools)
        : this(budget, rankPools, includeGraphBuffers, hostPools, requireCheckedRelease: false) { }

    /// <summary>Require a native checked-release bridge for owners that retain
    /// their exact ticket after cleanup acknowledgment fails. Create durable
    /// owners under EnterAllocationOwner with a session-lifetime reservation.</summary>
    public GgmlCacheBudgetScope(MemoryBudget budget, IEnumerable<IEnumerable<string>> rankPools,
        bool includeGraphBuffers, IEnumerable<string>? hostPools, bool requireCheckedRelease)
    {
        ArgumentNullException.ThrowIfNull(budget);
        ArgumentNullException.ThrowIfNull(rankPools);
        _budget = budget;
        IncludesGraphBuffers = includeGraphBuffers;
        IncludesCheckedRelease = requireCheckedRelease;
        _hostPools = hostPools?.ToArray() ?? [];
        if (hostPools != null && (_hostPools.Length == 0 || _hostPools.Distinct(StringComparer.Ordinal).Count() != _hostPools.Length))
            throw new ArgumentException("Map host staging to one or more distinct pools, or use null to exclude it.", nameof(hostPools));
        if (_hostPools.Length > 0)
            budget.CanEverFit(_hostPools.Select(pool => new MemoryCharge(pool, 0)));
        _rankPools = rankPools.Select(pools => pools?.ToArray()
            ?? throw new ArgumentException("Every rank needs a pool mapping.", nameof(rankPools))).ToArray();
        if (_rankPools.Length == 0 || _rankPools.Any(p => p.Length == 0 || p.Distinct(StringComparer.Ordinal).Count() != p.Length))
            throw new ArgumentException("Map each rank to one or more distinct budget pools.", nameof(rankPools));
        foreach (var pools in _rankPools)
            budget.CanEverFit(pools.Select(pool => new MemoryCharge(pool, 0))); // Validate names before native registration.
        _handle = GCHandle.Alloc(this);
        try
        {
            IntPtr context = GCHandle.ToIntPtr(_handle);
            IntPtr reserve = Marshal.GetFunctionPointerForDelegate(ReserveRoot);
            IntPtr commit = Marshal.GetFunctionPointerForDelegate(CommitRoot);
            IntPtr release = Marshal.GetFunctionPointerForDelegate(ReleaseRoot);
            int attached = requireCheckedRelease
                ? GgmlNative.AttachSharedCacheBudgetWithCheckedRelease(context, reserve, commit, release,
                    Marshal.GetFunctionPointerForDelegate(CheckedReleaseRoot), includeGraphBuffers ? 1 : 0, IncludesHostBuffers ? 1 : 0)
                : IncludesHostBuffers
                ? GgmlNative.AttachSharedCacheBudgetWithHost(context, reserve, commit, release, includeGraphBuffers ? 1 : 0)
                : includeGraphBuffers
                ? GgmlNative.AttachSharedCacheBudgetEx(context, reserve, commit, release, 1)
                : GgmlNative.AttachSharedCacheBudget(context, reserve, commit, release);
            if (attached != 1)
                throw new InvalidOperationException("Install the GGML budget before covered cache/graph allocations, with no other active budget scope.");
        }
        catch (Exception error)
        {
            _handle.Free();
            if (requireCheckedRelease && error is EntryPointNotFoundException)
                throw new NotSupportedException(
                    "Checked cache ownership requires TSGgml_AttachSharedCacheBudgetWithCheckedRelease. Update the native library with the managed assemblies.", error);
            if (IncludesHostBuffers && error is EntryPointNotFoundException)
                throw new NotSupportedException(
                    "Expert host-buffer budgeting requires a native library with TSGgml_AttachSharedCacheBudgetWithHost. Update the native library together with the managed assemblies.", error);
            throw;
        }
    }

    /// <summary>Unexpected callback failures fail admission closed and never cross
    /// the unmanaged boundary. Ordinary quota rejection is not an error.</summary>
    public Exception? CallbackError { get { lock (_gate) return _callbackError; } }
    public int ActiveAllocations { get { lock (_gate) return _allocations.Count; } }
    /// <summary>Kind 0: lazy device copies; 1: preloaded weights; 2: graph/context
    /// buffers; 3: native host staging; 4: selected-expert cache; 5: joint arena;
    /// 6: retained host expert payload; 7: persistent speculative graphs/state.
    /// Excludes uninstrumented driver/OS memory.</summary>
    public IReadOnlyList<GgmlAllocationUsage> AllocationUsage
    {
        get
        {
            lock (_gate) return _allocations.Values.GroupBy(a => (a.Rank, a.Kind))
                .Select(g => new GgmlAllocationUsage(g.Key.Rank, g.Key.Kind, g.Count(),
                    g.Where(a => !a.Committed).Sum(a => a.Bytes), g.Where(a => a.Committed).Sum(a => a.Bytes)))
                .OrderBy(a => a.Rank).ThenBy(a => a.Kind).ToArray();
        }
    }
    public bool IncludesGraphBuffers { get; }
    /// <summary>Explicit opt-in to acknowledged native ticket release. Existing
    /// void-release owners retain their original behavior.</summary>
    public bool IncludesCheckedRelease { get; }
    public bool IncludesHostBuffers => _hostPools.Length != 0;

    /// <summary>Use one session-owned reservation for selected-expert cache
    /// allocations, independent of the currently executing request. Change only
    /// while model execution is quiescent. Its live children must be freed before
    /// removing/replacing this owner; resizing unused credit preserves refunds.</summary>
    public void SetSharedExpertEnvelope(BudgetReservation? envelope)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_disposing || _executionEnvelope != null)
                throw new InvalidOperationException("Shared expert ownership changes require quiescent execution.");
            if (ReferenceEquals(envelope, _sharedExpertEnvelope)) return;
            if (_allocations.Values.Any(a => a.Kind == 4))
                throw new InvalidOperationException("Selected-expert allocations still own the previous reservation.");
            if (envelope != null && !ReferenceEquals(envelope.Budget, _budget))
                throw new ArgumentException("Expert cache belongs to a different budget.", nameof(envelope));
            _sharedExpertEnvelope = envelope;
        }
    }

    /// <summary>Persistent RAM expert payload has a separate owner from request
    /// staging. Install/remove only while execution is quiescent and after the
    /// previous cache's native payload has been released.</summary>
    public void SetSharedHostExpertEnvelope(BudgetReservation? envelope)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_disposing || _executionEnvelope != null)
                throw new InvalidOperationException("Shared host expert ownership changes require quiescent execution.");
            if (ReferenceEquals(envelope, _sharedHostExpertEnvelope)) return;
            if (_allocations.Values.Any(a => a.Kind == 6))
                throw new InvalidOperationException("Host expert allocations still own the previous reservation.");
            if (envelope != null && (!IncludesHostBuffers || !ReferenceEquals(envelope.Budget, _budget)))
                throw new ArgumentException("Host expert cache needs this budget's host pool mapping.", nameof(envelope));
            _sharedHostExpertEnvelope = envelope;
        }
    }

    /// <summary>Separate lifetime for a complete multi-request arena. Kind 5
    /// includes its state replicas as well as graph workspace; no individual
    /// request receives its refund when a participant finishes.</summary>
    public void SetSharedJointEnvelope(BudgetReservation? envelope)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_disposing || _executionEnvelope != null)
                throw new InvalidOperationException("Shared joint ownership changes require quiescent execution.");
            if (ReferenceEquals(envelope, _sharedJointEnvelope)) return;
            if (_allocations.Values.Any(a => a.Kind == 5))
                throw new InvalidOperationException("Joint arena allocations still own their previous reservation.");
            if (envelope != null && !ReferenceEquals(envelope.Budget, _budget))
                throw new ArgumentException("Joint arena belongs to a different budget.", nameof(envelope));
            _sharedJointEnvelope = envelope;
        }
    }

    /// <summary>Persistent draft graphs and recurrent verification snapshots
    /// retain this owner across request scopes. No incidental request or root
    /// fallback is permitted for their dedicated native allocation kind.</summary>
    public void SetSharedSpeculativeEnvelope(BudgetReservation? envelope)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_disposing || _executionEnvelope != null)
                throw new InvalidOperationException("Shared speculative ownership changes require quiescent execution.");
            if (ReferenceEquals(envelope, _sharedSpeculativeEnvelope)) return;
            if (_allocations.Values.Any(a => a.Kind == 7))
                throw new InvalidOperationException("Speculative allocations still own their previous reservation.");
            if (envelope != null && (!IncludesGraphBuffers || !ReferenceEquals(envelope.Budget, _budget)))
                throw new ArgumentException("Speculative state needs this budget's graph allocation mapping.", nameof(envelope));
            _sharedSpeculativeEnvelope = envelope;
        }
    }

    /// <summary>Route synchronous model execution allocations through an admitted
    /// request, including callbacks on native worker threads. The caller must
    /// hold the model compute lock and may not overlap execution or await.</summary>
    public IDisposable EnterExecution(BudgetReservation envelope)
    {
        ArgumentNullException.ThrowIfNull(envelope);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_disposing) throw new InvalidOperationException("The GGML budget is being detached.");
            if (!ReferenceEquals(envelope.Budget, _budget)) throw new ArgumentException("Request belongs to a different budget.", nameof(envelope));
            if (_executionEnvelope != null) throw new InvalidOperationException("Concurrent or nested GGML execution envelopes are unsupported.");
            // Do not publish an execution envelope until its final disposable
            // owner exists. An allocation failure leaves admission retryable.
            BeforeExecutionOwnerAllocation?.Invoke();
            var execution = new Execution(this);
            _executionEnvelope = envelope;
            return execution;
        }
    }

    /// <summary>Temporarily route synchronous GGML allocations to another owner,
    /// for example a persistent checkpoint, including native worker callbacks.
    /// Hold the model compute lock throughout; do not await or overlap model
    /// work. Overrides may nest but must dispose in reverse order. Existing
    /// shared expert/joint owners retain their dedicated allocation kinds.</summary>
    public IDisposable EnterAllocationOwner(BudgetReservation envelope)
    {
        ArgumentNullException.ThrowIfNull(envelope);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_disposing) throw new InvalidOperationException("The GGML budget is being detached.");
            if (!ReferenceEquals(envelope.Budget, _budget))
                throw new ArgumentException("Allocation owner belongs to a different budget.", nameof(envelope));
            // The final handle exists before either owner reference changes.
            // OOM leaves the previous request/override fully active.
            BeforeAllocationOwnerAllocation?.Invoke();
            var owner = new AllocationOwner(this, _executionEnvelope, _allocationOwner);
            _executionEnvelope = envelope;
            _allocationOwner = owner;
            return owner;
        }
    }

    private sealed class AllocationOwner(GgmlCacheBudgetScope scope, BudgetReservation? previousEnvelope,
        AllocationOwner? previousOwner) : IDisposable
    {
        private bool _disposed;
        public void Dispose()
        {
            lock (scope._gate)
            {
                if (_disposed) return;
                if (!ReferenceEquals(scope._allocationOwner, this))
                    throw new InvalidOperationException("GGML allocation owners must be restored in reverse order.");
                // Restore only references; no handle/closure/signal allocation
                // can fail after removing the active persistent owner.
                scope._executionEnvelope = previousEnvelope;
                scope._allocationOwner = previousOwner;
                _disposed = true;
            }
        }
    }

    private sealed class Execution(GgmlCacheBudgetScope owner) : IDisposable
    {
        private bool _disposed;
        public void Dispose()
        {
            lock (owner._gate)
            {
                if (_disposed) return;
                if (owner._allocationOwner != null)
                    throw new InvalidOperationException("Restore GGML allocation owners before leaving execution.");
                owner._executionEnvelope = null;
                _disposed = true;
            }
        }
    }

    public void Dispose()
    {
        // Do not hold _gate while calling native: a callback can already own the
        // native registry gate while it waits for this scope's accounting lock.
        lock (_lifecycleGate)
        {
            if (_disposed) return;
            lock (_gate)
            {
                if (_allocations.Count != 0 || _executionEnvelope != null)
                    throw new InvalidOperationException("GGML allocations still own budget credit. Stop model work and release covered caches/graphs before disposing the budget scope.");
                _disposing = true;
            }
            try
            {
                if (GgmlNative.DetachSharedCacheBudget(GCHandle.ToIntPtr(_handle)) != 1)
                    throw new InvalidOperationException("A covered GGML allocation is still in flight. Retry disposal after model work and cleanup finish.");
                _handle.Free(); // Native detach proves no callback can still use it.
                lock (_gate) _disposed = true;
            }
            finally { lock (_gate) _disposing = false; }
        }
    }

    private static GgmlCacheBudgetScope Owner(IntPtr context) => (GgmlCacheBudgetScope)GCHandle.FromIntPtr(context).Target!;
    private static ulong ReserveNative(IntPtr context, int rank, int kind, long bytes)
    {
        var owner = Owner(context);
        lock (owner._gate)
        {
            if (owner._callbackError != null || owner._disposing || owner._disposed) return 0;
            BudgetReservation? reservation = null;
            try
            {
                if (bytes <= 0) return 0;
                string[] pools;
                if ((kind == 3 || kind == 6) && owner.IncludesHostBuffers)
                    pools = owner._hostPools;
                else if ((uint)rank < (uint)owner._rankPools.Length
                    && (kind == 0 || kind == 1 || kind == 4 || ((kind == 2 || kind == 5 || kind == 7) && owner.IncludesGraphBuffers)))
                    pools = owner._rankPools[rank];
                else return 0;
                var charges = pools.Select(pool => new MemoryCharge(pool, bytes));
                // A persistent RAM cache never borrows an incidental request
                // envelope or root credit without its explicitly admitted owner.
                if (kind == 6 && owner._sharedHostExpertEnvelope == null) return 0;
                if (kind == 7 && owner._sharedSpeculativeEnvelope == null) return 0;
                var allocationEnvelope = kind == 6 ? owner._sharedHostExpertEnvelope
                    : kind == 7 ? owner._sharedSpeculativeEnvelope
                    : kind == 4 && owner._sharedExpertEnvelope != null
                    ? owner._sharedExpertEnvelope : kind == 5 && owner._sharedJointEnvelope != null
                    ? owner._sharedJointEnvelope : owner._executionEnvelope;
                reservation = allocationEnvelope == null ? owner._budget.TryReserve(charges)
                    : allocationEnvelope.TryTake(charges);
                if (reservation == null)
                {
                    var refusal = new GgmlAllocationRefusal(rank, kind, bytes,
                        allocationEnvelope?.Charges, owner._budget.Snapshot());
                    owner._allocationRefusalCount++;
                    owner._firstAllocationRefusal ??= refusal;
                    owner._lastAllocationRefusal = refusal;
                    if (kind == 5) owner.WriteGraphMemoryDiagnostic("refused", 0, rank, bytes, owner._graphDiagnosticCall);
                    return 0;
                }
                ulong token = checked(++owner._nextToken);
                var allocation = new Allocation(reservation, rank, kind, bytes);
                if (kind == 5 && owner.GraphMemoryDiagnosticWriter != null)
                    allocation.GraphDiagnosticOriginCall = owner._graphDiagnosticCall;
                owner._allocations.Add(token, allocation);
                if (kind == 5) owner.WriteGraphMemoryDiagnostic("reserve", token, rank, bytes, allocation.GraphDiagnosticOriginCall);
                return token;
            }
            catch (Exception ex)
            {
                owner._callbackError ??= ex;
                try { reservation?.Dispose(); } catch (Exception cleanup) { owner._callbackError = cleanup; }
                return 0;
            }
        }
    }
    private static int CommitNative(IntPtr context, ulong token)
    {
        var owner = Owner(context);
        lock (owner._gate)
        {
            try
            {
                var allocation = owner._allocations[token];
                allocation.Reservation.Commit();
                allocation.Committed = true;
                if (allocation.Kind == 5) owner.WriteGraphMemoryDiagnostic("commit", token,
                    allocation.Rank, allocation.Bytes, allocation.GraphDiagnosticOriginCall);
                return 1;
            }
            catch (Exception ex) { owner._callbackError ??= ex; return 0; }
        }
    }
    private static void ReleaseNative(IntPtr context, ulong token) => ReleaseNativeChecked(context, token);

    private static int ReleaseNativeChecked(IntPtr context, ulong token)
    {
        var owner = Owner(context);
        lock (owner._gate)
        {
            try
            {
                if (!owner._allocations.TryGetValue(token, out var reservation)) return 0;
                reservation.Reservation.Dispose();
                owner._allocations.Remove(token);
                if (reservation.Kind == 5) owner.WriteGraphMemoryDiagnostic("release", token,
                    reservation.Rank, reservation.Bytes, reservation.GraphDiagnosticOriginCall);
                return 1;
            }
            // CallbackError blocks new reservations, but cleanup may retry the
            // same retained token. Dispose is idempotent if its ledger release
            // completed before a later notification failed.
            catch (Exception ex) { owner._callbackError ??= ex; return 0; }
        }
    }
}
