// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class UnifiedMemoryGgmlAllocationOwnerTests
{
    /// <summary>Exercise the real managed native-callback methods without native
    /// registration or buffers. No backend/device/model acceptance is implied.</summary>
    private sealed class Callbacks : IDisposable
    {
        private delegate ulong Reserve(IntPtr context, int rank, int kind, long bytes);
        private delegate int Commit(IntPtr context, ulong token);
        private delegate void Release(IntPtr context, ulong token);
        private delegate int ReleaseChecked(IntPtr context, ulong token);
        private readonly Reserve _reserve;
        private readonly Commit _commit;
        private readonly Release _release;
        private readonly ReleaseChecked _releaseChecked;
        private readonly GCHandle _handle;
        internal GgmlCacheBudgetScope Scope { get; }
        internal Exception? ExpectedCallbackError { get; set; }
        private static FieldInfo Field(string name) => typeof(GgmlCacheBudgetScope).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!;
        private static void Set(GgmlCacheBudgetScope scope, string name, object value) => Field(name).SetValue(scope, value);
        private static T Method<T>(string name) where T : Delegate =>
            typeof(GgmlCacheBudgetScope).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<T>();

        internal Callbacks(MemoryBudget budget, bool host = false)
        {
            Scope = (GgmlCacheBudgetScope)RuntimeHelpers.GetUninitializedObject(typeof(GgmlCacheBudgetScope));
            Set(Scope, "_gate", new object()); Set(Scope, "_lifecycleGate", new object());
            Set(Scope, "_budget", budget); Set(Scope, "_rankPools", new[] { new[] { "device" } });
            Set(Scope, "_hostPools", host ? new[] { "host" } : Array.Empty<string>());
            Set(Scope, "_allocations", Activator.CreateInstance(Field("_allocations").FieldType)!);
            Set(Scope, "<IncludesGraphBuffers>k__BackingField", true);
            _reserve = Method<Reserve>("ReserveNative"); _commit = Method<Commit>("CommitNative"); _release = Method<Release>("ReleaseNative");
            _releaseChecked = Method<ReleaseChecked>("ReleaseNativeChecked");
            _handle = GCHandle.Alloc(Scope);
        }
        internal ulong Allocate(long bytes, int kind = 2, bool onWorker = false)
        {
            ulong AllocateCore()
            {
                ulong token = _reserve(GCHandle.ToIntPtr(_handle), 0, kind, bytes);
                Assert.NotEqual(0UL, token); Assert.Equal(1, _commit(GCHandle.ToIntPtr(_handle), token));
                return token;
            }
            if (!onWorker) return AllocateCore();
            ulong result = 0; Exception? failure = null;
            var worker = new Thread(() => { try { result = AllocateCore(); } catch (Exception error) { failure = error; } });
            worker.Start(); Assert.True(worker.Join(TimeSpan.FromSeconds(5)));
            Assert.NotEqual(Environment.CurrentManagedThreadId, worker.ManagedThreadId);
            if (failure != null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(failure).Throw();
            return result;
        }
        internal void Free(ulong token) => _release(GCHandle.ToIntPtr(_handle), token);
        internal int FreeChecked(ulong token) => _releaseChecked(GCHandle.ToIntPtr(_handle), token);
        internal ulong TryReserve(long bytes, int kind) => _reserve(GCHandle.ToIntPtr(_handle), 0, kind, bytes);
        public void Dispose()
        {
            try
            {
                Assert.Equal(0, Scope.ActiveAllocations);
                Assert.Same(ExpectedCallbackError, Scope.CallbackError);
                Assert.Null(Field("_executionEnvelope").GetValue(Scope));
                Assert.Null(Field("_allocationOwner").GetValue(Scope));
            }
            finally { Set(Scope, "_disposed", true); _handle.Free(); }
        }
    }
    private static BudgetReservation Owner(MemoryBudget budget, long bytes = 100) => budget.Reserve([new("device", bytes)]);
    private static long Children(BudgetReservation owner) => Assert.Single(owner.LiveChildCharges).Bytes;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CheckedReleaseAcknowledgesPendingAndCommittedExactTokens(bool committed)
    {
        var budget = new MemoryBudget([new("device", 200), new("host", 200)]);
        using var callbacks = new Callbacks(budget, host: true);
        using var persistent = budget.Reserve([new("host", 100)]);
        ulong token;
        using (callbacks.Scope.EnterAllocationOwner(persistent))
            token = committed ? callbacks.Allocate(48, kind: 3, onWorker: true) : callbacks.TryReserve(48, 3);
        Assert.NotEqual(0UL, token);
        Assert.Equal(48, Children(persistent));
        Assert.Equal(0, callbacks.FreeChecked(token + 100));
        Assert.Equal(1, callbacks.Scope.ActiveAllocations);
        Assert.Equal(1, callbacks.FreeChecked(token));
        Assert.Equal(0, Children(persistent));
        Assert.Equal(0, callbacks.Scope.ActiveAllocations);
        Assert.Equal(0, callbacks.FreeChecked(token));
    }

    [Fact]
    public void CheckedReleaseCanRetrySameTokenAfterLedgerReleaseButNotificationFailure()
    {
        var budget = new MemoryBudget([new("device", 200), new("host", 200)]);
        using var callbacks = new Callbacks(budget, host: true);
        using var persistent = budget.Reserve([new("host", 100)]);
        ulong token;
        using (callbacks.Scope.EnterAllocationOwner(persistent)) token = callbacks.Allocate(48, kind: 3);
        persistent.Dispose(); // A live child remains charged after its owner closes.
        Assert.Equal(48, budget.Snapshot().Single(pool => pool.Pool == "host").Committed);
        _ = budget.ChangeSignal;
        var failure = new OutOfMemoryException("checked release notification failure after refund");
        budget.BeforeChangeNotification = () => throw failure;
        callbacks.ExpectedCallbackError = failure;
        Assert.Equal(0, callbacks.FreeChecked(token));
        Assert.Same(failure, callbacks.Scope.CallbackError);
        Assert.Equal(1, callbacks.Scope.ActiveAllocations); // Keep original token until acknowledged.
        Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "host").Committed);
        Assert.Equal(0UL, callbacks.TryReserve(1, 3)); // Sticky health failure blocks new work.
        budget.BeforeChangeNotification = null;
        Assert.Equal(1, callbacks.FreeChecked(token)); // Idempotent reservation disposal, then token removal.
        Assert.Equal(0, callbacks.Scope.ActiveAllocations);
        Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "host").Committed);
        Assert.Same(failure, callbacks.Scope.CallbackError);
        Assert.Equal(0, callbacks.FreeChecked(token));
    }

    private sealed class CountingDisposal(IDisposable? inner = null, bool failAfterDispose = false) : IDisposable
    {
        internal int Count { get; private set; }
        public void Dispose()
        {
            ++Count; inner?.Dispose();
            if (failAfterDispose) throw new IOException("batch host scope disposal failure");
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void BatchExecutionEndsDiagnosticAttributionOnSuccessAndException(bool failBody, bool failHostDispose)
    {
        var budget = new MemoryBudget([new("device", 200)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget);
        var lines = new List<string>();
        callbacks.Scope.GraphMemoryDiagnosticWriter = lines.Add;
        var host = new CountingDisposal(failAfterDispose: failHostDispose);
        var native = new CountingDisposal(callbacks.Scope.EnterExecution(request));
        var scopeType = typeof(GlmUnifiedMemoryModelSession).GetNestedType("OwnedExecution", BindingFlags.NonPublic)!;
        var batch = (IDisposable)Activator.CreateInstance(scopeType,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic, null,
            [host, native, callbacks.Scope], null)!;
        ulong retained = 0;
        void Execute()
        {
            using (batch)
            {
                callbacks.Scope.SetGraphMemoryDiagnosticContext(33, 2, 0, 0, false);
                retained = callbacks.Allocate(20, kind: 5, onWorker: true);
                if (failBody) throw new IOException("batch body failure");
            }
        }
        if (failBody || failHostDispose) Assert.Throws<IOException>(Execute);
        else Execute();
        Assert.Equal(1, host.Count);
        Assert.Equal(1, native.Count);
        Assert.Single(lines, line => line.Contains("event=call-end call=1 "));
        Assert.Equal(20, Children(request));
        callbacks.Free(retained);
        Assert.Contains(lines, line => line.Contains("event=release call=0 ")
            && line.Contains("origin_call=1 ") && line.Contains("owned_bytes=0 "));
        Assert.Equal(0, Children(request));
    }

    [Fact]
    public void GraphDiagnosticsDistinguishPendingCommitReleaseAndOriginalCallWithoutChangingOwnership()
    {
        var budget = new MemoryBudget([new("device", 200)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget);
        using var execution = callbacks.Scope.EnterExecution(request);
        var lines = new List<string>();
        callbacks.Scope.GraphMemoryDiagnosticWriter = lines.Add;
        callbacks.Scope.SetGraphMemoryDiagnosticContext(1872, 512, 0, 1872, true);
        ulong committed = callbacks.Allocate(20, kind: 5, onWorker: true);
        ulong pending = callbacks.TryReserve(30, kind: 5);
        Assert.NotEqual(0UL, pending);
        ulong otherKind = callbacks.Allocate(7, kind: 2);
        Assert.Equal(57, Children(request));
        Assert.Contains(lines, line => line.Contains("event=reserve ") && line.Contains("bytes=20 ")
            && line.Contains("pending_bytes=20 committed_bytes=0 owned_bytes=20"));
        Assert.Contains(lines, line => line.Contains("event=commit ")
            && line.Contains("pending_bytes=0 committed_bytes=20 owned_bytes=20"));
        Assert.Contains(lines, line => line.Contains("event=reserve ")
            && line.Contains("pending_bytes=30 committed_bytes=20 owned_bytes=50 peak_owned_bytes=50"));
        Assert.All(lines, line => Assert.Contains("admitted_rows=512 output_rows=0 submitted_tokens=1872 layer_major=1", line));
        callbacks.Scope.SetGraphMemoryDiagnosticContext(1873, 1, 0, 1, false);
        callbacks.Free(pending);
        Assert.Contains(lines, line => line.Contains("event=release call=2 ") && line.Contains("origin_call=1 ")
            && line.Contains("pending_bytes=0 committed_bytes=20 owned_bytes=20 peak_owned_bytes=50"));
        callbacks.Scope.ClearGraphMemoryDiagnosticContext();
        callbacks.Free(committed); callbacks.Free(otherKind);
        Assert.Contains(lines, line => line.Contains("event=release call=0 ") && line.Contains("origin_call=1 ")
            && line.Contains("pending_bytes=0 committed_bytes=0 owned_bytes=0 peak_owned_bytes=50"));
        Assert.DoesNotContain(lines, line => line.Contains("kind=2"));
        Assert.Equal(0, Children(request));
    }

    [Fact]
    public void FailingGraphDiagnosticWriterCannotChangeSuccessfulCallbacksOrQuotaRefusal()
    {
        var budget = new MemoryBudget([new("device", 200)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget);
        using var execution = callbacks.Scope.EnterExecution(request);
        callbacks.Scope.GraphMemoryDiagnosticWriter = _ => throw new IOException("diagnostic writer failure");
        callbacks.Scope.SetGraphMemoryDiagnosticContext(512, 512, 0, 512, false);
        ulong token = callbacks.Allocate(80, kind: 5, onWorker: true);
        Assert.Equal(0UL, callbacks.TryReserve(21, kind: 5));
        Assert.Equal(80, Children(request));
        Assert.Null(callbacks.Scope.CallbackError);
        callbacks.Free(token);
        callbacks.Scope.ClearGraphMemoryDiagnosticContext();
        Assert.Null(callbacks.Scope.CallbackError);
        Assert.Equal(0, Children(request));
    }

    [Fact]
    public void UnconfiguredGraphDiagnosticsLeaveCallbacksUnlabelledAndAccountingUnchanged()
    {
        var budget = new MemoryBudget([new("device", 200)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget);
        using var execution = callbacks.Scope.EnterExecution(request);
        callbacks.Scope.SetGraphMemoryDiagnosticContext(512, 512, 0, 512, true);
        ulong silent = callbacks.Allocate(20, kind: 5);
        callbacks.Free(silent);
        var lines = new List<string>();
        callbacks.Scope.GraphMemoryDiagnosticWriter = lines.Add;
        callbacks.Scope.SetGraphMemoryDiagnosticContext(256, 1, 0, 1, false);
        ulong observed = callbacks.Allocate(30, kind: 5);
        callbacks.Free(observed);
        callbacks.Scope.ClearGraphMemoryDiagnosticContext();
        Assert.All(lines, line => Assert.Contains("call=1 ", line));
        Assert.Equal(0, Children(request));
    }

    [Fact]
    public void WorkerCheckpointAllocationsKeepPersistentOwnershipAfterRequestRelease()
    {
        var budget = new MemoryBudget([new("device", 256)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget); using var checkpoint = Owner(budget);
        ulong requestA, requestB, retained;
        using (callbacks.Scope.EnterExecution(request))
        {
            requestA = callbacks.Allocate(20, onWorker: true);
            using (callbacks.Scope.EnterAllocationOwner(checkpoint))
            {
                retained = callbacks.Allocate(64, onWorker: true);
                Assert.Equal(64, Children(checkpoint)); Assert.Equal(20, Children(request));
            }
            requestB = callbacks.Allocate(16, onWorker: true);
            Assert.Equal(36, Children(request)); Assert.Equal(64, Children(checkpoint));
        }
        request.Dispose();
        Assert.Equal(36, Assert.Single(request.OwnedCharges).Bytes);
        Assert.Equal(100, Assert.Single(checkpoint.OwnedCharges).Bytes);
        callbacks.Free(requestA); callbacks.Free(requestB);
        Assert.Equal(0, Assert.Single(request.OwnedCharges).Bytes);
        checkpoint.Dispose(); // Closing its unused allowance cannot refund the live copy.
        Assert.Equal(64, Assert.Single(budget.Snapshot()).Committed);
        callbacks.Free(retained);
        Assert.Equal(0, Assert.Single(budget.Snapshot()).Committed);
        Assert.Equal(0, Assert.Single(budget.Snapshot()).Reserved);
    }

    [Fact]
    public void OutOfOrderOwnerAndParentDisposalCannotClearTheActiveOverride()
    {
        var budget = new MemoryBudget([new("device", 300)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget); using var outer = Owner(budget); using var inner = Owner(budget);
        using var execution = callbacks.Scope.EnterExecution(request);
        using var a = callbacks.Scope.EnterAllocationOwner(outer);
        using var b = callbacks.Scope.EnterAllocationOwner(inner);
        Assert.Throws<InvalidOperationException>(() => a.Dispose());
        Assert.Throws<InvalidOperationException>(() => execution.Dispose());
        Assert.Throws<InvalidOperationException>(() => callbacks.Scope.Dispose()); // Rejected before native detach.
        ulong innerToken = callbacks.Allocate(12); Assert.Equal(12, Children(inner));
        Assert.Equal(0, Children(outer)); Assert.Equal(0, Children(request));
        b.Dispose();
        ulong outerToken = callbacks.Allocate(13); Assert.Equal(13, Children(outer));
        a.Dispose();
        ulong requestToken = callbacks.Allocate(14); Assert.Equal(14, Children(request));
        execution.Dispose(); a.Dispose(); b.Dispose(); // Idempotent after valid restoration.
        callbacks.Free(innerToken); callbacks.Free(outerToken); callbacks.Free(requestToken);
    }

    [Fact]
    public void OverrideHandlePreparationOomPreservesThePreviousOwnerAndAllowsRetry()
    {
        var budget = new MemoryBudget([new("device", 300)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget); using var outer = Owner(budget); using var candidate = Owner(budget);
        using var execution = callbacks.Scope.EnterExecution(request);
        using var current = callbacks.Scope.EnterAllocationOwner(outer);
        var before = budget.Snapshot().ToArray();
        callbacks.Scope.BeforeAllocationOwnerAllocation = () => throw new OutOfMemoryException("override owner injection");
        var error = Assert.Throws<OutOfMemoryException>(() => callbacks.Scope.EnterAllocationOwner(candidate));
        Assert.Equal("override owner injection", error.Message); Assert.Equal(before, budget.Snapshot().ToArray());
        ulong previous = callbacks.Allocate(11); Assert.Equal(11, Children(outer)); Assert.Equal(0, Children(candidate));
        callbacks.Scope.BeforeAllocationOwnerAllocation = null;
        using (callbacks.Scope.EnterAllocationOwner(candidate))
        {
            ulong retry = callbacks.Allocate(17); Assert.Equal(17, Children(candidate)); callbacks.Free(retry);
        }
        ulong restored = callbacks.Allocate(7); Assert.Equal(18, Children(outer));
        callbacks.Free(previous); callbacks.Free(restored);
    }

    [Fact]
    public void ForeignOwnerIsRejectedAndExecutionStillCannotNest()
    {
        var budget = new MemoryBudget([new("device", 200)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget); using var checkpoint = Owner(budget);
        using var foreign = Owner(new MemoryBudget([new("device", 100)]));
        Assert.Throws<ArgumentException>(() => callbacks.Scope.EnterAllocationOwner(foreign));
        using (callbacks.Scope.EnterAllocationOwner(checkpoint)) // Quiescent maintenance; previous owner is null.
        {
            Assert.Throws<InvalidOperationException>(() => callbacks.Scope.EnterExecution(request));
            ulong maintenance = callbacks.Allocate(9); Assert.Equal(9, Children(checkpoint)); callbacks.Free(maintenance);
        }
        using (callbacks.Scope.EnterExecution(request))
        {
            Assert.Throws<ArgumentException>(() => callbacks.Scope.EnterAllocationOwner(foreign));
            Assert.Throws<InvalidOperationException>(() => callbacks.Scope.EnterExecution(request));
            using (callbacks.Scope.EnterAllocationOwner(checkpoint))
                Assert.Throws<InvalidOperationException>(() => callbacks.Scope.EnterExecution(request));
            ulong restored = callbacks.Allocate(8); Assert.Equal(8, Children(request)); callbacks.Free(restored);
        }
        callbacks.Scope.BeforeExecutionOwnerAllocation = () => throw new OutOfMemoryException("execution owner injection");
        Assert.Throws<OutOfMemoryException>(() => callbacks.Scope.EnterExecution(request));
        callbacks.Scope.BeforeExecutionOwnerAllocation = null;
        using (callbacks.Scope.EnterExecution(request)) { }
    }

    [Fact]
    public void SuccessfulRestoreAllocatesNoOwnerAndSharedKindsKeepTheirDedicatedEnvelopes()
    {
        var budget = new MemoryBudget([new("device", 400)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget); using var checkpoint = Owner(budget);
        using var experts = Owner(budget); using var joint = Owner(budget);
        callbacks.Scope.SetSharedExpertEnvelope(experts); callbacks.Scope.SetSharedJointEnvelope(joint);
        using (callbacks.Scope.EnterExecution(request))
        {
            for (int i = 0; i < 32; i++) { using var warm = callbacks.Scope.EnterAllocationOwner(checkpoint); }
            var owner = callbacks.Scope.EnterAllocationOwner(checkpoint);
            ulong state = callbacks.Allocate(21), expert = callbacks.Allocate(22, 4), arena = callbacks.Allocate(23, 5);
            Assert.Equal(21, Children(checkpoint)); Assert.Equal(22, Children(experts)); Assert.Equal(23, Children(joint));
            Assert.Equal(0, Children(request));
            long before = GC.GetAllocatedBytesForCurrentThread();
            owner.Dispose();
            long restoredBytes = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.Equal(0, restoredBytes);
            ulong restored = callbacks.Allocate(24); Assert.Equal(24, Children(request));
            callbacks.Free(state); callbacks.Free(expert); callbacks.Free(arena); callbacks.Free(restored);
        }
        callbacks.Scope.SetSharedExpertEnvelope(null); callbacks.Scope.SetSharedJointEnvelope(null);
    }

    [Fact]
    public void RamExpertPayloadKeepsPersistentHostOwnerWhileStagingUsesTheRequest()
    {
        var budget = new MemoryBudget([new("device", 200), new("host", 300)]);
        using var callbacks = new Callbacks(budget, host: true);
        using var request = budget.Reserve([new("host", 40), new("device", 80)]);
        using var retained = budget.Reserve([new("host", 200)]);
        Assert.Equal(0UL, callbacks.TryReserve(10, 6)); // No implicit root/request fallback.
        callbacks.Scope.SetSharedHostExpertEnvelope(retained);
        ulong payload, staging;
        using (callbacks.Scope.EnterExecution(request))
        {
            payload = callbacks.Allocate(150, 6, onWorker: true);
            staging = callbacks.Allocate(32, 3, onWorker: true);
            Assert.Equal(150, Children(retained));
            Assert.Equal(32, request.LiveChildCharges.Single(c => c.Pool == "host").Bytes);
            Assert.Throws<InvalidOperationException>(() => callbacks.Scope.SetSharedHostExpertEnvelope(null));
        }
        request.Dispose(); callbacks.Free(staging);
        Assert.Equal(150, budget.Snapshot().Single(p => p.Pool == "host").Committed);
        Assert.Throws<InvalidOperationException>(() => callbacks.Scope.SetSharedHostExpertEnvelope(null));
        callbacks.Free(payload);
        callbacks.Scope.SetSharedHostExpertEnvelope(null);
        Assert.Equal(0UL, callbacks.TryReserve(10, 6));
        Assert.Null(callbacks.Scope.CallbackError);
    }

    [Fact]
    public void SpeculativeGraphsAndSnapshotsCannotBorrowRequestOrUnownedRootCredit()
    {
        var budget = new MemoryBudget([new("device", 400)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget, 100);
        using var retained = Owner(budget, 200);
        using (callbacks.Scope.EnterExecution(request))
            Assert.Equal(0UL, callbacks.TryReserve(10, 7));
        callbacks.Scope.SetSharedSpeculativeEnvelope(retained);
        ulong graph, snapshot;
        using (callbacks.Scope.EnterExecution(request))
        {
            graph = callbacks.Allocate(80, 7, onWorker: true);
            snapshot = callbacks.Allocate(90, 7, onWorker: true);
            Assert.Equal(0, Children(request)); Assert.Equal(170, Children(retained));
            Assert.Equal(0UL, callbacks.TryReserve(31, 7));
            Assert.Throws<InvalidOperationException>(() => callbacks.Scope.SetSharedSpeculativeEnvelope(null));
        }
        request.Dispose();
        Assert.Equal(170, Assert.Single(budget.Snapshot()).Committed);
        Assert.Throws<InvalidOperationException>(() => callbacks.Scope.SetSharedSpeculativeEnvelope(null));
        callbacks.Free(graph); callbacks.Free(snapshot);
        callbacks.Scope.SetSharedSpeculativeEnvelope(null);
        Assert.Equal(0UL, callbacks.TryReserve(10, 7));
    }
}
