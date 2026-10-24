// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

/// <summary>Actual native/managed ticket ABI only; no backend, CUDA allocation,
/// model or VMM mapping is exercised by these cases.</summary>
public sealed class GgmlCheckedCacheBudgetInteropTests
{
    private sealed class Bridge : IDisposable
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int ReserveTicket(int rank, int kind, long bytes, int commit);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int ReleaseTicket();
        private readonly IntPtr _module;
        internal ReserveTicket Reserve { get; }
        internal ReleaseTicket Release { get; }
        internal Bridge()
        {
            _module = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            try
            {
                Reserve = Marshal.GetDelegateForFunctionPointer<ReserveTicket>(NativeLibrary.GetExport(_module, "TSGgml_TestCheckedCacheTicketReserve"));
                Release = Marshal.GetDelegateForFunctionPointer<ReleaseTicket>(NativeLibrary.GetExport(_module, "TSGgml_TestCheckedCacheTicketRelease"));
            }
            catch { NativeLibrary.Free(_module); throw; }
        }
        public void Dispose() => NativeLibrary.Free(_module);
    }

    [GlmSnapshotBoundaryTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ActualCheckedCallbackReturnsDurableHostCreditWithoutDeviceCharge(bool committed)
    {
        using var bridge = new Bridge();
        var budget = new MemoryBudget([new("device", 128), new("host", 256)]);
        using (var scope = new GgmlCacheBudgetScope(budget, [["device"]], false, ["host"], requireCheckedRelease: true))
        {
            Assert.True(scope.IncludesCheckedRelease);
            using var durable = budget.Reserve([new("host", 100)]);
            try
            {
                using (scope.EnterAllocationOwner(durable)) Assert.Equal(1, bridge.Reserve(0, 3, 48, committed ? 1 : 0));
                var usage = Assert.Single(scope.AllocationUsage);
                Assert.Equal(3, usage.Kind);
                Assert.Equal(committed ? 48 : 0, usage.CommittedBytes);
                Assert.Equal(committed ? 0 : 48, usage.PendingBytes);
                Assert.Equal(48, Assert.Single(durable.LiveChildCharges).Bytes);
                Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "device").Committed);
                Assert.Throws<InvalidOperationException>(() => scope.Dispose());
                Assert.Equal(1, bridge.Release());
                Assert.Equal(1, bridge.Release());
                Assert.Equal(0, scope.ActiveAllocations);
                Assert.Equal(0, Assert.Single(durable.LiveChildCharges).Bytes);
                Assert.Null(scope.CallbackError);
            }
            finally { Assert.Equal(1, bridge.Release()); }
        }
        using var legacy = new GgmlCacheBudgetScope(budget, [["device"]], false, ["host"]);
        Assert.False(legacy.IncludesCheckedRelease);
        Assert.Equal(0, bridge.Reserve(0, 3, 16, 1)); // Never reinterpret void release as acknowledgment.
    }

    [GlmNativeHooksFact]
    public void ActualFailedAcknowledgmentKeepsNativeRegistrationAndRetriesExactToken()
    {
        using var bridge = new Bridge();
        var budget = new MemoryBudget([new("device", 128), new("host", 256)]);
        using var scope = new GgmlCacheBudgetScope(budget, [["device"]], false, ["host"], requireCheckedRelease: true);
        using var durable = budget.Reserve([new("host", 100)]);
        try
        {
            using (scope.EnterAllocationOwner(durable)) Assert.Equal(1, bridge.Reserve(0, 3, 48, 1));
            durable.Dispose();
            _ = budget.ChangeSignal;
            var error = new OutOfMemoryException("interop release notification failed after ledger refund");
            budget.BeforeChangeNotification = () => throw error;
            Assert.Equal(0, bridge.Release());
            Assert.Same(error, scope.CallbackError);
            Assert.Equal(1, scope.ActiveAllocations);
            Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "host").Committed);
            Assert.Throws<InvalidOperationException>(() => scope.Dispose());
            Assert.Throws<InvalidOperationException>(() => new GgmlCacheBudgetScope(budget, [["device"]]));
            budget.BeforeChangeNotification = null;
            Assert.Equal(1, bridge.Release());
            Assert.Equal(0, scope.ActiveAllocations);
            Assert.Same(error, scope.CallbackError); // Cleanup recovery does not erase model health failure.
        }
        finally
        {
            budget.BeforeChangeNotification = null;
            Assert.Equal(1, bridge.Release());
        }
    }
}
