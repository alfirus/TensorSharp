// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Real ownership transactions and injected synchronous native release.
/// These tests cover the lending policy, not CUDA or full-model qualification.</summary>
public sealed class GlmUnifiedExpertLendingTests
{
    private const long MiB = 1L << 20;
    private const string Host = UnifiedMemoryModelSession.HostPool, Device = UnifiedMemoryModelSession.DevicePool;
    private static MemoryCharge[] Charges(long host, long device) => [new(Host, host), new(Device, device)];
    private static MemoryCharge[] DeviceCharges(long bytes) => [new(Device, bytes)];
    private static long DeviceBytes(BudgetReservation owner) => owner.OwnedCharges.Single(c => c.Pool == Device).Bytes;
    private static GlmUnifiedPhysicalObservation Observe(MemoryBudget budget, long limit = 12000 * MiB)
        => new(default, budget.Snapshot(), Charges(12000 * MiB, limit), 0, 0);

    private sealed class Native : IHostMoeExpertCacheScopeNative
    {
        internal long Budget;
        internal Action<long>? OnUpdate;
        public ulong Begin(long bytes, int rows, int layers, int decodeRows) { Budget = bytes; return 1; }
        public long Update(ulong token, long bytes) { OnUpdate?.Invoke(bytes); Budget = bytes; return 0; }
        public long Trim(ulong token, long bytes) => 0;
        public long End(ulong token) => 0;
        public GgmlHostMoeExpertCacheUsage ReadUsage() => new(0, Budget, 0, 0, 0);
    }

    [Fact]
    public void MaximumIsOnlyPermissionAndKeepsStartupBudgetAndExplicitOperatorCap()
    {
        Assert.Equal(8000 * MiB, GlmUnifiedExpertLending.MaximumBudget(4000 * MiB,
            12000 * MiB, 4000 * MiB, long.MaxValue, true));
        Assert.Equal(5000 * MiB, GlmUnifiedExpertLending.MaximumBudget(4000 * MiB,
            12000 * MiB, 4000 * MiB, 5000 * MiB, true));
        Assert.Equal(4000 * MiB, GlmUnifiedExpertLending.MaximumBudget(4000 * MiB,
            12000 * MiB, 4000 * MiB, long.MaxValue, false));
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var envelope = budget.Reserve(DeviceCharges(4000 * MiB));
        using var cache = new GgmlHostMoeExpertCacheScope(8000 * MiB, 1024, 3, new Native(), 3);
        cache.UpdateBudget(4000 * MiB); // Before phase planning or expert payload.
        Assert.Equal(4000 * MiB, cache.BudgetBytes);
        Assert.Equal(8000 * MiB, cache.MaximumBudgetBytes);
        Assert.Equal(4000 * MiB, DeviceBytes(envelope));
        Assert.Equal(8000 * MiB, budget.Snapshot().Single(p => p.Pool == Device).Available);
    }

    [Fact]
    public void OneToThreeToOneRequestsRepayAndRecoverWithoutFreeingLiveSlotsOrGraphs()
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var dense = budget.Reserve(DeviceCharges(4000 * MiB)); dense.Commit();
        using var primary = budget.Reserve(DeviceCharges(500 * MiB)); primary.Commit();
        using var joint = budget.Reserve(Charges(100 * MiB, 700 * MiB));
        using var liveGraph = joint.TryTake(Charges(50 * MiB, 500 * MiB))!; liveGraph.Commit();
        using var experts = budget.Reserve(DeviceCharges(4000 * MiB));
        using var retained = experts.TryTake(DeviceCharges(3500 * MiB))!; retained.Commit();
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(8000 * MiB, 1024, 3, native, 3);
        cache.UpdateBudget(4000 * MiB);
        var physical = Observe(budget);
        long solo = GlmUnifiedExpertLending.RecoveryTarget(physical, cache.BudgetBytes, cache.MaximumBudgetBytes);
        Assert.Equal(6672 * MiB, solo);
        Assert.True(UnifiedMemoryExpertCapacity.Resize(cache, experts, Device, solo, 2000 * MiB, physical.Limits));
        using var loanPayload = experts.TryTake(DeviceCharges(2500 * MiB))!; loanPayload.Commit();

        // The two added slots and N=3 graph coexist with the original request.
        physical = Observe(budget);
        Assert.True(GlmUnifiedPrefillAdmission.TrySelect(physical.Ledger, physical.Limits,
            new(100 * MiB, 700 * MiB), default, cache.BudgetBytes, 2000 * MiB, 1,
            _ => (new InferenceMemoryBytes(100 * MiB, 1300 * MiB), new(0, 1000 * MiB)), out var selected));
        Assert.Equal(5072 * MiB, selected.ExpertTargetBytes);
        native.OnUpdate = target =>
        {
            Assert.Equal(solo, DeviceBytes(experts)); // No refund before native fence/free.
            Assert.Equal(6000 * MiB, experts.LiveChildCharges.Single().Bytes);
            loanPayload.Dispose();
        };
        Assert.True(UnifiedMemoryExpertCapacity.Resize(cache, experts, Device,
            selected.ExpertTargetBytes, 2000 * MiB, physical.Limits));
        native.OnUpdate = null;
        physical = Observe(budget);
        Assert.True(physical.Fits(Charges(0, 1600 * MiB)));
        Assert.True(joint.TryResizeTotal(Charges(100 * MiB, 1300 * MiB), physical.Limits));
        using var addedSlots = budget.Reserve(DeviceCharges(1000 * MiB)); addedSlots.Commit();
        Assert.Equal(500 * MiB, DeviceBytes(primary));
        Assert.Equal(500 * MiB, DeviceBytes(liveGraph));

        addedSlots.Dispose();
        Assert.True(joint.TryResizeTotal(Charges(100 * MiB, 700 * MiB)));
        physical = Observe(budget);
        long recovered = GlmUnifiedExpertLending.RecoveryTarget(physical, cache.BudgetBytes, cache.MaximumBudgetBytes);
        Assert.Equal(solo, recovered);
        Assert.True(UnifiedMemoryExpertCapacity.Resize(cache, experts, Device, recovered, 2000 * MiB, physical.Limits));
        Assert.Equal(3500 * MiB, experts.LiveChildCharges.Single().Bytes);
        Assert.Equal(500 * MiB, DeviceBytes(liveGraph));
        Assert.Equal(128 * MiB, budget.Snapshot().Single(p => p.Pool == Device).Available);
    }

    [Theory]
    [InlineData(1000)] // Streamed image preparation.
    [InlineData(700)]  // Optional independent prefix copy.
    public void ImageAndPrefixRepayOnlyBorrowedCapacityBeforeTheirOwnReservation(long neededMiB)
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var fixedOwners = budget.Reserve(Charges(100 * MiB, 5200 * MiB));
        using var experts = budget.Reserve(DeviceCharges(6672 * MiB));
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(8000 * MiB, 1024, 3, native, 3);
        cache.UpdateBudget(6672 * MiB);
        using var active = experts.TryTake(DeviceCharges(3500 * MiB))!; active.Commit();
        using var loan = experts.TryTake(DeviceCharges(2500 * MiB))!; loan.Commit();
        native.OnUpdate = _ => loan.Dispose();
        var needed = Charges(200 * MiB, neededMiB * MiB);
        var physical = Observe(budget);
        long target = GlmUnifiedExpertLending.RepaymentTarget(physical, needed, cache.BudgetBytes, 4000 * MiB);
        Assert.InRange(target, 4000 * MiB, cache.BudgetBytes - 1);
        Assert.True(UnifiedMemoryExpertCapacity.Resize(cache, experts, Device, target, 2000 * MiB, physical.Limits));
        physical = Observe(budget);
        using var imageOrCopy = budget.TryReserve(needed, physical.Limits);
        Assert.NotNull(imageOrCopy);
        Assert.Equal(3500 * MiB, experts.LiveChildCharges.Single().Bytes);
        Assert.Equal(5200 * MiB, DeviceBytes(fixedOwners));
    }

    [Fact]
    public void LaterGraphGrowthRepaysIncrementalDemandAndPreservesReplacementChildren()
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var fixedOwners = budget.Reserve(Charges(100 * MiB, 4500 * MiB));
        using var joint = budget.Reserve(Charges(100 * MiB, 700 * MiB));
        using var live = joint.TryTake(Charges(50 * MiB, 500 * MiB))!; live.Commit();
        using var experts = budget.Reserve(DeviceCharges(6672 * MiB));
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(8000 * MiB, 1024, 3, native, 3);
        cache.UpdateBudget(6672 * MiB);
        var target = GlmUnifiedPrefillAdmission.PreserveReplacement(new(150 * MiB, 1800 * MiB),
            new(50 * MiB, 500 * MiB), 2);
        var growth = Charges(target.Host - 100 * MiB, target.Device - 700 * MiB);
        var physical = Observe(budget);
        long repay = GlmUnifiedExpertLending.RepaymentTarget(physical, growth, cache.BudgetBytes, 2000 * MiB);
        Assert.Equal(5572 * MiB, repay);
        Assert.True(UnifiedMemoryExpertCapacity.Resize(cache, experts, Device, repay, 2000 * MiB, physical.Limits));
        Assert.True(joint.TryResizeTotal(Charges(target.Host, target.Device), Observe(budget).Limits));
        Assert.Equal(500 * MiB, DeviceBytes(live));
        Assert.Equal(128 * MiB, budget.Snapshot().Single(p => p.Pool == Device).Available);
    }

    [Fact]
    public void FreshPhysicalLossAfterRepaymentStillRefusesImageAndKeepsExistingOwnership()
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var fixedOwners = budget.Reserve(Charges(100 * MiB, 5200 * MiB));
        using var experts = budget.Reserve(DeviceCharges(6672 * MiB));
        using var cache = new GgmlHostMoeExpertCacheScope(8000 * MiB, 1024, 3, new Native(), 3);
        cache.UpdateBudget(6672 * MiB);
        var needed = Charges(200 * MiB, 1000 * MiB);
        var physical = Observe(budget);
        long target = GlmUnifiedExpertLending.RepaymentTarget(physical, needed, cache.BudgetBytes, 4000 * MiB);
        Assert.True(UnifiedMemoryExpertCapacity.Resize(cache, experts, Device, target, 2000 * MiB, physical.Limits));
        var fresh = Observe(budget, 11000 * MiB); // Another untracked owner consumed the released allowance.
        var before = budget.Snapshot().ToArray();
        Assert.False(fresh.Fits(needed));
        Assert.Throws<MemoryPressureException>(() => budget.TryReserve(needed, fresh.Limits));
        Assert.Equal(before, budget.Snapshot().ToArray());
        Assert.Equal(5200 * MiB, DeviceBytes(fixedOwners));
    }

    [Fact]
    public void RecoveryHonorsPhysicalCapacityAndPendingPromisesWithoutDoubleCountingChildren()
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var futureSlotsAndGraphs = budget.Reserve(Charges(100 * MiB, 3000 * MiB));
        using var experts = budget.Reserve(DeviceCharges(4000 * MiB));
        using var child = experts.TryTake(DeviceCharges(3000 * MiB))!; child.Commit();
        Assert.Equal(4872 * MiB, GlmUnifiedExpertLending.RecoveryTarget(Observe(budget, 8000 * MiB),
            4000 * MiB, 8000 * MiB));
        Assert.Equal(4000 * MiB, GlmUnifiedExpertLending.RecoveryTarget(Observe(budget, 7500 * MiB),
            4000 * MiB, 8000 * MiB)); // Below the useful-growth threshold.
        Assert.Equal(4000 * MiB, GlmUnifiedExpertLending.RecoveryTarget(Observe(budget),
            4000 * MiB, 4000 * MiB)); // Explicit growth disabled.
    }

    [Theory]
    [InlineData(true, 388, 4500, 7872, true)]
    [InlineData(false, 388, 4500, 7872, true)]
    [InlineData(true, 100, 4500, 7872, false)]
    [InlineData(false, 100, 4500, 7872, false)]
    [InlineData(true, 388, 3500, 6872, true)]
    [InlineData(true, 388, 500, 4000, false)]
    public void DeviceRecoveryRetainsGlobalHostBarrierAndRecomputesFromFreshAllowance(bool hintResult,
        long freshHostMiB, long freshDeviceMiB, long expectedTargetMiB, bool expectedResize)
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var viewPromise = budget.Reserve([new(Host, 400 * MiB)]);
        using var viewControl = viewPromise.TryTake([new(Host, 12 * MiB)])!; viewControl.Commit();
        using var dense = budget.Reserve(DeviceCharges(4000 * MiB)); dense.Commit();
        using var experts = budget.Reserve(DeviceCharges(4000 * MiB));
        using var payload = experts.TryTake(DeviceCharges(3500 * MiB))!; payload.Commit();
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(8000 * MiB, 512, 3, native, 3);
        cache.UpdateBudget(4000 * MiB);
        int updates = 0, hints = 0, samples = 0;
        native.OnUpdate = target => { updates++; Assert.Equal(target, DeviceBytes(experts)); };
        var before = budget.Snapshot().ToArray();
        var physical = GlmUnifiedPhysicalObservation.Create(new(20000 * MiB, 100 * MiB,
            16000 * MiB, 4500 * MiB), before, 12000 * MiB, 12000 * MiB, 0, 0);
        long proposed = GlmUnifiedExpertLending.RecoveryTarget(physical, cache.BudgetBytes, cache.MaximumBudgetBytes);
        Assert.Equal(7872 * MiB, proposed);
        Assert.Equal(0, physical.Shortfall(Device, Charges(0, proposed - cache.BudgetBytes)));
        Assert.Equal(288 * MiB, physical.Shortfall(Host, Charges(0, proposed - cache.BudgetBytes)));
        // Real single-pool Device growth must still reject an unchanged global
        // Host owner under the supplied Host limit; no native update can run.
        bool resized = UnifiedMemoryExpertCapacity.Resize(cache, experts, Device, proposed, 2000 * MiB, physical.Limits);
        Assert.False(resized); Assert.Equal(0, updates);
        Assert.Equal(before, budget.Snapshot().ToArray());
        physical = physical with { Ledger = budget.Snapshot() };
        Assert.True(GlmUnifiedExpertLending.TryRefreshSourceViewRecovery(true, resized, physical,
            cache.BudgetBytes, proposed, cache.MaximumBudgetBytes, 512 * MiB,
            () => { hints++; return hintResult; }, () =>
            {
                samples++;
                return GlmUnifiedPhysicalObservation.Create(new(20000 * MiB, freshHostMiB * MiB,
                    16000 * MiB, freshDeviceMiB * MiB), budget.Snapshot(), 12000 * MiB, 12000 * MiB, 0, 0);
            }, out var fresh, out long target, out var hinted));
        Assert.Equal(expectedTargetMiB * MiB, target); Assert.Equal(hintResult, hinted);
        Assert.Equal(1, hints); Assert.Equal(1, samples);
        Assert.Equal(before, budget.Snapshot().ToArray()); // Hint/sample themselves reserve/refund nothing.
        if (target > cache.BudgetBytes)
            resized = UnifiedMemoryExpertCapacity.Resize(cache, experts, Device, target, 2000 * MiB, fresh.Limits);
        Assert.Equal(expectedResize, resized); Assert.Equal(expectedResize ? 1 : 0, updates);
        Assert.Equal((expectedResize ? expectedTargetMiB : 4000) * MiB, cache.BudgetBytes);
        Assert.Equal(cache.BudgetBytes, DeviceBytes(experts));
        Assert.Equal(before.Single(x => x.Pool == Host), budget.Snapshot().Single(x => x.Pool == Host));
        Assert.Equal(400 * MiB, viewPromise.OwnedCharges.Single().Bytes);
        Assert.Equal(12 * MiB, viewPromise.LiveChildCharges.Single().Bytes);
        Assert.Equal(3500 * MiB, experts.LiveChildCharges.Single().Bytes);
        Assert.Equal(4000 * MiB, DeviceBytes(dense));
    }

    [Theory]
    [InlineData(false, false, 7872, 100, 4500)] // Immutable mode OFF.
    [InlineData(true, true, 7872, 100, 4500)]   // Initial resize already succeeded.
    [InlineData(true, false, 4000, 100, 4500)]  // No Device growth proposal.
    [InlineData(true, false, 7872, 100, 3500)]  // Device growth no longer fits.
    [InlineData(true, false, 7872, 10000, 500)] // Device-only deficit.
    [InlineData(true, false, 7872, 10000, 4500)]// No Host shortfall.
    public void SourceViewRecoveryHintRequiresFailedGrowthAndHostOnlyShortfall(bool sourceView,
        bool resizeSucceeded, long proposedMiB, long hostMiB, long deviceMiB)
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 12000 * MiB, 388 * MiB, 12 * MiB),
            new(Device, 12000 * MiB, 500 * MiB, 7500 * MiB) };
        var physical = GlmUnifiedPhysicalObservation.Create(new(20000 * MiB, hostMiB * MiB,
            16000 * MiB, deviceMiB * MiB), ledger, 12000 * MiB, 12000 * MiB, 0, 0);
        int hints = 0, samples = 0;
        Assert.False(GlmUnifiedExpertLending.TryRefreshSourceViewRecovery(sourceView, resizeSucceeded,
            physical, 4000 * MiB, proposedMiB * MiB, 8000 * MiB, 512 * MiB,
            () => { hints++; return true; }, () => { samples++; return physical; },
            out var refreshed, out long target, out var hinted));
        Assert.Equal(0, hints); Assert.Equal(0, samples); Assert.Null(hinted);
        Assert.Equal(proposedMiB * MiB, target); Assert.Equal(physical, refreshed);
    }

    [Fact]
    public void FatalCheckedRecoveryHintNeverSamplesOrChangesRealOwnersOrNativeCapacity()
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var viewPromise = budget.Reserve([new(Host, 400 * MiB)]);
        using var control = viewPromise.TryTake([new(Host, 12 * MiB)])!; control.Commit();
        using var dense = budget.Reserve(DeviceCharges(4000 * MiB)); dense.Commit();
        using var experts = budget.Reserve(DeviceCharges(4000 * MiB));
        using var payload = experts.TryTake(DeviceCharges(3500 * MiB))!; payload.Commit();
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(8000 * MiB, 512, 3, native, 3);
        cache.UpdateBudget(4000 * MiB);
        int updates = 0, samples = 0;
        native.OnUpdate = _ => updates++;
        var before = budget.Snapshot().ToArray();
        var physical = GlmUnifiedPhysicalObservation.Create(new(20000 * MiB, 100 * MiB,
            16000 * MiB, 4500 * MiB), before, 12000 * MiB, 12000 * MiB, 0, 0);
        long target = GlmUnifiedExpertLending.RecoveryTarget(physical, 4000 * MiB, 8000 * MiB);
        Assert.False(UnifiedMemoryExpertCapacity.Resize(cache, experts, Device, target, 2000 * MiB, physical.Limits));
        Assert.Throws<InvalidOperationException>(() => GlmUnifiedExpertLending.TryRefreshSourceViewRecovery(true,
            false, physical, 4000 * MiB, target, 8000 * MiB, 512 * MiB,
            () => GgmlGlmNative.CompleteCheckedMappedResidencyTrim(() => -1),
            () => { samples++; return physical; }, out _, out _, out _));
        Assert.Equal(0, samples); Assert.Equal(0, updates);
        Assert.Equal(before, budget.Snapshot().ToArray());
        Assert.Equal(4000 * MiB, cache.BudgetBytes); Assert.Equal(3500 * MiB, experts.LiveChildCharges.Single().Bytes);
    }

    [Fact]
    public void ImpossibleHostOrDeviceDemandDoesNotEvictAndRepaymentNeverGrowsAShrunkCache()
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var fixedOwners = budget.Reserve(Charges(11900 * MiB, 5200 * MiB));
        using var experts = budget.Reserve(DeviceCharges(6672 * MiB));
        var physical = Observe(budget);
        Assert.Equal(6672 * MiB, GlmUnifiedExpertLending.RepaymentTarget(physical,
            Charges(200 * MiB, 700 * MiB), 6672 * MiB, 4000 * MiB));
        Assert.Equal(6672 * MiB, GlmUnifiedExpertLending.RepaymentTarget(physical,
            Charges(0, 4000 * MiB), 6672 * MiB, 4000 * MiB));
        Assert.Equal(3000 * MiB, GlmUnifiedExpertLending.RepaymentTarget(physical,
            Charges(0, 700 * MiB), 3000 * MiB, 3000 * MiB));
    }
}
