// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class GlmUnifiedHostExpertCapacityTests
{
    private const long MiB = 1L << 20;
    private const string Host = UnifiedMemoryModelSession.HostPool, Device = UnifiedMemoryModelSession.DevicePool;
    private static MemoryCharge[] Charges(long host, long device = 0) => [new(Host, host), new(Device, device)];
    private static GlmUnifiedPhysicalObservation Observe(MemoryBudget budget, long limit = 12000 * MiB)
        => new(default, budget.Snapshot(), Charges(limit, 12000 * MiB), 0, 0);
    private sealed class Native : IHostMoeRamCacheNative
    {
        internal Action<long>? OnConfigure;
        internal long Capacity;
        public void Configure(long bytes) { OnConfigure?.Invoke(bytes); Capacity = bytes; }
        public GgmlHostMoeRamCacheUsage ReadUsage() => new(Capacity, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    [Fact]
    public void HostLoanRepaysOnlyAfterPhysicalReleaseThenRecoversAfterRequestCompletion()
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var fixedOwner = budget.Reserve(Charges(2000 * MiB, 4000 * MiB)); fixedOwner.Commit();
        using var envelope = budget.Reserve([new(Host, 0)]);
        var native = new Native();
        using var cache = new GgmlHostMoeRamCacheScope(native);
        var physical = Observe(budget);
        long target = GlmUnifiedHostExpertCapacity.RecoveryTarget(physical, 0, 10000 * MiB);
        Assert.Equal(9744 * MiB, target);
        Assert.True(GlmUnifiedHostExpertCapacity.Resize(cache, envelope, target, physical.Limits));
        using var payload = envelope.TryTake([new(Host, 9000 * MiB)])!; payload.Commit();
        var needed = Charges(3000 * MiB, 1000 * MiB);
        physical = Observe(budget);
        target = GlmUnifiedHostExpertCapacity.RepaymentTarget(physical, needed, cache.CapacityBytes);
        Assert.Equal(6744 * MiB, target);
        native.OnConfigure = bytes =>
        {
            Assert.Equal(9744 * MiB, Assert.Single(envelope.OwnedCharges).Bytes);
            Assert.Equal(9000 * MiB, Assert.Single(envelope.LiveChildCharges).Bytes);
            payload.Dispose(); // Native release must precede the credit shrink.
        };
        Assert.True(GlmUnifiedHostExpertCapacity.Resize(cache, envelope, target, physical.Limits));
        native.OnConfigure = null;
        Assert.True(Observe(budget).Fits(needed));
        using var request = budget.Reserve(needed); request.Commit();
        Assert.Equal(target, GlmUnifiedHostExpertCapacity.RecoveryTarget(Observe(budget), target, 10000 * MiB));
        request.Dispose();
        target = GlmUnifiedHostExpertCapacity.RecoveryTarget(Observe(budget), target, 10000 * MiB);
        Assert.Equal(9744 * MiB, target);
        Assert.True(GlmUnifiedHostExpertCapacity.Resize(cache, envelope, target, Observe(budget).Limits));
        Assert.Equal(4000 * MiB, fixedOwner.OwnedCharges.Single(c => c.Pool == Device).Bytes);
    }

    [Fact]
    public void RecoveryHonorsPhysicalLimitsAndOperatorCapWithoutAllocatingPayload()
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var pending = budget.Reserve(Charges(2000 * MiB));
        Assert.Equal(3744 * MiB, GlmUnifiedHostExpertCapacity.RecoveryTarget(Observe(budget, 6000 * MiB), 0, 10000 * MiB));
        Assert.Equal(1024 * MiB, GlmUnifiedHostExpertCapacity.RecoveryTarget(Observe(budget), 0, 1024 * MiB));
        using var cache = new GgmlHostMoeRamCacheScope(new Native());
        using var envelope = budget.Reserve([new(Host, 0)]);
        Assert.True(GlmUnifiedHostExpertCapacity.Resize(cache, envelope, 1024 * MiB, Observe(budget).Limits));
        Assert.Equal(0, Assert.Single(envelope.LiveChildCharges).Bytes);
        Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == Host).Committed);
    }

    [Fact]
    public void FailedGrowthRestoresReservationAndFailedNativeDisposalCanRetry()
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var envelope = budget.Reserve([new(Host, 0)]);
        var native = new Native();
        using var cache = new GgmlHostMoeRamCacheScope(native);
        native.OnConfigure = _ => throw new InvalidOperationException("injected failure");
        Assert.Throws<InvalidOperationException>(() => GlmUnifiedHostExpertCapacity.Resize(cache, envelope, 1024 * MiB, Observe(budget).Limits));
        Assert.Equal(0, Assert.Single(envelope.OwnedCharges).Bytes);
        Assert.Equal(0, cache.CapacityBytes);
        Assert.Throws<InvalidOperationException>(cache.Dispose);
        Assert.Throws<InvalidOperationException>(() => new GgmlHostMoeRamCacheScope(new Native()));
        native.OnConfigure = null;
        Assert.True(GlmUnifiedHostExpertCapacity.Resize(cache, envelope, 1024 * MiB, Observe(budget).Limits));
        cache.Dispose();
        Assert.Equal(0, native.Capacity);
        Assert.Throws<ObjectDisposedException>(() => cache.Configure(1));
    }

    [Fact]
    public void ExclusiveScopeCannotClearAnotherCacheAndFailedConstructionReleasesItsClaim()
    {
        var failed = new Native { OnConfigure = _ => throw new InvalidOperationException("constructor failure") };
        Assert.Throws<InvalidOperationException>(() => new GgmlHostMoeRamCacheScope(failed));
        var native = new Native();
        using var first = new GgmlHostMoeRamCacheScope(native);
        first.Configure(1024);
        bool touched = false;
        Assert.Throws<InvalidOperationException>(() => new GgmlHostMoeRamCacheScope(
            new Native { OnConfigure = _ => touched = true }));
        Assert.False(touched);
        Assert.Equal(1024, native.Capacity);
        first.Dispose();
        using var next = new GgmlHostMoeRamCacheScope(new Native());
        Assert.Equal(0, next.CapacityBytes);
    }

    [Fact]
    public void MappedShrinkKeepsHostEnvelopeUntilChargedPinsRetire()
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var envelope = budget.Reserve([new(Host, 0)]);
        var native = new MappedRamCacheNative();
        using var cache = new GgmlHostMoeRamCacheScope(native,
            MappedRamCacheNative.Reader(MappedRamCacheNative.Environment()));
        Assert.True(GlmUnifiedHostExpertCapacity.Resize(cache, envelope, 128 * MiB, Observe(budget).Limits));
        using var payload = envelope.TryTake([new(Host, 64 * MiB)])!; payload.Commit();
        native.Usage = default(GgmlHostMoeMappedRamCacheUsage) with
        { OwnedChargeBytes = 64 * MiB, LockedPageBytes = 60 * MiB, MetadataChargeBytes = 4 * MiB, ActiveLeases = 1 };
        Assert.Throws<InvalidOperationException>(() => GlmUnifiedHostExpertCapacity.Resize(cache, envelope, 32 * MiB, Observe(budget).Limits));
        Assert.Equal(128 * MiB, envelope.OwnedCharges.Single().Bytes);
        Assert.Equal(64 * MiB, envelope.LiveChildCharges.Single().Bytes);
        Assert.Equal(128 * MiB, cache.CapacityBytes);
        payload.Dispose(); native.Usage = default;
        Assert.True(GlmUnifiedHostExpertCapacity.Resize(cache, envelope, 32 * MiB, Observe(budget).Limits));
        Assert.Equal(32 * MiB, envelope.OwnedCharges.Single().Bytes);
    }

    [Fact]
    public void FailedMappedGrowthRetainsReservationUntilQuarantineCleanupCanRetry()
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var envelope = budget.Reserve([new(Host, 0)]);
        var native = new MappedRamCacheNative();
        using var cache = new GgmlHostMoeRamCacheScope(native,
            MappedRamCacheNative.Reader(MappedRamCacheNative.Environment()));
        BudgetReservation? payload = null;
        native.OnConfigure = bytes =>
        {
            if (bytes != 0)
            {
                payload = envelope.TryTake([new(Host, 64 * MiB)])!; payload.Commit();
                native.Usage = default(GgmlHostMoeMappedRamCacheUsage) with
                { OwnedChargeBytes = 64 * MiB, LockedPageBytes = 64 * MiB, QuarantineChargeBytes = 64 * MiB };
                throw new InvalidOperationException("injected mapped admission failure");
            }
            throw new InvalidOperationException("injected quarantine cleanup failure");
        };
        try
        {
            var error = Assert.Throws<InvalidOperationException>(() => GlmUnifiedHostExpertCapacity.Resize(
                cache, envelope, 128 * MiB, Observe(budget).Limits));
            Assert.IsType<AggregateException>(error.InnerException);
            Assert.Equal(128 * MiB, envelope.OwnedCharges.Single().Bytes);
            Assert.Equal(64 * MiB, envelope.LiveChildCharges.Single().Bytes);
            Assert.Equal(0, cache.CapacityBytes);
            Assert.Throws<InvalidOperationException>(cache.Dispose);
            Assert.Throws<InvalidOperationException>(() => new GgmlHostMoeRamCacheScope(new MappedRamCacheNative(),
                MappedRamCacheNative.Reader(MappedRamCacheNative.Environment())));
        }
        finally
        {
            native.OnConfigure = null;
            payload?.Dispose(); native.Usage = default;
        }
        // Cache.CapacityBytes is still zero, but its failed attempt retained
        // the envelope. Equal-target resize must retry native cleanup first.
        Assert.True(GlmUnifiedHostExpertCapacity.Resize(cache, envelope, 0, Observe(budget).Limits));
        Assert.Equal(0, envelope.OwnedCharges.Single().Bytes);
        Assert.Equal(0, native.Capacity);
    }

    [Fact]
    public void MappedGrowthRollbackRestoresNativeLazyBudgetBeforeHostCredit()
    {
        var budget = new MemoryBudget(Charges(12000 * MiB, 12000 * MiB));
        using var envelope = budget.Reserve([new(Host, 0)]);
        var native = new MappedRamCacheNative();
        using var cache = new GgmlHostMoeRamCacheScope(native,
            MappedRamCacheNative.Reader(MappedRamCacheNative.Environment()));
        native.OnConfigure = bytes =>
        {
            Assert.Equal(128 * MiB, envelope.OwnedCharges.Single().Bytes);
            if (bytes != 0)
                native.Usage = default(GgmlHostMoeMappedRamCacheUsage) with { PendingAdmissions = 1 };
            else native.Usage = default;
        };
        Assert.Throws<InvalidOperationException>(() => GlmUnifiedHostExpertCapacity.Resize(
            cache, envelope, 128 * MiB, Observe(budget).Limits));
        native.OnConfigure = null;
        Assert.Equal(new long[] { 0, 128 * MiB, 0 }, native.Configurations);
        Assert.Equal(0, native.Capacity);
        Assert.Equal(0, cache.CapacityBytes);
        Assert.Equal(0, envelope.OwnedCharges.Single().Bytes);
    }
}
