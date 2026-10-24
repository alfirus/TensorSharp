// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Globalization;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Real ledger and injected hardware observations; no CUDA/model load.</summary>
public sealed class GlmUnifiedAdmissionDiagnosticsTests
{
    private const string Host = UnifiedMemoryModelSession.HostPool;
    private const string Device = UnifiedMemoryModelSession.DevicePool;

    [Fact]
    public void CommitConstrainedHostRefusalReportsTheFailingPoolAndDoesNotAddReservedCreditBack()
    {
        var budget = new MemoryBudget([new(Host, 1000), new(Device, 1000)]);
        using var root = budget.Reserve([new(Host, 300), new(Device, 100)]);
        using var payload = root.TryTake([new(Host, 200), new(Device, 80)])!;
        payload.Commit();
        var before = budget.Snapshot();
        var hardware = new InferenceHardwareMemory(2000, 1000, 1600, 800)
        { HostCommitLimit = 1500, HostCommitAvailable = 180 };
        var sample = GlmUnifiedPhysicalObservation.Create(hardware, before, 1000, 1000, 50, 100);
        Assert.Equal(330, sample.Limits.Single(c => c.Pool == Host).Bytes);
        Assert.Equal(780, sample.Limits.Single(c => c.Pool == Device).Bytes);
        MemoryCharge[] growth = [new(Host, 31), new(Device, 0)];
        Assert.False(sample.Fits(growth));
        string text = sample.Describe(growth);
        Assert.Contains($"{Host}: reserved=100, committed=200, capacity=1000, physical-limit=330, growth=31, combined-post=331, shortfall=1", text);
        Assert.Contains($"{Device}: reserved=20, committed=80, capacity=1000, physical-limit=780, growth=0, combined-post=100, shortfall=0", text);
        Assert.Contains("host-available=1000, host-commit-limit=1500, host-commit-available=180, host-allocation-available=180", text);
        Assert.Equal(before.ToArray(), budget.Snapshot().ToArray());
    }

    [Fact]
    public void ExistingJointOwnershipIsReplacedOnceAndCommittedChildrenAreNeverCreditedAsFreed()
    {
        var budget = new MemoryBudget([new(Host, 2000), new(Device, 2000)]);
        using var joint = budget.Reserve([new(Host, 600), new(Device, 900)]);
        using var liveGraph = joint.TryTake([new(Host, 200), new(Device, 500)])!;
        liveGraph.Commit();
        var before = budget.Snapshot();
        var sample = GlmUnifiedPhysicalObservation.Create(new(3000, 500, 3000, 500), before, 2000, 2000, 0, 0);
        // New joint target (400/700) plus request (100/200), less the
        // already-owned 600/900. Live graph ownership is part of that root.
        MemoryCharge[] growth = [new(Host, 400 - 600 + 100), new(Device, 700 - 900 + 200)];
        Assert.True(sample.Fits(growth));
        string text = sample.Describe(growth);
        Assert.Contains("growth=-100, combined-post=500, shortfall=0", text);
        Assert.Contains("growth=0, combined-post=900, shortfall=0", text);
        Assert.Equal(500, joint.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(before.ToArray(), budget.Snapshot().ToArray());
    }

    [Fact]
    public void DeviceDeficitAndFreshHardwareChangeRemainDistinctFromHostDeficit()
    {
        var budget = new MemoryBudget([new(Host, 1000), new(Device, 1000)]);
        using var live = budget.Reserve([new(Host, 100), new(Device, 500)]);
        live.Commit();
        var ledger = budget.Snapshot();
        var initial = GlmUnifiedPhysicalObservation.Create(new(2000, 1000, 1600, 200), ledger, 1000, 1000, 50, 100);
        MemoryCharge[] growth = [new(Host, 20), new(Device, 101)];
        Assert.False(initial.Fits(growth));
        Assert.Contains("physical-limit=600, growth=101, combined-post=601, shortfall=1", initial.Describe(growth));
        var fresh = GlmUnifiedPhysicalObservation.Create(new(2000, 1000, 1600, 201), ledger, 1000, 1000, 50, 100);
        Assert.True(fresh.Fits(growth));
        // Retaining an observation for diagnostics must not resample or
        // rewrite the original refusal after the hardware allowance changes.
        Assert.Contains("device-available=200", initial.Describe(growth));
        Assert.Contains("shortfall=1", initial.Describe(growth));
        Assert.Equal(ledger.ToArray(), budget.Snapshot().ToArray());
    }

    [Fact]
    public void HeadroomExhaustionNeverSubtractsExistingCommitmentOrGrantsReservedCredit()
    {
        var budget = new MemoryBudget([new(Host, 1000), new(Device, 1000)]);
        using var owner = budget.Reserve([new(Host, 200), new(Device, 300)]);
        using var live = owner.TryTake([new(Host, 80), new(Device, 90)])!;
        live.Commit();
        var sample = GlmUnifiedPhysicalObservation.Create(new(2000, 10, 1600, 20), budget.Snapshot(), 1000, 1000, 50, 100);
        Assert.Equal(80, sample.Limits.Single(c => c.Pool == Host).Bytes);
        Assert.Equal(90, sample.Limits.Single(c => c.Pool == Device).Bytes);
        Assert.False(sample.Fits([new(Host, 0), new(Device, 0)]));
        Assert.Contains("growth=0, combined-post=200, shortfall=120", sample.Describe([new(Host, 0)]));
    }

    [Fact]
    public void UnknownCommitIsExplicitAndConfiguredLedgerCeilingStillLimitsAdmission()
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 50, 30, 0), new(Device, 70, 0, 0) };
        var sample = GlmUnifiedPhysicalObservation.Create(new(2000, 1000, 1600, 800), ledger, 1000, 1000, 0, 0);
        Assert.False(sample.Fits([new(Host, 21)]));
        Assert.Contains("host-commit-limit=unknown, host-commit-available=unknown", sample.Describe([new(Host, 21)]));
        Assert.Contains("physical-limit=1000, growth=21, combined-post=51, shortfall=1", sample.Describe([new(Host, 21)]));
    }

    [Fact]
    public void DiagnosticArithmeticDoesNotOverflowOrDependOnProcessCulture()
    {
        var original = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("fr-FR");
            var ledger = new MemoryPoolSnapshot[] { new(Host, long.MaxValue, long.MaxValue, 0), new(Device, 1, 0, 0) };
            var sample = GlmUnifiedPhysicalObservation.Create(new(long.MaxValue, long.MaxValue, 1, 1), ledger,
                long.MaxValue, 1, 0, 0);
            Assert.False(sample.Fits([new(Host, long.MaxValue)]));
            string text = sample.Describe([new(Host, long.MaxValue)]);
            Assert.Contains("combined-post=18446744073709551614", text);
            Assert.Contains("shortfall=9223372036854775807", text);
        }
        finally { CultureInfo.CurrentCulture = original; }
    }

    [Fact]
    public void SuccessfulMappedHintWithoutFreshAllowanceStillRefusesAndDoesNotRefundLiveChildren()
    {
        var budget = new MemoryBudget([new(Host, 1000), new(Device, 1000)]);
        using var root = budget.Reserve([new(Host, 200), new(Device, 200)]);
        using var state = root.TryTake([new(Host, 100), new(Device, 100)])!;
        state.Commit();
        var before = budget.Snapshot();
        var sample = GlmUnifiedPhysicalObservation.Create(new(2000, 50, 1600, 800), before, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(Host, 25), new(Device, 25)];
        int hints = 0, samples = 0;
        Assert.True(sample.TryRefreshMappedHostPressure(growth, () => { hints++; return true; },
            () => { samples++; return sample; }, out var refreshed, out var hinted));
        Assert.True(hinted);
        Assert.Equal(1, hints); Assert.Equal(1, samples);
        Assert.False(refreshed.Fits(growth));
        Assert.Equal(75, refreshed.Shortfall(Host, growth));
        Assert.Equal(100, root.LiveChildCharges.Single(c => c.Pool == Host).Bytes);
        Assert.Equal(before.ToArray(), budget.Snapshot().ToArray());
    }

    [Fact]
    public void MappedHostRetryUsesOneFreshObservationAndExactLimitWithoutAnyBudgetIncrease()
    {
        var budget = new MemoryBudget([new(Host, 1000), new(Device, 1000)]);
        using var root = budget.Reserve([new(Host, 200), new(Device, 200)]);
        using var state = root.TryTake([new(Host, 100), new(Device, 100)])!;
        state.Commit();
        var before = budget.Snapshot();
        var sample = GlmUnifiedPhysicalObservation.Create(new(2000, 50, 1600, 800), before, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(Host, 25), new(Device, 25)];
        int hints = 0, samples = 0;
        Assert.True(sample.TryRefreshMappedHostPressure(growth, () => { hints++; return true; },
            () => { samples++; return GlmUnifiedPhysicalObservation.Create(new(2000, 125, 1600, 800), budget.Snapshot(), 1000, 1000, 0, 0); },
            out var refreshed, out var hinted));
        Assert.True(hinted); Assert.Equal(1, hints); Assert.Equal(1, samples);
        Assert.True(refreshed.Fits(growth)); // committed 100 + available 125 == owned 200 + requested 25.
        Assert.Equal(225, refreshed.Limits.Single(c => c.Pool == Host).Bytes);
        Assert.False(sample.Fits(growth));
        Assert.Equal(before.ToArray(), budget.Snapshot().ToArray());
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(1, 1)]
    [InlineData(0, 0)]
    public void DeviceOrCombinedDeficitAndAlreadyFittingRequestNeverHintsMappings(int hostDeficit, int deviceDeficit)
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 1000, 100, 100), new(Device, 1000, 100, 100) };
        var sample = GlmUnifiedPhysicalObservation.Create(new(2000, 100 - hostDeficit, 1600, 100 - deviceDeficit),
            ledger, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(Host, 0), new(Device, 0)];
        int hints = 0, samples = 0;
        Assert.False(sample.TryRefreshMappedHostPressure(growth, () => { hints++; return true; },
            () => { samples++; return sample; }, out var refreshed, out var hinted));
        Assert.Null(hinted); Assert.Equal(0, hints); Assert.Equal(0, samples);
        Assert.Equal(sample, refreshed);
    }

    [Fact]
    public void FalseMappedHintCannotAuthorizeGrowthButAnIndependentlyFreshHardwareAllowanceCan()
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 1000, 100, 100), new(Device, 1000, 100, 100) };
        var sample = GlmUnifiedPhysicalObservation.Create(new(2000, 50, 1600, 800), ledger, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(Host, 25), new(Device, 25)];
        Assert.True(sample.TryRefreshMappedHostPressure(growth, () => false,
            () => GlmUnifiedPhysicalObservation.Create(new(2000, 125, 1600, 800), ledger, 1000, 1000, 0, 0),
            out var refreshed, out var hinted));
        Assert.False(hinted);
        Assert.True(refreshed.Fits(growth));
        Assert.Equal(ledger, refreshed.Ledger);
    }

    [Fact]
    public void FreeingMappedPhysicalRamCannotConcealAnUnchangedHostCommitShortfall()
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 1000, 100, 100), new(Device, 1000, 100, 100) };
        var hardware = new InferenceHardwareMemory(2000, 50, 1600, 800)
        { HostCommitLimit = 3000, HostCommitAvailable = 50 };
        var sample = GlmUnifiedPhysicalObservation.Create(hardware, ledger, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(Host, 25), new(Device, 25)];
        Assert.True(sample.TryRefreshMappedHostPressure(growth, () => true,
            () => GlmUnifiedPhysicalObservation.Create(hardware with { HostAvailable = 1000 }, ledger, 1000, 1000, 0, 0),
            out var refreshed, out _));
        Assert.False(refreshed.Fits(growth));
        Assert.Equal(75, refreshed.Shortfall(Host, growth));
        Assert.Contains("host-available=1000, host-commit-limit=3000, host-commit-available=50, host-allocation-available=50",
            refreshed.Describe(growth));
    }

    [Theory]
    [InlineData(true, 900, 512)]
    [InlineData(false, 900, 512)]
    [InlineData(true, 400, 12)]
    [InlineData(false, 400, 12)]
    public void NarrowSourceViewPrefillReselectsFromFreshPhysicsWithoutChangingOwnedCredit(
        bool hintResult, long freshHostAvailable, int expectedRows)
    {
        var budget = new MemoryBudget([new(Host, 1000), new(Device, 1000)]);
        using var joint = budget.Reserve([new(Host, 150), new(Device, 300)]);
        using var request = budget.Reserve([new(Host, 50), new(Device, 100)]);
        using var experts = budget.Reserve([new(Device, 100)]);
        using var jointChildren = joint.TryTake([new(Host, 75), new(Device, 200)])!;
        using var requestChildren = request.TryTake([new(Host, 25), new(Device, 50)])!;
        jointChildren.Commit(); requestChildren.Commit(); experts.Commit();
        var before = budget.Snapshot().ToArray();
        var jointOwned = new InferenceMemoryBytes(150, 300);
        var requestOwned = new InferenceMemoryBytes(50, 100);
        (InferenceMemoryBytes Joint, InferenceMemoryBytes Request) Estimate(int rows)
            => (new(438 + rows, 600), new(50, 100));
        var initial = GlmUnifiedPhysicalObservation.Create(new(2000, 400, 1600, 650), before, 1000, 1000, 0, 0);
        Assert.True(GlmUnifiedPrefillAdmission.TrySelect(initial.Ledger, initial.Limits,
            jointOwned, requestOwned, 100, 100, 512, Estimate, out var narrow));
        Assert.Equal(12, narrow.Rows); // A fitting width bypassed the old all-width refusal hint.
        var widest = Estimate(512);
        MemoryCharge[] growth = [new(Host, widest.Joint.Host - jointOwned.Host + widest.Request.Host - requestOwned.Host),
            new(Device, widest.Joint.Device - jointOwned.Device + widest.Request.Device - requestOwned.Device)];
        Assert.Equal(800, growth[0].Bytes); Assert.Equal(300, growth[1].Bytes);
        bool attempted = false;
        int hints = 0, samples = 0;
        Assert.True(initial.TryRefreshNarrowSourceViewPrefill(true, narrow.Rows, 512, 0, growth, ref attempted,
            () => { hints++; return hintResult; },
            () => { samples++; return GlmUnifiedPhysicalObservation.Create(new(2000, freshHostAvailable, 1600, 650),
                budget.Snapshot(), 1000, 1000, 0, 0); }, out var fresh));
        Assert.True(GlmUnifiedPrefillAdmission.TrySelect(fresh.Ledger, fresh.Limits,
            jointOwned, requestOwned, 100, 100, 512, Estimate, out var selected));
        Assert.Equal(expectedRows, selected.Rows);
        Assert.True(attempted); Assert.Equal(1, hints); Assert.Equal(1, samples);
        Assert.False(fresh.TryRefreshNarrowSourceViewPrefill(true, 12, 512, 0, growth, ref attempted,
            () => { hints++; return true; }, () => { samples++; return fresh; }, out _));
        Assert.Equal(1, hints); Assert.Equal(1, samples);
        Assert.Equal(before, budget.Snapshot().ToArray());
        Assert.Equal(200, joint.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(50, request.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(100, experts.OwnedCharges.Single(c => c.Pool == Device).Bytes);
    }

    [Theory]
    [InlineData(false, 12, 1, 0, 0, false)] // Mode OFF.
    [InlineData(true, 512, 1, 0, 0, false)] // No accepted downsizing.
    [InlineData(true, 12, 0, 1, 0, false)] // Device-only deficit.
    [InlineData(true, 12, 1, 2, 1, false)] // Beyond admitted expert credit.
    [InlineData(true, 12, 1, 1, 0, false)] // Existing expert floor cannot be spent.
    [InlineData(true, 12, 0, 0, 0, false)] // No physical Host deficit.
    [InlineData(true, 12, 1, 2, 2, true)] // Selector may reclaim this exact Device delta.
    public void NarrowSourceViewHintHonorsModeWidthAndActualDeviceReclaimBound(bool sourceView,
        int selectedRows, int hostDeficit, int deviceDeficit, long reclaimable, bool expected)
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 1000, 100, 100), new(Device, 1000, 100, 100) };
        var physical = GlmUnifiedPhysicalObservation.Create(new(2000, 125 - hostDeficit, 1600, 125 - deviceDeficit),
            ledger, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(Host, 25), new(Device, 25)];
        bool attempted = false;
        int hints = 0, samples = 0;
        Assert.Equal(expected, physical.TryRefreshNarrowSourceViewPrefill(sourceView, selectedRows, 512,
            reclaimable, growth, ref attempted, () => { hints++; return true; },
            () => { samples++; return physical; }, out var fresh));
        Assert.Equal(expected, attempted);
        Assert.Equal(expected ? 1 : 0, hints); Assert.Equal(hints, samples);
        Assert.Equal(physical, fresh); // No synthetic Device or Host allowance.
        Assert.Equal(deviceDeficit, fresh.Shortfall(Device, growth));
    }

    [Fact]
    public void NarrowSourceViewHintUsesBothOwnedRootsAndReplacementFloorWithoutRefundingChildren()
    {
        var budget = new MemoryBudget([new(Host, 10000), new(Device, 10000)]);
        using var joint = budget.Reserve([new(Host, 1000), new(Device, 2000)]);
        using var request = budget.Reserve([new(Host, 200), new(Device, 300)]);
        using var children = joint.TryTake([new(Host, 700), new(Device, 1500)])!;
        using var requestChildren = request.TryTake([new(Host, 100), new(Device, 100)])!;
        children.Commit(); requestChildren.Commit();
        var before = budget.Snapshot().ToArray();
        var replacement = GlmUnifiedPrefillAdmission.PreserveReplacement(new(400, 700), new(700, 1500), 2);
        Assert.Equal(new InferenceMemoryBytes(900, 1850), replacement);
        MemoryCharge[] growth = [new(Host, replacement.Host - 1000 + 600 - 200),
            new(Device, replacement.Device - 2000 + 500 - 300)];
        Assert.Equal(300, growth[0].Bytes); Assert.Equal(50, growth[1].Bytes);
        var physical = GlmUnifiedPhysicalObservation.Create(new(20000, 600, 20000, 10000), before, 10000, 10000, 0, 0);
        Assert.Equal(100, physical.Shortfall(Host, growth));
        bool attempted = false;
        Assert.True(physical.TryRefreshNarrowSourceViewPrefill(true, 12, 512, 0, growth, ref attempted, () => true,
            () => GlmUnifiedPhysicalObservation.Create(new(20000, 700, 20000, 10000), budget.Snapshot(),
                10000, 10000, 0, 0), out var fresh));
        Assert.True(fresh.Fits(growth)); // Exact fresh allowance, no owned root counted twice.
        Assert.Equal(before, budget.Snapshot().ToArray());
        Assert.Equal(1500, joint.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(100, request.LiveChildCharges.Single(c => c.Pool == Host).Bytes);
    }

    [Fact]
    public void NarrowSourceViewHintCannotRepairConfiguredHostCeilingOrResampleAfterFatal()
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 200, 100, 100), new(Device, 1000, 100, 100) };
        var physical = GlmUnifiedPhysicalObservation.Create(new(2000, 1000, 1600, 800), ledger, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(Host, 25), new(Device, 25)];
        bool attempted = false;
        int hints = 0, samples = 0;
        Assert.False(physical.TryRefreshNarrowSourceViewPrefill(true, 12, 512, 0, growth, ref attempted,
            () => { hints++; return true; }, () => { samples++; return physical; }, out _));
        Assert.False(attempted); Assert.Equal(0, hints); Assert.Equal(0, samples);
        ledger = [new(Host, 1000, 100, 100), new(Device, 1000, 100, 100)];
        physical = GlmUnifiedPhysicalObservation.Create(new(2000, 50, 1600, 800), ledger, 1000, 1000, 0, 0);
        Assert.Throws<InvalidOperationException>(() => physical.TryRefreshNarrowSourceViewPrefill(true, 12, 512, 0,
            growth, ref attempted, () => { hints++; throw new InvalidOperationException("checked fatal"); },
            () => { samples++; return physical; }, out _));
        Assert.True(attempted); Assert.Equal(1, hints); Assert.Equal(0, samples);
        Assert.Equal(ledger, physical.Ledger);
    }

    [Theory]
    [InlineData(-1, 100)]
    [InlineData(101, 100)]
    [InlineData(0, 0)]
    public void InvalidDeviceObservationCannotConcealARefusal(long available, long total)
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 100, 0, 0), new(Device, 100, 0, 0) };
        Assert.Throws<ArgumentException>(() => GlmUnifiedPhysicalObservation.Create(new(200, 100, total, available),
            ledger, 100, 100, 0, 0));
    }
}
