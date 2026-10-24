// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class GlmUnifiedMtpDraftPlanTests
{
    [Theory]
    [InlineData(20, 0, 240)] // Graph-only children cannot stand in for the first snapshot.
    [InlineData(160, 140, 240)] // Keep the forecast's remaining alignment allowance.
    [InlineData(180, 160, 250)] // Actual larger snapshots remain fully owned.
    [InlineData(60, 40, 240)] // Partially allocated snapshot arenas are still children.
    public void SnapshotForecastIsOnlyReducedByTheDedicatedLiveAllocation(long live, long snapshot, long expected)
    {
        var target = GlmUnifiedMtpDraftPlan.Target(new(10, 70), new(5, live), 150, snapshot);
        Assert.Equal(15, target.Host);
        Assert.Equal(expected, target.Device);
    }

    [Theory]
    [InlineData(20, -1)]
    [InlineData(20, 21)]
    public void UnavailableOrUnownedSnapshotEvidenceCannotGrantCredit(long live, long snapshot)
        => Assert.Throws<InvalidOperationException>(() =>
            GlmUnifiedMtpDraftPlan.Target(new(10, 70), new(5, live), 150, snapshot));

    [Fact]
    public void ReservationShrinkPreservesSnapshotAndGraphChildrenAndFirstCaptureCapacity()
    {
        var budget = new MemoryBudget([new("host", 1000), new("device", 1000)]);
        using var envelope = budget.Reserve([new("host", 10), new("device", 400)]);
        using var graph = envelope.TryTake([new("device", 20)])!;
        graph.Commit();
        var first = GlmUnifiedMtpDraftPlan.Target(new(10, 70), new(0, 20), 150, 0);
        Assert.True(envelope.TryResizeTotal([new("host", first.Host), new("device", first.Device)]));
        using var snapshot = envelope.TryTake([new("device", 140)])!;
        snapshot.Commit();
        var captured = GlmUnifiedMtpDraftPlan.Target(new(10, 70), new(0, 160), 150, 140);
        Assert.True(envelope.TryResizeTotal([new("host", captured.Host), new("device", captured.Device)]));
        Assert.Equal(160, envelope.LiveChildCharges.Single(c => c.Pool == "device").Bytes);
        Assert.Equal(240, envelope.OwnedCharges.Single(c => c.Pool == "device").Bytes);
        using var replacement = envelope.TryTake([new("device", 70)])!;
        Assert.NotNull(replacement);
        replacement.Commit();
        Assert.Equal(230, budget.Snapshot().Single(p => p.Pool == "device").Committed);
        snapshot.Dispose();
        var released = GlmUnifiedMtpDraftPlan.Target(new(10, 70), new(0, 90), 150, 0);
        Assert.Equal(310, released.Device); // The next snapshot again needs its complete allowance.
        Assert.True(envelope.TryResizeTotal([new("host", released.Host), new("device", released.Device)]));
        using var nextSnapshot = envelope.TryTake([new("device", 150)])!;
        Assert.NotNull(nextSnapshot);
    }

    [Fact]
    public void LowerReservationDoesNotAuthorizeOptionalGrowthBeyondPhysicalCapacity()
    {
        var budget = new MemoryBudget([new("device", 1000)]);
        using var draft = budget.Reserve([new("device", 400)]);
        using var snapshot = draft.TryTake([new("device", 140)])!;
        snapshot.Commit();
        var target = GlmUnifiedMtpDraftPlan.Target(new(0, 70), new(0, 140), 150, 140);
        Assert.True(draft.TryResizeTotal([new("device", target.Device)], [new("device", 220)]));
        Assert.Throws<MemoryPressureException>(() => budget.TryReserve([new("device", 1)], [new("device", 220)]));
        Assert.Equal(140, draft.LiveChildCharges.Single().Bytes);
    }
}
