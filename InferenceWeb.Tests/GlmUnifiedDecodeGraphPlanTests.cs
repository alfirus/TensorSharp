// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>CPU admission and ownership checks only. These tests do not certify
/// that a native graph fits its workspace forecast or qualify CUDA execution.</summary>
public sealed class GlmUnifiedDecodeGraphPlanTests
{
    private const long MiB = 1L << 20;

    [Theory]
    [InlineData(null, false)]
    [InlineData("", false)]
    [InlineData("0", false)]
    [InlineData("true", false)]
    [InlineData("01", false)]
    [InlineData(" 1", false)]
    [InlineData("1 ", false)]
    [InlineData("1", true)]
    public void OnlyExactOptInEnablesThePlan(string? value, bool expected)
        => Assert.Equal(expected, GlmUnifiedDecodeGraphPlan.Enabled(value));

    [Fact]
    public void LegacyShapesDoNotClaimSingleNativeGraphExecution()
    {
        var shape = new GlmUnifiedGraphShape(256, 1, false);
        Assert.Equal(0, shape.SubmittedTokens);
        Assert.False(shape.SingleNativeGraph);
        Assert.False(Eligible(shape));
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 3)]
    [InlineData(3, 512)]
    public void ActualSubmissionFitsExactlyOneNativeUbatch(int submitted, int ubatch)
        => Assert.True(GlmUnifiedDecodeGraphPlan.SingleAcquire(submitted, ubatch, true, 1,
            layerMajor: false, overrides: false, tracing: false));

    [Theory]
    [InlineData(0, 512, true, 1, false, false, false)]
    [InlineData(-1, 512, true, 1, false, false, false)]
    [InlineData(1, 0, true, 1, false, false, false)]
    [InlineData(3, 2, true, 1, false, false, false)]
    [InlineData(1, 512, false, 1, false, false, false)]
    [InlineData(1, 512, true, 0, false, false, false)]
    [InlineData(1, 512, true, 2, false, false, false)]
    [InlineData(1, 512, true, 1, true, false, false)]
    [InlineData(1, 512, true, 1, false, true, false)]
    [InlineData(1, 512, true, 1, false, false, true)]
    public void SingleAcquireRejectsEveryUnqualifiedPath(int submitted, int ubatch, bool native,
        int tensorParallel, bool layerMajor, bool overrides, bool tracing)
        => Assert.False(GlmUnifiedDecodeGraphPlan.SingleAcquire(submitted, ubatch, native,
            tensorParallel, layerMajor, overrides, tracing));

    [Theory]
    [InlineData(1, 0)]
    [InlineData(1, 1)]
    [InlineData(2, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 1)]
    [InlineData(3, 3)]
    public void EligibleDecodeRetainsActualInputAndOutputWidths(int rows, int outputs)
        => Assert.True(Eligible(Shape(rows, outputs)));

    [Fact]
    public void EveryEligibilityGuardMustHold()
    {
        var shape = Shape();
        foreach (var invalid in new[]
        {
            shape with { ContextTokens = 0 },
            shape with { Rows = 0, SubmittedTokens = 0 },
            shape with { Rows = 4, SubmittedTokens = 4 },
            shape with { Batched = true },
            shape with { OutputRows = -1 },
            shape with { OutputRows = 2 },
            shape with { SubmittedTokens = 0 },
            shape with { SubmittedTokens = 2 },
            shape with { SingleNativeGraph = false }
        }) Assert.False(Eligible(invalid));

        Assert.False(GlmUnifiedDecodeGraphPlan.Eligible(shape, 0, 3, true, false));
        Assert.False(GlmUnifiedDecodeGraphPlan.Eligible(shape, 512, 0, true, false));
        Assert.False(GlmUnifiedDecodeGraphPlan.Eligible(shape, 512, 4, true, false));
        Assert.False(GlmUnifiedDecodeGraphPlan.Eligible(shape, 512, 3, false, false));
        Assert.False(GlmUnifiedDecodeGraphPlan.Eligible(shape, 512, 3, true, true));
        Assert.False(GlmUnifiedDecodeGraphPlan.Eligible(Shape(2, 1), 512, 1, true, false));
        Assert.False(GlmUnifiedDecodeGraphPlan.Eligible(Shape(3, 1), 2, 3, true, false));
    }

    [Fact]
    public void CappedRowsCannotHideASecondAcquireOrSkipRestoringTheBroadPlan()
    {
        var decode = Shape(3, 1);
        var multiple = decode with { SubmittedTokens = 6, SingleNativeGraph = false };
        Assert.Equal(decode.Rows, multiple.Rows);
        Assert.Equal(decode.ContextTokens, multiple.ContextTokens);
        Assert.True(Eligible(decode));
        Assert.False(Eligible(multiple));
        Assert.NotEqual(decode, multiple);
        Assert.True(GlmUnifiedDecodeGraphPlan.MustPrepare(multiple, decode, eligible: false));
        Assert.Equal(new InferenceMemoryBytes(200, 400), GlmUnifiedDecodeGraphPlan.Target(
            new(200, 400), new(10, 50), new(70, 70), eligible: false));
    }

    [Fact]
    public void OutputRowsRemainPartOfAdmissionIdentity()
    {
        var lastOnly = Shape(3, 1);
        var verify = lastOnly with { OutputRows = 3 };
        Assert.NotEqual(lastOnly, verify);
        Assert.True(GlmUnifiedDecodeGraphPlan.MustPrepare(verify, lastOnly, eligible: false));
        Assert.True(Eligible(lastOnly));
        Assert.True(Eligible(verify));
    }

    [Fact]
    public void EveryEligibleCallRechecksChildrenEvenWhenManagedShapeIsUnchanged()
    {
        var shape = Shape(3, 3);
        // Native keys also contain hidden-output and per-row KV shapes, which
        // can change while this managed shape remains identical.
        Assert.True(GlmUnifiedDecodeGraphPlan.MustPrepare(shape, shape, eligible: true));
        Assert.True(GlmUnifiedDecodeGraphPlan.MustPrepare(shape, null, eligible: true));
        Assert.False(GlmUnifiedDecodeGraphPlan.MustPrepare(shape, shape, eligible: false));
        // A new prefill promise changes eligibility without changing tokens or
        // native dispatch metadata. Restore the broad reservation regardless.
        Assert.True(GlmUnifiedDecodeGraphPlan.MustPrepare(shape, shape, eligible: false, previousEligible: true));
        Assert.True(GlmUnifiedDecodeGraphPlan.MustPrepare(shape, null, eligible: false));
        Assert.True(GlmUnifiedDecodeGraphPlan.MustPrepare(shape with { ContextTokens = 512 }, shape, eligible: false));
    }

    [Fact]
    public void DeviceTargetKeepsTheBroadHostPromiseAndOneWholeReplacement()
    {
        var target = GlmUnifiedDecodeGraphPlan.Target(new(200, 400), new(35, 50), new(70, 80), true);
        Assert.Equal(200, target.Host);
        Assert.Equal(130, target.Device);
        // A smaller broad device estimate cannot revoke a larger live owner.
        Assert.Equal(580, GlmUnifiedDecodeGraphPlan.Target(new(200, 400), new(35, 500), new(70, 80), true).Device);
    }

    [Fact]
    public void PendingAndCommittedChildrenRemainOwnedBesideOneNewGraph()
    {
        var budget = new MemoryBudget([new("host", 1000), new("device", 1000)]);
        using var envelope = budget.Reserve([new("host", 200), new("device", 400)]);
        using var retained = envelope.TryTake([new("host", 10), new("device", 20)])!;
        retained.Commit();
        using var pending = envelope.TryTake([new("device", 30)])!;
        var live = Live(envelope);
        Assert.Equal(new InferenceMemoryBytes(10, 50), live);

        var target = GlmUnifiedDecodeGraphPlan.Target(new(200, 400), live, new(70, 70), true);
        Assert.True(envelope.TryResizeTotal([new("host", target.Host), new("device", target.Device)]));
        Assert.Equal(200, envelope.OwnedCharges.Single(c => c.Pool == "host").Bytes);
        Assert.Equal(120, envelope.OwnedCharges.Single(c => c.Pool == "device").Bytes);
        Assert.Equal(live, Live(envelope));
        using var replacement = envelope.TryTake([new("device", 70)])!;
        Assert.NotNull(replacement);
        replacement.Commit();
        Assert.Null(envelope.TryTake([new("device", 1)]));
        Assert.Equal(120, Live(envelope).Device);
        Assert.Equal(90, budget.Snapshot().Single(p => p.Pool == "device").Committed);
    }

    [Fact]
    public void FailedPartialGenerationMustReleaseBeforeItsCreditCanBeRetried()
    {
        var budget = new MemoryBudget([new("host", 1000), new("device", 1000)]);
        using var envelope = budget.Reserve([new("host", 200), new("device", 150)]);
        using var retained = envelope.TryTake([new("device", 80)])!;
        retained.Commit();
        using var partial = envelope.TryTake([new("device", 40)])!;
        partial.Commit();
        using var pending = envelope.TryTake([new("device", 30)])!;
        Assert.Null(envelope.TryTake([new("device", 1)]));
        Assert.Equal(220, GlmUnifiedDecodeGraphPlan.Target(new(200, 400), Live(envelope), new(70, 70), true).Device);

        partial.Dispose();
        pending.Dispose();
        var retry = GlmUnifiedDecodeGraphPlan.Target(new(200, 400), Live(envelope), new(70, 70), true);
        Assert.Equal(150, retry.Device);
        Assert.True(envelope.TryResizeTotal([new("host", retry.Host), new("device", retry.Device)]));
        using var completed = envelope.TryTake([new("device", 70)])!;
        Assert.NotNull(completed);
        completed.Commit();
        retained.Dispose();
        Assert.Equal(70, Live(envelope).Device);
        Assert.Equal(140, GlmUnifiedDecodeGraphPlan.Target(new(200, 400), Live(envelope), new(70, 70), true).Device);
    }

    [Fact]
    public void ReplacementPromiseCannotBypassPhysicalLimits()
    {
        var budget = new MemoryBudget([new("host", 1000), new("device", 1000)]);
        using var envelope = budget.Reserve([new("host", 200), new("device", 90)]);
        using var retained = envelope.TryTake([new("device", 90)])!;
        retained.Commit();
        var target = GlmUnifiedDecodeGraphPlan.Target(new(200, 400), Live(envelope), new(70, 70), true);
        Assert.False(envelope.TryResizeTotal([new("host", target.Host), new("device", target.Device)],
            [new("host", 1000), new("device", 159)]));
        Assert.Equal(90, envelope.OwnedCharges.Single(c => c.Pool == "device").Bytes);
        Assert.True(envelope.TryResizeTotal([new("host", target.Host), new("device", target.Device)],
            [new("host", 1000), new("device", 160)]));
        Assert.Throws<MemoryPressureException>(() => budget.TryReserve([new("device", 1)], [new("device", 160)]));
        Assert.Equal(90, Live(envelope).Device);
    }

    [Fact]
    public void TargetRejectsInconsistentOwnershipAndNegativeWorkspace()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            GlmUnifiedDecodeGraphPlan.Target(new(9, 400), new(10, 50), new(70, 70), true));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            GlmUnifiedDecodeGraphPlan.Target(new(200, 400), new(10, 50), new(-1, 70), true));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            GlmUnifiedDecodeGraphPlan.Target(new(200, 400), new(10, 50), new(70, -1), true));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            GlmUnifiedDecodeGraphPlan.Target(new(200, 400), new(10, -1), new(70, 70), true));
    }

    [Fact]
    public void DeviceAdditionIsCheckedWithoutChangingTheLegacyBranch()
    {
        Assert.Throws<OverflowException>(() =>
            GlmUnifiedDecodeGraphPlan.Target(new(200, 400), new(10, long.MaxValue), new(70, 1), true));
        Assert.Equal(new InferenceMemoryBytes(200, 400),
            GlmUnifiedDecodeGraphPlan.Target(new(200, 400), new(10, long.MaxValue), new(70, 1), false));
    }

    [Theory]
    [InlineData(-1, true, 512)]
    [InlineData(0, true, 512)]
    [InlineData(64 * MiB - 1, true, 512)]
    [InlineData(64 * MiB, true, 64)]
    [InlineData(512 * MiB, true, 64)]
    [InlineData(64 * MiB, false, 512)]
    [InlineData(1024 * MiB, false, 512)]
    public void LowerRecoveryThresholdRequiresTheFirstEligibleBoundaryAndActualRelease(
        long releasedBytes, bool firstEligibleDecode, long expectedMiB)
        => Assert.Equal(expectedMiB * MiB,
            GlmUnifiedDecodeGraphPlan.RecoveryMinimum(releasedBytes, firstEligibleDecode));

    [Fact]
    public void FirstDecodeRecoveryPreservesMarginPendingPromisesAndTheLaterThreshold()
    {
        const string host = UnifiedMemoryModelSession.HostPool, device = UnifiedMemoryModelSession.DevicePool;
        var budget = new MemoryBudget([new(host, 1000 * MiB), new(device, 1000 * MiB)]);
        using var fixedState = budget.Reserve([new(device, 300 * MiB)]);
        fixedState.Commit();
        using var pendingGraphPromise = budget.Reserve([new(host, 200 * MiB), new(device, 100 * MiB)]);
        using var experts = budget.Reserve([new(device, 400 * MiB)]);
        using var residentExperts = experts.TryTake([new(device, 350 * MiB)])!;
        residentExperts.Commit();
        var physical = Observe(budget, 1000 * MiB);
        long firstMinimum = GlmUnifiedDecodeGraphPlan.RecoveryMinimum(200 * MiB, true);
        long laterMinimum = GlmUnifiedDecodeGraphPlan.RecoveryMinimum(200 * MiB, false);
        long target = GlmUnifiedExpertLending.RecoveryTarget(physical, 400 * MiB, 800 * MiB, firstMinimum);

        Assert.Equal(472 * MiB, target); // 200 MiB slack minus the existing 128 MiB margin.
        Assert.Equal(400 * MiB, GlmUnifiedExpertLending.RecoveryTarget(physical, 400 * MiB, 800 * MiB));
        Assert.Equal(400 * MiB, GlmUnifiedExpertLending.RecoveryTarget(physical, 400 * MiB, 800 * MiB, laterMinimum));
        Assert.True(experts.TryResizeTotal([new(device, target)], physical.Limits));
        Assert.Equal(128 * MiB, budget.Snapshot().Single(p => p.Pool == device).Available);
        Assert.Equal(350 * MiB, experts.LiveChildCharges.Single().Bytes);
        Assert.Equal(100 * MiB, pendingGraphPromise.OwnedCharges.Single(c => c.Pool == device).Bytes);
        Assert.Equal(300 * MiB, fixedState.OwnedCharges.Single().Bytes);
    }

    [Fact]
    public void FirstDecodeRecoveryStillUsesFreshPhysicalCapacityAndMiBRounding()
    {
        const string host = UnifiedMemoryModelSession.HostPool, device = UnifiedMemoryModelSession.DevicePool;
        var budget = new MemoryBudget([new(host, 1000 * MiB), new(device, 1000 * MiB)]);
        using var fixedOwners = budget.Reserve([new(device, 400 * MiB)]);
        fixedOwners.Commit();
        using var experts = budget.Reserve([new(device, 400 * MiB)]);
        long minimum = GlmUnifiedDecodeGraphPlan.RecoveryMinimum(64 * MiB, true);
        Assert.Equal(464 * MiB, GlmUnifiedExpertLending.RecoveryTarget(
            Observe(budget, 992 * MiB), 400 * MiB, 800 * MiB, minimum));
        Assert.Equal(400 * MiB, GlmUnifiedExpertLending.RecoveryTarget(
            Observe(budget, 992 * MiB - 1), 400 * MiB, 800 * MiB, minimum));
        Assert.Equal(400 * MiB, GlmUnifiedExpertLending.RecoveryTarget(
            Observe(budget, 1000 * MiB), 400 * MiB, 463 * MiB, minimum));

        long staleTarget = GlmUnifiedExpertLending.RecoveryTarget(
            Observe(budget, 1000 * MiB), 400 * MiB, 800 * MiB, minimum);
        var fresh = Observe(budget, 871 * MiB);
        Assert.Equal(400 * MiB, GlmUnifiedExpertLending.RecoveryTarget(fresh, 400 * MiB, 800 * MiB, minimum));
        Assert.False(experts.TryResizeTotal([new(device, staleTarget)], fresh.Limits));
        Assert.Equal(400 * MiB, experts.OwnedCharges.Single().Bytes);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            GlmUnifiedExpertLending.RecoveryTarget(fresh, 400 * MiB, 800 * MiB, minimumGrowthBytes: 0));
    }

    private static GlmUnifiedGraphShape Shape(int rows = 1, int outputs = 1)
        => new(256, rows, false, outputs, SubmittedTokens: rows, SingleNativeGraph: true);

    private static bool Eligible(GlmUnifiedGraphShape shape)
        => GlmUnifiedDecodeGraphPlan.Eligible(shape, 512, 3, primaryDecode: true, outstandingPrefill: false);

    private static InferenceMemoryBytes Live(BudgetReservation envelope)
        => new(envelope.LiveChildCharges.Single(c => c.Pool == "host").Bytes,
            envelope.LiveChildCharges.Single(c => c.Pool == "device").Bytes);

    private static GlmUnifiedPhysicalObservation Observe(MemoryBudget budget, long deviceLimit)
        => new(default, budget.Snapshot(),
            [new(UnifiedMemoryModelSession.HostPool, 1000 * MiB), new(UnifiedMemoryModelSession.DevicePool, deviceLimit)], 0, 0);
}
