// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Managed forecasts and ownership only; native copy/fence behavior
/// and performance require the separate native staging fixtures.</summary>
public sealed class GlmUnifiedFileStagingPlanTests
{
    private const long MiB = 1L << 20;

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("0")]
    [InlineData("true")]
    [InlineData("01")]
    [InlineData(" 1")]
    [InlineData("1 ")]
    public void OnlyExactOptInChangesTheHostForecast(string? option)
    {
        Assert.False(GlmUnifiedFileStagingPlan.Resolve(option, nativeCapable: false));
        Assert.False(GlmUnifiedFileStagingPlan.ResolvePrefillRing(option, nativeCapable: false));
        // Removing the flag cannot undo a process-static native opt-in. A
        // mismatched later session must not silently lose its host forecast.
        Assert.Throws<NotSupportedException>(() => GlmUnifiedFileStagingPlan.Resolve(option, nativeCapable: true));
        Assert.Throws<NotSupportedException>(() => GlmUnifiedFileStagingPlan.ResolvePrefillRing(option, nativeCapable: true));
    }

    [Fact]
    public void ExactOptInRequiresTheMatchingEnabledNativeCapability()
    {
        Assert.True(GlmUnifiedFileStagingPlan.Resolve("1", nativeCapable: true));
        Assert.Throws<NotSupportedException>(() => GlmUnifiedFileStagingPlan.Resolve("1", nativeCapable: false));
        Assert.True(GlmUnifiedFileStagingPlan.ResolvePrefillRing("1", nativeCapable: true));
        Assert.Throws<NotSupportedException>(() => GlmUnifiedFileStagingPlan.ResolvePrefillRing("1", nativeCapable: false));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void IndependentModesShareOneRetainedExpansionInEveryOwner(bool decode, bool prefill)
    {
        var ordinary = Geometry();
        var configured = ordinary with { ExpandedDecodeStaging = decode, ExpandedPrefillRingStaging = prefill };
        long delta = decode || prefill ? 40 * MiB : 0;
        foreach (int rows in new[] { 1, 3, 512 })
        {
            var before = ordinary.Workspace(2048, rows, 1);
            var after = configured.Workspace(2048, rows, 1);
            Assert.Equal(delta, after.Host-before.Host);
            Assert.Equal(before.Device, after.Device);
            foreach (int generations in new[] { 1, 2, 33 })
            {
                before = ordinary.GraphPeak(2048, rows, false, generations);
                after = configured.GraphPeak(2048, rows, false, generations);
                Assert.Equal(delta, after.Host-before.Host);
                Assert.Equal(before.Device, after.Device);
            }
        }
        Assert.Equal(ordinary.SlotBytes(2048), configured.SlotBytes(2048));
        var mtp = new GlmUnifiedMtpGeometry(0, 0, null!, 2048) { ShapeSpecificPlan = true };
        foreach (var purpose in new[] { GlmUnifiedMtpGraphPurpose.DraftStep, GlmUnifiedMtpGraphPurpose.Snapshot,
            GlmUnifiedMtpGraphPurpose.CatchUp })
        {
            int rows = purpose == GlmUnifiedMtpGraphPurpose.CatchUp ? 512 : 1;
            Assert.Equal(GlmUnifiedMtpDraftPlan.Workspace(ordinary with { Mtp = mtp }, 2048, rows, purpose),
                GlmUnifiedMtpDraftPlan.Workspace(configured with { Mtp = mtp }, 2048, rows, purpose));
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(512)]
    public void RetainedArenaAllowanceSurvivesEveryLaterRequestShape(int rows)
    {
        var ordinary = Geometry();
        Assert.False(ordinary.ExpandedDecodeStaging);
        var expanded = ordinary with { ExpandedDecodeStaging = true };
        var before = ordinary.Workspace(2048, rows, 1);
        var after = expanded.Workspace(2048, rows, 1);
        Assert.Equal(40 * MiB, after.Host - before.Host);
        Assert.Equal(before.Device, after.Device);
        Assert.Equal(ordinary.SlotBytes(2048), expanded.SlotBytes(2048));
        Assert.Equal(ordinary.ExecutionPlacementReserve(2048, rows, 1), expanded.ExecutionPlacementReserve(2048, rows, 1));
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(2, false)]
    [InlineData(33, false)]
    [InlineData(1, true)]
    [InlineData(2, true)]
    [InlineData(33, true)]
    public void JointOwnerCountsOneArenaExpansionAcrossGraphGenerations(int generations, bool batched)
    {
        var ordinary = Geometry();
        var expanded = ordinary with { ExpandedDecodeStaging = true };
        var before = ordinary.GraphPeak(2048, 3, batched, generations);
        var after = expanded.GraphPeak(2048, 3, batched, generations);
        Assert.Equal(40 * MiB, after.Host - before.Host);
        Assert.Equal(before.Device, after.Device);
    }

    [Theory]
    [InlineData((int)GlmUnifiedMtpGraphPurpose.DraftStep, 1)]
    [InlineData((int)GlmUnifiedMtpGraphPurpose.Snapshot, 1)]
    [InlineData((int)GlmUnifiedMtpGraphPurpose.CatchUp, 512)]
    public void DraftOwnerDoesNotDuplicateTheExecutingRequestsHostArena(int purpose, int rows)
    {
        var ordinary = Geometry() with { Mtp = new(0, 0, null!, 2048) { ShapeSpecificPlan = true } };
        var expanded = ordinary with { ExpandedDecodeStaging = true };
        Assert.Equal(GlmUnifiedMtpDraftPlan.Workspace(ordinary, 2048, rows, (GlmUnifiedMtpGraphPurpose)purpose),
            GlmUnifiedMtpDraftPlan.Workspace(expanded, 2048, rows, (GlmUnifiedMtpGraphPurpose)purpose));
        Assert.Equal(GlmUnifiedMtpDraftPlan.SnapshotForecast(ordinary), GlmUnifiedMtpDraftPlan.SnapshotForecast(expanded));
    }

    [Fact]
    public void BaseAndPendingDeltaFitSeventyTwoAndKeepTheirChildFloorThroughRelease()
    {
        var budget = new MemoryBudget([new("host", 72 * MiB)]);
        using var request = budget.Reserve([new("host", 72 * MiB)]);
        using var retained = request.TryTake([new("host", 32 * MiB)])!;
        retained.Commit();
        using var delta = request.TryTake([new("host", GlmUnifiedFileStagingPlan.AdditionalHostBytes)])!;
        Assert.NotNull(delta);
        Assert.Equal(32 * MiB, Assert.Single(budget.Snapshot()).Committed);
        Assert.Equal(40 * MiB, Assert.Single(budget.Snapshot()).Reserved);
        // Even before its physical commit, the delta cannot be refunded by a
        // phase transition shrinking the parent's desired staging forecast.
        Assert.True(request.TryResizeTotal([new("host", 32 * MiB)]));
        Assert.Equal(72 * MiB, Assert.Single(request.OwnedCharges).Bytes);
        delta.Commit();
        request.Dispose();
        Assert.Equal(72 * MiB, Assert.Single(budget.Snapshot()).Committed);
        retained.Dispose();
        Assert.Equal(40 * MiB, Assert.Single(budget.Snapshot()).Committed);
        delta.Dispose();
        Assert.Equal(0, Assert.Single(budget.Snapshot()).Committed);
    }

    [Fact]
    public void PhysicalRefusalLeavesTheOrdinaryArenaAndItsOwnerUntouched()
    {
        var budget = new MemoryBudget([new("host", 72 * MiB)]);
        using var request = budget.Reserve([new("host", 32 * MiB)]);
        using var retained = request.TryTake([new("host", 32 * MiB)])!;
        retained.Commit();
        Assert.False(request.TryResizeTotal([new("host", 72 * MiB)], [new("host", 71 * MiB)]));
        Assert.Equal(32 * MiB, Assert.Single(request.OwnedCharges).Bytes);
        Assert.Null(request.TryTake([new("host", GlmUnifiedFileStagingPlan.AdditionalHostBytes)]));
        Assert.Equal(32 * MiB, Assert.Single(budget.Snapshot()).Committed);
        Assert.Equal(0, Assert.Single(budget.Snapshot()).Reserved);
        Assert.True(request.TryResizeTotal([new("host", 72 * MiB)], [new("host", 72 * MiB)]));
    }

    private static GlmUnifiedMemoryGeometry Geometry() => new()
    {
        Architecture = "glm5next", Layers = 42, RecurrentLayers = 31, DeclaredContextTokens = 2048,
        Hidden = 4096, Heads = 32, Vocab = 154880, KvRow = 512, IndexerDim = 128,
        KdaDim = 128, ConvKernel = 4, HyperConnections = 4, ExpertFf = 2048, UsedExperts = 8
    };
}
