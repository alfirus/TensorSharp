// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

// Allocation forecasts only; native intermediate-state equality and failed
// transfer poisoning require the separately gated CUDA model fixtures.
public sealed class GlmUnifiedMtpPrefixPlanTests
{
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("0")]
    [InlineData("true")]
    [InlineData("01")]
    [InlineData(" 1")]
    [InlineData("1 ")]
    public void NoImplicitPrefixCapabilityOptIn(string? option)
    {
        Assert.False(GlmUnifiedMtpDraftPlan.ResolvePrefixCommit(option, false));
        Assert.False(GlmUnifiedMtpDraftPlan.ResolvePrefixCommit(option, true));
    }

    [Fact]
    public void ExplicitPrefixOptInRefusesAnOldNativeLibrary()
    {
        Assert.Throws<NotSupportedException>(() => GlmUnifiedMtpDraftPlan.ResolvePrefixCommit("1", false));
        Assert.True(GlmUnifiedMtpDraftPlan.ResolvePrefixCommit("1", true));
    }

    [Fact]
    public void PersistentMirrorAndOneLayerScratchBelongToDifferentOwners()
    {
        var baseline = Geometry(false);
        var prefix = Geometry(true);
        long originalSnapshot = GlmUnifiedMtpDraftPlan.SnapshotForecast(baseline);
        Assert.True(originalSnapshot > 0);
        Assert.Equal(2 * originalSnapshot, GlmUnifiedMtpDraftPlan.SnapshotForecast(prefix));
        var oldGraph = baseline.Workspace(256, 2, 2);
        var newGraph = prefix.Workspace(256, 2, 2);
        Assert.Equal(oldGraph.Host, newGraph.Host);
        Assert.Equal(4L * prefix.Heads * prefix.KdaDim * prefix.KdaDim, newGraph.Device - oldGraph.Device);
        Assert.True(newGraph.Device - oldGraph.Device < originalSnapshot);
        Assert.Equal(GlmUnifiedMtpDraftPlan.Workspace(baseline, 256, 1, GlmUnifiedMtpGraphPurpose.Snapshot),
            GlmUnifiedMtpDraftPlan.Workspace(prefix, 256, 1, GlmUnifiedMtpGraphPurpose.Snapshot));
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 1)]
    [InlineData(3, 1)]
    [InlineData(3, 3)]
    [InlineData(512, 1)]
    public void OtherTargetShapesKeepTheirExistingWorkspace(int rows, int outputs)
        => Assert.Equal(Geometry(false).Workspace(256, rows, outputs), Geometry(true).Workspace(256, rows, outputs));

    [Fact]
    public void PartialSnapshotAllocationCannotFundTheMissingMirror()
    {
        long snapshot = GlmUnifiedMtpDraftPlan.SnapshotForecast(Geometry(false));
        var budget = new MemoryBudget([new("device", 4 * snapshot)]);
        using var owner = budget.Reserve([new("device", 3 * snapshot)]);
        using var firstArena = owner.TryTake([new("device", snapshot)])!;
        firstArena.Commit();
        var target = GlmUnifiedMtpDraftPlan.Target(new(0, snapshot / 2), new(0, snapshot), 2 * snapshot, snapshot);
        Assert.True(owner.TryResizeTotal([new("device", target.Device)]));
        using var prefixArena = owner.TryTake([new("device", snapshot)])!;
        Assert.NotNull(prefixArena);
        // A pending arena already owns its bytes; it cannot be refunded merely
        // because the native snapshot is not yet published as a valid window.
        Assert.True(owner.TryResizeTotal([new("device", snapshot)]));
        Assert.Equal(2 * snapshot, owner.OwnedCharges.Single().Bytes); // Total targets clamp to live children.
        Assert.Equal(2 * snapshot, owner.LiveChildCharges.Single().Bytes);
        Assert.True(owner.TryResizeTotal([new("device", target.Device)]));
        prefixArena.Commit();
        var retained = GlmUnifiedMtpDraftPlan.Target(new(0, snapshot / 2), new(0, 2 * snapshot), 2 * snapshot, 2 * snapshot);
        Assert.Equal(target, retained);
        using var graph = owner.TryTake([new("device", snapshot / 2)])!;
        Assert.NotNull(graph);
        graph.Commit();
        Assert.Equal(target.Device, budget.Snapshot().Single().Committed);
    }

    private static GlmUnifiedMemoryGeometry Geometry(bool enabled)
    {
        var layout = new GlmUnifiedExpertLayout(new(GgmlTensorType.Q8_0, 1024, 0, true),
            new(GgmlTensorType.Q8_0, 1024, 0, true), new(GgmlTensorType.Q8_0, 1024, 0, true));
        return new()
        {
            Layers = 4, RecurrentLayers = 3, DeclaredContextTokens = 2048,
            Heads = 8, KdaDim = 128, ConvKernel = 4, Hidden = 1024,
            HyperConnections = 4, Vocab = 1024, ExpertFf = 256, UsedExperts = 2,
            Mtp = new(1024, 1024, layout, 2048) { PrefixStateCommit = enabled }
        };
    }
}
