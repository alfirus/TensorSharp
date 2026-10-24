// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Pure shape and ownership forecasts; no native allocation, CUDA
/// execution, numerical parity or throughput qualification is implied.</summary>
public sealed class GlmUnifiedMtpShapePlanTests
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
    public void OnlyExactOptInEnablesShapePlanning(string? option)
    {
        Assert.False(GlmUnifiedMtpDraftPlan.ResolveShapePlan(option, nativeCapable: false));
        Assert.False(GlmUnifiedMtpDraftPlan.ResolveShapePlan(option, nativeCapable: true));
    }

    [Fact]
    public void ExactOptInRequiresItsNativeCapability()
    {
        Assert.True(GlmUnifiedMtpDraftPlan.ResolveShapePlan("1", nativeCapable: true));
        Assert.Throws<NotSupportedException>(() =>
            GlmUnifiedMtpDraftPlan.ResolveShapePlan("1", nativeCapable: false));
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(3, 1)]
    [InlineData(3, 8)]
    [InlineData(3, 9)]
    [InlineData(3, 512)]
    public void SharedDraftKeyRetainsTheCompleteTrunkForecast(int decodeRows, int prefillRows)
    {
        var trunk = Geometry();
        var shared = WithDraft(trunk, trunk.ExpertLayerLayouts[0]!, shapeSpecific: false);
        long expected = Forecast(trunk, decodeRows, prefillRows);
        Assert.Equal(expected, Forecast(shared, decodeRows, prefillRows));
        Assert.Equal(expected, Forecast(shared with
        {
            Mtp = shared.Mtp! with { ShapeSpecificPlan = true }
        }, decodeRows, prefillRows));
    }

    [Theory]
    [InlineData(3, 1)]
    [InlineData(1, 8)]
    [InlineData(3, 9)]
    [InlineData(3, 512)]
    public void DistinctDraftAddsOnlyItsRowOneFloorAndIgnoresMemberOrder(int decodeRows, int prefillRows)
    {
        var trunk = Geometry();
        var draft = DraftLayout();
        var legacy = WithDraft(trunk, draft, shapeSpecific: false);
        var shaped = WithDraft(trunk, draft, shapeSpecific: true);
        var draftOnly = trunk with { Layers = 1, ExpertLayerLayouts = [draft], Mtp = null };
        long expected = checked(Forecast(trunk, decodeRows, prefillRows) + Forecast(draftOnly, 1, 1));

        Assert.Equal(expected, Forecast(shaped, decodeRows, prefillRows));
        Assert.True(expected < Forecast(legacy, decodeRows, prefillRows));
        Assert.Equal(expected, Forecast(shaped with
        {
            ExpertLayerLayouts = trunk.ExpertLayerLayouts.Reverse().ToArray()
        }, decodeRows, prefillRows));

        // Without the capability, the draft keeps the same complete shape as
        // another trunk member, including every retained compact variant.
        var legacyAsTrunk = trunk with
        {
            Layers = trunk.Layers + 1,
            ExpertLayerLayouts = [draft, .. trunk.ExpertLayerLayouts]
        };
        Assert.Equal(Forecast(legacyAsTrunk, decodeRows, prefillRows),
            Forecast(legacy, decodeRows, prefillRows));
    }

    [Fact]
    public void SharedGroupCombinesLargestPayloadWithTheTrunkWidth()
    {
        var trunk = Geometry();
        var small = trunk.ExpertLayerLayouts[0]!;
        var large = small with
        {
            Gate = small.Gate with
            {
                BytesPerExpert = small.Gate.BytesPerExpert + 4096,
                TensorTailBytes = small.Gate.TensorTailBytes + 128
            }
        };
        var draftHasLargest = WithDraft(trunk, large, shapeSpecific: true);
        var trunkHasLargest = WithDraft(trunk with
        {
            ExpertLayerLayouts = [large, .. trunk.ExpertLayerLayouts.Skip(1)]
        }, small, shapeSpecific: true);

        Assert.Equal(Forecast(trunkHasLargest, 3, 512), Forecast(draftHasLargest, 3, 512));
        Assert.True(Forecast(draftHasLargest, 3, 512) > Forecast(trunk, 3, 512));
    }

    [Fact]
    public void WideCatchUpRemovesExactlyTheAbsentFfnAndLogitsForecast()
    {
        var legacy = WithDraft(Geometry(), DraftLayout(), shapeSpecific: false);
        var shaped = legacy with { Mtp = legacy.Mtp! with { ShapeSpecificPlan = true } };
        var before = GlmUnifiedMtpDraftPlan.Workspace(legacy, 2048, 512, GlmUnifiedMtpGraphPurpose.CatchUp);
        var after = GlmUnifiedMtpDraftPlan.Workspace(shaped, 2048, 512, GlmUnifiedMtpGraphPurpose.CatchUp);
        long absent = 256 * MiB + 4L * shaped.Vocab;

        Assert.Equal(absent, before.Host - after.Host);
        Assert.Equal(absent, before.Device - after.Device);
        Assert.True(after.Host > 0);
        Assert.True(after.Device > 0);
        var shorterContext = GlmUnifiedMtpDraftPlan.Workspace(shaped, 1024, 512, GlmUnifiedMtpGraphPurpose.CatchUp);
        Assert.Equal(4L * 1024 * 512, after.Host - shorterContext.Host);
        Assert.Equal(4L * 1024 * 512, after.Device - shorterContext.Device);
    }

    [Theory]
    [InlineData((int)GlmUnifiedMtpGraphPurpose.DraftStep)]
    [InlineData((int)GlmUnifiedMtpGraphPurpose.Snapshot)]
    public void ScalarDraftAndSnapshotKeepTheLegacyWorkspace(int purposeValue)
    {
        var purpose = (GlmUnifiedMtpGraphPurpose)purposeValue;
        var trunk = Geometry();
        var legacy = WithDraft(trunk, DraftLayout(), shapeSpecific: false);
        var shaped = legacy with { Mtp = legacy.Mtp! with { ShapeSpecificPlan = true } };
        var oldDraft = trunk with { Layers = 1, RecurrentLayers = 0, HyperConnections = 1, Mtp = null };
        var expected = oldDraft.Workspace(2048, 1, 1);

        Assert.Equal(expected, GlmUnifiedMtpDraftPlan.Workspace(legacy, 2048, 1, purpose));
        Assert.Equal(expected, GlmUnifiedMtpDraftPlan.Workspace(shaped, 2048, 1, purpose));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedMtpDraftPlan.Workspace(shaped, 2048, 2, purpose));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedMtpDraftPlan.Workspace(legacy, 2048, 512, purpose));
    }

    [Fact]
    public void AbsentDraftCapabilityAndInvalidPurposeCannotSelectReducedWorkspace()
    {
        var trunk = Geometry();
        var oldDraft = trunk with { Layers = 1, RecurrentLayers = 0, HyperConnections = 1 };
        Assert.Equal(oldDraft.Workspace(2048, 512, 1),
            GlmUnifiedMtpDraftPlan.Workspace(trunk, 2048, 512, GlmUnifiedMtpGraphPurpose.CatchUp));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedMtpDraftPlan.Workspace(trunk, 2048, 1,
            (GlmUnifiedMtpGraphPurpose)int.MaxValue));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(140)]
    [InlineData(160)]
    public void ReducedCatchUpTargetPreservesGraphChildrenAndMissingSnapshot(long snapshotMiB)
    {
        var legacy = WithDraft(Geometry(), DraftLayout(), shapeSpecific: false);
        var shaped = legacy with { Mtp = legacy.Mtp! with { ShapeSpecificPlan = true } };
        var before = GlmUnifiedMtpDraftPlan.Workspace(legacy, 2048, 512, GlmUnifiedMtpGraphPurpose.CatchUp);
        var after = GlmUnifiedMtpDraftPlan.Workspace(shaped, 2048, 512, GlmUnifiedMtpGraphPurpose.CatchUp);
        long graphBytes = 20 * MiB, snapshotBytes = snapshotMiB * MiB, snapshotForecast = 150 * MiB;
        var target = GlmUnifiedMtpDraftPlan.Target(after, new(5 * MiB, graphBytes + snapshotBytes),
            snapshotForecast, snapshotBytes);
        var legacyTarget = GlmUnifiedMtpDraftPlan.Target(before, new(5 * MiB, graphBytes + snapshotBytes),
            snapshotForecast, snapshotBytes);

        Assert.Equal(after.Host + 5 * MiB, target.Host);
        Assert.Equal(after.Device + graphBytes + Math.Max(snapshotForecast, snapshotBytes), target.Device);
        Assert.Equal(before.Host - after.Host, legacyTarget.Host - target.Host);
        Assert.Equal(before.Device - after.Device, legacyTarget.Device - target.Device);
    }

    private static long Forecast(GlmUnifiedMemoryGeometry geometry, int decodeRows, int prefillRows)
        => GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, decodeRows, prefillRows);

    private static GlmUnifiedMemoryGeometry WithDraft(GlmUnifiedMemoryGeometry geometry,
        GlmUnifiedExpertLayout layout, bool shapeSpecific) => geometry with
        {
            Mtp = new(64 * MiB, 288 * layout.BytesPerExpert, layout, 2048)
            {
                ShapeSpecificPlan = shapeSpecific
            }
        };

    private static GlmUnifiedExpertLayout Layout(GgmlTensorType gate, GgmlTensorType up,
        GgmlTensorType down, long bytes) => new(new(gate, bytes / 3, 0, true),
            new(up, bytes / 3, 0, true), new(down, bytes - 2 * (bytes / 3), 0, true));

    private static GlmUnifiedExpertLayout DraftLayout()
        => Layout(GgmlTensorType.Q8_0, GgmlTensorType.Q8_0, GgmlTensorType.Q8_0, 9_109_504);

    private static GlmUnifiedMemoryGeometry Geometry() => new()
    {
        Layers = 3, DeclaredContextTokens = 32768, RecurrentLayers = 1,
        Hidden = 4096, Heads = 64, Vocab = 154880, KvRow = 512, IndexerDim = 128,
        KdaDim = 128, ConvKernel = 4, HyperConnections = 4, ExpertFf = 2048,
        UsedExperts = 8, ExpertCount = 288, DenseDeviceBytes = 1024 * MiB,
        ExpertLayerBytes = [30L << 30, 30L << 30, 30L << 30],
        ExpertLayerLayouts =
        [
            Layout(GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ3_XXS, 7_536_640),
            Layout(GgmlTensorType.IQ1_S, GgmlTensorType.IQ1_S, GgmlTensorType.IQ3_XXS, 6_488_064),
            Layout(GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ4_XS, 8_781_824)
        ]
    };
}
