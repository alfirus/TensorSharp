// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Pure metadata/policy tests. No native allocator, CUDA initialization,
/// model execution or performance qualification is implied.</summary>
public sealed class GlmSelectedExpertMemoryPlanTests
{
    private const long GiB = 1L << 30;
    private static GlmUnifiedExpertLayout Layout(GgmlTensorType gate, GgmlTensorType up,
        GgmlTensorType down, long bytes) => new(new(gate, bytes / 3, 0, true),
            new(up, bytes / 3, 0, true), new(down, bytes - 2 * (bytes / 3), 0, true));

    private static GlmUnifiedMemoryGeometry Geometry() => new()
    {
        Layers = 3, DeclaredContextTokens = 32768, RecurrentLayers = 1,
        Hidden = 4096, Heads = 64, Vocab = 154880, KvRow = 512, IndexerDim = 128,
        KdaDim = 128, ConvKernel = 4, HyperConnections = 4, ExpertFf = 2048,
        UsedExperts = 8, ExpertCount = 288, DenseDeviceBytes = GiB,
        ExpertLayerBytes = [30 * GiB, 30 * GiB, 30 * GiB],
        ExpertLayerLayouts = [Layout(GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ3_XXS, 7_536_640),
            Layout(GgmlTensorType.IQ1_S, GgmlTensorType.IQ1_S, GgmlTensorType.IQ3_XXS, 6_488_064),
            Layout(GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ4_XS, 8_781_824)]
    };

    private static UnifiedMemoryModelOptions Options(int prefill = 128, int requests = 3, long maximum = long.MaxValue)
        => new() { EnableGlmSelectedExperts = true, PrefillChunkTokens = prefill,
            MaximumRunningRequests = requests, MaximumExpertCacheBytes = maximum };

    [Theory]
    [InlineData(8, 0.595553)]
    [InlineData(16, 3.155874)]
    [InlineData(32, 6.306023)]
    [InlineData(64, 7.110228)]
    [InlineData(128, 7.148325)]
    public void CompleteUnionAndAllVariantForecastExceedsRecordedNativeMinimum(int rows, double nativeGiB)
    {
        var geometry = Geometry();
        long forecast = GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, rows);
        // Rounded native measurements are comparison evidence, not exact
        // allocator values and not a substitute for the post-load check.
        Assert.True(forecast >= nativeGiB * GiB);
        long union = Math.Min(288, 8L * (rows > 8 ? rows : 3));
        Assert.True(forecast > 3 * union * 8_781_824);
        Assert.True(forecast < 3 * union * (4096L * 2048 * 3 * 4)); // no F32 weight copy
    }

    [Fact]
    public void ResourceGrowthAndShrinkSelectLargestFeasiblePrefillWithoutMutatingCaller()
    {
        var geometry = Geometry();
        var options = Options();
        var original = options with { };
        var small = GlmSelectedExpertMemoryPlan.Select(geometry, options, 32768, 4 * GiB, GiB / 2);
        var large = GlmSelectedExpertMemoryPlan.Select(geometry, options, 32768, 11 * GiB, GiB / 2);
        var shrunk = GlmSelectedExpertMemoryPlan.Select(geometry, options, 32768, 4 * GiB, GiB / 2);
        Assert.True(small.PrefillRows > 8);
        Assert.True(small.PrefillRows < large.PrefillRows);
        Assert.Equal(128, large.PrefillRows);
        Assert.Equal(small, shrunk);
        Assert.Equal(original, options);
        Assert.Equal(128, options.PrefillChunkTokens);
        Assert.Equal(3, small.CpuExpertLayers);
        Assert.True(small.CacheMinimumForecastBytes <= small.CacheCapacityBytes);
        long nextReserve = geometry.ExecutionPlacementReserve(32768, small.PrefillRows + 1, 3);
        long nextCapacity = GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(4 * GiB,
            geometry.DenseDeviceBytes, nextReserve, GiB / 2, options.MaximumExpertCacheBytes);
        Assert.True(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, small.PrefillRows + 1) > nextCapacity);
        Assert.Equal(small.CacheCapacityBytes + geometry.ExecutionPlacementReserve(32768, small.PrefillRows, 3),
            small.ExecutionReserveBytes);
        Assert.True(small.ExecutionReserveBytes + geometry.DenseDeviceBytes + GiB / 2 <= 4 * GiB);
    }

    [Fact]
    public void OperatorAndCallerCapsArePreservedWhileTwoGiBIsNoLongerAnImplicitCap()
    {
        var geometry = Geometry();
        var automatic = GlmSelectedExpertMemoryPlan.Select(geometry, Options(), 32768, 10 * GiB, GiB / 2);
        var limited = GlmSelectedExpertMemoryPlan.Select(geometry, Options(maximum: 2 * GiB), 32768, 10 * GiB, GiB / 2);
        Assert.True(automatic.CacheCapacityBytes > 2 * GiB);
        Assert.Equal(2 * GiB, limited.CacheCapacityBytes);
        Assert.InRange(limited.PrefillRows, 9, 10);
        var capped = GlmSelectedExpertMemoryPlan.Select(geometry, Options(prefill: 7), 32768, 10 * GiB, 0);
        Assert.Equal(7, capped.PrefillRows);
        Assert.Equal(128, GlmSelectedExpertMemoryPlan.Select(geometry, Options(prefill: 512), 32768, 10 * GiB, 0).PrefillRows);
        Assert.Equal(4, GlmSelectedExpertMemoryPlan.Select(geometry, Options(), 4, 10 * GiB, 0).PrefillRows);
    }

    [Fact]
    public void NarrowVariantsProtectMaximumDecodeUnionAndFailClosedBelowItsFloor()
    {
        var geometry = Geometry();
        long narrow = GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 1);
        Assert.Equal(narrow, GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 8));
        Assert.True(narrow > GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 1, 8));
        Assert.True(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 9) > narrow);
        Assert.Equal(8, GlmSelectedExpertMemoryPlan.Select(geometry, Options(maximum: narrow), 32768, 10 * GiB, 0).PrefillRows);
        Assert.Throws<MemoryPressureException>(() => GlmSelectedExpertMemoryPlan.Select(geometry,
            Options(maximum: narrow - 1), 32768, 10 * GiB, 0));
        Assert.Throws<MemoryPressureException>(() => GlmSelectedExpertMemoryPlan.Select(geometry,
            Options(maximum: 0), 32768, 10 * GiB, 0));
        Assert.Throws<MemoryPressureException>(() => GlmSelectedExpertMemoryPlan.Select(geometry,
            Options(), 32768, geometry.DenseDeviceBytes, GiB));
    }

    [Fact]
    public void EqualGroupQuotasUseMaximumPayloadAndIgnoreDuplicateTypeTriples()
    {
        var geometry = Geometry();
        var duplicate = geometry with { Layers = 4, ExpertLayerLayouts = [.. geometry.ExpertLayerLayouts, geometry.ExpertLayerLayouts[0]] };
        Assert.Equal(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 16),
            GlmSelectedExpertMemoryPlan.ForecastMinimum(duplicate, 3, 16));
        var uniformLargest = geometry with { ExpertLayerLayouts = geometry.ExpertLayerLayouts.Select(l =>
            Layout(l.Gate.Type, l.Up.Type, l.Down.Type, 8_781_824)).ToArray() };
        Assert.Equal(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 16),
            GlmSelectedExpertMemoryPlan.ForecastMinimum(uniformLargest, 3, 16));
        var withTails = geometry with { ExpertLayerLayouts = geometry.ExpertLayerLayouts.Select(l =>
            l with { Gate = l.Gate with { TensorTailBytes = 512 } }).ToArray() };
        Assert.Equal(3 * 512, GlmSelectedExpertMemoryPlan.ForecastMinimum(withTails, 3, 16)
            - GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 16));
    }

    [Fact]
    public void DisabledAndFullyResidentPlansPreserveOriginalPrefillAndUseNoExpertEnvelope()
    {
        var geometry = Geometry();
        var disabled = GlmSelectedExpertMemoryPlan.Select(geometry, Options() with { EnableGlmSelectedExperts = false }, 32768, 10 * GiB, 0);
        Assert.Equal(128, disabled.PrefillRows);
        Assert.Equal(0, disabled.CacheCapacityBytes);
        Assert.Equal(0, disabled.CacheMinimumForecastBytes);
        var resident = GlmSelectedExpertMemoryPlan.Select(geometry with { ExpertLayerLayouts = [] }, Options(), 32768, 100 * GiB, 0);
        Assert.Equal(128, resident.PrefillRows);
        Assert.Equal(0, resident.CpuExpertLayers);
        Assert.Equal(0, resident.CacheCapacityBytes);
        Assert.False(new UnifiedMemoryModelOptions().EnableGlmSelectedExperts);
    }

    [Fact]
    public void CheckedSixtyFourBitMathRejectsOverflowAndUnsupportedOrIncompleteLayouts()
    {
        var geometry = Geometry();
        var overflow = geometry with { ExpertLayerLayouts = [Layout(GgmlTensorType.Q8_0, GgmlTensorType.Q8_0, GgmlTensorType.Q8_0, long.MaxValue), null, null] };
        Assert.Throws<OverflowException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(overflow, 3, 128));
        Assert.Throws<OverflowException>(() => GlmSelectedExpertMemoryPlan.Select(geometry with { DenseDeviceBytes = long.MaxValue }, Options(), 32768, long.MaxValue, 0));
        var unsupported = geometry with { ExpertLayerLayouts = [geometry.ExpertLayerLayouts[0] with
            { Gate = geometry.ExpertLayerLayouts[0].Gate with { SupportsSelectedTier = false } }, null, null] };
        Assert.Throws<NotSupportedException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(unsupported, 3, 8));
        Assert.Throws<NotSupportedException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry with { ExpertLayerLayouts = [] }, 3, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 4, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 129));
    }
}
