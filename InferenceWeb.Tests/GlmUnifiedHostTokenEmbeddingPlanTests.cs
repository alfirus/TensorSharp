// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory.Planning;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>CPU-only certificate, source-layout and memory-planning checks.
/// Original CUDA GET_ROWS parity and full-model speed require device validation.</summary>
public sealed class GlmUnifiedHostTokenEmbeddingPlanTests
{
    private const long Table = GlmUnifiedHostTokenEmbeddingPlan.TableBytes;
    private const long Pack = GlmUnifiedHostTokenEmbeddingPlan.PackBytes;

    [Theory]
    [InlineData(null, false)]
    [InlineData("0", false)]
    [InlineData("1", true)]
    public void ExactModeContract(string? value, bool expected)
        => Assert.Equal(expected, GlmUnifiedHostTokenEmbeddingPlan.Resolve(value));

    [Theory]
    [InlineData("")]
    [InlineData("true")]
    [InlineData("01")]
    [InlineData(" 1")]
    [InlineData("1 ")]
    [InlineData("-1")]
    [InlineData("2")]
    public void InvalidModeCannotSelectPlacement(string value)
        => Assert.Throws<ArgumentException>(() => GlmUnifiedHostTokenEmbeddingPlan.Resolve(value));

    [Fact]
    public void DisabledCapabilityDoesNotCallNewNativeAbi()
        => GlmUnifiedHostTokenEmbeddingPlan.RequireSupport(false, () => throw new EntryPointNotFoundException());

    [Fact]
    public void RequestedCapabilityMustBeVersionOneAndCannotHideMissingEntryPoint()
    {
        GlmUnifiedHostTokenEmbeddingPlan.RequireSupport(true, () => 1);
        foreach (int version in new[] { 0, 2, -1 })
            Assert.Throws<NotSupportedException>(() => GlmUnifiedHostTokenEmbeddingPlan.RequireSupport(true, () => version));
        Assert.Throws<EntryPointNotFoundException>(() => GlmUnifiedHostTokenEmbeddingPlan.RequireSupport(true,
            () => throw new EntryPointNotFoundException()));
    }

    [Theory]
    [InlineData(false, null)]
    [InlineData(false, "0")]
    [InlineData(true, "1")]
    public void ImmutableModeAcceptsOnlyMatchingLoadSnapshot(bool planned, string? current)
        => GlmUnifiedHostTokenEmbeddingPlan.ValidateMode(planned, current);

    [Theory]
    [InlineData(false, "1")]
    [InlineData(true, null)]
    [InlineData(true, "0")]
    public void EnvironmentDriftCannotSelectAnUnplannedLoader(bool planned, string? current)
        => Assert.Throws<InvalidOperationException>(() => GlmUnifiedHostTokenEmbeddingPlan.ValidateMode(planned, current));

    [Theory]
    [InlineData(1)]
    [InlineData(128)]
    [InlineData(512)]
    public void SinglePrimaryProfileSupportsCapOffAndEveryBoundedChunk(int rows)
    {
        var options = Options() with { PrefillChunkTokens = rows, AdaptiveGlmPrefillChunkTokens = rows,
            LayerMajorPrefillTokens = 0 };
        Assert.True(GlmUnifiedHostTokenEmbeddingPlan.ProfileOptions(options, false));
        Assert.False(GlmUnifiedLayerMajorGraphCapPlan.ProfileOptions(options, false));
        Assert.False(GlmUnifiedHostTokenEmbeddingPlan.ProfileOptions(options, true));
    }

    [Fact]
    public void OtherSlotOwnersDraftAndUnboundedAdaptiveChunksAreRejected()
    {
        var o = Options();
        foreach (var invalid in new[] { o with { MaximumRunningRequests = 2 }, o with { RetainedConversationCount = 1 },
            o with { PrefixCheckpointCount = 1 }, o with { EnableGlmFlashMtp = true },
            o with { EnableAdaptiveGlmPrefill = true, AdaptiveGlmPrefillChunkTokens = 513 } })
            Assert.False(GlmUnifiedHostTokenEmbeddingPlan.ProfileOptions(invalid, false));
    }

    [Fact]
    public void ExactMetadataAllowsSeparateOutputIncludingAdjacentExtentsAndOtherShard()
    {
        Validate();
        Validate(output: new("source.gguf", 64 + Table, Table));
        Validate(output: new("other.gguf", 64, Table));
    }

    [Fact]
    public void QuantizationShapeExtentAndOutputIdentityCannotBeInferredOrDiscounted()
    {
        Assert.Throws<NotSupportedException>(() => Validate(type: GgmlTensorType.F16));
        Assert.Throws<NotSupportedException>(() => Validate(shape: [4096, 154879]));
        Assert.Throws<NotSupportedException>(() => Validate(shape: [4096, 154880, 1]));
        Assert.Throws<NotSupportedException>(() => Validate(source: new("source.gguf", 64, Table - 1)));
        Assert.Throws<NotSupportedException>(() => Validate(source: new("source.gguf", -1, Table)));
        Assert.Throws<NotSupportedException>(() => Validate(source: new("source.gguf", long.MaxValue, Table)));
        Assert.Throws<NotSupportedException>(() => Validate(output: new("SOURCE.GGUF", 64, Table)));
        Assert.Throws<NotSupportedException>(() => Validate(output: new("source.gguf", 64 + Table - 1, Table)));
        Assert.Throws<NotSupportedException>(() => Validate(output: new("other.gguf", long.MaxValue, Table)));
        Assert.Throws<NotSupportedException>(() => Validate(output: new("other.gguf", 0, 0)));
        Assert.Throws<NotSupportedException>(() => Validate(outputShape: [4095, 154880]));
    }

    [Fact]
    public void DiscountIsExactlyOneInputTablePreservingOutputAllowancesSourceAndExperts()
    {
        var original = Geometry();
        var mapped = GlmUnifiedHostTokenEmbeddingPlan.Apply(original);
        Assert.True(mapped.HostTokenEmbedding);
        Assert.False(original.HostTokenEmbedding);
        Assert.Equal(original.DenseDeviceBytes - Table, mapped.DenseDeviceBytes);
        Assert.Equal(Table + (128L << 10), mapped.DenseDeviceBytes); // Output and both allowances remain.
        Assert.Equal(original.SourceBytes, mapped.SourceBytes);
        Assert.Same(original.ExpertLayerBytes, mapped.ExpertLayerBytes);
        Assert.Throws<NotSupportedException>(() => GlmUnifiedHostTokenEmbeddingPlan.Apply(mapped));
        foreach (var invalid in new[] { original with { Architecture = "glm-dsa" }, original with { Hidden = 4095 },
            original with { Vocab = 154879 }, original with { DenseDeviceBytes = Table - 1 },
            original with { Mtp = new GlmUnifiedMtpGeometry(0, 0, null!, 2048) } })
            Assert.Throws<NotSupportedException>(() => GlmUnifiedHostTokenEmbeddingPlan.Apply(invalid));
    }

    [Theory]
    [InlineData(1, 1, false)]
    [InlineData(3, 2, false)]
    [InlineData(512, 33, false)]
    [InlineData(1, 2, true)]
    [InlineData(512, 2, true)]
    public void HostPackIsFixedOnceWhileEveryRetainedGraphPaysItsPackedGpuInput(int rows, int generations, bool expanded)
    {
        var original = Geometry() with { ExpandedDecodeStaging = expanded };
        var mapped = GlmUnifiedHostTokenEmbeddingPlan.Apply(original);
        Assert.Equal(Pack, mapped.Workspace(2048, rows).Host - original.Workspace(2048, rows).Host);
        Assert.Equal(rows * 2304L, mapped.Workspace(2048, rows).Device - original.Workspace(2048, rows).Device);
        Assert.Equal(Pack, mapped.GraphPeak(2048, rows, false, generations).Host
            - original.GraphPeak(2048, rows, false, generations).Host);
        Assert.Equal(generations * rows * 2304L, mapped.GraphPeak(2048, rows, false, generations).Device
            - original.GraphPeak(2048, rows, false, generations).Device);
        Assert.Throws<ArgumentOutOfRangeException>(() => mapped.Workspace(2048, 513));
        var oldStrict = GlmUnifiedLayerMajorGraphCapPlan.Target(original.GraphPeak(2048, rows, false), default, 512L << 20);
        var newStrict = GlmUnifiedLayerMajorGraphCapPlan.Target(mapped.GraphPeak(2048, rows, false), default, 512L << 20);
        Assert.Equal(oldStrict.Device, newStrict.Device); // Native cap includes its new input payload.
        Assert.Equal(Pack, newStrict.Host - oldStrict.Host);
    }

    [Fact]
    public void LoadedCertificateMustAcknowledgeEveryActualFieldExactly()
    {
        long[] actual = [1, 1, 12, 4096, 154880, Table, 2304, Pack, Pack, 1, 1, 512];
        GlmUnifiedHostTokenEmbeddingPlan.ValidateLoadedInfo(actual);
        for (int field = 0; field < actual.Length; ++field)
        {
            var invalid = (long[])actual.Clone();
            invalid[field] = field is 0 or 1 or 9 or 10 ? 0 : invalid[field] + 1;
            Assert.Throws<NotSupportedException>(() => GlmUnifiedHostTokenEmbeddingPlan.ValidateLoadedInfo(invalid));
        }
        Assert.Throws<NotSupportedException>(() => GlmUnifiedHostTokenEmbeddingPlan.ValidateLoadedInfo(actual[..11]));
        Assert.Throws<NotSupportedException>(() => GlmUnifiedHostTokenEmbeddingPlan.ValidateLoadedInfo([.. actual, 0]));
        Assert.Throws<NotSupportedException>(() => GlmUnifiedHostTokenEmbeddingPlan.ValidateLoadedInfo([]));
    }

    private static UnifiedMemoryModelOptions Options() => new()
        { MaximumRunningRequests = 1, RetainedConversationCount = 0, PrefixCheckpointCount = 0 };

    private static void Validate(GgmlTensorType type = GgmlTensorType.Q4_K, ulong[]? shape = null,
        GgufFileRegion? source = null, ulong[]? outputShape = null, GgufFileRegion? output = null)
        => GlmUnifiedHostTokenEmbeddingPlan.ValidateLayout(type, shape ?? [4096, 154880],
            source ?? new("source.gguf", 64, Table), outputShape ?? [4096, 154880],
            output ?? new("source.gguf", 64 + Table, Table));

    private static GlmUnifiedMemoryGeometry Geometry() => new()
    {
        Architecture = "glm5next", Layers = 45, RecurrentLayers = 34, DeclaredContextTokens = 32768,
        Hidden = 4096, Heads = 64, Vocab = 154880, KvRow = 512, IndexerDim = 128,
        KdaDim = 128, ConvKernel = 4, HyperConnections = 4, ExpertFf = 2048, UsedExperts = 8,
        DenseDeviceBytes = 2 * Table + (128L << 10), SourceBytes = 90L << 30, ExpertLayerBytes = [123, 456]
    };
}
