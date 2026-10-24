// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>Profile-1 raw Q4_K row packing. The source remains a pageable,
/// immutable file mapping; no resident-RAM credit or CPU dequantization is implied.</summary>
internal static class GlmUnifiedHostTokenEmbeddingPlan
{
    internal const string EnvironmentVariable = "TS_GLM_HOST_TOKEN_EMBEDDING";
    internal const int Version = 1, Hidden = 4096, Vocabulary = 154880, MaximumRows = 512;
    internal const long RowBytes = 2304, TableBytes = 356843520, PackBytes = RowBytes * MaximumRows;

    internal static bool Resolve(string? value) => value switch
    {
        null or "0" => false,
        "1" => true,
        _ => throw new ArgumentException(EnvironmentVariable + " must be unset, 0 or 1.")
    };

    internal static void RequireSupport(bool enabled, Func<int> version)
    {
        if (enabled && version() != Version)
            throw new NotSupportedException("Host token embedding requires native profile 1.");
    }

    internal static void ValidateMode(bool planned, string? current)
    {
        if (Resolve(current) != planned)
            throw new InvalidOperationException(EnvironmentVariable + " changed after the immutable model policy was selected.");
    }

    internal static bool ProfileOptions(UnifiedMemoryModelOptions options, bool media)
        => !media && !options.EnableGlmFlashMtp && options.MaximumRunningRequests == 1
            && options.RetainedConversationCount == 0 && options.PrefixCheckpointCount == 0
            && options.ResolveGlmPrefillChunkTokens() is >= 1 and <= MaximumRows;

    internal static void ValidateLayout(GgmlTensorType type, ReadOnlySpan<ulong> shape,
        GgufFileRegion source, ReadOnlySpan<ulong> outputShape, GgufFileRegion output)
    {
        if (type != GgmlTensorType.Q4_K || shape.Length != 2 || shape[0] != Hidden || shape[1] != Vocabulary
            || source.ByteLength != TableBytes || source.Offset < 0 || string.IsNullOrWhiteSpace(source.Path)
            || outputShape.Length != 2 || outputShape[0] != Hidden || outputShape[1] != Vocabulary
            || output.Offset < 0 || output.ByteLength <= 0 || string.IsNullOrWhiteSpace(output.Path)
            || source.Offset > long.MaxValue - source.ByteLength || output.Offset > long.MaxValue - output.ByteLength)
            throw new NotSupportedException("Host token embedding requires the exact Q4_K 4096 x 154880 source and a separate output tensor.");
        if (string.Equals(source.Path, output.Path, StringComparison.OrdinalIgnoreCase)
            && source.Offset < output.Offset + output.ByteLength && output.Offset < source.Offset + source.ByteLength)
            throw new NotSupportedException("Host token embedding cannot share or overlap the GPU output source extent.");
    }

    internal static GlmUnifiedMemoryGeometry Apply(GlmUnifiedMemoryGeometry geometry)
    {
        if (!GlmDsaArchitecture.IsGlm5Next(geometry.Architecture) || geometry.Hidden != Hidden
            || geometry.Vocab != Vocabulary || geometry.Mtp != null || geometry.HostTokenEmbedding
            || geometry.DenseDeviceBytes < TableBytes)
            throw new NotSupportedException("Host token embedding requires an unmodified, trunk-only profile-1 geometry.");
        // Keep all alignment/metadata allowances and the separate output table.
        // This certificate discounts exactly one input table, never its mapping.
        return geometry with { HostTokenEmbedding = true, DenseDeviceBytes = checked(geometry.DenseDeviceBytes - TableBytes) };
    }

    internal static void ValidateLoadedInfo(ReadOnlySpan<long> info)
    {
        if (info.Length != 12 || info[0] != Version || info[1] != 1 || info[2] != (long)GgmlTensorType.Q4_K
            || info[3] != Hidden || info[4] != Vocabulary || info[5] != TableBytes || info[6] != RowBytes
            || info[7] != PackBytes || info[8] != PackBytes || info[9] != 1 || info[10] != 1 || info[11] != MaximumRows)
            throw new NotSupportedException("The loaded GLM handle did not certify the exact host token embedding layout, ownership and CUDA input allocation.");
    }
}
