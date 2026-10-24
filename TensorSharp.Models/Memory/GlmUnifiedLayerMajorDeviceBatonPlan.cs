// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Globalization;
using TensorSharp.Memory.Planning;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>One independently charged full-call F32 residual bank. This is an
/// additional Device promise, never spare graph credit or a Host refund.</summary>
internal static class GlmUnifiedLayerMajorDeviceBatonPlan
{
    internal const string EnvironmentVariable = "TS_GLM_LAYER_MAJOR_DEVICE_BATON";
    internal static bool Resolve(string? value) => value switch {
        null or "0" => false, "1" => true,
        _ => throw new ArgumentException(EnvironmentVariable + " must be unset, 0 or 1.")
    };
    // Only the opt-in caller uses this before its early profile guard. Preserve
    // ordinary explicit-option/MAX_CONTEXT precedence and snapshot it once.
    internal static UnifiedMemoryModelOptions ResolveContextOptions(UnifiedMemoryModelOptions options, string? configured)
    {
        ArgumentNullException.ThrowIfNull(options);
        int context = options.ContextTokens;
        if (context == 0 && (!int.TryParse(configured, NumberStyles.None,
                CultureInfo.InvariantCulture, out context) || context <= 0))
            throw new NotSupportedException("Device baton requires ContextTokens=2048 or MAX_CONTEXT=2048.");
        if (context != 2048)
            throw new NotSupportedException("Device baton requires effective context exactly 2048.");
        return options.ContextTokens == context ? options : options with { ContextTokens = context };
    }
    internal static void ValidateMode(bool planned, string? value)
    {
        if (Resolve(value) != planned) throw new InvalidOperationException("Layer-major device baton mode changed after load.");
    }
    internal static bool ProfileOptions(UnifiedMemoryModelOptions o, bool media)
        => !media && o.ContextTokens == 2048 && o.MaximumRunningRequests == 1
            && o.RetainedConversationCount == 0 && o.PrefixCheckpointCount == 0
            && o.EnableGlmSelectedExperts && o.EnableAdaptiveGlmPrefill && o.EnableExpertCacheCapacityGrowth
            && !o.EnableGlmFlashMtp && !o.EnableAsyncExpertPrefill && o.MaximumHostExpertCacheBytes == 0
            && o.ResolveGlmPrefillChunkTokens() == 512 && o.LayerMajorPrefillTokens > 512;
    internal static long Bytes(GlmUnifiedMemoryGeometry geometry, int submittedTokens, int rows, bool layerMajor)
    {
        if (!geometry.LayerMajorDeviceBaton || !layerMajor || submittedTokens <= rows) return 0;
        if (rows != 512 || submittedTokens > 2048 || submittedTokens < 1
            || geometry.Hidden != 4096 || geometry.HyperConnections != 4)
            throw new NotSupportedException("Device baton requires actual context2048/chunk512 layer-major geometry.");
        return checked(4L * geometry.Hidden * geometry.HyperConnections * submittedTokens);
    }
    internal static InferenceMemoryBytes AddOnce(InferenceMemoryBytes graph, long bankBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bankBytes);
        return new(graph.Host, checked(graph.Device + bankBytes));
    }
    // This certificate is emitted only after the native bank's physical free
    // and checked ticket acknowledgment, before logits/head publication.
    internal static long CompletedRelease(ReadOnlySpan<long> info, ref long observedCall)
    {
        if (info.Length != 12 || info[0] != 1 || info[1] != 1 || info[2] != 0
            || info[3] != 0 || info[4] != 0 || info[5] < observedCall || info[11] != 1)
            throw new InvalidOperationException("Layer-major bank owner did not finish checked retirement.");
        if (info[5] == observedCall) return 0;
        long tokens = info[6] / (4L * 4096 * 4);
        if (tokens <= 512 || tokens > 2048 || info[6] != tokens * (4L * 4096 * 4)
            || info[7] != checked(44L * info[6]) || info[8] != info[7]
            || info[9] != 44L * ((tokens + 511) / 512) || info[10] != info[9])
            throw new InvalidOperationException("Layer-major bank completion has inconsistent transfer proof.");
        observedCall = info[5]; return info[6];
    }
    internal static bool Complete(int result, bool strictGraph, bool bankMode, Func<bool> legacyFallback)
        => GlmUnifiedLayerMajorGraphCapPlan.Complete(result,strictGraph || bankMode,legacyFallback);

    internal static long RecoveryMinimum(long actualReleasedCredit, long checkedFreedBankBytes)
        => GlmUnifiedDecodeGraphPlan.RecoveryMinimum(Math.Min(actualReleasedCredit, checkedFreedBankBytes), true);
}
