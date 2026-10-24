// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Globalization;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>One durable Host slab for causal cache metadata. This promise does
/// not describe resident model bytes, allocator headers or additional VRAM.</summary>
internal static class GlmUnifiedCausalRouteNeighborPlan
{
    internal const string EnvironmentVariable = "TS_HOST_MOE_CAUSAL_ROUTE_NEIGHBOR";
    internal static bool Resolve(string? value) => value switch {
        null or "0" => false, "1" => true,
        _ => throw new ArgumentException(EnvironmentVariable + " must be unset, 0 or 1.")
    };
    internal static void ValidateMode(bool planned, string? value)
    {
        if (Resolve(value) != planned)
            throw new InvalidOperationException("Causal route-neighbor mode changed after load.");
    }
    internal static UnifiedMemoryModelOptions ResolveContextOptions(UnifiedMemoryModelOptions options, string? configured)
    {
        ArgumentNullException.ThrowIfNull(options);
        int context = options.ContextTokens;
        if (context == 0 && (!int.TryParse(configured, NumberStyles.None,
                CultureInfo.InvariantCulture, out context) || context <= 0))
            throw new NotSupportedException("Causal route-neighbor requires ContextTokens=2048 or MAX_CONTEXT=2048.");
        if (context != 2048)
            throw new NotSupportedException("Causal route-neighbor requires effective context exactly 2048.");
        return options.ContextTokens == context ? options : options with { ContextTokens = context };
    }
    internal static bool ProfileOptions(UnifiedMemoryModelOptions o, bool media)
        => !media && o.ContextTokens == 2048 && o.MaximumRunningRequests == 1
            && o.RetainedConversationCount == 0 && o.PrefixCheckpointCount == 0
            && o.EnableGlmSelectedExperts && o.EnableAdaptiveGlmPrefill && o.EnableExpertCacheCapacityGrowth
            && !o.EnableGlmFlashMtp && !o.EnableAsyncExpertPrefill && o.MaximumHostExpertCacheBytes == 0
            && o.ResolveGlmPrefillChunkTokens() == 512 && o.LayerMajorPrefillTokens > 512;
    internal static long ValidateHostPlan(ReadOnlySpan<long> plan)
    {
        if (plan.Length != 8 || plan[0] != 1 || plan[1] != 1 || plan[2] <= 0 || plan[2] > (1L << 20)
            || plan[3] != plan[2] || plan[4] != 42 || plan[5] != 288 || plan[6] != 64 || plan[7] != 8)
            throw new NotSupportedException("Native causal route-neighbor Host plan does not certify profile 1.");
        return plan[2];
    }
    private static void ValidateInfo(ReadOnlySpan<long> info)
    {
        if (info.Length != 16 || info[0] != 1)
            throw new InvalidOperationException("Native causal route-neighbor ownership certificate is malformed.");
        foreach (long value in info)
            if (value < 0) throw new InvalidOperationException("Native causal route-neighbor certificate has a negative field.");
        for (int i = 1; i <= 5; ++i)
            if (info[i] > 1) throw new InvalidOperationException("Native causal route-neighbor certificate has an invalid Boolean.");
        if (info[9] > 1)
            throw new InvalidOperationException("Native causal route-neighbor pending-query certificate is invalid.");
    }
    internal static void ValidateLoadedInfo(ReadOnlySpan<long> info)
    {
        ValidateInfo(info);
        if (info[1] != 1) throw new NotSupportedException("The loaded GLM handle did not capture the requested causal policy.");
        for (int i = 2; i < info.Length; ++i)
            if (info[i] != 0) throw new InvalidOperationException("A newly loaded causal policy already owns state.");
    }
    internal static void ValidateActiveInfo(ReadOnlySpan<long> info, long bytes)
    {
        ValidateInfo(info);
        if (bytes <= 0 || info[1] != 1 || info[2] != 1 || info[3] != 0 || info[4] != 0 || info[5] != 0
            || info[6] != bytes || info[7] != bytes || info[8] != bytes || info[9] != 0)
            throw new InvalidOperationException("Native causal route-neighbor did not certify its complete charged Host slab.");
    }
    internal static void ValidateClosedInfo(ReadOnlySpan<long> info, long bytes)
    {
        ValidateInfo(info);
        if (bytes < 0 || info[1] != 1 || info[3] != 0 || info[4] != 1
            || (info[6] != 0 && info[6] != bytes) || info[7] != 0 || info[8] != 0 || info[9] != 0)
            throw new InvalidOperationException("Native causal route-neighbor has not completed checked retirement.");
    }
}
