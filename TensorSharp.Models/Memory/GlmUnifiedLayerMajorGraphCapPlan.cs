// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using TensorSharp.Memory.Planning;

namespace TensorSharp.Models;

/// <summary>An explicit experimental limit on declared scheduler payload. Native
/// enforcement proves the bound for successful calls; it does not promise that
/// an eligible graph fits, or bound CUDA libraries, scratch and driver pools.</summary>
internal static class GlmUnifiedLayerMajorGraphCapPlan
{
    internal const string EnvironmentVariable = "TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB";
    internal const int Version = 1;
    internal const int ChunkRows = 512;
    internal const int Context = 2048;
    internal const long GenerationAllowance = 64L << 20;

    internal static long Resolve(string? value) => value switch
    {
        null or "0" => 0,
        "512" => 512L << 20,
        "768" => 768L << 20,
        _ => throw new ArgumentException(EnvironmentVariable + " must be unset, 0, 512 or 768.")
    };

    internal static bool ValidCap(long bytes) => bytes is (512L << 20) or (768L << 20);

    internal static void RequireCertificate(int version, bool configured)
    {
        if (version != Version || !configured)
            throw new NotSupportedException("The native GLM handle did not certify strict layer-major graph cap profile 1.");
    }

    internal static long Certify(long cap, Func<int> version, Func<int, long, bool> configure)
    {
        if (cap == 0) return 0; // Disabled scopes must not depend on the new ABI.
        if (!ValidCap(cap)) throw new ArgumentOutOfRangeException(nameof(cap));
        int actual = version();
        if (actual != Version) RequireCertificate(actual, false);
        RequireCertificate(actual, configure(Version, cap));
        return cap;
    }

    internal static bool ProfileGeometry(GlmUnifiedMemoryGeometry g, int context)
        => GlmDsaArchitecture.IsGlm5Next(g.Architecture) && context == Context && g.DeclaredContextTokens >= Context
            && g.Layers == 45 && g.RecurrentLayers == 34 && g.Hidden == 4096 && g.Heads == 64
            && g.Vocab == 154880 && g.KvRow == 512 && g.IndexerDim == 128
            && g.KdaDim == 128 && g.ConvKernel == 4 && g.HyperConnections == 4
            && g.ExpertFf == 2048 && g.UsedExperts == 8 && g.ExpertCount == 288 && g.Mtp == null;

    internal static bool ProfileOptions(UnifiedMemoryModelOptions options, bool media)
        => options.EnableGlmSelectedExperts && !options.EnableGlmFlashMtp && !media
            && options.MaximumRunningRequests == 1 && options.RetainedConversationCount == 0
            && options.PrefixCheckpointCount == 0 && options.ResolveGlmPrefillChunkTokens() == ChunkRows
            && options.LayerMajorPrefillTokens > ChunkRows;

    // Select the actual strategy before admission. A short tail must never
    // acquire the strict credit and then enter ordinary full-trunk fallback.
    internal static bool Select(int submitted, int chunk, int context, int nPast,
        bool native, int tensorParallel, bool overrides, bool tracing)
        => native && tensorParallel == 1 && !overrides && !tracing && chunk == ChunkRows
            && context == Context && nPast >= 0 && submitted > chunk && submitted <= context - nPast
            && (submitted % chunk == 0 || submitted % chunk >= 9);

    internal static bool Eligible(GlmUnifiedGraphShape shape, long cap, int logicalContext, bool primaryText)
        => ValidCap(cap) && primaryText && shape.StrictLayerMajor && !shape.Batched
            && shape.Rows == ChunkRows && shape.OutputRows is 0 or 1 && !shape.SingleNativeGraph
            && shape.ContextTokens > 0 && shape.ContextTokens <= logicalContext
            && Select(shape.SubmittedTokens, shape.Rows, logicalContext,
                shape.ContextTokens - shape.SubmittedTokens, true, 1, false, false);

    internal static bool MustPrepare(GlmUnifiedGraphShape shape, GlmUnifiedGraphShape? previous,
        bool strict, bool previousStrict)
        => strict || previousStrict || previous != shape;

    internal static InferenceMemoryBytes Target(InferenceMemoryBytes broad, InferenceMemoryBytes live, long cap)
    {
        if (!ValidCap(cap) || broad.Host < 0 || broad.Device < 0 || live.Host < 0 || live.Device < 0)
            throw new ArgumentOutOfRangeException(nameof(cap));
        long generation = checked(cap + GenerationAllowance);
        return new(Math.Max(broad.Host, live.Host), Math.Max(checked(2 * generation), checked(live.Device + generation)));
    }

    internal static InferenceMemoryBytes PreservePromise(InferenceMemoryBytes graph, InferenceMemoryBytes promise,
        bool executingRequest, bool strict)
        => new(Math.Max(graph.Host, promise.Host), executingRequest && strict ? graph.Device
            : Math.Max(graph.Device, promise.Device));

    // Negative means the legacy native export requested ordinary fallback.
    // A certified handle must fail closed even if an incompatible DLL returns it.
    internal static bool AllowLegacyFallback(int result, bool strictHandle) => result < 0 && !strictHandle;

    internal static bool Complete(int result, bool strictHandle, Func<bool> fallback)
    {
        if (result >= 0) return result == 1;
        if (!AllowLegacyFallback(result, strictHandle))
            throw new InvalidOperationException("Strict native layer-major execution requested an unauthorized ordinary fallback.");
        return fallback();
    }
}
