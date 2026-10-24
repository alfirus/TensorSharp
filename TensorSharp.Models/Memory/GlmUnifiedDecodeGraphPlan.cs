// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using TensorSharp.Memory.Planning;

namespace TensorSharp.Models;

/// <summary>Experimental device-only admission for one native graph acquisition.
/// This is one conservative replacement forecast alongside every live child,
/// not a claim that a native cache key will hit or that its size is exact.</summary>
internal static class GlmUnifiedDecodeGraphPlan
{
    internal static bool Enabled(string? value) => value == "1";

    internal static bool SingleAcquire(int submittedTokens, int nativeUbatch, bool nativeExecutor,
        int tensorParallel, bool layerMajor, bool overrides, bool tracing)
        => nativeExecutor && tensorParallel == 1 && !layerMajor && !overrides && !tracing
            && submittedTokens > 0 && submittedTokens <= nativeUbatch;

    internal static bool Eligible(GlmUnifiedGraphShape shape, int prefillRows, int decodeRows,
        bool primaryDecode, bool outstandingPrefill)
        => primaryDecode && !outstandingPrefill && shape.SingleNativeGraph && !shape.Batched
            && shape.ContextTokens > 0 && decodeRows is >= 1 and <= 3 && prefillRows > 0
            && shape.Rows > 0 && shape.Rows <= decodeRows && shape.Rows <= prefillRows
            && shape.SubmittedTokens == shape.Rows && shape.OutputRows >= 0 && shape.OutputRows <= shape.Rows;

    internal static bool MustPrepare(GlmUnifiedGraphShape shape, GlmUnifiedGraphShape? previous, bool eligible,
        bool previousEligible = false)
        => eligible || previousEligible || previous != shape;

    internal static long RecoveryMinimum(long releasedDeviceBytes, bool firstEligibleDecode)
        => firstEligibleDecode && releasedDeviceBytes >= (64L << 20) ? 64L << 20 : 512L << 20;

    internal static InferenceMemoryBytes Target(InferenceMemoryBytes broad, InferenceMemoryBytes live,
        InferenceMemoryBytes workspace, bool eligible)
    {
        if (!eligible) return broad;
        if (live.Host < 0 || live.Device < 0 || workspace.Host < 0 || workspace.Device < 0
            || broad.Host < live.Host)
            throw new ArgumentOutOfRangeException(nameof(live));
        // Native metadata is not fully described by allocator children. Preserve
        // the existing Host forecast; only device buffers use live+one here.
        return new(broad.Host, checked(live.Device + workspace.Device));
    }
}
