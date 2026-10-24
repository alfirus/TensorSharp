// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;

namespace TensorSharp.Models;

/// <summary>The optional arena retains its expanded allocation across shapes.
/// It remains host staging owned by the executing request or joint envelope;
/// it is neither a device graph generation nor a speculative kind-7 child.</summary>
internal static class GlmUnifiedFileStagingPlan
{
    internal const string EnvironmentVariable = "TS_HOST_MOE_DECODE_STAGING_72MB";
    internal const string PrefillRingEnvironmentVariable = "TS_HOST_MOE_PREFILL_RING_72MB";
    internal const long AdditionalHostBytes = 40L << 20;

    internal static bool Resolve(string? option, bool nativeCapable)
        => ResolveMode(option, nativeCapable, EnvironmentVariable);

    internal static bool ResolvePrefillRing(string? option, bool nativeCapable)
        => ResolveMode(option, nativeCapable, PrefillRingEnvironmentVariable);

    private static bool ResolveMode(string? option, bool nativeCapable, string environmentVariable)
    {
        // Native environment flags are latched for the process. Never underprice
        // a later session after its caller removes an already active opt-in.
        if (option != "1") return !nativeCapable ? false
            : throw new NotSupportedException("Native GLM staging is already enabled; preserve "
                + environmentVariable + "=1 or start a fresh process before loading another session.");
        return nativeCapable ? true
            : throw new NotSupportedException(environmentVariable + "=1 requires the matching enabled native staging capability.");
    }
}
