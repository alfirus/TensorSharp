// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

internal static partial class GgmlNative
{
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeSetLayerResidency")]
    internal static partial int HostMoeSetLayerResidency(int enable);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeLayerResidencyHint")]
    internal static partial int HostMoeLayerResidencyHint(IntPtr gate, IntPtr up, IntPtr down,
        long gateBytes, long upBytes, long downBytes, int experts);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeLayerResidencyStats")]
    internal static partial int HostMoeLayerResidencyStats(out long loads, out long hits, out long loadBytes);
}

/// <summary>Cumulative layer-major prefill residency counters for this process:
/// complete expert stacks staged from file, chunk executions that reused the
/// staged stack, and the staged payload bytes.</summary>
public readonly record struct GgmlHostMoeLayerResidencyUsage(long Loads, long Hits, long LoadBytes);

public partial class GgmlBasicOps
{
    /// <summary>Enter or leave layer-major prefill. While on, a file-backed
    /// host-MoE seam stages its layer's complete expert stack on the device once
    /// and reuses it for every following chunk of that layer, bypassing the
    /// decode expert cache. Leaving frees the staged layer. Call only between
    /// forwards, under the model's compute lock.</summary>
    public static void SetHostMoeLayerResidency(bool enabled)
    {
        if (GgmlNative.HostMoeSetLayerResidency(enabled ? 1 : 0) != 1)
            throw new InvalidOperationException("GGML layer-major expert residency change failed: " + GgmlNative.LastNativeError());
    }

    /// <summary>Name the expert stacks of the layer that will run next, so their
    /// file read overlaps the current layer's compute. IntPtr.Zero clears it.</summary>
    public static void SetHostMoeLayerResidencyHint(IntPtr gate, IntPtr up, IntPtr down,
        long gateBytes, long upBytes, long downBytes, int experts)
    {
        if (GgmlNative.HostMoeLayerResidencyHint(gate, up, down, gateBytes, upBytes, downBytes, experts) != 1)
            throw new InvalidOperationException("GGML layer residency hint failed: " + GgmlNative.LastNativeError());
    }

    public static GgmlHostMoeLayerResidencyUsage GetHostMoeLayerResidencyUsage()
    {
        if (GgmlNative.HostMoeLayerResidencyStats(out long loads, out long hits, out long bytes) != 1)
            throw new InvalidOperationException("GGML layer residency usage read failed: " + GgmlNative.LastNativeError());
        return new(loads, hits, bytes);
    }
}
