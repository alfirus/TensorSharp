// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

public static partial class GgmlGlmNative
{
    [DllImport(DllName, CallingConvention = Conv)]
    private static extern int TSGgml_GlmCausalRouteNeighborVersion();
    [DllImport(DllName, CallingConvention = Conv)]
    private static extern int TSGgml_GlmCausalRouteNeighborEnabled();
    [DllImport(DllName, CallingConvention = Conv)]
    private static extern int TSGgml_GlmCausalRouteNeighborHostPlan(int profile, [Out] long[] values, int length);
    [DllImport(DllName, CallingConvention = Conv)]
    private static extern int TSGgml_GlmConfigureCausalRouteNeighbor(IntPtr handle, ulong scope, int profile, long expectedHostBytes);
    [DllImport(DllName, CallingConvention = Conv)]
    private static extern int TSGgml_GlmCausalRouteNeighborInfo(IntPtr handle, [Out] long[] values, int length);
    [DllImport(DllName, CallingConvention = Conv)]
    private static extern int TSGgml_GlmCloseCausalRouteNeighbor(IntPtr handle);

    public static void RequireCausalRouteNeighborSupport()
        => CompleteCausalRouteNeighborHandshake(true, TSGgml_GlmCausalRouteNeighborVersion,
            TSGgml_GlmCausalRouteNeighborEnabled, () => TSGgml_GlmCausalRouteNeighborInfo(IntPtr.Zero,new long[16],16),
            () => TSGgml_GlmCloseCausalRouteNeighbor(IntPtr.Zero));
    internal static void CompleteCausalRouteNeighborHandshake(bool requested, Func<int> version, Func<int> enabled,
        Func<int> nullInfo, Func<int> nullClose)
    {
        if (!requested) return;
        try {
            if (version() != 1 || enabled() != 1 || nullInfo() != 0 || nullClose() != 1)
                throw new NotSupportedException("Causal route-neighbor requires matching requested native profile 1 and checked lifecycle exports.");
        } catch (EntryPointNotFoundException error) {
            throw new NotSupportedException("Causal route-neighbor requires the matching GgmlOps native library.",error);
        }
    }
    public static long[] CausalRouteNeighborHostPlan()
    {
        RequireCausalRouteNeighborSupport();
        var values = new long[8];
        if (TSGgml_GlmCausalRouteNeighborHostPlan(1,values,values.Length) != 1)
            throw new NotSupportedException("Native causal route-neighbor Host planning was refused.");
        return values;
    }
    internal static void ConfigureCausalRouteNeighbor(IntPtr handle, ulong scope, long expectedHostBytes)
    {
        if (TSGgml_GlmConfigureCausalRouteNeighbor(handle,scope,1,expectedHostBytes) != 1)
            throw new InvalidOperationException("Native causal route-neighbor configuration failed; its owner must complete checked cleanup.");
    }
    public static long[] CausalRouteNeighborInfo(IntPtr handle)
    {
        var values = new long[16];
        if (TSGgml_GlmCausalRouteNeighborInfo(handle,values,values.Length) != 1)
            throw new InvalidOperationException("Native causal route-neighbor ownership information was refused.");
        return values;
    }
    public static void CloseCausalRouteNeighbor(IntPtr handle)
        => CompleteCausalRouteNeighborClose(() => TSGgml_GlmCloseCausalRouteNeighbor(handle));
    internal static void CompleteCausalRouteNeighborClose(Func<int> close)
    {
        if (close() != 1)
            throw new InvalidOperationException("Native causal route-neighbor checked close failed; retain its model, ticket and Host envelope for retry.");
    }
}
