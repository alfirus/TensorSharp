// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class GlmDsaModel
{
    private long _causalRouteNeighborHostBytes;
    internal void ConfigureUnifiedCausalRouteNeighbor(GgmlHostMoeExpertCacheScope scope, long bytes)
    {
        lock (_nativeSync)
        {
            if (MemoryPolicy?.EnableGlmCausalRouteNeighbor != true || _exec is not NativeGlmExecutor native
                || _causalRouteNeighborHostBytes != 0 || bytes <= 0)
                throw new InvalidOperationException("The causal route-neighbor model owner is unavailable or already configured.");
            // Adopt the expected size before the native call can retain a slab
            // or failed-ACK ticket. The before-delete bridge is already installed.
            _causalRouteNeighborHostBytes = bytes;
            native.ConfigureCausalRouteNeighbor(scope,bytes);
            GlmUnifiedCausalRouteNeighborPlan.ValidateActiveInfo(native.CausalRouteNeighborInfo,bytes);
        }
    }
    internal long[] UnifiedCausalRouteNeighborInfo
    {
        get {
            lock (_nativeSync)
                return _exec is NativeGlmExecutor native ? native.CausalRouteNeighborInfo
                    : throw new NotSupportedException("Causal route-neighbor requires the native GLM executor.");
        }
    }
    internal void PrepareUnifiedCausalRouteNeighborDispose()
    {
        lock (_nativeSync)
        {
            if (MemoryPolicy?.EnableGlmCausalRouteNeighbor != true) return;
            if (_exec is not NativeGlmExecutor native)
                throw new InvalidOperationException("Causal route-neighbor lost its native cleanup owner.");
            native.CloseCausalRouteNeighbor();
            GlmUnifiedCausalRouteNeighborPlan.ValidateClosedInfo(native.CausalRouteNeighborInfo,_causalRouteNeighborHostBytes);
        }
    }
}
