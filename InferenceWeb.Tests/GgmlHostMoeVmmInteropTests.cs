// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.GGML;
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

/// <summary>Actual additive native ABI without creating a CUDA backend, VMM
/// arena/table, model or payload. Device cache qualification is separate.</summary>
public sealed class GgmlHostMoeVmmInteropTests
{
    [GlmNativeHooksFact]
    public void ActualNativeLoaderRejectsUnguardedVmmBeforeBackendOrFileWork()
    {
        using var environment = new NativeEnvScope();
        environment.Set(GgmlHostMoeExpertCacheScope.VmmWeightsEnvironmentVariable, "1");
        environment.Set("TS_GLM_HOST_TOKEN_EMBEDDING", "0");
        environment.Set("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", "0");
        IntPtr model = GgmlGlmNative.LoadModel("must-not-be-opened-vmm-guard.gguf", 1, 2048, 512, 1,
            backendName: "CUDA", tp: 1, ctxIsHardLimit: true, loadMtp: false);
        Assert.Equal(IntPtr.Zero, model);
        Assert.Contains("VMM expert weights require", GgmlNative.LastNativeError("missing refusal"));
    }

    [GlmNativeHooksFact]
    public void ActualPureHostPlanAndCheckedFreeHandshakeCreateNoBudgetOwners()
    {
        var budget = new MemoryBudget([new("device", 1), new("host", 128L << 20)]);
        using var native = new GgmlCacheBudgetScope(budget, [["device"]], true, ["host"], requireCheckedRelease: true);
        long first = GgmlHostMoeExpertCacheScope.PlanVmmWeightsHostBytes();
        Assert.True(first > (8L << 20));
        Assert.Equal(first, GgmlHostMoeExpertCacheScope.PlanVmmWeightsHostBytes());
        GgmlGlmNative.RequireCheckedFreeSupport();
        Assert.True(GgmlGlmNative.FreeChecked(IntPtr.Zero, required: true));
        Assert.Equal(0, native.ActiveAllocations);
        Assert.Empty(native.AllocationUsage);
        Assert.Null(native.CallbackError);
        Assert.All(budget.Snapshot(), pool => { Assert.Equal(0, pool.Reserved); Assert.Equal(0, pool.Committed); });
    }

    [GlmNativeHooksFact]
    public void ActualOffExpertScopeDoesNotCreateVmmMetadataOrTickets()
    {
        using var environment = new NativeEnvScope();
        environment.Set(GgmlHostMoeExpertCacheScope.VmmWeightsEnvironmentVariable, "0");
        var budget = new MemoryBudget([new("device", 1L << 20), new("host", 128L << 20)]);
        using var native = new GgmlCacheBudgetScope(budget, [["device"]], true, ["host"], requireCheckedRelease: true);
        using (var experts = new GgmlHostMoeExpertCacheScope(1L << 20, 512, 3))
        {
            var usage = experts.ReadVmmUsage();
            Assert.Equal(1, usage.Version);
            Assert.Equal(0, usage.Configured);
            Assert.Equal(0, usage.HostTicketOwners);
            Assert.Equal(0, usage.LiveTables);
            Assert.Equal(0, usage.OwnedDeviceBytes);
            Assert.Equal(0, usage.PendingTicketBytes);
            Assert.Equal(0, usage.QuarantinedBytes);
            experts.PrepareForModelDispose();
        }
        Assert.Equal(0, native.ActiveAllocations);
        Assert.Null(native.CallbackError);
    }
}
