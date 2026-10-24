// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
public sealed class GlmStrictGraphCapCheckpointTheoryAttribute : TheoryAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda,Models,NativeTestHooks";
    public GlmStrictGraphCapCheckpointTheoryAttribute()
    {
        Skip = string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_GLM_STRICT_GRAPH_CAP_MODEL"))
            ? "Requires explicit TS_TEST_GLM_STRICT_GRAPH_CAP_MODEL for the full checkpoint failure test."
            : Environment.GetEnvironmentVariable("TS_TEST_GLM_NATIVE_HOOKS") != "1"
                ? "Requires TS_TEST_GLM_NATIVE_HOOKS=1 and matching native test exports."
                : TestGates.CudaSkip;
    }
}

[CollectionDefinition("GLM strict graph cap checkpoint", DisableParallelization = true)]
public sealed class GlmStrictGraphCapCheckpointCollection { }

/// <summary>Explicit full-checkpoint failure qualification. This deliberately
/// retains production profile gates and never substitutes a tiny model. Run
/// alone with the model path and native test hooks; skipped is not coverage.</summary>
[Collection("GLM strict graph cap checkpoint")]
public sealed class GlmStrictGraphCapCheckpointTests
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ArmFailure(IntPtr model, int layer);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate long FailureCount(IntPtr model);
    private const long MiB = 1L << 20, GiB = 1L << 30;

    [GlmStrictGraphCapCheckpointTheory]
    [InlineData(512)]
    [InlineData(768)]
    public void CompletedLayerFailurePublishesNothingPoisonsSlotAndCheckedResetRecovers(long capMiB)
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_GLM_STRICT_GRAPH_CAP_MODEL")!;
        Assert.True(File.Exists(path), "The explicit full checkpoint path must exist; this is not synthetic fallback.");
        using var env = ConfigureEnvironment();
        GlmUnifiedMemoryGeometry geometry;
        using (var file = GgufFile.OpenForMemoryPlanning(path)) geometry = GlmUnifiedMemoryGeometry.Read(file);
        Assert.True(GlmUnifiedLayerMajorGraphCapPlan.ProfileGeometry(geometry, 2048));
        int groups = geometry.ExpertLayerLayouts.OfType<GlmUnifiedExpertLayout>().Select(layout => layout.Types).Distinct().Count();
        Assert.True(groups > 0);
        long expertForecast = GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 1, 512);
        var hardware = InferenceHardwareMemory.CaptureCuda();
        var host = hardware.HostAllocationCapacity();
        long hostHeadroom = Math.Max(512 * MiB, Math.Min(host.Total, 28 * GiB) / 16);
        long deviceHeadroom = Math.Max(768 * MiB, Math.Min(hardware.DeviceTotal, 14 * GiB) / 16);
        long hostCeiling = Math.Min(28 * GiB, Math.Max(0, host.Available - hostHeadroom));
        long deviceCeiling = Math.Min(14 * GiB, Math.Max(0, hardware.DeviceAvailable - deviceHeadroom));
        Assert.True(hostCeiling > 0 && deviceCeiling > 0, "No capacity remains after physical headroom; fixture unavailable.");
        var budget = new MemoryBudget([new("host", hostCeiling), new("device", deviceCeiling)]);
        GgmlBasicOps.ClearHostBufferCache();
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        using var native = new GgmlCacheBudgetScope(budget, [["device"], ["host"]], true, ["host"]);
        using var joint = budget.Reserve([new("host", 0), new("device", 0)]);
        native.SetSharedJointEnvelope(joint);
        using var expertOwner = budget.Reserve([new("device", expertForecast)]);
        native.SetSharedExpertEnvelope(expertOwner);
        using var experts = new GgmlHostMoeExpertCacheScope(expertForecast, 512, groups, 1);
        IntPtr model = IntPtr.Zero, library = IntPtr.Zero;
        ArmFailure? arm = null;
        long firedBefore = -1, firedAfter = -1;
        try
        {
            GgmlGlmNative.RequireUnifiedBudgetSupport();
            Assert.Equal(1, GgmlGlmNative.LayerMajorGraphCapVersion());
            model = GgmlGlmNative.LoadModel(path, 1, 2048, 512, 8, nCpuMoe: int.MaxValue,
                backendName: "CUDA", tp: 1, ctxIsHardLimit: true, loadMtp: false);
            Assert.NotEqual(IntPtr.Zero, model);
            Assert.False(GgmlGlmNative.HasDraftHead(model));
            Assert.Equal(groups, GgmlGlmNative.SelectedExpertGroupCount(model));
            long minimum = GgmlGlmNative.SelectedExpertMinimumBudget(model, 1);
            Assert.InRange(minimum, 1, expertForecast);
            experts.UpdateBudget(minimum);
            Assert.True(expertOwner.TryResizeTotal([new("device", minimum)]));
            Assert.True(GgmlGlmNative.ConfigureGraphCache(model, 1));
            Assert.True(GgmlGlmNative.ConfigureSharedGraphOwner(model, true));
            Assert.True(GgmlGlmNative.ConfigureSelectedExpertSeams(model, true));
            Assert.True(GgmlGlmNative.ConfigureLayerMajorGraphCap(model, 1, capMiB * MiB));
            Assert.True(GgmlGlmNative.ConfigureLayerMajorGraphCap(model, 1, capMiB * MiB));
            Assert.False(GgmlGlmNative.ConfigureLayerMajorGraphCap(model, 1, (capMiB == 512 ? 768 : 512) * MiB));
            Assert.False(GgmlGlmNative.ConfigureLayerMajorGraphCap(model, 2, capMiB * MiB));
            Assert.False(GgmlGlmNative.ConfigureGraphCache(model, 2));
            Assert.False(GgmlGlmNative.ConfigureSharedGraphOwner(model, false));
            Assert.False(GgmlGlmNative.ConfigureSelectedExpertSeams(model, false));
            Assert.True(GgmlGlmNative.ConfigureGraphCache(model, 1));
            Assert.True(GgmlGlmNative.ConfigureSharedGraphOwner(model, true));
            Assert.True(GgmlGlmNative.ConfigureSelectedExpertSeams(model, true));
            var peak = geometry.GraphPeak(2048, 512, false);
            Assert.True(joint.TryResizeTotal([new("host", checked(peak.Host + geometry.LayerMajorHostBytes(521))),
                new("device", peak.Device)]), "Full broad/strict replacement ownership must fit before execution.");
            library = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            arm = Marshal.GetDelegateForFunctionPointer<ArmFailure>(NativeLibrary.GetExport(library,
                "TSGgml_GlmTestFailStrictLayerMajorAfterLayer"));
            var fired = Marshal.GetDelegateForFunctionPointer<FailureCount>(NativeLibrary.GetExport(library,
                "TSGgml_GlmTestStrictLayerMajorFailureCount"));
            var baseline = new float[GgmlGlmNative.VocabSize(model)];
            var baselineNext = new float[baseline.Length];
            Assert.True(GgmlGlmNative.Forward(model, [65, 66, 67], baseline));
            Assert.True(GgmlGlmNative.Forward(model, [68], baselineNext));
            Assert.All(baseline, value => Assert.True(float.IsFinite(value)));
            Assert.True(GgmlGlmNative.ResetChecked(model));
            Assert.True(GgmlGlmNative.ConfigureGraphCache(model, 1, trim: 1));
            int head = GgmlGlmNative.NPast(model);
            Assert.Equal(0, head);
            firedBefore = fired(model);
            Assert.True(firedBefore >= 0);
            Assert.Equal(1, arm(model, 0));
            const float sentinel = -1234567.25f;
            int sentinelBits = BitConverter.SingleToInt32Bits(sentinel);
            var logits = Enumerable.Repeat(sentinel, baseline.Length).ToArray();
            int[] prompt = Enumerable.Range(0, 521).Select(i => 65 + i % 59).ToArray();
            Assert.Equal(0, GgmlGlmNative.ForwardLayerMajor(model, prompt, logits, 512));
            firedAfter = fired(model);
            Assert.Equal(firedBefore + 1, firedAfter); // Preflight or allocation refusal is not this proof.
            Assert.Equal(head, GgmlGlmNative.NPast(model));
            Assert.All(logits, value => Assert.Equal(sentinelBits, BitConverter.SingleToInt32Bits(value)));
            Assert.False(GgmlGlmNative.Forward(model, [65], logits));
            Assert.Equal(head, GgmlGlmNative.NPast(model));
            Assert.All(logits, value => Assert.Equal(sentinelBits, BitConverter.SingleToInt32Bits(value)));
            Assert.Equal(1, arm(model, -1));
            Assert.True(GgmlGlmNative.ResetChecked(model));
            Assert.True(GgmlGlmNative.Forward(model, [65, 66, 67], logits));
            Assert.Equal(baseline, logits);
            Assert.True(GgmlGlmNative.Forward(model, [68], logits));
            Assert.Equal(baselineNext, logits);
            Assert.Equal(4, GgmlGlmNative.NPast(model));
            Assert.Equal(firedAfter, fired(model));
            Assert.Null(native.CallbackError);
        }
        finally
        {
            if (model != IntPtr.Zero)
            {
                arm?.Invoke(model, -1);
                GgmlGlmNative.Free(model);
            }
            experts.Dispose();
            native.SetSharedExpertEnvelope(null);
            native.SetSharedJointEnvelope(null);
            if (library != IntPtr.Zero) NativeLibrary.Free(library);
        }
        Assert.Null(native.CallbackError);
        Assert.Equal(0, native.ActiveAllocations);
        Console.WriteLine($"[GLM-STRICT-GRAPH-CAP-FAILURE-TEST] cap_mib={capMiB} submitted=521 completed_layer=0 fired_before={firedBefore} fired_after={firedAfter} sentinel_exact=1 head_unpublished=1 continuation_refused=1 checked_reset_recovery_exact=1 host_ceiling={hostCeiling} device_ceiling={deviceCeiling} host_headroom={hostHeadroom} device_headroom={deviceHeadroom} scope=full_checkpoint_failure_safety_not_performance");
    }

    private static NativeEnvScope ConfigureEnvironment()
    {
        var env = new NativeEnvScope();
        foreach (string name in new[] { "TS_GLM_TRACE", "TS_GLM_TRACE_DUMP_DIR", "TS_GLM_NODES_PER_LAYER",
            "TS_GLM_MOE_MMAP", "TS_GLM_TP_SHARD", "TS_GLM_TP_OVERSUBSCRIBE" }) env.Set(name, null);
        foreach (string name in new[] { "TS_GLM_FLASH_MTP", "TS_GLM_FLASH_MTP_SHAPE_PLAN", "TS_GLM_FLASH_MTP_PREFIX_COMMIT",
            "TS_HOST_MOE_RAM_CACHE_MAPPED", "TS_HOST_MOE_RAM_CACHE_PINNED", "TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD",
            "TS_HOST_MOE_RAM_CACHE_VICTIM", "TS_HOST_MOE_LAYER_PREFETCH", "TS_HOST_MOE_DECODE_PREFETCH",
            "TS_HOST_MOE_DECODE_STAGING_72MB", "TS_HOST_MOE_PREFILL_RING_72MB", "TS_HOST_MOE_PREFILL_RING_COALESCE16" }) env.Set(name, "0");
        return env;
    }
}
