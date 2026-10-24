// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
public sealed class GlmHostTokenEmbeddingCheckpointTheoryAttribute : TheoryAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda,Models,NativeTestHooks";
    public GlmHostTokenEmbeddingCheckpointTheoryAttribute()
    {
        Skip = string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_GLM_HOST_TOKEN_EMBEDDING_MODEL"))
            ? "Requires explicit TS_TEST_GLM_HOST_TOKEN_EMBEDDING_MODEL for the full checkpoint raw-pack failure test."
            : Environment.GetEnvironmentVariable("TS_TEST_GLM_NATIVE_HOOKS") != "1"
                ? "Requires TS_TEST_GLM_NATIVE_HOOKS=1 and matching native test exports."
                : TestGates.CudaSkip;
    }
}

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
public sealed class GlmPrefillDownOverlapCheckpointTheoryAttribute : TheoryAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda,Models,NativeTestHooks";
    public GlmPrefillDownOverlapCheckpointTheoryAttribute()
    {
        Skip = string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_GLM_PREFILL_DOWN_OVERLAP_MODEL"))
            ? "Requires explicit TS_TEST_GLM_PREFILL_DOWN_OVERLAP_MODEL for the full checkpoint overlap failure test."
            : Environment.GetEnvironmentVariable("TS_TEST_GLM_NATIVE_HOOKS") != "1"
                ? "Requires TS_TEST_GLM_NATIVE_HOOKS=1 and matching native overlap test exports."
                : TestGates.CudaSkip;
    }
}

[CollectionDefinition("GLM host token embedding checkpoint", DisableParallelization = true)]
public sealed class GlmHostTokenEmbeddingCheckpointCollection { }

/// <summary>Explicit real-checkpoint safety and full-vocabulary parity, run
/// alone with accounted CUDA/Host ownership. A skipped test is not coverage.</summary>
[Collection("GLM host token embedding checkpoint")]
public sealed class GlmHostTokenEmbeddingCheckpointTests
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ArmPackFailure(IntPtr model, int successfulPacksBeforeFailure);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate long PackFailureCount(IntPtr model);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ArmOverlapFailure(int stage);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ReadOverlapStats([Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] ulong[] values, int length);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ReadOverlapCapability();
    private sealed class NativeModule : IDisposable
    {
        internal IntPtr Handle { get; } = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
        public void Dispose() => NativeLibrary.Free(Handle);
    }
    private const long MiB = 1L << 20, GiB = 1L << 30;
    private const float Sentinel = -1234567.25f;

    [GlmHostTokenEmbeddingCheckpointTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void SecondPackFailurePublishesNothingPoisonsSlotAndResetRestoresResidentBaseline(bool layerMajor)
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_GLM_HOST_TOKEN_EMBEDDING_MODEL")!;
        Assert.True(File.Exists(path), "The explicit actual checkpoint must exist; synthetic fallback is not coverage.");
        using var env = ConfigureEnvironment();
        using var library = new NativeModule(); // Outlives both model and expert teardown paths.
        float[] resident = [], residentNext = [];
        // The old loader must run with mode0. Mode1 uses only the new explicit
        // loader; a successful resident fallback cannot satisfy this fixture.
        WithModel(path, false, (model, _) =>
        {
            resident = new float[GgmlGlmNative.VocabSize(model)];
            residentNext = new float[resident.Length];
            Assert.True(GgmlGlmNative.Forward(model, [65, 66, 67], resident));
            Assert.True(GgmlGlmNative.Forward(model, [68], residentNext));
            Assert.All(resident, value => Assert.True(float.IsFinite(value)));
            Assert.All(residentNext, value => Assert.True(float.IsFinite(value)));
        });

        long firedBefore = -1, firedAfter = -1;
        WithModel(path, true, (model, native) =>
        {
            GlmUnifiedHostTokenEmbeddingPlan.ValidateLoadedInfo(GgmlGlmNative.HostTokenEmbeddingInfo(model));
            ArmPackFailure? arm = null;
            try
            {
                arm = Marshal.GetDelegateForFunctionPointer<ArmPackFailure>(NativeLibrary.GetExport(library.Handle,
                    "TSGgml_GlmTestFailHostTokenEmbeddingPack"));
                var fired = Marshal.GetDelegateForFunctionPointer<PackFailureCount>(NativeLibrary.GetExport(library.Handle,
                    "TSGgml_GlmTestHostTokenEmbeddingPackFailureCount"));
                int vocab = GgmlGlmNative.VocabSize(model);
                Assert.Equal(resident.Length, vocab);
                Assert.Equal(0, GgmlGlmNative.NPast(model));
                firedBefore = fired(model);
                Assert.True(firedBefore >= 0);
                Assert.Equal(1, arm(model, 1));
                var logits = Enumerable.Repeat(Sentinel, vocab).ToArray();
                int[] prompt = Enumerable.Range(0, 521).Select(i => 65 + i % 59).ToArray();
                var invalid = (int[])prompt.Clone();
                invalid[^1] = vocab;
                if (layerMajor) Assert.Equal(0, GgmlGlmNative.ForwardLayerMajor(model, invalid, logits, 512));
                else Assert.False(GgmlGlmNative.Forward(model, invalid, logits));
                Assert.Equal(firedBefore, fired(model));
                Assert.Equal(0, GgmlGlmNative.NPast(model));
                AssertSentinel(logits);
                // A valid one-pack call must still succeed without Reset. If
                // invalid-tail validation consumed the countdown, it fails here.
                Assert.True(GgmlGlmNative.Forward(model, [65, 66, 67], logits));
                AssertLogitsExact(resident, logits);
                Assert.Equal(firedBefore, fired(model));
                Assert.Equal(1, arm(model, -1));
                Assert.True(GgmlGlmNative.Forward(model, [68], logits));
                AssertLogitsExact(residentNext, logits);
                Assert.Equal(4, GgmlGlmNative.NPast(model));

                Assert.True(GgmlGlmNative.ResetChecked(model));
                Assert.True(GgmlGlmNative.ConfigureGraphCache(model, 1, trim: 1));
                Assert.Equal(0, GgmlGlmNative.NPast(model));
                Assert.Equal(1, arm(model, 1));
                Array.Fill(logits, Sentinel);
                // Countdown1 is consumed only by actual pack copies. One real
                // 512-row chunk completes compute/fences before the second
                // 9-row copier writes a strict raw-row prefix then fails.
                if (layerMajor) Assert.Equal(0, GgmlGlmNative.ForwardLayerMajor(model, prompt, logits, 512));
                else Assert.False(GgmlGlmNative.Forward(model, prompt, logits));
                firedAfter = fired(model);
                Assert.Equal(firedBefore + 1, firedAfter); // Not preflight/cap/refusal coverage.
                Assert.Equal(0, GgmlGlmNative.NPast(model));
                AssertSentinel(logits);
                Assert.False(GgmlGlmNative.Forward(model, [65], logits));
                Assert.Equal(0, GgmlGlmNative.NPast(model));
                AssertSentinel(logits);
                Assert.Equal(firedAfter, fired(model));

                Assert.Equal(1, arm(model, -1));
                Assert.True(GgmlGlmNative.ResetChecked(model));
                Assert.True(GgmlGlmNative.Forward(model, [65, 66, 67], logits));
                AssertLogitsExact(resident, logits);
                Assert.True(GgmlGlmNative.Forward(model, [68], logits));
                AssertLogitsExact(residentNext, logits);
                Assert.Equal(4, GgmlGlmNative.NPast(model));
                Assert.Equal(firedAfter, fired(model));
                Assert.Null(native.CallbackError);
            }
            finally
            {
                arm?.Invoke(model, -1);
            }
        });
        Console.WriteLine($"[GLM-HOST-TOKEN-EMBEDDING-FAILURE-TEST] layer_major={(layerMajor ? 1 : 0)} submitted=521 first_chunk=512 failing_chunk=9 fired_before={firedBefore} fired_after={firedAfter} invalid_tail_preflight=1 hook_unconsumed=1 sentinel_exact=1 head_unpublished=1 continuation_refused=1 checked_reset_full_vocab_resident_bitexact=1 scope=full_checkpoint_failure_safety_not_performance");
    }

    // The native flag is process-latched: run this separately from the raw-pack
    // fixture with a fresh process, rather than changing an existing handle's mode.
    [GlmPrefillDownOverlapCheckpointTheory]
    [InlineData(27)]
    [InlineData(31)]
    public void SubmittedOverlapFailureDrainsBothStreamsPublishesNothingAndResetRecovers(int stage)
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_GLM_PREFILL_DOWN_OVERLAP_MODEL")!;
        Assert.True(File.Exists(path), "The explicit actual checkpoint must exist; synthetic fallback is not coverage.");
        using var env = ConfigureEnvironment();
        env.Set("TS_HOST_MOE_PREFILL_DOWN_OVERLAP", "1");
        env.Set("TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "0");
        env.Set("TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS", "0");
        using var library = new NativeModule();
        var version = Marshal.GetDelegateForFunctionPointer<ReadOverlapCapability>(NativeLibrary.GetExport(library.Handle,
            "TSGgml_HostMoePrefillDownOverlapVersion"));
        var enabled = Marshal.GetDelegateForFunctionPointer<ReadOverlapCapability>(NativeLibrary.GetExport(library.Handle,
            "TSGgml_HostMoePrefillDownOverlapEnabled"));
        Assert.Equal(1, version());
        Assert.Equal(1, enabled());
        var arm = Marshal.GetDelegateForFunctionPointer<ArmOverlapFailure>(NativeLibrary.GetExport(library.Handle,
            "TSGgml_TestHostMoePrefillDownOverlapFailNext"));
        var read = Marshal.GetDelegateForFunctionPointer<ReadOverlapStats>(NativeLibrary.GetExport(library.Handle,
            "TSGgml_TestHostMoePrefillDownOverlapStats"));
        ulong[] Snapshot()
        {
            var values = new ulong[5];
            Assert.Equal(1, read(values, values.Length));
            return values;
        }
        ulong[] before = [], after = [];
        WithModel(path, true, (model, native) =>
        {
            GlmUnifiedHostTokenEmbeddingPlan.ValidateLoadedInfo(GgmlGlmNative.HostTokenEmbeddingInfo(model));
            try
            {
                Assert.Equal(1, arm(0));
                int vocab = GgmlGlmNative.VocabSize(model);
                var baseline = new float[vocab];
                var baselineNext = new float[vocab];
                Assert.True(GgmlGlmNative.Forward(model, [65, 66, 67], baseline));
                Assert.True(GgmlGlmNative.Forward(model, [68], baselineNext));
                Assert.All(baseline, value => Assert.True(float.IsFinite(value)));
                Assert.All(baselineNext, value => Assert.True(float.IsFinite(value)));
                Assert.Equal(4, GgmlGlmNative.NPast(model));
                Assert.True(GgmlGlmNative.ResetChecked(model));
                Assert.True(GgmlGlmNative.ConfigureGraphCache(model, 1, trim: 1));
                Assert.Equal(0, GgmlGlmNative.NPast(model));
                before = Snapshot();
                Assert.Equal(1, arm(stage));
                var logits = Enumerable.Repeat(Sentinel, vocab).ToArray();
                int[] prompt = Enumerable.Range(0, 521).Select(i => 65 + i % 59).ToArray();
                Assert.Equal(0, GgmlGlmNative.ForwardLayerMajor(model, prompt, logits, 512));
                after = Snapshot();
                // These deltas exclude an admission/preflight failure: the
                // first real overlap prefix was queued and both streams drained.
                Assert.Equal(before[0], after[0]);
                Assert.Equal(before[1] + 1, after[1]);
                Assert.Equal(before[2], after[2]);
                Assert.Equal(before[3] + 1, after[3]);
                Assert.Equal(before[4] + (stage == 31 ? 1UL : 0UL), after[4]);
                Assert.Equal(0, GgmlGlmNative.NPast(model));
                AssertSentinel(logits);
                Assert.False(GgmlGlmNative.Forward(model, [65], logits));
                Assert.Equal(0, GgmlGlmNative.NPast(model));
                AssertSentinel(logits);
                Assert.Equal(after, Snapshot());
                Assert.Equal(1, arm(0));
                Assert.True(GgmlGlmNative.ResetChecked(model));
                Assert.True(GgmlGlmNative.Forward(model, [65, 66, 67], logits));
                AssertLogitsExact(baseline, logits);
                Assert.True(GgmlGlmNative.Forward(model, [68], logits));
                AssertLogitsExact(baselineNext, logits);
                Assert.Equal(4, GgmlGlmNative.NPast(model));
                Assert.Equal(after, Snapshot());
                Assert.Null(native.CallbackError);
            }
            finally
            {
                Assert.Equal(1, arm(0));
            }
        });
        Console.WriteLine($"[GLM-PREFILL-DOWN-OVERLAP-FAILURE-TEST] stage={stage} host_embedding=1 layer_major=1 submitted=521 prefix_before={before[1]} prefix_after={after[1]} drains_before={before[3]} drains_after={after[3]} partial_down_reads_before={before[4]} partial_down_reads_after={after[4]} completed_unchanged=1 suffix_unsubmitted=1 sentinel_exact=1 head_unpublished=1 continuation_refused=1 checked_reset_full_vocab_host_bitexact=1 scope=full_checkpoint_failure_safety_not_performance");
    }

    private static void AssertSentinel(float[] logits)
        => Assert.All(logits, value => Assert.Equal(BitConverter.SingleToInt32Bits(Sentinel), BitConverter.SingleToInt32Bits(value)));

    private static void AssertLogitsExact(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i)
            Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
    }

    private static void WithModel(string path, bool hostEmbedding, Action<IntPtr, GgmlCacheBudgetScope> action)
    {
        using var mode = new NativeEnvScope();
        mode.Set(GlmUnifiedHostTokenEmbeddingPlan.EnvironmentVariable, hostEmbedding ? "1" : "0");
        GlmUnifiedMemoryGeometry geometry;
        using (var file = GgufFile.OpenForMemoryPlanning(path))
            geometry = GlmUnifiedMemoryGeometry.Read(file, hostTokenEmbedding: hostEmbedding);
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
        Assert.True(hostCeiling > 0 && deviceCeiling > 0, "No physical room after headroom; fixture unavailable, not passing coverage.");
        var budget = new MemoryBudget([new("host", hostCeiling), new("device", deviceCeiling)]);
        GgmlBasicOps.ClearHostBufferCache();
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        using var native = new GgmlCacheBudgetScope(budget, [["device"], ["host"]], true, ["host"]);
        using var joint = budget.Reserve([new("host", 0), new("device", 0)]);
        native.SetSharedJointEnvelope(joint);
        using var expertOwner = budget.Reserve([new("device", expertForecast)]);
        native.SetSharedExpertEnvelope(expertOwner);
        using var experts = new GgmlHostMoeExpertCacheScope(expertForecast, 512, groups, 1);
        IntPtr model = IntPtr.Zero;
        try
        {
            GgmlGlmNative.RequireUnifiedBudgetSupport();
            if (hostEmbedding)
            {
                Assert.Equal(1, GgmlGlmNative.HostTokenEmbeddingVersion());
                model = GgmlGlmNative.LoadModelWithHostTokenEmbedding(path, 1, 2048, 512, 8,
                    int.MaxValue, "CUDA", 1, true);
            }
            else model = GgmlGlmNative.LoadModel(path, 1, 2048, 512, 8, nCpuMoe: int.MaxValue,
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
            var peak = geometry.GraphPeak(2048, 512, false);
            Assert.True(joint.TryResizeTotal([new("host", peak.Host), new("device", peak.Device)]),
                "The full broad ordinary graph ownership envelope must fit before executing the failure case.");
            action(model, native);
        }
        finally
        {
            if (model != IntPtr.Zero) GgmlGlmNative.Free(model);
            experts.Dispose();
            native.SetSharedExpertEnvelope(null);
            native.SetSharedJointEnvelope(null);
        }
        Assert.Null(native.CallbackError);
        Assert.Equal(0, native.ActiveAllocations);
        expertOwner.Dispose();
        joint.Dispose();
        Assert.All(budget.Snapshot(), pool =>
        {
            Assert.Equal(0, pool.Reserved);
            Assert.Equal(0, pool.Committed);
        });
        Console.WriteLine($"[GLM-HOST-TOKEN-EMBEDDING-CHECKPOINT-OWNER] mode={(hostEmbedding ? 1 : 0)} active_allocations=0 callback_error=0 ledger_reserved=0 ledger_committed=0 host_ceiling={hostCeiling} device_ceiling={deviceCeiling} host_headroom={hostHeadroom} device_headroom={deviceHeadroom}");
    }

    private static NativeEnvScope ConfigureEnvironment()
    {
        var env = new NativeEnvScope();
        foreach (string name in new[] { "TS_GLM_TRACE", "TS_GLM_TRACE_DUMP_DIR", "TS_GLM_NODES_PER_LAYER",
            "TS_GLM_MOE_MMAP", "TS_GLM_TP_SHARD", "TS_GLM_TP_OVERSUBSCRIBE" }) env.Set(name, null);
        foreach (string name in new[] { "TS_HOST_MOE_RAM_CACHE_MB", "TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", "TS_GLM_FLASH_MTP", "TS_GLM_FLASH_MTP_SHAPE_PLAN",
            "TS_GLM_FLASH_MTP_PREFIX_COMMIT", "TS_HOST_MOE_RAM_CACHE_MAPPED", "TS_HOST_MOE_RAM_CACHE_PINNED",
            "TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD", "TS_HOST_MOE_RAM_CACHE_VICTIM", "TS_HOST_MOE_LAYER_PREFETCH",
            "TS_HOST_MOE_DECODE_PREFETCH", "TS_HOST_MOE_DECODE_STAGING_72MB", "TS_HOST_MOE_PREFILL_RING_72MB",
            "TS_HOST_MOE_PREFILL_RING_COALESCE16", "TS_HOST_MOE_PREFILL_DOWN_OVERLAP" }) env.Set(name, "0");
        return env;
    }
}
