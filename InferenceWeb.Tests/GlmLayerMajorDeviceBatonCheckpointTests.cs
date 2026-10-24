// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
public sealed class GlmLayerMajorDeviceBatonCheckpointFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda,Models";
    public GlmLayerMajorDeviceBatonCheckpointFactAttribute()
    {
        Skip = string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_GLM_LAYER_MAJOR_DEVICE_BATON_MODEL"))
            ? "Requires explicit TS_TEST_GLM_LAYER_MAJOR_DEVICE_BATON_MODEL; run alone in a fresh process."
            : TestGates.CudaSkip;
    }
}
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
public sealed class GlmLayerMajorDeviceBatonFaultCheckpointFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda,Models,NativeTestHooks";
    public GlmLayerMajorDeviceBatonFaultCheckpointFactAttribute()
    {
        Skip = string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_GLM_LAYER_MAJOR_DEVICE_BATON_MODEL"))
            ? "Requires explicit TS_TEST_GLM_LAYER_MAJOR_DEVICE_BATON_MODEL."
            : Environment.GetEnvironmentVariable("TS_TEST_GLM_NATIVE_HOOKS") != "1"
                ? "Requires TS_TEST_GLM_NATIVE_HOOKS=1 and original raw-pack fault exports."
                : TestGates.CudaSkip;
    }
}
[CollectionDefinition("GLM layer-major Device baton checkpoint", DisableParallelization = true)]
public sealed class GlmLayerMajorDeviceBatonCheckpointCollection { }

/// <summary>Production scheduler/admission and real model lifetime. Synthetic
/// token IDs are valid vocabulary entries, not the canonical quality prompt.
/// Skipped tests are not coverage. No fabricated native handle, execution
/// envelope, replacement release callback or new native fault ABI is used.</summary>
[Collection("GLM layer-major Device baton checkpoint")]
public sealed class GlmLayerMajorDeviceBatonCheckpointTests
{
    private const long GiB = 1L << 30, Bank1872 = 122683392, Bank521 = 34144256;
    private const float Sentinel = -1234567.25f;
    private const BindingFlags PrivateInstance = BindingFlags.Instance | BindingFlags.NonPublic;
    private static CheckpointOwner? s_retainedCleanupOwner;

    [GlmLayerMajorDeviceBatonCheckpointFact]
    public void Actual117MiBBankCompletesContinuesResetsAndCheckedTeardownReleasesDeclaredPayload()
        => WithActualSession(owner =>
        {
            var model = owner.Model;
            var native = owner.Native;
            var executor = owner.NativeExecutor;
            IntPtr handle = owner.Handle;
            AssertFresh(owner);
            owner.PrepareRequest(Prompt(1872));
            ExecuteAdmittedStep(owner, prefill: true, rows: 1872);
            Assert.Equal(1872, GgmlGlmNative.NPast(handle));
            Assert.Equal(1872, owner.Sequence!.NumComputedTokens);
            long[] completed = GgmlGlmNative.LayerMajorDeviceBatonInfo(handle);
            Assert.Equal(new long[] {1,1,0,0,0,1,Bank1872,44*Bank1872,44*Bank1872,176,176,1}, completed);
            Assert.Equal(0, completed[3]); Assert.Equal(0, completed[4]);
            Assert.True(native.ActiveAllocations > 0); // Graphs/model remain real owners.
            Assert.Null(native.CallbackError);
            ExecuteAdmittedStep(owner, prefill: false, rows: 1);
            Assert.Equal(1873, GgmlGlmNative.NPast(handle));
            Assert.Equal(1873, owner.Sequence.NumComputedTokens);
            Assert.Equal(completed, GgmlGlmNative.LayerMajorDeviceBatonInfo(handle));
            Assert.Equal(handle, ReadPrivate<IntPtr>(executor, "_handle"));
            var ledger = owner.Session.Budget.Snapshot().ToArray();
            var allocations = native.AllocationUsage.ToArray();
            // This ends the request: no sequence work runs after the public
            // reset makes its scheduler-computed count intentionally obsolete.
            lock (model.GpuComputeLock) model.ResetKVCache();
            Assert.Equal(0, GgmlGlmNative.NPast(handle));
            Assert.Equal(completed, GgmlGlmNative.LayerMajorDeviceBatonInfo(handle));
            Assert.Equal(ledger, owner.Session.Budget.Snapshot().ToArray());
            Assert.Equal(allocations, native.AllocationUsage.ToArray());
            owner.ReleaseRequest(recoverCapacity: true);
            AssertRequestReleased(owner);
            var retainedExperts = owner.Experts.ReadUsage();
            var retainedAllocations = native.AllocationUsage.ToArray();
            Assert.True(retainedExperts.AccountedBytes > 0);
            lock (model.GpuComputeLock) model.PrepareUnifiedNativeDispose();
            Assert.Equal(completed, GgmlGlmNative.LayerMajorDeviceBatonInfo(handle));
            // This ordinary profile closes only the independent bank here.
            // Expert caches remain owned until the actual Session.Dispose.
            Assert.Equal(retainedExperts, owner.Experts.ReadUsage());
            Assert.Equal(retainedAllocations, native.AllocationUsage.ToArray());
            // Unrelated graph kind5 tickets may still be live. Only this bank's
            // independent native certificate is zero before model deletion.
            owner.Dispose();
            AssertFullyDisposed(owner, executor, native);
            Console.WriteLine("[GLM-DEVICE-BATON-CHECKPOINT-TEST] profile=1 case=happy host_embedding=1 context=2048 chunk=512 submitted=1872 bank_bytes=122683392 input_calls=176 output_calls=176 input_bytes=5398069248 output_bytes=5398069248 scheduler_admitted=1 production_batch_executor=1 closed_bank_before_n1=1 continuation_head=1873 public_reset_head=0 reset_same_bank_stats_ledger=1 request_credit_released=1 checked_close_before_experts=1 ordinary_experts_retained_until_session_dispose=1 actual_executor_handle_zero=1 active_allocations=0 ledger_reserved=0 ledger_committed=0 token_fixture=deterministic_valid_ids scope=real_checkpoint_finite_logits_normal_declared_payload_teardown_not_canonical_quality_or_performance");
        });

    [GlmLayerMajorDeviceBatonFaultCheckpointFact]
    public void ExistingSecondPackFaultRetainsBankPublishesNothingPoisonsAndCheckedCloseOwnsTeardown()
        => WithActualSession(owner =>
        {
            var model = owner.Model;
            var native = owner.Native;
            var executor = owner.NativeExecutor;
            IntPtr handle = owner.Handle;
            AssertFresh(owner);
            // Obtain the REAL model-owned caller buffer through admitted normal
            // execution; do not inject a private execution context or fake array.
            owner.PrepareRequest(Prompt(9));
            ExecuteAdmittedStep(owner, prefill: true, rows: 9);
            owner.ReleaseRequest(recoverCapacity: true);
            AssertRequestReleased(owner);
            Assert.Equal(0, GgmlGlmNative.NPast(handle));
            float[] caller = ReadPrivate<float[]>(model, "_logitsBuffer");
            Assert.Equal(154880, caller.Length);
            Assert.Equal(new long[] {1,1,0,0,0,0,0,0,0,0,0,1}, GgmlGlmNative.LayerMajorDeviceBatonInfo(handle));
            owner.PrepareRequest(Prompt(521));
            owner.ArmPackFault(1);
            long firedBefore = owner.PackFaultCount();
            Array.Fill(caller, Sentinel);
            lock (model.GpuComputeLock)
            {
                var output = owner.Scheduler!.Schedule();
                var work = Assert.Single(output.ScheduledWork);
                Assert.Same(owner.Sequence, work.Sequence);
                Assert.True(work.IsPrefill); Assert.Equal(521, work.NumScheduledTokens);
                Assert.NotNull(owner.Sequence!.MemoryEnvelope);
                var result = Assert.Single(owner.Executor!.ExecuteStep(output));
                Assert.Same(owner.Sequence, result.Sequence);
                Assert.NotNull(result.Error); Assert.False(result.SlotUnavailable);
                Assert.Equal(0, result.TokensForwarded);
                Assert.Equal(0, owner.Sequence.NumComputedTokens);
                owner.Scheduler.NotifyStepCompleted(output);
            }
            Assert.Equal(firedBefore + 1, owner.PackFaultCount());
            owner.ArmPackFault(-1);
            // Reaching the second raw pack proves first512 returned through its
            // real graph/output-D2D boundary. The native FAILED baton record
            // separately exposes output_calls=1/output_bytes=33554432.
            long[] failed = GgmlGlmNative.LayerMajorDeviceBatonInfo(handle);
            Assert.Equal(new long[] {1,1,1,Bank521,Bank521,0,0,0,0,0,0,1}, failed);
            Assert.Equal(0, GgmlGlmNative.NPast(handle));
            Assert.All(caller, value => Assert.Equal(Sentinel, value));
            Assert.Null(owner.Sequence!.LastLogits);
            // AllocationUsage groups by rank/kind; a graph joint arena can
            // share this group. Info above independently certifies this bank.
            var failedJoint = Assert.Single(native.AllocationUsage.Where(a => a.Rank == 0 && a.Kind == 5));
            Assert.True(failedJoint.Allocations >= 1);
            Assert.True(failedJoint.CommittedBytes >= Bank521);
            Assert.Equal(0, failedJoint.PendingBytes);
            Assert.Null(native.CallbackError);
            var ledger = owner.Session.Budget.Snapshot().ToArray();
            var allocations = native.AllocationUsage.ToArray();
            // These REAL calls are refusal probes: no execution/admission
            // context is fabricated, and the native guards refuse before work.
            Assert.False(GgmlGlmNative.Forward(handle, [65], caller));
            Assert.False(GgmlGlmNative.ResetChecked(handle));
            Assert.Equal(0, GgmlGlmNative.NPast(handle));
            Assert.All(caller, value => Assert.Equal(Sentinel, value));
            Assert.Equal(failed, GgmlGlmNative.LayerMajorDeviceBatonInfo(handle));
            Assert.Equal(ledger, owner.Session.Budget.Snapshot().ToArray());
            Assert.Equal(allocations, native.AllocationUsage.ToArray());
            Assert.Equal(handle, ReadPrivate<IntPtr>(executor, "_handle"));
            // Exercise the production bridge before ANY expert trim/release.
            // It closes the bank first; failed model poison is intentionally
            // terminal even after physical retirement succeeds.
            var retainedExperts = owner.Experts.ReadUsage();
            Assert.True(retainedExperts.AccountedBytes > 0);
            lock (model.GpuComputeLock) model.PrepareUnifiedNativeDispose();
            long[] closed = GgmlGlmNative.LayerMajorDeviceBatonInfo(handle);
            Assert.Equal(new long[] {1,1,1,0,0,0,0,0,0,0,0,1}, closed);
            Assert.Equal(retainedExperts, owner.Experts.ReadUsage());
            // The checked bridge removes exactly one committed bank ticket,
            // retaining every other grouped graph/staging/expert owner.
            var expectedAfterBank = allocations.Select(a => a.Rank == 0 && a.Kind == 5
                ? a with { Allocations = a.Allocations - 1, CommittedBytes = a.CommittedBytes - Bank521 }
                : a).Where(a => a.Allocations != 0).ToArray();
            Assert.Equal(expectedAfterBank, native.AllocationUsage.ToArray());
            Assert.Equal(handle, ReadPrivate<IntPtr>(executor, "_handle"));
            // Terminal shutdown has no optional cache readmission. The actual
            // scheduler Abort/holder-release/NotifyMemoryReleased still run;
            // physical children remain charged until checked model deletion.
            owner.ReleaseRequest(recoverCapacity: false);
            AssertRequestReleased(owner);
            owner.Dispose();
            AssertFullyDisposed(owner, executor, native);
            Console.WriteLine($"[GLM-DEVICE-BATON-CHECKPOINT-TEST] profile=1 case=pack_failure host_embedding=1 context=2048 chunk=512 submitted=521 first_chunk=512 failing_chunk=9 bank_bytes=34144256 fired_before={firedBefore} fired_after={firedBefore+1} scheduler_admitted=1 production_batch_executor=1 real_first_chunk_output_d2d=1 sentinel_exact=1 head_unpublished=1 completed_calls=0 retained_live_bank_bytes=34144256 retained_ticket_bytes=34144256 continuation_refused=1 reset_refused=1 checked_bank_close_before_experts=1 kind5_committed_delta=34144256 kind5_owner_delta=1 ordinary_experts_retained_until_session_dispose=1 terminal_optional_capacity_recovery_skipped=1 request_credit_released=1 actual_executor_handle_zero=1 active_allocations=0 ledger_reserved=0 ledger_committed=0 scope=real_checkpoint_injected_raw_pack_failure_safety_not_vendor_free_error_or_canonical_quality_or_performance");
        });

    private static int[] Prompt(int count) => Enumerable.Range(0,count).Select(i => 65+i%59).ToArray();
    private static UnifiedMemoryModelOptions Options() => new() {
        ContextTokens=0, PrefillChunkTokens=512, AdaptiveGlmPrefillChunkTokens=512, LayerMajorPrefillTokens=8192,
        MaximumRunningRequests=1, RetainedConversationCount=0, PrefixCheckpointCount=0,
        MaximumHostBytes=28*GiB, MaximumDeviceBytes=14*GiB, MaximumHostExpertCacheBytes=0,
        EnableGlmSelectedExperts=true, EnableAdaptiveGlmPrefill=true, EnableExpertCacheCapacityGrowth=true,
        EnableAsyncExpertPrefill=false, EnableGlmFlashMtp=false
    };
    private static void WithActualSession(Action<CheckpointOwner> body)
    {
        string path=Environment.GetEnvironmentVariable("TS_TEST_GLM_LAYER_MAJOR_DEVICE_BATON_MODEL")!;
        Assert.True(File.Exists(path),"Explicit real checkpoint required; a synthetic substitute is not coverage.");
        Assert.Null(s_retainedCleanupOwner);
        using var speculation=new EnvScope();speculation.ClearSpeculationVars();
        using var environment=ConfigureEnvironment();
        CheckpointOwner? owner=null;Exception? original=null;
        try {
            owner=new CheckpointOwner(GlmUnifiedMemoryModelSession.Create(path,null,BackendType.GgmlCuda,Options()));
            body(owner);
        } catch(UnifiedMemoryModelLoadException error) {
            owner=error.UnreleasedSession==null?null:new CheckpointOwner(error.UnreleasedSession);original=error;throw;
        } catch(Exception error) {original=error;throw;}
        finally {
            if(owner!=null)try{owner.Dispose();}
            catch(Exception cleanup){s_retainedCleanupOwner=owner;throw original==null
                ?new AggregateException("Checked cleanup failed; complete checkpoint owner retained.",cleanup)
                :new AggregateException("Checkpoint assertion and cleanup failed; complete owner retained.",original,cleanup);}
        }
    }
    private static void AssertFresh(CheckpointOwner owner)
    {
        Assert.Equal(2048,owner.Session.ContextTokens);Assert.Equal(512,owner.Session.PrefillChunkTokens);
        Assert.Equal(2048,owner.Model.MaxContextLength);Assert.Equal(3,owner.Model.UnifiedExpertGroups);
        Assert.True(owner.Model.UnifiedHostTokenEmbeddingCertified);Assert.False(owner.Model.HasDraftHead);
        Assert.True(owner.Native.IncludesCheckedRelease);
        Assert.True(ReadPrivate<bool>(owner.NativeExecutor,"_requireCheckedFree"));
        Assert.Null(ReadPrivateNullable(owner.NativeExecutor,"_releaseForTests"));
        Assert.False(ReadPrivate<bool>(owner.Session,"_fileSourceViewRequested"));
        Assert.False(ReadPrivate<bool>(owner.Session,"_vmmWeightsRequested"));
        GlmUnifiedHostTokenEmbeddingPlan.ValidateLoadedInfo(GgmlGlmNative.HostTokenEmbeddingInfo(owner.Handle));
        Assert.Equal(new long[] {1,1,0,0,0,0,0,0,0,0,0,1},GgmlGlmNative.LayerMajorDeviceBatonInfo(owner.Handle));
        Assert.Null(owner.Native.CallbackError);
    }
    private static void ExecuteAdmittedStep(CheckpointOwner owner,bool prefill,int rows)
    {
        lock(owner.Model.GpuComputeLock) {
            var output=owner.Scheduler!.Schedule();var work=Assert.Single(output.ScheduledWork);
            Assert.Same(owner.Sequence,work.Sequence);Assert.Equal(prefill,work.IsPrefill);Assert.Equal(rows,work.NumScheduledTokens);
            Assert.Equal(SequenceStatus.Running,owner.Sequence!.Status);Assert.NotNull(owner.Sequence.MemoryEnvelope);
            var result=Assert.Single(owner.Executor!.ExecuteStep(output));
            Assert.Same(owner.Sequence,result.Sequence);Assert.Null(result.Error);Assert.False(result.SlotUnavailable);
            Assert.Equal(prefill,result.IsPrefill);Assert.Equal(rows,result.TokensForwarded);
            AssertFinite(owner.Sequence.LastLogits!);
            if(prefill) {
                Assert.True(owner.Sequence.NumComputedTokens>=owner.Sequence.PromptTokens.Count);
                Assert.NotNull(owner.Scheduler.Config.MemoryAdmission!.AfterPrefill);
                owner.Scheduler.Config.MemoryAdmission.AfterPrefill(owner.Sequence);
            }
            owner.Scheduler.NotifyStepCompleted(output);
        }
    }
    private static void AssertRequestReleased(CheckpointOwner owner)
    {
        Assert.Null(owner.Sequence!.MemoryEnvelope);Assert.Equal(0,owner.Scheduler!.RunningCount);
        Assert.Equal(owner.Pool!.NumBlocks,owner.Pool.NumFreeBlocks);Assert.Equal(0,owner.Pool.Storage.AllocatedBytes);
    }
    private static void AssertFullyDisposed(CheckpointOwner owner,NativeGlmExecutor executor,GgmlCacheBudgetScope native)
    {
        Assert.Equal(IntPtr.Zero,ReadPrivate<IntPtr>(executor,"_handle"));Assert.Null(owner.Session.Model);
        Assert.Null(native.CallbackError);Assert.Equal(0,native.ActiveAllocations);Assert.Empty(native.AllocationUsage);
        Assert.All(owner.Session.Budget.Snapshot(),p=>{Assert.Equal(0,p.Reserved);Assert.Equal(0,p.Committed);});
        owner.Session.Dispose();
    }
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int ArmPackFailure(IntPtr model,int countdown);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate long CountPackFailures(IntPtr model);
    private sealed class PackHooks : IDisposable
    {
        private IntPtr _module;
        private readonly ArmPackFailure _arm;
        private readonly CountPackFailures _count;
        internal PackHooks() {
            _module=NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            try {
                _arm=Marshal.GetDelegateForFunctionPointer<ArmPackFailure>(NativeLibrary.GetExport(_module,"TSGgml_GlmTestFailHostTokenEmbeddingPack"));
                _count=Marshal.GetDelegateForFunctionPointer<CountPackFailures>(NativeLibrary.GetExport(_module,"TSGgml_GlmTestHostTokenEmbeddingPackFailureCount"));
            } catch {NativeLibrary.Free(_module);_module=IntPtr.Zero;throw;}
        }
        internal void Arm(IntPtr model,int countdown)=>Assert.Equal(1,_arm(model,countdown));
        internal long Count(IntPtr model)=>_count(model);
        public void Dispose(){if(_module==IntPtr.Zero)return;NativeLibrary.Free(_module);_module=IntPtr.Zero;}
    }
    private sealed class CheckpointOwner(IUnifiedMemoryModelSession session) : IDisposable
    {
        internal IUnifiedMemoryModelSession Session {get;}=session;
        internal GlmDsaModel Model=>Assert.IsType<GlmDsaModel>(Session.Model);
        internal GgmlCacheBudgetScope Native=>ReadPrivate<GgmlCacheBudgetScope>(Session,"_native");
        internal GgmlHostMoeExpertCacheScope Experts=>ReadPrivate<GgmlHostMoeExpertCacheScope>(Session,"_experts");
        internal NativeGlmExecutor NativeExecutor=>ReadPrivate<NativeGlmExecutor>(Model,"_exec");
        internal IntPtr Handle=>ReadPrivate<IntPtr>(NativeExecutor,"_handle");
        internal BlockPool? Pool {get;private set;}
        internal ContinuousBatchScheduler? Scheduler {get;private set;}
        internal BatchExecutor? Executor {get;private set;}
        internal SequenceState? Sequence {get;private set;}
        private bool _requestReleased=true,_disposed;
        private int _requests;
        private PackHooks? _hooks;
        internal void ArmPackFault(int countdown)=>(_hooks??=new PackHooks()).Arm(Handle,countdown);
        internal long PackFaultCount()=>(_hooks??=new PackHooks()).Count(Handle);
        internal void PrepareRequest(int[] prompt) {
            Assert.True(_requestReleased);Pool?.Storage.Dispose();
            var model=Model;var config=Session.ConfigureScheduler(new SchedulerConfig{EnablePrefixCaching=false});
            Assert.NotNull(config.MemoryAdmission);Assert.Same(Session.Budget,config.MemoryAdmission.Budget);
            Assert.True(config.MemoryAdmission.UsePerSequenceHolders);Assert.False(config.EnablePrefixCaching);
            Assert.Equal(1,config.MaxNumRunningSequences);
            Pool=new BlockPool(config.NumBlocks,config.BlockSize,model.ComputeKVBlockByteSize(config.BlockSize));
            Scheduler=new ContinuousBatchScheduler(config,Pool,NullLogger.Instance,supportsCrossSequenceKvReuse:false);
            Executor=new BatchExecutor(model,Pool,Scheduler,NullLogger.Instance);
            Sequence=new SequenceState("actual-device-baton-"+(++_requests),prompt,maxNewTokens:4,blockSize:config.BlockSize,samplingConfig:null);
            Scheduler.Submit(Sequence);_requestReleased=false;
        }
        internal void ReleaseRequest(bool recoverCapacity) {
            if(_requestReleased||Sequence==null||Scheduler==null)return;
            var model=Model;
            lock(model.GpuComputeLock) {
                Scheduler.Abort(Sequence.RequestId);
                Executor?.DiscardReleasedFusedCacheBookkeeping(Sequence.RequestId);
                Executor?.ReleaseLinearSpeculativeContext(Sequence.RequestId);
                model.OnSequenceReleased(Sequence.RequestId);Executor?.RadixCache?.Drain();
                Scheduler.NotifyMemoryReleased(Sequence.RequestId);_requestReleased=true;
                if(recoverCapacity)Scheduler.Config.MemoryAdmission?.AfterSequenceMemoryRelease?.Invoke(Scheduler.GetRunningSequencesSnapshot());
            }
        }
        public void Dispose() {
            if(_disposed)return;
            // Model shutdown is terminal. The real production disposal bridge
            // is first so any retained bank drains before expert/slot mutation.
            // A refusal leaves every scheduler/envelope/callback/module alive.
            if(Session.Model is GlmDsaModel model) {
                lock(model.GpuComputeLock) {
                    _hooks?.Arm(Handle,-1);
                    model.PrepareUnifiedNativeDispose();
                }
            }
            ReleaseRequest(recoverCapacity:false);Pool?.Storage.Dispose();Session.Dispose();
            _hooks?.Dispose();_hooks=null;_disposed=true;
        }
    }
    private static T ReadPrivate<T>(object owner,string name)
        =>Assert.IsType<T>(owner.GetType().GetField(name,PrivateInstance)!.GetValue(owner));
    private static object? ReadPrivateNullable(object owner,string name)
        =>owner.GetType().GetField(name,PrivateInstance)!.GetValue(owner);
    private static void AssertFinite(float[] logits) {
        Assert.Equal(154880,logits.Length);Assert.All(logits,x=>Assert.True(float.IsFinite(x)));Assert.Contains(logits,x=>x!=0);
    }
    private static NativeEnvScope ConfigureEnvironment()
    {
        var env=new NativeEnvScope();
        foreach(string name in new[]{"TS_GLM_TRACE","TS_GLM_TRACE_DUMP_DIR","TS_GLM_NODES_PER_LAYER","TS_GLM_MOE_MMAP","TS_GLM_TP_SHARD","TS_GLM_TP_OVERSUBSCRIBE"})env.Set(name,null);
        foreach(string name in new[]{"TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB","TS_GLM_FLASH_MTP","TS_GLM_FLASH_MTP_SHAPE_PLAN","TS_GLM_FLASH_MTP_PREFIX_COMMIT","TS_GLM_DECODE_GRAPH_RESERVE","TS_GLM_OP_OFFLOAD","TS_GGML_STRICT_RECURRENT_PROJECTIONS","TS_GLM_SELECTED_EXPERT_TIMING","TS_GLM_GRAPH_MEMORY_DIAGNOSTICS","TS_DSV4_PERF","TS_HOST_MOE_RAM_CACHE_MB","TS_HOST_MOE_RAM_CACHE_MAPPED","TS_HOST_MOE_RAM_CACHE_PINNED","TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD","TS_HOST_MOE_RAM_CACHE_VICTIM","TS_HOST_MOE_RAM_CACHE_FREQUENCY","TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY","TS_HOST_MOE_LAYER_PREFETCH","TS_HOST_MOE_DECODE_PREFETCH","TS_HOST_MOE_PREFILL_RING_COALESCE16","TS_HOST_MOE_PREFILL_DOWN_OVERLAP","TS_HOST_MOE_PHASE_COLD_GROWTH","TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER","TS_HOST_MOE_EXPERT_CACHE_PREFETCH","TS_HOST_MOE_EXPERT_CACHE_LFU","TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS","TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS","TS_HOST_MOE_FILE_READ_WORKERS","TS_HOST_MOE_FILE_RANDOM_ACCESS","TS_HOST_MOE_FILE_READ_SORT","TS_HOST_MOE_ROUTE_TRACE","TS_HOST_MOE_ROUTE_LOOKUP_INDEX","TS_HOST_MOE_FILE_SOURCE_VIEW","TS_HOST_MOE_FILE_SOURCE_VIEW_PREFETCH","TS_HOST_MOE_VMM_WEIGHTS"})env.Set(name,"0");
        env.Set("TS_GLM_NATIVE","1");env.Set("TS_GLM_HOST_TOKEN_EMBEDDING","1");env.Set("TS_GLM_LAYER_MAJOR_DEVICE_BATON","1");
        env.Set("TS_HOST_MOE_DECODE_REBALANCE","1");env.Set("TS_HOST_MOE_DECODE_STAGING_72MB","1");env.Set("TS_HOST_MOE_PREFILL_RING_72MB","1");
        env.Set("MAX_CONTEXT","2048");env.Set("TS_SCHED_PREFILL_CHUNK","512");env.Set("TS_UNIFIED_LAYER_MAJOR_TOKENS","8192");
        return env;
    }
}
