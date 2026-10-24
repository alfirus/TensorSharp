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
public sealed class GlmCausalRouteNeighborCheckpointFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda,Models";
    public GlmCausalRouteNeighborCheckpointFactAttribute()
    {
        Skip = string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_GLM_CAUSAL_ROUTE_NEIGHBOR_MODEL"))
            ? "Requires explicit TS_TEST_GLM_CAUSAL_ROUTE_NEIGHBOR_MODEL; run alone in a fresh process."
            : TestGates.CudaSkip;
    }
}
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
public sealed class GlmCausalRouteNeighborFaultCheckpointTheoryAttribute : TheoryAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda,Models,NativeTestHooks";
    public GlmCausalRouteNeighborFaultCheckpointTheoryAttribute()
    {
        Skip = string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_GLM_CAUSAL_ROUTE_NEIGHBOR_MODEL"))
            ? "Requires explicit TS_TEST_GLM_CAUSAL_ROUTE_NEIGHBOR_MODEL."
            : Environment.GetEnvironmentVariable("TS_TEST_GLM_NATIVE_HOOKS") != "1"
                ? "Requires TS_TEST_GLM_NATIVE_HOOKS=1 and the actual causal failure exports."
                : TestGates.CudaSkip;
    }
}
[CollectionDefinition("GLM causal route-neighbor checkpoint", DisableParallelization = true)]
public sealed class GlmCausalRouteNeighborCheckpointCollection { }

/// <summary>Real scheduler admission, completed metadata activity and checked
/// lifecycle. Valid synthetic token IDs are not a canonical quality prompt.
/// No native handle, execution context, release callback or future routes are
/// fabricated. An unavailable checkpoint/device/export is not passing coverage.</summary>
[Collection("GLM causal route-neighbor checkpoint")]
public sealed class GlmCausalRouteNeighborCheckpointTests
{
    private const long GiB = 1L << 30;
    private const float Sentinel = -1234567.25f;
    private const BindingFlags PrivateInstance = BindingFlags.Instance | BindingFlags.NonPublic;
    private static CheckpointOwner? s_retainedCleanupOwner;

    [GlmCausalRouteNeighborCheckpointFact]
    public void ChargedHistorySkipsWidePrefillLearnsCompletedDecodeResetsAndClosesBeforeModel()
        => WithActualSession(owner =>
        {
            var model=owner.Model;var native=owner.Native;var executor=owner.NativeExecutor;
            IntPtr handle=owner.Handle;
            long bytes=AssertFresh(owner);
            long[] initial=model.UnifiedCausalRouteNeighborInfo;
            // Two real unchanged PP graph chunks, followed by enough completed
            // N1 visits to observe an eligible past-only suffix in this model.
            owner.PrepareRequest(Prompt(521));
            ExecuteAdmittedStep(owner,prefill:true,rows:521);
            Assert.Equal(521,GgmlGlmNative.NPast(handle));
            long[] prefill=model.UnifiedCausalRouteNeighborInfo;
            AssertActive(prefill,bytes);
            Assert.Equal(initial[10],prefill[10]);Assert.Equal(initial[11],prefill[11]);
            Assert.Equal(initial[14],prefill[14]);Assert.Equal(initial[15],prefill[15]);
            Assert.True(prefill[12]>initial[12]);
            for(int step=0;step<32;++step) {
                long[] before=model.UnifiedCausalRouteNeighborInfo;
                ExecuteAdmittedStep(owner,prefill:false,rows:1);
                long[] after=model.UnifiedCausalRouteNeighborInfo;
                AssertActive(after,bytes);
                Assert.Equal(before[10]+1,after[10]);
                Assert.Equal(before[11]+42,after[11]);
                Assert.Equal(before[14]+before[15]+42,after[14]+after[15]);
                Assert.Equal(522+step,GgmlGlmNative.NPast(handle));
                Assert.Equal(522+step,owner.Sequence!.NumComputedTokens);
            }
            long[] learned=model.UnifiedCausalRouteNeighborInfo;
            Assert.Equal(prefill[10]+32,learned[10]);Assert.Equal(prefill[11]+1344,learned[11]);
            Assert.True(learned[14]>prefill[14],"Actual completed N1 calls must include a real past-route match.");
            Assert.Equal(handle,ReadPrivate<IntPtr>(executor,"_handle"));
            lock(model.GpuComputeLock)model.ResetKVCache();
            Assert.Equal(0,GgmlGlmNative.NPast(handle));
            long[] reset=model.UnifiedCausalRouteNeighborInfo;
            AssertActive(reset,bytes);Assert.True(reset[12]>learned[12]);
            foreach(int index in new[]{10,11,14,15})Assert.Equal(learned[index],reset[index]);
            owner.ReleaseRequest(recoverCapacity:true);AssertRequestReleased(owner);
            CloseSlabBeforeOrdinaryCache(owner,bytes);
            owner.Dispose();AssertFullyDisposed(owner,executor,native);
            Console.WriteLine($"[GLM-CAUSAL-ROUTE-NEIGHBOR-CHECKPOINT-TEST] profile=1 case=happy host_embedding=1 context=2048 chunk=512 submitted=521 n1_tokens=32 committed_rows=1344 matched_queries={learned[14]-prefill[14]} declared_host_bytes={bytes} extra_device_bytes=0 scheduler_admitted=1 production_batch_executor=1 pp_no_learning=1 public_reset_head=0 reset_same_slab_ticket=1 cumulative_counts_retained=1 checked_close_before_experts=1 ordinary_experts_retained_until_session_dispose=1 actual_executor_handle_zero=1 active_allocations=0 ledger_reserved=0 ledger_committed=0 scope=real_checkpoint_completed_metadata_and_declared_payload_teardown_not_canonical_quality_or_saved_io_or_performance");
        });

    [GlmCausalRouteNeighborFaultCheckpointTheory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void ActualQueryUploadOrFinalDrainFailurePublishesNothingAndCheckedResetRecovers(int stage)
        => WithActualSession(owner =>
        {
            var model=owner.Model;var native=owner.Native;var executor=owner.NativeExecutor;
            IntPtr handle=owner.Handle;long bytes=AssertFresh(owner);
            owner.PrepareRequest(Prompt(9));
            ExecuteAdmittedStep(owner,prefill:true,rows:9);
            Assert.Equal(9,GgmlGlmNative.NPast(handle));
            float[] caller=ReadPrivate<float[]>(model,"_logitsBuffer");
            Assert.Equal(154880,caller.Length);
            if(stage==2) {
                // A real checked maintenance boundary retains the metadata
                // owner but clears GPU weights. The next selected upload must
                // actually occur before the one-shot upload hook may fire.
                lock(model.GpuComputeLock)owner.Experts.Trim(0);
                Assert.Equal(0,owner.Experts.ReadUsage().AccountedBytes);
                AssertActive(model.UnifiedCausalRouteNeighborInfo,bytes);
            }
            long[] beforeFault=model.UnifiedCausalRouteNeighborInfo;
            int outputCount=owner.Sequence!.OutputTokens.Count;
            float[] previousLogits=(float[])owner.Sequence.LastLogits!.Clone();
            owner.ArmCausalFault(stage);long fired=owner.CausalFaultCount();
            Array.Fill(caller,Sentinel);
            lock(model.GpuComputeLock) {
                var output=owner.Scheduler!.Schedule();var work=Assert.Single(output.ScheduledWork);
                Assert.Same(owner.Sequence,work.Sequence);Assert.False(work.IsPrefill);
                Assert.Equal(1,work.NumScheduledTokens);Assert.NotNull(owner.Sequence.MemoryEnvelope);
                var result=Assert.Single(owner.Executor!.ExecuteStep(output));
                Assert.Same(owner.Sequence,result.Sequence);Assert.NotNull(result.Error);
                Assert.False(result.SlotUnavailable);Assert.Equal(0,result.TokensForwarded);
                Assert.Equal(9,owner.Sequence.NumComputedTokens);
                owner.Scheduler.NotifyStepCompleted(output);
            }
            Assert.Equal(fired+1,owner.CausalFaultCount());owner.ArmCausalFault(0);
            long[] failed=model.UnifiedCausalRouteNeighborInfo;
            AssertOwnership(failed,bytes);Assert.Equal(1,failed[5]);Assert.Equal(0,failed[9]);
            Assert.Equal(beforeFault[10],failed[10]);Assert.Equal(beforeFault[11],failed[11]);
            Assert.Equal(9,GgmlGlmNative.NPast(handle));
            // Production BatchExecutor records its sampled INPUT token before
            // Forward. A failed Forward leaves that bookkeeping entry while
            // returning Error/TokensForwarded0, not a successful emitted token.
            Assert.Equal(outputCount+1,owner.Sequence.OutputTokens.Count);
            Assert.Equal(previousLogits,owner.Sequence.LastLogits);
            Assert.All(caller,value=>Assert.Equal(Sentinel,value));
            Assert.Equal(handle,ReadPrivate<IntPtr>(executor,"_handle"));
            Assert.Null(native.CallbackError);
            var allocations=native.AllocationUsage.ToArray();var ledger=owner.Session.Budget.Snapshot().ToArray();
            Assert.False(GgmlGlmNative.Forward(handle,[65],caller));
            Assert.Equal(9,GgmlGlmNative.NPast(handle));
            Assert.All(caller,value=>Assert.Equal(Sentinel,value));
            Assert.Equal(failed,model.UnifiedCausalRouteNeighborInfo);
            Assert.Equal(allocations,native.AllocationUsage.ToArray());
            Assert.Equal(ledger,owner.Session.Budget.Snapshot().ToArray());
            // This controller explicitly supports checked reset recovery.
            // A failed ordinary continuation cannot resume the poisoned slot.
            lock(model.GpuComputeLock)model.ResetKVCache();
            long[] recovered=model.UnifiedCausalRouteNeighborInfo;
            AssertActive(recovered,bytes);Assert.True(recovered[12]>failed[12]);
            foreach(int index in new[]{10,11,14,15})Assert.Equal(failed[index],recovered[index]);
            Assert.Equal(0,GgmlGlmNative.NPast(handle));
            owner.ReleaseRequest(recoverCapacity:false);AssertRequestReleased(owner);
            owner.PrepareRequest(Prompt(9));
            ExecuteAdmittedStep(owner,prefill:true,rows:9);
            ExecuteAdmittedStep(owner,prefill:false,rows:1);
            Assert.Equal(10,GgmlGlmNative.NPast(handle));
            AssertActive(model.UnifiedCausalRouteNeighborInfo,bytes);
            owner.ReleaseRequest(recoverCapacity:true);AssertRequestReleased(owner);
            CloseSlabBeforeOrdinaryCache(owner,bytes);
            owner.Dispose();AssertFullyDisposed(owner,executor,native);
            Console.WriteLine($"[GLM-CAUSAL-ROUTE-NEIGHBOR-CHECKPOINT-TEST] profile=1 case=fault stage={stage} fired_before={fired} fired_after={fired+1} actual_miss_upload_required={(stage==2?1:0)} declared_host_bytes={bytes} pending_after_abort=0 sentinel_exact=1 last_logits_unchanged=1 pre_forward_sample_retained=1 head_unpublished=1 ordinary_continuation_refused=1 actual_checked_reset_recovered=1 reset_same_slab_ticket=1 recovered_head=10 checked_close_before_experts=1 actual_executor_handle_zero=1 active_allocations=0 ledger_reserved=0 ledger_committed=0 scope=real_checkpoint_injected_policy_boundary_safety_not_vendor_error_or_canonical_quality_or_saved_io_or_performance");
        });

    private static void AssertOwnership(long[] info,long bytes)
    {
        Assert.Equal(16,info.Length);Assert.Equal(1,info[0]);Assert.Equal(1,info[1]);
        Assert.Equal(1,info[2]);Assert.Equal(0,info[3]);Assert.Equal(0,info[4]);
        Assert.Equal(bytes,info[6]);Assert.Equal(bytes,info[7]);Assert.Equal(bytes,info[8]);
    }
    private static void AssertActive(long[] info,long bytes)
    {
        AssertOwnership(info,bytes);Assert.Equal(0,info[5]);Assert.Equal(0,info[9]);
    }
    private static long AssertFresh(CheckpointOwner owner)
    {
        Assert.Equal(2048,owner.Session.ContextTokens);Assert.Equal(512,owner.Session.PrefillChunkTokens);
        Assert.Equal(2048,owner.Model.MaxContextLength);Assert.Equal(3,owner.Model.UnifiedExpertGroups);
        Assert.True(owner.Model.UnifiedHostTokenEmbeddingCertified);Assert.False(owner.Model.HasDraftHead);
        Assert.True(owner.Native.IncludesCheckedRelease);
        Assert.True(ReadPrivate<bool>(owner.NativeExecutor,"_requireCheckedFree"));
        Assert.Null(ReadPrivateNullable(owner.NativeExecutor,"_releaseForTests"));
        Assert.True(ReadPrivate<bool>(owner.Session,"_causalRouteNeighborRequested"));
        Assert.False(ReadPrivate<bool>(owner.Session,"_fileSourceViewRequested"));
        Assert.False(ReadPrivate<bool>(owner.Session,"_vmmWeightsRequested"));
        GlmUnifiedHostTokenEmbeddingPlan.ValidateLoadedInfo(GgmlGlmNative.HostTokenEmbeddingInfo(owner.Handle));
        long bytes=GlmUnifiedCausalRouteNeighborPlan.ValidateHostPlan(GgmlGlmNative.CausalRouteNeighborHostPlan());
        long[] info=owner.Model.UnifiedCausalRouteNeighborInfo;
        AssertActive(info,bytes);
        foreach(int index in new[]{10,11,14,15})Assert.Equal(0,info[index]);
        Assert.Null(owner.Native.CallbackError);
        return bytes;
    }
    private static void CloseSlabBeforeOrdinaryCache(CheckpointOwner owner,long bytes)
    {
        var native=owner.Native;var retainedExperts=owner.Experts.ReadUsage();
        var before=native.AllocationUsage.ToArray();long[] info=owner.Model.UnifiedCausalRouteNeighborInfo;
        AssertActive(info,bytes);Assert.True(retainedExperts.AccountedBytes>0);
        // AllocationUsage is grouped by rank/kind. Other staging/model owners
        // may share kind3; only this exact slab's checked ticket is removed.
        var host=Assert.Single(before.Where(a=>a.Rank==0&&a.Kind==3));
        Assert.True(host.Allocations>=1);Assert.True(host.CommittedBytes>=bytes);
        lock(owner.Model.GpuComputeLock)owner.Model.PrepareUnifiedNativeDispose();
        long[] closed=owner.Model.UnifiedCausalRouteNeighborInfo;
        GlmUnifiedCausalRouteNeighborPlan.ValidateClosedInfo(closed,bytes);
        Assert.Equal(0,closed[5]);
        foreach(int index in new[]{10,11,12,13,14,15})Assert.Equal(info[index],closed[index]);
        Assert.Equal(retainedExperts,owner.Experts.ReadUsage());
        var expected=before.Select(a=>a.Rank==0&&a.Kind==3
            ? a with {Allocations=a.Allocations-1,CommittedBytes=a.CommittedBytes-bytes}:a)
            .Where(a=>a.Allocations!=0).ToArray();
        Assert.Equal(expected,native.AllocationUsage.ToArray());
        Assert.Null(native.CallbackError);
    }
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
        string path=Environment.GetEnvironmentVariable("TS_TEST_GLM_CAUSAL_ROUTE_NEIGHBOR_MODEL")!;
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
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int ArmCausalFailure(IntPtr model,int stage);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate long CountCausalFailures(IntPtr model);
    private sealed class CausalHooks : IDisposable
    {
        private IntPtr _module;
        private readonly ArmCausalFailure _arm;
        private readonly CountCausalFailures _count;
        internal CausalHooks() {
            _module=NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            try {
                _arm=Marshal.GetDelegateForFunctionPointer<ArmCausalFailure>(NativeLibrary.GetExport(_module,"TSGgml_GlmTestFailCausalRouteNeighbor"));
                _count=Marshal.GetDelegateForFunctionPointer<CountCausalFailures>(NativeLibrary.GetExport(_module,"TSGgml_GlmTestCausalRouteNeighborFailureCount"));
            } catch {NativeLibrary.Free(_module);_module=IntPtr.Zero;throw;}
        }
        internal void Arm(IntPtr model,int stage)=>Assert.Equal(1,_arm(model,stage));
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
        private CausalHooks? _hooks;
        internal void ArmCausalFault(int stage)=>(_hooks??=new CausalHooks()).Arm(Handle,stage);
        internal long CausalFaultCount()=>(_hooks??=new CausalHooks()).Count(Handle);
        internal void PrepareRequest(int[] prompt) {
            Assert.True(_requestReleased);Pool?.Storage.Dispose();
            var model=Model;var config=Session.ConfigureScheduler(new SchedulerConfig{EnablePrefixCaching=false});
            Assert.NotNull(config.MemoryAdmission);Assert.Same(Session.Budget,config.MemoryAdmission.Budget);
            Assert.True(config.MemoryAdmission.UsePerSequenceHolders);Assert.False(config.EnablePrefixCaching);
            Assert.Equal(1,config.MaxNumRunningSequences);
            Pool=new BlockPool(config.NumBlocks,config.BlockSize,model.ComputeKVBlockByteSize(config.BlockSize));
            Scheduler=new ContinuousBatchScheduler(config,Pool,NullLogger.Instance,supportsCrossSequenceKvReuse:false);
            Executor=new BatchExecutor(model,Pool,Scheduler,NullLogger.Instance);
            Sequence=new SequenceState("actual-causal-route-neighbor-"+(++_requests),prompt,maxNewTokens:64,blockSize:config.BlockSize,samplingConfig:null);
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
            // is first so any retained metadata retires before expert/slot mutation.
            // A refusal leaves every scheduler/envelope/callback/module alive.
            if(Session.Model is GlmDsaModel model) {
                lock(model.GpuComputeLock) {
                    _hooks?.Arm(Handle,0);
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
        foreach(string name in new[]{"TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB","TS_GLM_FLASH_MTP","TS_GLM_FLASH_MTP_SHAPE_PLAN","TS_GLM_FLASH_MTP_PREFIX_COMMIT","TS_GLM_DECODE_GRAPH_RESERVE","TS_GLM_OP_OFFLOAD","TS_GGML_STRICT_RECURRENT_PROJECTIONS","TS_GLM_SELECTED_EXPERT_TIMING","TS_GLM_GRAPH_MEMORY_DIAGNOSTICS","TS_DSV4_PERF","TS_HOST_MOE_RAM_CACHE_MB","TS_HOST_MOE_RAM_CACHE_MAPPED","TS_HOST_MOE_RAM_CACHE_PINNED","TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD","TS_HOST_MOE_RAM_CACHE_VICTIM","TS_HOST_MOE_RAM_CACHE_FREQUENCY","TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY","TS_HOST_MOE_LAYER_PREFETCH","TS_HOST_MOE_DECODE_PREFETCH","TS_HOST_MOE_PREFILL_RING_COALESCE16","TS_HOST_MOE_PREFILL_DOWN_OVERLAP","TS_HOST_MOE_PHASE_COLD_GROWTH","TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER","TS_HOST_MOE_EXPERT_CACHE_PREFETCH","TS_HOST_MOE_EXPERT_CACHE_LFU","TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS","TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS","TS_HOST_MOE_FILE_READ_WORKERS","TS_HOST_MOE_FILE_RANDOM_ACCESS","TS_HOST_MOE_FILE_READ_SORT","TS_HOST_MOE_ROUTE_TRACE","TS_HOST_MOE_FILE_SOURCE_VIEW","TS_HOST_MOE_FILE_SOURCE_VIEW_PREFETCH","TS_HOST_MOE_VMM_WEIGHTS"})env.Set(name,"0");
        env.Set("TS_GLM_NATIVE","1");env.Set("TS_GLM_HOST_TOKEN_EMBEDDING","1");env.Set("TS_GLM_LAYER_MAJOR_DEVICE_BATON","0");env.Set("TS_HOST_MOE_CAUSAL_ROUTE_NEIGHBOR","1");
        env.Set("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS","1");env.Set("TS_HOST_MOE_EXPERT_CACHE_TIMING","1");
        env.Set("TS_GLM_FLASH_MTP_FORWARD_TIMING","1");
        env.Set("TS_HOST_MOE_DECODE_REBALANCE","1");env.Set("TS_HOST_MOE_DECODE_STAGING_72MB","1");env.Set("TS_HOST_MOE_PREFILL_RING_72MB","1");
        env.Set("TS_HOST_MOE_EXPERT_CACHE_RING_READS","2");
        env.Set("MAX_CONTEXT","2048");env.Set("TS_SCHED_PREFILL_CHUNK","512");env.Set("TS_UNIFIED_LAYER_MAJOR_TOKENS","8192");
        return env;
    }
}
