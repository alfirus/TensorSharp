// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
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
public sealed class GlmVmmCheckedTeardownCheckpointFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda,Models";
    public GlmVmmCheckedTeardownCheckpointFactAttribute()
    {
        Skip = string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_GLM_VMM_CHECKED_TEARDOWN_MODEL"))
            ? "Requires explicit TS_TEST_GLM_VMM_CHECKED_TEARDOWN_MODEL; run this full-checkpoint lifecycle test alone in a fresh process."
            : TestGates.CudaSkip;
    }
}

[CollectionDefinition("GLM VMM checked teardown checkpoint", DisableParallelization = true)]
public sealed class GlmVmmCheckedTeardownCheckpointCollection { }

/// <summary>Actual nonnull checked-free refusal and normal ordered teardown.
/// Run only this explicit checkpoint test in a fresh process: native modes are
/// process-latched. Skipped/unavailable is not coverage. No test-only native
/// profile, fabricated model handle or substitute release callback is used.</summary>
[Collection("GLM VMM checked teardown checkpoint")]
public sealed class GlmVmmCheckedTeardownCheckpointTests
{
    private const long GiB = 1L << 30;
    private const BindingFlags PrivateInstance = BindingFlags.Instance | BindingFlags.NonPublic;
    // A failed checked cleanup must keep the entire real owner, its callbacks,
    // model/source/backend and tickets alive. One test => one bounded slot.
    private static CheckpointOwner? s_retainedCleanupOwner;

    [GlmVmmCheckedTeardownCheckpointFact]
    public void ActualLiveVmmRefusesModelFreeRetainsExecutionThenCheckedCloseReleasesDeclaredPayload()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_GLM_VMM_CHECKED_TEARDOWN_MODEL")!;
        Assert.True(File.Exists(path), "The explicit real checkpoint must exist; no synthetic fallback is coverage.");
        Assert.Null(s_retainedCleanupOwner);
        using var speculation = new EnvScope();
        speculation.ClearSpeculationVars();
        using var environment = ConfigureEnvironment();
        var options = new UnifiedMemoryModelOptions
        {
            ContextTokens = 2048,
            PrefillChunkTokens = 512,
            AdaptiveGlmPrefillChunkTokens = 512,
            LayerMajorPrefillTokens = 8192,
            MaximumRunningRequests = 1,
            RetainedConversationCount = 0,
            PrefixCheckpointCount = 0,
            MaximumHostBytes = 28 * GiB,
            MaximumDeviceBytes = 14 * GiB,
            MaximumHostExpertCacheBytes = 0,
            EnableGlmSelectedExperts = true,
            EnableAdaptiveGlmPrefill = true,
            EnableExpertCacheCapacityGrowth = true,
            EnableAsyncExpertPrefill = false,
            EnableGlmFlashMtp = false
        };
        CheckpointOwner? cleanupOwner = null;
        Exception? originalFailure = null;
        try
        {
            // This is the production loader/planner: it owns physical headroom,
            // exact source registration and the immutable real VMM profile.
            var session = GlmUnifiedMemoryModelSession.Create(path, null, BackendType.GgmlCuda, options);
            cleanupOwner = new CheckpointOwner(session);
            var model = Assert.IsType<GlmDsaModel>(session.Model);
            var experts = ReadPrivate<GgmlHostMoeExpertCacheScope>(session, "_experts");
            var native = ReadPrivate<GgmlCacheBudgetScope>(session, "_native");
            var executor = ReadPrivate<NativeGlmExecutor>(model, "_exec");
            Assert.Null(ReadPrivateNullable(executor, "_releaseForTests"));
            Assert.True(ReadPrivate<bool>(executor, "_requireCheckedFree"));
            IntPtr handle = ReadPrivate<IntPtr>(executor, "_handle");
            Assert.NotEqual(IntPtr.Zero, handle);
            Assert.True(native.IncludesCheckedRelease);
            Assert.Equal(2048, session.ContextTokens);
            Assert.Equal(512, session.PrefillChunkTokens);
            Assert.Equal(2048, model.MaxContextLength);
            Assert.Equal(3, model.UnifiedExpertGroups);
            Assert.True(model.UnifiedHostTokenEmbeddingCertified);
            GlmUnifiedHostTokenEmbeddingPlan.ValidateLoadedInfo(GgmlGlmNative.HostTokenEmbeddingInfo(handle));
            Assert.False(GgmlGlmNative.HasDraftHead(handle));
            Assert.Equal(154880, GgmlGlmNative.VocabSize(handle));
            var empty = experts.ReadVmmUsage();
            Assert.Equal(1, empty.Configured);
            Assert.Equal(0, empty.Closed);
            Assert.Equal(0, empty.LiveTables);
            Assert.Equal(0, empty.OwnedDeviceBytes);
            Assert.Equal(2, empty.HostTicketOwners);
            Assert.True(empty.DeclaredHostBytes > 0);

            // Real Schedule() performs physical admission and creates the
            // request envelope. BatchExecutor enters the production serial
            // owner and binds the actual per-sequence holder around Forward.
            // No fabricated envelope/private execution context is assigned.
            int[] prompt = Enumerable.Range(0, 9).Select(i => 65 + i).ToArray();
            var config = session.ConfigureScheduler(new SchedulerConfig { EnablePrefixCaching = false });
            cleanupOwner.PrepareRequest(config, model, prompt);
            var sequence = cleanupOwner.Sequence!;
            Assert.NotNull(config.MemoryAdmission);
            Assert.Same(session.Budget, config.MemoryAdmission.Budget);
            Assert.True(config.MemoryAdmission.UsePerSequenceHolders);
            Assert.False(config.EnablePrefixCaching);
            Assert.Equal(1, config.MaxNumRunningSequences);
            ExecuteAdmittedStep(cleanupOwner, model, prefill: true, rows: 9);
            Assert.Equal(9, GgmlGlmNative.NPast(handle));
            ExecuteAdmittedStep(cleanupOwner, model, prefill: false, rows: 1);
            Assert.Equal(10, GgmlGlmNative.NPast(handle));
            Assert.Equal(10, sequence.NumComputedTokens);
            Assert.NotNull(sequence.MemoryEnvelope);
            Assert.True(model.HasFusedSequenceCache(sequence.RequestId));
            var live = experts.ReadVmmUsage();
            Assert.Equal(1, live.Configured);
            Assert.Equal(0, live.Closed);
            Assert.Equal(3, live.LiveTables);
            Assert.Equal(2, live.HostTicketOwners);
            Assert.True(live.OwnedDeviceBytes > 0);
            Assert.Equal(0, live.PendingTicketBytes);
            Assert.Equal(0, live.QuarantinedBytes);
            Assert.Equal(live.OwnedDeviceBytes, experts.ReadUsage().AccountedBytes);
            Assert.Equal(live.OwnedDeviceBytes,
                native.AllocationUsage.Where(x => x.Kind == 4).Sum(x => x.CommittedBytes));
            Assert.Null(native.CallbackError);
            var ledgerBeforeRefusal = session.Budget.Snapshot().ToArray();
            var allocationsBeforeRefusal = native.AllocationUsage.ToArray();

            // Call the REAL exported checked free with a REAL nonnull owner,
            // deliberately before the session's normal scope-close guard.
            Assert.False(GgmlGlmNative.FreeChecked(handle, required: true));
            Assert.Equal(handle, ReadPrivate<IntPtr>(executor, "_handle"));
            Assert.Same(model, session.Model);
            Assert.Equal(10, GgmlGlmNative.NPast(handle));
            Assert.Equal(live, experts.ReadVmmUsage());
            Assert.Equal(ledgerBeforeRefusal, session.Budget.Snapshot().ToArray());
            Assert.Equal(allocationsBeforeRefusal, native.AllocationUsage.ToArray());
            Assert.Null(native.CallbackError);

            // A retained pointer alone is insufficient: actual continuation
            // must still execute successfully and advance the real slot head.
            ExecuteAdmittedStep(cleanupOwner, model, prefill: false, rows: 1);
            Assert.Equal(11, GgmlGlmNative.NPast(handle));
            Assert.Equal(11, sequence.NumComputedTokens);
            Assert.Equal(handle, ReadPrivate<IntPtr>(executor, "_handle"));
            Assert.Equal(3, experts.ReadVmmUsage().LiveTables);

            // The engine releases model holders/fences before returning the
            // scheduler's request credit. Use that actual ordering, with no
            // prefix retention. No request executes after this boundary.
            cleanupOwner.ReleaseRequest();
            Assert.Null(sequence.MemoryEnvelope);
            Assert.Equal(0, cleanupOwner.Scheduler!.RunningCount);
            Assert.Equal(cleanupOwner.Pool!.NumBlocks, cleanupOwner.Pool.NumFreeBlocks);
            Assert.Equal(0, cleanupOwner.Pool.Storage.AllocatedBytes);
            int quiescentHead = GgmlGlmNative.NPast(handle);

            // Reuse the actual production model/session disposal bridge. It
            // performs Trim(0), checked final arena close and zero-owner proof.
            model.PrepareUnifiedNativeDispose();
            var closed = experts.ReadVmmUsage();
            Assert.Equal(1, closed.Closed);
            Assert.Equal(0, closed.LiveTables);
            Assert.Equal(0, closed.HostTicketOwners);
            Assert.Equal(0, closed.OwnedDeviceBytes);
            Assert.Equal(0, closed.PendingTicketBytes);
            Assert.Equal(0, closed.QuarantinedBytes);
            Assert.Equal(0, experts.ReadUsage().AccountedBytes);
            Assert.DoesNotContain(native.AllocationUsage, x => x.Kind == 4);
            Assert.True(native.ActiveAllocations > 0, "The still-live model must retain its nonexpert declared owners until model deletion.");
            Assert.Equal(handle, ReadPrivate<IntPtr>(executor, "_handle"));
            Assert.Equal(quiescentHead, GgmlGlmNative.NPast(handle));

            // Normal session teardown invokes the real required checked-free
            // path with that nonnull handle. NativeGlmExecutor clears it only
            // after an actual true result; no manual field clearing/deletion.
            cleanupOwner.Dispose();
            cleanupOwner = null;
            Assert.Equal(IntPtr.Zero, ReadPrivate<IntPtr>(executor, "_handle"));
            Assert.Null(session.Model);
            Assert.Null(native.CallbackError);
            Assert.Equal(0, native.ActiveAllocations);
            Assert.Empty(native.AllocationUsage);
            Assert.All(session.Budget.Snapshot(), pool =>
            {
                Assert.Equal(0, pool.Reserved);
                Assert.Equal(0, pool.Committed);
            });
            session.Dispose(); // Completed cleanup stays idempotent.
            Console.WriteLine($"[GLM-VMM-CHECKED-TEARDOWN-TEST] profile=1 host_embedding=1 context=2048 chunk=512 pp_rows=9 decode_rows=1 scheduler_admitted=1 production_batch_executor=1 request_credit_released=1 live_tables_before=3 owned_device_before={live.OwnedDeviceBytes} checked_free_nonnull_refused=1 same_handle_head_ledger=1 continuation_head=11 scope_close_before_model_free=1 closed_tables=0 closed_host_owners=0 closed_device_bytes=0 actual_executor_handle_zero=1 active_allocations=0 ledger_reserved=0 ledger_committed=0 scope=real_checkpoint_normal_declared_payload_teardown_not_legacy_fault_ack_or_total_os_memory_or_performance");
        }
        catch (UnifiedMemoryModelLoadException error)
        {
            cleanupOwner = error.UnreleasedSession == null ? null : new CheckpointOwner(error.UnreleasedSession);
            originalFailure = error;
            throw;
        }
        catch (Exception error)
        {
            originalFailure = error;
            throw;
        }
        finally
        {
            if (cleanupOwner != null)
            {
                try { cleanupOwner.Dispose(); }
                catch (Exception cleanup)
                {
                    s_retainedCleanupOwner = cleanupOwner;
                    throw originalFailure == null
                        ? new AggregateException("Actual checked teardown failed; complete owner retained for cleanup retry.", cleanup)
                        : new AggregateException("Checkpoint assertion and checked teardown failed; complete owner retained for cleanup retry.", originalFailure, cleanup);
                }
            }
        }
    }

    private static void ExecuteAdmittedStep(CheckpointOwner owner, GlmDsaModel model, bool prefill, int rows)
    {
        lock (model.GpuComputeLock)
        {
            var output = owner.Scheduler!.Schedule();
            var work = Assert.Single(output.ScheduledWork);
            Assert.Same(owner.Sequence, work.Sequence);
            Assert.Equal(prefill, work.IsPrefill);
            Assert.Equal(rows, work.NumScheduledTokens);
            Assert.Equal(SequenceStatus.Running, owner.Sequence!.Status);
            Assert.NotNull(owner.Sequence.MemoryEnvelope);
            var result = Assert.Single(owner.Executor!.ExecuteStep(output));
            Assert.Same(owner.Sequence, result.Sequence);
            Assert.Null(result.Error);
            Assert.False(result.SlotUnavailable);
            Assert.Equal(prefill, result.IsPrefill);
            Assert.Equal(rows, result.TokensForwarded);
            AssertFinite(owner.Sequence.LastLogits!);
            if (prefill)
            {
                Assert.True(owner.Sequence.NumComputedTokens >= owner.Sequence.PromptTokens.Count);
                Assert.NotNull(owner.Scheduler.Config.MemoryAdmission!.AfterPrefill);
                owner.Scheduler.Config.MemoryAdmission.AfterPrefill(owner.Sequence);
            }
            owner.Scheduler.NotifyStepCompleted(output);
        }
    }

    // Retain all real scheduler/request owners as well as the model on any
    // failed cleanup; dropping the scheduler could lose its envelope retry.
    private sealed class CheckpointOwner(IUnifiedMemoryModelSession session) : IDisposable
    {
        public BlockPool? Pool { get; private set; }
        public ContinuousBatchScheduler? Scheduler { get; private set; }
        public BatchExecutor? Executor { get; private set; }
        public SequenceState? Sequence { get; private set; }
        private bool _requestReleased;

        public void PrepareRequest(SchedulerConfig config, GlmDsaModel model, int[] prompt)
        {
            Pool = new BlockPool(config.NumBlocks, config.BlockSize, model.ComputeKVBlockByteSize(config.BlockSize));
            Scheduler = new ContinuousBatchScheduler(config, Pool, NullLogger.Instance,
                supportsCrossSequenceKvReuse: false);
            Executor = new BatchExecutor(model, Pool, Scheduler, NullLogger.Instance);
            Sequence = new SequenceState("actual-vmm-checked-teardown", prompt, maxNewTokens: 4,
                blockSize: config.BlockSize, samplingConfig: null);
            Scheduler.Submit(Sequence);
        }

        public void ReleaseRequest()
        {
            if (_requestReleased || Sequence == null || Scheduler == null) return;
            var model = Assert.IsType<GlmDsaModel>(session.Model);
            lock (model.GpuComputeLock)
            {
                // Abort marks a finished request and returns its scheduler
                // blocks. Native holder cleanup must succeed before the real
                // envelope is returned. Each operation is retry-safe.
                Scheduler.Abort(Sequence.RequestId);
                Executor?.DiscardReleasedFusedCacheBookkeeping(Sequence.RequestId);
                Executor?.ReleaseLinearSpeculativeContext(Sequence.RequestId);
                model.OnSequenceReleased(Sequence.RequestId);
                Executor?.RadixCache?.Drain();
                Scheduler.NotifyMemoryReleased(Sequence.RequestId);
                _requestReleased = true;
                Scheduler.Config.MemoryAdmission?.AfterSequenceMemoryRelease?.Invoke(
                    Scheduler.GetRunningSequencesSnapshot());
            }
        }

        public void Dispose()
        {
            ReleaseRequest();
            Pool?.Storage.Dispose();
            session.Dispose();
        }
    }

    private static T ReadPrivate<T>(object owner, string name)
        => Assert.IsType<T>(owner.GetType().GetField(name, PrivateInstance)!.GetValue(owner));

    private static object? ReadPrivateNullable(object owner, string name)
        => owner.GetType().GetField(name, PrivateInstance)!.GetValue(owner);

    private static void AssertFinite(float[] logits)
    {
        Assert.Equal(154880, logits.Length);
        Assert.All(logits, value => Assert.True(float.IsFinite(value)));
        Assert.Contains(logits, value => value != 0);
    }

    private static NativeEnvScope ConfigureEnvironment()
    {
        var env = new NativeEnvScope();
        foreach (string name in new[] { "TS_GLM_TRACE", "TS_GLM_TRACE_DUMP_DIR", "TS_GLM_NODES_PER_LAYER",
            "TS_GLM_MOE_MMAP", "TS_GLM_TP_SHARD", "TS_GLM_TP_OVERSUBSCRIBE" }) env.Set(name, null);
        foreach (string name in new[] { "TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", "TS_GLM_FLASH_MTP", "TS_GLM_FLASH_MTP_SHAPE_PLAN",
            "TS_GLM_FLASH_MTP_PREFIX_COMMIT", "TS_GLM_DECODE_GRAPH_RESERVE", "TS_GLM_OP_OFFLOAD", "TS_GGML_STRICT_RECURRENT_PROJECTIONS",
            "TS_GLM_SELECTED_EXPERT_TIMING", "TS_GLM_GRAPH_MEMORY_DIAGNOSTICS", "TS_DSV4_PERF",
            "TS_HOST_MOE_RAM_CACHE_MB", "TS_HOST_MOE_RAM_CACHE_MAPPED", "TS_HOST_MOE_RAM_CACHE_PINNED",
            "TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD", "TS_HOST_MOE_RAM_CACHE_VICTIM", "TS_HOST_MOE_RAM_CACHE_FREQUENCY",
            "TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY", "TS_HOST_MOE_LAYER_PREFETCH", "TS_HOST_MOE_DECODE_PREFETCH",
            "TS_HOST_MOE_PREFILL_RING_COALESCE16", "TS_HOST_MOE_PREFILL_DOWN_OVERLAP", "TS_HOST_MOE_PHASE_COLD_GROWTH",
            "TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER", "TS_HOST_MOE_EXPERT_CACHE_PREFETCH", "TS_HOST_MOE_EXPERT_CACHE_LFU",
            "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS", "TS_HOST_MOE_FILE_READ_WORKERS",
            "TS_HOST_MOE_FILE_RANDOM_ACCESS", "TS_HOST_MOE_FILE_READ_SORT", "TS_HOST_MOE_ROUTE_TRACE" }) env.Set(name, "0");
        env.Set("TS_GLM_NATIVE", "1");
        env.Set("TS_GLM_HOST_TOKEN_EMBEDDING", "1");
        env.Set("TS_HOST_MOE_VMM_WEIGHTS", "1");
        env.Set("TS_HOST_MOE_DECODE_REBALANCE", "1");
        env.Set("TS_HOST_MOE_DECODE_STAGING_72MB", "1");
        env.Set("TS_HOST_MOE_PREFILL_RING_72MB", "1");
        env.Set("MAX_CONTEXT", "2048");
        env.Set("TS_SCHED_PREFILL_CHUNK", "512");
        env.Set("TS_UNIFIED_LAYER_MAJOR_TOKENS", "8192");
        return env;
    }
}
