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
public sealed class GlmFileSourceViewCheckedTeardownCheckpointFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda,Models";
    public GlmFileSourceViewCheckedTeardownCheckpointFactAttribute()
    {
        Skip = string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_GLM_FILE_SOURCE_VIEW_CHECKED_TEARDOWN_MODEL"))
            ? "Requires explicit TS_TEST_GLM_FILE_SOURCE_VIEW_CHECKED_TEARDOWN_MODEL; run this full-checkpoint lifecycle test alone in a fresh process."
            : TestGates.CudaSkip;
    }
}

[CollectionDefinition("GLM source-view checked teardown checkpoint", DisableParallelization = true)]
public sealed class GlmFileSourceViewCheckedTeardownCheckpointCollection { }

/// <summary>Actual nonnull checked-free refusal and normal ordered teardown.
/// Run only this explicit checkpoint test in a fresh process: native modes are
/// process-latched. Skipped/unavailable is not coverage. No test-only native
/// profile, fabricated model handle or substitute release callback is used.</summary>
[Collection("GLM source-view checked teardown checkpoint")]
public sealed class GlmFileSourceViewCheckedTeardownCheckpointTests
{
    private const long GiB = 1L << 30;
    private const BindingFlags PrivateInstance = BindingFlags.Instance | BindingFlags.NonPublic;
    // A failed checked cleanup must keep the entire real owner, its callbacks,
    // model/source/backend and tickets alive. One test => one bounded slot.
    private static CheckpointOwner? s_retainedCleanupOwner;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ReadPressureStats(ulong token,
        [Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 2)] long[] values, int length);

    private sealed class PressureStatsReader : IDisposable
    {
        private IntPtr _module;
        private readonly ReadPressureStats _read;
        internal PressureStatsReader()
        {
            _module = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            try { _read = Marshal.GetDelegateForFunctionPointer<ReadPressureStats>(NativeLibrary.GetExport(
                _module, "TSGgml_HostMoeFileSourceViewPressureStats")); }
            catch { NativeLibrary.Free(_module); _module = IntPtr.Zero; throw; }
        }
        internal long[] Snapshot(ulong token)
        {
            var values = new long[8];
            Assert.Equal(1, _read(token, values, values.Length));
            Assert.Equal(1, values[0]);
            Assert.All(values, value => Assert.True(value >= 0));
            return values;
        }
        public void Dispose()
        {
            if (_module == IntPtr.Zero) return;
            NativeLibrary.Free(_module);
            _module = IntPtr.Zero;
        }
    }

    [GlmFileSourceViewCheckedTeardownCheckpointFact]
    public void ActualLiveSourceViewsRefuseModelFreeRetainExecutionThenCheckedCloseReleasesDeclaredPayload()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_GLM_FILE_SOURCE_VIEW_CHECKED_TEARDOWN_MODEL")!;
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
            // exact source registration and the immutable real source-view profile.
            var session = GlmUnifiedMemoryModelSession.Create(path, null, BackendType.GgmlCuda, options);
            cleanupOwner = new CheckpointOwner(session);
            var model = Assert.IsType<GlmDsaModel>(session.Model);
            var experts = ReadPrivate<GgmlHostMoeExpertCacheScope>(session, "_experts");
            var native = ReadPrivate<GgmlCacheBudgetScope>(session, "_native");
            var executor = ReadPrivate<NativeGlmExecutor>(model, "_exec");
            Assert.Null(ReadPrivateNullable(executor, "_releaseForTests"));
            Assert.True(ReadPrivate<bool>(executor, "_requireCheckedFree"));
            Assert.True(ReadPrivate<bool>(executor, "_requireCheckedMappedTrim"));
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
            var empty = experts.ReadFileSourceViewUsage();
            Assert.Equal(1, empty.Configured);
            Assert.Equal(0, empty.Closed);
            Assert.Equal(126, empty.OwnedRecords);
            Assert.Equal(126, empty.LiveViews);
            Assert.Equal(0, empty.ActiveReaders);
            Assert.Equal(0, empty.PendingRecords);
            Assert.Equal(0, empty.QuarantinedRecords);
            Assert.True(empty.OwnedControlBytes > 0);
            Assert.Equal(empty.OwnedControlBytes, empty.HeldControlTicketBytes);
            Assert.True(empty.TotalHostPromise > empty.OwnedControlBytes);

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
            var live = experts.ReadFileSourceViewUsage();
            Assert.Equal(1, live.Configured);
            Assert.Equal(0, live.Closed);
            Assert.Equal(126, live.OwnedRecords);
            Assert.Equal(126, live.LiveViews);
            Assert.Equal(0, live.ActiveReaders);
            Assert.Equal(0, live.PendingRecords);
            Assert.Equal(0, live.QuarantinedRecords);
            Assert.Equal(empty.TotalHostPromise, live.TotalHostPromise);
            Assert.Equal(empty.HeldControlTicketBytes, live.HeldControlTicketBytes);
            Assert.Equal(3, GgmlBasicOps.GetHostMoeExpertCacheEntries().Length);
            long cacheBytes = experts.ReadUsage().AccountedBytes;
            Assert.True(cacheBytes > 0);
            Assert.Equal(cacheBytes,
                native.AllocationUsage.Where(x => x.Kind == 4).Sum(x => x.CommittedBytes));
            Assert.Null(native.CallbackError);
            // A small request need not encounter admission pressure. Exercise
            // the production checked working-set hint while quiescent; views,
            // their counters and the durable Host owner must stay unchanged.
            var ledgerBeforeTrim = session.Budget.Snapshot().ToArray();
            var allocationsBeforeTrim = native.AllocationUsage.ToArray();
            ulong expertToken = ReadPrivate<ulong>(experts, "_token");
            var hintBefore = cleanupOwner.ReadPressureHintStats(expertToken);
            Assert.Equal(126, hintBefore[7]);
            lock (model.GpuComputeLock) Assert.True(model.TrimUnifiedMappedResidency());
            var hintAfter = cleanupOwner.ReadPressureHintStats(expertToken);
            Assert.Equal(new long[] { 1, hintBefore[1] + 1, hintBefore[2] + 1,
                hintBefore[3] + 126, hintBefore[4] + 126, hintBefore[5], hintBefore[6], 126 }, hintAfter);
            Assert.Equal(live, experts.ReadFileSourceViewUsage());
            Assert.Equal(ledgerBeforeTrim, session.Budget.Snapshot().ToArray());
            Assert.Equal(allocationsBeforeTrim, native.AllocationUsage.ToArray());
            Assert.Equal(handle, ReadPrivate<IntPtr>(executor, "_handle"));
            Assert.Equal(10, GgmlGlmNative.NPast(handle));
            Assert.Null(native.CallbackError);
            var ledgerBeforeRefusal = session.Budget.Snapshot().ToArray();
            var allocationsBeforeRefusal = native.AllocationUsage.ToArray();

            // Call the REAL exported checked free with a REAL nonnull owner,
            // deliberately before the session's normal scope-close guard.
            Assert.False(GgmlGlmNative.FreeChecked(handle, required: true));
            Assert.Equal(handle, ReadPrivate<IntPtr>(executor, "_handle"));
            Assert.Same(model, session.Model);
            Assert.Equal(10, GgmlGlmNative.NPast(handle));
            Assert.Equal(live, experts.ReadFileSourceViewUsage());
            Assert.Equal(ledgerBeforeRefusal, session.Budget.Snapshot().ToArray());
            Assert.Equal(allocationsBeforeRefusal, native.AllocationUsage.ToArray());
            Assert.Null(native.CallbackError);

            // A retained pointer alone is insufficient: actual continuation
            // must still execute successfully and advance the real slot head.
            ExecuteAdmittedStep(cleanupOwner, model, prefill: false, rows: 1);
            Assert.Equal(11, GgmlGlmNative.NPast(handle));
            Assert.Equal(11, sequence.NumComputedTokens);
            Assert.Equal(handle, ReadPrivate<IntPtr>(executor, "_handle"));
            Assert.Equal(126, experts.ReadFileSourceViewUsage().LiveViews);

            // The engine releases model holders/fences before returning the
            // scheduler's request credit. Use that actual ordering, with no
            // prefix retention. No request executes after this boundary.
            cleanupOwner.ReleaseRequest();
            Assert.Null(sequence.MemoryEnvelope);
            Assert.Equal(0, cleanupOwner.Scheduler!.RunningCount);
            Assert.Equal(cleanupOwner.Pool!.NumBlocks, cleanupOwner.Pool.NumFreeBlocks);
            Assert.Equal(0, cleanupOwner.Pool.Storage.AllocatedBytes);
            // Exercise the real between-request pressure path after trimming
            // retained checkpoints. Release may keep an idle primary KV slot;
            // this boundary does not claim that every native slot is absent.
            model.TrimUnifiedRetainedSlots();
            int quiescentHead = GgmlGlmNative.NPast(handle);
            var betweenRequestUsage = experts.ReadFileSourceViewUsage();
            var betweenRequestLedger = session.Budget.Snapshot().ToArray();
            var betweenRequestAllocations = native.AllocationUsage.ToArray();
            var betweenRequestHintBefore = cleanupOwner.ReadPressureHintStats(expertToken);
            lock (model.GpuComputeLock) Assert.True(model.TrimUnifiedMappedResidency());
            var betweenRequestHintAfter = cleanupOwner.ReadPressureHintStats(expertToken);
            Assert.Equal(new long[] { 1, betweenRequestHintBefore[1] + 1, betweenRequestHintBefore[2] + 1,
                betweenRequestHintBefore[3] + 126, betweenRequestHintBefore[4] + 126,
                betweenRequestHintBefore[5], betweenRequestHintBefore[6], 126 }, betweenRequestHintAfter);
            Assert.Equal(betweenRequestUsage, experts.ReadFileSourceViewUsage());
            Assert.Equal(betweenRequestLedger, session.Budget.Snapshot().ToArray());
            Assert.Equal(betweenRequestAllocations, native.AllocationUsage.ToArray());
            Assert.Equal(handle, ReadPrivate<IntPtr>(executor, "_handle"));
            Assert.Equal(quiescentHead, GgmlGlmNative.NPast(handle));
            Assert.Null(native.CallbackError);
            experts.Trim(0);
            Assert.Equal(0, experts.ReadUsage().AccountedBytes);
            Assert.Equal(126, experts.ReadFileSourceViewUsage().LiveViews);
            Assert.DoesNotContain(native.AllocationUsage, x => x.Kind == 4);

            // Reuse the actual production model/session disposal bridge. It
            // detaches/closes views and checked control tickets before deletion.
            model.PrepareUnifiedNativeDispose();
            var closed = experts.ReadFileSourceViewUsage();
            Assert.Equal(1, closed.Closed);
            Assert.Equal(0, closed.OwnedRecords);
            Assert.Equal(0, closed.LiveViews);
            Assert.Equal(0, closed.ActiveReaders);
            Assert.Equal(0, closed.MappedVaBytes);
            Assert.Equal(0, closed.OwnedControlBytes);
            Assert.Equal(0, closed.HeldControlTicketBytes);
            Assert.Equal(0, closed.PendingRecords);
            Assert.Equal(0, closed.QuarantinedRecords);
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
            Console.WriteLine($"[GLM-SOURCE-VIEW-CHECKED-TEARDOWN-TEST] profile=1 host_embedding=1 vmm=0 context=2048 chunk=512 pp_rows=9 decode_rows=1 scheduler_admitted=1 production_batch_executor=1 request_credit_released=1 live_views_before=126 held_control_before={live.HeldControlTicketBytes} host_promise={live.TotalHostPromise} cache_device_before={cacheBytes} checked_pressure_hint=1 hint_range_calls=126 hint_accepted_ranges=126 unchanged_views=126 same_trim_stats=1 same_trim_charges=1 between_request_pressure_hint=1 between_request_hint_range_calls=126 between_request_hint_accepted_ranges=126 between_request_same_views_charges_head_handle=1 checked_free_nonnull_refused=1 same_handle_head_ledger=1 continuation_head=11 trim0_retained_views=126 scope_close_before_model_free=1 closed_views=0 closed_control_bytes=0 held_control_bytes=0 actual_executor_handle_zero=1 active_allocations=0 ledger_reserved=0 ledger_committed=0 scope=real_checkpoint_normal_declared_payload_teardown_not_legacy_fault_ack_or_total_os_memory_or_performance");
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
        private PressureStatsReader? _pressureStats;

        internal long[] ReadPressureHintStats(ulong token)
            => (_pressureStats ??= new PressureStatsReader()).Snapshot(token);

        public void PrepareRequest(SchedulerConfig config, GlmDsaModel model, int[] prompt)
        {
            Pool = new BlockPool(config.NumBlocks, config.BlockSize, model.ComputeKVBlockByteSize(config.BlockSize));
            Scheduler = new ContinuousBatchScheduler(config, Pool, NullLogger.Instance,
                supportsCrossSequenceKvReuse: false);
            Executor = new BatchExecutor(model, Pool, Scheduler, NullLogger.Instance);
            Sequence = new SequenceState("actual-source-view-checked-teardown", prompt, maxNewTokens: 4,
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
            _pressureStats?.Dispose(); // Outlives actual source/model/callback cleanup.
            _pressureStats = null;
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
        env.Set("TS_HOST_MOE_VMM_WEIGHTS", "0");
        env.Set("TS_HOST_MOE_FILE_SOURCE_VIEW", "1");
        env.Set("TS_HOST_MOE_DECODE_REBALANCE", "1");
        env.Set("TS_HOST_MOE_DECODE_STAGING_72MB", "1");
        env.Set("TS_HOST_MOE_PREFILL_RING_72MB", "1");
        env.Set("MAX_CONTEXT", "2048");
        env.Set("TS_SCHED_PREFILL_CHUNK", "512");
        env.Set("TS_UNIFIED_LAYER_MAJOR_TOKENS", "8192");
        return env;
    }
}
