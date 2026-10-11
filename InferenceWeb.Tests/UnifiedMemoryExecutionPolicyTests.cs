// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp;
using TensorSharp.Memory;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

/// <summary>Actual CPU scheduler/executor checks; no native or model qualification.</summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class UnifiedMemoryExecutionPolicyTests
{
    private static readonly ExecutionOptions Serial = new()
    {
        BatchedPathDisabled = true, PerSeqFusedEnabled = false,
        BatchedFusedDecodeEnabled = false, RetainedFusedCacheBudget = 0,
        PrefixCheckpointBudget = 0, KvHolderPoolMax = 0
    };

    private static SchedulerConfig Config(RequestMemoryAdmission admission, int lanes = 1) => new()
    {
        BlockSize = 8, NumBlocks = 32, MaxNumRunningSequences = lanes,
        MaxNumBatchedTokens = 8, MaxPrefillChunkSize = 8, SoloPrefillChunkSize = 8,
        EnablePrefixCaching = false, StopRepetition = false,
        MemoryAdmission = admission, ExecutionOptionsOverride = Serial
    };
    private static SequenceState Request(string id) => new(id, [1, 2, 3, 4], 3, 8, SamplingConfig.Greedy);
    private sealed class NoOpScope : IDisposable { public void Dispose() { } }

    [Fact]
    public void InitiallyOversizedPeakCanShrinkAtPhysicalBoundaryBeforeAdmission()
    {
        var budget = new MemoryBudget([new("device", 100)]); int samples = 0;
        var cfg = Config(new(budget, _ => [new("device", samples == 0 ? 110 : 80)])
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { samples++; return [new("device", 100)]; },
            EnterSerialExecution = _ => new NoOpScope()
        });
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool); var seq = Request("shrink-at-boundary");
        scheduler.Submit(seq);
        Assert.Equal(0, samples); Assert.Null(seq.MemoryEnvelope);
        Assert.Same(seq, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
        Assert.Equal(1, samples); Assert.Equal(80, Assert.Single(seq.MemoryEnvelope!.Charges).Bytes);
        scheduler.Abort(seq.RequestId); scheduler.NotifyMemoryReleased(seq.RequestId);
        Assert.Equal(100, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void InitiallyOversizedPeakStillFailsWhenRefreshCannotMakeItFit()
    {
        var budget = new MemoryBudget([new("device", 100)]); int samples = 0;
        var cfg = Config(new(budget, _ => [new("device", 110)])
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { samples++; return [new("device", 100)]; },
            EnterSerialExecution = _ => throw new InvalidOperationException("Must not execute")
        });
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool); var seq = Request("too-large-even-after-boundary");
        scheduler.Submit(seq);
        Assert.Empty(scheduler.Schedule().ScheduledWork);
        Assert.Equal(1, samples); Assert.Equal(SequenceStatus.FinishedError, seq.Status);
        Assert.IsType<MemoryPressureException>(seq.Error); Assert.Null(seq.MemoryEnvelope);
        Assert.Equal(0, scheduler.WaitingCount); Assert.Equal(0, scheduler.RunningCount);
        Assert.Equal(100, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void OrdinaryAdmissionStillRejectsOversizedPeakAtSubmission()
    {
        var budget = new MemoryBudget([new("device", 100)]); int samples = 0;
        var cfg = Config(new(budget, _ => [new("device", 110)])
        {
            BeforeAdmission = _ => { samples++; return [new("device", 100)]; },
            EnterSerialExecution = _ => new NoOpScope()
        });
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool); var seq = Request("ordinary-too-large");
        Assert.Throws<MemoryPressureException>(() => scheduler.Submit(seq));
        Assert.Equal(0, samples); Assert.Null(seq.MemoryEnvelope);
        Assert.Equal(0, scheduler.WaitingCount); Assert.Equal(100, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void QueuedFollowerReestimatesAfterPriorRequestMaterializesRetainedAllocation()
    {
        var budget = new MemoryBudget([new("device", 80)]);
        bool retained = false;
        int samples = 0, estimates = 0;
        var cfg = Config(new(budget, _ => { estimates++; return [new("device", retained ? 20 : 80)]; })
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { samples++; return [new("device", 80)]; },
            EnterSerialExecution = _ => new NoOpScope()
        });
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool);
        var first = Request("first"); var next = Request("queued-before-materialization");
        scheduler.Submit(first); scheduler.Submit(next);
        Assert.Same(first, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
        using var materialized = first.MemoryEnvelope!.TryTake([new("device", 60)])!;
        materialized.Commit(); retained = true;
        Assert.True(scheduler.Abort(first.RequestId)); scheduler.NotifyMemoryReleased(first.RequestId);
        Assert.Equal(60, budget.Snapshot().Single().Committed);
        Assert.Same(next, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
        Assert.Equal(20, Assert.Single(next.MemoryEnvelope!.Charges).Bytes);
        Assert.Equal(2, samples); Assert.Equal(4, estimates);
        scheduler.Abort(next.RequestId); scheduler.NotifyMemoryReleased(next.RequestId);
        materialized.Dispose();
        Assert.Equal(80, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void ReestimateRunsAfterPhysicalTrimAndCanIncreaseTheIncrementalReservation()
    {
        var budget = new MemoryBudget([new("device", 80)]);
        using var retained = budget.Reserve([new("device", 60)]); retained.Commit();
        bool trimmed = false;
        var cfg = Config(new(budget, _ => [new("device", trimmed ? 80 : 20)])
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { retained.Dispose(); trimmed = true; return [new("device", 80)]; },
            EnterSerialExecution = _ => new NoOpScope()
        });
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool); var seq = Request("trimmed");
        scheduler.Submit(seq);
        Assert.Single(scheduler.Schedule().ScheduledWork);
        Assert.Equal(80, Assert.Single(seq.MemoryEnvelope!.Charges).Bytes);
        scheduler.Abort(seq.RequestId); scheduler.NotifyMemoryReleased(seq.RequestId);
        Assert.Equal(80, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void RefreshedImpossiblePeakFailsBeforeExecutionWithoutLeakingReservation()
    {
        var budget = new MemoryBudget([new("device", 80)]); bool observed = false;
        var cfg = Config(new(budget, _ => [new("device", observed ? 81 : 20)])
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { observed = true; return [new("device", 80)]; },
            EnterSerialExecution = _ => throw new InvalidOperationException("Must not execute")
        });
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool); var seq = Request("too-large-after-refresh");
        scheduler.Submit(seq);
        Assert.Empty(scheduler.Schedule().ScheduledWork);
        Assert.Equal(SequenceStatus.FinishedError, seq.Status);
        Assert.IsType<MemoryPressureException>(seq.Error);
        Assert.Null(seq.MemoryEnvelope); Assert.Equal(80, budget.Snapshot().Single().Available);
    }

    [Theory]
    [InlineData(false, true, 1)]
    [InlineData(true, false, 1)]
    [InlineData(true, true, 2)]
    public void IncrementalRefreshRefusesMissingBoundaryExecutionOrMultipleLanes(bool sample, bool execute, int lanes)
    {
        var budget = new MemoryBudget([new("device", 80)]);
        var cfg = Config(new(budget, _ => [new("device", 20)])
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = sample ? _ => [new("device", 80)] : null,
            EnterSerialExecution = execute ? _ => new NoOpScope() : null
        }, lanes);
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        Assert.Throws<ArgumentException>(() => new ContinuousBatchScheduler(cfg, pool));
    }

    [Fact]
    public void SchedulerCopiesPreserveTheExplicitImmutableExecutionPolicy()
    {
        using var env = new EnvScope(); env.Set("TS_SCHED_DISABLE_BATCHED", "0");
        var cfg = Config(new(new MemoryBudget([new("device", 80)]), _ => []));
        Assert.Same(Serial, cfg.ResolveExecutionOptions());
        Assert.Same(Serial, cfg.WithBlockSize(16).ExecutionOptionsOverride);
        Assert.Same(Serial, cfg.WithSpeculation(SpeculationOptions.Disabled).ExecutionOptionsOverride);
        Assert.False(new SchedulerConfig().ResolveExecutionOptions().BatchedPathDisabled);
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED"));
    }

    [Fact]
    public void CapabilityReportDistinguishesAdmittedEnginePolicyFromEnvironmentAndModelFusion()
    {
        var cfg = Config(new(new MemoryBudget([new("device", 80)]), _ => [])
        { EnterSerialExecution = _ => new NoOpScope() });
        var caps = new ExecutionCapabilities
        {
            SupportsBatchedPagedAttention = true, SupportsLinearKvMigration = false,
            SupportsPerSequenceFusedForward = true, SupportsRetainedFusedCache = true
        };
        string report = ExecutionPlanner.BuildCapabilityReport(caps, cfg.ResolveExecutionOptions(), cfg);
        Assert.Contains("active per-engine policy: SchedulerConfig.ExecutionOptionsOverride", report);
        Assert.Contains("does not set environment variables", report);
        Assert.DoesNotContain("active TS_* overrides:", report);
        Assert.Contains("N=1 single-sequence fused fast path: disabled (per-engine policy selects the serial model route)", report);
        Assert.Contains("admitted serial execution: model.Forward", report);
        Assert.Contains("model-owned fused kernels remain available where supported by the model/backend", report);
    }

    [Fact]
    public void OrdinaryCapabilityReportPreservesEnvironmentOverrideWording()
    {
        var caps = new ExecutionCapabilities
        {
            SupportsBatchedPagedAttention = true, SupportsPerSequenceFusedForward = true,
            SupportsRetainedFusedCache = true
        };
        string report = ExecutionPlanner.BuildCapabilityReport(caps, Serial, new SchedulerConfig());
        Assert.Contains("batched paged attention: disabled (TS_SCHED_DISABLE_BATCHED)", report);
        Assert.Contains("per-sequence fused concurrent decode: disabled (TS_PER_SEQ_FUSED=0)", report);
        Assert.Contains("retained fused-cache continuation: off (TS_RETAINED_FUSED_CACHE_MAX=0)", report);
        Assert.Contains("active TS_* overrides: " + Serial.DescribeOverrides(), report);
        Assert.DoesNotContain("per-engine policy", report);
        Assert.DoesNotContain("admitted serial execution", report);
    }

    [Fact]
    public void EvenDefaultValuedEnginePolicyIsNamedWithoutClaimingFusedKernelSupport()
    {
        var cfg = new SchedulerConfig { ExecutionOptionsOverride = ExecutionOptions.Default };
        string report = ExecutionPlanner.BuildCapabilityReport(new ExecutionCapabilities(), cfg.ResolveExecutionOptions(), cfg);
        Assert.Contains("active per-engine policy: SchedulerConfig.ExecutionOptionsOverride; default execution settings", report);
        Assert.DoesNotContain("active TS_* overrides", report);
        Assert.DoesNotContain("model-owned fused kernels", report);
    }

    [Fact]
    public async Task ActualExecutorUsesSerialOverrideAndSamplesOnceForEachRequestNotEachToken()
    {
        using var env = new EnvScope(); env.Set("TS_SCHED_DISABLE_BATCHED", "0");
        var budget = new MemoryBudget([new("device", 80)]); int samples = 0, scopes = 0;
        using var model = new RoutedModel();
        using var engine = new InferenceEngine(model, Config(new(budget, _ => [new("device", 20)])
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { Assert.True(Monitor.IsEntered(model.GpuComputeLock)); samples++; return [new("device", 80)]; },
            EnterSerialExecution = seq => { Assert.NotNull(seq.MemoryEnvelope); scopes++; return new NoOpScope(); }
        }));
        var handles = new[] { engine.SubmitRequest(Request("a")), engine.SubmitRequest(Request("b")) };
        await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.All(handles, h => Assert.Equal(new[] { 5, 5, 5 }, h.Sequence.OutputTokens));
        Assert.Equal(2, samples); Assert.True(scopes >= 6); Assert.True(model.Forwards >= 6);
        Assert.Equal(0, model.BatchedForwards);
        Assert.Throws<NotSupportedException>(() => engine.UpdateSpeculation(new SpeculationOptions { Enabled = true }));
        engine.Dispose(); Assert.Equal(80, budget.Snapshot().Single().Available);
    }

    private sealed class RoutedModel : IModelArchitecture, IBatchedPagedModel
    {
        public object GpuComputeLock { get; } = new();
        public ModelConfig Config { get; } = new() { VocabSize = 8 };
        public ITokenizer Tokenizer { get; } = new Tokens();
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public bool SupportsLinearKVMigration => true;
        public int Forwards, BatchedForwards;
        public float[] Forward(int[] tokens) { Forwards++; return [0, 0, 0, 0, 0, 10, 0, 0]; }
        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext context)
        { BatchedForwards++; throw new InvalidOperationException("The admitted serial route must not call ForwardBatch."); }
        public void ResetKVCache() { }
        public void TruncateKVCache(int tokens) { }
        public void Dispose() { }
        public long ComputeKVBlockByteSize(int count) => count * 4L;
        public bool TryExtractKVBlock(int start, int count, Span<byte> target) { target.Clear(); return true; }
        public bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;
    }
    private sealed class Tokens : ITokenizer
    {
        public string[] Vocab { get; } = Enumerable.Range(0, 8).Select(x => x.ToString()).ToArray();
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public int VocabSize => 8;
        public List<int> Encode(string text, bool addSpecial = true) => [];
        public string Decode(List<int> ids) => string.Join(",", ids);
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => buffer.Add((byte)('0' + tokenId));
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string token) => -1;
    }
}
