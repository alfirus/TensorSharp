// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Memory;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

public partial class SpeculativeExecutionTests
{
    private sealed class EmptyAllocationScope : IDisposable { public void Dispose() { } }

    private sealed class TrackedSpeculator(IDraftHead head, Action disposed,
        Action<int>? committed = null) : ISpeculator
    {
        private readonly DraftHeadSpeculator _inner = new(head, VocabSize, HiddenSize, 2);
        public string Name => "tracked-linear-draft";
        public int MaxDraftTokens => _inner.MaxDraftTokens;
        public float MinDraftProb { get => _inner.MinDraftProb; set => _inner.MinDraftProb = value; }
        public float DefaultMinDraftProb => _inner.DefaultMinDraftProb;
        public bool NeedsHiddenState => true;
        public bool HandlesOwnPrefill => false;
        public int Propose(in DraftContext context, List<int> output) => _inner.Propose(context, output);
        public void Commit(int[] tokens, float[]? hidden, int start)
        {
            _inner.Commit(tokens, hidden, start);
            committed?.Invoke(start);
        }
        public void Reset() => _inner.Reset();
        public void Dispose() { disposed(); _inner.Dispose(); }
    }

    private static SchedulerConfig AdmittedSpeculationConfig(MemoryBudget budget, string algorithm) => new()
    {
        BlockSize = BlockSize, NumBlocks = 64, MaxNumRunningSequences = 1,
        MaxNumBatchedTokens = 64, MaxPrefillChunkSize = 64, SoloPrefillChunkSize = 64,
        EnablePrefixCaching = false, StopRepetition = false,
        Speculation = new() { Enabled = true, SpeculatorName = algorithm,
            MaxDraftTokens = 2, MaxDraftTokensExplicit = true },
        ExecutionOptionsOverride = new() { BatchedPathDisabled = true, PerSeqFusedEnabled = false,
            BatchedFusedDecodeEnabled = false, KvHolderPoolMax = 0,
            RetainedFusedCacheBudget = 0, PrefixCheckpointBudget = 0 },
        MemoryAdmission = new(budget, _ => [new("ram", 256)])
        {
            AllowSpeculativeSerialExecution = true, ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => [new("ram", 1024)],
            EnterSerialExecution = _ => new EmptyAllocationScope()
        }
    };

    private static SequenceState LifetimeSequence(string id) => new(id,
        Enumerable.Range(1, 5).ToArray(), 24, BlockSize,
        new() { Temperature = 0f, TopK = 0, TopP = 1f, RepetitionPenalty = 1f });

    [Fact]
    public async Task AdmittedSingleTokenWarmupStaysInsideTheSpeculativeContext()
    {
        var budget = new MemoryBudget([new("ram", 1024)]);
        var model = new FakeSpeculativeModel();
        using var engine = new InferenceEngine(model,
            AdmittedSpeculationConfig(budget, SpeculatorRegistry.DraftHead), NullLogger.Instance);
        var sequence = new SequenceState("warmup", new[] { 1, 2, 3 }, 1, BlockSize,
            new() { Temperature = 0f, TopK = 0, TopP = 1f, RepetitionPenalty = 1f });
        await engine.SubmitRequest(sequence).Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(SequenceStatus.FinishedLengthCapped, sequence.Status);
        Assert.NotNull(sequence.SpecStats);
        Assert.Equal(0, sequence.SpecStats.VerifySteps);
        Assert.Equal(1, sequence.SpecStats.PlainSteps);
        Assert.Equal(0, model.ForwardCalls);
        Assert.Equal(new[] { model.ExpectedNext(2) }, sequence.OutputTokens);
        Assert.Equal(1024, budget.Snapshot().Single().Available);
        Assert.Empty(model.ProtocolViolations);
    }

    [Theory]
    [InlineData("finish")]
    [InlineData("cancel")]
    [InlineData("error")]
    public async Task AdmittedLinearSpeculationRetiresBeforeRefundAndCanRunAgain(string ending)
    {
        var budget = new MemoryBudget([new("ram", 1024)]);
        var model = new FakeSpeculativeModel();
        int disposed = 0;
        string algorithm = "lifetime-" + Guid.NewGuid().ToString("N");
        InferenceEngine? runningEngine = null;
        bool first = true;
        SpeculatorRegistry.Register(algorithm, (target, _) => new TrackedSpeculator((IDraftHead)target,
            () =>
            {
                // Execution buffers are still owned when their disposer runs.
                Assert.Equal(768, budget.Snapshot().Single().Available);
                Interlocked.Increment(ref disposed);
            }, start =>
            {
                if (!first || start != 0) return;
                if (ending == "cancel") runningEngine!.Abort("first");
                if (ending == "error") throw new InvalidOperationException("injected draft catch-up failure");
            }));
        using var engine = runningEngine = new InferenceEngine(model,
            AdmittedSpeculationConfig(budget, algorithm), NullLogger.Instance);
        var request = engine.SubmitRequest(LifetimeSequence("first"));
        if (ending == "error")
            await Assert.ThrowsAsync<InvalidOperationException>(() => request.Completion.WaitAsync(TimeSpan.FromSeconds(5)));
        else
            await request.Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(SpinWait.SpinUntil(() => budget.Snapshot().Single().Available == 1024, TimeSpan.FromSeconds(5)));
        Assert.Equal(1, Volatile.Read(ref disposed));
        Assert.NotNull(request.Sequence.SpecStats);
        Assert.Null(request.Sequence.MemoryEnvelope);
        Assert.Equal(0, model.ForwardCalls);

        first = false;
        var next = engine.SubmitRequest(LifetimeSequence("next"));
        await next.Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(SpinWait.SpinUntil(() => budget.Snapshot().Single().Available == 1024, TimeSpan.FromSeconds(5)));
        Assert.Equal(2, Volatile.Read(ref disposed));
        Assert.True(next.Sequence.SpecStats!.VerifySteps > 0);
        Assert.Empty(model.ProtocolViolations);
        Assert.Equal(0, model.ForwardCalls);
        Assert.Throws<NotSupportedException>(() => engine.UpdateSpeculation(SpeculationOptions.Disabled));
    }

    [Fact]
    public async Task LinearSpeculativeDisposalFailureRetainsRequestCreditUntilRetry()
    {
        var budget = new MemoryBudget([new("ram", 1024)]);
        var model = new FakeSpeculativeModel();
        int attempts = 0;
        int fail = 1;
        string algorithm = "lifetime-refusal-" + Guid.NewGuid().ToString("N");
        SpeculatorRegistry.Register(algorithm, (target, _) => new TrackedSpeculator((IDraftHead)target, () =>
        {
            Assert.Equal(768, budget.Snapshot().Single().Available);
            Interlocked.Increment(ref attempts);
            if (Volatile.Read(ref fail) != 0) throw new InvalidOperationException("injected retirement failure");
        }));
        using var engine = new InferenceEngine(model, AdmittedSpeculationConfig(budget, algorithm), NullLogger.Instance);
        try
        {
            var request = engine.SubmitRequest(LifetimeSequence("retry"));
            await request.Completion.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal(1, engine.PendingSequenceReleaseCount);
            Assert.Equal(768, budget.Snapshot().Single().Available);
            Assert.NotNull(request.Sequence.MemoryEnvelope);
            Assert.Equal(1, Volatile.Read(ref attempts));
            Volatile.Write(ref fail, 0);
            engine.Abort("retry");
            Assert.True(SpinWait.SpinUntil(() => budget.Snapshot().Single().Available == 1024, TimeSpan.FromSeconds(5)));
            Assert.Equal(0, engine.PendingSequenceReleaseCount);
            Assert.Null(request.Sequence.MemoryEnvelope);
            Assert.Equal(2, Volatile.Read(ref attempts));
        }
        finally { Volatile.Write(ref fail, 0); }
    }
}
