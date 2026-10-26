// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class GreedyForwardExecutorTests
{
    private const int Vocab = 32;

    [Fact]
    public void Decode_UsesExactPendingTokenAtFuturePosition_WithoutLogitRows()
    {
        var model = new GreedyModel();
        using var run = new Run(model,SamplingConfig.Greedy);
        run.Prefill();
        Assert.Equal(0,model.GreedyCalls);
        Assert.NotNull(run.Sequence.LastLogits);

        var first = run.Step();
        Assert.Null(first.Error);
        Assert.Equal(3,first.SampledToken);
        Assert.Equal(new[] { 1,2,3 },model.Cache);
        Assert.Equal(3,run.Sequence.NumComputedTokens);
        Assert.Equal(4,run.Sequence.PendingDeviceToken);
        Assert.Equal(3,run.Sequence.PendingDevicePosition);
        Assert.Null(run.Sequence.LastLogits);
        Assert.False(run.Sequence.KvStateInPagedStorage);

        var second = run.Step();
        Assert.Null(second.Error);
        Assert.Equal(4,second.SampledToken);
        Assert.Equal(new[] { 3,4 },run.Sequence.OutputTokens);
        Assert.Equal(new[] { 3,4 },model.GreedyInputs);
        Assert.Equal(new[] { 1,2,3,4 },model.Cache);
        Assert.Equal(4,run.Sequence.PendingDevicePosition);
        Assert.Single(model.ForwardInputs); // prefill only
    }

    [Fact]
    public void Decline_ForwardsOnce_AndClearsStalePendingPrediction()
    {
        var model = new GreedyModel { Decline = true };
        using var run = new Run(model,SamplingConfig.Greedy);
        run.Prefill();
        run.Sequence.PendingDeviceToken=7;
        run.Sequence.PendingDevicePosition=100;

        var result=run.Step();

        Assert.Null(result.Error);
        Assert.Equal(3,result.SampledToken);
        Assert.Equal(1,model.GreedyCalls);
        Assert.Equal(2,model.ForwardInputs.Count);
        Assert.Equal(new[] { 3 },model.ForwardInputs[1]);
        Assert.Equal(new[] { 1,2,3 },model.Cache);
        Assert.Equal(new[] { 3 },run.Sequence.OutputTokens);
        Assert.Null(run.Sequence.PendingDeviceToken);
        Assert.NotNull(run.Sequence.LastLogits);
        Assert.Equal(4,Array.IndexOf(run.Sequence.LastLogits,run.Sequence.LastLogits.Max()));
    }

    [Theory]
    [InlineData("temperature")]
    [InlineData("repetition")]
    [InlineData("presence")]
    [InlineData("frequency")]
    [InlineData("grammar")]
    [InlineData("thinking-mask")]
    public void IneligibleSampler_RetainsFullLogitForward(string policy)
    {
        var config=SamplingConfig.Greedy;
        switch (policy)
        {
            case "temperature": config.Temperature=0.5f;config.Seed=42;break;
            case "repetition": config.RepetitionPenalty=1.2f;break;
            case "presence": config.PresencePenalty=0.2f;break;
            case "frequency": config.FrequencyPenalty=0.2f;break;
            case "grammar": config.Grammar=new GrammarConstraint(Grammar.JsonObject(),new Tokenizer());break;
            case "thinking-mask": config.ThinkingBudget=new ThinkingTokenBudget(2,5,
                openTokenId:6,openAtStart:false,suppressUnopenedEnd:true);break;
        }
        var model=new GreedyModel();
        using var run=new Run(model,config);
        run.Prefill();
        var result=run.Step();
        Assert.Null(result.Error);
        Assert.Equal(0,model.GreedyCalls);
        Assert.Equal(2,model.ForwardInputs.Count);
        Assert.NotNull(run.Sequence.LastLogits);
        Assert.Null(run.Sequence.PendingDeviceToken);
    }

    [Fact]
    public void SingleTokenPrefill_AlwaysRequestsFullLogits()
    {
        var model=new GreedyModel();
        using var run=new Run(model,SamplingConfig.Greedy);
        Assert.Null(run.Step(prefill:true,count:1).Error);
        Assert.Null(run.Step(prefill:true,count:1).Error);
        Assert.Equal(0,model.GreedyCalls);
        Assert.Equal(2,model.ForwardInputs.Count);
        Assert.NotNull(run.Sequence.LastLogits);
    }

    [Fact]
    public void ForcedThinkingClose_OverridesRemotePendingToken()
    {
        var config=SamplingConfig.Greedy;
        config.ThinkingBudget=new ThinkingTokenBudget(1,5);
        var model=new GreedyModel();
        using var run=new Run(model,config);
        run.Prefill();
        Assert.Null(run.Step().Error);
        Assert.Equal(4,run.Sequence.PendingDeviceToken);
        var closed=run.Step();
        Assert.Null(closed.Error);
        Assert.Equal(5,closed.SampledToken);
        Assert.Equal(new[] { 3,5 },model.GreedyInputs);
        Assert.Equal(new[] { 3,5 },run.Sequence.OutputTokens);
        Assert.Null(run.Sequence.LastLogits);
    }

    [Fact]
    public void FirstTokenConstraint_StillUsesPrefillLogitsBeforeGreedyForward()
    {
        var config=SamplingConfig.Greedy;config.FirstTokenAllowList=new[] { 6 };
        var model=new GreedyModel();
        using var run=new Run(model,config);
        run.Prefill();
        Assert.Equal(6,run.Step().SampledToken);
        Assert.Equal(4,run.Step().SampledToken);
        Assert.Equal(new[] { 6,4 },model.GreedyInputs);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(Vocab)]
    public void InvalidReturnedToken_FailsClosedWithoutRetry(int next)
    {
        var model=new GreedyModel { Next=next };
        using var run=new Run(model,SamplingConfig.Greedy);
        run.Prefill();
        var result=run.Step();
        Assert.IsType<InvalidOperationException>(result.Error);
        Assert.Same(result.Error,run.Sequence.Error);
        Assert.Single(model.ForwardInputs);
        Assert.Equal(1,model.GreedyCalls);
        Assert.Equal(3,model.Cache.Count); // the attempted forward already advanced
        Assert.Equal(2,run.Sequence.NumComputedTokens);
        Assert.Null(run.Sequence.LastLogits);
        Assert.Null(run.Sequence.PendingDeviceToken);
    }

    [Fact]
    public void TransportExceptionAfterAdvance_FailsClosedWithoutFullForwardRetry()
    {
        var model=new GreedyModel { ThrowAfterAdvance=true };
        using var run=new Run(model,SamplingConfig.Greedy);
        run.Prefill();
        var result=run.Step();
        Assert.IsType<IOException>(result.Error);
        Assert.Single(model.ForwardInputs);
        Assert.Equal(1,model.GreedyCalls);
        Assert.Equal(3,model.Cache.Count);
        Assert.Equal(2,run.Sequence.NumComputedTokens);
        Assert.Null(run.Sequence.LastLogits);
        Assert.Null(run.Sequence.PendingDeviceToken);
    }

    [Fact]
    public void ModelWithoutOptionalInterface_PreservesOrdinaryForwarding()
    {
        var model=new PlainModel();
        using var run=new Run(model,SamplingConfig.Greedy);
        run.Prefill();
        Assert.Equal(3,run.Step().SampledToken);
        Assert.Equal(4,run.Step().SampledToken);
        Assert.Equal(3,model.ForwardInputs.Count);
        Assert.Equal(new[] { 1,2,3,4 },model.Cache);
        Assert.NotNull(run.Sequence.LastLogits);
        Assert.Null(run.Sequence.PendingDeviceToken);
    }

    private sealed class Run : IDisposable
    {
        private readonly BlockPool pool;
        private readonly BatchExecutor executor;
        public SequenceState Sequence { get; }
        public Run(PlainModel model,SamplingConfig config)
        {
            var settings=new SchedulerConfig { BlockSize=8,NumBlocks=16,MaxNumBatchedTokens=32,MaxNumRunningSequences=1 };
            pool=new BlockPool(16,8,0);
            var scheduler=new ContinuousBatchScheduler(settings,pool,NullLogger.Instance,supportsCrossSequenceKvReuse:false);
            executor=new BatchExecutor(model,pool,scheduler,NullLogger.Instance);
            Sequence=new SequenceState("remote-greedy",new[] { 1,2 },8,8,config);
            foreach (var block in pool.AllocateNew(4)!) Sequence.BlockTable.AppendBlock(block);
            Sequence.Status=SequenceStatus.Running;
        }
        public void Prefill()=>Assert.Null(Step(prefill:true,count:2).Error);
        public SequenceStepResult Step(bool prefill=false,int count=1)
        {
            var work=new SchedulerOutput();
            work.ScheduledWork.Add(new ScheduledSequenceWork(Sequence,count,Sequence.NumComputedTokens==0,prefill));
            return Assert.Single(executor.ExecuteStep(work));
        }
        public void Dispose()=>pool.Storage.Dispose();
    }

    private class PlainModel : IModelArchitecture
    {
        public ModelConfig Config { get; }=new() { VocabSize=Vocab };
        public ITokenizer Tokenizer { get; }=new Tokenizer();
        public IMultimodalInjector MultimodalInjector=>null!;
        public IBackendExecutionPlan ExecutionPlan=>null!;
        public bool SupportsKVCacheTruncation=>true;
        public List<int[]> ForwardInputs { get; }=new();
        public List<int> Cache { get; }=new();
        public float[] Forward(int[] tokens)
        {
            ForwardInputs.Add((int[])tokens.Clone());Cache.AddRange(tokens);
            var logits=new float[Vocab];logits[Cache.Count<=2 ? 3 : 4]=10;
            return logits;
        }
        public void ResetKVCache()=>Cache.Clear();
        public void TruncateKVCache(int count)=>Cache.RemoveRange(count,Cache.Count-count);
        public void Dispose() { }
    }

    private sealed class GreedyModel : PlainModel,IGreedyForwardModel
    {
        public bool Decline { get; init; }
        public bool ThrowAfterAdvance { get; init; }
        public int Next { get; init; }=4;
        public int GreedyCalls { get; private set; }
        public List<int> GreedyInputs { get; }=new();
        public bool TryForwardGreedyToken(int token,out int nextToken)
        {
            ++GreedyCalls;GreedyInputs.Add(token);nextToken=Next;
            if (Decline) return false;
            Cache.Add(token);
            if (ThrowAfterAdvance) throw new IOException("Remote peer failed after advancing its cache.");
            return true;
        }
    }

    private sealed class Tokenizer : ITokenizer
    {
        public string[] Vocab { get; }=Enumerable.Range(0,GreedyForwardExecutorTests.Vocab)
            .Select(i=>i==3 ? "{" : i==4 ? "}" : "x").ToArray();
        public int BosTokenId=>-1;
        public int[] EosTokenIds=>Array.Empty<int>();
        public int VocabSize=>Vocab.Length;
        public List<int> Encode(string text,bool addSpecial=true)=>new();
        public string Decode(List<int> ids)=>string.Concat(ids.Select(i=>Vocab[i]));
        public void AppendTokenBytes(int token,List<byte> bytes)=>bytes.AddRange(System.Text.Encoding.UTF8.GetBytes(Vocab[token]));
        public bool IsEos(int token)=>false;
        public int LookupToken(string token)=>Array.IndexOf(Vocab,token);
    }
}
