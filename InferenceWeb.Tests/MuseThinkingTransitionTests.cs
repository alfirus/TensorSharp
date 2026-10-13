// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Reflection;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

public sealed class MuseThinkingTransitionTests
{
    private const int Eot = 0, Eom = 1, Start = 2, Message = 3, Self = 4, Thought = 5,
        Assistant = 6, User = 7, Answer = 8, Tool = 9, ToolArguments = 10;
    private const string Transition = "<|eom|><|start|>assistant to=user<|message|>";

    [Fact]
    public async Task ForcedTransitionIsForwardedAndActivatesAnswerGrammarInTheSameGeneration()
    {
        using var model = new ReasoningModel();
        using var engine = Engine(model);
        var config = Config(model.Tokenizer, 2, grammar: true);
        var sequence = new SequenceState("muse-transition", new[] { Start, Assistant }, 32, 8, config);
        var handle = engine.SubmitRequest(sequence);
        int[] output = await Collect(handle);
        int[] transition = model.Tokenizer.Encode(Transition, false).ToArray();
        Assert.Equal(new[] { Self, Message, Thought, Thought }.Concat(transition).Append(Answer), output);
        Assert.Equal("eos", (await handle.Completion).FinishReason);
        Assert.Contains(Eom, model.Forwarded);
        Assert.Contains(User, model.Forwarded);
        Assert.True(config.Grammar!.IsActive && config.Grammar.IsComplete && !config.Grammar.IsDead);

        var parser = new MuseGlimmerOutputParser();
        parser.Init(true, null);
        var visible = new StringBuilder();
        foreach (int token in output) visible.Append(parser.Add(model.Tokenizer.Decode(new() { token }), false).Content);
        visible.Append(parser.Add("", true).Content);
        Assert.Equal("{\"ok\":true}", visible.ToString());
    }

    [Theory]
    [InlineData(User)]
    [InlineData(Tool)]
    public void NaturalAnswerAndToolChannelsDoNotConsumeTheThinkingBudget(int recipient)
    {
        var tokenizer = new PieceTokenizer();
        var sampler = new TokenSampler(Config(tokenizer, 1));
        var history = new List<int> { recipient, Message };
        for (int i = 0; i < 20; i++)
        {
            history.Add(recipient == User ? Answer : ToolArguments);
            Assert.False(sampler.TryGetForcedThinkingToken(history, out _));
        }
    }

    [Fact]
    public void ToolsAvailableTransitionPreservesTheModelsToolRecipientChoice()
    {
        var tokenizer = new PieceTokenizer();
        var config = ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, tokenizer, "muse-glimmer", 1,
            out bool installed, toolsAvailable: true);
        Assert.True(installed);
        var sampler = new TokenSampler(config);
        var history = new List<int> { Self, Message, Thought };
        foreach (int expected in new[] { Eom, Start, Assistant })
        {
            Assert.True(sampler.TryGetForcedThinkingToken(history, out int actual));
            Assert.Equal(expected, actual);
            history.Add(actual);
        }
        Assert.False(sampler.TryGetForcedThinkingToken(history, out _));
        foreach (int token in new[] { Tool, Message, ToolArguments, Eot })
        {
            history.Add(token);
            Assert.False(sampler.TryGetForcedThinkingToken(history, out _));
        }
        var parser = new MuseGlimmerOutputParser();
        parser.Init(true, null);
        var calls = new List<ToolCall>();
        foreach (int token in history)
        {
            var parsed = parser.Add(tokenizer.Decode(new() { token }), false);
            if (parsed.ToolCalls != null) calls.AddRange(parsed.ToolCalls);
        }
        var call = Assert.Single(calls);
        Assert.Equal("knowledge_search", call.Name);
        Assert.Equal("x", call.Arguments!["query"]);
    }

    [Fact]
    public void RepeatedSelfReentryHasFiniteHandoffsAndRollbackRestoresTheBudgetState()
    {
        var tokenizer = new PieceTokenizer();
        var config = ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, tokenizer, "muse-glimmer", 1,
            out bool installed, toolsAvailable: true);
        Assert.True(installed);
        var sequence = new SequenceState("reentry", new[] { Start, Assistant }, 100, 8, config);
        var sampler = sequence.GetOrCreateSampler();
        foreach (int token in new[] { Self, Message, Thought }) sequence.AppendOutputToken(token);
        for (int attempt = 0; attempt < MessageThinkingTokenBudget.MaxTransitions; attempt++)
        {
            foreach (int expected in new[] { Eom, Start, Assistant })
            {
                Assert.True(sampler.TryGetForcedThinkingToken(sequence.OutputTokens, out int actual));
                Assert.Equal(expected, actual);
                sequence.AppendOutputToken(actual);
            }
            Assert.False(sampler.TryGetForcedThinkingToken(sequence.OutputTokens, out _));
            Assert.False(sequence.MessageThinkingBudgetExhausted);
            sequence.AppendOutputToken(Self);
            sequence.AppendOutputToken(Message);
        }
        Assert.False(sampler.TryGetForcedThinkingToken(sequence.OutputTokens, out _));
        Assert.True(sequence.MessageThinkingBudgetExhausted);
        Assert.False(sequence.MessageThinkingBudgetExhaustedAt(sequence.OutputTokens.Count - 1));
        Assert.True(sequence.MessageThinkingBudgetExhaustedAt(sequence.OutputTokens.Count));
        Assert.False(sampler.TryRequestThinkingClosure(sequence.OutputTokens));

        sequence.OutputTokens.RemoveRange(3, sequence.OutputTokens.Count - 3);
        Assert.True(sampler.TryGetForcedThinkingToken(sequence.OutputTokens, out int afterRollback));
        Assert.Equal(Eom, afterRollback);
        Assert.False(sequence.MessageThinkingBudgetExhausted);
        Assert.False(sequence.MessageThinkingBudgetExhaustedAt(100));
    }

    [Fact]
    public void HeaderTokensAreExcludedAndReasoningAcrossMessagesAccumulates()
    {
        var tokenizer = new PieceTokenizer();
        var sampler = new TokenSampler(Config(tokenizer, 2));
        var history = new List<int> { Self, Message, Thought, Eom, Start, Assistant, Self, Message };
        Assert.False(sampler.TryGetForcedThinkingToken(history, out _));
        history.Add(Thought);
        Assert.True(sampler.TryGetForcedThinkingToken(history, out int next));
        Assert.Equal(Eom, next);
    }

    [Fact]
    public void IncompleteUtf8TokensInsideReasoningStillCountTowardTheLimit()
    {
        var tokenizer = new PieceTokenizer { SplitUtf8 = true };
        var sampler = new TokenSampler(Config(tokenizer, 2));
        var history = new List<int> { Self, Message, Thought };
        Assert.False(sampler.TryGetForcedThinkingToken(history, out _));
        history.Add(ToolArguments);
        Assert.True(sampler.TryGetForcedThinkingToken(history, out int next));
        Assert.Equal(Eom, next);
    }

    [Fact]
    public void RepeatedPeeksDoNotAdvanceAndRollbackReconstructsTheTransition()
    {
        var tokenizer = new PieceTokenizer();
        var sampler = new TokenSampler(Config(tokenizer, 2));
        var history = new List<int> { Self, Message, Thought, Thought };
        int[] transition = tokenizer.Encode(Transition, false).ToArray();
        foreach (int expected in transition)
        {
            Assert.True(sampler.TryGetForcedThinkingToken(history, out int first));
            Assert.True(sampler.TryGetForcedThinkingToken(history, out int second));
            Assert.Equal(expected, first);
            Assert.Equal(first, second);
            history.Add(first);
        }
        Assert.False(sampler.TryGetForcedThinkingToken(history, out _));
        history.RemoveRange(4, history.Count - 4);
        Assert.True(sampler.TryGetForcedThinkingToken(history, out int restarted));
        Assert.Equal(Eom, restarted);
        history.Add(Eom);
        Assert.True(sampler.TryGetForcedThinkingToken(history, out int resumed));
        Assert.Equal(Start, resumed);
        history.RemoveRange(2, history.Count - 2);
        Assert.False(sampler.TryGetForcedThinkingToken(history, out _));
    }

    [Fact]
    public void ActiveGrammarNeverYieldsToForcedTransitionEvenAfterClosureBegins()
    {
        var tokenizer = new PieceTokenizer();
        var config = Config(tokenizer, 1);
        var sampler = new TokenSampler(config);
        var history = new List<int> { Self, Message, Thought };
        Assert.True(sampler.TryGetForcedThinkingToken(history, out int close));
        history.Add(close);
        config.Grammar = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
        Assert.False(sampler.TryGetForcedThinkingToken(history, out _));
        var logits = new float[tokenizer.VocabSize];
        logits[Thought] = 100;
        logits[Answer] = 1;
        Assert.Equal(Answer, sampler.Sample(logits, history));
    }

    [Fact]
    public void ClonedPoliciesHaveIndependentSamplerState()
    {
        var tokenizer = new PieceTokenizer();
        var config = Config(tokenizer, 1);
        var first = new TokenSampler(config);
        var second = new TokenSampler(config.Clone());
        var history = new List<int> { Self, Message, Thought };
        Assert.True(first.TryGetForcedThinkingToken(history, out int close));
        Assert.Equal(Eom, close);
        Assert.False(second.TryGetForcedThinkingToken(new List<int> { Self, Message }, out _));
        Assert.True(second.TryGetForcedThinkingToken(history, out int independentClose));
        Assert.Equal(Eom, independentClose);
    }

    [Fact]
    public void InstallingTagBudgetDoesNotRetainAnEarlierMessagePolicy()
    {
        var tokenizer = new PieceTokenizer();
        var muse = Config(tokenizer, 1);
        tokenizer.Vocab[ToolArguments] = "</think>";
        var tag = ChatGenerationPipeline.WithThinkingBudget(muse, tokenizer, "deepseek41", 2, out bool installed);
        Assert.True(installed);
        Assert.Null(tag.MessageThinkingBudget);
        Assert.NotNull(tag.ThinkingBudget);
        Assert.NotNull(muse.MessageThinkingBudget);
        Assert.Null(muse.ThinkingBudget);
    }

    [Fact]
    public void DirectAnswerPromptSuffixDoesNotCountVisibleAnswerTokens()
    {
        var tokenizer = new PieceTokenizer();
        var config = ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, tokenizer, "muse-glimmer", 1,
            out bool installed, enableThinking: false, generationSuffix: " to=user<|message|>");
        Assert.True(installed);
        var sampler = new TokenSampler(config);
        Assert.False(sampler.TryGetForcedThinkingToken(Enumerable.Repeat(Answer, 20).ToList(), out _));
    }

    [Theory]
    [InlineData("TakePendingOrSample", false)]
    [InlineData("PeekPendingOrSample", true)]
    public void TransitionOverridesPendingDeviceArgmaxWithoutHostLogits(string method, bool retainsPending)
    {
        var tokenizer = new PieceTokenizer();
        var sequence = new SequenceState("device", new[] { Start, Assistant }, 32, 8, Config(tokenizer, 1));
        foreach (int token in new[] { Self, Message, Thought }) sequence.AppendOutputToken(token);
        sequence.PendingDeviceToken = Thought;
        sequence.PendingDevicePosition = sequence.NumComputedTokens;
        Assert.True(sequence.GetOrCreateSampler().IsPlainGreedyArgmax);
        var operation = typeof(BatchExecutor).GetMethod(method, BindingFlags.Static | BindingFlags.NonPublic)!;
        Assert.Equal(Eom, (int)operation.Invoke(null, new object[] { sequence })!);
        Assert.Equal(retainsPending, sequence.PendingDeviceToken.HasValue);
        Assert.Null(sequence.LastLogits);
    }

    [Fact]
    public async Task CancellationDuringTransitionStopsWithoutGeneratingAnAnswer()
    {
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        using var model = new ReasoningModel { OnClosingForward = () =>
        {
            entered.Set();
            if (!release.Wait(TimeSpan.FromSeconds(5))) throw new TimeoutException();
        } };
        using var engine = Engine(model);
        using var cancellation = new CancellationTokenSource();
        var sequence = new SequenceState("cancel", new[] { Start, Assistant }, 32, 8, Config(model.Tokenizer, 1));
        var handle = engine.SubmitRequest(sequence, cancellation.Token);
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(5)));
            cancellation.Cancel();
        }
        finally { release.Set(); }
        Assert.DoesNotContain(Answer, await Collect(handle));
        Assert.Equal("aborted", (await handle.Completion).FinishReason);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(7)]
    public async Task TransitionNeverExtendsTheOriginalOutputLimit(int limit)
    {
        using var model = new ReasoningModel();
        using var engine = Engine(model);
        var handle = engine.SubmitRequest(new SequenceState("limited", new[] { Start, Assistant }, limit, 8, Config(model.Tokenizer, 2)));
        int[] output = await Collect(handle);
        Assert.Equal(limit, output.Length);
        Assert.Equal("max_tokens", (await handle.Completion).FinishReason);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void UnusableTransitionKeepsTheExistingFallback(bool eosDelimiter)
    {
        var tokenizer = new PieceTokenizer { EomIsEos = eosDelimiter, CorruptEncoding = !eosDelimiter };
        var original = SamplingConfig.Greedy;
        Assert.Same(original, ChatGenerationPipeline.WithThinkingBudget(original, tokenizer, "muse-glimmer", 2, out bool installed));
        Assert.False(installed);
    }

    [Fact]
    public void UnsupportedTokenizerEncodingKeepsTheExistingFallback()
    {
        var tokenizer = new PieceTokenizer { ThrowEncoding = true };
        var original = SamplingConfig.Greedy;
        Assert.Same(original, ChatGenerationPipeline.WithThinkingBudget(original, tokenizer, "muse-glimmer", 2, out bool installed));
        Assert.False(installed);
    }

    [Fact]
    public void LiteralFramingSpellingsCannotReplaceTrainedControlTokenIds()
    {
        var tokenizer = new PieceTokenizer { LiteralFramingEncoding = true };
        tokenizer.Vocab[ToolArguments] = Transition;
        var original = SamplingConfig.Greedy;
        Assert.Same(original, ChatGenerationPipeline.WithThinkingBudget(original, tokenizer, "muse-glimmer", 2, out bool installed));
        Assert.False(installed);
    }

    private static SamplingConfig Config(ITokenizer tokenizer, int budget, bool grammar = false)
    {
        var original = SamplingConfig.Greedy;
        if (grammar)
        {
            original.Grammar = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
            original.Grammar.ActivateAfter("to=user<|message|>");
        }
        var config = ChatGenerationPipeline.WithThinkingBudget(original, tokenizer, "muse-glimmer", budget, out bool installed);
        Assert.True(installed);
        if (grammar) Assert.NotSame(original.Grammar, config.Grammar);
        return config;
    }

    private static InferenceEngine Engine(ReasoningModel model) => new(model, new SchedulerConfig
    {
        MaxNumBatchedTokens = 8, MaxNumRunningSequences = 2, MaxPrefillChunkSize = 8,
        NumBlocks = 128, BlockSize = 8, EnablePrefixCaching = false, DecodeQuantumTokens = 1,
        StopRepetition = false,
    }, NullLogger.Instance);

    private static async Task<int[]> Collect(InferenceRequestHandle handle)
    {
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var output = new List<int>();
        await foreach (int token in handle.Tokens.ReadAllAsync(deadline.Token)) output.Add(token);
        await handle.Completion.WaitAsync(deadline.Token);
        return output.ToArray();
    }

    private sealed class PieceTokenizer : ITokenizer, ISpecialTokenVocabulary
    {
        public bool EomIsEos { get; init; }
        public bool CorruptEncoding { get; init; }
        public bool ThrowEncoding { get; init; }
        public bool SplitUtf8 { get; init; }
        public bool LiteralFramingEncoding { get; init; }
        public string[] Vocab { get; } = { "<|eot|>", "<|eom|>", "<|start|>", "<|message|>", " to=self", "reason",
            "assistant", " to=user", "{\"ok\":true}", " to=knowledge_search",
            "<atem:function_calls><atem:invoke name=\"knowledge_search\"><atem:parameter name=\"query\">x</atem:parameter></atem:invoke></atem:function_calls>" };
        public int BosTokenId => -1;
        public int[] EosTokenIds => EomIsEos ? new[] { Eot, Eom } : new[] { Eot };
        public int VocabSize => Vocab.Length;
        public IReadOnlyCollection<int> SpecialTokenIds => new[] { Eot, Eom, Start, Message };
        public List<int> Encode(string text, bool addSpecial = true)
        {
            if (ThrowEncoding) throw new NotSupportedException("Synthetic unsupported encoding.");
            if (LiteralFramingEncoding) return new() { ToolArguments };
            if (CorruptEncoding) return new() { Thought };
            var tokens = new List<int>();
            while (text.Length > 0)
            {
                int token = Enumerable.Range(0, Vocab.Length).OrderByDescending(id => Vocab[id].Length)
                    .FirstOrDefault(id => text.StartsWith(Vocab[id], StringComparison.Ordinal), -1);
                if (token < 0) throw new InvalidOperationException("Unknown fixture text: " + text);
                tokens.Add(token);
                text = text[Vocab[token].Length..];
            }
            return tokens;
        }
        public string Decode(List<int> tokens) => string.Concat(tokens.Select(token => Vocab[token]));
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.AddRange(SplitUtf8 && token == Thought
            ? new byte[] { 0xE4, 0xB8 } : SplitUtf8 && token == ToolArguments ? new byte[] { 0xAD }
            : Encoding.UTF8.GetBytes(Vocab[token]));
        public bool IsEos(int token) => token == Eot || EomIsEos && token == Eom;
        public int LookupToken(string token) => Array.IndexOf(Vocab, token);
    }

    private sealed class ReasoningModel : IModelArchitecture
    {
        private readonly List<int> _state = new();
        public List<int> Forwarded { get; } = new();
        public Action? OnClosingForward { get; init; }
        public ModelConfig Config { get; } = new() { Architecture = "muse-glimmer", VocabSize = 11 };
        public ITokenizer Tokenizer { get; } = new PieceTokenizer();
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => "muse-thinking-transition-test";
        public float[] Forward(int[] tokens)
        {
            _state.AddRange(tokens);
            Forwarded.AddRange(tokens);
            if (tokens.Contains(Eom)) OnClosingForward?.Invoke();
            int next = _state.Contains(User) ? (_state[^1] == Answer ? Eot : Answer)
                : _state[^1] == Assistant ? Self : _state[^1] == Self ? Message : Thought;
            var logits = new float[Config.VocabSize];
            logits[next] = 20;
            return logits;
        }
        public void ResetKVCache() => _state.Clear();
        public void TruncateKVCache(int count) => _state.RemoveRange(count, _state.Count - count);
        public long ComputeKVBlockByteSize(int count) => count;
        public bool TryExtractKVBlock(int start, int count, Span<byte> destination)
        {
            if (destination.Length != count || start + count > _state.Count) return false;
            for (int i = 0; i < count; i++) destination[i] = (byte)_state[start + i];
            return true;
        }
        public bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source)
        {
            if (start != _state.Count || source.Length != count) return false;
            foreach (byte token in source) _state.Add(token);
            return true;
        }
        public void Dispose() { }
    }
}
