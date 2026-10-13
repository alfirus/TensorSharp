// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Knowledge;
using TensorSharp.Knowledge;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server.Knowledge;

namespace InferenceWeb.Tests;

/// <summary>CPU-only model/scheduler tests: no model downloads or accelerator.</summary>
public sealed class KnowledgeModelRerankerTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "knowledge-model-rank-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _environment = new();
    private ModelService? _service;
    private ScriptModel? _model;
    private const string Input = "{\"query\":\"cobalt\",\"candidates\":[{\"id\":0,\"excerpt\":\"topic\"},{\"id\":1,\"excerpt\":\"fact\"},{\"id\":2,\"excerpt\":\"detail\"}]}";

    public KnowledgeModelRerankerTests() => _environment.ClearSpeculationVars();
    public void Dispose()
    { _service?.Dispose(); _environment.Dispose(); if (Directory.Exists(_root)) Directory.Delete(_root, true); }

    private void Load()
    {
        Directory.CreateDirectory(_root);
        string path = Path.Combine(_root, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(path);
        _service = new ModelService(NullLogger<ModelService>.Instance,
            (file, _, _, _) => _model = new ScriptModel(file));
        _service.EngineHost.SchedulerConfigOverride = new SchedulerConfig
        {
            BlockSize = 16, NumBlocks = 2048, MaxNumRunningSequences = 1,
            MaxNumBatchedTokens = 4096, MaxPrefillChunkSize = 4096, SoloPrefillChunkSize = 4096,
            EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
        };
        _service.LoadModel(path, null, "cpu");
    }

    [Fact]
    public void HostCanOptOutWithoutUnloadingItsAnswerModel()
    {
        Load();
        _environment.Set("TS_KNOWLEDGE_RERANK", "off");
        Assert.Null(_service!.CreateKnowledgeReranker());
        Assert.True(_service.IsLoaded);
        _environment.Set("TS_KNOWLEDGE_RERANK", "1");
        Assert.IsType<ModelKnowledgeReranker>(_service.CreateKnowledgeReranker());
    }

    [Fact]
    public async Task IsolatedRankGenerationBetweenAnswerRoundsUsesSingleSlotSchedulerWithoutHistoryLeak()
    {
        Load();
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        using var answerSession = new ChatSession();
        var history = new List<ChatMessage> { new() { Role = "user", Content = "PRIVATE ORIGINAL TASK" } };
        var initial = new StringBuilder();
        await foreach (var piece in _service!.ChatStreamAsync(answerSession, history, 32, deadline.Token)) initial.Append(piece);
        Assert.Contains("ready", initial.ToString());
        string order = await new ModelKnowledgeReranker(_service).RankAsync(Input, deadline.Token);
        Assert.True(KnowledgeReranking.TryParseOrder(order, 3, out var ids), order);
        Assert.Equal(new[] { 2, 1, 0 }, ids);
        Assert.Equal(0, _service.EngineHost.TryGetEngine().RunningCount);
        Assert.Equal(0, _service.EngineHost.TryGetEngine().WaitingCount);
        Assert.DoesNotContain("PRIVATE ORIGINAL TASK", _model!.Prompts.Single(prompt => prompt.Contains("Rate each candidate")));
        history.Add(new() { Role = "assistant", Content = "ready" });
        history.Add(new() { Role = "user", Content = "Continue the original task" });
        var final = new StringBuilder();
        await foreach (var piece in _service.ChatStreamAsync(answerSession, history, 32, deadline.Token)) final.Append(piece);
        Assert.Contains("ready", final.ToString());
        Assert.DoesNotContain("Rate each candidate", _model.Prompts.Last());
    }

    [Fact]
    public async Task CancelledRankAwaitsSingleSlotEngineCleanupBeforeTheAnswerCanResume()
    {
        Load();
        _model!.DelayForward = true;
        using var cancellation = new CancellationTokenSource();
        var ranker = new ModelKnowledgeReranker(_service!);
        Task<string> ranking = ranker.RankAsync(Input, cancellation.Token);
        await _model.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => ranking);
        Assert.Equal(0, _service!.EngineHost.TryGetEngine().RunningCount);
        Assert.Equal(0, _service.EngineHost.TryGetEngine().WaitingCount);
        _model.DelayForward = false;
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        using var session = new ChatSession();
        var answer = new StringBuilder();
        await foreach (var piece in _service.ChatStreamAsync(session,
            new() { new() { Role = "user", Content = "Answer after cancellation" } }, 32, deadline.Token)) answer.Append(piece);
        Assert.Contains("ready", answer.ToString());
    }

    [Theory]
    [InlineData("{\"scores\":[[0,0],[1,2],[2,3]]}", true)]
    [InlineData("```json\n{\"scores\":[[0,0],[1,2],[2,3]]}\n```", false)]
    [InlineData("{\"scores\":[[1,2],[2,3]]}", false)]
    [InlineData("{\"scores\":[[0,0],[1,2],[2,3],[0,0]]}", false)]
    [InlineData("{\"scores\":[[0,0],[1,2],[3,3]]}", false)]
    [InlineData("{\"scores\":[[0,0],[1,2],[2,3]],\"answer\":\"bad\"}", false)]
    public void PrivateGrammarConstrainsShapeCountAndMembership(string text, bool accepted)
    {
        var tokenizer = new CharTokenizer();
        var config = ModelKnowledgeReranker.CreateSampling(Input, tokenizer, "gemma4");
        Assert.True(config.Grammar!.IsActive);
        bool allowed = true;
        int index = 0;
        foreach (char character in text)
        {
            if (!Allows(config.Grammar, character)) { allowed = false; break; }
            config.Grammar.Accept(character);
            index++;
        }
        Assert.True(accepted == (allowed && config.Grammar.IsComplete), $"Accepted={accepted}; consumed={index}/{text.Length}; complete={config.Grammar.IsComplete}");
    }

    [Fact]
    public void GrammarStateIsPrivateAndOnlyHeaderProtocolsDelayActivation()
    {
        var tokenizer = new CharTokenizer();
        var first = ModelKnowledgeReranker.CreateSampling(Input, tokenizer, "muse-glimmer");
        var second = ModelKnowledgeReranker.CreateSampling(Input, tokenizer, "muse-glimmer");
        Assert.NotSame(first.Grammar, second.Grammar);
        Assert.True(first.Grammar!.IsActive);
        first.Grammar.Accept('{');
        Assert.False(Allows(first.Grammar, '{'));
        Assert.True(Allows(second.Grammar!, '{'));
        Assert.False(ModelKnowledgeReranker.CreateSampling(Input, tokenizer, "gpt-oss").Grammar!.IsActive);
    }

    private static bool Allows(GrammarConstraint grammar, int token)
    { var mask = grammar.CurrentMask(); return (mask[token >> 6] & (1UL << (token & 63))) != 0; }

    private sealed class ScriptModel : ModelBase
    {
        private const string Marker = "<SPECIAL_11>Assistant\n";
        private const string Template = "{{ '<SPECIAL_10>System\\n' }}{% for message in messages %}{{ message.role + ': ' + message.content + '\\n' }}{% endfor %}{{ '<SPECIAL_11>Assistant\\n' }}";
        private readonly StringBuilder _seen = new();
        private string _script = "";
        public bool DelayForward;
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public List<string> Prompts { get; } = new();
        public ScriptModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "nemotron_h", VocabSize = 128, NumLayers = 1, ChatTemplate = Template };
            Tokenizer = new CharTokenizer(); _maxContextLength = 32768;
        }
        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override string KVStateFingerprint => "knowledge-rank-probe";
        public override long ComputeKVBlockByteSize(int tokenCount) => tokenCount * 4L;
        public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
        public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;
        protected override float[] ForwardCore(int[] tokens)
        {
            Started.TrySetResult();
            if (DelayForward) Thread.Sleep(30);
            foreach (int token in tokens) _seen.Append((char)token);
            string seen = _seen.ToString();
            int start = seen.LastIndexOf(Marker, StringComparison.Ordinal);
            var logits = Enumerable.Repeat(-1000f, 128).ToArray();
            if (start < 0) { logits[1] = 1000; return logits; }
            start += Marker.Length;
            if (seen.AsSpan(start).StartsWith("<think></think>")) start += "<think></think>".Length;
            int at = seen.Length - start;
            if (at == 0)
            { _script = seen.Contains("Rate each candidate") ? "{\"scores\":[[0,0],[1,2],[2,3]]}" : "ready"; Prompts.Add(seen); }
            logits[at < _script.Length ? _script[at] : 1] = 1000;
            return logits;
        }
        protected override void ResetKVCacheCore() => _seen.Clear();
    }
    private sealed class CharTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 128).Select(id => ((char)id).ToString()).ToArray();
        public int VocabSize => 128;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 1 };
        public bool IsEos(int id) => id == 1;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(character => character < 128 ? (int)character : '?').ToList();
        public string Decode(List<int> tokens) => new(tokens.Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }
}
