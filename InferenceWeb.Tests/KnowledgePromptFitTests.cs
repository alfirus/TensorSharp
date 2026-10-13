// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>Real prompt rendering and engine capacity with a CPU-only model that never generates.</summary>
public sealed class KnowledgePromptFitTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "knowledge-prompt-fit-" + Guid.NewGuid().ToString("N"));
    private readonly List<ModelService> _services = new();
    private readonly EnvScope _environment = new();

    public KnowledgePromptFitTests() => _environment.ClearSpeculationVars();

    public void Dispose()
    {
        foreach (var service in _services) service.Dispose();
        _environment.Dispose();
        if (Directory.Exists(_root)) Directory.Delete(_root, true);
    }

    private (ModelService Service, FitModel Model) Create(int modelCapacity, int poolCapacity)
    {
        Directory.CreateDirectory(_root);
        string path = Path.Combine(_root, "fit-" + Guid.NewGuid().ToString("N") + ".gguf");
        BackendFailureWarmupTests.WriteProbeGguf(path);
        FitModel? model = null;
        var service = new ModelService(NullLogger<ModelService>.Instance,
            (file, _, _, _) => model = new FitModel(file, modelCapacity));
        _services.Add(service);
        // An explicit operator block limit prevents the scheduler from expanding
        // its default pool automatically to the model's larger context window.
        _environment.Set("TS_SCHED_NUM_BLOCKS", (poolCapacity / 16).ToString(System.Globalization.CultureInfo.InvariantCulture));
        service.EngineHost.SchedulerConfigOverride = new SchedulerConfig
        {
            BlockSize = 16, NumBlocks = poolCapacity / 16, MaxNumRunningSequences = 1,
            MaxNumBatchedTokens = 1024, MaxPrefillChunkSize = 1024, SoloPrefillChunkSize = 1024,
            EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
        };
        service.LoadModel(path, null, "cpu");
        return (service, model!);
    }

    [Fact]
    public void ActualPromptTokensAndRequestedAnswerMustFitAlongsideSafetyMargin()
    {
        var (service, model) = Create(modelCapacity: 4096, poolCapacity: 8192);
        // This tokenizer maps each character to one token, including CJK. The
        // template adds seven tokens for ANSWER:, independent of UTF-8 byte length.
        var messages = new List<ChatMessage> { new() { Role = "user", Content = new string('界', 200) } };
        int exactReplyAllowance = 4096 - 1024 - 207;
        Assert.True(service.KnowledgePromptFits(messages, exactReplyAllowance, null, false));
        Assert.False(service.KnowledgePromptFits(messages, exactReplyAllowance + 1, null, false));
        Assert.False(service.KnowledgePromptFits(messages, 4096, null, false));
        Assert.Equal(0, model.ForwardCalls);
    }

    [Fact]
    public void SmallerEnginePoolBoundsThePromptEvenWhenTheModelAdvertisesMoreContext()
    {
        var (service, model) = Create(modelCapacity: 8192, poolCapacity: 2048);
        var messages = new List<ChatMessage> { new() { Role = "user", Content = new string('x', 200) } };
        Assert.Equal(8192, service.ContextTokens);
        var pool = service.EngineHost.TryGetEngine().PoolStats;
        Assert.Equal(2048, pool.totalBlocks * pool.blockSize);
        Assert.True(service.KnowledgePromptFits(messages, 2048 - 1024 - 207, null, false));
        Assert.False(service.KnowledgePromptFits(messages, 2048 - 1024 - 206, null, false));
        Assert.Equal(0, model.ForwardCalls);
    }

    [Fact]
    public void ToolDeclarationsConsumeContextAndMediaCandidatesRemainOnOrdinaryRetrieval()
    {
        var (service, model) = Create(modelCapacity: 4096, poolCapacity: 8192);
        var messages = new List<ChatMessage> { new() { Role = "user", Content = new string('x', 200) } };
        int exactReplyAllowance = 4096 - 1024 - 207;
        Assert.True(service.KnowledgePromptFits(messages, exactReplyAllowance, null, false));
        var tools = new List<ToolFunction> { new() { Name = "inspect", Description = "additional tool text" } };
        Assert.False(service.KnowledgePromptFits(messages, exactReplyAllowance, tools, false));
        messages[0].ImagePaths = new() { "not-read.png" };
        Assert.False(service.KnowledgePromptFits(messages, 1, null, false));
        messages[0].ImagePaths = null;
        messages[0].AudioPaths = new() { "not-read.wav" };
        Assert.False(service.KnowledgePromptFits(messages, 1, null, false));
        Assert.Equal(0, model.ForwardCalls);
    }

    [Fact]
    public void MissingModelCannotApproveACompleteDocument()
    {
        using var service = new ModelService();
        Assert.False(service.KnowledgePromptFits(new() { new() { Role = "user", Content = "report" } }, 1, null, false));
    }

    private sealed class FitModel : ModelBase
    {
        private const string Template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            + "{% if tools %}{% for tool in tools %}{{ tool.function.description }}{% endfor %}{% endif %}"
            + "{% if add_generation_prompt %}{{ 'ANSWER:' }}{% endif %}";
        public int ForwardCalls { get; private set; }
        public FitModel(string path, int capacity) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "llama", VocabSize = 128, NumLayers = 1, ChatTemplate = Template };
            Tokenizer = new CharacterTokenizer();
            _maxContextLength = capacity;
        }
        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override string KVStateFingerprint => "knowledge-prompt-fit";
        public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
        public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
        public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;
        protected override float[] ForwardCore(int[] tokens)
        {
            ForwardCalls++;
            throw new InvalidOperationException("A context-fit check must not run inference.");
        }
        protected override void ResetKVCacheCore() { }
    }

    private sealed class CharacterTokenizer : ITokenizer
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
