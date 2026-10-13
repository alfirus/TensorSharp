// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.DependencyInjection;
using TensorSharp.AgentHost.Knowledge;
using TensorSharp.Knowledge;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Knowledge;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Responses;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>Real adapters, retrieval, model scheduler and tool loop, with deterministic CPU token logits.</summary>
public sealed class KnowledgeProtocolTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "knowledge-protocol-" + Guid.NewGuid().ToString("N"));
    private readonly KnowledgeStore _knowledge;
    private readonly string _modelPath;
    private readonly InMemoryResponsesStore _responses = new();
    private ModelService? _service;
    private ScriptModel? _model;

    public KnowledgeProtocolTests()
    {
        Directory.CreateDirectory(_root);
        _modelPath = Path.Combine(_root, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(_modelPath);
        _knowledge = new KnowledgeStore(Path.Combine(_root, "knowledge"), new BuiltInKnowledgeExtractor());
    }

    public void Dispose()
    {
        _service?.Dispose(); _responses.Dispose(); _knowledge.Dispose();
        if (Directory.Exists(_root)) Directory.Delete(_root, true);
    }

    private async Task<KnowledgeDocument> Import()
    {
        var collection = await _knowledge.CreateCollectionAsync("selected");
        var hidden = await _knowledge.CreateCollectionAsync("hidden");
        await _knowledge.ImportAsync(hidden.Id, "secret.txt", new MemoryStream(Encoding.UTF8.GetBytes("cobalt hidden password NIGHTJAR-999")));
        return await _knowledge.ImportAsync(collection.Id, "field-notes.txt", new MemoryStream(Encoding.UTF8.GetBytes("The cobalt access code is ORCHARD-731.")));
    }

    private void Load(params string[] scripts)
    {
        _service = new ModelService(NullLogger<ModelService>.Instance,
            (path, _, _, _) => _model = new ScriptModel(path, scripts));
        _service.EngineHost.SchedulerConfigOverride = new SchedulerConfig
        {
            BlockSize = 16, NumBlocks = 4096, MaxNumRunningSequences = 2,
            MaxNumBatchedTokens = 4096, MaxPrefillChunkSize = 4096, SoloPrefillChunkSize = 4096,
            EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
        };
        _service.LoadModel(_modelPath, null, "cpu");
    }

    private static Dictionary<string, object?> Request(string protocol, bool stream, string? collection)
    {
        var body = new Dictionary<string, object?>
        {
            ["model"] = "probe.gguf", ["stream"] = stream, ["skills"] = Array.Empty<string>(),
            ["skills_discovery"] = false, ["multi_agent"] = false, ["max_tokens"] = 900,
        };
        const string question = "What is the cobalt access code? Cite the source.\nANSWER:";
        if (protocol == "responses") { body["input"] = question; body["store"] = true; }
        else if (protocol == "generate") body["prompt"] = question;
        else body["messages"] = new[] { new { role = "user", content = question } };
        if (collection != null) body["knowledgeCollections"] = new[] { collection };
        return body;
    }

    private async Task<(DefaultHttpContext Context, string Text)> Invoke(string protocol, object request, bool withStore = true)
    {
        _service ??= new ModelService();
        var options = ServerOptionsBuilder.Build(new[] { "--model", _modelPath, "--backend", "cpu", "--no-skills", "--max-tokens", "900" }, _root);
        var uploads = new UploadStoragePolicy(Path.Combine(_root, "uploads"));
        var skills = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var workspaces = new SessionWorkspaceManager(Path.Combine(_root, "workspaces"));
        var context = new DefaultHttpContext();
        context.Request.ContentType = "application/json";
        context.Request.Body = new MemoryStream(JsonSerializer.SerializeToUtf8Bytes(request));
        context.Response.Body = new MemoryStream();
        KnowledgeStore? knowledge = withStore ? _knowledge : null;
        switch (protocol)
        {
            case "chat": await new OpenAIChatAdapter(_service, options, uploads, skills, null, workspaces, NullLoggerFactory.Instance, knowledge).ChatCompletionsAsync(context); break;
            case "responses": await new OpenAIResponsesAdapter(_service, options, uploads, skills, null, workspaces, NullLoggerFactory.Instance, _responses, knowledge).CreateResponseAsync(context); break;
            case "ollama": await new OllamaAdapter(_service, options, uploads, skills, null, workspaces, NullLoggerFactory.Instance, knowledge).ChatAsync(context); break;
            case "generate": await new OllamaAdapter(_service, options, uploads, skills, null, workspaces, NullLoggerFactory.Instance, knowledge).GenerateAsync(context); break;
            default: throw new ArgumentException(protocol);
        }
        context.Response.Body.Position = 0;
        return (context, await new StreamReader(context.Response.Body).ReadToEndAsync());
    }

    private static (string Answer, JsonElement Final) Read(string protocol, bool stream, string text)
    {
        if (!stream)
        {
            var body = JsonSerializer.Deserialize<JsonElement>(text);
            return (protocol switch
            {
                "chat" => body.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!,
                "responses" => body.GetProperty("output").EnumerateArray().Where(item => item.GetProperty("type").GetString() == "message")
                    .Select(item => item.GetProperty("content")[0].GetProperty("text").GetString()).FirstOrDefault() ?? "",
                "ollama" => body.GetProperty("message").GetProperty("content").GetString()!,
                _ => body.GetProperty("response").GetString()!,
            }, body);
        }
        var frames = text.Split('\n').Where(line => protocol is "chat" or "responses" ? line.StartsWith("data: ") && line != "data: [DONE]" : line.StartsWith('{'))
            .Select(line => JsonSerializer.Deserialize<JsonElement>(line.StartsWith("data: ") ? line[6..] : line)).ToArray();
        Assert.DoesNotContain(frames, frame => frame.TryGetProperty("replace", out _));
        string answer = string.Concat(frames.Select(frame => protocol switch
        {
            "chat" => frame.GetProperty("choices")[0].GetProperty("delta").TryGetProperty("content", out var content) ? content.GetString() : "",
            "responses" => frame.GetProperty("type").GetString() == "response.output_text.delta" ? frame.GetProperty("delta").GetString() : "",
            "ollama" => frame.GetProperty("message").GetProperty("content").GetString(),
            _ => frame.GetProperty("response").GetString(),
        }));
        return (answer, protocol == "responses" ? frames[^1].GetProperty("response") : frames[^1]);
    }

    [Theory]
    [InlineData("chat", false)] [InlineData("chat", true)]
    [InlineData("responses", false)] [InlineData("responses", true)]
    [InlineData("ollama", false)] [InlineData("ollama", true)]
    [InlineData("generate", false)] [InlineData("generate", true)]
    public async Task SelectedKnowledgeReachesInferenceAndCitationsAreAuthoritativeAndStateless(string protocol, bool stream)
    {
        var document = await Import();
        await _knowledge.ImportAsync(document.CollectionId, "unused.txt", new MemoryStream(Encoding.UTF8.GetBytes("The cobalt access code has an unused supporting note COPPER-UNUSED.")));
        string wrong = KnowledgeTools.SourceUrl(document.CollectionId, document.FileName, document.Version);
        string right = KnowledgeTools.SourceUrl(document.CollectionId, document.Id, document.Version);
        Load($"ORCHARD-731 [source]({wrong})", "No knowledge selected.");
        var response = await Invoke(protocol, Request(protocol, stream, document.CollectionId));
        Assert.Equal(200, response.Context.Response.StatusCode);
        var (answer, final) = Read(protocol, stream, response.Text);
        Assert.Equal($"ORCHARD-731 [source]({right})", answer);
        Assert.Equal(1, final.GetProperty("knowledge_citation_repairs").GetInt32());
        var source = Assert.Single(final.GetProperty("knowledge_sources").EnumerateArray());
        Assert.Equal(document.Id, source.GetProperty("documentId").GetString());
        Assert.Equal(right, source.GetProperty("sourceUrl").GetString());
        Assert.Empty(final.GetProperty("knowledge_tool_steps").EnumerateArray());
        Assert.Contains("ORCHARD-731", _model!.Prompts[0]);
        Assert.Contains("COPPER-UNUSED", _model.Prompts[0]);
        Assert.DoesNotContain("NIGHTJAR-999", _model.Prompts[0]);
        if (protocol == "responses")
        {
            Assert.True(_responses.TryGet(final.GetProperty("id").GetString()!, out var stored));
            Assert.Contains("knowledge_sources", stored.Json);
            Assert.Contains(right, stored.Json);
        }
        var next = await Invoke(protocol, Request(protocol, stream, null));
        var nextReply = Read(protocol, stream, next.Text);
        Assert.False(nextReply.Final.TryGetProperty("knowledge_sources", out _));
        Assert.DoesNotContain("ORCHARD-731", _model.Prompts[^1]);
        Assert.DoesNotContain("Reference data retrieved", _model.Prompts[^1]);
    }

    [Theory]
    [InlineData("chat", false)] [InlineData("chat", true)]
    [InlineData("responses", false)] [InlineData("responses", true)]
    [InlineData("ollama", false)] [InlineData("ollama", true)]
    public async Task KnowledgeToolsExecuteInsideTheProtocolLoopWithSkillsDisabled(string protocol, bool stream)
    {
        var document = await Import();
        Load("<TOOLCALL>[knowledge_search(query=\"cobalt\")]</TOOLCALL>", "The access code is ORCHARD-731.");
        var response = await Invoke(protocol, Request(protocol, stream, document.CollectionId));
        var (answer, final) = Read(protocol, stream, response.Text);
        Assert.Equal("The access code is ORCHARD-731.", answer);
        Assert.Empty(final.GetProperty("knowledge_sources").EnumerateArray());
        var step = Assert.Single(final.GetProperty("knowledge_tool_steps").EnumerateArray());
        Assert.Equal(KnowledgeTools.Search, step.GetProperty("tool").GetString());
        Assert.True(step.GetProperty("ok").GetBoolean());
        Assert.True(step.GetProperty("resultBytes").GetInt32() > 0);
        Assert.Equal(2, _model!.Prompts.Count);
    }

    [Theory]
    [InlineData("chat")] [InlineData("responses")] [InlineData("ollama")]
    public async Task ToolChoiceNoneKeepsRetrievalAndSuppressesKnowledgeTools(string protocol)
    {
        var document = await Import();
        Load("ORCHARD-731");
        var request = Request(protocol, false, document.CollectionId);
        request["tool_choice"] = "none";
        var response = await Invoke(protocol, request);
        Assert.Empty(Read(protocol, false, response.Text).Final.GetProperty("knowledge_sources").EnumerateArray());
        Assert.DoesNotContain("knowledge_search", _model!.Prompts[0]);
        Assert.Contains("ORCHARD-731", _model.Prompts[0]);
    }

    [Theory]
    [InlineData("chat", false)] [InlineData("chat", true)]
    [InlineData("responses", false)] [InlineData("responses", true)]
    [InlineData("ollama", false)] [InlineData("ollama", true)]
    public async Task CallerOwnedKnowledgeToolNameRemainsAnExternalToolCall(string protocol, bool stream)
    {
        var document = await Import();
        Load("<TOOLCALL>[knowledge_search(query=\"cobalt\")]</TOOLCALL>");
        var request = Request(protocol, stream, document.CollectionId);
        var function = new { name = KnowledgeTools.Search, description = "Caller owns this tool", parameters = new
        { type = "object", properties = new { query = new { type = "string" } }, required = new[] { "query" } } };
        request["tools"] = protocol == "responses"
            ? (object)new[] { new { type = "function", function.name, function.description, function.parameters } }
            : new[] { new { type = "function", function } };
        var response = await Invoke(protocol, request);
        Assert.Equal(200, response.Context.Response.StatusCode);
        var (_, final) = Read(protocol, stream, response.Text);
        Assert.Empty(final.GetProperty("knowledge_tool_steps").EnumerateArray());
        Assert.Contains(protocol == "responses" ? "function_call" : "tool_calls", response.Text);
        Assert.Contains("knowledge_search", response.Text);
        Assert.Single(_model!.Prompts);
    }

    [Theory]
    [InlineData("chat", false)] [InlineData("chat", true)]
    [InlineData("responses", false)] [InlineData("responses", true)]
    public async Task StructuredOutputsKeepTheirValidatedContentAndCarrySourcesSeparately(string protocol, bool stream)
    {
        var document = await Import();
        string right = KnowledgeTools.SourceUrl(document.CollectionId, document.Id, document.Version);
        string wrong = KnowledgeTools.SourceUrl(document.CollectionId, document.FileName, document.Version);
        string answer = JsonSerializer.Serialize(new { source = right, untouched = wrong });
        Load(answer);
        var request = Request(protocol, stream, document.CollectionId);
        request["tool_choice"] = "none";
        object schema = new { type = "object", properties = new { source = new { type = "string" }, untouched = new { type = "string" } }, required = new[] { "source", "untouched" }, additionalProperties = false };
        if (protocol == "chat") request["response_format"] = new { type = "json_schema", json_schema = new { name = "result", strict = true, schema } };
        else request["text"] = new { format = new { type = "json_schema", name = "result", strict = true, schema } };
        var response = await Invoke(protocol, request);
        Assert.Equal(200, response.Context.Response.StatusCode);
        var (output, final) = Read(protocol, stream, response.Text);
        Assert.Equal(right, JsonSerializer.Deserialize<JsonElement>(output).GetProperty("source").GetString());
        Assert.Equal(wrong, JsonSerializer.Deserialize<JsonElement>(output).GetProperty("untouched").GetString());
        Assert.Equal(0, final.GetProperty("knowledge_citation_repairs").GetInt32());
        Assert.Single(final.GetProperty("knowledge_sources").EnumerateArray());
    }

    [Fact]
    public async Task ExecutedToolPreviewIsUtf8BoundedAndModelToolClaimsAreNotRecorded()
    {
        var document = await Import();
        var scope = new KnowledgeToolScope(_knowledge, new[] { document.CollectionId });
        var options = ServerOptionsBuilder.Build(new[] { "--no-skills" }, _root);
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var plan = SkillRequestPlan.Create(registry, Array.Empty<string>(), false, null, "nemotron_h", 8192, options,
            out _, knowledge: scope, multiAgent: false)!;
        var request = new KnowledgeProtocolRequest(scope) { Plan = plan };
        request.Observe(ChatStreamUpdate.ToolProgress("writing", "imaginary"));
        request.Observe(ChatStreamUpdate.ToolProgress("finished", "imaginary"));
        request.Observe(ChatStreamUpdate.ToolProgress("running", "shell", piece: "3915888\n " + new string('文', 500)));
        // The next CJK rune does not fit; later ASCII must not turn the prefix
        // into a discontinuous preview that silently skips output in the middle.
        request.Observe(ChatStreamUpdate.ToolProgress("running", "shell", piece: "BAD"));
        plan.Invocations.Add(new(1, "shell", null, null, true, 1000));
        request.Observe(ChatStreamUpdate.ToolProgress("finished", "shell"));
        var final = JsonSerializer.SerializeToElement(request.WithSources(new { done = true }));
        var step = Assert.Single(final.GetProperty("knowledge_tool_steps").EnumerateArray());
        Assert.Equal("shell", step.GetProperty("tool").GetString());
        Assert.StartsWith("3915888\n", step.GetProperty("output").GetString());
        Assert.Equal(510, Encoding.UTF8.GetByteCount(step.GetProperty("output").GetString()!));
        Assert.True(step.GetProperty("outputTruncated").GetBoolean());
        Assert.DoesNotContain("BAD", step.GetProperty("output").GetString());
        Assert.DoesNotContain("imaginary", final.ToString());
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void AdaptersResolveWithOptionalKnowledgeWithoutChangingExistingConstructors(bool withStore)
    {
        var services = new ServiceCollection();
        services.AddLogging(); services.AddSingleton<ModelService>();
        services.AddSingleton(ServerOptionsBuilder.Build(new[] { "--no-skills" }, _root));
        services.AddSingleton(new UploadStoragePolicy(Path.Combine(_root, "uploads")));
        services.AddSingleton(new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() }));
        services.AddSingleton(new SessionWorkspaceManager(Path.Combine(_root, "workspaces")));
        services.AddSingleton<ICodeRunner>(_ => null!);
        services.AddSingleton<IResponsesStore>(_responses);
        if (withStore) services.AddSingleton(_knowledge);
        services.AddSingleton<OpenAIChatAdapter>(); services.AddSingleton<OpenAIResponsesAdapter>(); services.AddSingleton<OllamaAdapter>();
        using var provider = services.BuildServiceProvider();
        Assert.NotNull(provider.GetRequiredService<OpenAIChatAdapter>());
        Assert.NotNull(provider.GetRequiredService<OpenAIResponsesAdapter>());
        Assert.NotNull(provider.GetRequiredService<OllamaAdapter>());
    }

    [Theory]
    [InlineData("chat")] [InlineData("responses")] [InlineData("ollama")] [InlineData("generate")]
    public async Task InvalidOrUnavailableSelectionIsJson400BeforeStreaming(string protocol)
    {
        var document = await Import();
        foreach (object? selection in new object?[] { null, "all", new object?[] { null }, new[] { new string('a', 32) } })
        {
            var request = Request(protocol, true, document.CollectionId);
            request["knowledgeCollections"] = selection;
            var response = await Invoke(protocol, request);
            Assert.Equal(400, response.Context.Response.StatusCode);
            Assert.StartsWith("application/json", response.Context.Response.ContentType);
            Assert.True(JsonSerializer.Deserialize<JsonElement>(response.Text).TryGetProperty("error", out _));
        }
        var unavailable = await Invoke(protocol, Request(protocol, true, document.CollectionId), withStore: false);
        Assert.Equal(400, unavailable.Context.Response.StatusCode);
        Assert.Contains("no personal knowledge library", unavailable.Text);
    }

    private sealed class ScriptModel : ModelBase
    {
        private const string AssistantMarker = "<SPECIAL_11>Assistant\n";
        private const string Template = "{{ '<SPECIAL_10>System\\n' }}{% for message in messages %}{{ message.role + ': ' + message.content + '\\n' }}{% endfor %}{{ '<SPECIAL_11>Assistant\\n' }}";
        private readonly string[] _scripts;
        private readonly StringBuilder _seen = new();
        private string _script = "";
        public List<string> Prompts { get; } = new();
        public ScriptModel(string path, string[] scripts) : base(path, BackendType.Cpu)
        {
            _scripts = scripts;
            Config = new ModelConfig { Architecture = "nemotron_h", VocabSize = 128, NumLayers = 1, ChatTemplate = Template };
            Tokenizer = new CharTokenizer(); _maxContextLength = 32768;
        }
        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override string KVStateFingerprint => "knowledge-protocol";
        public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
        public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
        public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;
        protected override float[] ForwardCore(int[] tokens)
        {
            foreach (int token in tokens) _seen.Append((char)token);
            string seen = _seen.ToString();
            int start = seen.LastIndexOf(AssistantMarker, StringComparison.Ordinal);
            if (start >= 0) start += AssistantMarker.Length;
            else { start = seen.LastIndexOf("\nANSWER:", StringComparison.Ordinal); if (start >= 0) start += "\nANSWER:".Length; }
            var logits = Enumerable.Repeat(-1000f, 128).ToArray();
            if (start < 0) { logits[1] = 1000; return logits; }
            if (seen.AsSpan(start).StartsWith("<think></think>")) start += "<think></think>".Length;
            int at = seen.Length - start;
            if (at == 0) { _script = _scripts[Math.Min(Prompts.Count, _scripts.Length - 1)]; Prompts.Add(seen); }
            logits[at < _script.Length ? _script[at] : 1] = 1000;
            return logits;
        }
        protected override void ResetKVCacheCore() => _seen.Clear();
    }

    private sealed class CharTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 128).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => 128;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 1 };
        public bool IsEos(int id) => id == 1;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => c < 128 ? (int)c : '?').ToList();
        public string Decode(List<int> tokens) => new(tokens.Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }
}
