using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server.Endpoints;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>Real bundle, HTTP routes and progressive disclosure; model quality is a separate live probe.</summary>
public sealed class TopSkillsIntegrationTests
{
    private static string Repository
    {
        get
        {
            DirectoryInfo? directory = new(AppContext.BaseDirectory);
            while (directory != null && !File.Exists(Path.Combine(directory.FullName, "TensorSharp.sln")))
                directory = directory.Parent;
            // Some checkouts use the XML solution format instead.
            if (directory == null)
            {
                directory = new DirectoryInfo(AppContext.BaseDirectory);
                while (directory != null && !Directory.Exists(Path.Combine(directory.FullName, "TensorAgent", "skills")))
                    directory = directory.Parent;
            }
            return directory?.FullName ?? throw new InvalidOperationException("Cannot find the repository skill bundle.");
        }
    }

    private static string Bundle => Path.Combine(Repository, "TensorAgent", "skills");
    private static readonly string[] Imported = ["find-skills", "grill-me", "grill-with-docs", "tdd", "handoff"];
    private static SkillRegistry Registry() => new(new SkillRegistryOptions { Roots = [Bundle] });
    private static ServerHostingOptions Options(string directory) => ServerOptionsBuilder.Build(
        ["--backend", "ggml_cpu", "--no-multi-agent", "--skills-dir", Bundle], directory);

    [Fact]
    public async Task NonstreamingCompletionReturnsJsonContentTypeOverRealHttp()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-completion-json-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string modelPath = Path.Combine(root, "probe.gguf");
            BackendFailureWarmupTests.WriteProbeGguf(modelPath);
            using var model = new ModelService(NullLogger<ModelService>.Instance,
                (path, _, _, _) => new JsonProbeModel(path));
            model.EngineHost.SchedulerConfigOverride = new SchedulerConfig
            {
                BlockSize = 16, NumBlocks = 4096, MaxNumRunningSequences = 2,
                MaxNumBatchedTokens = 4096, MaxPrefillChunkSize = 4096, SoloPrefillChunkSize = 4096,
                EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
            };
            model.LoadModel(modelPath, null, "cpu");
            ServerHostingOptions options = ServerOptionsBuilder.Build(
                ["--model", modelPath, "--backend", "cpu", "--no-multi-agent", "--no-skills"], root);
            var adapter = new OpenAIChatAdapter(model, options, new UploadStoragePolicy(root), Registry(), null,
                new SessionWorkspaceManager(Path.Combine(root, "workspaces")), NullLoggerFactory.Instance);
            var builder = WebApplication.CreateBuilder(new WebApplicationOptions { EnvironmentName = "Production" });
            builder.Logging.ClearProviders();
            builder.WebHost.UseUrls("http://127.0.0.1:0");
            await using var app = builder.Build();
            app.MapPost("/v1/chat/completions", (HttpContext context) => adapter.ChatCompletionsAsync(context));
            await app.StartAsync();
            string address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.Single();
            using var client = new HttpClient { BaseAddress = new Uri(address) };
            using HttpResponseMessage response = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "probe.gguf", stream = false, think = false, multi_agent = false, max_tokens = 32,
                messages = new[] { new { role = "user", content = "Say ready." } },
            });
            response.EnsureSuccessStatusCode();
            Assert.Equal("application/json", response.Content.Headers.ContentType!.MediaType);
            using JsonDocument body = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            Assert.Equal("chat.completion", body.RootElement.GetProperty("object").GetString());
            Assert.Equal("ready", body.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!.Trim());
            await app.StopAsync();
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    // Deterministic token logits exercise the scheduler/adapter transport. This is
    // deliberately distinct from the separately recorded real-model quality tasks.
    private sealed class JsonProbeModel : ModelBase
    {
        private const string Marker = "<SPECIAL_11>Assistant\n";
        private readonly StringBuilder _seen = new();
        public JsonProbeModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "nemotron_h", VocabSize = 128, NumLayers = 1,
                ChatTemplate = "{{ '<SPECIAL_10>System\\n' }}{% for message in messages %}{{ message.role + ': ' + message.content + '\\n' }}{% endfor %}{{ '<SPECIAL_11>Assistant\\n' }}" };
            Tokenizer = new JsonProbeTokenizer();
            _maxContextLength = 32768;
        }
        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override string KVStateFingerprint => "json-content-type-probe";
        public override long ComputeKVBlockByteSize(int count) => 4L * count;
        public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
        public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;
        protected override float[] ForwardCore(int[] tokens)
        {
            foreach (int token in tokens) _seen.Append((char)token);
            string seen = _seen.ToString();
            int start = seen.LastIndexOf(Marker, StringComparison.Ordinal);
            var logits = Enumerable.Repeat(-1000f, 128).ToArray();
            if (start < 0) { logits[1] = 1000; return logits; }
            start += Marker.Length;
            if (seen.AsSpan(start).StartsWith("<think></think>")) start += "<think></think>".Length;
            int at = seen.Length - start;
            const string answer = "ready";
            logits[at < answer.Length ? answer[at] : 1] = 1000;
            return logits;
        }
        protected override void ResetKVCacheCore() => _seen.Clear();
    }

    private sealed class JsonProbeTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 128).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => 128;
        public int BosTokenId => -1;
        public int[] EosTokenIds => [1];
        public bool IsEos(int id) => id == 1;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => c < 128 ? (int)c : '?').ToList();
        public string Decode(List<int> tokens) => new(tokens.Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }

    [Fact]
    public void ImportedBundlesMatchTheirPinnedProvenanceAndHaveNoUnreachableDependencies()
    {
        SkillRegistry registry = Registry();
        using JsonDocument provenance = JsonDocument.Parse(File.ReadAllText(Path.Combine(Bundle, "top-skills-provenance.json")));
        Assert.Equal(Imported.Order(), provenance.RootElement.GetProperty("installed").EnumerateArray().Select(e => e.GetProperty("name").GetString()).Order());
        foreach (JsonElement record in provenance.RootElement.GetProperty("installed").EnumerateArray())
        {
            string name = record.GetProperty("name").GetString()!;
            Assert.True(registry.TryGet(name, out Skill skill));
            Assert.Empty(skill.Manifest.Warnings);
            Assert.Equal("MIT", skill.Manifest.License);
            Assert.Equal(record.GetProperty("revision").GetString(), skill.Manifest.Metadata["revision"]);
            Assert.DoesNotContain("Skill tool", skill.Manifest.Body, StringComparison.Ordinal);
            foreach (JsonProperty file in record.GetProperty("integrated_files").EnumerateObject())
            {
                Assert.True(skill.TryReadResource(file.Name, 48 * 1024, 0, out _, out string? error), error);
                string hash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.Combine(skill.RootDirectory, file.Name)))).ToLowerInvariant();
                Assert.Equal(file.Value.GetString(), hash);
            }
        }
        // These stubs require external executables or mutate toolchain configuration;
        // a valid Markdown manifest alone is not a successful capability check.
        foreach (string excluded in new[] { "agent-browser", "setup-matt-pocock-skills", "hyperframes-cli", "improve-codebase-architecture" })
            Assert.False(registry.TryGet(excluded, out _));
    }

    [Theory]
    [InlineData("grill-me", "SKILL.md")]
    [InlineData("grill-with-docs", "GLOSSARY-FORMAT.md")]
    [InlineData("tdd", "tests.md")]
    [InlineData("handoff", "SKILL.md")]
    public async Task ExplicitSelectionDisclosesRealResourcesWithDiscoveryDisabled(string name, string resource)
    {
        SkillRequestPlan plan = SkillRequestPlan.Create(Registry(), [name], false, null, "gemma4", 8192,
            Options(Path.GetTempPath()), out IReadOnlyList<string> unknown);
        Assert.Empty(unknown);
        Assert.Single(plan.ToolContext.Reachable);
        int calls = 0;
        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            [new ChatMessage { Role = "user", Content = "Use the selected workflow." }], plan.Tools,
            plan.ToolContext, (messages, _, _) =>
            {
                calls++;
                if (calls == 1)
                    return Task.FromResult(new SkillTurnOutput(new ParsedOutput
                    {
                        ToolCalls = [new ToolCall { Name = SkillTools.ReadToolName,
                            Arguments = new Dictionary<string, object> { ["skill"] = name, ["path"] = resource } }],
                    }));
                Assert.Contains(messages, m => m.Role == "tool" && m.Content.Contains(resource, StringComparison.Ordinal));
                return Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = "Workflow resource received." }));
            }, plan.LoopOptions);
        Assert.Equal(2, result.Rounds);
        Assert.True(Assert.Single(result.Invocations).Ok);
        Assert.False(result.HitRoundLimit);
        Assert.Equal("Workflow resource received.", result.Output.Parsed!.Content);
    }

    [Fact]
    public async Task RealHttpCatalogAndResourceRoutesExposeTheSharedBundle()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-top-skills-http-" + Guid.NewGuid().ToString("N"));
        try
        {
            var builder = WebApplication.CreateBuilder(new WebApplicationOptions { EnvironmentName = "Production" });
            builder.Logging.ClearProviders();
            builder.WebHost.UseUrls("http://127.0.0.1:0");
            builder.Services.AddSingleton(Registry());
            builder.Services.AddSingleton(Options(root));
            builder.Services.AddSingleton(new UploadStoragePolicy(root));
            builder.Services.AddSingleton<SkillsAdapter>();
            await using var app = builder.Build();
            app.MapSkillEndpoints();
            await app.StartAsync();
            string address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.Single();
            using var client = new HttpClient { BaseAddress = new Uri(address) };
            using JsonDocument catalog = JsonDocument.Parse(await client.GetStringAsync("/v1/skills"));
            string[] ids = catalog.RootElement.GetProperty("data").EnumerateArray().Select(e => e.GetProperty("id").GetString()!).ToArray();
            foreach (string name in Imported)
            {
                Assert.Contains(name, ids);
                using JsonDocument detail = JsonDocument.Parse(await client.GetStringAsync("/api/skills/" + name));
                Assert.Equal(name, detail.RootElement.GetProperty("id").GetString());
            }
            using HttpResponseMessage reference = await client.GetAsync("/api/skills/grill-me/files/SKILL.md");
            reference.EnsureSuccessStatusCode();
            Assert.Equal("text/plain", reference.Content.Headers.ContentType!.MediaType);
            Assert.Contains("frontier", await reference.Content.ReadAsStringAsync(), StringComparison.OrdinalIgnoreCase);
            await app.StopAsync();
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }
}
