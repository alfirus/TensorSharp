// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using TensorSharp.AgentHost.Knowledge;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Knowledge;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

public sealed class CompleteKnowledgeOriginalAccessTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "complete-original-" + Guid.NewGuid().ToString("N"));
    private readonly KnowledgeStore _store;
    public CompleteKnowledgeOriginalAccessTests() => _store = new KnowledgeStore(_root, new BuiltInKnowledgeExtractor());
    public void Dispose() { _store.Dispose(); Directory.Delete(_root, true); }

    private async Task<(KnowledgeToolScope Scope, KnowledgeDocument Document)> Prepare()
    {
        var collection = await _store.CreateCollectionAsync("selected");
        var document = await _store.ImportAsync(collection.Id, "scores.csv",
            new MemoryStream(Encoding.UTF8.GetBytes("score\n95\n72\n")));
        var scope = new KnowledgeToolScope(_store, new[] { collection.Id });
        var hit = (await _store.SearchAsync("score", new() { CollectionIds = scope.CollectionIds }))[0];
        var complete = _store.ReadDocumentContext(hit.Reference, collectionIds: scope.CollectionIds)!;
        Assert.True(KnowledgeTools.TrySerializeCompleteDocumentEvidence(complete, 64000, out string payload));
        scope.RecordCompleteDocumentEvidence(complete, payload);
        Assert.True(scope.Retrieval.ShouldSynthesize);
        return (scope, document);
    }

    private static ToolCall Open(KnowledgeDocument document) => new()
    {
        Name = KnowledgeTools.Open, Arguments = new()
        { ["collection_id"] = document.CollectionId, ["document_id"] = document.Id, ["version"] = document.Version },
    };

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public async Task CompleteSnapshotAllowsOriginalThenCodeButRepeatedOpenStillTerminates(bool streaming, bool repeatOpen)
    {
        var (scope, document) = await Prepare();
        var workspace = new SessionWorkspaceManager(Path.Combine(_root, "workspaces")).GetOrCreate("calculation");
        var runner = new CsvCalculationRunner();
        int generations = 0;
        ParsedOutput Next(List<ChatMessage> messages, List<ToolFunction>? tools)
        {
            generations++;
            Assert.DoesNotContain(tools!, tool => tool.Name is KnowledgeTools.Search or KnowledgeTools.Read or KnowledgeTools.List);
            if (repeatOpen)
            {
                if (generations <= 5)
                {
                    Assert.Equal(generations <= 4, tools!.Any(tool => tool.Name == KnowledgeTools.Open));
                    return new() { ToolCalls = new() { Open(document) } };
                }
                Assert.Equal(6, generations);
                Assert.Empty(tools!);
                Assert.False(scope.Retrieval.CanAccessOriginal);
                return new() { Content = "Answer from the available source." };
            }
            Assert.Contains(tools!, tool => tool.Name == KnowledgeTools.Open);
            Assert.Contains(tools!, tool => tool.Name == SkillToolNames.Shell);
            if (generations == 1) return new() { ToolCalls = new() { Open(document) } };
            if (generations == 2)
            {
                var opened = Assert.Single(messages, message => message.Content?.Contains("\"path\":\"knowledge-") == true);
                string path = JsonDocument.Parse(opened.Content).RootElement.GetProperty("path").GetString()!;
                Assert.Equal("score\n95\n72\n", File.ReadAllText(Path.Combine(workspace.WorkDirectory, path)));
                return new() { ToolCalls = new() { new() { Name = SkillToolNames.Shell, Arguments = new() { ["command"] = path } } } };
            }
            Assert.Equal(3, generations);
            Assert.Contains(messages, message => message.Content == "23");
            return new() { Content = "The computed difference is 23." };
        }
        if (streaming)
        {
            var options = ServerOptionsBuilder.Build(new[] { "--model", "x.gguf" }, _root);
            var plan = SkillRequestPlan.Create(null, Array.Empty<string>(), false, null, "nemotron_h_moe",
                32768, options, out _, knowledge: scope, multiAgent: false, codeRunner: runner, workspace: workspace)!;
            async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
                [EnumeratorCancellation] CancellationToken cancellationToken)
            {
                var output = Next(messages, tools);
                yield return ChatStreamUpdate.Text("<think></think>" + (output.ToolCalls?.Count > 0
                    ? "<tool_call>" + JsonSerializer.Serialize(new { name = output.ToolCalls[0].Name, arguments = output.ToolCalls[0].Arguments }) + "</tool_call>"
                    : output.Content));
                await Task.Yield();
                yield return new ChatStreamUpdate("", true, 10, 20, 5, 0, 0, 0, "stop");
            }
            await foreach (var _ in SkillChatLoop.RunAsync("nemotron_h_moe", new(), plan, true, Generate, null, CancellationToken.None)) { }
        }
        else
        {
            var result = await SkillAgentLoop.RunAsync(new(), KnowledgeTools.Merge(new[] { runner.Declare() }),
                new SkillToolContext(Array.Empty<Skill>()) { Knowledge = scope, Workspace = workspace, CodeRunner = runner },
                (messages, tools, cancellationToken) => Task.FromResult(new SkillTurnOutput(Next(messages, tools))),
                new() { MaxRounds = 64, ClientTools = Array.Empty<ToolFunction>() });
            Assert.False(result.HitRoundLimit);
        }
        Assert.Equal(repeatOpen ? 6 : 3, generations);
        Assert.Equal(repeatOpen ? 0 : 1, runner.Executions);
        Assert.Single(Directory.GetFiles(workspace.WorkDirectory, "knowledge-*"));
    }

    [Fact]
    public async Task CompleteOriginalAccessNeedsCodeWorkspaceAndKeepsTheSixteenCallCeiling()
    {
        var (scope, document) = await Prepare();
        Assert.Contains("alreadyRetrieved", KnowledgeTools.Execute(Open(document), scope).Content);
        var workspace = new SessionWorkspaceManager(Path.Combine(_root, "workspaces")).GetOrCreate("limit");
        for (int index = 0; index < 16; index++)
        {
            var original = index == 0 ? document : await _store.ImportAsync(document.CollectionId, $"data{index}.txt",
                new MemoryStream(Encoding.UTF8.GetBytes($"value {index}")));
            Assert.True(scope.Retrieval.CanAccessOriginal);
            Assert.Contains("\"path\":\"knowledge-", KnowledgeTools.Execute(Open(original), scope, workspace).Content);
        }
        Assert.False(scope.Retrieval.CanAccessOriginal);
        Assert.Contains("alreadyRetrieved", KnowledgeTools.Execute(Open(document), scope, workspace).Content);
        Assert.Equal(16, Directory.GetFiles(workspace.WorkDirectory, "knowledge-*").Length);
    }

    [Fact]
    public async Task ExplicitTerminalRecoveryClosesOriginalAccessAndAllHostTools()
    {
        var (scope, document) = await Prepare();
        var workspace = new SessionWorkspaceManager(Path.Combine(_root, "workspaces")).GetOrCreate("terminal");
        var runner = new CsvCalculationRunner();
        var options = ServerOptionsBuilder.Build(new[] { "--model", "x.gguf" }, _root);
        var plan = SkillRequestPlan.Create(null, Array.Empty<string>(), false, null, "nemotron_h_moe",
            32768, options, out _, knowledge: scope, multiAgent: false, codeRunner: runner, workspace: workspace)!;
        scope.Retrieval.RequestSynthesis();
        plan.KnowledgeAnswerRecovery = true;
        Assert.False(scope.Retrieval.CanAccessOriginal);
        Assert.Contains("alreadyRetrieved", KnowledgeTools.Execute(Open(document), scope, workspace).Content);
        async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            Assert.Empty(tools);
            yield return ChatStreamUpdate.Text("<think></think>Answer from the indexed document.");
            await Task.Yield();
            yield return new ChatStreamUpdate("", true, 10, 20, 5, 0, 0, 0, "stop");
        }
        await foreach (var _ in SkillChatLoop.RunAsync("nemotron_h_moe", new(), plan, true, Generate, null, CancellationToken.None)) { }
        Assert.Empty(Directory.GetFiles(workspace.WorkDirectory, "knowledge-*"));
        Assert.Equal(0, runner.Executions);
    }

    private sealed class CsvCalculationRunner : ICodeRunner
    {
        public bool CanRun => true;
        public string? UnavailableReason => null;
        public bool CanInstallPackages => false;
        public int Executions { get; private set; }
        public ToolFunction Declare() => new() { Name = SkillToolNames.Shell, Description = "Calculate from the original CSV" };
        public string InstallPackages(string language, IReadOnlyList<string> packages, SessionWorkspace workspace,
            Action<string>? onOutput = null) => throw new NotSupportedException();
        public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null, SessionWorkspace? workspace = null, IReadOnlyList<string>? skillDirectories = null)
        {
            Assert.True(workspace!.TryResolve(call.Arguments["command"].ToString()!, out string path, out _));
            int[] scores = File.ReadAllLines(path).Skip(1).Select(int.Parse).ToArray();
            Executions++;
            return SkillToolResult.Success((scores.Max() - scores.Min()).ToString());
        }
    }
}
