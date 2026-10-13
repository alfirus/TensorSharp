// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using TensorSharp.AgentHost.Knowledge;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Knowledge;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Knowledge;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

public sealed class KnowledgeRetrievalLoopTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "knowledge-loop-" + Guid.NewGuid().ToString("N"));
    private readonly KnowledgeStore _store;
    public KnowledgeRetrievalLoopTests() => _store = new KnowledgeStore(_root, new BuiltInKnowledgeExtractor());
    public void Dispose() { _store.Dispose(); Directory.Delete(_root, true); }

    private async Task<KnowledgeToolScope> Prepare()
    {
        var collection = await _store.CreateCollectionAsync("selected");
        await _store.ImportAsync(collection.Id, "report.txt", new MemoryStream(Encoding.UTF8.GetBytes(
            "Alex Aurora cobalt orchard. Mathematics 95: strong reasoning. Writing 72: needs more detail.")));
        return new KnowledgeToolScope(_store, new[] { collection.Id });
    }

    private static ToolCall Search(string query = "Alex") => new()
    { Name = KnowledgeTools.Search, Arguments = new() { ["query"] = query } };

    [Fact]
    public async Task RepeatedCallsStopAfterThreeUnproductiveAttemptsAndKeepOriginalEvidence()
    {
        var scope = await Prepare();
        Assert.Contains("Mathematics 95", KnowledgeTools.Execute(Search(), scope).Content);
        for (int i = 0; i < 3; i++)
        {
            var repeated = KnowledgeTools.Execute(Search(" Alex "), scope);
            Assert.True(repeated.Ok);
            Assert.DoesNotContain("Mathematics 95", repeated.Content);
            Assert.Contains("alreadyRetrieved", repeated.Content);
        }
        Assert.True(scope.Retrieval.ShouldSynthesize);
        Assert.Contains("Mathematics 95", Assert.Single(scope.Sources).Text);
    }

    [Fact]
    public async Task DifferentQueriesReturningTheSameEvidenceAlsoStop()
    {
        var scope = await Prepare();
        foreach (string query in new[] { "Alex", "Aurora", "cobalt", "orchard" })
            Assert.True(KnowledgeTools.Execute(Search(query), scope).Ok);
        Assert.True(scope.Retrieval.ShouldSynthesize);
        Assert.Single(scope.Sources);
    }

    [Fact]
    public async Task ChildReceivesEvidenceEvenWhenParentAlreadyRetrievedIt()
    {
        var scope = await Prepare();
        var parent = new SkillToolContext(Array.Empty<Skill>()) { Knowledge = scope };
        Assert.Contains("Mathematics 95", SkillTools.Execute(Search(), parent).Content);
        using var child = AgentWorkspace.Create(parent, "knowledge-loop-child-" + Guid.NewGuid().ToString("N"));
        Assert.Same(scope, child.Context.Knowledge);
        Assert.Contains("Mathematics 95", SkillTools.Execute(Search(), child.Context).Content);
        Assert.DoesNotContain("Mathematics 95", SkillTools.Execute(Search(), child.Context).Content);
        Assert.False(scope.Retrieval.ShouldSynthesize);
    }

    [Fact]
    public async Task ProvenanceRetainsEveryEmittedExcerptBeyondFifty()
    {
        var scope = await Prepare();
        var hit = (await _store.SearchAsync("Alex", new() { CollectionIds = scope.CollectionIds }))[0];
        scope.RecordSources(Enumerable.Range(0, 60).Select(i => hit with { Reference = $"kb://emitted/{i}", ChunkIndex = i }));
        Assert.Equal(60, scope.Sources.Count);
        Assert.Contains(scope.Sources, s => s.Reference == "kb://emitted/59");
        scope.RecordSources(new[] { hit with { Reference = "kb://emitted/59", ChunkIndex = 59, Text = hit.Text + " Expanded." } });
        Assert.EndsWith(" Expanded.", scope.Sources.Single(s => s.Reference == "kb://emitted/59").Text);
    }

    [Fact]
    public async Task NewEvidenceResetsStallCountAndExpandedTextIsProgress()
    {
        var scope = await Prepare();
        var hit = (await _store.SearchAsync("Alex", new() { CollectionIds = scope.CollectionIds }))[0];
        scope.RecordSources(new[] { hit with { Text = "Alex", Truncated = true } });
        Assert.True(KnowledgeTools.Execute(Search("nothingmatchingone"), scope).Ok);
        Assert.True(KnowledgeTools.Execute(Search("nothingmatchingtwo"), scope).Ok);
        var read = KnowledgeTools.Execute(new ToolCall
        { Name = KnowledgeTools.Read, Arguments = new() { ["reference"] = hit.Reference, ["adjacent_chunks"] = 0 } }, scope);
        Assert.Contains("Mathematics 95", read.Content);
        Assert.False(scope.Retrieval.ShouldSynthesize);
        Assert.True(KnowledgeTools.Execute(Search("nothingmatchingthree"), scope).Ok);
        Assert.True(KnowledgeTools.Execute(Search("nothingmatchingfour"), scope).Ok);
        Assert.False(scope.Retrieval.ShouldSynthesize);
        Assert.True(KnowledgeTools.Execute(Search("nothingmatchingfive"), scope).Ok);
        Assert.True(scope.Retrieval.ShouldSynthesize);
    }

    [Fact]
    public async Task LongComparisonsCanReadNewEvidenceButStillHaveARequestCeiling()
    {
        var scope = await Prepare();
        for (int i = 0; i < 18; i++)
            await _store.ImportAsync(scope.CollectionIds[0], $"subject{i}.txt",
                new MemoryStream(Encoding.UTF8.GetBytes($"subjecttoken{i} grade {i + 70}, feedback improving.")));
        for (int i = 0; i < 16; i++)
        {
            Assert.False(scope.Retrieval.ShouldSynthesize);
            Assert.Contains($"subjecttoken{i}", KnowledgeTools.Execute(Search($"subjecttoken{i}"), scope).Content);
        }
        Assert.True(scope.Retrieval.ShouldSynthesize);
        Assert.DoesNotContain("subjecttoken16", KnowledgeTools.Execute(Search("subjecttoken16"), scope).Content);
    }

    [Fact]
    public async Task CumulativeEvidenceCountsUtf8AndOnlyTheExpansionOfTheSameExcerpt()
    {
        var selected = await Prepare();
        var scope = new KnowledgeToolScope(_store, selected.CollectionIds, 4096);
        var hit = (await _store.SearchAsync("Alex", new() { CollectionIds = scope.CollectionIds }))[0];
        string initial = new('界', 2000);
        scope.RecordSources(new[] { hit with { Text = initial } });
        Assert.Equal(8192, scope.Retrieval.MaxEvidenceBytes);
        Assert.Equal(6000, scope.Retrieval.EvidenceBytes);
        scope.RecordSources(new[] { hit with { Text = initial }, hit with { Text = initial[..1000] } });
        Assert.Equal(6000, scope.Retrieval.EvidenceBytes);
        Assert.False(scope.Retrieval.ShouldSynthesize);
        string expanded = initial + string.Concat(Enumerable.Repeat("😀", 600));
        scope.RecordSources(new[] { hit with { Text = expanded } });
        Assert.Equal(8400, scope.Retrieval.EvidenceBytes);
        Assert.True(scope.Retrieval.ShouldSynthesize);
    }

    [Fact]
    public async Task RenderedEvidenceChargesFullExpandedReadButNotDuplicateNoticesOrReplay()
    {
        var selected = await Prepare();
        foreach (string name in new[] { "first", "second" })
            await _store.ImportAsync(selected.CollectionIds[0], name + ".txt",
                new MemoryStream(Encoding.UTF8.GetBytes("sharedsubject " + new string('界', 1100))));
        var scope = new KnowledgeToolScope(_store, selected.CollectionIds, 4096);
        string search = KnowledgeTools.Execute(Search("sharedsubject"), scope).Content;
        Assert.Equal(Encoding.UTF8.GetByteCount(search), scope.Retrieval.EmittedEvidenceBytes);
        var excerpt = scope.Sources.First(s => s.Truncated);
        long priorNovel = scope.Retrieval.EvidenceBytes;
        long priorRendered = scope.Retrieval.EmittedEvidenceBytes;
        var call = new ToolCall { Name = KnowledgeTools.Read,
            Arguments = new() { ["reference"] = excerpt.Reference, ["adjacent_chunks"] = 0 } };
        string read = KnowledgeTools.Execute(call, scope).Content;
        Assert.Contains("sharedsubject", read);
        Assert.Equal(priorRendered + Encoding.UTF8.GetByteCount(read), scope.Retrieval.EmittedEvidenceBytes);
        long novelIncrement = scope.Retrieval.EvidenceBytes - priorNovel;
        Assert.True(novelIncrement > 0);
        Assert.True(novelIncrement < Encoding.UTF8.GetByteCount(read));
        long delivered = scope.Retrieval.EmittedEvidenceBytes;
        Assert.Contains("alreadyRetrieved", KnowledgeTools.Execute(call, scope).Content);
        scope.GetBoundedEvidence();
        KnowledgeTools.BuildSynthesisMessages(Array.Empty<ChatMessage>(), scope);
        Assert.Equal(delivered, scope.Retrieval.EmittedEvidenceBytes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task EagerAndRenderedResponsesCapEvidenceWithinTheSameBatch(bool streaming)
    {
        var selected = await Prepare();
        var scope = new KnowledgeToolScope(_store, selected.CollectionIds, 2048);
        foreach (string name in new[] { "firstsubject", "secondsubject", "latersubject" })
            await _store.ImportAsync(scope.CollectionIds[0], name + ".txt",
                new MemoryStream(Encoding.UTF8.GetBytes(name + " " + new string('界', 1100))));
        var eager = await KnowledgeRequestContext.PrepareAsync(new()
            { new() { Role = "user", Content = "Compare Alex" } }, scope, toolsAvailable: true);
        string eagerContent = eager.Last(m => m.Role == "user").Content;
        int start = eagerContent.IndexOf('\n') + 1;
        int end = eagerContent.IndexOf("\nEnd of reference data.", StringComparison.Ordinal);
        Assert.Equal(Encoding.UTF8.GetByteCount(eagerContent[start..end]), scope.Retrieval.EmittedEvidenceBytes);
        Assert.Equal(4096, scope.Retrieval.MaxEmittedEvidenceBytes);
        int generations = 0;
        ParsedOutput Next(List<ChatMessage> messages, List<ToolFunction>? tools)
        {
            generations++;
            if (generations == 1) return new() { ToolCalls = new()
                { Search("firstsubject"), Search("secondsubject"), Search("latersubject") } };
            Assert.Empty(tools!);
            Assert.Contains("firstsubject", messages[^1].Content);
            Assert.Contains("secondsubject", messages[^1].Content);
            Assert.DoesNotContain("latersubject", messages[^1].Content);
            return new() { Content = "The supported answer." };
        }
        if (streaming)
        {
            var options = ServerOptionsBuilder.Build(new[] { "--model", "x.gguf" }, _root);
            var plan = SkillRequestPlan.Create(null, Array.Empty<string>(), false, null, "nemotron_h_moe",
                32768, options, out _, knowledge: scope, multiAgent: false)!;
            async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
                [EnumeratorCancellation] CancellationToken ct)
            {
                var next = Next(messages, tools);
                yield return ChatStreamUpdate.Text("<think></think>" + (next.ToolCalls?.Count > 0
                    ? string.Concat(next.ToolCalls.Select(call => "<tool_call>" + JsonSerializer.Serialize(new { name = call.Name, arguments = call.Arguments }) + "</tool_call>"))
                    : next.Content));
                await Task.Yield();
                yield return new ChatStreamUpdate("", true, 10, 20, 5, 0, 0, 0, "stop");
            }
            await foreach (var _ in SkillChatLoop.RunAsync("nemotron_h_moe", eager, plan, true, Generate, null, CancellationToken.None)) { }
        }
        else
        {
            var result = await SkillAgentLoop.RunAsync(eager, KnowledgeTools.Merge(null),
                new SkillToolContext(Array.Empty<Skill>()) { Knowledge = scope },
                (messages, tools, ct) => Task.FromResult(new SkillTurnOutput(Next(messages, tools))));
            Assert.Equal("The supported answer.", result.Output.Parsed!.Content);
        }
        Assert.Equal(2, generations);
        Assert.True(scope.Retrieval.EvidenceBytes < scope.Retrieval.MaxEvidenceBytes);
        Assert.True(scope.Retrieval.EmittedEvidenceBytes >= scope.Retrieval.MaxEmittedEvidenceBytes);
        Assert.True(scope.Retrieval.EmittedEvidenceBytes < scope.Retrieval.MaxEmittedEvidenceBytes + scope.MaxCharacters);
        Assert.DoesNotContain(scope.Sources, source => source.FileName == "latersubject.txt");
    }

    [Fact]
    public async Task ExhaustedParentEvidenceBudgetDoesNotPreventAChildFromReceivingEvidence()
    {
        var selected = await Prepare();
        var scope = new KnowledgeToolScope(_store, selected.CollectionIds, 1024);
        var hit = (await _store.SearchAsync("Alex", new() { CollectionIds = scope.CollectionIds }))[0];
        scope.RecordSources(new[] { hit with { Text = new string('a', 4096) } });
        Assert.True(scope.Retrieval.ShouldSynthesize);
        using var child = AgentWorkspace.Create(new SkillToolContext(Array.Empty<Skill>()) { Knowledge = scope },
            "knowledge-budget-child-" + Guid.NewGuid().ToString("N"));
        Assert.Equal(4096, child.Context.KnowledgeRetrieval!.MaxEvidenceBytes);
        Assert.False(child.Context.KnowledgeRetrieval.ShouldSynthesize);
        // Independent budget accounting remains independent even when the root is at its ceiling.
        scope.RecordSources(new[] { hit }, child.Context.KnowledgeRetrieval);
        Assert.Equal(Encoding.UTF8.GetByteCount(hit.Text), child.Context.KnowledgeRetrieval.EvidenceBytes);
        Assert.Equal(4096, scope.Retrieval.EvidenceBytes);
        Assert.Contains("Mathematics 95", scope.GetBoundedEvidence(child.Context.KnowledgeRetrieval));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task CrossingEvidenceBudgetSkipsRemainingSearchesInTheSameBatch(bool streaming)
    {
        var selected = await Prepare();
        var scope = new KnowledgeToolScope(_store, selected.CollectionIds, 2048);
        var hit = (await _store.SearchAsync("Alex", new() { CollectionIds = scope.CollectionIds }))[0];
        // Model evidence received in earlier steps is already close to the request's
        // cumulative budget; one more useful result must finish the retrieval phase.
        scope.RecordSources(new[] { hit with { Reference = "kb://earlier/evidence", Text = new string('e', 4064) } });
        var later = await _store.ImportAsync(scope.CollectionIds[0], "later.txt",
            new MemoryStream(Encoding.UTF8.GetBytes("laterquery This document must not be retrieved after the budget is reached.")));
        int generations = 0;
        ParsedOutput Next(List<ChatMessage> messages, List<ToolFunction>? tools)
        {
            generations++;
            if (generations == 1)
            {
                Assert.False(scope.Retrieval.ShouldSynthesize);
                return new() { ToolCalls = new() { Search(), Search("laterquery"), Search("different query") } };
            }
            Assert.Empty(tools!);
            Assert.True(scope.Retrieval.ShouldSynthesize);
            Assert.Contains("Mathematics 95", messages[^1].Content);
            return new() { Content = "The supported answer." };
        }
        if (streaming)
        {
            var options = ServerOptionsBuilder.Build(new[] { "--model", "x.gguf" }, _root);
            var plan = SkillRequestPlan.Create(null, Array.Empty<string>(), false, null, "nemotron_h_moe",
                32768, options, out _, knowledge: scope, multiAgent: false)!;
            async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
                [EnumeratorCancellation] CancellationToken ct)
            {
                var next = Next(messages, tools);
                string text = "<think></think>" + (next.ToolCalls?.Count > 0
                    ? string.Concat(next.ToolCalls.Select(call => "<tool_call>" + JsonSerializer.Serialize(new { name = call.Name, arguments = call.Arguments }) + "</tool_call>"))
                    : next.Content);
                yield return ChatStreamUpdate.Text(text);
                await Task.Yield();
                yield return new ChatStreamUpdate("", true, 10, 20, 5, 0, 0, 0, "stop");
            }
            await foreach (var _ in SkillChatLoop.RunAsync("nemotron_h_moe", new(), plan, true, Generate, null, CancellationToken.None)) { }
        }
        else
        {
            var result = await SkillAgentLoop.RunAsync(new(), KnowledgeTools.Merge(null),
                new SkillToolContext(Array.Empty<Skill>()) { Knowledge = scope },
                (messages, tools, ct) => Task.FromResult(new SkillTurnOutput(Next(messages, tools))));
            Assert.Equal("The supported answer.", result.Output.Parsed!.Content);
        }
        Assert.Equal(2, generations);
        Assert.DoesNotContain(scope.Sources, source => source.DocumentId == later.Id);
        Assert.True(scope.Retrieval.EvidenceBytes >= scope.Retrieval.MaxEvidenceBytes);
        Assert.True(scope.Retrieval.EvidenceBytes < scope.Retrieval.MaxEvidenceBytes + scope.MaxCharacters);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task AgentLoopSwitchesToSynthesisWithEvidenceAndNoHostTools(bool toolRolesRendered)
    {
        var scope = await Prepare();
        var context = new SkillToolContext(Array.Empty<Skill>()) { Knowledge = scope };
        int generations = 0;
        Task<SkillTurnOutput> Generate(List<ChatMessage> messages, List<ToolFunction>? tools, CancellationToken ct)
        {
            generations++;
            if (tools?.Count > 0)
                return Task.FromResult(new SkillTurnOutput(new ParsedOutput { ToolCalls = new() { Search() } }));
            Assert.Contains(messages, m => m.Content?.Contains("Mathematics 95") == true);
            Assert.Contains(KnowledgeRetrievalProgress.SynthesisInstruction, messages[^1].Content);
            Assert.Equal("user", messages[^1].Role);
            Assert.IsAssignableFrom<HostContinuationMessage>(messages[^1]);
            return Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = "Mathematics is stronger; writing needs more detail." }));
        }
        var result = await SkillAgentLoop.RunAsync(new() { new() { Role = "user", Content = "Compare Alex's grades and comments." } },
            KnowledgeTools.Merge(null), context, Generate,
            new() { MaxRounds = 64, ToolResultsAreRendered = toolRolesRendered, ClientTools = Array.Empty<ToolFunction>() });
        Assert.Equal(5, generations);
        Assert.Equal(5, result.Rounds);
        Assert.False(result.HitRoundLimit);
        Assert.Equal("Mathematics is stronger; writing needs more detail.", result.Output.Parsed!.Content);
        Assert.Equal(4, result.Invocations.Count);
    }

    [Fact]
    public async Task StreamingLoopSynthesizesAfterStallAndStreamsTheAnswer()
    {
        var scope = await Prepare();
        var options = ServerOptionsBuilder.Build(new[] { "--model", "x.gguf", "--skills-max-rounds", "64" }, _root);
        var plan = SkillRequestPlan.Create(null, Array.Empty<string>(), false, null, "nemotron_h_moe",
            32768, options, out var unknown, knowledge: scope, multiAgent: false);
        Assert.Empty(unknown);
        Assert.NotNull(plan);
        int generations = 0;
        async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
            [EnumeratorCancellation] CancellationToken ct)
        {
            generations++;
            string response;
            if (tools.Count > 0) response = "<think></think><tool_call>{\"name\":\"knowledge_search\",\"arguments\":{\"query\":\"Alex\"}}</tool_call>";
            else
            {
                Assert.Contains(messages, m => m.Content?.Contains("Mathematics 95") == true);
                Assert.Equal("user", messages[^1].Role);
                Assert.IsAssignableFrom<HostContinuationMessage>(messages[^1]);
                response = "<think></think>Mathematics is stronger; writing needs more detail.";
            }
            foreach (char c in response)
            {
                ct.ThrowIfCancellationRequested();
                yield return ChatStreamUpdate.Text(c.ToString());
                await Task.Yield();
            }
            yield return new ChatStreamUpdate("", true, 10, 20, 5, 0, 0, 0, "stop");
        }
        var updates = new List<ChatStreamUpdate>();
        await foreach (var update in SkillChatLoop.RunAsync("nemotron_h_moe",
            new() { new() { Role = "user", Content = "Compare Alex's grades." } }, plan!, true, Generate, null, CancellationToken.None))
            updates.Add(update);
        Assert.Equal(5, generations);
        Assert.Equal("Mathematics is stronger; writing needs more detail.", string.Concat(updates.Where(u => !u.Done).Select(u => u.Piece)));
        Assert.True(updates.Count(u => !string.IsNullOrEmpty(u.Piece)) > 1);
        Assert.Single(updates, u => u.Done);
    }

    [Fact]
    public async Task FinalSynthesisPreservesClientOwnedTools()
    {
        var scope = await Prepare();
        var client = new ToolFunction { Name = "client_lookup", Description = "client" };
        int generations = 0;
        Task<SkillTurnOutput> Generate(List<ChatMessage> messages, List<ToolFunction>? tools, CancellationToken ct)
        {
            generations++;
            if (generations == 1) return Task.FromResult(new SkillTurnOutput(new ParsedOutput { ToolCalls = new() { Search() } }));
            Assert.Same(client, Assert.Single(tools!));
            return Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = "Answer" }));
        }
        var result = await SkillAgentLoop.RunAsync(new(), KnowledgeTools.Merge(new[] { client }),
            new SkillToolContext(Array.Empty<Skill>()) { Knowledge = scope }, Generate,
            new() { MaxRounds = 1, ClientTools = new[] { client } });
        Assert.Equal("Answer", result.Output.Parsed!.Content);
        Assert.True(result.HitRoundLimit);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RetrievalCompletionStillAllowsRequestedCodeWork(bool streaming)
    {
        var scope = await Prepare();
        var runner = new CalculationRunner();
        var workspace = new SessionWorkspaceManager(Path.Combine(_root, "workspaces")).GetOrCreate("calculation");
        var document = _store.ListDocuments(scope.CollectionIds[0]).Single();
        int generations = 0;
        ParsedOutput Next(List<ChatMessage> messages, List<ToolFunction>? tools)
        {
            generations++;
            if (generations == 1) return new() { ToolCalls = new() { new()
            {
                Name = KnowledgeTools.Open,
                Arguments = new() { ["collection_id"] = document.CollectionId, ["document_id"] = document.Id, ["version"] = document.Version },
            } } };
            if (generations <= 5) return new() { ToolCalls = new() { Search() } };
            Assert.DoesNotContain(tools!, t => KnowledgeTools.IsTool(t.Name));
            Assert.Contains(tools!, t => t.Name == SkillToolNames.Shell);
            if (generations == 6)
            {
                Assert.Contains(KnowledgeRetrievalProgress.CompleteTaskInstruction, messages[^1].Content);
                Assert.IsAssignableFrom<HostContinuationMessage>(messages[^1]);
                var opened = Assert.Single(messages, m => m.Content?.Contains("\"path\":\"knowledge-") == true);
                string originalPath = JsonDocument.Parse(opened.Content).RootElement.GetProperty("path").GetString()!;
                Assert.True(File.Exists(Path.Combine(workspace.WorkDirectory, originalPath)));
                return new() { ToolCalls = new() { new() { Name = SkillToolNames.Shell, Arguments = new() { ["command"] = "95 - 72" } } } };
            }
            Assert.Contains(messages, m => m.Content == "23");
            return new() { Content = "The computed grade difference is 23." };
        }
        if (streaming)
        {
            var options = ServerOptionsBuilder.Build(new[] { "--model", "x.gguf" }, _root);
            var plan = SkillRequestPlan.Create(null, Array.Empty<string>(), false, null, "nemotron_h_moe",
                32768, options, out _, knowledge: scope, multiAgent: false, codeRunner: runner, workspace: workspace)!;
            async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
                [EnumeratorCancellation] CancellationToken ct)
            {
                var next = Next(messages, tools);
                string text = "<think></think>" + (next.ToolCalls?.Count > 0
                    ? "<tool_call>" + JsonSerializer.Serialize(new { name = next.ToolCalls[0].Name, arguments = next.ToolCalls[0].Arguments }) + "</tool_call>"
                    : next.Content);
                yield return ChatStreamUpdate.Text(text);
                await Task.Yield();
                yield return new ChatStreamUpdate("", true, 10, 20, 5, 0, 0, 0, "stop");
            }
            var content = new StringBuilder();
            await foreach (var update in SkillChatLoop.RunAsync("nemotron_h_moe", new(), plan, true, Generate, null, CancellationToken.None))
                if (!update.Done) content.Append(update.Piece);
            Assert.Equal("The computed grade difference is 23.", content.ToString());
        }
        else
        {
            var result = await SkillAgentLoop.RunAsync(new(), KnowledgeTools.Merge(new[] { runner.Declare() }),
                new SkillToolContext(Array.Empty<Skill>()) { Knowledge = scope, CodeRunner = runner, Workspace = workspace },
                (messages, tools, ct) => Task.FromResult(new SkillTurnOutput(Next(messages, tools))),
                new() { MaxRounds = 8, ClientTools = Array.Empty<ToolFunction>() });
            Assert.Equal("The computed grade difference is 23.", result.Output.Parsed!.Content);
            Assert.False(result.HitRoundLimit);
        }
        Assert.Equal(7, generations);
        Assert.Equal(1, runner.Executions);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task IgnoringDisabledRetrievalTerminatesWithoutRemovingNormalCodeContinuation(bool streaming)
    {
        var scope = await Prepare();
        Assert.Contains("Mathematics 95", KnowledgeTools.Execute(Search(), scope).Content);
        scope.Retrieval.RequestSynthesis();
        long delivered = scope.Retrieval.EmittedEvidenceBytes;
        var runner = new CalculationRunner();
        var workspace = new SessionWorkspaceManager(Path.Combine(_root, "workspaces")).GetOrCreate("ignored-stop");
        int generations = 0;
        ParsedOutput Next(List<ChatMessage> messages, List<ToolFunction>? tools)
        {
            generations++;
            if (generations == 1)
            {
                Assert.DoesNotContain(tools!, tool => KnowledgeTools.IsTool(tool.Name));
                Assert.Contains(tools!, tool => tool.Name == SkillToolNames.Shell);
                var ignored = Search("Never look this up");
                ignored.Id = "ignored-lookup";
                return new() { ToolCalls = new() { ignored } };
            }
            Assert.Equal(2, generations);
            Assert.Empty(tools!);
            Assert.Contains("Mathematics 95", messages[^1].Content);
            Assert.IsAssignableFrom<HostContinuationMessage>(messages[^1]);
            if (!streaming) Assert.Single(messages, message => message.ToolCallId == "ignored-lookup");
            return new() { Content = "Mathematics is stronger." };
        }
        if (streaming)
        {
            var options = ServerOptionsBuilder.Build(new[] { "--model", "x.gguf" }, _root);
            var plan = SkillRequestPlan.Create(null, Array.Empty<string>(), false, null, "nemotron_h_moe",
                32768, options, out _, knowledge: scope, multiAgent: false, codeRunner: runner, workspace: workspace)!;
            async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
                [EnumeratorCancellation] CancellationToken ct)
            {
                var next = Next(messages, tools);
                yield return ChatStreamUpdate.Text("<think></think>" + (next.ToolCalls?.Count > 0
                    ? "<tool_call>" + JsonSerializer.Serialize(new { name = next.ToolCalls[0].Name, arguments = next.ToolCalls[0].Arguments }) + "</tool_call>"
                    : next.Content));
                await Task.Yield();
                yield return new ChatStreamUpdate("", true, 10, 20, 5, 0, 0, 0, "stop");
            }
            var content = new StringBuilder();
            await foreach (var update in SkillChatLoop.RunAsync("nemotron_h_moe", new(), plan, true, Generate, null, CancellationToken.None))
                if (!update.Done) content.Append(update.Piece);
            Assert.Equal("Mathematics is stronger.", content.ToString());
        }
        else
        {
            var result = await SkillAgentLoop.RunAsync(new(), KnowledgeTools.Merge(new[] { runner.Declare() }),
                new SkillToolContext(Array.Empty<Skill>()) { Knowledge = scope, CodeRunner = runner, Workspace = workspace },
                (messages, tools, ct) => Task.FromResult(new SkillTurnOutput(Next(messages, tools))),
                new() { MaxRounds = 64, ClientTools = Array.Empty<ToolFunction>() });
            Assert.Equal("Mathematics is stronger.", result.Output.Parsed!.Content);
            Assert.False(result.HitRoundLimit);
            Assert.Equal(2, result.Rounds);
        }
        Assert.Equal(2, generations);
        Assert.Equal(0, runner.Executions);
        Assert.Equal(delivered, scope.Retrieval.EmittedEvidenceBytes);
    }

    [Fact]
    public async Task OversizedReadWindowClampsToTheSupportedRangeAndCanonicalizesTheDuplicate()
    {
        var selected = await Prepare();
        await _store.ImportAsync(selected.CollectionIds[0], "long.txt",
            new MemoryStream(Encoding.UTF8.GetBytes(string.Join("\n\n", Enumerable.Range(0, 12)
                .Select(index => $"SECTION{index} " + new string((char)('a' + index), 600))))));
        var scope = new KnowledgeToolScope(_store, selected.CollectionIds);
        var hit = (await _store.SearchAsync("SECTION5", new() { CollectionIds = scope.CollectionIds }))[0];
        var call = new ToolCall { Name = KnowledgeTools.Read,
            Arguments = new() { ["reference"] = hit.Reference, ["adjacent_chunks"] = 3 } };
        var read = KnowledgeTools.Execute(call, scope);
        Assert.True(read.Ok);
        Assert.NotEmpty(scope.Sources);
        Assert.All(scope.Sources, source => Assert.InRange(Math.Abs(source.ChunkIndex - hit.ChunkIndex), 0, 2));
        call.Arguments["adjacent_chunks"] = 2;
        Assert.Contains("alreadyRetrieved", KnowledgeTools.Execute(call, scope).Content);
        var fresh = new KnowledgeToolScope(_store, selected.CollectionIds);
        call.Arguments["adjacent_chunks"] = -1;
        Assert.False(KnowledgeTools.Execute(call, fresh).Ok);
        call.Arguments["adjacent_chunks"] = "many";
        Assert.False(KnowledgeTools.Execute(call, fresh).Ok);
    }

    [Theory]
    [InlineData("thinking_budget", false)]
    [InlineData("stop", true)]
    public async Task AnswerlessThinkingBudgetKeepsRecoverySignalInsteadOfSyntheticExhaustion(string reason, bool hasNotice)
    {
        var scope = await Prepare();
        KnowledgeTools.Execute(Search(), scope);
        scope.Retrieval.RequestSynthesis();
        var options = ServerOptionsBuilder.Build(new[] { "--model", "x.gguf" }, _root);
        var plan = SkillRequestPlan.Create(null, Array.Empty<string>(), false, null, "nemotron_h_moe",
            32768, options, out _, knowledge: scope, multiAgent: false)!;
        plan.KnowledgeAnswerRecovery = true;
        async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
            [EnumeratorCancellation] CancellationToken ct)
        {
            Assert.Empty(tools);
            yield return ChatStreamUpdate.Text("<think>Need more evidence</think><tool_call>{\"name\":\"knowledge_search\",\"arguments\":{\"query\":\"Alex\"}}</tool_call>");
            await Task.Yield();
            yield return new ChatStreamUpdate("", true, 10, 20, 5, 0, 0, 0, reason);
        }
        var updates = new List<ChatStreamUpdate>();
        await foreach (var update in SkillChatLoop.RunAsync("nemotron_h_moe", new(), plan, true, Generate, null, CancellationToken.None))
            updates.Add(update);
        Assert.Equal(reason, Assert.Single(updates, update => update.Done).FinishReason);
        string visible = string.Concat(updates.Where(update => !update.Done).Select(update => update.Piece));
        Assert.Equal(hasNotice, !string.IsNullOrWhiteSpace(visible));
        Assert.Empty(plan.Invocations);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RecoveryStartsWithRecalledEvidenceAndNoFurtherRetrieval(bool streaming)
    {
        var scope = await Prepare();
        Assert.Contains("Mathematics 95", KnowledgeTools.Execute(Search(), scope).Content);
        scope.Retrieval.RequestSynthesis();
        var original = new List<ChatMessage> { new() { Role = "user", Content = "Compare the grades." } };
        void Inspect(List<ChatMessage> messages, List<ToolFunction>? tools)
        {
            Assert.Empty(tools!);
            Assert.Equal(original[0], messages[0]);
            Assert.Contains("Mathematics 95", messages[^1].Content);
            Assert.IsAssignableFrom<HostContinuationMessage>(messages[^1]);
        }
        if (streaming)
        {
            var options = ServerOptionsBuilder.Build(new[] { "--model", "x.gguf" }, _root);
            var plan = SkillRequestPlan.Create(null, Array.Empty<string>(), false, null, "nemotron_h_moe",
                32768, options, out _, knowledge: scope, multiAgent: false)!;
            async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
                [EnumeratorCancellation] CancellationToken ct)
            {
                Inspect(messages, tools);
                yield return ChatStreamUpdate.Text("<think></think>Mathematics is stronger.");
                await Task.Yield();
                yield return new ChatStreamUpdate("", true, 10, 20, 5, 0, 0, 0, "stop");
            }
            var content = new StringBuilder();
            await foreach (var update in SkillChatLoop.RunAsync("nemotron_h_moe", original, plan, true, Generate, null, CancellationToken.None))
                if (!update.Done) content.Append(update.Piece);
            Assert.Equal("Mathematics is stronger.", content.ToString());
        }
        else
        {
            var result = await SkillAgentLoop.RunAsync(original, KnowledgeTools.Merge(null),
                new SkillToolContext(Array.Empty<Skill>()) { Knowledge = scope }, (messages, tools, ct) =>
                {
                    Inspect(messages, tools);
                    return Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = "Mathematics is stronger." }));
                });
            Assert.Equal(1, result.Rounds);
        }
        Assert.Single(original);
    }

    [Fact]
    public async Task SynthesisReplayReplacesOldPayloadsPreservesToolPairingAndStaysBounded()
    {
        var scope = await Prepare();
        var hit = (await _store.SearchAsync("Alex", new() { CollectionIds = scope.CollectionIds }))[0];
        scope.RecordSources(Enumerable.Range(0, 30).Select(i => hit with
        {
            Reference = $"kb://excerpt/{i}", ChunkIndex = i, Location = new() { Page = i + 1 },
            Text = $"EXCERPT_{i} " + string.Concat(Enumerable.Repeat("成绩优秀😀仍需努力。", 200)),
        }));
        string snapshot = scope.GetBoundedEvidence();
        Assert.True(Encoding.UTF8.GetByteCount(snapshot) <= scope.MaxCharacters);
        Assert.Contains("EXCERPT_0 ", snapshot);
        Assert.Contains("EXCERPT_29 ", snapshot);
        Assert.DoesNotContain("�", snapshot);
        Assert.True(JsonDocument.Parse(snapshot).RootElement.GetProperty("truncated").GetBoolean());
        var original = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Compare grades" },
            new() { Role = "assistant", ToolCalls = new() { new() { Name = KnowledgeTools.Search, Id = "lookup-1" } } },
            new KnowledgeToolResultMessage { Role = "tool", ToolCallId = "lookup-1", Content = "OLD_EVIDENCE_" + new string('x', 30000) },
            new() { Role = "tool", ToolCallId = "code-1", Content = "Preserve code result" },
        };
        var once = KnowledgeTools.BuildSynthesisMessages(original, scope);
        var twice = KnowledgeTools.BuildSynthesisMessages(once, scope);
        Assert.Single(twice.OfType<KnowledgeEvidenceReplayMessage>());
        Assert.Equal("lookup-1", twice[2].ToolCallId);
        Assert.DoesNotContain("OLD_EVIDENCE_", twice[2].Content);
        Assert.Contains(twice, m => m.Content == "Preserve code result");
        Assert.StartsWith("OLD_EVIDENCE_", original[2].Content);
    }

    [Fact]
    public async Task ApprovedCompleteDocumentPreservesEveryMiddleExcerptDuringRecovery()
    {
        var selected = await Prepare();
        var scope = new KnowledgeToolScope(_store, selected.CollectionIds, 1024);
        var source = (await _store.SearchAsync("Alex", new() { CollectionIds = scope.CollectionIds }))[0];
        var complete = Enumerable.Range(0, 30).Select(index => source with
        {
            Reference = $"kb://complete/{index}", ChunkIndex = index,
            Location = new() { Page = index / 4 + 1 },
            Text = $"SUBJECT_{index} " + new string('x', 800), Truncated = false,
        }).ToArray();
        Assert.True(KnowledgeTools.TrySerializeCompleteDocumentEvidence(complete, 64000, out string payload));
        int bytes = Encoding.UTF8.GetByteCount(payload);
        Assert.True(bytes > scope.MaxCharacters);
        Assert.Equal(0, scope.Retrieval.EmittedEvidenceBytes);
        Assert.False(scope.Retrieval.HasCompleteDocumentEvidence);

        // The chat host performs the actual tokenizer/context check between candidate
        // serialization and this explicit commit; serialization alone has no effects.
        scope.RecordCompleteDocumentEvidence(complete, payload);
        Assert.True(scope.Retrieval.HasCompleteDocumentEvidence);
        Assert.True(scope.Retrieval.ShouldSynthesize);
        Assert.Equal(bytes, scope.Retrieval.EmittedEvidenceBytes);
        Assert.Equal(payload, scope.GetBoundedEvidence());
        Assert.Equal(30, scope.Sources.Count);
        scope.RecordCompleteDocumentEvidence(complete, payload);
        Assert.Equal(bytes, scope.Retrieval.EmittedEvidenceBytes);
        var replay = KnowledgeTools.BuildSynthesisMessages(Array.Empty<ChatMessage>(), scope);
        Assert.Equal(KnowledgeTools.CreateEvidenceReplayMessage(payload).Content, replay[^1].Content);
        var repeated = KnowledgeTools.BuildSynthesisMessages(replay, scope);
        foreach (int index in Enumerable.Range(0, 30)) Assert.Contains($"SUBJECT_{index} ", repeated[^1].Content);
        Assert.Equal(bytes, scope.Retrieval.EmittedEvidenceBytes);
        Assert.Single(repeated.OfType<KnowledgeEvidenceReplayMessage>());
        Assert.Contains("\"indexedDocumentComplete\":true", payload);

        using var child = AgentWorkspace.Create(new SkillToolContext(Array.Empty<Skill>()) { Knowledge = scope },
            "complete-evidence-child-" + Guid.NewGuid().ToString("N"));
        Assert.False(child.Context.KnowledgeRetrieval!.HasCompleteDocumentEvidence);
        Assert.False(child.Context.KnowledgeRetrieval.ShouldSynthesize);
        Assert.DoesNotContain("SUBJECT_", scope.GetBoundedEvidence(child.Context.KnowledgeRetrieval));
    }

    [Fact]
    public async Task CompleteDocumentCandidateRejectsClippingMixedDocumentsAndOutOfScopeCommit()
    {
        var scope = await Prepare();
        var source = (await _store.SearchAsync("Alex", new() { CollectionIds = scope.CollectionIds }))[0];
        var complete = new[] { source with { Text = new string('界', 3000), Truncated = false } };
        Assert.True(KnowledgeTools.TrySerializeCompleteDocumentEvidence(complete, 20000, out string payload));
        int bytes = Encoding.UTF8.GetByteCount(payload);
        Assert.True(KnowledgeTools.TrySerializeCompleteDocumentEvidence(complete, bytes, out string exact));
        Assert.Equal(payload, exact);
        Assert.False(KnowledgeTools.TrySerializeCompleteDocumentEvidence(complete, bytes - 1, out string declined));
        Assert.Empty(declined);
        Assert.False(KnowledgeTools.TrySerializeCompleteDocumentEvidence(new[] { source with { Truncated = true } }, 20000, out _));
        Assert.False(KnowledgeTools.TrySerializeCompleteDocumentEvidence(new[] { source, source with { DocumentId = "other" } }, 20000, out _));
        Assert.False(KnowledgeTools.TrySerializeCompleteDocumentEvidence(Array.Empty<KnowledgeSearchResult>(), 20000, out _));
        Assert.Throws<ArgumentException>(() => scope.RecordCompleteDocumentEvidence(complete, payload + " "));
        var otherCollection = await _store.CreateCollectionAsync("other");
        var other = new KnowledgeToolScope(_store, new[] { otherCollection.Id });
        Assert.Throws<ArgumentException>(() => other.RecordCompleteDocumentEvidence(complete, payload));
        Assert.False(scope.Retrieval.HasCompleteDocumentEvidence);
        Assert.False(scope.Retrieval.ShouldSynthesize);
        Assert.Empty(scope.Sources);
        Assert.Equal(0, scope.Retrieval.EmittedEvidenceBytes);
    }

    [Fact]
    public async Task RequestCancellationReachesAnInFlightSearch()
    {
        var embedding = new WaitingEmbedding();
        string path = Path.Combine(_root, "embedding");
        using var store = new KnowledgeStore(path, new BuiltInKnowledgeExtractor(), embedding);
        var collection = await store.CreateCollectionAsync("selected");
        await store.ImportAsync(collection.Id, "report.txt", new MemoryStream(Encoding.UTF8.GetBytes("Alex report")));
        using var cancellation = new CancellationTokenSource();
        var context = new SkillToolContext(Array.Empty<Skill>())
        { Knowledge = new KnowledgeToolScope(store, new[] { collection.Id }) };
        Task<SkillToolResult> execution = Task.Run(() => SkillTools.Execute(Search(), context, cancellationToken: cancellation.Token));
        await embedding.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => execution.WaitAsync(TimeSpan.FromSeconds(5)));
    }

    [Fact]
    public void SerializedBudgetPreservesSeveralSubjectsAndUnicode()
    {
        var hits = Enumerable.Range(0, 3).Select(i => new KnowledgeSearchResult
        {
            CollectionId = new string('a', 32), DocumentId = new string((char)('b' + i), 32),
            Reference = $"kb://subject/{i}", FileName = $"subject{i}.txt", Version = 1,
            Text = string.Concat(Enumerable.Repeat("成绩优秀😀需要提高。", 200)),
        }).ToArray();
        string json = KnowledgeTools.SerializeResults(hits, 4096, out var included);
        Assert.True(Encoding.UTF8.GetByteCount(json) <= 4096);
        Assert.Equal(3, included.Count);
        Assert.All(included, hit => { Assert.True(hit.Text.Length > 40); Assert.True(hit.Truncated); });
        Assert.DoesNotContain("�", json);
        Assert.True(JsonDocument.Parse(json).RootElement.GetProperty("truncated").GetBoolean());
    }

    private sealed class WaitingEmbedding : IKnowledgeEmbeddingProvider
    {
        public string ModelId => "waiting-fixture";
        public int Dimensions => 2;
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public async Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, bool isQuery, CancellationToken cancellationToken = default)
        {
            if (isQuery)
            {
                Started.TrySetResult();
                await Task.Delay(Timeout.Infinite, cancellationToken);
            }
            return texts.Select(_ => new[] { 1f, 0f }).ToArray();
        }
    }

    private sealed class CalculationRunner : ICodeRunner
    {
        public bool CanRun => true;
        public string? UnavailableReason => null;
        public bool CanInstallPackages => false;
        public int Executions { get; private set; }
        public ToolFunction Declare() => new() { Name = SkillToolNames.Shell, Description = "Calculate the requested result" };
        public string InstallPackages(string language, IReadOnlyList<string> packages, SessionWorkspace workspace,
            Action<string>? onOutput = null) => throw new NotSupportedException();
        public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null, SessionWorkspace? workspace = null, IReadOnlyList<string>? skillDirectories = null)
        {
            Assert.Equal("95 - 72", call.Arguments["command"].ToString());
            Executions++;
            return SkillToolResult.Success((95 - 72).ToString());
        }
    }
}
