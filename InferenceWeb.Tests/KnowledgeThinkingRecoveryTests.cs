using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using TensorSharp.AgentHost.Knowledge;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Knowledge;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;
using TensorSharp.Server.Knowledge;

namespace InferenceWeb.Tests;

public sealed class KnowledgeThinkingRecoveryTests
{
    private const string FullReportQuestion = "Compare Mira's grades and teacher comments.";
    private const string MiddleComment = "MIDDLE_COMMENT: Explain the reasoning in writing instead of skipping intermediate steps.";

    [Fact]
    public async Task CompleteSmallDocumentRetainsMiddleCommentsAcrossSynthesisAndTranscriptCloning()
    {
        string root = Path.Combine(Path.GetTempPath(), "knowledge-full-report-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var store = new KnowledgeStore(root, new SyntheticReportExtractor());
            var collection = await store.CreateCollectionAsync("reports");
            var document = await store.ImportAsync(collection.Id, "report.txt", new MemoryStream([1]));
            var scope = new KnowledgeToolScope(store, new[] { collection.Id });
            var original = new List<ChatMessage> { new() { Role = "user", Content = FullReportQuestion } };
            int fitChecks = 0;
            var prepared = await KnowledgeRequestContext.PrepareAsync(original, scope, toolsAvailable: true,
                completeDocumentFits: candidate =>
                {
                    fitChecks++;
                    Assert.Contains(MiddleComment, string.Join('\n', candidate.Select(message => message.Content)));
                    return true;
                });
            Assert.Equal(1, fitChecks);
            Assert.True(scope.Retrieval.HasCompleteDocumentEvidence);
            Assert.True(scope.Retrieval.ShouldSynthesize);
            Assert.Equal(20, scope.Sources.Count);
            Assert.All(scope.Sources, source => { Assert.Equal(document.Id, source.DocumentId); Assert.False(source.Truncated); });
            Assert.Equal(FullReportQuestion, original[0].Content);
            // Tool-capable preparation commits evidence to the scope, then the loop
            // inserts it exactly once. Carrying both forms doubled the actual prompt.
            Assert.DoesNotContain(MiddleComment, string.Join('\n', prepared.Select(message => message.Content)));

            var synthesized = KnowledgeTools.BuildSynthesisMessages(prepared, scope);
            var replay = Assert.Single(synthesized.OfType<KnowledgeEvidenceReplayMessage>());
            Assert.Contains(MiddleComment, replay.Content);
            string payload = scope.GetBoundedEvidence();
            using var json = JsonDocument.Parse(payload);
            Assert.True(json.RootElement.GetProperty("indexedDocumentComplete").GetBoolean());
            Assert.False(json.RootElement.GetProperty("truncated").GetBoolean());
            var excerpts = Assert.Single(json.RootElement.GetProperty("sources").EnumerateArray()).GetProperty("excerpts");
            Assert.Equal(20, excerpts.GetArrayLength());
            Assert.Contains(MiddleComment, excerpts[10].GetProperty("text").GetString());

            var cloned = synthesized.Select(ConversationTranscriptStore.CloneShallow).ToList();
            var repeated = KnowledgeTools.BuildSynthesisMessages(cloned, scope);
            Assert.Single(repeated.OfType<KnowledgeEvidenceReplayMessage>());
            Assert.Equal(1, CountOccurrences(string.Join('\n', repeated.Select(message => message.Content)), MiddleComment));
            Assert.Contains(repeated, message => message.Role == "user" && message.Content == FullReportQuestion);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public async Task RejectedCompleteDocumentFallsBackToTwoBoundedOrientationExcerpts()
    {
        string root = Path.Combine(Path.GetTempPath(), "knowledge-full-rejected-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var store = new KnowledgeStore(root, new SyntheticReportExtractor());
            var collection = await store.CreateCollectionAsync("reports");
            await store.ImportAsync(collection.Id, "report.txt", new MemoryStream([1]));
            var scope = new KnowledgeToolScope(store, new[] { collection.Id });
            int fitChecks = 0;
            var prepared = await KnowledgeRequestContext.PrepareAsync(
                new() { new() { Role = "user", Content = FullReportQuestion } }, scope, toolsAvailable: true,
                completeDocumentFits: candidate =>
                {
                    fitChecks++;
                    Assert.Contains(MiddleComment, string.Join('\n', candidate.Select(message => message.Content)));
                    return false;
                });
            Assert.Equal(1, fitChecks);
            Assert.False(scope.Retrieval.HasCompleteDocumentEvidence);
            Assert.False(scope.Retrieval.ShouldSynthesize);
            Assert.Equal(2, scope.Sources.Count);
            Assert.InRange(scope.Retrieval.EmittedEvidenceBytes, 1, 4000);
            Assert.EndsWith(FullReportQuestion, prepared.Last(message => message.Role == "user").Content);
            Assert.DoesNotContain(prepared, message => message is KnowledgeEvidenceReplayMessage);
            Assert.DoesNotContain("\"indexedDocumentComplete\":true", string.Join('\n', prepared.Select(message => message.Content)));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public async Task MixedTopDocumentsDoNotExpandAnArbitraryCompleteSource()
    {
        string root = Path.Combine(Path.GetTempPath(), "knowledge-full-mixed-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var store = new KnowledgeStore(root, new SyntheticReportExtractor(sectionCount: 3));
            var collection = await store.CreateCollectionAsync("reports");
            await store.ImportAsync(collection.Id, "first.txt", new MemoryStream([1]));
            await store.ImportAsync(collection.Id, "second.txt", new MemoryStream([2]));
            var scope = new KnowledgeToolScope(store, new[] { collection.Id });
            var search = await store.SearchAsync(FullReportQuestion, new()
            { CollectionIds = scope.CollectionIds, Limit = 6, MaxCharacters = scope.MaxCharacters });
            Assert.Equal(6, search.Count);
            Assert.Equal(2, search.Select(source => source.DocumentId).Distinct().Count());
            bool checkedFit = false;
            await KnowledgeRequestContext.PrepareAsync(
                new() { new() { Role = "user", Content = FullReportQuestion } }, scope, toolsAvailable: true,
                completeDocumentFits: _ => { checkedFit = true; return true; });
            Assert.False(checkedFit);
            Assert.False(scope.Retrieval.HasCompleteDocumentEvidence);
            Assert.Equal(2, scope.Sources.Count);
            Assert.InRange(scope.Retrieval.EmittedEvidenceBytes, 1, 4000);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public async Task ToolFreeCompleteDocumentAppearsOnceBeforeTheOriginalQuestion()
    {
        string root = Path.Combine(Path.GetTempPath(), "knowledge-full-direct-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var store = new KnowledgeStore(root, new SyntheticReportExtractor());
            var collection = await store.CreateCollectionAsync("reports");
            await store.ImportAsync(collection.Id, "report.txt", new MemoryStream([1]));
            var scope = new KnowledgeToolScope(store, new[] { collection.Id });
            var original = new List<ChatMessage> { new() { Role = "user", Content = FullReportQuestion } };
            var prepared = await KnowledgeRequestContext.PrepareAsync(original, scope, toolsAvailable: false,
                completeDocumentFits: _ => true);
            Assert.True(scope.Retrieval.HasCompleteDocumentEvidence);
            Assert.Equal(20, scope.Sources.Count);
            Assert.Equal(FullReportQuestion, original[0].Content);
            var user = Assert.Single(prepared.Where(message => message.Role == "user"));
            Assert.EndsWith(FullReportQuestion, user.Content);
            Assert.Contains(scope.GetBoundedEvidence(), user.Content);
            Assert.Equal(1, CountOccurrences(string.Join('\n', prepared.Select(message => message.Content)), MiddleComment));
            Assert.DoesNotContain(prepared, message => message is KnowledgeEvidenceReplayMessage);
            Assert.DoesNotContain(prepared.Where(message => message.Role == "system"), message => message.Content.Contains(MiddleComment));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    private static int CountOccurrences(string text, string value) => text.Split(value, StringSplitOptions.None).Length - 1;

    private sealed class SyntheticReportExtractor(int sectionCount = 20) : IKnowledgeExtractor
    {
        public Task<KnowledgeExtraction> ExtractAsync(KnowledgeExtractionContext context,
            CancellationToken cancellationToken = default) => Task.FromResult(new KnowledgeExtraction
            {
                Segments = Enumerable.Range(0, sectionCount).Select(index => new KnowledgeSegment(
                    $"Mira grades and teacher comments in {context.FileName}, section {index:D2}. "
                    + (index == 10 ? MiddleComment + " " : "")
                    + string.Join(' ', Enumerable.Repeat($"Detailed assessment of subject {index:D2} and its evidence.", 12)),
                    new KnowledgeLocation { Section = $"Subject {index:D2}" }, "text")).ToArray(),
            });
    }

    [Fact]
    public async Task ToolCapablePreparationLeavesRoomForTargetedFollowUpEvidence()
    {
        string root = Path.Combine(Path.GetTempPath(), "knowledge-orientation-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var store = new KnowledgeStore(root, new BuiltInKnowledgeExtractor());
            var collection = await store.CreateCollectionAsync("reports");
            for (int i = 0; i < 6; i++)
                await store.ImportAsync(collection.Id, $"report-{i}.txt", new MemoryStream(Encoding.UTF8.GetBytes(
                    string.Join(' ', Enumerable.Repeat($"Alex report {i} has scores and detailed observations about subject {i}.", 100)))));
            const string question = "Compare Alex's scores and comments.";
            var orientation = new KnowledgeToolScope(store, new[] { collection.Id });
            var messages = await KnowledgeRequestContext.PrepareAsync(
                new() { new() { Role = "user", Content = question } }, orientation, toolsAvailable: true);
            Assert.Equal(2, orientation.Sources.Count);
            Assert.InRange(orientation.Retrieval.EmittedEvidenceBytes, 1, 4000);
            Assert.False(orientation.Retrieval.ShouldSynthesize);
            Assert.EndsWith(question, messages.Last(message => message.Role == "user").Content);

            var direct = new KnowledgeToolScope(store, new[] { collection.Id });
            await KnowledgeRequestContext.PrepareAsync(new() { new() { Role = "user", Content = question } },
                direct, toolsAvailable: false);
            Assert.True(direct.Sources.Count > orientation.Sources.Count);
            Assert.InRange(direct.Retrieval.EmittedEvidenceBytes, 4001, 12000);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task ThinkingRetryAnswersWithPreviouslyReadEvidenceWithoutRestartingRetrieval(bool codeEnabled, bool hasClientTool)
    {
        string root = Path.Combine(Path.GetTempPath(), "knowledge-thinking-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var store = new KnowledgeStore(root, new BuiltInKnowledgeExtractor());
            var collection = await store.CreateCollectionAsync("reports");
            await store.ImportAsync(collection.Id, "grades.txt", new MemoryStream(Encoding.UTF8.GetBytes("Alex mathematics 95.")));
            await store.ImportAsync(collection.Id, "comments.txt", new MemoryStream(Encoding.UTF8.GetBytes("Writing 72. Needs proofreading and supporting detail.")));
            var scope = new KnowledgeToolScope(store, new[] { collection.Id });
            var original = new List<ChatMessage> { new() { Role = "user", Content = "Compare Alex's scores and teacher comments." } };
            var prepared = new List<ChatMessage> { new() { Role = "user", Content = "Eager source block plus original question." } };
            var options = ServerOptionsBuilder.Build(new[] { "--model", "x.gguf" }, root);
            var runner = new RecoveryRunner();
            var client = new ToolFunction { Name = "client_lookup", Description = "Caller-owned lookup" };
            var plan = SkillRequestPlan.Create(null, Array.Empty<string>(), false, hasClientTool ? new() { client } : null, "nemotron_h_moe",
                8192, options, out var unknown, knowledge: scope, multiAgent: false, codeRunner: codeEnabled ? runner : null);
            Assert.Empty(unknown);
            Assert.NotNull(plan);
            if (codeEnabled) Assert.Contains(plan.Tools, tool => tool.Name == SkillToolNames.Shell);
            // Evidence discovered by tools during the over-budget attempt is absent
            // from original/prepared history, but must survive the rescue generation.
            foreach (string query in new[] { "mathematics", "proofreading" })
                KnowledgeTools.Execute(new ToolCall { Name = KnowledgeTools.Search, Arguments = new() { ["query"] = query } }, scope);
            var retry = WebUiChatService.PrepareKnowledgeThinkingRetry(original, prepared, scope, plan!);
            Assert.Contains(retry, message => message.Role == "user" && message.Content == original[0].Content);
            Assert.Contains(retry, message => message.Role == "system" && message.Content.Contains(KnowledgeTools.Guidance));
            Assert.DoesNotContain(retry.Where(message => message.Role == "system"), message => message.Content.Contains("Writing 72"));
            Assert.True(scope.Retrieval.ShouldSynthesize);
            Assert.True(plan.KnowledgeAnswerRecovery);
            int generations = 0;
            async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
                [EnumeratorCancellation] CancellationToken cancellationToken)
            {
                generations++;
                Assert.DoesNotContain(tools, tool => KnowledgeTools.IsTool(tool.Name));
                Assert.DoesNotContain(tools, tool => SkillTools.IsBuiltInTool(tool.Name));
                if (hasClientTool) Assert.Same(client, Assert.Single(tools));
                else Assert.Empty(tools);
                string content = string.Join('\n', messages.Select(message => message.Content));
                Assert.Contains(original[0].Content, content);
                Assert.Contains("mathematics 95", content);
                Assert.Contains("Writing 72", content);
                Assert.Contains("proofreading", content);
                Assert.DoesNotContain("Eager source block", content);
                yield return ChatStreamUpdate.Text("Mathematics is stronger; writing needs proofreading.");
                await Task.Yield();
                yield return new ChatStreamUpdate("", true, 10, 20, 5, 0, 0, 0, "stop");
            }
            var updates = new List<ChatStreamUpdate>();
            await foreach (var update in SkillChatLoop.RunAsync("nemotron_h_moe", retry, plan!, false,
                Generate, null!, CancellationToken.None)) updates.Add(update);
            Assert.Equal(1, generations);
            Assert.Contains("writing needs proofreading", string.Concat(updates.Select(update => update.Piece)));
            Assert.Empty(plan!.Invocations);
            Assert.Equal(0, runner.Executions);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public async Task ThinkingRetryPreservesRequiredArtifactToolsEvenWithAStaleRecoveryFlag()
    {
        string root = Path.Combine(Path.GetTempPath(), "knowledge-thinking-guard-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var store = new KnowledgeStore(Path.Combine(root, "knowledge"), new BuiltInKnowledgeExtractor());
            var collection = await store.CreateCollectionAsync("reports");
            var scope = new KnowledgeToolScope(store, new[] { collection.Id });
            var original = new List<ChatMessage> { new() { Role = "user", Content = "Create the requested presentation." } };
            var workspace = new SessionWorkspaceManager(Path.Combine(root, "workspaces")).GetOrCreate("guarded");
            var options = ServerOptionsBuilder.Build(new[] { "--model", "x.gguf" }, root);
            var plan = SkillRequestPlan.Create(null, Array.Empty<string>(), false, null, "nemotron_h_moe",
                8192, options, out _, knowledge: scope, multiAgent: false, codeRunner: new RecoveryRunner(), workspace: workspace)!;
            Assert.True(WorkspaceArtifactCompletionRequirement.TryCreate(".pptx", workspace,
                new CodeArtifactStore(Path.Combine(root, "artifacts")), "/api/artifacts", out var requirement));
            plan.CompletionRequirement = requirement;
            var retry = WebUiChatService.PrepareKnowledgeThinkingRetry(original, original, scope, plan);
            Assert.False(plan.KnowledgeAnswerRecovery);
            // A guard installed after the flag was set must still take precedence.
            plan.KnowledgeAnswerRecovery = true;
            using var cancellation = new CancellationTokenSource();
            bool inspected = false;
            int generations = 0;
            async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
                [EnumeratorCancellation] CancellationToken ct)
            {
                Assert.Contains(tools, tool => tool.Name == SkillToolNames.Shell);
                Assert.DoesNotContain(tools, tool => KnowledgeTools.IsTool(tool.Name));
                Assert.Contains(messages, message => message.Content?.Contains(KnowledgeRetrievalProgress.CompleteTaskInstruction) == true);
                if (++generations == 1)
                {
                    yield return ChatStreamUpdate.Text("<tool_call>{\"name\":\"knowledge_search\",\"arguments\":{\"query\":\"missing evidence\"}}</tool_call>");
                    yield return new ChatStreamUpdate("", true, 10, 20, 5, 0, 0, 0, "stop");
                    yield break;
                }
                inspected = true;
                cancellation.Cancel();
                ct.ThrowIfCancellationRequested();
                await Task.Yield();
                yield break;
            }
            await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
            {
                await foreach (var _ in SkillChatLoop.RunAsync("nemotron_h_moe", retry, plan, false,
                    Generate, null!, cancellation.Token)) { }
            });
            Assert.True(inspected);
            Assert.Equal(2, generations);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    private sealed class RecoveryRunner : ICodeRunner
    {
        public bool CanRun => true;
        public string? UnavailableReason => null;
        public bool CanInstallPackages => false;
        public int Executions { get; private set; }
        public ToolFunction Declare() => new() { Name = SkillToolNames.Shell, Description = "Code tool" };
        public string InstallPackages(string language, IReadOnlyList<string> packages, SessionWorkspace workspace,
            Action<string>? onOutput = null) => throw new NotSupportedException();
        public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null, SessionWorkspace? workspace = null, IReadOnlyList<string>? skillDirectories = null)
        {
            Executions++;
            return SkillToolResult.Failure("A terminal recovery must not execute code.");
        }
    }
}
