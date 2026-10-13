// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using Xunit.Abstractions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;

namespace TensorAgent.Tests;

[Collection(LiveModelCollection.Name)]
public sealed class KnowledgeLiveTests(ITestOutputHelper output) : LiveModelHarness
{
    [LiveModelFact]
    public async Task ChineseComparisonUsesEnglishReportAndListsOnlyCitedSources()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        Start(model, weights);
        await LoadAsync(model);
        using var create = await Client.PostAsJsonAsync("/api/knowledge/collections", new { name = "Synthetic school reports" });
        create.EnsureSuccessStatusCode();
        string id = (await create.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("collection").GetProperty("id").GetString()!;
        using var form = new MultipartFormDataContent();
        form.Add(new ByteArrayContent(Encoding.UTF8.GetBytes(
            "Alex Rowan school report. Mathematics: 95 out of 100. Teacher: strong reasoning and explains solutions clearly. "
            + "Writing: 72 out of 100. Teacher: creative ideas but needs proofreading, punctuation and supporting detail. "
            + "Science: 86 out of 100. Teacher: curious and participates well; written explanations need more evidence. "
            + "These are final scores; all three subjects pass.")), "files", "student-report.txt");
        form.Add(new ByteArrayContent(Encoding.UTF8.GetBytes(
            "学校课程介绍：数学、写作、科学的成绩比较方法。学生的亮点和不足可以综合老师的评语分析。"
            + "这份课程介绍没有任何学生的成绩或个人评语。")), "files", "course-overview.txt");
        using var imported = await Client.PostAsync($"/api/knowledge/collections/{id}/documents", form);
        imported.EnsureSuccessStatusCode();
        Assert.Empty((await imported.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("errors").EnumerateArray());
        JsonElement session = await OpenSessionAsync();
        using var deadline = new CancellationTokenSource(TimeSpan.FromMinutes(5));
        var frames = await StreamAsync(new
        {
            sessionId = session.GetProperty("sessionId").GetString(), knowledgeCollections = new[] { id },
            skills = Array.Empty<string>(), skills_discovery = false,
            messages = new[] { new { role = "user", content = "Alex Rowan的哪个成绩比较好，哪个成绩比较差？请综合老师评语，用中文比较分数、亮点和不足，并引用实际使用的来源。" } },
            maxTokens = 2048, think = false,
        }, deadline.Token);
        string answer = TextOf(frames);
        Assert.Contains("95", answer);
        Assert.Contains("72", answer);
        Assert.Contains("数学", answer);
        Assert.Contains("写作", answer);
        Assert.Contains(frames, frame => frame.TryGetProperty("done", out JsonElement done) && done.ValueKind == JsonValueKind.True);
        Assert.DoesNotContain(frames, frame => frame.TryGetProperty("error", out JsonElement error)
            && error.ValueKind is not (JsonValueKind.Null or JsonValueKind.Undefined));
        Assert.DoesNotContain(frames, frame => frame.TryGetProperty("truncated", out JsonElement truncated)
            && truncated.ValueKind == JsonValueKind.True);
        Assert.InRange(frames.Count(frame => frame.TryGetProperty("skill_step", out JsonElement step)
            && step.GetString()!.StartsWith("knowledge_", StringComparison.Ordinal)), 0, 16);
        var sources = frames.Last(frame => frame.TryGetProperty("knowledgeSources", out _)).GetProperty("knowledgeSources");
        Assert.Contains(sources.EnumerateArray(), source => source.GetProperty("fileName").GetString() == "student-report.txt");
        foreach (JsonElement source in sources.EnumerateArray())
            Assert.True(answer.Contains(source.GetProperty("sourceUrl").GetString()!, StringComparison.Ordinal)
                || answer.Contains(source.GetProperty("reference").GetString()!, StringComparison.Ordinal),
                "An uncited source was displayed. Answer: " + answer);
    }

    [LiveModelFact]
    public async Task MultipleTechnicalSourcesSupportRequestedTableWithoutUnusedCitations()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        var rankingEvents = new ConcurrentQueue<string>();
        using var logging = LoggerFactory.Create(builder => builder.AddProvider(new RankingLogProvider(rankingEvents)));
        Start(model, weights, loggerFactory: logging);
        await LoadAsync(model);
        using var create = await Client.PostAsJsonAsync("/api/knowledge/collections", new { name = "Synthetic operations manuals" });
        create.EnsureSuccessStatusCode();
        string id = (await create.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("collection").GetProperty("id").GetString()!;
        using var form = new MultipartFormDataContent();
        foreach (var (fileName, content) in new[]
        {
            ("orion-overview.txt", "Orion service documentation overview. Restart and rollback procedures, readiness checks and follow-up actions "
                + "are described in the individual operations manuals. This overview does not specify any checks or follow-up actions."),
            ("orion-restart.txt", "Orion service restart procedure. Required readiness check: WAIT_FOR_QUORUM. "
                + "Follow-up action: CHECK_REPLICA_HEALTH. These steps apply to restarting the currently deployed version."),
            ("orion-rollback.txt", "Orion service rollback procedure. Required readiness check: VERIFY_SCHEMA_V7. "
                + "Follow-up action: REPLAY_AUDIT_QUEUE. These steps apply to reverting to the preceding deployed version."),
        })
            form.Add(new ByteArrayContent(Encoding.UTF8.GetBytes(content)), "files", fileName);
        using var imported = await Client.PostAsync($"/api/knowledge/collections/{id}/documents", form);
        imported.EnsureSuccessStatusCode();
        Assert.Empty((await imported.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("errors").EnumerateArray());
        JsonElement session = await OpenSessionAsync();
        using var deadline = new CancellationTokenSource(TimeSpan.FromMinutes(2));
        const string query = "How do the Orion service restart and rollback procedures differ? "
            + "Return a Markdown table with one row per operation and columns Operation, Readiness check, and Follow-up action. "
            + "Preserve the exact check and action identifiers, and cite the manuals you use.";
        var frames = await StreamAsync(new
        {
            sessionId = session.GetProperty("sessionId").GetString(), knowledgeCollections = new[] { id },
            skills = Array.Empty<string>(), skills_discovery = false,
            messages = new[] { new { role = "user", content = query } },
            maxTokens = 1024, think = false,
        }, deadline.Token);
        string answer = TextOf(frames);
        string renderedIdentifiers = answer.Replace("\\_", "_", StringComparison.Ordinal);
        string diagnostic = string.Join('\n', rankingEvents) + "\nSynthetic answer:\n" + answer;
        foreach (string message in rankingEvents) output.WriteLine(message);
        Assert.True(rankingEvents.Any(message => message.Contains("rerank applied:", StringComparison.Ordinal)), diagnostic);
        foreach (string identifier in new[] { "WAIT_FOR_QUORUM", "CHECK_REPLICA_HEALTH", "VERIFY_SCHEMA_V7", "REPLAY_AUDIT_QUEUE" })
            Assert.True(renderedIdentifiers.Contains(identifier, StringComparison.Ordinal), "Missing " + identifier + "\n" + diagnostic);
        Assert.Matches(@"(?i)\|[^\r\n]*Operation[^\r\n]*Readiness check[^\r\n]*Follow-up action[^\r\n]*\|", answer);
        var rows = renderedIdentifiers.Split('\n').Where(line => line.TrimStart().StartsWith('|')).Select(line => line.Split('|')).ToArray();
        var restart = Assert.Single(rows, row => row.Length >= 4 && row[1].Contains("restart", StringComparison.OrdinalIgnoreCase));
        var rollback = Assert.Single(rows, row => row.Length >= 4 && row[1].Contains("rollback", StringComparison.OrdinalIgnoreCase));
        Assert.Contains("WAIT_FOR_QUORUM", restart[2]);
        Assert.Contains("CHECK_REPLICA_HEALTH", restart[3]);
        Assert.Contains("VERIFY_SCHEMA_V7", rollback[2]);
        Assert.Contains("REPLAY_AUDIT_QUEUE", rollback[3]);
        Assert.Contains(frames, frame => frame.TryGetProperty("done", out JsonElement done) && done.ValueKind == JsonValueKind.True);
        Assert.DoesNotContain(frames, frame => frame.TryGetProperty("error", out JsonElement error)
            && error.ValueKind is not (JsonValueKind.Null or JsonValueKind.Undefined));
        Assert.DoesNotContain(frames, frame => frame.TryGetProperty("truncated", out JsonElement truncated)
            && truncated.ValueKind == JsonValueKind.True);
        var sources = frames.Last(frame => frame.TryGetProperty("knowledgeSources", out _)).GetProperty("knowledgeSources");
        Assert.Equal(new[] { "orion-restart.txt", "orion-rollback.txt" }.Order(),
            sources.EnumerateArray().Select(source => source.GetProperty("fileName").GetString()!).Order());
        foreach (JsonElement source in sources.EnumerateArray())
        {
            Assert.True(answer.Contains(source.GetProperty("sourceUrl").GetString()!, StringComparison.Ordinal)
                || answer.Contains(source.GetProperty("reference").GetString()!, StringComparison.Ordinal));
            using var original = await Client.GetAsync(source.GetProperty("sourceUrl").GetString());
            original.EnsureSuccessStatusCode();
        }
    }

    private sealed class RankingLogProvider(ConcurrentQueue<string> events) : ILoggerProvider
    {
        public ILogger CreateLogger(string categoryName) => new RankingLogger(events);
        public void Dispose() { }
        private sealed class RankingLogger(ConcurrentQueue<string> events) : ILogger
        {
            public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
            public bool IsEnabled(LogLevel logLevel) => true;
            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
                Func<TState, Exception?, string> formatter)
            {
                string message = formatter(state, exception);
                if (message.StartsWith("Knowledge rerank ", StringComparison.Ordinal)) events.Enqueue(message);
            }
        }
    }

    [LiveModelFact]
    public async Task UploadedPrivateEvidenceAnswersThroughRealChatAndSurvivesConversationPersistence()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        AgentAppHost host = Start(model, weights);
        await LoadAsync(model);
        using var create = await Client.PostAsJsonAsync("/api/knowledge/collections", new { name = "Field notebook" });
        create.EnsureSuccessStatusCode();
        var collection = (await create.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("collection");
        string id = collection.GetProperty("id").GetString()!;
        const string fact = "The Larkspur expedition's agreed rendezvous code is AMBER-KIWI-731. The code was generated for this private field notebook.";
        using var form = new MultipartFormDataContent();
        form.Add(new ByteArrayContent(Encoding.UTF8.GetBytes(fact)), "files", "field-notebook.txt");
        using var imported = await Client.PostAsync($"/api/knowledge/collections/{id}/documents", form);
        imported.EnsureSuccessStatusCode();
        Assert.Empty((await imported.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("errors").EnumerateArray());
        JsonElement session = await OpenSessionAsync();
        string sessionId = session.GetProperty("sessionId").GetString()!;
        var frames = await StreamAsync(new
        {
            sessionId, knowledgeCollections = new[] { id }, skills = Array.Empty<string>(), skills_discovery = false,
            messages = new[] { new { role = "user", content = "What is the Larkspur expedition rendezvous code? Use the selected field notebook and cite the source link." } },
            maxTokens = 256, think = false,
        });
        string answer = TextOf(frames);
        Assert.Contains("AMBER-KIWI-731", answer, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("/api/knowledge/collections/", answer);
        JsonElement cited = frames.Last(frame => frame.TryGetProperty("knowledgeSources", out _)).GetProperty("knowledgeSources")[0];
        using var original = await Client.GetAsync(cited.GetProperty("sourceUrl").GetString());
        original.EnsureSuccessStatusCode();
        Assert.Equal(fact, await original.Content.ReadAsStringAsync());
        // The final replacement, when present, must be what users and saved history
        // receive. Resolve every local source URL in the final visible prose as well.
        foreach (Match match in Regex.Matches(answer, @"/api/knowledge/collections/[a-f0-9]{32}/documents/[^\s)<>""`]+/source\?version=[0-9]+"))
        {
            using var proseSource = await Client.GetAsync(match.Value);
            proseSource.EnsureSuccessStatusCode();
            Assert.Equal(fact, await proseSource.Content.ReadAsStringAsync());
        }
        Assert.Contains(frames, frame => frame.TryGetProperty("done", out _));
        var saved = host.Conversations.Load(session.GetProperty("conversationId").GetString()!)!;
        Assert.Equal(new[] { id }, saved.KnowledgeCollections);
        Assert.NotEmpty(saved.Messages[^1].KnowledgeSources!);
        Assert.Equal(answer, saved.Messages[^1].Content);
        Assert.DoesNotContain(saved.Messages, message => message.Content.Contains("untrustedSourceData", StringComparison.Ordinal));
    }
}
