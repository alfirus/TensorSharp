// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using TensorSharp.AgentHost.Knowledge;
using TensorSharp.Knowledge;
using TensorSharp.Server.Knowledge;

namespace InferenceWeb.Tests;

public sealed class KnowledgeCitationTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "knowledge-citations-" + Guid.NewGuid().ToString("N"));
    private KnowledgeStore? _store;
    public void Dispose() { _store?.Dispose(); if (Directory.Exists(_root)) Directory.Delete(_root, true); }

    private async Task<(KnowledgeToolScope Scope, KnowledgeSearchResult Hit)> Prepare(string fileName = "source.txt")
    {
        _store = new KnowledgeStore(_root, new BuiltInKnowledgeExtractor());
        var collection = await _store.CreateCollectionAsync("citations");
        await _store.ImportAsync(collection.Id, fileName, new MemoryStream(Encoding.UTF8.GetBytes("Unique cobalt evidence.")));
        var scope = new KnowledgeToolScope(_store, new[] { collection.Id });
        var hits = await _store.SearchAsync("cobalt", new() { CollectionIds = scope.CollectionIds });
        scope.RecordSources(hits);
        return (scope, Assert.Single(hits));
    }

    private static void IssueCompleteCitation(KnowledgeToolScope scope, KnowledgeSearchResult hit)
    {
        var complete = scope.Store.ReadDocumentContext(hit.Reference, 64000, scope.CollectionIds)!;
        Assert.True(KnowledgeTools.TrySerializeCompleteDocumentEvidence(complete, 96000, out string payload));
        using var json = JsonDocument.Parse(payload);
        Assert.Equal("[K1]", Assert.Single(json.RootElement.GetProperty("sources").EnumerateArray())
            .GetProperty("citationMarker").GetString());
        scope.RecordCompleteDocumentEvidence(complete, payload);
    }

    [Fact]
    public async Task OrdinaryMarkersAreStableAndIssuedOnlyForBudgetedDeliveredEvidence()
    {
        var (scope, first) = await Prepare();
        var document = await scope.Store.ImportAsync(first.CollectionId, "second.txt",
            new MemoryStream(Encoding.UTF8.GetBytes("Second cobalt evidence.")));
        var second = (await scope.Store.SearchAsync("cobalt", new() { CollectionIds = scope.CollectionIds }))
            .Single(hit => hit.DocumentId == document.Id);
        var candidates = new[] { first, second };
        // A serialization preview may reserve names, but cannot authorize a citation.
        string preview = KnowledgeTools.SerializeResults(candidates, 12000, out _, scope: scope);
        using var previewJson = JsonDocument.Parse(preview);
        Assert.Equal(new[] { "[K1]", "[K2]" }, previewJson.RootElement.GetProperty("results").EnumerateArray()
            .Select(result => result.GetProperty("citationMarker").GetString()));
        Assert.Empty(KnowledgeCitations.UsedSources("[K1] [K2]", scope));

        int budget = 128;
        IReadOnlyList<KnowledgeSearchResult> included;
        string bounded;
        do { bounded = KnowledgeTools.SerializeResults(candidates, budget++, out included, scope: scope); }
        while (included.Count == 0 && budget < 2000);
        Assert.Single(included);
        Assert.True(Encoding.UTF8.GetByteCount(bounded) < budget);
        scope.RecordSources(included, issueCitationMarkers: true);
        // Retrieving/recording the excluded candidate separately is not marker delivery.
        scope.RecordSources(new[] { second });
        Assert.Single(KnowledgeCitations.UsedSources("[K1] [K2]", scope));
        Assert.Equal(("[K2]", 0), KnowledgeTools.NormalizeCitationUrls("[K2]", scope));

        string later = KnowledgeTools.SerializeResults(new[] { second }, 4000, out included, scope: scope);
        scope.RecordSources(included, issueCitationMarkers: true);
        Assert.Contains("\"citationMarker\":\"[K2]\"", later);
        string url = KnowledgeTools.SourceUrl(second.CollectionId, second.DocumentId, second.Version);
        Assert.Equal(($"Second fact[2]({url})。", 1), KnowledgeTools.NormalizeCitationUrls("Second fact[K2]。", scope));
        var onlyUsed = Assert.Single(KnowledgeCitations.UsedSources("Second fact[K2]。", scope));
        Assert.Equal(second.DocumentId, onlyUsed.DocumentId);
        Assert.Equal(url, onlyUsed.Reference);
        Assert.Equal(-1, onlyUsed.ChunkIndex);
        Assert.Null(onlyUsed.Location);
        Assert.Equal(2, KnowledgeCitations.UsedSources("Both facts[K1][K2]。", scope).Count);
        string firstUrl = KnowledgeTools.SourceUrl(first.CollectionId, first.DocumentId, first.Version);
        Assert.Equal(($"Both facts[1]({firstUrl})[2]({url})。", 2),
            KnowledgeTools.NormalizeCitationUrls("Both facts[K1][K2]。", scope));
        foreach (string adjacent in new[] { "【K1】【K2】", "［K1］［K2］", "【K1】[K2]", "[K1]［K2］" })
        {
            Assert.Equal(2, KnowledgeCitations.UsedSources("综合两份资料" + adjacent + "。", scope).Count);
            Assert.Equal(($"综合两份资料[1]({firstUrl})[2]({url})。", 2),
                KnowledgeTools.NormalizeCitationUrls("综合两份资料" + adjacent + "。", scope));
        }
        Assert.Equal(("[K1][outside]", 0), KnowledgeTools.NormalizeCitationUrls("[K1][outside]", scope));
        Assert.Equal(("[K3]", 0), KnowledgeTools.NormalizeCitationUrls("[K3]", scope));

        var request = new KnowledgeProtocolRequest(scope);
        const string json = "{\"source\":\"[K2]\"}";
        Assert.Equal(json, request.Normalize(json, structured: true));
        var response = JsonSerializer.SerializeToElement(request.WithSources(new { answer = json }));
        Assert.Equal(second.DocumentId, Assert.Single(response.GetProperty("knowledge_sources").EnumerateArray())
            .GetProperty("documentId").GetString());
        Assert.Contains("\"citationMarker\":\"[K2]\"", scope.GetBoundedEvidence());
    }

    [Fact]
    public async Task EagerSearchAndReadIssueTheSameMarkerAcrossDeliveryPaths()
    {
        var (seed, hit) = await Prepare();
        var scope = new KnowledgeToolScope(seed.Store, seed.CollectionIds);
        var messages = await KnowledgeRequestContext.PrepareAsync(
            new() { new() { Role = "user", Content = "cobalt" } }, scope, toolsAvailable: true);
        Assert.Contains("\"citationMarker\":\"[K1]\"", messages[^1].Content);
        Assert.Single(KnowledgeCitations.UsedSources("[K1]", scope));
        // Separate consumers may need the same source, while marker identity stays request-wide.
        var child = new KnowledgeRetrievalProgress();
        var read = KnowledgeTools.Execute(new() { Name = KnowledgeTools.Read,
            Arguments = new() { ["reference"] = hit.Reference } }, scope, retrieval: child);
        Assert.True(read.Ok);
        Assert.Contains("\"citationMarker\":\"[K1]\"", read.Content);
        Assert.Single(KnowledgeCitations.UsedSources("[K1]", scope));

        var searchScope = new KnowledgeToolScope(seed.Store, seed.CollectionIds);
        var search = KnowledgeTools.Execute(new() { Name = KnowledgeTools.Search,
            Arguments = new() { ["query"] = "cobalt" } }, searchScope);
        Assert.True(search.Ok);
        Assert.Contains("\"citationMarker\":\"[K1]\"", search.Content);
        Assert.Single(KnowledgeCitations.UsedSources("[K1]", searchScope));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task OrdinaryMarkersRejectOtherScopesAndChangedSourceVersions(bool delete)
    {
        var (scope, hit) = await Prepare();
        KnowledgeTools.SerializeResults(new[] { hit }, 4000, out var included, scope: scope);
        scope.RecordSources(included, issueCitationMarkers: true);
        var hidden = await scope.Store.CreateCollectionAsync("hidden");
        var other = new KnowledgeToolScope(scope.Store, new[] { hidden.Id });
        KnowledgeTools.SerializeResults(new[] { hit }, 4000, out var outside, scope: other);
        other.RecordSources(outside, issueCitationMarkers: true);
        foreach (string marker in new[] { "[K1]", "【K1】", "［K1］" })
        {
            Assert.Empty(KnowledgeCitations.UsedSources(marker, other));
            Assert.Equal((marker, 0), KnowledgeTools.NormalizeCitationUrls(marker, other));
        }
        if (delete) await scope.Store.DeleteDocumentAsync(hit.CollectionId, hit.DocumentId);
        else await scope.Store.ReindexAsync(hit.CollectionId, hit.DocumentId);
        foreach (string marker in new[] { "[K1]", "【K1】", "［K1］" })
        {
            Assert.Empty(KnowledgeCitations.UsedSources(marker, scope));
            Assert.Equal((marker, 0), KnowledgeTools.NormalizeCitationUrls(marker, scope));
        }
    }

    [Fact]
    public async Task CompleteDocumentCannotRebindAnEarlierOrdinaryMarker()
    {
        var (scope, first) = await Prepare();
        KnowledgeTools.SerializeResults(new[] { first }, 4000, out var included, scope: scope);
        scope.RecordSources(included, issueCitationMarkers: true);
        var document = await scope.Store.ImportAsync(first.CollectionId, "complete.txt",
            new MemoryStream(Encoding.UTF8.GetBytes("Complete indigo evidence.")));
        var second = Assert.Single(await scope.Store.SearchAsync("indigo", new() { CollectionIds = scope.CollectionIds }));
        var complete = scope.Store.ReadDocumentContext(second.Reference, 64000, scope.CollectionIds)!;
        Assert.True(KnowledgeTools.TrySerializeCompleteDocumentEvidence(complete, 96000, out string conflicting));
        Assert.Throws<ArgumentException>(() => scope.RecordCompleteDocumentEvidence(complete, conflicting));
        Assert.Equal(first.DocumentId, Assert.Single(KnowledgeCitations.UsedSources("[K1]", scope)).DocumentId);
        Assert.Empty(KnowledgeCitations.UsedSources("[K2]", scope));
        Assert.True(scope.TrySerializeCompleteDocumentEvidence(complete, 96000, out string payload));
        Assert.Contains("\"citationMarker\":\"[K2]\"", payload);
        scope.RecordCompleteDocumentEvidence(complete, payload);
        Assert.Equal(first.DocumentId, Assert.Single(KnowledgeCitations.UsedSources("[K1]", scope)).DocumentId);
        Assert.Equal(document.Id, Assert.Single(KnowledgeCitations.UsedSources("[K2]", scope)).DocumentId);
    }

    [Fact]
    public async Task CompleteDocumentMarkerBindsOnlyItsIssuedDocumentAndProducesCanonicalUiMetadata()
    {
        var (scope, hit) = await Prepare();
        Assert.Equal(("[K1]", 0), KnowledgeTools.NormalizeCitationUrls("[K1]", scope));
        Assert.Empty(KnowledgeCitations.UsedSources("[K1]", scope));
        IssueCompleteCitation(scope, hit);
        var other = await scope.Store.ImportAsync(hit.CollectionId, "other.txt",
            new MemoryStream(Encoding.UTF8.GetBytes("Unused copper evidence.")));
        scope.RecordSources(await scope.Store.SearchAsync("copper", new() { CollectionIds = scope.CollectionIds }));
        string url = KnowledgeTools.SourceUrl(hit.CollectionId, hit.DocumentId, hit.Version);
        string answer = "Cobalt evidence[K1]。来源：[K1]";
        var normalized = KnowledgeTools.NormalizeCitationUrls(answer, scope);
        Assert.Equal(answer.Replace("[K1]", $"[1]({url})", StringComparison.Ordinal), normalized.Text);
        Assert.Equal(2, normalized.RepairCount);
        var source = Assert.Single(KnowledgeCitations.UsedSources(answer, scope));
        Assert.Equal(hit.DocumentId, source.DocumentId);
        Assert.NotEqual(other.Id, source.DocumentId);
        Assert.Equal(url, source.Reference);
        Assert.Equal(-1, source.ChunkIndex);
        Assert.Null(source.Location);

        var shown = new StringBuilder(answer);
        var frames = WebUiChatService.KnowledgeCitationFrames(shown, scope)
            .Select(frame => JsonSerializer.SerializeToElement(frame)).ToArray();
        Assert.Equal(normalized.Text, frames[0].GetProperty("replace").GetString());
        Assert.Equal(normalized.Text, shown.ToString());
        var metadata = Assert.Single(frames[^1].GetProperty("knowledgeSources").EnumerateArray());
        Assert.Equal(url, metadata.GetProperty("sourceUrl").GetString());
        Assert.Equal(url, metadata.GetProperty("reference").GetString());
        Assert.Equal(hit.DocumentId, metadata.GetProperty("documentId").GetString());
    }

    [Fact]
    public async Task StructuredMarkerKeepsValidatedJsonAndBindsCanonicalSourcesSeparately()
    {
        var (scope, hit) = await Prepare();
        IssueCompleteCitation(scope, hit);
        const string json = "{\"answer\":\"Cobalt\",\"citation\":\"[K1]\"}";
        var request = new KnowledgeProtocolRequest(scope);
        Assert.Equal(json, request.Normalize(json, structured: true));
        var payload = JsonSerializer.SerializeToElement(request.WithSources(new { answer = json }));
        Assert.Equal(json, payload.GetProperty("answer").GetString());
        var source = Assert.Single(payload.GetProperty("knowledge_sources").EnumerateArray());
        Assert.Equal(KnowledgeTools.SourceUrl(hit.CollectionId, hit.DocumentId, hit.Version), source.GetProperty("sourceUrl").GetString());
        Assert.Equal(0, payload.GetProperty("knowledge_citation_repairs").GetInt32());
    }

    [Theory]
    [InlineData("【K1】")]
    [InlineData("［K1］")]
    public async Task LocalizedMarkerBracketsPreserveIssuedIdentityInChineseProseAndStructuredJson(string marker)
    {
        var (scope, hit) = await Prepare();
        string paragraph = "从资料来看，已有结论有明确的原文依据" + marker + "。尚未说明的部分不能据此推断。来源：" + marker;
        Assert.Empty(KnowledgeCitations.UsedSources(paragraph, scope));
        Assert.Equal((paragraph, 0), KnowledgeTools.NormalizeCitationUrls(paragraph, scope));
        IssueCompleteCitation(scope, hit);
        string url = KnowledgeTools.SourceUrl(hit.CollectionId, hit.DocumentId, hit.Version);
        Assert.Equal((paragraph.Replace(marker, $"[1]({url})", StringComparison.Ordinal), 2),
            KnowledgeTools.NormalizeCitationUrls(paragraph, scope));
        var used = Assert.Single(KnowledgeCitations.UsedSources(paragraph, scope));
        Assert.Equal(url, used.Reference);
        Assert.Equal(-1, used.ChunkIndex);
        Assert.Null(used.Location);

        string json = "{\"answer\":\"结论有原文依据\",\"citation\":\"" + marker + "\"}";
        var request = new KnowledgeProtocolRequest(scope);
        Assert.Equal(json, request.Normalize(json, structured: true));
        var payload = JsonSerializer.SerializeToElement(request.WithSources(new { answer = json }));
        Assert.Equal(json, payload.GetProperty("answer").GetString());
        Assert.Equal(url, Assert.Single(payload.GetProperty("knowledge_sources").EnumerateArray()).GetProperty("sourceUrl").GetString());

        foreach (string unsupported in new[]
        {
            "【K2】", "［K2］", "【k1】", "［K01］", "【K1]", "[K1】", "［K1】", "【K1］",
            "[" + marker + "]", "【" + marker + "】", "［" + marker + "］", "\\" + marker, "!" + marker,
            marker + "[outside]", marker + "【outside】", marker + "［outside］", marker + "(https://example.com)",
            "https://example.com/" + marker, "//example.com/" + marker, "/prefix/" + marker,
            "https://example.com?q=value" + marker, "https://example.com/" + marker + marker,
        })
        {
            Assert.Equal((unsupported, 0), KnowledgeTools.NormalizeCitationUrls(unsupported, scope));
            Assert.Empty(KnowledgeCitations.UsedSources(unsupported, scope));
        }
    }

    [Fact]
    public async Task MarkersNeverRepairUnknownScopeExternalUrlsOrModelCreatedLinks()
    {
        var (scope, hit) = await Prepare();
        IssueCompleteCitation(scope, hit);
        foreach (string unsupported in new[]
        {
            "[K2]", "[K10]", "[k1]", "[[K1]]", "\\[K1]", "![K1]", "[K1](https://example.com)",
            "https://example.com/[K1]", "//example.com/[K1]", "/prefix/[K1]", "https://example.com?q=value[K1]",
            "https://api.knowledge/collections/" + hit.CollectionId + "/documents/" + hit.DocumentId + "/source?version=1",
        })
        {
            Assert.Equal((unsupported, 0), KnowledgeTools.NormalizeCitationUrls(unsupported, scope));
            Assert.Empty(KnowledgeCitations.UsedSources(unsupported, scope));
        }
        var otherRequest = new KnowledgeToolScope(scope.Store, scope.CollectionIds);
        otherRequest.RecordSources(scope.Sources);
        Assert.Equal(("[K1]", 0), KnowledgeTools.NormalizeCitationUrls("[K1]", otherRequest));
        Assert.Empty(KnowledgeCitations.UsedSources("[K1]", otherRequest));

        var hidden = await scope.Store.CreateCollectionAsync("hidden");
        var hiddenScope = new KnowledgeToolScope(scope.Store, new[] { hidden.Id });
        var complete = scope.Store.ReadDocumentContext(hit.Reference, 64000, scope.CollectionIds)!;
        Assert.True(KnowledgeTools.TrySerializeCompleteDocumentEvidence(complete, 96000, out string payload));
        Assert.Throws<ArgumentException>(() => hiddenScope.RecordCompleteDocumentEvidence(complete, payload));
        Assert.Equal(("[K1]", 0), KnowledgeTools.NormalizeCitationUrls("[K1]", hiddenScope));
        Assert.Empty(KnowledgeCitations.UsedSources("[K1]", hiddenScope));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task IssuedMarkerCannotEndorseReindexedOrDeletedSources(bool delete)
    {
        var (scope, hit) = await Prepare();
        IssueCompleteCitation(scope, hit);
        if (delete) await scope.Store.DeleteDocumentAsync(hit.CollectionId, hit.DocumentId);
        else await scope.Store.ReindexAsync(hit.CollectionId, hit.DocumentId);
        Assert.Equal(("Evidence [K1]", 0), KnowledgeTools.NormalizeCitationUrls("Evidence [K1]", scope));
        Assert.Empty(KnowledgeCitations.UsedSources("Evidence [K1]", scope));
    }

    [Fact]
    public async Task OnlyExactRetrievedFilenameUrlsBindToAnAuthoritativeSource()
    {
        var (scope, hit) = await Prepare("星图 notes (1).txt");
        string wrong = KnowledgeTools.SourceUrl(hit.CollectionId, hit.FileName, hit.Version);
        string right = KnowledgeTools.SourceUrl(hit.CollectionId, hit.DocumentId, hit.Version);
        string text = $"[notebook]({wrong}) and <{wrong}>.";
        var repaired = KnowledgeTools.NormalizeCitationUrls(text, scope);
        Assert.Equal(text.Replace(wrong, right, StringComparison.Ordinal), repaired.Text);
        Assert.Equal(2, repaired.RepairCount);
        foreach (string unchanged in new[]
        {
            right, wrong.Replace("version=1", "version=2"), wrong.Replace(hit.CollectionId, new string('a', 32)),
            KnowledgeTools.SourceUrl(hit.CollectionId, Guid.NewGuid().ToString("N"), 1),
            KnowledgeTools.SourceUrl(hit.CollectionId, "unknown.txt", 1),
            "https://example.com" + wrong, "/prefix" + wrong, wrong + "&other=true", wrong + "#page=2", wrong + "0",
            wrong + ".5", wrong.Replace("notes", "NOTES"),
        })
        {
            Assert.Equal((unchanged, 0), KnowledgeTools.NormalizeCitationUrls(unchanged, scope));
        }
        Assert.Equal((wrong, 0), KnowledgeTools.NormalizeCitationUrls(wrong, null));
        var unretrieved = new KnowledgeToolScope(scope.Store, scope.CollectionIds);
        Assert.Equal((wrong, 0), KnowledgeTools.NormalizeCitationUrls(wrong, unretrieved));
        await scope.Store.ReindexAsync(hit.CollectionId, hit.DocumentId);
        Assert.Equal((wrong, 0), KnowledgeTools.NormalizeCitationUrls(wrong, scope));
    }

    [Fact]
    public async Task AmbiguousEvidenceOrDeletedSourcesAreNotRepaired()
    {
        var (scope, hit) = await Prepare();
        string wrong = KnowledgeTools.SourceUrl(hit.CollectionId, hit.FileName, hit.Version);
        // A duplicated filename from an external retriever must not choose one ID.
        var second = await scope.Store.ImportAsync(hit.CollectionId, "other.txt", new MemoryStream(Encoding.UTF8.GetBytes("Other cobalt evidence.")));
        scope.RecordSources(new[] { hit with { Reference = "kb://ambiguous", DocumentId = second.Id } });
        Assert.Equal((wrong, 0), KnowledgeTools.NormalizeCitationUrls(wrong, scope));
        var single = new KnowledgeToolScope(scope.Store, scope.CollectionIds);
        single.RecordSources(new[] { hit, hit with { Reference = hit.Reference + "neighbor", ChunkIndex = 1 } });
        Assert.Equal(1, KnowledgeTools.NormalizeCitationUrls(wrong, single).RepairCount);
        await scope.Store.DeleteDocumentAsync(hit.CollectionId, hit.DocumentId);
        Assert.Equal((wrong, 0), KnowledgeTools.NormalizeCitationUrls(wrong, single));
    }

    [Theory]
    [InlineData("来源：", "。")]
    [InlineData("成绩较好，", "；")]
    [InlineData("评语（", "）")]
    [InlineData("依据【", "】")]
    public async Task ChinesePunctuationDelimitsExactCitationsAndFilenameRepairs(string before, string after)
    {
        var (scope, hit) = await Prepare("成绩单.txt");
        string url = KnowledgeTools.SourceUrl(hit.CollectionId, hit.DocumentId, hit.Version);
        string wrong = KnowledgeTools.SourceUrl(hit.CollectionId, hit.FileName, hit.Version);
        Assert.Equal(hit.DocumentId,
            Assert.Single(KnowledgeCitations.UsedSources(before + url + after, scope)).DocumentId);
        Assert.Equal(hit,
            Assert.Single(KnowledgeCitations.UsedSources(before + hit.Reference + after, scope)));
        Assert.Equal((before + url + after, 1),
            KnowledgeTools.NormalizeCitationUrls(before + wrong + after, scope));
        var frames = WebUiChatService.KnowledgeCitationFrames(new StringBuilder(before + wrong + after), scope)
            .Select(frame => JsonSerializer.SerializeToElement(frame)).ToArray();
        Assert.Equal(hit.DocumentId, Assert.Single(frames[^1].GetProperty("knowledgeSources").EnumerateArray())
            .GetProperty("documentId").GetString());

        foreach (string prefix in new[] { "https://untrusted.example", "//untrusted.example", "/prefix" })
        {
            Assert.Empty(KnowledgeCitations.UsedSources(before + prefix + url + after, scope));
            string external = before + prefix + wrong + after;
            Assert.Equal((external, 0), KnowledgeTools.NormalizeCitationUrls(external, scope));
        }
    }

    [Fact]
    public async Task FinalParserFlushIsNormalizedBeforeDoneAndCarriesExplicitRepairCount()
    {
        var (scope, hit) = await Prepare();
        string wrong = KnowledgeTools.SourceUrl(hit.CollectionId, hit.FileName, hit.Version);
        string right = KnowledgeTools.SourceUrl(hit.CollectionId, hit.DocumentId, hit.Version);
        using var session = new ChatSession();
        var shown = new StringBuilder("Evidence: ");
        var frames = WebUiChatService.FinalFrames(new FlushParser($"[source]({wrong})"), false, null,
            session, Stopwatch.StartNew(), 1, 0, 0, false, shown, knowledge: scope)
            .Select(frame => JsonSerializer.SerializeToElement(frame)).ToArray();
        Assert.Equal($"Evidence: [source]({right})", shown.ToString());
        Assert.Contains(wrong, frames[0].GetProperty("token").GetString());
        Assert.Equal(shown.ToString(), frames[1].GetProperty("replace").GetString());
        Assert.Equal(1, frames[2].GetProperty("knowledgeCitationRepairs").GetInt32());
        Assert.Equal(hit.DocumentId, Assert.Single(frames[3].GetProperty("knowledgeSources").EnumerateArray()).GetProperty("documentId").GetString());
        Assert.True(frames[^1].GetProperty("done").GetBoolean());
    }

    [Fact]
    public async Task OnlyFinalAnswerCitationsSelectSourcesAndDocumentLinksDoNotClaimUnusedChunks()
    {
        var (scope, hit) = await Prepare();
        var other = await scope.Store.ImportAsync(hit.CollectionId, "other.txt", new MemoryStream(Encoding.UTF8.GetBytes("Unused copper evidence.")));
        scope.RecordSources(await scope.Store.SearchAsync("copper", new() { CollectionIds = scope.CollectionIds }));
        scope.RecordSources(new[] { hit with { Reference = hit.Reference + "neighbor", ChunkIndex = 1, Location = new() { Page = 2 } } });
        string url = KnowledgeTools.SourceUrl(hit.CollectionId, hit.DocumentId, hit.Version);

        Assert.Empty(KnowledgeCitations.UsedSources("The answer does not cite a source.", scope));
        Assert.Empty(KnowledgeCitations.UsedSources(hit.FileName, scope));
        var cited = Assert.Single(KnowledgeCitations.UsedSources($"Evidence [source]({url}) and <{url}>。", scope));
        Assert.Equal(hit.DocumentId, cited.DocumentId);
        Assert.NotEqual(other.Id, cited.DocumentId);
        Assert.Equal(url, cited.Reference);
        Assert.Equal(-1, cited.ChunkIndex);
        Assert.Null(cited.Location);
        Assert.Equal(hit, Assert.Single(KnowledgeCitations.UsedSources($"Evidence [{hit.Reference}]", scope)));
        Assert.Equal(hit, Assert.Single(KnowledgeCitations.UsedSources($"成绩较好【{hit.Reference}】。", scope)));
        Assert.Equal(hit, Assert.Single(KnowledgeCitations.UsedSources($"[{hit.Reference}] [{hit.Reference}neighbor]", scope)));

        var frames = WebUiChatService.KnowledgeCitationFrames(new StringBuilder("No citations."), scope)
            .Select(frame => JsonSerializer.SerializeToElement(frame)).ToArray();
        Assert.Empty(Assert.Single(frames).GetProperty("knowledgeSources").EnumerateArray());
    }

    [Fact]
    public async Task SourceSelectionRejectsUnretrievedStaleDeletedAndNonExactReferences()
    {
        var (scope, hit) = await Prepare();
        string url = KnowledgeTools.SourceUrl(hit.CollectionId, hit.DocumentId, hit.Version);
        foreach (string unsupported in new[]
        {
            "https://untrusted.example" + url, "/prefix" + url, url + "0", url + ".5", url + "&version=2",
            url.Replace("version=1", "version=2"), KnowledgeTools.SourceUrl(hit.CollectionId, hit.FileName, hit.Version),
            KnowledgeTools.SourceUrl(new string('a', 32), hit.DocumentId, hit.Version), hit.Reference + "9",
        }) Assert.Empty(KnowledgeCitations.UsedSources(unsupported, scope));
        Assert.Empty(KnowledgeCitations.UsedSources(url, new KnowledgeToolScope(scope.Store, scope.CollectionIds)));
        await scope.Store.ReindexAsync(hit.CollectionId, hit.DocumentId);
        Assert.Empty(KnowledgeCitations.UsedSources(url, scope));
        await scope.Store.DeleteDocumentAsync(hit.CollectionId, hit.DocumentId);
        Assert.Empty(KnowledgeCitations.UsedSources(hit.Reference, scope));
    }

    [Fact]
    public async Task LateCitedDocumentRetainsProvenanceAfterManyEarlierChunks()
    {
        var (scope, hit) = await Prepare();
        scope.RecordSources(Enumerable.Range(1, 50).Select(index => hit with
        { Reference = hit.Reference + "-" + index, ChunkIndex = index }));
        var late = await scope.Store.ImportAsync(hit.CollectionId, "late.txt", new MemoryStream(Encoding.UTF8.GetBytes("Late indigo evidence.")));
        scope.RecordSources(await scope.Store.SearchAsync("indigo", new() { CollectionIds = scope.CollectionIds }));
        string url = KnowledgeTools.SourceUrl(late.CollectionId, late.Id, late.Version);
        Assert.Equal(late.Id, Assert.Single(KnowledgeCitations.UsedSources($"Used [source]({url}).", scope)).DocumentId);
    }

    private sealed class FlushParser(string final) : IOutputParser
    {
        public bool HasThinkingSupport => false;
        public bool HasToolSupport => false;
        public bool AlwaysRequired => false;
        public void Init(bool enableThinking, List<ToolFunction>? tools) { }
        public ParsedOutput Add(string text, bool done) => new() { Content = done ? final : "" };
    }
}
