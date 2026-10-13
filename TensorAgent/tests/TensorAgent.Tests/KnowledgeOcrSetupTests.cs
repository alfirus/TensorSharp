// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorSharp.Knowledge;
using TensorSharp.Server.Knowledge;

namespace TensorAgent.Tests;

public sealed class KnowledgeOcrSetupTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-ocr-setup-" + Guid.NewGuid().ToString("N"));

    public void Dispose() { try { Directory.Delete(_root, true); } catch { } }

    private static PdfOcrSetupStatus Status(string state, long downloaded = 0, long total = 0, bool retryable = false) =>
        JsonSerializer.Deserialize<PdfOcrSetupStatus>(JsonSerializer.Serialize(new
        { state, stage = "PDF tools", downloadedBytes = downloaded, totalBytes = total, error = (string?)null, retryable }),
            new JsonSerializerOptions(JsonSerializerDefaults.Web))!;

    private static HttpClient Client(LoopbackServer server, bool authenticated = true)
    {
        var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl) };
        if (authenticated) client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={server.Token}");
        return client;
    }

    [Fact]
    public async Task SetupRemainsAuthenticatedAndReportsProgressWhilePreparationIsPending()
    {
        using var store = new KnowledgeStore(_root, new BuiltInKnowledgeExtractor());
        using var server = new LoopbackServer();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var status = Status("not_started");
        int calls = 0;
        server.MapKnowledge(store, () => Volatile.Read(ref status), async ct =>
        {
            Interlocked.Increment(ref calls);
            Volatile.Write(ref status, Status("downloading", 25, 100));
            entered.TrySetResult();
            await release.Task.WaitAsync(ct);
            Volatile.Write(ref status, Status("ready", 100, 100));
        });
        server.Start();
        using var stranger = Client(server, false);
        using var deniedStatus = await stranger.GetAsync("/api/knowledge/ocr/status");
        using var deniedPrepare = await stranger.PostAsJsonAsync("/api/knowledge/ocr/prepare", new { });
        Assert.Equal(HttpStatusCode.Forbidden, deniedStatus.StatusCode);
        Assert.Equal(HttpStatusCode.Forbidden, deniedPrepare.StatusCode);
        Assert.Equal(0, calls);

        using var client = Client(server);
        using var before = await client.GetFromJsonAsync<JsonDocument>("/api/knowledge/ocr/status");
        Assert.True(before!.RootElement.GetProperty("managed").GetBoolean());
        Assert.Equal("not_started", before.RootElement.GetProperty("status").GetProperty("state").GetString());
        Assert.Equal(0, calls);
        Task<HttpResponseMessage> preparing = client.PostAsJsonAsync("/api/knowledge/ocr/prepare", new { });
        try
        {
            await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
            using var progress = await client.GetFromJsonAsync<JsonDocument>("/api/knowledge/ocr/status");
            Assert.Equal("downloading", progress!.RootElement.GetProperty("status").GetProperty("state").GetString());
            Assert.Equal(25, progress.RootElement.GetProperty("status").GetProperty("downloadedBytes").GetInt64());
            Assert.False(preparing.IsCompleted);
        }
        finally { release.TrySetResult(); }
        using var complete = await preparing.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(HttpStatusCode.OK, complete.StatusCode);
        using var ready = await complete.Content.ReadFromJsonAsync<JsonDocument>();
        Assert.Equal("ready", ready!.RootElement.GetProperty("status").GetProperty("state").GetString());
        Assert.Equal(1, calls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task FailedSetupReturnsRetryableStatusAndAnExplicitRetryCanComplete(bool internalDeadline)
    {
        using var store = new KnowledgeStore(_root, new BuiltInKnowledgeExtractor());
        using var server = new LoopbackServer();
        var status = Status("not_started");
        int calls = 0;
        server.MapKnowledge(store, () => status, _ =>
        {
            if (++calls == 1)
            {
                status = Status("failed", retryable: true);
                if (internalDeadline) throw new OperationCanceledException("The setup deadline elapsed.");
                throw new IOException("Host detail must not replace the safe route error.");
            }
            status = Status("ready");
            return Task.CompletedTask;
        });
        server.Start();
        using var client = Client(server);
        using var failure = await client.PostAsJsonAsync("/api/knowledge/ocr/prepare", new { });
        Assert.Equal(HttpStatusCode.ServiceUnavailable, failure.StatusCode);
        string body = await failure.Content.ReadAsStringAsync();
        Assert.DoesNotContain("Host detail", body);
        using var failed = JsonDocument.Parse(body);
        Assert.True(failed.RootElement.GetProperty("status").GetProperty("retryable").GetBoolean());
        using var retried = await client.PostAsJsonAsync("/api/knowledge/ocr/prepare", new { });
        Assert.Equal(HttpStatusCode.OK, retried.StatusCode);
        using var ready = await retried.Content.ReadFromJsonAsync<JsonDocument>();
        Assert.Equal("ready", ready!.RootElement.GetProperty("status").GetProperty("state").GetString());
        Assert.Equal(2, calls);
    }

    [Fact]
    public async Task NativeOrManuallyConfiguredProviderDoesNotClaimManagedReadiness()
    {
        using var store = new KnowledgeStore(_root, new BuiltInKnowledgeExtractor());
        using var server = new LoopbackServer();
        server.MapKnowledge(store);
        server.Start();
        using var client = Client(server);
        using var status = await client.GetFromJsonAsync<JsonDocument>("/api/knowledge/ocr/status");
        using var preparation = await client.PostAsJsonAsync("/api/knowledge/ocr/prepare", new { });
        using var prepared = await preparation.Content.ReadFromJsonAsync<JsonDocument>();
        foreach (var response in new[] { status!, prepared! })
        {
            Assert.False(response.RootElement.GetProperty("managed").GetBoolean());
            Assert.Equal(JsonValueKind.Null, response.RootElement.GetProperty("status").ValueKind);
        }
    }

    [Fact]
    public async Task BatchReportsOnlyUnpublishedSetupFailuresByInputOrdinalAndPreservesSuccesses()
    {
        using var store = new KnowledgeStore(_root, new ProbeExtractor(async (context, ct) =>
        {
            string text = await File.ReadAllTextAsync(context.FilePath, ct);
            if (text == "fail") throw new PdfOcrSetupException("Private installation diagnostic.");
            return new KnowledgeExtraction { Segments = [new KnowledgeSegment(text)] };
        }));
        var collection = await store.CreateCollectionAsync("mixed batch");
        using var server = new LoopbackServer();
        server.MapKnowledge(store);
        server.Start();
        using var client = Client(server);
        string route = $"/api/knowledge/collections/{collection.Id}/documents";
        using var form = new MultipartFormDataContent();
        form.Add(new ByteArrayContent(Encoding.UTF8.GetBytes("first searchable document")), "files", "scan.pdf");
        form.Add(new ByteArrayContent(Encoding.UTF8.GetBytes("fail")), "files", "scan.pdf");
        form.Add(new ByteArrayContent(Encoding.UTF8.GetBytes("last searchable document")), "files", "notes.txt");
        using var response = await client.PostAsync(route, form);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        string body = await response.Content.ReadAsStringAsync();
        Assert.DoesNotContain("Private installation diagnostic", body);
        using var batch = JsonDocument.Parse(body);
        Assert.Equal(2, batch.RootElement.GetProperty("documents").GetArrayLength());
        var error = Assert.Single(batch.RootElement.GetProperty("errors").EnumerateArray());
        Assert.Equal(1, error.GetProperty("fileIndex").GetInt32());
        Assert.Equal("scan.pdf", error.GetProperty("fileName").GetString());
        Assert.Equal("pdf_ocr_setup_failed", error.GetProperty("code").GetString());
        Assert.Equal(2, store.ListDocuments(collection.Id).Count);
        var scan = store.ListDocuments(collection.Id).Single(d => d.FileName == "scan.pdf");
        var notes = store.ListDocuments(collection.Id).Single(d => d.FileName == "notes.txt");

        using var retry = new MultipartFormDataContent();
        retry.Add(new ByteArrayContent(Encoding.UTF8.GetBytes("retried searchable document")), "files", "scan.pdf");
        using var retried = await client.PostAsync(route, retry);
        Assert.Equal(HttpStatusCode.OK, retried.StatusCode);
        // Imports with the same name intentionally version the existing document.
        Assert.Equal(2, store.ListDocuments(collection.Id).Count);
        Assert.Equal(scan.Version + 1, store.ListDocuments(collection.Id).Single(d => d.Id == scan.Id).Version);
        Assert.Equal(notes.Version, store.ListDocuments(collection.Id).Single(d => d.Id == notes.Id).Version);
    }

    [Fact]
    public async Task ReindexSetupFailureHasAuthoritativeCodeAndLeavesExistingVersionReadable()
    {
        bool fail = false;
        using var store = new KnowledgeStore(_root, new ProbeExtractor((_, _) => fail
            ? throw new PdfOcrSetupException("Internal setup diagnostic.")
            : Task.FromResult(new KnowledgeExtraction { Segments = [new KnowledgeSegment("persistent evidence")] })));
        var collection = await store.CreateCollectionAsync("reindex");
        var original = await store.ImportAsync(collection.Id, "scan.pdf", new MemoryStream([1, 2, 3]));
        using var server = new LoopbackServer();
        server.MapKnowledge(store);
        server.Start();
        using var client = Client(server);
        string route = $"/api/knowledge/collections/{collection.Id}/documents/{original.Id}/reindex";
        fail = true;
        using var rejected = await client.PostAsJsonAsync(route, new { });
        Assert.Equal(HttpStatusCode.ServiceUnavailable, rejected.StatusCode);
        using var error = await rejected.Content.ReadFromJsonAsync<JsonDocument>();
        Assert.Equal("pdf_ocr_setup_failed", error!.RootElement.GetProperty("code").GetString());
        Assert.Equal(original.Version, Assert.Single(store.ListDocuments(collection.Id)).Version);
        Assert.Single(await store.SearchAsync("persistent", new() { CollectionIds = [collection.Id] }));
        fail = false;
        using var retried = await client.PostAsJsonAsync(route, new { });
        Assert.Equal(HttpStatusCode.OK, retried.StatusCode);
        Assert.Equal(original.Version + 1, Assert.Single(store.ListDocuments(collection.Id)).Version);
    }

    private sealed class ProbeExtractor(Func<KnowledgeExtractionContext, CancellationToken, Task<KnowledgeExtraction>> extract) : IKnowledgeExtractor
    {
        public Task<KnowledgeExtraction> ExtractAsync(KnowledgeExtractionContext context, CancellationToken cancellationToken = default) =>
            extract(context, cancellationToken);
    }

    [Fact]
    public void PdfToolsCacheIsSeparateFromDurableKnowledgeAndModelStorage()
    {
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache"));
        paths.EnsureCreated();
        Assert.StartsWith(paths.CacheRoot + Path.DirectorySeparatorChar, paths.PdfOcrDirectory, StringComparison.Ordinal);
        Assert.StartsWith(paths.DataRoot + Path.DirectorySeparatorChar, paths.KnowledgeDirectory, StringComparison.Ordinal);
        Assert.NotEqual(paths.ModelsDirectory, paths.PdfOcrDirectory);
        File.WriteAllText(Path.Combine(paths.KnowledgeDirectory, "original.txt"), "durable original");
        Directory.Delete(paths.CacheRoot, true);
        Assert.Equal("durable original", File.ReadAllText(Path.Combine(paths.KnowledgeDirectory, "original.txt")));
    }
}
