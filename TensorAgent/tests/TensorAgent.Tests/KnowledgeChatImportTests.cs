// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Sessions;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.Knowledge;

namespace TensorAgent.Tests;

public sealed class KnowledgeChatImportTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-chat-knowledge-" + Guid.NewGuid().ToString("N"));
    private readonly string _uploads;
    private readonly ConversationStore _conversations;
    private readonly KnowledgeStore _knowledge;
    private readonly CodeArtifactStore _artifacts;
    private readonly LoopbackServer _server;
    private readonly HttpClient _client;

    public KnowledgeChatImportTests()
    {
        _uploads = Path.Combine(_root, "uploads");
        Directory.CreateDirectory(_uploads);
        _conversations = new ConversationStore(Path.Combine(_root, "chats"));
        _knowledge = new KnowledgeStore(Path.Combine(_root, "knowledge"), new BuiltInKnowledgeExtractor());
        _artifacts = new CodeArtifactStore(Path.Combine(_root, "artifacts"));
        _server = new LoopbackServer();
        _server.MapKnowledge(_knowledge);
        _server.MapKnowledgeChats(_knowledge, _conversations, _uploads, _artifacts);
        _server.Start();
        _client = new HttpClient { BaseAddress = new Uri(_server.BaseUrl) };
        _client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={_server.Token}");
    }

    public void Dispose()
    {
        _client.Dispose();
        _server.Dispose();
        _knowledge.Dispose();
        try { Directory.Delete(_root, true); } catch (IOException) { }
    }

    private Conversation Chat(string title, params StoredMessage[] messages)
    {
        var chat = new Conversation { Title = title, ModelId = "test-model", Messages = [.. messages] };
        _conversations.Save(chat);
        return chat;
    }

    private async Task<JsonElement> Import(string collection, params string[] ids)
    {
        using var response = await _client.PostAsJsonAsync($"/api/knowledge/collections/{collection}/chats", new { conversationIds = ids });
        string body = await response.Content.ReadAsStringAsync();
        Assert.True(response.IsSuccessStatusCode, response.StatusCode + ": " + body);
        return JsonSerializer.Deserialize<JsonElement>(body);
    }

    [Fact]
    public async Task PickerListsOnlyNonemptyChatsWithSearchAndPagination()
    {
        _conversations.Create();
        Chat("Orchard planning", new StoredMessage { Content = "first" });
        Chat("ORCHARD harvest", new StoredMessage { Content = "second" });
        Chat("Unrelated", new StoredMessage { Content = "third" });
        var page = await _client.GetFromJsonAsync<JsonElement>("/api/knowledge/chats?q=orchard&limit=1");
        Assert.Equal(2, page.GetProperty("total").GetInt32());
        Assert.Single(page.GetProperty("chats").EnumerateArray());
        Assert.Equal(1, page.GetProperty("nextOffset").GetInt32());
        var next = await _client.GetFromJsonAsync<JsonElement>("/api/knowledge/chats?q=orchard&limit=1&offset=1");
        Assert.NotEqual(page.GetProperty("chats")[0].GetProperty("id").GetString(), next.GetProperty("chats")[0].GetProperty("id").GetString());
        Assert.Equal(JsonValueKind.Null, next.GetProperty("nextOffset").ValueKind);
        Assert.Equal(1, next.GetProperty("chats")[0].GetProperty("messageCount").GetInt32());
        var all = await _client.GetFromJsonAsync<JsonElement>("/api/knowledge/chats");
        Assert.Equal(3, all.GetProperty("total").GetInt32());
    }

    [Fact]
    public async Task ImportedTranscriptRetrievesBothSidesAndReimportsInPlace()
    {
        var collection = await _knowledge.CreateCollectionAsync("Saved chats");
        var chat = Chat("Garden decision", new StoredMessage { Content = "The copper orchard needs water." },
            new StoredMessage { Role = "assistant", Content = "Use sapphire irrigation on Tuesdays.", Thinking = "hidden-azimuth-reasoning" });
        var imported = await Import(collection.Id, chat.Id, chat.Id);
        Assert.Empty(imported.GetProperty("errors").EnumerateArray());
        Assert.Equal(chat.Id, Assert.Single(imported.GetProperty("importedChats").EnumerateArray()).GetProperty("id").GetString());
        string documentId = Assert.Single(imported.GetProperty("documents").EnumerateArray()).GetProperty("id").GetString()!;

        using var response = await _client.PostAsJsonAsync("/api/knowledge/search", new { query = "sapphire irrigation", collectionIds = new[] { collection.Id } });
        var result = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.True(response.IsSuccessStatusCode, result.ToString());
        var found = Assert.Single(result.GetProperty("results").EnumerateArray());
        Assert.Contains("sapphire irrigation", found.GetProperty("text").GetString());
        Assert.DoesNotContain("hidden-azimuth", found.GetProperty("text").GetString());
        Assert.Equal("Chats/" + chat.Id + "/conversation.md", found.GetProperty("relativePath").GetString());
        var userResult = await _knowledge.SearchAsync("copper orchard", new KnowledgeSearchOptions { CollectionIds = [collection.Id] });
        Assert.Contains(userResult, source => source.Text.Contains("copper orchard", StringComparison.Ordinal));
        Assert.Empty(await _knowledge.SearchAsync("hidden-azimuth-reasoning", new KnowledgeSearchOptions { CollectionIds = [collection.Id] }));
        Assert.Empty(await _knowledge.SearchAsync("sapphire", new KnowledgeSearchOptions { CollectionIds = [] }));

        var unchanged = await Import(collection.Id, chat.Id);
        Assert.Equal(documentId, unchanged.GetProperty("documents")[0].GetProperty("id").GetString());
        Assert.Equal(1, unchanged.GetProperty("documents")[0].GetProperty("version").GetInt32());
        chat.Title = "Renamed garden decision";
        chat.Messages.Add(new StoredMessage { Role = "assistant", Content = "The revised heliotrope schedule is Fridays." });
        _conversations.Save(chat);
        var refreshed = await Import(collection.Id, chat.Id);
        Assert.Equal(documentId, refreshed.GetProperty("documents")[0].GetProperty("id").GetString());
        Assert.Equal(2, refreshed.GetProperty("documents")[0].GetProperty("version").GetInt32());
        Assert.Single(_knowledge.ListDocuments(collection.Id));
        Assert.Single(await _knowledge.SearchAsync("heliotrope", new KnowledgeSearchOptions { CollectionIds = [collection.Id] }));
        Assert.Equal(3, _conversations.Load(chat.Id)!.Messages.Count);
    }

    [Fact]
    public async Task LocalAttachmentsGeneratedMediaAndArtifactsOutliveChatAndUploadDeletion()
    {
        await File.WriteAllTextAsync(Path.Combine(_uploads, "input.txt"), "Attached nebula measurements");
        byte[] image = Convert.FromBase64String("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/gZ8AAAAASUVORK5CYII=");
        await File.WriteAllBytesAsync(Path.Combine(_uploads, "picture.png"), image);
        await File.WriteAllBytesAsync(Path.Combine(_uploads, "answer.wav"), [1, 2, 3, 4]);
        string artifactDirectory = Path.Combine(_artifacts.Root, "run1", "out");
        Directory.CreateDirectory(artifactDirectory);
        await File.WriteAllTextAsync(Path.Combine(artifactDirectory, "answer.txt"), "Produced quasar calculation");
        var chat = Chat("Astronomy", new StoredMessage { Content = "Look at my measurements.",
            TextFilePaths = ["input.txt"], TextFileNames = ["measurements.txt"], ImagePaths = ["picture.png"],
            Attachments = [new StoredAttachment { File = "input.txt", FileName = "measurements.txt" },
                new StoredAttachment { File = "picture.png", FileName = "picture", MediaType = "image" }] },
            new StoredMessage { Role = "assistant", Content = "Here are the results.", ImageUrl = "/uploads/picture.png",
                AudioUrl = "/uploads/answer.wav", Artifacts = [new StoredArtifact { Name = "answer.txt", Url = "/api/code/artifacts/run1/out/answer.txt" }] });
        var collection = await _knowledge.CreateCollectionAsync("Astronomy");
        var imported = await Import(collection.Id, chat.Id);
        Assert.Empty(imported.GetProperty("errors").EnumerateArray());
        Assert.Equal(4, imported.GetProperty("importedChats")[0].GetProperty("attachmentCount").GetInt32());
        Assert.Equal(5, imported.GetProperty("documents").GetArrayLength());
        _conversations.Delete(chat.Id);
        Directory.Delete(_uploads, true);
        Directory.Delete(_artifacts.Root, true);
        var documents = _knowledge.ListDocuments(collection.Id);
        Assert.Equal(2, documents.Count(document => document.Kind == "media"));
        var picture = Assert.Single(documents, document => document.FileName == "picture.png");
        using var original = await _client.GetAsync($"/api/knowledge/collections/{collection.Id}/documents/{picture.Id}/source?version={picture.Version}&inline=1");
        Assert.Equal(HttpStatusCode.OK, original.StatusCode);
        Assert.Equal("image/png", original.Content.Headers.ContentType!.MediaType);
        Assert.Equal(image, await original.Content.ReadAsByteArrayAsync());
        Assert.Contains(await _knowledge.SearchAsync("nebula measurements", new KnowledgeSearchOptions { CollectionIds = [collection.Id] }),
            source => source.FileName == "measurements.txt" && source.Text.Contains("nebula", StringComparison.Ordinal));
        Assert.Single(await _knowledge.SearchAsync("quasar calculation", new KnowledgeSearchOptions { CollectionIds = [collection.Id] }));
        var importedAgain = await Import(collection.Id, chat.Id);
        Assert.Empty(importedAgain.GetProperty("importedChats").EnumerateArray());
        Assert.Equal(5, _knowledge.ListDocuments(collection.Id).Count);
    }

    [Fact]
    public async Task GeneratedAndMisnamedMediaUseTheirStoredExtensionForIndexingAndInlineThumbnails()
    {
        byte[] image = Convert.FromBase64String("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/gZ8AAAAASUVORK5CYII=");
        await File.WriteAllBytesAsync(Path.Combine(_uploads, "generated.png"), image);
        await File.WriteAllBytesAsync(Path.Combine(_uploads, "misnamed.png"), image);
        var chat = Chat("Generated pictures", new StoredMessage { Content = "Generated image result", ImagePaths = ["generated.png", "misnamed.png"],
            Attachments = [new StoredAttachment { File = "generated.png", FileName = "Generated image", MediaType = "image" },
                new StoredAttachment { File = "misnamed.png", FileName = "Portrait.txt", MediaType = "image" }] });
        var collection = await _knowledge.CreateCollectionAsync("Pictures");
        var imported = await Import(collection.Id, chat.Id);
        Assert.Empty(imported.GetProperty("errors").EnumerateArray());
        var media = _knowledge.ListDocuments(collection.Id).Where(source => source.Kind == "media").ToArray();
        Assert.Equal(new[] { "Generated image.png", "Portrait.png" }, media.Select(source => source.FileName).Order(StringComparer.Ordinal));
        foreach (var source in media)
        {
            Assert.Equal(KnowledgeDocumentStatus.MetadataOnly, source.Status);
            using var response = await _client.GetAsync($"/api/knowledge/collections/{collection.Id}/documents/{source.Id}/source?version={source.Version}&inline=1");
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            Assert.Equal("image/png", response.Content.Headers.ContentType!.MediaType);
            Assert.Equal(image, await response.Content.ReadAsByteArrayAsync());
        }
    }

    [Fact]
    public async Task BatchKeepsGoodTranscriptsAndSeparatelyReportsMissingChatsAndAttachments()
    {
        var collection = await _knowledge.CreateCollectionAsync("History");
        var good = Chat("Good", new StoredMessage { Content = "Visible historical conversation", ImagePaths = ["expired.png"] });
        var empty = _conversations.Create();
        string missing = Guid.NewGuid().ToString("N");
        var imported = await Import(collection.Id, good.Id, empty.Id, missing);
        Assert.Equal(good.Id, Assert.Single(imported.GetProperty("importedChats").EnumerateArray()).GetProperty("id").GetString());
        Assert.Single(imported.GetProperty("documents").EnumerateArray());
        var errors = imported.GetProperty("errors").EnumerateArray().ToArray();
        Assert.Equal(3, errors.Length);
        Assert.Equal("expired.png", Assert.Single(errors, error => error.GetProperty("conversationId").GetString() == good.Id).GetProperty("fileName").GetString());
        Assert.All(errors.Where(error => error.GetProperty("conversationId").GetString() != good.Id), error => Assert.False(error.TryGetProperty("fileName", out _)));
        var source = Assert.Single(_knowledge.ListDocuments(collection.Id));
        var hits = await _knowledge.SearchAsync("historical conversation", new KnowledgeSearchOptions { CollectionIds = [collection.Id] });
        // The filename conversation.md also matches the header chunk. Require
        // the actual transcript body and the correct imported source identity.
        Assert.All(hits, hit => Assert.Equal(source.Id, hit.DocumentId));
        Assert.Contains(hits, hit => hit.Text.Contains("Visible historical conversation", StringComparison.Ordinal));
    }

    [Fact]
    public async Task BothHistoryRoutesRequireAuthenticationAndInvalidSelectionDoesNotImport()
    {
        var collection = await _knowledge.CreateCollectionAsync("History");
        var chat = Chat("Good", new StoredMessage { Content = "Selected words" });
        using var unauthenticated = new HttpClient { BaseAddress = _client.BaseAddress };
        using var listing = await unauthenticated.GetAsync("/api/knowledge/chats");
        Assert.Equal(HttpStatusCode.Forbidden, listing.StatusCode);
        using var importing = await unauthenticated.PostAsJsonAsync($"/api/knowledge/collections/{collection.Id}/chats", new { conversationIds = new[] { chat.Id } });
        Assert.Equal(HttpStatusCode.Forbidden, importing.StatusCode);
        foreach (string[] selection in new[] { Array.Empty<string>(), new[] { chat.Id, "../../outside" }, Enumerable.Repeat(chat.Id, 33).ToArray() })
        {
            using var invalid = await _client.PostAsJsonAsync($"/api/knowledge/collections/{collection.Id}/chats", new { conversationIds = selection });
            Assert.Equal(HttpStatusCode.BadRequest, invalid.StatusCode);
            Assert.Empty(_knowledge.ListDocuments(collection.Id));
        }
        using var absent = await _client.PostAsJsonAsync($"/api/knowledge/collections/{Guid.NewGuid():N}/chats", new { conversationIds = new[] { chat.Id } });
        Assert.Equal(HttpStatusCode.NotFound, absent.StatusCode);
    }

    [SkippableFact]
    public async Task AttachmentLinksOutsideTheirStoreAreReportedAndNeverCopied()
    {
        string outside = Path.Combine(_root, "secret.txt");
        await File.WriteAllTextAsync(outside, "private outside-semaphore");
        try { File.CreateSymbolicLink(Path.Combine(_uploads, "link.txt"), outside); }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        { Skip.If(true, "Creating symbolic links is unavailable: " + error.Message); }
        var chat = Chat("Guarded", new StoredMessage { Content = "A selected chat", TextFilePaths = ["link.txt"],
            Artifacts = [new StoredArtifact { Name = "escape.txt", Url = "/api/code/artifacts/run1/%2e%2e/secret.txt" }] });
        var collection = await _knowledge.CreateCollectionAsync("Guarded");
        var imported = await Import(collection.Id, chat.Id);
        Assert.Equal(2, imported.GetProperty("errors").GetArrayLength());
        Assert.Single(imported.GetProperty("documents").EnumerateArray());
        Assert.Single(imported.GetProperty("importedChats").EnumerateArray());
        Assert.Empty(await _knowledge.SearchAsync("outside-semaphore", new KnowledgeSearchOptions { CollectionIds = [collection.Id] }));
    }
}
