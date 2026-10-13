using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TensorSharp.Knowledge;
using TensorSharp.Server.Endpoints;

namespace InferenceWeb.Tests;

public sealed class KnowledgeEndpointTests
{
    [Fact]
    public async Task RealHttpLibraryLifecycleScopesAndDownloadPolicy()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-knowledge-http-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            var builder = WebApplication.CreateBuilder(new WebApplicationOptions { EnvironmentName = "Production" });
            builder.Logging.ClearProviders();
            builder.WebHost.UseUrls("http://127.0.0.1:0");
            builder.Services.AddSingleton(_ => new KnowledgeStore(root, new BuiltInKnowledgeExtractor()));
            await using var app = builder.Build();
            app.MapKnowledgeEndpoints();
            await app.StartAsync();
            string address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.Single();
            using var client = new HttpClient { BaseAddress = new Uri(address) };
            using var created = await client.PostAsJsonAsync("/api/knowledge/collections", new { name = "Project Orion" });
            created.EnsureSuccessStatusCode();
            var collection = JsonDocument.Parse(await created.Content.ReadAsStringAsync()).RootElement.GetProperty("collection");
            string id = collection.GetProperty("id").GetString()!;
            string documents = $"/api/knowledge/collections/{id}/documents";
            using var upload = new MultipartFormDataContent();
            upload.Add(new ByteArrayContent(Encoding.UTF8.GetBytes("The secret launch code is Cedar-481.")), "file", "launch.html");
            using var imported = await client.PostAsync(documents, upload);
            imported.EnsureSuccessStatusCode();
            var metadata = JsonDocument.Parse(await imported.Content.ReadAsStringAsync()).RootElement.GetProperty("documents")[0];
            string documentId = metadata.GetProperty("id").GetString()!;
            using var search = await client.PostAsJsonAsync("/api/knowledge/search", new { query = "launch code", collectionIds = new[] { id } });
            search.EnsureSuccessStatusCode();
            var hit = JsonDocument.Parse(await search.Content.ReadAsStringAsync()).RootElement.GetProperty("results")[0];
            Assert.Contains("Cedar-481", hit.GetProperty("text").GetString());
            using var source = await client.GetAsync(hit.GetProperty("sourceUrl").GetString());
            source.EnsureSuccessStatusCode();
            Assert.Equal("application/octet-stream", source.Content.Headers.ContentType!.MediaType);
            Assert.Equal("attachment", source.Content.Headers.ContentDisposition!.DispositionType);
            Assert.Equal("nosniff", source.Headers.GetValues("X-Content-Type-Options").Single());
            using var denied = await client.PostAsJsonAsync("/api/knowledge/read", new { reference = hit.GetProperty("reference").GetString(), collectionIds = Array.Empty<string>() });
            Assert.Equal(HttpStatusCode.NotFound, denied.StatusCode);
            using var empty = await client.PostAsJsonAsync("/api/knowledge/search", new { query = "launch", collectionIds = Array.Empty<string>() });
            Assert.Equal(0, JsonDocument.Parse(await empty.Content.ReadAsStringAsync()).RootElement.GetProperty("results").GetArrayLength());
            using var reindex = await client.PostAsJsonAsync(documents + "/" + documentId + "/reindex", new { });
            reindex.EnsureSuccessStatusCode();
            using var stale = await client.GetAsync(hit.GetProperty("sourceUrl").GetString());
            Assert.Equal(HttpStatusCode.Conflict, stale.StatusCode);
            using var deleted = await client.DeleteAsync(documents + "/" + documentId);
            deleted.EnsureSuccessStatusCode();
            var listing = JsonDocument.Parse(await client.GetStringAsync(documents));
            Assert.Empty(listing.RootElement.GetProperty("documents").EnumerateArray());
            using var malformed = await client.PostAsJsonAsync("/api/knowledge/search", new { query = "launch", collectionIds = new[] { id }, limit = "bad" });
            Assert.Equal(HttpStatusCode.BadRequest, malformed.StatusCode);

            using var multiple = new MultipartFormDataContent();
            multiple.Add(new StringContent("first"), "file", "first.txt");
            multiple.Add(new StringContent("second"), "file", "second.txt");
            using var refused = await client.PostAsync(documents, multiple);
            Assert.Equal(HttpStatusCode.BadRequest, refused.StatusCode);
            Assert.Empty(JsonDocument.Parse(await client.GetStringAsync(documents)).RootElement.GetProperty("documents").EnumerateArray());
            await app.StopAsync();
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }
}
