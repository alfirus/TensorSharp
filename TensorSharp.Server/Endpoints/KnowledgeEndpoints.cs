using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Routing;
using Microsoft.AspNetCore.WebUtilities;
using Microsoft.Net.Http.Headers;
using TensorSharp.Knowledge;

namespace TensorSharp.Server.Endpoints;

/// <summary>Opt-in personal-library API. Mount behind the application's authentication boundary
/// for remote use. Collection identifiers are retrieval scope, never authorization credentials.</summary>
public static class KnowledgeEndpoints
{
    public static IEndpointRouteBuilder MapKnowledgeEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("/api/knowledge");
        group.MapGet("/collections", (KnowledgeStore store) => Results.Json(new { collections = store.ListCollections() }));
        group.MapPost("/collections", (HttpContext ctx, KnowledgeStore store) => Guard(async () =>
        {
            var body = await Body(ctx);
            return Results.Json(new { collection = await store.CreateCollectionAsync(Required(body, "name"), Optional(body, "description"), ctx.RequestAborted) });
        }));
        group.MapDelete("/collections/{collection}", (string collection, HttpContext ctx, KnowledgeStore store) => Guard(async () =>
            Results.Json(new { deleted = await store.DeleteCollectionAsync(collection, ctx.RequestAborted) })));
        group.MapGet("/collections/{collection}/documents", (string collection, HttpContext ctx, KnowledgeStore store) => Guard(() =>
        {
            int offset = PageNumber(ctx, "offset", 0, 0, 10000), limit = PageNumber(ctx, "limit", 100, 1, 200);
            var all = store.ListDocuments(collection);
            var documents = all.Skip(offset).Take(limit).ToArray();
            return Task.FromResult<IResult>(Results.Json(new { documents, total = all.Count, nextOffset = offset + documents.Length < all.Count ? (int?)(offset + documents.Length) : null }));
        }));
        group.MapPost("/collections/{collection}/documents", (string collection, HttpContext ctx, KnowledgeStore store) => Guard(async () =>
        {
            // Stream a bounded multipart upload directly into the store. Never buffer an entire
            // request in memory or let form binding create an unbounded collection of files.
            if (!MediaTypeHeaderValue.TryParse(ctx.Request.ContentType, out var type) ||
                !string.Equals(type.MediaType.Value, "multipart/form-data", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Expected multipart/form-data with one file per request.");
            string boundary = HeaderUtilities.RemoveQuotes(type.Boundary).Value ?? "";
            if (boundary.Length is < 1 or > 200) throw new ArgumentException("Invalid multipart boundary.");
            long limit = store.Options.MaxFileBytes + 65536;
            if (ctx.Features.Get<IHttpMaxRequestBodySizeFeature>() is { IsReadOnly: false } bodySize) bodySize.MaxRequestBodySize = limit;
            if (ctx.Request.ContentLength > limit) throw new KnowledgeLimitException("Upload exceeds the request size limit.");
            var reader = new MultipartReader(boundary, ctx.Request.Body) { BodyLengthLimit = store.Options.MaxFileBytes, HeadersCountLimit = 16, HeadersLengthLimit = 8192 };
            var section = await reader.ReadNextSectionAsync(ctx.RequestAborted) ?? throw new ArgumentException("Upload contains no file.");
            if (!ContentDispositionHeaderValue.TryParse(section.ContentDisposition, out var disposition) || !disposition.DispositionType.Equals("form-data") ||
                (!disposition.FileName.HasValue && !disposition.FileNameStar.HasValue)) throw new ArgumentException("Upload must contain a file section.");
            string fileName = HeaderUtilities.RemoveQuotes(disposition.FileNameStar.HasValue ? disposition.FileNameStar : disposition.FileName).Value ?? "";
            // Validate the complete envelope before a durable commit. A second file must not
            // silently disappear or leave a partial success behind an error response.
            await using var staged = new FileStream(Path.Combine(Path.GetTempPath(), "ts-knowledge-http-" + Guid.NewGuid().ToString("N")),
                FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 81920, FileOptions.Asynchronous | FileOptions.DeleteOnClose);
            byte[] copyBuffer = new byte[81920];
            int copied;
            while ((copied = await section.Body.ReadAsync(copyBuffer, ctx.RequestAborted)) != 0)
            {
                if (staged.Length + copied > store.Options.MaxFileBytes) throw new KnowledgeLimitException("File exceeds the upload limit.");
                await staged.WriteAsync(copyBuffer.AsMemory(0, copied), ctx.RequestAborted);
            }
            if (await reader.ReadNextSectionAsync(ctx.RequestAborted) != null) throw new ArgumentException("Upload one file per request.");
            staged.Position = 0;
            var document = await store.ImportAsync(collection, fileName, staged, section.ContentType, ctx.RequestAborted);
            return Results.Json(new { documents = new[] { document } });
        })).DisableRequestTimeout();
        group.MapDelete("/collections/{collection}/documents/{document}", (string collection, string document, HttpContext ctx, KnowledgeStore store) => Guard(async () =>
            Results.Json(new { deleted = await store.DeleteDocumentAsync(collection, document, ctx.RequestAborted) })));
        group.MapPost("/collections/{collection}/documents/{document}/reindex", (string collection, string document, HttpContext ctx, KnowledgeStore store) => Guard(async () =>
            Results.Json(new { document = await store.ReindexAsync(collection, document, ctx.RequestAborted) }))).DisableRequestTimeout();
        group.MapGet("/collections/{collection}/documents/{document}/source", (string collection, string document, int version, HttpContext ctx, KnowledgeStore store) => Guard(() =>
        {
            var metadata = store.ListDocuments(collection).FirstOrDefault(d => d.Id == document) ?? throw new KeyNotFoundException("Document was not found.");
            var stream = store.OpenOriginal(collection, document, version);
            ctx.Response.Headers.XContentTypeOptions = "nosniff";
            ctx.Response.Headers.CacheControl = "no-store";
            return Task.FromResult<IResult>(Results.File(stream, "application/octet-stream", metadata.FileName, enableRangeProcessing: true));
        }));
        group.MapPost("/search", (HttpContext ctx, KnowledgeStore store) => Guard(async () =>
        {
            var body = await Body(ctx);
            var results = await store.SearchAsync(Required(body, "query"), new KnowledgeSearchOptions
            { CollectionIds = Scope(body), Limit = Number(body, "limit", 8), MaxCharacters = Number(body, "maxCharacters", 12000) }, ctx.RequestAborted);
            return Results.Json(new { results = results.Select(Result) });
        }));
        group.MapPost("/read", (HttpContext ctx, KnowledgeStore store) => Guard(async () =>
        {
            var body = await Body(ctx);
            var results = store.Read(Required(body, "reference"), Number(body, "maxCharacters", 12000), Number(body, "adjacentChunks", 0), Scope(body));
            return Results.Json(new { results = results.Select(Result) });
        }));
        return endpoints;
    }

    private static object Result(KnowledgeSearchResult result) => new
    {
        result.Reference, result.CollectionId, result.DocumentId, result.FileName, result.Version,
        result.ChunkIndex, result.Text, result.Location, result.Score, result.Status, result.Truncated, result.Modality,
        sourceUrl = $"/api/knowledge/collections/{result.CollectionId}/documents/{result.DocumentId}/source?version={result.Version}"
    };

    private static async Task<JsonElement> Body(HttpContext ctx)
    {
        const int limit = 32768;
        if (ctx.Request.ContentLength > limit) throw new KnowledgeLimitException("Knowledge JSON exceeds 32 KiB.");
        using var buffer = new MemoryStream();
        byte[] bytes = new byte[4096];
        int count;
        while ((count = await ctx.Request.Body.ReadAsync(bytes, ctx.RequestAborted)) != 0)
        {
            if (buffer.Length + count > limit) throw new KnowledgeLimitException("Knowledge JSON exceeds 32 KiB.");
            buffer.Write(bytes, 0, count);
        }
        using var document = JsonDocument.Parse(buffer.ToArray(), new JsonDocumentOptions { MaxDepth = 16 });
        if (document.RootElement.ValueKind != JsonValueKind.Object) throw new ArgumentException("Expected a JSON object.");
        return document.RootElement.Clone();
    }

    private static string Required(JsonElement body, string name) => Optional(body, name) ?? throw new ArgumentException($"{name} is required.");
    private static int PageNumber(HttpContext ctx, string name, int fallback, int min, int max) =>
        !ctx.Request.Query.TryGetValue(name, out var value) ? fallback : int.TryParse(value, out int n) && n >= min && n <= max ? n : throw new ArgumentException($"{name} must be between {min} and {max}.");
    private static string? Optional(JsonElement body, string name) => !body.TryGetProperty(name, out var value) || value.ValueKind == JsonValueKind.Null ? null : value.ValueKind == JsonValueKind.String ? value.GetString() : throw new ArgumentException($"{name} must be a string.");
    private static int Number(JsonElement body, string name, int fallback) => !body.TryGetProperty(name, out var value) ? fallback : value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int number) ? number : throw new ArgumentException($"{name} must be an integer.");
    private static string[] Scope(JsonElement body)
    {
        if (!body.TryGetProperty("collectionIds", out var scope) || scope.ValueKind != JsonValueKind.Array || scope.GetArrayLength() > 100) throw new ArgumentException("collectionIds must be an array of at most 100 selected collection IDs.");
        return scope.EnumerateArray().Select(value => value.ValueKind == JsonValueKind.String ? value.GetString()! : throw new ArgumentException("collectionIds must contain strings.")).ToArray();
    }

    private static async Task<IResult> Guard(Func<Task<IResult>> action)
    {
        try { return await action().ConfigureAwait(false); }
        catch (KnowledgeLimitException error) { return Results.Json(new { error = error.Message }, statusCode: 413); }
        catch (KnowledgeStaleReferenceException error) { return Results.Json(new { error = error.Message }, statusCode: 409); }
        catch (KeyNotFoundException error) { return Results.Json(new { error = error.Message }, statusCode: 404); }
        catch (UnauthorizedAccessException) { return Results.Json(new { error = "The requested source is outside the selected scope." }, statusCode: 403); }
        catch (Exception error) when (error is ArgumentException or InvalidDataException or JsonException or System.Xml.XmlException)
        { return Results.Json(new { error = error.Message }, statusCode: 400); }
    }
}
