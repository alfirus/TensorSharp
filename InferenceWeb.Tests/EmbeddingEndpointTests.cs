using System.Buffers.Binary;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TensorSharp.Models.Embeddings;
using TensorSharp.Server.Endpoints;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;

namespace InferenceWeb.Tests;

public class EmbeddingEndpointTests
{
    internal static ServerHostingOptions Options(bool embeddings = true) => new(
        startupModelPath: "/private/models/test-encoder.gguf", startupMmProjPath: null,
        defaultBackend: "ggml_cpu", supportedBackends: null, defaultMaxTokens: 100,
        maxTokensPinned: false, defaultVideoFrames: 0, defaultVideoFps: 0,
        defaultVideoWidth: 0, defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: null,
        uploadDirectory: Path.GetTempPath(), logDirectory: Path.GetTempPath(),
        fileLoggingEnabled: false, samplingDefaults: null, embeddingsEnabled: embeddings);

    private sealed class FakeModel : IEmbeddingModel
    {
        public string ModelName => "test-encoder";
        public string Architecture => "bert";
        public int Dimensions => 3;
        public int MaxTokens => 8;
        public int VocabularySize => 256;
        public int Calls { get; private set; }
        public bool? LastTruncate { get; private set; }
        public CancellationToken LastCancellation { get; private set; }
        public int[][] LastInputs { get; private set; }
        public int[] Tokenize(string text, bool truncate = false)
        {
            LastTruncate = truncate;
            int[] tokens = new[] { 1 }.Concat(text.Select(c => (int)c)).Append(2).ToArray();
            if (tokens.Length > MaxTokens)
            {
                if (!truncate) throw new ArgumentException("Input exceeds the model context length.");
                tokens = tokens[..MaxTokens];
                tokens[^1] = 2;
            }
            return tokens;
        }
        public Task<EmbeddingBatchResult> EmbedTokensAsync(IReadOnlyList<int[]> inputs, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Calls++;
            LastCancellation = cancellationToken;
            LastInputs = inputs.ToArray();
            return Task.FromResult(new EmbeddingBatchResult(
                inputs.Select(tokens => new float[] { tokens[0], 4, 12 }).ToArray(), inputs.Sum(tokens => tokens.Length)));
        }
        public void Dispose() { }
    }

    private static DefaultHttpContext Context(string path, string json)
    {
        var context = new DefaultHttpContext();
        context.Request.Path = path;
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(json));
        context.Response.Body = new MemoryStream();
        return context;
    }

    private static JsonElement ReadResponse(HttpContext context)
    {
        context.Response.Body.Position = 0;
        using var document = JsonDocument.Parse(context.Response.Body);
        return document.RootElement.Clone();
    }

    [Fact]
    public async Task OpenAI_TokenBatch_PreservesOrderAndCountsActualTokens()
    {
        var model = new FakeModel();
        var context = Context("/v1/embeddings", """{"model":"test-encoder","input":[[3,7,2],[5,2]]}""");
        await new EmbeddingAdapter(model, Options()).OpenAIAsync(context);
        JsonElement body = ReadResponse(context);
        Assert.Equal(200, context.Response.StatusCode);
        Assert.Equal("list", body.GetProperty("object").GetString());
        Assert.Equal(5, body.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
        Assert.Equal(5, body.GetProperty("usage").GetProperty("total_tokens").GetInt32());
        Assert.Equal(1, model.Calls);
        Assert.Equal(new[] { 3, 7, 2 }, model.LastInputs[0]);
        Assert.Equal(new[] { 5, 2 }, model.LastInputs[1]);
        Assert.Null(model.LastTruncate);
        var data = body.GetProperty("data");
        Assert.Equal(0, data[0].GetProperty("index").GetInt32());
        Assert.Equal(1, data[1].GetProperty("index").GetInt32());
        Assert.True(data[1].GetProperty("embedding")[0].GetSingle() > data[0].GetProperty("embedding")[0].GetSingle());
    }

    [Fact]
    public async Task OpenAI_DimensionsThenNormalization_Base64IsLittleEndianFloat32()
    {
        var context = Context("/v1/embeddings", """{"model":"test-encoder.gguf","input":[3,2],"dimensions":2,"encoding_format":"base64"}""");
        await new EmbeddingAdapter(new FakeModel(), Options()).OpenAIAsync(context);
        byte[] bytes = Convert.FromBase64String(ReadResponse(context).GetProperty("data")[0].GetProperty("embedding").GetString()!);
        Assert.Equal(8, bytes.Length);
        Assert.Equal(0.6f, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(0, 4)), 6);
        Assert.Equal(0.8f, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(4, 4)), 6);
    }

    [Theory]
    [InlineData("{", 400)]
    [InlineData("[]", 400)]
    [InlineData("null", 400)]
    [InlineData("""{"model":42,"input":"abc"}""", 400)]
    [InlineData("""{"model":"missing","input":"abc"}""", 404)]
    [InlineData("""{"model":"test-encoder"}""", 400)]
    [InlineData("""{"model":"test-encoder","input":[]}""", 400)]
    [InlineData("""{"model":"test-encoder","input":""}""", 400)]
    [InlineData("""{"model":"test-encoder","input":["ok",2]}""", 400)]
    [InlineData("""{"model":"test-encoder","input":[[2],[]]}""", 400)]
    [InlineData("""{"model":"test-encoder","input":[-1]}""", 400)]
    [InlineData("""{"model":"test-encoder","input":[256]}""", 400)]
    [InlineData("""{"model":"test-encoder","input":[1.5]}""", 400)]
    [InlineData("""{"model":"test-encoder","input":[true]}""", 400)]
    [InlineData("""{"model":"test-encoder","input":"ok","dimensions":"2"}""", 400)]
    [InlineData("""{"model":"test-encoder","input":"ok","dimensions":0}""", 400)]
    [InlineData("""{"model":"test-encoder","input":"ok","dimensions":4}""", 400)]
    [InlineData("""{"model":"test-encoder","input":"ok","encoding_format":"int8"}""", 400)]
    public async Task InvalidRequests_ReturnProtocolErrorsBeforeInference(string json, int status)
    {
        var model = new FakeModel();
        var context = Context("/v1/embeddings", json);
        await new EmbeddingAdapter(model, Options()).OpenAIAsync(context);
        Assert.Equal(status, context.Response.StatusCode);
        Assert.Equal("invalid_request_error", ReadResponse(context).GetProperty("error").GetProperty("type").GetString());
        Assert.Equal(0, model.Calls);
    }

    [Theory]
    [InlineData("""[{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}}]""")]
    [InlineData("""[{"content":[{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,aGVsbG8="}}]}]""")]
    [InlineData("""[{"content":[{"type":"video_url","video_url":{"url":"https://example.com/video.mp4"}}]}]""")]
    [InlineData("""[{"content":[]}]""")]
    public async Task MediaInputRequiresLoadedProjectorAndReturnsProtocolErrorsWithoutTextInference(string input)
    {
        foreach (bool openAI in new[] { true, false })
        {
            var model = new FakeModel();
            using var adapter = new EmbeddingAdapter(model, Options());
            var context = Context(openAI ? "/v1/embeddings" : "/api/embed", "{\"model\":\"test-encoder\",\"input\":" + input + "}");
            if (openAI) await adapter.OpenAIAsync(context);
            else await adapter.OllamaAsync(context);
            Assert.Equal(400, context.Response.StatusCode);
            var error = ReadResponse(context).GetProperty("error");
            if (openAI) Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
            else Assert.Equal(JsonValueKind.String, error.ValueKind);
            Assert.Equal(0, model.Calls);
        }
    }

    [Theory]
    [InlineData("/v1/embeddings", false, false, 400)]
    [InlineData("/api/embed", true, false, 200)]
    [InlineData("/api/embed", true, true, 400)]
    public async Task ContextOverflow_OpenAIRejects_OllamaTruncationIsExplicit(string path, bool ollama, bool disableTruncation, int status)
    {
        var model = new FakeModel();
        var context = Context(path, JsonSerializer.Serialize(new
            { model = "test-encoder", input = "too many tokens", truncate = !disableTruncation }));
        var adapter = new EmbeddingAdapter(model, Options());
        if (ollama) await adapter.OllamaAsync(context);
        else await adapter.OpenAIAsync(context);
        Assert.Equal(status, context.Response.StatusCode);
        if (status == 200)
        {
            Assert.True(model.LastTruncate);
            Assert.Equal(2, model.LastInputs[0][^1]);
            Assert.Equal(8, ReadResponse(context).GetProperty("prompt_eval_count").GetInt32());
        }
    }

    [Fact]
    public async Task Ollama_HasItsOwnResponseShape()
    {
        var model = new FakeModel();
        var adapter = new EmbeddingAdapter(model, Options());
        var modern = Context("/api/embed", """{"model":"test-encoder","input":["one","two"]}""");
        await adapter.OllamaAsync(modern);
        JsonElement body = ReadResponse(modern);
        Assert.Equal(2, body.GetProperty("embeddings").GetArrayLength());
        Assert.Equal(10, body.GetProperty("prompt_eval_count").GetInt32());
        Assert.True(body.GetProperty("total_duration").GetInt64() >= 0);
        Assert.Equal(0, body.GetProperty("load_duration").GetInt64());
    }

    [Theory]
    [InlineData("\"\"")]
    [InlineData("[]")]
    public async Task Ollama_EmptyInput_DoesNotRunEncoder(string input)
    {
        var model = new FakeModel();
        var context = Context("/api/embed", "{\"model\":\"test-encoder\",\"input\":" + input + "}");
        await new EmbeddingAdapter(model, Options()).OllamaAsync(context);
        Assert.Equal(200, context.Response.StatusCode);
        Assert.Equal(0, ReadResponse(context).GetProperty("embeddings").GetArrayLength());
        Assert.Equal(0, model.Calls);
    }

    [Fact]
    public async Task RequestCancellation_IsPropagatedToModel()
    {
        var model = new FakeModel();
        using var cancellation = new CancellationTokenSource();
        var context = Context("/v1/embeddings", """{"model":"test-encoder","input":"abc"}""");
        context.RequestAborted = cancellation.Token;
        await new EmbeddingAdapter(model, Options()).OpenAIAsync(context);
        Assert.Equal(cancellation.Token, model.LastCancellation);
        cancellation.Cancel();
        var cancelled = Context("/v1/embeddings", """{"model":"test-encoder","input":"abc"}""");
        cancelled.RequestAborted = cancellation.Token;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => new EmbeddingAdapter(model, Options()).OpenAIAsync(cancelled));
        Assert.Equal(1, model.Calls);
    }

    [Fact]
    public async Task OversizedBatch_IsRejectedBeforeTokenizationOrInference()
    {
        var model = new FakeModel();
        var context = Context("/v1/embeddings", JsonSerializer.Serialize(new
        {
            model = "test-encoder",
            input = Enumerable.Repeat("one", EmbeddingAdapter.MaxBatchInputs + 1).ToArray(),
        }));
        await new EmbeddingAdapter(model, Options()).OpenAIAsync(context);
        Assert.Equal(400, context.Response.StatusCode);
        Assert.Equal(0, model.Calls);
        Assert.Null(model.LastTruncate);
    }

    [Fact]
    public async Task MappedHttpRoutes_ServeEmbeddingsAndDiscovery_RejectChat()
    {
        var builder = WebApplication.CreateBuilder(new WebApplicationOptions { EnvironmentName = "Production" });
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddSingleton(Options());
        builder.Services.AddSingleton<IEmbeddingModel>(new FakeModel());
        builder.Services.AddSingleton<EmbeddingAdapter>();
        builder.Services.AddSingleton<OpenAIChatAdapter>();
        builder.Services.AddSingleton<OpenAIResponsesAdapter>();
        builder.Services.AddSingleton<WebUiAdapter>();
        builder.Services.AddSingleton<OllamaAdapter>();
        await using var app = builder.Build();
        app.UseEmbeddingModelGuard();
        app.MapOpenAIEndpoints();
        app.MapOllamaEndpoints();
        app.MapWebUiEndpoints();
        await app.StartAsync();
        string address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.Single();
        using var client = new HttpClient { BaseAddress = new Uri(address) };
        foreach (string path in new[] { "/v1/embeddings", "/api/embed" })
        {
            using var response = await client.PostAsJsonAsync(path, new { model = "test-encoder", input = new[] { "one", "two" } });
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        }
        // Ollama's deprecated /api/embeddings is not served.
        using var removed = await client.PostAsJsonAsync("/api/embeddings", new { model = "test-encoder", prompt = "one" });
        Assert.Equal(HttpStatusCode.NotFound, removed.StatusCode);
        foreach (string path in new[] { "/v1/models", "/api/tags" })
        {
            string json = await client.GetStringAsync(path);
            Assert.Contains("embedding", json);
            Assert.DoesNotContain("/private", json);
        }
        using var show = await client.PostAsJsonAsync("/api/show", new { model = "test-encoder" });
        string showBody = await show.Content.ReadAsStringAsync();
        Assert.Contains("bert.embedding_length", showBody);
        Assert.Contains("embedding", showBody);
        Assert.DoesNotContain("/private", showBody);
        JsonElement webModels = await client.GetFromJsonAsync<JsonElement>("/api/models");
        Assert.Equal("test-encoder.gguf", webModels.GetProperty("loaded").GetString());
        Assert.Equal(3, webModels.GetProperty("embeddingDimensions").GetInt32());
        Assert.DoesNotContain("/private", webModels.GetRawText());
        using var chat = await client.PostAsJsonAsync("/v1/chat/completions", new { model = "test-encoder" });
        Assert.Equal(HttpStatusCode.BadRequest, chat.StatusCode);
        Assert.Contains("embedding model", await chat.Content.ReadAsStringAsync());
        await app.StopAsync();
    }

    [Fact]
    public async Task StandaloneEmbeddingRoutes_RequireNoChatServices()
    {
        var builder = WebApplication.CreateBuilder(new WebApplicationOptions { EnvironmentName = "Production" });
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddTensorSharpEmbeddings(Options());
        // Replace only the expensive encoder, retaining the public hosting registration.
        builder.Services.AddSingleton<IEmbeddingModel>(new FakeModel());
        await using var app = builder.Build();
        app.MapEmbeddingEndpoints();
        await app.StartAsync();
        string address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.Single();
        using var client = new HttpClient { BaseAddress = new Uri(address) };
        foreach (string path in new[] { "/v1/embeddings", "/api/embed" })
        {
            using var response = await client.PostAsJsonAsync(path, new { model = "test-encoder", input = "one" });
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        }
        Assert.Contains("test-encoder", await client.GetStringAsync("/v1/models"));
        Assert.Contains("test-encoder", await client.GetStringAsync("/api/tags"));
        using var show = await client.PostAsJsonAsync("/api/show", new { model = "test-encoder" });
        Assert.Equal(HttpStatusCode.OK, show.StatusCode);
        Assert.Null(app.Services.GetService<ModelService>());
        Assert.Null(app.Services.GetService<OpenAIChatAdapter>());
        await app.StopAsync();
    }

    [Theory]
    [InlineData("/v1/embeddings", false)]
    [InlineData("/v1/embeddings", true)]
    [InlineData("/api/embed", true)]
    [InlineData("/api/show", true)]
    public async Task OversizedJsonBodies_Return413ForContentLengthAndChunkedRequests(string path, bool chunked)
    {
        var builder = WebApplication.CreateBuilder(new WebApplicationOptions { EnvironmentName = "Production" });
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddTensorSharpEmbeddings(Options());
        var model = new FakeModel();
        builder.Services.AddSingleton<IEmbeddingModel>(model);
        await using var app = builder.Build();
        app.MapEmbeddingEndpoints();
        await app.StartAsync();
        string address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.Single();
        using var client = new HttpClient { BaseAddress = new Uri(address) };
        using var request = new HttpRequestMessage(HttpMethod.Post, path)
        {
            Content = new OversizedJsonContent(chunked),
        };
        request.Headers.TransferEncodingChunked = chunked;
        request.Headers.ExpectContinue = true;
        using var response = await client.SendAsync(request);
        Assert.Equal(HttpStatusCode.RequestEntityTooLarge, response.StatusCode);
        string json = await response.Content.ReadAsStringAsync();
        Assert.True(json.Length < 512, "The rejection must not echo the oversized request.");
        using var document = JsonDocument.Parse(json);
        Assert.Contains("16 MiB", json);
        if (path.StartsWith("/v1"))
            Assert.Equal("invalid_request_error", document.RootElement.GetProperty("error").GetProperty("type").GetString());
        else
            Assert.Equal(JsonValueKind.String, document.RootElement.GetProperty("error").ValueKind);
        Assert.Equal(0, model.Calls);
        Assert.Null(model.LastTruncate);
        await app.StopAsync();
    }

    private sealed class OversizedJsonContent(bool chunked) : HttpContent
    {
        private static readonly byte[] Prefix = Encoding.UTF8.GetBytes("{\"model\":\"test-encoder\",\"input\":\"");
        private static readonly byte[] Suffix = Encoding.UTF8.GetBytes("\"}");
        private static readonly byte[] Chunk = Enumerable.Repeat((byte)'a', 8192).ToArray();
        protected override bool TryComputeLength(out long length)
        {
            length = EmbeddingAdapter.MaxRequestBodyBytes + 1L;
            return !chunked;
        }
        protected override async Task SerializeToStreamAsync(Stream stream, TransportContext? context)
        {
            await stream.WriteAsync(Prefix);
            int remaining = EmbeddingAdapter.MaxRequestBodyBytes + 1 - Prefix.Length - Suffix.Length;
            while (remaining > 0)
            {
                int count = Math.Min(remaining, Chunk.Length);
                await stream.WriteAsync(Chunk.AsMemory(0, count));
                remaining -= count;
            }
            await stream.WriteAsync(Suffix);
        }
    }

    [Theory]
    [InlineData("/api/show")]
    [InlineData("/v1/embeddings")]
    public async Task InvalidModelPathCharacters_Return400(string path)
    {
        var context = Context(path, """{"model":"bad/\u0000","input":"abc"}""");
        var adapter = new EmbeddingAdapter(new FakeModel(), Options());
        if (path == "/api/show") await adapter.ShowAsync(context);
        else await adapter.OpenAIAsync(context);
        Assert.Equal(400, context.Response.StatusCode);
        Assert.Contains("valid model identifier", ReadResponse(context).GetRawText());
    }

    [Fact]
    public void DimensionReduction_NormalizesSubnormalValuesWithoutOverflow()
    {
        float[] vector = EmbeddingAdapter.ResizeAndNormalize([float.Epsilon, 1f], 1);
        Assert.Equal(new[] { 1f }, vector);
        Assert.All(vector, value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void Normalization_HandlesLargeFiniteValuesWithoutLosingUnitLength()
    {
        float[] vector = EmbeddingAdapter.ResizeAndNormalize([float.MaxValue, -float.MaxValue], 2);
        Assert.All(vector, value => Assert.True(float.IsFinite(value)));
        Assert.Equal(1d / Math.Sqrt(2d), vector[0], 6);
        Assert.Equal(-1d / Math.Sqrt(2d), vector[1], 6);
        Assert.InRange(vector.Sum(value => (double)value * value), 0.999999, 1.000001);
    }
}
