using TensorSharp.Knowledge;
using TensorSharp.Runtime;
using TensorSharp.Server.Knowledge;

namespace InferenceWeb.Tests;

/// <summary>Media import contracts without a production model or accelerator.</summary>
public sealed class KnowledgeMediaExtractorTests
{
    [Theory]
    [InlineData("muse-glimmer", null, " to=self<|message|>Guess the hidden brand.<|eom|><|start|>assistant<|message|>TensorSharp promotional banner.<|eot|>")]
    [InlineData("gemma4", "<|channel>thought\n", "Guess the hidden brand.<channel|>TensorSharp promotional banner.")]
    [InlineData("nemotron_h", "<think></think>", "Guess the hidden brand.</think>TensorSharp promotional banner.")]
    public async Task RawGenerationIndexesOnlyTheAnswerAcrossMediaModelProtocols(string architecture, string? suffix, string raw)
    {
        using var generation = new CancellationTokenSource();
        var updates = new List<ChatStreamUpdate> { ChatStreamUpdate.Text("") with { RawGenerationSuffix = suffix } };
        // Real generation splits protocol markers across pieces.
        updates.AddRange(raw.Select(value => ChatStreamUpdate.Text(value.ToString())));
        string content = await ModelKnowledgeMediaExtractor.CollectAsync(Stream(updates.ToArray()), architecture,
            generation, CancellationToken.None);
        Assert.Equal("TensorSharp promotional banner.", content);
    }

    [Fact]
    public async Task ThinkingWithoutAnAnswerCannotBeImportedAsSourceEvidence()
    {
        using var generation = new CancellationTokenSource();
        await Assert.ThrowsAsync<InvalidDataException>(() => ModelKnowledgeMediaExtractor.CollectAsync(Stream(
            ChatStreamUpdate.Text("") with { RawGenerationSuffix = "<|channel>thought\n" },
            ChatStreamUpdate.Text("The image might contain TensorSharp, but I have not described it.")),
            "gemma4", generation, CancellationToken.None));
    }

    [Fact]
    public async Task AlreadyParsedOutputPreservesLiteralTextAndAppliesRetractions()
    {
        using var generation = new CancellationTokenSource();
        string content = await ModelKnowledgeMediaExtractor.CollectAsync(Stream(
            ChatStreamUpdate.Parsed("Maybe a secret", null, null),
            ChatStreamUpdate.Parsed("Visible text: <channel|> TensorSharp", "Maybe a secret", null, "Maybe a secret")),
            "gemma4", generation, CancellationToken.None);
        Assert.Equal("Visible text: <channel|> TensorSharp", content);
    }

    [Fact]
    public async Task ToolOutputAndInterruptedGenerationCannotBecomeSuccessfulExtraction()
    {
        using var generation = new CancellationTokenSource();
        await Assert.ThrowsAsync<InvalidDataException>(() => ModelKnowledgeMediaExtractor.CollectAsync(Stream(
            ChatStreamUpdate.Parsed("TensorSharp", null, new[] { new ToolCall { Name = "invented_tool" } })),
            "gemma4", generation, CancellationToken.None));
        await Assert.ThrowsAsync<InvalidDataException>(() => ModelKnowledgeMediaExtractor.CollectAsync(Stream(
            ChatStreamUpdate.Text("Incomplete image description"),
            ChatStreamUpdate.Text("") with { Done = true, FinishReason = "cancelled" }),
            "gemma4", generation, CancellationToken.None));
    }

    [Fact]
    public async Task AnswerLimitDoesNotCountDiscardedReasoningAsImageContent()
    {
        using var generation = new CancellationTokenSource();
        string content = await ModelKnowledgeMediaExtractor.CollectAsync(Stream(
            ChatStreamUpdate.Text("") with { RawGenerationSuffix = "<|channel>thought\n" },
            ChatStreamUpdate.Text(new string('x', 20000) + "<channel|>TensorSharp banner")),
            "gemma4", generation, CancellationToken.None);
        Assert.Equal("TensorSharp banner", content);
    }

    [Fact]
    public async Task OversizedRawOutputCancelsAndDrainsGenerationCleanup()
    {
        using var generation = new CancellationTokenSource();
        bool cleanedUp = false;
        async IAsyncEnumerable<ChatStreamUpdate> Produce()
        {
            try
            {
                yield return ChatStreamUpdate.Text(new string('x', 64001));
                await Task.Yield();
                generation.Token.ThrowIfCancellationRequested();
                yield return ChatStreamUpdate.Text("must not be indexed");
            }
            finally { cleanedUp = true; }
        }
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => ModelKnowledgeMediaExtractor.CollectAsync(
            Produce(), "gemma4", generation, CancellationToken.None));
        Assert.True(generation.IsCancellationRequested);
        Assert.True(cleanedUp);
    }

    [Fact]
    public async Task OversizedAnswerAndUserCancellationFailInsteadOfImportingPartialText()
    {
        using var generation = new CancellationTokenSource();
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => ModelKnowledgeMediaExtractor.CollectAsync(Stream(
            ChatStreamUpdate.Text(new string('x', 16001))), "gemma4", generation, CancellationToken.None));
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => ModelKnowledgeMediaExtractor.CollectAsync(Stream(
            ChatStreamUpdate.Text("TensorSharp banner")), "gemma4", cancelled, cancelled.Token));
    }

    [Theory]
    [InlineData("upload", "image/png", ".png")]
    [InlineData("upload.blob", "IMAGE/JPEG; charset=binary", ".jpg")]
    [InlineData("upload", "audio/x-wav", ".wav")]
    [InlineData("upload", "video/mp4", ".mp4")]
    [InlineData("photo.PNG", "image/jpeg", ".PNG")]
    [InlineData("upload.bin", "application/octet-stream", ".bin")]
    public void MimeOnlyMediaReceivesAKnownPrivateCodecSuffix(string name, string type, string expected)
        => Assert.Equal(expected, ModelKnowledgeMediaExtractor.ResolveExtension(new("original", name, type)));

    [Fact]
    public async Task MimeOnlyImageImportReportsItsActualVisionRequirement()
    {
        string path = Path.GetTempFileName();
        try
        {
            using var model = new ModelService();
            var extraction = await new BuiltInKnowledgeExtractor(mediaExtractor: new ModelKnowledgeMediaExtractor(model))
                .ExtractAsync(new(path, "uploaded-image", "image/png"));
            Assert.Equal(KnowledgeDocumentStatus.MetadataOnly, extraction.Status);
            Assert.Empty(extraction.Segments);
            Assert.Contains(extraction.Warnings, value => value.Contains("loaded vision model"));
            Assert.StartsWith("tensorsharp-media-v2:", extraction.ExtractorId);
        }
        finally { File.Delete(path); }
    }

    private static async IAsyncEnumerable<ChatStreamUpdate> Stream(params ChatStreamUpdate[] updates)
    {
        foreach (var update in updates) yield return update;
        await Task.CompletedTask;
    }
}
