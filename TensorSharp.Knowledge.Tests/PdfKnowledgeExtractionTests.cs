using TensorSharp.Knowledge;

namespace TensorSharp.Knowledge.Tests;

/// <summary>Coverage and failure semantics use a deterministic provider here. Real native OCR
/// recognition is verified separately against these image-only fixtures in InferenceWeb.Tests.</summary>
public sealed class PdfKnowledgeExtractionTests
{
    private static KnowledgeExtractionContext Fixture(string name) => new(
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "Knowledge", name), name, "application/pdf");

    [Fact]
    public async Task EveryScannedPageReceivesCompletePageContextAndKeepsItsRealNumber()
    {
        var provider = new RecordingProvider((context, _) => Task.FromResult(new KnowledgePdfPageExtraction
        {
            Text = $"Recognized page {context.PageNumber}. Page 999 is literal text, not provenance.", ExtractorId = "test-ocr-v1"
        }));
        var result = await new BuiltInKnowledgeExtractor(pdfPageExtractor: provider).ExtractAsync(Fixture("scanned-four-pages.pdf"));
        Assert.Equal(new[] { 1, 2, 3, 4 }, provider.Calls.Select(c => c.PageNumber));
        Assert.All(provider.Calls, call => Assert.Equal("scanned-four-pages.pdf", call.FileName));
        Assert.All(provider.Calls, call => Assert.Equal(16_777_216, call.MaxPixels));
        Assert.Equal(new[] { 1, 2, 3, 4 }, result.Segments.Select(s => s.Location!.Page!.Value));
        Assert.All(result.Segments, segment => Assert.Equal("ocr", segment.Modality));
        Assert.Equal(KnowledgeDocumentStatus.Partial, result.Status);
        Assert.Contains("test-ocr-v1", result.ExtractorId);
        Assert.Contains(result.Warnings, warning => warning.Contains("pages 1-4"));
        Assert.DoesNotContain(result.Warnings, warning => warning.Contains("did not produce"));
    }

    [Fact]
    public async Task MixedPdfPreservesSelectableTextAndRecognizesWholeRasterPage()
    {
        var provider = new RecordingProvider((_, _) => Task.FromResult(new KnowledgePdfPageExtraction
        { Text = "Scanned body code CEDAR-274. Lower image table total 5831.", ExtractorId = "test-ocr" }));
        var result = await new BuiltInKnowledgeExtractor(pdfPageExtractor: provider).ExtractAsync(Fixture("mixed-two-pages.pdf"));
        Assert.Equal(2, Assert.Single(provider.Calls).PageNumber);
        Assert.Contains(result.Segments, segment => segment.Location!.Page == 1 && segment.Modality == "text");
        Assert.Contains(result.Segments, segment => segment.Location!.Page == 2 && segment.Modality == "text"
            && segment.Text.Count(c => !char.IsWhiteSpace(c)) > 16);
        Assert.Contains(result.Segments, segment => segment.Location!.Page == 2 && segment.Modality == "ocr"
            && segment.Text.Contains("Lower image table"));
    }

    [Fact]
    public async Task MissingOcrCannotClaimAnImageOnlyPdfIsIndexed()
    {
        var result = await new BuiltInKnowledgeExtractor().ExtractAsync(Fixture("scanned-four-pages.pdf"));
        Assert.Equal(KnowledgeDocumentStatus.MetadataOnly, result.Status);
        Assert.Empty(result.Segments);
        Assert.Contains(result.Warnings, warning => warning.Contains("No PDF OCR provider") && warning.Contains("1-4"));
    }

    [Fact]
    public async Task MissingPagesAreReportedWhileSuccessfulPagesRemainSearchable()
    {
        var provider = new RecordingProvider((context, _) => Task.FromResult(new KnowledgePdfPageExtraction
        { Text = context.PageNumber % 2 == 0 ? "Recovered text" : null, ExtractorId = "test-ocr" }));
        var result = await new BuiltInKnowledgeExtractor(pdfPageExtractor: provider).ExtractAsync(Fixture("scanned-four-pages.pdf"));
        Assert.Equal(new[] { 2, 4 }, result.Segments.Select(s => s.Location!.Page!.Value));
        Assert.Equal(KnowledgeDocumentStatus.Partial, result.Status);
        Assert.Contains(result.Warnings, warning => warning.Contains("did not produce text for pages 1, 3"));
    }

    [Fact]
    public async Task PageBudgetExplicitlyReportsSkippedPages()
    {
        var provider = new RecordingProvider((_, _) => Task.FromResult(new KnowledgePdfPageExtraction { Text = "Visible text" }));
        var result = await new BuiltInKnowledgeExtractor(new KnowledgeOptions { MaxPdfOcrPages = 2 }, pdfPageExtractor: provider)
            .ExtractAsync(Fixture("scanned-four-pages.pdf"));
        Assert.Equal(2, provider.Calls.Count);
        Assert.Equal(2, result.Segments.Count);
        Assert.Contains(result.Warnings, warning => warning.Contains("page limit (2)") && warning.Contains("3-4"));
    }

    [Fact]
    public async Task OversizedEmbeddedImagesNeverReachTheNativeRenderer()
    {
        var provider = new RecordingProvider((_, _) => throw new InvalidOperationException("Must not render"));
        var result = await new BuiltInKnowledgeExtractor(new KnowledgeOptions { MaxPdfEmbeddedImagePixels = 1000 }, pdfPageExtractor: provider)
            .ExtractAsync(Fixture("scanned-four-pages.pdf"));
        Assert.Empty(provider.Calls);
        Assert.Equal(KnowledgeDocumentStatus.MetadataOnly, result.Status);
        Assert.Contains(result.Warnings, warning => warning.Contains("embedded images exceed"));
    }

    [Fact]
    public async Task MultipleLegalImagesCannotExceedTheAggregatePagePixelBudget()
    {
        const long budget = 1_200_000;
        var input = Fixture("mixed-two-pages.pdf");
        using (var pdf = UglyToad.PdfPig.PdfDocument.Open(input.FilePath))
        {
            long[] pixels = pdf.GetPage(2).GetImages().Select(image => (long)image.WidthInSamples * image.HeightInSamples).ToArray();
            Assert.Equal(2, pixels.Length);
            Assert.All(pixels, count => Assert.InRange(count, 1, budget));
            Assert.True(pixels.Sum() > budget);
        }
        var provider = new RecordingProvider((_, _) => throw new InvalidOperationException("Must not render"));
        var result = await new BuiltInKnowledgeExtractor(new KnowledgeOptions { MaxPdfEmbeddedImagePixels = budget }, pdfPageExtractor: provider)
            .ExtractAsync(input);
        Assert.Empty(provider.Calls);
        Assert.Equal(KnowledgeDocumentStatus.Partial, result.Status);
        Assert.Equal(new[] { 1, 2 }, result.Segments.Select(segment => segment.Location!.Page!.Value));
        Assert.All(result.Segments, segment => Assert.Equal("text", segment.Modality));
        Assert.Contains(result.Warnings, warning => warning.Contains("Page 2:") && warning.Contains("aggregate pixel"));
    }

    [Fact]
    public async Task RecoverableOcrErrorPreservesAllSelectableText()
    {
        var provider = new RecordingProvider((_, _) => throw new InvalidDataException("Page raster could not be decoded."));
        var result = await new BuiltInKnowledgeExtractor(pdfPageExtractor: provider).ExtractAsync(Fixture("mixed-two-pages.pdf"));
        Assert.Equal(KnowledgeDocumentStatus.Partial, result.Status);
        Assert.Equal(new[] { 1, 2 }, result.Segments.Select(segment => segment.Location!.Page!.Value));
        Assert.All(result.Segments, segment => Assert.Equal("text", segment.Modality));
        Assert.Contains(result.Warnings, warning => warning.Contains("Page 2: OCR could not read this page"));
    }

    [Fact]
    public async Task RecoverableOcrErrorDoesNotPreventLaterScannedPagesBeingIndexed()
    {
        var provider = new RecordingProvider((context, _) => context.PageNumber == 2
            ? throw new InvalidDataException("Vision could not process this page.")
            : Task.FromResult(new KnowledgePdfPageExtraction { Text = "Readable scan" }));
        var result = await new BuiltInKnowledgeExtractor(pdfPageExtractor: provider).ExtractAsync(Fixture("scanned-four-pages.pdf"));
        Assert.Equal(new[] { 1, 3, 4 }, result.Segments.Select(segment => segment.Location!.Page!.Value));
        Assert.Contains(result.Warnings, warning => warning.Contains("did not produce text for pages 2"));
    }

    [Fact]
    public async Task OcrProviderResourceLimitIsFatalRatherThanSilentlyPartial()
    {
        var provider = new RecordingProvider((_, _) => throw new KnowledgeLimitException("OCR text exceeds character budget."));
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => new BuiltInKnowledgeExtractor(pdfPageExtractor: provider)
            .ExtractAsync(Fixture("scanned-four-pages.pdf")));
        Assert.Single(provider.Calls);
    }

    [Fact]
    public async Task CharacterLimitAppliesAcrossAllOcrPages()
    {
        var provider = new RecordingProvider((_, _) => Task.FromResult(new KnowledgePdfPageExtraction { Text = new string('a', 60) }));
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => new BuiltInKnowledgeExtractor(
            new KnowledgeOptions { MaxExtractedCharacters = 100 }, pdfPageExtractor: provider).ExtractAsync(Fixture("scanned-four-pages.pdf")));
        Assert.Equal(new[] { 100, 40 }, provider.Calls.Select(c => c.MaxCharacters));
    }

    [Fact]
    public async Task CancellationStopsBeforeLaterPagesAndPropagatesToCaller()
    {
        using var source = new CancellationTokenSource();
        var provider = new RecordingProvider(async (_, token) =>
        {
            source.Cancel();
            await Task.Delay(Timeout.Infinite, token);
            return new KnowledgePdfPageExtraction();
        });
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => new BuiltInKnowledgeExtractor(pdfPageExtractor: provider)
            .ExtractAsync(Fixture("scanned-four-pages.pdf"), source.Token));
        Assert.Single(provider.Calls);
    }

    [Fact]
    public async Task PageDeadlineReportsPartialCoverageAndContinuesOtherPages()
    {
        var provider = new RecordingProvider(async (context, token) =>
        {
            if (context.PageNumber == 1) await Task.Delay(Timeout.Infinite, token);
            return new KnowledgePdfPageExtraction { Text = "Recovered later page" };
        });
        var result = await new BuiltInKnowledgeExtractor(new KnowledgeOptions { PdfOcrPageTimeoutSeconds = 1 }, pdfPageExtractor: provider)
            .ExtractAsync(Fixture("scanned-four-pages.pdf"));
        Assert.Equal(new[] { 2, 3, 4 }, result.Segments.Select(s => s.Location!.Page!.Value));
        Assert.Contains(result.Warnings, warning => warning.Contains("Page 1: OCR exceeded its time limit"));
    }

    [Fact]
    public async Task ProviderUnavailabilityProducesActionableCoverageWarnings()
    {
        var provider = new RecordingProvider((_, _) => throw new NotSupportedException("Install a local OCR provider."));
        var result = await new BuiltInKnowledgeExtractor(pdfPageExtractor: provider).ExtractAsync(Fixture("scanned-four-pages.pdf"));
        Assert.Equal(KnowledgeDocumentStatus.MetadataOnly, result.Status);
        Assert.Contains(result.Warnings, warning => warning.Contains("Install a local OCR provider"));
    }

    private sealed class RecordingProvider(Func<KnowledgePdfPageContext, CancellationToken, Task<KnowledgePdfPageExtraction>> extract)
        : IKnowledgePdfPageExtractor
    {
        public List<KnowledgePdfPageContext> Calls { get; } = [];
        public Task<KnowledgePdfPageExtraction> ExtractPageAsync(KnowledgePdfPageContext context, CancellationToken cancellationToken = default)
        {
            Calls.Add(context);
            return extract(context, cancellationToken);
        }
    }
}
