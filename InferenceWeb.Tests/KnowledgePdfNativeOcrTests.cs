using TensorSharp.Knowledge;
using TensorSharp.Models.Media;
using TensorSharp.Server.Knowledge;

namespace InferenceWeb.Tests;

public sealed class ApplePdfOcrFactAttribute : FactAttribute
{
    public ApplePdfOcrFactAttribute()
    {
        if (!ApplePdfOcr.IsSupported) Skip = "Requires actual Apple CoreGraphics and Vision OCR (macOS 13+/iOS 16+).";
    }
}

/// <summary>Actual system OCR, not mocked extraction. Other platforms are explicitly skipped.</summary>
public sealed class KnowledgePdfNativeOcrTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-native-pdf-" + Guid.NewGuid().ToString("N"));
    public KnowledgePdfNativeOcrTests() => Directory.CreateDirectory(_root);
    public void Dispose() => Directory.Delete(_root, true);
    private string Fixture(string name)
    {
        string path = Path.Combine(_root, name);
        using var source = typeof(KnowledgePdfNativeOcrTests).Assembly.GetManifestResourceStream("InferenceWeb.Tests.Fixtures.Knowledge." + name)!;
        using var output = File.Create(path);
        source.CopyTo(output);
        return path;
    }

    [ApplePdfOcrFact]
    public void ActualOcrReadsAllFourScannedPageCodesAndTotals()
    {
        string path = Fixture("scanned-four-pages.pdf");
        string[] codes = ["CEDAR-4821", "ORBIT-7356", "HARBOR-9064", "BAMBOO-2187"];
        string[] totals = ["46", "43", "42", "45"];
        for (int page = 1; page <= 4; page++)
        {
            string text = ApplePdfOcr.RecognizePage(path, page, 16_777_216);
            Assert.Contains(codes[page - 1], text, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(totals[page - 1], text);
        }
    }

    [ApplePdfOcrFact]
    public void ActualOcrReadsChineseAndBothRasterBlocks()
    {
        string scanned = ApplePdfOcr.RecognizePage(Fixture("scanned-four-pages.pdf"), 4, 16_777_216);
        Assert.True(scanned.Contains("竹林", StringComparison.Ordinal), scanned);
        Assert.Contains("蓝色", scanned);
        Assert.Contains("二十七", scanned);
        string mixed = ApplePdfOcr.RecognizePage(Fixture("mixed-two-pages.pdf"), 2, 16_777_216);
        Assert.Contains("ORBIT-7356", mixed, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Amber bolts", mixed, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("43", mixed);
    }

    [ApplePdfOcrFact]
    public async Task RuntimeIndexesScannedPagesWithoutAChatModelAndKeepsTheirProvenance()
    {
        string path = Fixture("scanned-four-pages.pdf");
        using var store = KnowledgeRuntime.CreateStore(Path.Combine(_root, "store"));
        var collection = await store.CreateCollectionAsync("Scans");
        await using var input = File.OpenRead(path);
        var document = await store.ImportAsync(collection.Id, "scanned.pdf", input, "application/pdf");
        Assert.Equal(KnowledgeDocumentStatus.Partial, document.Status);
        Assert.Contains("apple-vision", document.ExtractorId);
        for (int page = 1; page <= 4; page++)
        {
            string code = new[] { "CEDAR-4821", "ORBIT-7356", "HARBOR-9064", "BAMBOO-2187" }[page - 1];
            var hits = await store.SearchAsync(code, new KnowledgeSearchOptions { CollectionIds = [collection.Id] });
            Assert.NotEmpty(hits);
            Assert.Equal(page, hits[0].Location?.Page);
            Assert.Contains(code, hits[0].Text, StringComparison.OrdinalIgnoreCase);
        }
        await using var original = store.OpenOriginal(collection.Id, document.Id, document.Version);
        using var bytes = new MemoryStream();
        await original.CopyToAsync(bytes);
        Assert.Equal(await File.ReadAllBytesAsync(path), bytes.ToArray());
    }

    [ApplePdfOcrFact]
    public void RenderingHonorsPixelBudgetAndRejectsCancellationAndMissingPages()
    {
        string path = Fixture("scanned-four-pages.pdf");
        byte[] png = ApplePdfOcr.RenderPagePng(path, 1, 800_000);
        var (width, height) = MediaCodecs.Image.ReadDimensions(png);
        Assert.InRange((long)width * height, 1, 800_000);
        Assert.Throws<ArgumentOutOfRangeException>(() => ApplePdfOcr.RecognizePage(path, 5, 800_000));
        Assert.Throws<ArgumentOutOfRangeException>(() => ApplePdfOcr.RecognizePage(path, 1, 1));
        Assert.Throws<PdfOcrLimitException>(() => ApplePdfOcr.RecognizePage(path, 1, 800_000, maxCharacters: 10));
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() => ApplePdfOcr.RecognizePage(path, 1, 800_000, cancellation.Token));
        Assert.Throws<InvalidDataException>(() => ApplePdfOcr.RecognizePage(Path.Combine(_root, "absent.pdf"), 1, 800_000));
    }
}
