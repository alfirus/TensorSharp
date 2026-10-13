using TensorSharp.Knowledge;
using TensorSharp.Server.Knowledge;

namespace InferenceWeb.Tests;

public sealed class KnowledgePdfBridgeTests
{
    [Fact]
    public async Task ExhaustedCharacterBudgetFailsBeforeNativeFileAccess()
    {
        var context = new KnowledgePdfPageContext("must-not-be-opened.pdf", "scan.pdf", 1, 0, 800_000);
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => new PdfKnowledgePageExtractor().ExtractPageAsync(context));
    }

    [ApplePdfOcrFact]
    public async Task ActualNativeOutputLimitRemainsFatalThroughTheKnowledgeAdapter()
    {
        string path = Path.Combine(Path.GetTempPath(), "knowledge-ocr-budget-" + Guid.NewGuid().ToString("N") + ".pdf");
        try
        {
            using (var source = typeof(KnowledgePdfBridgeTests).Assembly.GetManifestResourceStream(
                "InferenceWeb.Tests.Fixtures.Knowledge.scanned-four-pages.pdf")!)
            await using (var output = File.Create(path))
                await source.CopyToAsync(output);
            var context = new KnowledgePdfPageContext(path, "scan.pdf", 1, 10, 800_000);
            await Assert.ThrowsAsync<KnowledgeLimitException>(() => new PdfKnowledgePageExtractor().ExtractPageAsync(context));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}
