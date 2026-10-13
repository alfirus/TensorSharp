using System.IO.Compression;
using System.Text;
using TensorSharp.Knowledge;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.Fonts.Standard14Fonts;
using UglyToad.PdfPig.Writer;

namespace TensorSharp.Knowledge.Tests;

public sealed class BuiltInKnowledgeExtractorTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "knowledge-extractor-tests-" + Guid.NewGuid().ToString("N"));
    public BuiltInKnowledgeExtractorTests() => Directory.CreateDirectory(_root);
    public void Dispose() => Directory.Delete(_root, true);
    private async Task<KnowledgeExtraction> Extract(string name, string text, KnowledgeOptions? options = null)
    {
        string path = Path.Combine(_root, "input");
        await File.WriteAllTextAsync(path, text, new UTF8Encoding(false));
        return await new BuiltInKnowledgeExtractor(options).ExtractAsync(new(path, name, null));
    }

    [Fact]
    public async Task MarkdownPreservesSectionsAndSourceLines()
    {
        var result = await Extract("guide.md", "# Launch\nRelease code is Cedar-481.\n## Owner\nMira owns deployment.");
        Assert.Equal(2, result.Segments.Count);
        Assert.Equal("Launch", result.Segments[0].Location!.Section);
        Assert.Equal(1, result.Segments[0].Location!.StartLine);
        Assert.Equal(2, result.Segments[0].Location!.EndLine);
        Assert.Equal("Owner", result.Segments[1].Location!.Section);
        Assert.Equal(3, result.Segments[1].Location!.StartLine);
        Assert.Equal("tensorsharp-managed-documents-v1", result.ExtractorId);
    }

    [Fact]
    public async Task CsvKeepsHeadersQuotedDelimitersNewlinesAndRecordNumbers()
    {
        var result = await Extract("sales.csv", "Product,Revenue,Note\n\"Blue, boat\",129,\"first\nsecond\"\nTrain,840,\"quoted \"\"text\"\"\"\n");
        Assert.Equal(2, result.Segments.Count);
        Assert.Equal("Product: Blue, boat | Revenue: 129 | Note: first\nsecond", result.Segments[0].Text);
        Assert.Equal(2, result.Segments[0].Location!.Row);
        Assert.Contains("quoted \"text\"", result.Segments[1].Text);
        Assert.Equal(3, result.Segments[1].Location!.Row);
    }

    [Theory]
    [InlineData("a,b\n\"broken,b")]
    [InlineData("a,b\n\"closed\"junk,b")]
    public async Task MalformedTablesFailWithoutPretendingToHaveIndexed(string text)
        => await Assert.ThrowsAsync<InvalidDataException>(() => Extract("bad.csv", text));

    [Fact]
    public async Task JsonLinesKeepCompleteRowsAndRejectMalformedRecords()
    {
        var result = await Extract("events.jsonl", "{\"status\":\"approved\"}\n\n{\"status\":\"released\"}");
        Assert.Equal(3, result.Segments[1].Location!.Row);
        Assert.Contains("released", result.Segments[1].Text);
        await Assert.ThrowsAnyAsync<System.Text.Json.JsonException>(() => Extract("bad.jsonl", "{\"ok\":1}\n{broken}"));
    }

    [Theory]
    [InlineData("clip.srt", "1\n00:01:02,125 --> 00:01:08,750\nUse the cobalt valve.\n")]
    [InlineData("clip.vtt", "WEBVTT\n\n01:02.125 --> 01:08.750\nUse the cobalt valve.\n")]
    public async Task SubtitlesCarryExactTimeReferences(string name, string text)
    {
        var result = await Extract(name, text);
        Assert.Equal(62125, result.Segments[0].Location!.StartMilliseconds);
        Assert.Equal(68750, result.Segments[0].Location!.EndMilliseconds);
        Assert.Equal("audio", result.Segments[0].Modality);
    }

    [Fact]
    public async Task HtmlDoesNotIndexScriptsOrFetchExternalResources()
    {
        var result = await Extract("page.html", "<script>secret bogus</script><h1>Price</h1><p>EUR &amp; USD</p><img src='https://invalid.local/private'>");
        string text = string.Join("", result.Segments.Select(s => s.Text));
        Assert.DoesNotContain("bogus", text);
        Assert.DoesNotContain("invalid.local", text);
        Assert.Contains("EUR & USD", text);
    }

    [Fact]
    public async Task UnknownAndMediaWithoutExtractorAreMetadataOnly()
    {
        foreach (string name in new[] { "image.png", "recording.wav", "video.mp4", "legacy.doc", "data.bin" })
        {
            var result = await Extract(name, "Not a text document despite these bytes.");
            Assert.Equal(KnowledgeDocumentStatus.MetadataOnly, result.Status);
            Assert.Empty(result.Segments);
            Assert.NotEmpty(result.Warnings);
        }
    }

    [Fact]
    public async Task CharacterLimitsAndBinaryTextFailClosed()
    {
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => Extract("large.txt", new string('x', 1025), new KnowledgeOptions { MaxExtractedCharacters = 1024 }));
        await Assert.ThrowsAsync<InvalidDataException>(() => Extract("binary.txt", "abc\0def"));
    }

    [Fact]
    public async Task PdfRetainsRealPageProvenance()
    {
        var builder = new PdfDocumentBuilder();
        var font = builder.AddStandard14Font(Standard14Font.Helvetica);
        builder.AddPage(PageSize.A4).AddText("Launch code Cedar-481", 12, new PdfPoint(40, 700), font);
        builder.AddPage(PageSize.A4).AddText("Mira is the owner", 12, new PdfPoint(40, 700), font);
        string path = Path.Combine(_root, "input");
        await File.WriteAllBytesAsync(path, builder.Build());
        var result = await new BuiltInKnowledgeExtractor().ExtractAsync(new(path, "guide.pdf", null));
        Assert.Equal(2, result.Segments.Count);
        Assert.Equal(2, result.Segments[1].Location!.Page);
        Assert.Contains("Mira", result.Segments[1].Text);
        Assert.Equal(KnowledgeDocumentStatus.Partial, result.Status);
    }

    [Fact]
    public async Task OfficeReadersPreserveWorkbookCellCoordinatesAndPresentationOrder()
    {
        string path = Zip(new Dictionary<string, string>
        {
            ["xl/workbook.xml"] = "<workbook xmlns:r='urn:relationships'><sheets><sheet name='Budget' r:id='r1'/></sheets></workbook>",
            ["xl/_rels/workbook.xml.rels"] = "<Relationships><Relationship Id='r1' Target='worksheets/sheet1.xml'/></Relationships>",
            ["xl/worksheets/sheet1.xml"] = "<worksheet><sheetData><row><c r='B7' t='inlineStr'><is><t>Cobalt</t></is></c><c r='C7'><v>815</v></c></row></sheetData></worksheet>"
        });
        var result = await new BuiltInKnowledgeExtractor().ExtractAsync(new(path, "budget.xlsx", null));
        Assert.Equal("Budget", result.Segments[0].Location!.Sheet);
        Assert.Equal(7, result.Segments[0].Location!.Row);
        Assert.Contains("C7=815", result.Segments[0].Text);
        path = Zip(new Dictionary<string, string>
        {
            ["ppt/presentation.xml"] = "<presentation xmlns:r='urn:relationships'><sldIdLst><sldId r:id='r2'/><sldId r:id='r1'/></sldIdLst></presentation>",
            ["ppt/_rels/presentation.xml.rels"] = "<Relationships><Relationship Id='r1' Target='slides/slide1.xml'/><Relationship Id='r2' Target='slides/slide2.xml'/></Relationships>",
            ["ppt/slides/slide1.xml"] = "<slide><p><t>Second visible slide</t></p></slide>",
            ["ppt/slides/slide2.xml"] = "<slide><p><t>First visible slide</t></p></slide>"
        });
        result = await new BuiltInKnowledgeExtractor().ExtractAsync(new(path, "plan.pptx", null));
        Assert.Contains("First visible", result.Segments[0].Text);
        Assert.Equal(1, result.Segments[0].Location!.Page);
        Assert.Contains("Second visible", result.Segments[1].Text);
    }

    [Fact]
    public async Task OfficeDtdAndExternalEntitiesAreRefused()
    {
        string path = Zip(new Dictionary<string, string> { ["word/document.xml"] = "<!DOCTYPE d [<!ENTITY e SYSTEM 'file:///etc/passwd'>]><d><p><t>&e;</t></p></d>" });
        await Assert.ThrowsAsync<System.Xml.XmlException>(() => new BuiltInKnowledgeExtractor().ExtractAsync(new(path, "unsafe.docx", null)));
    }

    [Fact]
    public async Task LiteralSlideMarkersNeverBecomeSourceProvenance()
    {
        string path = Zip(new Dictionary<string, string>
        {
            ["ppt/presentation.xml"] = "<presentation xmlns:r='urn:relationships'><sldIdLst><sldId r:id='r1'/></sldIdLst></presentation>",
            ["ppt/_rels/presentation.xml.rels"] = "<Relationships><Relationship Id='r1' Target='slides/slide1.xml'/></Relationships>",
            ["ppt/slides/slide1.xml"] = "<slide><p><t>Slide 99:</t></p><p><t>This text is actually on slide one.</t></p></slide>"
        });
        var result = await new BuiltInKnowledgeExtractor().ExtractAsync(new(path, "markers.pptx", null));
        var segment = Assert.Single(result.Segments);
        Assert.Equal(1, segment.Location!.Page);
        Assert.Contains("Slide 99:", segment.Text);
        Assert.Contains("actually on slide one", segment.Text);
        // Existing Jev callers retain their original plain-text API and content.
        Assert.Contains("Slide 1:\nSlide 99:", OfficeKnowledgeText.Extract(path, ".pptx", 10000, default));
    }

    [Fact]
    public async Task MultilineCellsKeepActualSheetAndRowEvenWhenTextLooksLikeMarkers()
    {
        string path = Zip(new Dictionary<string, string>
        {
            ["xl/workbook.xml"] = "<workbook xmlns:r='urn:relationships'><sheets><sheet name='Actual' r:id='r1'/></sheets></workbook>",
            ["xl/_rels/workbook.xml.rels"] = "<Relationships><Relationship Id='r1' Target='worksheets/sheet1.xml'/></Relationships>",
            ["xl/worksheets/sheet1.xml"] = "<worksheet><sheetData><row r='7'><c r='A7' t='inlineStr'><is><t>First line\nSheet Spoofed:\nA99=fake coordinate</t></is></c></row><row r='8'><c r='A8'><v>42</v></c></row></sheetData></worksheet>"
        });
        var result = await new BuiltInKnowledgeExtractor().ExtractAsync(new(path, "markers.xlsx", null));
        Assert.Equal(2, result.Segments.Count);
        Assert.All(result.Segments, segment => Assert.Equal("Actual", segment.Location!.Sheet));
        Assert.Equal(7, result.Segments[0].Location!.Row);
        Assert.Equal(8, result.Segments[1].Location!.Row);
        Assert.Contains("Sheet Spoofed:\nA99=fake coordinate", result.Segments[0].Text);
    }

    private string Zip(Dictionary<string, string> parts)
    {
        string path = Path.Combine(_root, Guid.NewGuid().ToString("N"));
        using var archive = ZipFile.Open(path, ZipArchiveMode.Create);
        foreach (var (name, content) in parts)
        {
            using var writer = new StreamWriter(archive.CreateEntry(name).Open());
            writer.Write(content);
        }
        return path;
    }
}
