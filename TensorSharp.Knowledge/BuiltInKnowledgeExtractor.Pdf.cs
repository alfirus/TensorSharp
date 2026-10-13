using System.Security.Cryptography;
using System.Text;
using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;

namespace TensorSharp.Knowledge;

public sealed partial class BuiltInKnowledgeExtractor
{
    private async Task<KnowledgeExtraction> ExtractPdfAsync(KnowledgeExtractionContext context, CancellationToken ct)
    {
        using var stream = File.OpenRead(context.FilePath);
        using var document = PdfDocument.Open(stream);
        if (document.NumberOfPages > Math.Min(_options.MaxSegments, 4096))
            throw new KnowledgeLimitException("PDF has too many pages; split the document.");
        var segments = new List<KnowledgeSegment>();
        var warnings = new List<string>();
        var recognizedPages = new List<int>();
        var missingPages = new List<int>();
        var limitedPages = new List<int>();
        var identities = new SortedSet<string>(StringComparer.Ordinal);
        int attemptedPages = 0, diagnosticsOmitted = 0;
        bool prepared = false;
        long characters = 0;
        void AddText(string text, int page, string modality)
        {
            if (string.IsNullOrWhiteSpace(text)) return;
            characters += text.Length;
            if (characters > _options.MaxExtractedCharacters || segments.Count >= _options.MaxSegments)
                throw new KnowledgeLimitException("PDF text exceeds the extraction limit; split the document.");
            segments.Add(new(text, new KnowledgeLocation { Page = page, Label = "Page " + page }, modality));
        }
        void AddDiagnostic(int page, string warning)
        {
            // Keep persisted metadata bounded even when thousands of pages cannot be read.
            if (warnings.Count < 80) warnings.Add($"Page {page}: {warning[..Math.Min(warning.Length, 2000)]}");
            else diagnosticsOmitted++;
        }
        for (int pageNumber = 1; pageNumber <= document.NumberOfPages; pageNumber++)
        {
            ct.ThrowIfCancellationRequested();
            var page = document.GetPage(pageNumber);
            string text = ContentOrderTextExtractor.GetText(page, true);
            AddText(text, pageNumber, "text");
            // Inspect headers only, never decode an embedded image. A selectable heading or
            // footer must not prevent recognition of a scanned body elsewhere on the page.
            bool hasImages = false, oversized = false;
            int imageCount = 0;
            long embeddedPixels = 0;
            foreach (var image in page.GetImages())
            {
                ct.ThrowIfCancellationRequested();
                hasImages = true;
                long pixels = (long)image.WidthInSamples * image.HeightInSamples;
                // Subtract before adding so even adversarial image headers cannot overflow.
                // Repeated image placements count again: rendering may decode each one.
                if (++imageCount > 4096 || image.WidthInSamples <= 0 || image.HeightInSamples <= 0
                    || pixels > _options.MaxPdfEmbeddedImagePixels - embeddedPixels)
                { oversized = true; break; }
                embeddedPixels += pixels;
            }
            if (!hasImages && !string.IsNullOrWhiteSpace(text)) continue;
            if (_pdfPages == null)
            { missingPages.Add(pageNumber); continue; }
            if (attemptedPages >= _options.MaxPdfOcrPages)
            { limitedPages.Add(pageNumber); continue; }
            if (oversized)
            {
                missingPages.Add(pageNumber);
                AddDiagnostic(pageNumber, "OCR was not attempted because embedded images exceed the configured per-page aggregate pixel or count limit. Split or downsample this page and reindex.");
                continue;
            }
            attemptedPages++;
            if (!prepared)
            {
                // Dependency provisioning may require a first-use download. Do it only for an
                // eligible OCR page, outside the page deadline and before committing any document.
                await _pdfPages.PrepareAsync(ct).ConfigureAwait(false);
                prepared = true;
            }
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
            deadline.CancelAfter(TimeSpan.FromSeconds(_options.PdfOcrPageTimeoutSeconds));
            KnowledgePdfPageExtraction result;
            try
            {
                result = await _pdfPages.ExtractPageAsync(new(context.FilePath, context.FileName, pageNumber,
                    (int)Math.Min(int.MaxValue, _options.MaxExtractedCharacters - characters), _options.MaxPdfPagePixels), deadline.Token).ConfigureAwait(false);
                deadline.Token.ThrowIfCancellationRequested();
            }
            catch (OperationCanceledException) when (!ct.IsCancellationRequested)
            {
                missingPages.Add(pageNumber);
                AddDiagnostic(pageNumber, "OCR exceeded its time limit; increase the page timeout or split the document and reindex.");
                continue;
            }
            catch (NotSupportedException error)
            {
                missingPages.Add(pageNumber);
                AddDiagnostic(pageNumber, error.Message);
                continue;
            }
            catch (InvalidDataException error)
            {
                // A failed page renderer/recognizer must not discard the other pages or
                // this page's selectable text. Contract/limit violations remain fatal:
                // returned metadata is validated below, and KnowledgeLimitException is
                // deliberately a different exception type.
                missingPages.Add(pageNumber);
                AddDiagnostic(pageNumber, "OCR could not read this page: " + error.Message);
                continue;
            }
            ct.ThrowIfCancellationRequested();
            if (result == null || result.Warnings == null || result.Warnings.Any(w => w == null)
                || result.ExtractorId?.Length > 1000)
                throw new InvalidDataException("PDF OCR provider returned invalid metadata.");
            if (result.ExtractorId != null) identities.Add(result.ExtractorId);
            foreach (string warning in result.Warnings.Take(81)) AddDiagnostic(pageNumber, warning);
            if (result.Warnings.Count > 81) diagnosticsOmitted += result.Warnings.Count - 81;
            if (string.IsNullOrWhiteSpace(result.Text))
            { missingPages.Add(pageNumber); continue; }
            if (result.Text.Length > _options.MaxExtractedCharacters - characters)
                throw new KnowledgeLimitException("PDF OCR text exceeds the extraction limit; split the document.");
            // The full-page OCR may repeat the existing text layer exactly. Keep the original
            // selectable text in that case, retaining OCR output when it adds page content.
            if (!EquivalentPdfText(text, result.Text)) AddText(result.Text, pageNumber, "ocr");
            recognizedPages.Add(pageNumber);
        }
        if (recognizedPages.Count > 0)
            warnings.Add($"OCR read complete rendered pages {PageRanges(recognizedPages)}. Recognized text may contain errors; verify important names, numbers and tables against the original.");
        if (missingPages.Count > 0)
            warnings.Add($"OCR did not produce text for pages {PageRanges(missingPages)}. These pages may be blank or contain unreadable scans; their selectable text, if present, is preserved."
                + (_pdfPages == null ? " No PDF OCR provider is configured. Configure a provider and reindex to read scanned content." : ""));
        if (limitedPages.Count > 0)
            warnings.Add($"OCR page limit ({_options.MaxPdfOcrPages}) was reached. Pages {PageRanges(limitedPages)} were not OCR-indexed; split the document or increase the limit and reindex.");
        if (diagnosticsOmitted > 0) warnings.Add($"{diagnosticsOmitted} additional page diagnostics omitted.");
        warnings.Add("PDF page numbers and extracted text are indexed. Visual meaning, exact table structure and layout coordinates are not interpreted.");
        string identity = "tensorsharp-pdf-v2";
        if (identities.Count == 1 && identities.First().Length <= 900) identity += ":" + identities.First();
        else if (identities.Count > 0) identity += ":ocr-sha256:" + Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(string.Join('\n', identities))));
        return Complete(segments, warnings) with { ExtractorId = identity };
    }

    private static bool EquivalentPdfText(string left, string right)
        => left.Where(c => !char.IsWhiteSpace(c)).SequenceEqual(right.Where(c => !char.IsWhiteSpace(c)));

    private static string PageRanges(List<int> pages)
    {
        var ranges = new List<string>();
        for (int index = 0; index < pages.Count; index++)
        {
            int start = pages[index], end = start;
            while (index + 1 < pages.Count && pages[index + 1] == end + 1) end = pages[++index];
            ranges.Add(start == end ? start.ToString() : $"{start}-{end}");
        }
        string result = string.Join(", ", ranges);
        return result.Length <= 3000 ? result : result[..3000] + $"… ({pages.Count} pages total)";
    }
}
