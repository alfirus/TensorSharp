using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace TensorSharp.Knowledge;

/// <summary>Local managed document readers. Originals are never rewritten. Media understanding
/// is supplied by a host-owned extractor; unsupported formats remain explicitly metadata-only.</summary>
public sealed partial class BuiltInKnowledgeExtractor : IKnowledgeExtractor
{
    private readonly KnowledgeOptions _options;
    private readonly IKnowledgeExtractor? _media;
    private readonly IKnowledgePdfPageExtractor? _pdfPages;
    private static readonly HashSet<string> TextExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".txt", ".md", ".markdown", ".log", ".cs", ".py", ".js", ".ts", ".tsx", ".jsx", ".java",
        ".c", ".cpp", ".h", ".rs", ".go", ".sql", ".yaml", ".yml", ".toml", ".ini", ".xml",
        ".sh", ".ps1", ".tex", ".rst", ".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".srt", ".vtt",
        ".html", ".htm"
    };

    public BuiltInKnowledgeExtractor(KnowledgeOptions? options = null, IKnowledgeExtractor? mediaExtractor = null,
        IKnowledgePdfPageExtractor? pdfPageExtractor = null)
    {
        _options = options ?? new();
        _media = mediaExtractor;
        _pdfPages = pdfPageExtractor;
        if (_options.MaxPdfOcrPages is < 0 or > 4096 || _options.MaxPdfPagePixels is < 65536 or > 64_000_000
            || _options.MaxPdfEmbeddedImagePixels is < 1 or > 256_000_000 || _options.PdfOcrPageTimeoutSeconds is < 1 or > 600)
            throw new ArgumentOutOfRangeException(nameof(options), "PDF OCR page, pixel and time limits are outside supported bounds.");
    }

    public async Task<KnowledgeExtraction> ExtractAsync(KnowledgeExtractionContext context, CancellationToken cancellationToken = default)
    {
        var result = await ExtractCoreAsync(context, cancellationToken).ConfigureAwait(false);
        return result with { ExtractorId = result.ExtractorId ?? "tensorsharp-managed-documents-v1" };
    }

    private async Task<KnowledgeExtraction> ExtractCoreAsync(KnowledgeExtractionContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string extension = Path.GetExtension(context.FileName).ToLowerInvariant();
        if (extension == ".pdf") return await ExtractPdfAsync(context, cancellationToken).ConfigureAwait(false);
        if (extension is ".docx" or ".pptx" or ".xlsx")
        {
            var segments = OfficeKnowledgeText.ExtractSegments(context.FilePath, extension,
                _options.MaxExtractedCharacters, _options.MaxSegments, cancellationToken);
            return Complete(segments.ToList(), new[] { extension switch
            {
                ".docx" => "DOCX text, tables and referenced notes extracted. Pagination, embedded images and layout are not rendered.",
                ".xlsx" => "Cell coordinates and cached values extracted. Formulas are not recalculated; charts and images are not interpreted.",
                _ => "Slide text extracted in presentation order. Images, charts, speaker notes and layout are not interpreted.",
            } });
        }
        if (!TextExtensions.Contains(extension) && !(extension.Length == 0 && context.ContentType?.StartsWith("text/", StringComparison.OrdinalIgnoreCase) == true))
            return _media == null ? MetadataOnly("No content extractor is available for this format. The original is preserved; load a compatible media model or configure an extractor and reindex.")
                : await _media.ExtractAsync(context, cancellationToken).ConfigureAwait(false);

        string source = await ReadTextAsync(context.FilePath, cancellationToken).ConfigureAwait(false);
        if (extension is ".srt" or ".vtt") return ExtractSubtitles(source, cancellationToken);
        if (extension is ".csv" or ".tsv") return ExtractTable(source, extension == ".csv" ? ',' : '\t', cancellationToken);
        if (extension is ".jsonl" or ".ndjson") return ExtractJsonLines(source, cancellationToken);
        if (extension == ".json")
        {
            using var json = JsonDocument.Parse(source, new JsonDocumentOptions { MaxDepth = 64 });
            if (json.RootElement.ValueKind == JsonValueKind.Array)
                return Complete(json.RootElement.EnumerateArray().Select((item, index) =>
                    new KnowledgeSegment(item.GetRawText(), new KnowledgeLocation { Row = index + 1, Label = "$[" + index + "]" }, "table")).ToList());
        }
        if (extension is ".html" or ".htm")
        {
            // Text-only conversion: never render markup, execute scripts, or fetch external resources.
            source = Regex.Replace(source, @"<(script|style|noscript)\b[^>]*>.*?</\1\s*>", "", RegexOptions.IgnoreCase | RegexOptions.Singleline, TimeSpan.FromSeconds(1));
            source = Regex.Replace(source, @"</?(?:p|div|br|li|tr|h[1-6])\b[^>]*>", "\n", RegexOptions.IgnoreCase, TimeSpan.FromSeconds(1));
            source = WebUtility.HtmlDecode(Regex.Replace(source, "<[^>]*>", " ", RegexOptions.None, TimeSpan.FromSeconds(1)));
        }
        return ExtractLines(source, extension is ".md" or ".markdown", cancellationToken);
    }

    public static KnowledgeExtraction MetadataOnly(string warning) => new()
    {
        Status = KnowledgeDocumentStatus.MetadataOnly,
        Warnings = new[] { warning }
    };

    private async Task<string> ReadTextAsync(string path, CancellationToken ct)
    {
        using var stream = File.OpenRead(path);
        using var reader = new StreamReader(stream, new UTF8Encoding(false, true), detectEncodingFromByteOrderMarks: true);
        var output = new StringBuilder();
        char[] buffer = new char[8192];
        int read;
        while ((read = await reader.ReadAsync(buffer.AsMemory(), ct).ConfigureAwait(false)) > 0)
        {
            if (output.Length + read > _options.MaxExtractedCharacters)
                throw new KnowledgeLimitException("Extracted text exceeds the configured character limit; split the file.");
            if (buffer.AsSpan(0, read).Contains('\0')) throw new InvalidDataException("The file contains binary data, not text.");
            output.Append(buffer, 0, read);
        }
        return output.ToString().Replace("\r\n", "\n", StringComparison.Ordinal).Replace('\r', '\n');
    }

    private KnowledgeExtraction Complete(List<KnowledgeSegment> segments, IReadOnlyList<string>? warnings = null)
    {
        if (segments.Count > _options.MaxSegments || segments.Sum(x => (long)x.Text.Length) > _options.MaxExtractedCharacters)
            throw new KnowledgeLimitException("Extracted content exceeds the configured segment or character limit; split the file.");
        return new KnowledgeExtraction
        {
            Segments = segments,
            Status = segments.Count == 0 ? KnowledgeDocumentStatus.MetadataOnly : warnings?.Count > 0 ? KnowledgeDocumentStatus.Partial : KnowledgeDocumentStatus.Ready,
            Warnings = warnings ?? (segments.Count == 0 ? new[] { "The file has no readable text." } : Array.Empty<string>())
        };
    }

    private KnowledgeExtraction ExtractLines(string text, bool markdown, CancellationToken ct)
    {
        var segments = new List<KnowledgeSegment>();
        string? section = null;
        var block = new StringBuilder();
        int start = 1;
        string[] lines = text.Split('\n');
        void Flush(int end)
        {
            if (block.Length > 0 && !string.IsNullOrWhiteSpace(block.ToString()))
                segments.Add(new(block.ToString().TrimEnd(), new KnowledgeLocation { StartLine = start, EndLine = end, Section = section }, "text"));
            block.Clear();
        }
        for (int i = 0; i < lines.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            bool heading = markdown && Regex.IsMatch(lines[i], @"^#{1,6}\s", RegexOptions.None, TimeSpan.FromSeconds(1));
            if (heading || block.Length >= _options.ChunkCharacters * 4)
            {
                Flush(i);
                start = i + 1;
            }
            if (heading) section = lines[i].TrimStart('#', ' ');
            block.AppendLine(lines[i]);
        }
        Flush(lines.Length);
        return Complete(segments);
    }

    private KnowledgeExtraction ExtractJsonLines(string text, CancellationToken ct)
    {
        var segments = new List<KnowledgeSegment>();
        int line = 0;
        foreach (string value in text.Split('\n'))
        {
            ct.ThrowIfCancellationRequested(); line++;
            if (string.IsNullOrWhiteSpace(value)) continue;
            using var json = JsonDocument.Parse(value, new JsonDocumentOptions { MaxDepth = 64 });
            segments.Add(new(value, new KnowledgeLocation { Row = line, StartLine = line, EndLine = line }, "table"));
            if (segments.Count > _options.MaxSegments) throw new KnowledgeLimitException("Too many JSON records.");
        }
        return Complete(segments);
    }

    private KnowledgeExtraction ExtractTable(string text, char separator, CancellationToken ct)
    {
        var segments = new List<KnowledgeSegment>();
        var fields = new List<string>();
        var field = new StringBuilder();
        string[]? headers = null;
        bool quoted = false, closedQuote = false;
        int row = 1;
        long total = 0;
        void EndRow()
        {
            fields.Add(field.ToString()); field.Clear(); closedQuote = false;
            if (headers == null) headers = fields.ToArray();
            else if (fields.Any(x => x.Length > 0))
            {
                string value = string.Join(" | ", fields.Select((cell, index) => $"{(index < headers.Length ? headers[index] : "Column " + (index + 1))}: {cell}"));
                total += value.Length;
                if (total > _options.MaxExtractedCharacters || segments.Count >= _options.MaxSegments) throw new KnowledgeLimitException("Table exceeds extraction limits.");
                segments.Add(new(value, new KnowledgeLocation { Row = row }, "table"));
            }
            fields.Clear(); row++;
        }
        for (int i = 0; i < text.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            char ch = text[i];
            if (quoted)
            {
                if (ch == '"')
                {
                    if (i + 1 < text.Length && text[i + 1] == '"') { field.Append('"'); i++; }
                    else { quoted = false; closedQuote = true; }
                }
                else field.Append(ch);
            }
            else if (ch == separator) { fields.Add(field.ToString()); field.Clear(); closedQuote = false; }
            else if (ch == '\n') EndRow();
            else if (ch == '"' && field.Length == 0 && !closedQuote) quoted = true;
            else if (closedQuote || ch == '"') throw new InvalidDataException("Malformed quoted CSV field.");
            else field.Append(ch);
        }
        if (quoted) throw new InvalidDataException("CSV has an unterminated quoted field.");
        if (field.Length > 0 || fields.Count > 0 || closedQuote) EndRow();
        if (segments.Count == 0 && headers?.Length > 0) segments.Add(new("Columns: " + string.Join(" | ", headers), new KnowledgeLocation { Row = 1 }, "table"));
        return Complete(segments);
    }

    private KnowledgeExtraction ExtractSubtitles(string text, CancellationToken ct)
    {
        var segments = new List<KnowledgeSegment>();
        string[] lines = text.Split('\n');
        var timing = new Regex(@"^(?:(\d+):)?(\d{2}):(\d{2})[,.](\d{3})\s+-->\s+(?:(\d+):)?(\d{2}):(\d{2})[,.](\d{3})", RegexOptions.CultureInvariant, TimeSpan.FromSeconds(1));
        for (int i = 0; i < lines.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            var match = timing.Match(lines[i]);
            if (!match.Success) continue;
            double Seconds(int group) => (match.Groups[group].Success ? double.Parse(match.Groups[group].Value, CultureInfo.InvariantCulture) * 3600 : 0)
                + double.Parse(match.Groups[group + 1].Value, CultureInfo.InvariantCulture) * 60
                + double.Parse(match.Groups[group + 2].Value, CultureInfo.InvariantCulture)
                + double.Parse(match.Groups[group + 3].Value, CultureInfo.InvariantCulture) / 1000;
            double start = Seconds(1), end = Seconds(5);
            if (end < start) throw new InvalidDataException("Subtitle cue ends before it starts.");
            var cue = new StringBuilder();
            while (++i < lines.Length && !string.IsNullOrWhiteSpace(lines[i])) cue.AppendLine(lines[i]);
            if (cue.Length > 0) segments.Add(new(cue.ToString().Trim(), new KnowledgeLocation
            { StartSeconds = start, EndSeconds = end, StartMilliseconds = (long)(start * 1000), EndMilliseconds = (long)(end * 1000), Label = "Transcript" }, "audio"));
        }
        if (segments.Count == 0 && !string.IsNullOrWhiteSpace(text)) throw new InvalidDataException("No valid subtitle cues were found.");
        return Complete(segments);
    }
}
