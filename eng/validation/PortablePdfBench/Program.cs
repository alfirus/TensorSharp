using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Knowledge;
using TensorSharp.Models.Media;
using TensorSharp.Server.Knowledge;

// Deliberately fails on missing native dependencies or failed assertions; no simulated OCR.
var arguments = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 >= args.Length || !args[i].StartsWith("--")) throw new ArgumentException("Expected --name value pairs.");
    arguments.Add(args[i], args[i + 1]);
}
string Get(string name, string fallback) => arguments.GetValueOrDefault("--" + name, fallback);
string output = Path.GetFullPath(Get("output", "artifacts/knowledge/crossplatform/portable"));
string fixtures = Path.GetFullPath(Get("fixtures", "InferenceWeb.Tests/Fixtures/Knowledge"));
int samples = int.Parse(Get("samples", "3"));
if (samples is < 1 or > 20) throw new ArgumentOutOfRangeException(nameof(samples));
var options = new PdfOcrOptions
{
    Provider = Get("provider", "tesseract"), Languages = Get("languages", "eng+chi_sim"),
    PdfToPpmExecutable = Get("pdftoppm", null!), TesseractExecutable = Get("tesseract", null!),
    TessdataDirectory = Get("tessdata", null!)
};
if (options.Provider != "tesseract") throw new ArgumentException("This benchmark requires the actual tesseract provider.");
Directory.CreateDirectory(output);
var timings = new List<object>();
var checks = new List<string>();
var warm = new List<double>();
var storeTimings = new List<object>();
var geometry = new List<object>();
var cropTextChecks = new List<object>();
var accuracyFailures = new List<string>();
bool managed = bool.Parse(Get("managed", "false"));
var setupEvents = new List<PdfOcrSetupStatus>();
double? setupMilliseconds = null;
string? managedDirectory = managed ? Path.GetFullPath(Get("cache", Path.Combine(output, "managed-ocr"))) : null;
PdfOcrProvisioner? provisioner = null;
string storeStatus = "not_run";
var timer = Stopwatch.StartNew();
string status = "failed";
string? error = null;
using var manifest = JsonDocument.Parse(await File.ReadAllTextAsync(Path.Combine(fixtures, "manifest.json")));
var cases = manifest.RootElement.GetProperty("files").EnumerateArray().SelectMany(file =>
    file.GetProperty("pages").EnumerateArray().Select(page => new PageCase(file.GetProperty("fileName").GetString()!, page.Clone()))).ToArray();
void Require(bool condition, string name)
{
    if (!condition) throw new InvalidDataException(name);
    checks.Add(name);
}
void CheckText(PageCase item, string text)
{
    string compact = string.Concat(text.Where(c => !char.IsWhiteSpace(c)));
    Require(compact.Contains(item.Code, StringComparison.OrdinalIgnoreCase), $"{item.File} page {item.Page}: code {item.Code} (whitespace normalized)");
    if (item.Data.TryGetProperty("expectedTotal", out var total))
        Require(text.Contains(total.GetString()!, StringComparison.Ordinal), $"{item.File} page {item.Page}: total {total}");
    if (item.Data.TryGetProperty("chineseExpected", out var phrases))
    {
        foreach (var phrase in phrases.EnumerateArray()) Require(compact.Contains(phrase.GetString()!), "Chinese phrase: " + phrase.GetString());
    }
}
string root = Path.Combine(Path.GetTempPath(), "ts-portable-pdf-bench-" + Guid.NewGuid().ToString("N"));
try
{
    if (managed)
    {
        provisioner = new PdfOcrProvisioner(managedDirectory!, Get("bundled", null!));
        provisioner.StatusChanged += value =>
        {
            // Preserve stage changes, not thousands of byte-progress events.
            lock (setupEvents)
                if (setupEvents.Count == 0 || setupEvents[^1].State != value.State || setupEvents[^1].Stage != value.Stage)
                    setupEvents.Add(value);
        };
        var setupClock = Stopwatch.StartNew();
        options = await provisioner.PrepareAsync();
        setupMilliseconds = setupClock.Elapsed.TotalMilliseconds;
        Require(provisioner.Status.State == "ready", "App-managed OCR preparation completed");
        Require(File.Exists(options.PdfToPpmExecutable) && File.Exists(options.TesseractExecutable), "Managed tool executables exist");
        Require(Path.GetFullPath(options.PdfToPpmExecutable).StartsWith(managedDirectory! + Path.DirectorySeparatorChar, StringComparison.Ordinal), "Renderer is inside app-private cache");
        Require(Path.GetFullPath(options.TesseractExecutable).StartsWith(managedDirectory! + Path.DirectorySeparatorChar, StringComparison.Ordinal), "OCR is inside app-private cache");
    }
    var bridge = new PdfKnowledgePageExtractor(options);
    var portable = new PortablePdfPageExtractor(options);
    for (int sample = 0; sample <= samples; sample++)
    {
        foreach (var item in cases)
        {
            var clock = Stopwatch.StartNew();
            var result = await bridge.ExtractPageAsync(new(Path.Combine(fixtures, item.File), item.File, item.Page, 100000, 16_777_216));
            double ms = clock.Elapsed.TotalMilliseconds;
            timings.Add(new { sample, item.File, item.Page, milliseconds = ms, result.ExtractorId, result.Warnings, result.Text,
                codeExactMatch = result.Text?.Contains(item.Code, StringComparison.OrdinalIgnoreCase) == true,
                codeWhitespaceNormalizedMatch = string.Concat((result.Text ?? "").Where(c => !char.IsWhiteSpace(c))).Contains(item.Code, StringComparison.OrdinalIgnoreCase) });
            if (sample > 0) warm.Add(ms);
            CheckText(item, result.Text ?? "");
            Require(result.ExtractorId?.StartsWith("poppler-tesseract-v1;") == true, "Portable engine provenance recorded");
        }
    }
    foreach (int budget in new[] { 16_777_216, 65_536 })
    {
        for (int page = 1; page <= 4; page++)
        {
            byte[] png = await portable.RenderPageAsync(new(Path.Combine(fixtures, "crop-rotation.pdf"), "crop-rotation.pdf", page, 100000, budget));
            byte[] rgba = ManagedMediaProvider.Instance.DecodeRgba(png, out int width, out int height);
            Require((long)width * height <= budget, $"Crop page {page}: pixel budget {budget}");
            Require(page % 2 == 1 ? width > height : height > width, $"Crop page {page}: rotation dimensions");
            string[] colors = page switch
            {
                1 => ["blue", "yellow", "red", "green"], 2 => ["red", "blue", "green", "yellow"],
                3 => ["green", "red", "yellow", "blue"], _ => ["yellow", "green", "blue", "red"]
            };
            (int X, int Y)[] points = [(width / 20, height / 20), (width * 19 / 20, height / 20),
                (width / 20, height * 19 / 20), (width * 19 / 20, height * 19 / 20)];
            for (int corner = 0; corner < 4; corner++)
            {
                byte[] expected = colors[corner] switch { "red" => [255, 0, 0, 255], "green" => [0, 255, 0, 255], "blue" => [0, 0, 255, 255], _ => [255, 255, 0, 255] };
                Require(rgba.AsSpan((points[corner].Y * width + points[corner].X) * 4, 4).SequenceEqual(expected),
                    $"Crop page {page}: corner {corner} at budget {budget}");
            }
            geometry.Add(new { page, budget, width, height });
            await File.WriteAllBytesAsync(Path.Combine(output, $"crop-{page}-{budget}.png"), png);
        }
    }
    foreach (int page in new[] { 2, 4 })
    {
        var crop = await bridge.ExtractPageAsync(new(Path.Combine(fixtures, "crop-rotation.pdf"), "crop-rotation.pdf", page, 100000, 16_777_216));
        string expected = $"CROP {(page - 1) * 90}";
        bool matched = crop.Text?.Contains(expected, StringComparison.OrdinalIgnoreCase) == true;
        cropTextChecks.Add(new { page, expected, matched, crop.Text, crop.ExtractorId });
        // Preserve a strict quality failure, while still exercising the independent store
        // checks below. Never replace a mistaken OCR character with the fixture answer.
        if (matched) checks.Add($"Rotated crop {page}: OCR upright content");
        else accuracyFailures.Add($"Rotated crop {page}: expected '{expected}', actual '{crop.Text}'");
        Require(!crop.Text!.Contains("OUTSIDE", StringComparison.OrdinalIgnoreCase) && !crop.Text.Contains("9463"), $"Rotated crop {page}: excludes off-crop text");
    }
    var documents = new List<KnowledgeDocument>();
    var references = new List<string>();
    var knowledgeOptions = new KnowledgeOptions();
    IKnowledgeExtractor extractor = new BuiltInKnowledgeExtractor(knowledgeOptions, pdfPageExtractor: bridge);
    using (var store = new KnowledgeStore(root, extractor, options: knowledgeOptions))
    {
        var excluded = await store.CreateCollectionAsync("Excluded collection");
        foreach (string file in cases.Select(c => c.File).Distinct())
        {
            var collection = await store.CreateCollectionAsync(file);
            await using var input = File.OpenRead(Path.Combine(fixtures, file));
            var clock = Stopwatch.StartNew();
            var document = await store.ImportAsync(collection.Id, file, input, "application/pdf");
            storeTimings.Add(new { operation = "import", file, milliseconds = clock.Elapsed.TotalMilliseconds, document.Status, document.Warnings });
            Require(document.Status == KnowledgeDocumentStatus.Partial && document.ChunkCount > 0, file + ": OCR remains marked partial with searchable content");
            Require(!document.Warnings.Any(w => w.Contains("did not produce text", StringComparison.OrdinalIgnoreCase)
                || w.Contains("could not read", StringComparison.OrdinalIgnoreCase)), file + ": no missing OCR pages");
            Require(document.ExtractorId?.Contains("poppler-tesseract-v1;") == true, file + ": durable extractor provenance");
            using var original = store.OpenOriginal(collection.Id, document.Id, document.Version);
            byte[] expectedOriginal = SHA256.HashData(await File.ReadAllBytesAsync(Path.Combine(fixtures, file)));
            Require(SHA256.HashData(original).SequenceEqual(expectedOriginal), file + ": original bytes preserved");
            foreach (var item in cases.Where(c => c.File == file))
            {
                clock.Restart();
                var hits = await store.SearchAsync(item.Code, new() { CollectionIds = [collection.Id] });
                storeTimings.Add(new { operation = "search", file, page = item.Page, milliseconds = clock.Elapsed.TotalMilliseconds });
                var hit = hits.FirstOrDefault(h => h.Location?.Page == item.Page && h.Text.Contains(item.Code, StringComparison.OrdinalIgnoreCase));
                Require(hit != null, file + $": retrieves code with page {item.Page} citation");
                references.Add(hit!.Reference);
                CheckText(item, string.Join("\n", store.Read(hit.Reference, collectionIds: [collection.Id]).Select(h => h.Text)));
                Require((await store.SearchAsync(item.Code, new() { CollectionIds = [excluded.Id] })).Count == 0, "Excluded collection cannot search imported code");
                Require((await store.SearchAsync(item.Code, new())).Count == 0, "Empty collection scope cannot search imported code");
                bool rejected = false;
                try { store.Read(hit.Reference, collectionIds: [excluded.Id]); } catch (KeyNotFoundException) { rejected = true; }
                Require(rejected, "Citation read enforces collection scope");
            }
            clock.Restart();
            var reindexed = await store.ReindexAsync(collection.Id, document.Id);
            storeTimings.Add(new { operation = "reindex", file, milliseconds = clock.Elapsed.TotalMilliseconds });
            Require(reindexed.Version == document.Version + 1 && reindexed.Status == KnowledgeDocumentStatus.Partial, file + ": reindex advances OCR version");
            bool stale = false;
            try { store.Read(references[^1]); } catch (KnowledgeStaleReferenceException) { stale = true; }
            Require(stale, "Old citation rejected after reindex");
            documents.Add(reindexed);
        }
    }
    using (var reopened = new KnowledgeStore(root, extractor, options: knowledgeOptions))
    {
        foreach (var document in documents)
        {
            foreach (var item in cases.Where(c => c.File == document.FileName))
            {
                var hits = await reopened.SearchAsync(item.Code, new() { CollectionIds = [document.CollectionId] });
                Require(hits.Any(h => h.Location?.Page == item.Page && h.Version == document.Version), "Reopened store retains page citations and current version");
            }
        }
    }
    storeStatus = "passed";
    status = accuracyFailures.Count == 0 ? "passed" : "failed";
    if (accuracyFailures.Count > 0) error = string.Join("; ", accuracyFailures);
}
catch (Exception failure) { error = failure.ToString(); Console.Error.WriteLine(error); }
finally
{
    provisioner?.Dispose();
    if (Directory.Exists(root)) Directory.Delete(root, true);
    warm.Sort();
    double? Percentile(double quantile) => warm.Count == 0 ? null : warm[(int)Math.Ceiling(quantile * warm.Count) - 1];
    var report = new
    {
        status, error, utc = DateTimeOffset.UtcNow, os = RuntimeInformation.OSDescription, architecture = RuntimeInformation.ProcessArchitecture.ToString(),
        runtime = RuntimeInformation.FrameworkDescription, elapsedSeconds = timer.Elapsed.TotalSeconds,
        methodology = "Actual local Poppler+Tesseract, pinned original synthetic fixtures; first pass separate, then sequential warm six-page passes. Child processes start for every page. OCR assertions ignore whitespace only; exact code matches and raw OCR are retained separately. Store retrieval still requires literal codes at the expected page. Durable import, reindex and scoped citation retrieval checked separately. No chat/embedding model or ggml executed. No performance-equivalence, real-world corpus or concurrency claim.",
        warmSamples = warm.Count, warmP50Ms = Percentile(.5), warmP95Ms = Percentile(.95), percentileMethod = "nearest-rank",
        fixtureHashes = Directory.GetFiles(fixtures, "*.pdf").ToDictionary(path => Path.GetFileName(path)!, path => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)))),
        managed, managedDirectory, setupMilliseconds, setupEvents,
        checks, accuracyFailures, timings, geometry, cropTextChecks, storeStatus, storeTimings
    };
    await File.WriteAllTextAsync(Path.Combine(output, "report.json"), JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine(JsonSerializer.Serialize(new { status, checks = checks.Count, report.warmSamples, report.warmP50Ms, report.warmP95Ms, output }));
}
return status == "passed" ? 0 : 1;

sealed record PageCase(string File, JsonElement Data)
{
    public int Page => Data.GetProperty("page").GetInt32();
    public string Code => Data.GetProperty("expectedCode").GetString()!;
}
