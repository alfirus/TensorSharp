using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Knowledge;
using TensorSharp.Models.Media;
using TensorSharp.Server.Knowledge;

if (args.Length > 0 && args[0] == "--cold-profile")
{
    await ColdProfile.RunAsync(args.Skip(1).ToArray());
    return;
}

string source = Path.GetFullPath(args.Length > 0 ? args[0] : "InferenceWeb.Tests/Fixtures/Knowledge/scanned-four-pages.pdf");
string output = Path.GetFullPath(args.Length > 1 ? args[1] : "artifacts/knowledge/pdf-ocr/benchmark");
Directory.CreateDirectory(output);
if (!ApplePdfOcr.IsSupported) throw new PlatformNotSupportedException("This benchmark requires actual Apple Vision OCR; no pass is recorded on unsupported platforms.");
string[] codes = ["CEDAR-4821", "ORBIT-7356", "HARBOR-9064", "BAMBOO-2187"];
var samples = new List<object>();
var latencies = new List<double>();
for (int sample = 0; sample < 4; sample++)
{
    for (int page = 1; page <= 4; page++)
    {
        var clock = Stopwatch.StartNew();
        string text = ApplePdfOcr.RecognizePage(source, page, 16_777_216);
        double ms = clock.Elapsed.TotalMilliseconds;
        if (!text.Contains(codes[page - 1], StringComparison.OrdinalIgnoreCase)) throw new Exception($"Missing code on page {page}: {text}");
        if (sample > 0) latencies.Add(ms);
        samples.Add(new { sample, page, milliseconds = ms, characters = text.Length, text });
        if (sample == 0) await File.WriteAllBytesAsync(Path.Combine(output, $"page-{page}.png"), ApplePdfOcr.RenderPagePng(source, page, 16_777_216));
    }
}
string root = Path.Combine(Path.GetTempPath(), "ts-pdf-bench-" + Guid.NewGuid().ToString("N"));
object imported;
try
{
    using var store = KnowledgeRuntime.CreateStore(root);
    var collection = await store.CreateCollectionAsync("OCR fixture");
    await using var stream = File.OpenRead(source);
    var clock = Stopwatch.StartNew();
    var document = await store.ImportAsync(collection.Id, Path.GetFileName(source), stream, "application/pdf");
    double importMs = clock.Elapsed.TotalMilliseconds;
    var retrieval = new List<object>();
    for (int page = 1; page <= 4; page++)
    {
        clock.Restart();
        var hits = await store.SearchAsync(codes[page - 1], new() { CollectionIds = [collection.Id] });
        double ms = clock.Elapsed.TotalMilliseconds;
        if (hits.Count == 0 || hits[0].Location?.Page != page) throw new Exception($"Retrieval failed on page {page}");
        retrieval.Add(new { page, milliseconds = ms, hitPage = hits[0].Location?.Page });
    }
    imported = new { importMs, document.Status, document.ExtractorId, document.Warnings, retrieval };
}
finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
latencies.Sort();
var report = new { utc = DateTimeOffset.UtcNow, runtime = RuntimeInformation.FrameworkDescription, os = RuntimeInformation.OSDescription,
    source, sha256 = Convert.ToHexStringLower(SHA256.HashData(await File.ReadAllBytesAsync(source))),
    methodology = "One cold process, first pass reported separately, then three sequential warm passes across four original synthetic scan pages. No chat model or network. OCR and durable import/retrieval measured separately. Not a customer-corpus or concurrency benchmark; OCR is fallible.",
    warmPages = latencies.Count, warmP50Ms = latencies[(latencies.Count - 1) / 2], warmP95Ms = latencies[(int)Math.Ceiling(.95 * latencies.Count) - 1], samples, imported };
await File.WriteAllTextAsync(Path.Combine(output, "report.json"), JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine(JsonSerializer.Serialize(new { output, report.warmP50Ms, report.warmP95Ms }));
