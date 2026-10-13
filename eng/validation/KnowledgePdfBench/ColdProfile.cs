using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.Knowledge;
using TensorSharp.Models.Media;
using TensorSharp.Server.Knowledge;
using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;

/// <summary>Fresh-process stage timings; no product instrumentation, model loading or warm-up.</summary>
internal static class ColdProfile
{
    public static async Task RunAsync(string[] args)
    {
        string mode = args.Length > 0 ? args[0] : "import";
        string source = Path.GetFullPath(args.Length > 1 ? args[1] : "InferenceWeb.Tests/Fixtures/Knowledge/scanned-four-pages.pdf");
        string output = Path.GetFullPath(args.Length > 2 ? args[2] : $"artifacts/knowledge/pdf-ocr/cold-{mode}.json");
        if (mode != "parse" && !ApplePdfOcr.IsSupported)
            throw new PlatformNotSupportedException("Native OCR timing requires actual Apple Vision; use parse for the managed-reader profile.");
        var stages = new List<object>();
        var wall = Stopwatch.StartNew();
        double last = 0;
        void Record(string name, object? detail = null)
        {
            double now = wall.Elapsed.TotalMilliseconds;
            ThreadPool.GetAvailableThreads(out int worker, out int completion);
            var stage = new { name, elapsedMs = now, stageMs = now - last,
                cpuMs = Process.GetCurrentProcess().TotalProcessorTime.TotalMilliseconds,
                threadId = Environment.CurrentManagedThreadId, availableWorkerThreads = worker, detail };
            last = now; stages.Add(stage);
            Console.WriteLine(JsonSerializer.Serialize(stage));
        }
        Record("entry");
        if (mode is "native-main" or "native-worker")
        {
            for (int page = 1; page <= 4; page++)
            {
                string text = mode == "native-main"
                    ? ApplePdfOcr.RecognizePage(source, page, 16_777_216)
                    : await Task.Run(() => ApplePdfOcr.RecognizePage(source, page, 16_777_216));
                Record("native-page", new { page, characters = text.Length });
            }
        }
        else if (mode == "parse")
        {
            using var stream = File.OpenRead(source);
            Record("open-file");
            using var pdf = PdfDocument.Open(stream);
            Record("pdfpig-open");
            for (int number = 1; number <= pdf.NumberOfPages; number++)
            {
                var page = pdf.GetPage(number);
                Record("pdfpig-get-page", new { page = number });
                string text = ContentOrderTextExtractor.GetText(page, true);
                Record("pdfpig-text", new { page = number, characters = text.Length });
                var images = page.GetImages().Select(image => new { image.WidthInSamples, image.HeightInSamples }).ToArray();
                Record("pdfpig-images", new { page = number, images });
            }
        }
        else if (mode is "import" or "extract")
        {
            var extractor = new TimedExtractor(new BuiltInKnowledgeExtractor(pdfPageExtractor:
                new TimedPageExtractor(new PdfKnowledgePageExtractor(), Record)), Record);
            if (mode == "extract")
            {
                var result = await extractor.ExtractAsync(new(source, Path.GetFileName(source), "application/pdf"));
                Record("extract-complete", new { result.Status, segments = result.Segments.Count });
            }
            else
            {
                string root = Path.Combine(Path.GetTempPath(), "knowledge-cold-profile-" + Guid.NewGuid().ToString("N"));
                try
                {
                    using var store = new KnowledgeStore(root, extractor);
                    Record("store-constructed");
                    var collection = await store.CreateCollectionAsync("Cold OCR profile");
                    Record("collection-created");
                    await using var input = File.OpenRead(source);
                    var doc = await store.ImportAsync(collection.Id, Path.GetFileName(source), input, "application/pdf");
                    Record("import-complete", new { doc.Status, doc.ChunkCount });
                    doc = await store.ReindexAsync(collection.Id, doc.Id);
                    Record("reindex-complete", new { doc.Status, doc.Version, doc.ChunkCount });
                }
                finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
            }
        }
        else throw new ArgumentException("Mode must be import, extract, parse, native-main or native-worker.");
        Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        await File.WriteAllTextAsync(output, JsonSerializer.Serialize(new
        {
            utc = DateTimeOffset.UtcNow, mode, source, runtime = RuntimeInformation.FrameworkDescription,
            os = RuntimeInformation.OSDescription, process = Environment.ProcessId, executable = Environment.ProcessPath,
            methodology = "Fresh process with no intentional warm-up; system file and OCR caches are not cleared. Sequential local stages. CPU time is process-wide. No chat model, embedding provider or network.",
            stages
        }, new JsonSerializerOptions { WriteIndented = true }));
    }

    private sealed class TimedExtractor(IKnowledgeExtractor inner, Action<string, object?> record) : IKnowledgeExtractor
    {
        public async Task<KnowledgeExtraction> ExtractAsync(KnowledgeExtractionContext context, CancellationToken ct = default)
        {
            record("extractor-enter", null);
            var result = await inner.ExtractAsync(context, ct);
            record("extractor-exit", new { segments = result.Segments.Count });
            return result;
        }
    }

    private sealed class TimedPageExtractor(IKnowledgePdfPageExtractor inner, Action<string, object?> record) : IKnowledgePdfPageExtractor
    {
        public async Task<KnowledgePdfPageExtraction> ExtractPageAsync(KnowledgePdfPageContext context, CancellationToken ct = default)
        {
            record("ocr-enter", new { page = context.PageNumber });
            var result = await inner.ExtractPageAsync(context, ct);
            record("ocr-exit", new { page = context.PageNumber, characters = result.Text?.Length });
            return result;
        }
    }
}
