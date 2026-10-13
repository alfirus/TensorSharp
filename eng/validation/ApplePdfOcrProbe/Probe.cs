using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Foundation;
using TensorSharp.Knowledge;
using TensorSharp.Models.Media;
using TensorSharp.Server.Knowledge;
using UIKit;

namespace ApplePdfOcrProbe;

internal static class Probe
{
    private const int MaxPixels = 16_777_216;
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
    private static string Fixture(string name) => Path.Combine(NSBundle.MainBundle.BundlePath, "Fixtures", name);
    private static void Check(bool condition, string message) { if (!condition) throw new InvalidOperationException(message); }

    public static async Task<bool> RunAsync()
    {
        string directory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
        Directory.CreateDirectory(directory);
        string resultPath = Path.Combine(directory, "probe-result.json");
        var cases = new List<object>();
        var timings = new List<object>();
        int failed = 0;
        var total = Stopwatch.StartNew();
        async Task Test(string name, Func<Task<object?>> action)
        {
            var timer = Stopwatch.StartNew();
            try
            {
                object? details = await action();
                cases.Add(new { name, passed = true, elapsedMs = timer.Elapsed.TotalMilliseconds, details });
                Console.WriteLine("PDF_PROBE_PASS " + name);
            }
            catch (Exception error)
            {
                failed++;
                cases.Add(new { name, passed = false, elapsedMs = timer.Elapsed.TotalMilliseconds, error = error.ToString() });
                Console.WriteLine("PDF_PROBE_FAIL " + name + " " + error);
            }
        }

        await Test("platform", () =>
        {
            Check(OperatingSystem.IsIOS() && ApplePdfOcr.IsSupported, "Probe must execute on supported iOS.");
            Check(!PortablePdfPageExtractor.IsSupported, "iOS must not advertise the desktop subprocess provider.");
            return Task.FromResult<object?>(new { os = RuntimeInformation.OSDescription, framework = RuntimeInformation.FrameworkDescription,
                architecture = RuntimeInformation.ProcessArchitecture.ToString(), ios = UIDevice.CurrentDevice.SystemVersion,
                simulator = IsSimulator, productionAssembly = typeof(ApplePdfOcr).Assembly.FullName });
        });

        string[] codes = ["CEDAR-4821", "ORBIT-7356", "HARBOR-9064", "BAMBOO-2187"];
        string[] totals = ["46", "43", "42", "45"];
        for (int page = 1; page <= 4; page++)
        {
            int number = page;
            await Test("scan-page-" + page, () =>
            {
                string text = Recognize(Fixture("scanned-four-pages.pdf"), number);
                Check(text.Contains(codes[number - 1], StringComparison.OrdinalIgnoreCase), text);
                Check(text.Contains(totals[number - 1], StringComparison.Ordinal), text);
                if (number == 4)
                    foreach (string phrase in new[] { "竹林", "蓝色", "二十七" }) Check(text.Contains(phrase), text);
                return Task.FromResult<object?>(new { text });
            });
        }

        await Test("mixed-selectable-and-multiple-image-blocks", () =>
        {
            string first = Recognize(Fixture("mixed-two-pages.pdf"), 1);
            string second = Recognize(Fixture("mixed-two-pages.pdf"), 2);
            Check(first.Contains("JUNIPER-5643", StringComparison.OrdinalIgnoreCase), first);
            foreach (string phrase in new[] { "ORBIT-7356", "Amber bolts", "43" })
                Check(second.Contains(phrase, StringComparison.OrdinalIgnoreCase), second);
            return Task.FromResult<object?>(new { first, second });
        });

        for (int page = 1; page <= 4; page++)
        {
            int number = page;
            await Test("crop-rotation-" + ((page - 1) * 90), () =>
            {
                foreach (int budget in new[] { MaxPixels, 65_536 })
                {
                    byte[] png = ApplePdfOcr.RenderPagePng(Fixture("crop-rotation.pdf"), number, budget);
                    byte[] pixels = ManagedMediaProvider.Instance.DecodeRgba(png, out int width, out int height);
                    (int wide, int high) = budget == MaxPixels ? (1666, 1111) : (313, 209);
                    Check((width, height) == (number % 2 == 1 ? (wide, high) : (high, wide)), $"Unexpected raster {width}x{height}.");
                    Check((long)width * height <= budget, "Raster exceeds pixel budget.");
                    CheckCorners(pixels, width, height, number);
                }
                string text = Recognize(Fixture("crop-rotation.pdf"), number);
                Check(text.Contains("CROP " + ((number - 1) * 90), StringComparison.OrdinalIgnoreCase), text);
                Check(!text.Contains("OUTSIDE") && !text.Contains("9463"), "Outside-crop text leaked: " + text);
                return Task.FromResult<object?>(new { text });
            });
        }

        await Test("limits-and-cancellation", () =>
        {
            string path = Fixture("scanned-four-pages.pdf");
            Throws<ArgumentOutOfRangeException>(() => ApplePdfOcr.RecognizePage(path, 0, MaxPixels));
            Throws<ArgumentOutOfRangeException>(() => ApplePdfOcr.RecognizePage(path, 5, MaxPixels));
            Throws<ArgumentOutOfRangeException>(() => ApplePdfOcr.RecognizePage(path, 1, 1));
            Throws<PdfOcrLimitException>(() => ApplePdfOcr.RecognizePage(path, 1, MaxPixels, maxCharacters: 10));
            using var canceled = new CancellationTokenSource();
            canceled.Cancel();
            Throws<OperationCanceledException>(() => ApplePdfOcr.RecognizePage(path, 1, MaxPixels, canceled.Token));
            Throws<InvalidDataException>(() => ApplePdfOcr.RecognizePage(Path.Combine(directory, "absent.pdf"), 1, MaxPixels));
            return Task.FromResult<object?>(null);
        });

        await Test("knowledge-import-search-provenance-and-original", async () =>
        {
            string root = Path.Combine(directory, "store-" + Guid.NewGuid().ToString("N"));
            // Explicitly unusable desktop paths prove auto selects the Apple branch on iOS.
            var pdf = new PdfKnowledgePageExtractor(new PdfOcrOptions { Provider = "auto",
                PdfToPpmExecutable = "/pdf-probe-missing-pdftoppm", TesseractExecutable = "/pdf-probe-missing-tesseract" });
            using var store = new KnowledgeStore(root, new BuiltInKnowledgeExtractor(pdfPageExtractor: pdf));
            var collection = await store.CreateCollectionAsync("iOS PDF OCR fixture");
            var documents = new List<object>();
            foreach (string name in new[] { "scanned-four-pages.pdf", "mixed-two-pages.pdf" })
            {
                var timer = Stopwatch.StartNew();
                await using var input = File.OpenRead(Fixture(name));
                var document = await store.ImportAsync(collection.Id, name, input, "application/pdf");
                Check(document.Status == KnowledgeDocumentStatus.Partial, $"Unexpected document status {document.Status}: {string.Join(';', document.Warnings)}");
                Check(document.ExtractorId?.Contains("apple-vision") == true, "Actual Apple OCR identity missing.");
                documents.Add(new { name, status = document.Status.ToString(), elapsedMs = timer.Elapsed.TotalMilliseconds, document.ExtractorId });
                await using var original = store.OpenOriginal(collection.Id, document.Id, document.Version);
                using var bytes = new MemoryStream();
                await original.CopyToAsync(bytes);
                byte[] expectedHash = SHA256.HashData(await File.ReadAllBytesAsync(Fixture(name)));
                Check(SHA256.HashData(bytes.ToArray()).SequenceEqual(expectedHash), "Original bytes changed.");
            }
            for (int i = 0; i < codes.Length; i++)
            {
                var hits = await store.SearchAsync(codes[i], new KnowledgeSearchOptions { CollectionIds = [collection.Id] });
                Check(hits.Any(hit => hit.FileName == "scanned-four-pages.pdf" && hit.Location?.Page == i + 1
                    && hit.Text.Contains(codes[i], StringComparison.OrdinalIgnoreCase)), "Search/provenance missing: " + codes[i]);
            }
            var chinese = await store.SearchAsync("蓝色 二十七", new KnowledgeSearchOptions { CollectionIds = [collection.Id] });
            Check(chinese.Any(hit => hit.Location?.Page == 4 && hit.Text.Contains("蓝色") && hit.Text.Contains("二十七")), "Chinese retrieval missing.");
            return new { documents };
        });

        int samples = int.TryParse(Environment.GetEnvironmentVariable("PDF_PROBE_SAMPLES"), out int requested) ? Math.Clamp(requested, 1, 10) : 3;
        await Test("ocr-benchmark", () =>
        {
            foreach (string name in new[] { "scanned-four-pages.pdf", "mixed-two-pages.pdf" })
            {
                int pages = name.StartsWith("scanned") ? 4 : 2;
                for (int page = 1; page <= pages; page++)
                {
                    var milliseconds = new List<double>();
                    int characters = 0;
                    for (int sample = 0; sample < samples; sample++)
                    {
                        var timer = Stopwatch.StartNew();
                        characters = Recognize(Fixture(name), page).Length;
                        milliseconds.Add(timer.Elapsed.TotalMilliseconds);
                    }
                    double[] sorted = milliseconds.Order().ToArray();
                    double median = (sorted[(samples - 1) / 2] + sorted[samples / 2]) / 2;
                    timings.Add(new { name, page, samplesMs = milliseconds, medianMs = median, minMs = sorted[0], maxMs = sorted[^1], characters });
                }
            }
            return Task.FromResult<object?>(new { samples, warm = true, notes = "Per-page render plus Vision OCR; earlier correctness cases warm frameworks. Simulator timings are host measurements, not phone performance." });
        });

        var result = new { status = "complete", passed = failed == 0, failed, elapsedMs = total.Elapsed.TotalMilliseconds,
            os = RuntimeInformation.OSDescription, framework = RuntimeInformation.FrameworkDescription,
            architecture = RuntimeInformation.ProcessArchitecture.ToString(), simulator = IsSimulator, cases, benchmarks = timings };
        string output = JsonSerializer.Serialize(result, Json);
        await File.WriteAllTextAsync(resultPath + ".tmp", output);
        File.Move(resultPath + ".tmp", resultPath, true);
        Console.WriteLine("PDF_PROBE_RESULT " + JsonSerializer.Serialize(result));
        return failed == 0;
    }

    private static string Recognize(string path, int page)
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(60));
        return ApplePdfOcr.RecognizePage(path, page, MaxPixels, timeout.Token);
    }

    private static void Throws<T>(Action action) where T : Exception
    {
        try { action(); }
        catch (T) { return; }
        throw new InvalidOperationException("Expected " + typeof(T).Name);
    }

    private static void CheckCorners(byte[] pixels, int width, int height, int page)
    {
        byte[][] colors = [[0, 0, 255, 255], [255, 255, 0, 255], [255, 0, 0, 255], [0, 255, 0, 255]];
        int[][] order = [[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]];
        (int x, int y)[] corners = [(width / 20, height / 20), (width * 19 / 20, height / 20),
            (width / 20, height * 19 / 20), (width * 19 / 20, height * 19 / 20)];
        for (int i = 0; i < corners.Length; i++)
        {
            var (x, y) = corners[i];
            Check(colors[order[page - 1][i]].AsSpan().SequenceEqual(pixels.AsSpan((y * width + x) * 4, 4)),
                $"Page {page}, corner {i} pixel mismatch: crop/rotation/DPI scale.");
        }
    }

    private static bool IsSimulator =>
#if PROBE_SIMULATOR
        true;
#else
        false;
#endif
}
