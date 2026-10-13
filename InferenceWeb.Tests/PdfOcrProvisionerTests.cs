using System.Net;
using System.Security.Cryptography;
using System.Text;
using TensorSharp.Knowledge;
using TensorSharp.Server.Knowledge;
using UglyToad.PdfPig.Writer;
using UglyToad.PdfPig.Fonts.Standard14Fonts;
using UglyToad.PdfPig.Core;

namespace InferenceWeb.Tests;

public sealed class PdfOcrProvisionerTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ocr-setup-tests-" + Guid.NewGuid().ToString("N"));

    [Fact]
    public void EmbeddedLockIncludesCompletePlatformClosuresAndPinnedBestLanguageCompanions()
    {
        var manifest = PdfOcrProvisioner.ReadManifest();
        Assert.Equal(new[] { "linux-arm64", "linux-x64", "win-x64" }, manifest.Runtimes.Select(r => r.Rid).Order());
        Assert.Equal(new[] { "chi_sim.traineddata", "chi_sim_vert.traineddata", "chi_tra.traineddata", "chi_tra_vert.traineddata", "eng.traineddata" },
            manifest.Languages.Select(a => a.FileName).Order(StringComparer.Ordinal));
        foreach (var runtime in manifest.Runtimes)
        {
            Assert.Contains(runtime.Packages, a => a.FileName.StartsWith("poppler-26."));
            Assert.Contains(runtime.Packages, a => a.FileName.StartsWith("tesseract-5."));
            Assert.Equal(runtime.Packages.Length, runtime.Packages.Select(a => a.FileName).Distinct().Count());
            using var provisioner = new PdfOcrProvisioner(_root, manifest, runtime.Rid, new HttpClient(new RejectNetwork()));
            Assert.Equal("not_started", provisioner.Status.State);
        }
    }

    [Fact]
    public async Task FirstUsePinsEveryDownloadPublishesHealthyPrefixAndSurvivesOfflineRestart()
    {
        using var fixture = new Fixture(_root);
        PdfOcrOptions installed;
        using (var first = fixture.Create())
        {
            var states = new List<string>();
            first.StatusChanged += status => { lock (states) states.Add(status.State); };
            installed = await first.PrepareAsync();
            Assert.Equal("ready", first.Status.State);
            Assert.Equal(first.Status.TotalBytes, first.Status.DownloadedBytes);
            Assert.Equal(3, fixture.Requests);
            Assert.Equal(1, fixture.Installs);
            Assert.Contains("downloading", states);
            Assert.Contains("installing", states);
            Assert.Equal("eng+chi_sim", installed.Languages);
            Assert.Contains("--offline", fixture.InstallArguments);
            Assert.Contains("--no-rc", fixture.InstallArguments);
            Assert.Contains("--no-env", fixture.InstallArguments);
            Assert.Contains("--skip-run-link-scripts", fixture.InstallArguments);
            string explicitFile = fixture.InstallArguments[Array.IndexOf(fixture.InstallArguments, "--file") + 1];
            string[] entries = await File.ReadAllLinesAsync(explicitFile);
            Assert.Equal("@EXPLICIT", entries[0]);
            Assert.All(entries.Skip(1), entry => Assert.True(new Uri(entry).IsFile));
            Assert.True(File.Exists(Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(installed.TesseractExecutable))!, "tensorsharp-ocr-lock.json")));
        }
        fixture.AllowNetwork = false;
        using var restarted = fixture.Create();
        var offline = await restarted.PrepareAsync();
        Assert.Equal(installed.TesseractExecutable, offline.TesseractExecutable);
        Assert.Equal(3, fixture.Requests);
        Assert.Equal(1, fixture.Installs);
        Assert.Equal("ready", restarted.Status.State);
    }

    [Fact]
    public async Task CorruptDownloadNeverExecutesAndRetryReusesVerifiedArchives()
    {
        using var fixture = new Fixture(_root) { CorruptPackage = true };
        using var manager = fixture.Create();
        var error = await Assert.ThrowsAsync<PdfOcrSetupException>(() => manager.PrepareAsync());
        Assert.Contains("verification failed", error.Message);
        Assert.Equal(0, fixture.Installs);
        Assert.False(File.Exists(Path.Combine(_root, "ready.json")));
        Assert.Empty(Directory.GetFiles(_root, "*.download", SearchOption.AllDirectories));
        Assert.Equal("failed", manager.Status.State);
        Assert.True(manager.Status.Retryable);
        fixture.CorruptPackage = false;
        await manager.PrepareAsync();
        Assert.Equal("ready", manager.Status.State);
        Assert.Equal(1, fixture.Installs);
        Assert.InRange(fixture.Requests, 4, 6); // Verified siblings survive cancellation; incomplete siblings retry.
    }

    [Fact]
    public async Task TwoManagersSerializeInstallationUsingPrivateFilesystemLock()
    {
        using var fixture = new Fixture(_root);
        using var one = fixture.Create();
        using var two = fixture.Create();
        var results = await Task.WhenAll(one.PrepareAsync(), two.PrepareAsync());
        Assert.Equal(results[0].TesseractExecutable, results[1].TesseractExecutable);
        Assert.Equal(1, fixture.Installs);
        Assert.Equal(3, fixture.Requests);
    }

    [Fact]
    public async Task CancellingOneWaiterDoesNotInterruptAnother()
    {
        using var fixture = new Fixture(_root) { HoldDownloads = true };
        using var manager = fixture.Create();
        using var cancellation = new CancellationTokenSource();
        Task<PdfOcrOptions> first = manager.PrepareAsync(cancellation.Token);
        await fixture.DownloadStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Task<PdfOcrOptions> second = manager.PrepareAsync();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => first);
        fixture.ReleaseDownloads.TrySetResult();
        await second.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal("ready", manager.Status.State);
        Assert.Equal(1, fixture.Installs);
    }

    [Fact]
    public async Task LastWaiterCancellationStopsWorkAndImmediateRetryWaitsForCleanup()
    {
        using var fixture = new Fixture(_root) { HoldDownloads = true };
        using var manager = fixture.Create();
        using var cancellation = new CancellationTokenSource();
        Task<PdfOcrOptions> first = manager.PrepareAsync(cancellation.Token);
        await fixture.DownloadStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => first);
        fixture.HoldDownloads = false;
        await manager.PrepareAsync().WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal("ready", manager.Status.State);
        Assert.Equal(1, fixture.Installs);
        Assert.Empty(Directory.GetFiles(_root, "*.download", SearchOption.AllDirectories));
    }

    [Fact]
    public async Task BundledSeedCanProvisionWithoutAnyNetworkAndCorruptInstalledModelRepairsFromCache()
    {
        using var fixture = new Fixture(_root) { AllowNetwork = false };
        string seed = Path.Combine(_root, "seed");
        foreach (var asset in fixture.AllAssets)
        {
            string path = Path.Combine(seed, asset.Sha256, asset.FileName);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            await File.WriteAllBytesAsync(path, fixture.Contents[asset.FileName]);
        }
        using var manager = fixture.Create(seed);
        var before = await manager.PrepareAsync();
        string model = Path.Combine(before.TessdataDirectory, "eng.traineddata");
        await File.WriteAllTextAsync(model, "broken");
        var after = await manager.PrepareAsync();
        Assert.NotEqual(before.TesseractExecutable, after.TesseractExecutable);
        Assert.Equal(fixture.Contents["eng.traineddata"], await File.ReadAllBytesAsync(Path.Combine(after.TessdataDirectory, "eng.traineddata")));
        Assert.Equal(2, fixture.Installs);
        Assert.Equal(0, fixture.Requests);
        // Never remove a previously published prefix: another running host can still use it.
        Assert.True(File.Exists(before.TesseractExecutable));
    }

    [Fact]
    public async Task InstallerFailurePublishesNothingCleansPrefixAndCanRetryOffline()
    {
        using var fixture = new Fixture(_root) { FailInstall = true };
        using var manager = fixture.Create();
        await Assert.ThrowsAsync<PdfOcrSetupException>(() => manager.PrepareAsync());
        Assert.False(File.Exists(Path.Combine(_root, "ready.json")));
        Assert.Empty(Directory.GetDirectories(Path.Combine(_root, "runtimes")));
        fixture.FailInstall = false;
        fixture.AllowNetwork = false;
        await manager.PrepareAsync();
        Assert.Equal("ready", manager.Status.State);
        Assert.Equal(3, fixture.Requests);
    }

    [Fact]
    public async Task SetupFailureCannotReplaceAnExistingKnowledgeDocument()
    {
        var provider = new PreparingProvider();
        using var store = new KnowledgeStore(Path.Combine(_root, "store"), new BuiltInKnowledgeExtractor(pdfPageExtractor: provider));
        var collection = await store.CreateCollectionAsync("atomic import");
        byte[] bytes = ScannedPdf();
        using var original = new MemoryStream(bytes);
        var first = await store.ImportAsync(collection.Id, "scan.pdf", original);
        Assert.Equal(1, provider.Preparations); // Four pages share preparation, outside their individual deadline.
        provider.Fail = true;
        using var replacement = new MemoryStream(bytes.Concat("\n"u8.ToArray()).ToArray());
        await Assert.ThrowsAsync<PdfOcrSetupException>(() => store.ImportAsync(collection.Id, "scan.pdf", replacement));
        await Assert.ThrowsAsync<PdfOcrSetupException>(() => store.ReindexAsync(collection.Id, first.Id));
        var retained = Assert.Single(store.ListDocuments(collection.Id));
        Assert.Equal(first.Version, retained.Version);
        using var content = store.OpenOriginal(collection.Id, first.Id, first.Version);
        Assert.Equal(SHA256.HashData(bytes), SHA256.HashData(content));
        Assert.Empty(Directory.GetFiles(Path.Combine(_root, "store", "staging")));
    }

    [Fact]
    public async Task SelectableTextPdfAndOversizedImagesDoNotPrepareDependencies()
    {
        Directory.CreateDirectory(_root);
        var builder = new PdfDocumentBuilder();
        var font = builder.AddStandard14Font(Standard14Font.Helvetica);
        builder.AddPage(200, 200).AddText("Selectable PDF stays offline", 12, new PdfPoint(10, 100), font);
        string path = Path.Combine(_root, "text.pdf");
        await File.WriteAllBytesAsync(path, builder.Build());
        int preparations = 0;
        var provider = new ManagedPdfKnowledgePageExtractor(_ =>
        {
            preparations++;
            throw new IOException("network forbidden");
        }, new());
        var extractor = new BuiltInKnowledgeExtractor(new() { MaxPdfEmbeddedImagePixels = 1000 }, pdfPageExtractor: provider);
        var text = await extractor.ExtractAsync(new(path, "text.pdf", "application/pdf"));
        Assert.Contains(text.Segments, s => s.Text.Contains("stays offline"));
        await File.WriteAllBytesAsync(path, ScannedPdf());
        await extractor.ExtractAsync(new(path, "scan.pdf", "application/pdf"));
        Assert.Equal(0, preparations);
    }

    [Fact]
    public async Task PreparationTimeIsOutsidePerPageTimeout()
    {
        Directory.CreateDirectory(_root);
        string path = Path.Combine(_root, "scan.pdf");
        await File.WriteAllBytesAsync(path, ScannedPdf());
        var provider = new PreparingProvider { Delay = TimeSpan.FromMilliseconds(1250) };
        var result = await new BuiltInKnowledgeExtractor(new() { PdfOcrPageTimeoutSeconds = 1 }, pdfPageExtractor: provider)
            .ExtractAsync(new(path, "scan.pdf", "application/pdf"));
        Assert.Equal(1, provider.Preparations);
        Assert.Equal(4, result.Segments.Count);
        Assert.DoesNotContain(result.Warnings, w => w.Contains("time limit"));
    }

    [Fact]
    public async Task NullBinaryCacheMarkerRepairsInsteadOfRepeatingFailure()
    {
        using var fixture = new Fixture(_root);
        using (var manager = fixture.Create()) await manager.PrepareAsync();
        string marker = Path.Combine(_root, "ready.json");
        var json = System.Text.Json.Nodes.JsonNode.Parse(await File.ReadAllTextAsync(marker))!;
        json["binaries"] = null;
        await File.WriteAllTextAsync(marker, json.ToJsonString());
        fixture.AllowNetwork = false;
        using var restart = fixture.Create();
        await restart.PrepareAsync();
        Assert.Equal("ready", restart.Status.State);
        Assert.Equal(2, fixture.Installs);
    }

    private static byte[] ScannedPdf()
    {
        using var resource = typeof(PdfOcrProvisionerTests).Assembly.GetManifestResourceStream("InferenceWeb.Tests.Fixtures.Knowledge.scanned-four-pages.pdf")!;
        using var output = new MemoryStream();
        resource.CopyTo(output);
        return output.ToArray();
    }

    private sealed class PreparingProvider : IKnowledgePdfPageExtractor
    {
        public bool Fail;
        public int Preparations;
        public TimeSpan Delay;
        public async Task PrepareAsync(CancellationToken cancellationToken = default)
        {
            Preparations++;
            if (Fail) throw new PdfOcrSetupException("offline setup failed");
            if (Delay > TimeSpan.Zero) await Task.Delay(Delay, cancellationToken);
        }
        public Task<KnowledgePdfPageExtraction> ExtractPageAsync(KnowledgePdfPageContext context, CancellationToken cancellationToken = default)
            => Task.FromResult(new KnowledgePdfPageExtraction { Text = "searchable extracted page " + context.PageNumber });
    }

    private sealed class RejectNetwork : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => throw new IOException("network is disabled for this test");
    }

    private sealed class Fixture : HttpMessageHandler
    {
        private readonly string _directory;
        private readonly HttpClient _client;
        private readonly PdfOcrManifest _manifest;
        public readonly Dictionary<string, byte[]> Contents = new()
        {
            ["micromamba"] = "bootstrap fixture"u8.ToArray(), ["native.conda"] = "native package fixture"u8.ToArray(),
            ["eng.traineddata"] = "language fixture"u8.ToArray()
        };
        public int Requests, Installs;
        public bool CorruptPackage, HoldDownloads, FailInstall;
        public bool AllowNetwork = true;
        public string[] InstallArguments = [];
        public TaskCompletionSource DownloadStarted = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource ReleaseDownloads = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public IEnumerable<PdfOcrDownloadAsset> AllAssets => _manifest.Runtimes[0].Packages
            .Prepend(_manifest.Runtimes[0].Bootstrap).Concat(_manifest.Languages);
        public Fixture(string directory)
        {
            _directory = directory;
            _client = new HttpClient(this, disposeHandler: false);
            PdfOcrDownloadAsset Asset(string name) => new() { FileName = name, Url = "https://fixtures.invalid/" + name,
                Bytes = Contents[name].Length, Sha256 = Convert.ToHexStringLower(SHA256.HashData(Contents[name])) };
            _manifest = new() { Version = 1, Id = "test-v1", Runtimes = [new() { Rid = "fixture-x64",
                Bootstrap = Asset("micromamba"), Packages = [Asset("native.conda")], RendererPath = "bin/pdftoppm", OcrPath = "bin/tesseract" }],
                Languages = [Asset("eng.traineddata")] };
        }
        public PdfOcrProvisioner Create(string? seed = null) => new(_directory, _manifest, "fixture-x64", _client, seed, Run);
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            if (!AllowNetwork) throw new IOException("network forbidden");
            Interlocked.Increment(ref Requests);
            DownloadStarted.TrySetResult();
            if (HoldDownloads) await ReleaseDownloads.Task.WaitAsync(ct);
            string file = request.RequestUri!.Segments[^1];
            byte[] bytes = Contents[file].ToArray();
            if (CorruptPackage && file == "native.conda") bytes[0] ^= 0x20;
            return new(HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) };
        }
        private async Task<PortablePdfProcessResult> Run(string exe, IReadOnlyList<string> arguments, string cwd,
            CancellationToken ct, IReadOnlyDictionary<string, string>? environment)
        {
            if (arguments[0] != "create") return new(0, "tesseract 5.5.3\n"u8.ToArray(), "", false);
            Interlocked.Increment(ref Installs);
            InstallArguments = arguments.ToArray();
            string prefix = InstallArguments[Array.IndexOf(InstallArguments, "--prefix") + 1];
            Directory.CreateDirectory(Path.Combine(prefix, "bin"));
            await File.WriteAllTextAsync(Path.Combine(prefix, "bin", "pdftoppm"), "renderer fixture", ct);
            await File.WriteAllTextAsync(Path.Combine(prefix, "bin", "tesseract"), "ocr fixture", ct);
            return new(FailInstall ? 1 : 0, [], FailInstall ? "installer fixture failure" : "", false);
        }
        protected override void Dispose(bool disposing) { if (disposing) _client.Dispose(); base.Dispose(disposing); }
    }

    public void Dispose() { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
}
