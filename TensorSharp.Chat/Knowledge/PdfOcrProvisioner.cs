using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Server.Knowledge;

public sealed record PdfOcrSetupStatus(string State, string Stage, long DownloadedBytes = 0,
    long TotalBytes = 0, string? Error = null, bool Retryable = false);

/// <summary>Dependency setup failed before the knowledge import or reindex was committed.</summary>
public sealed class PdfOcrSetupException(string message, Exception? inner = null) : IOException(message, inner);

/// <summary>Provisions a private, pinned Poppler/Tesseract runtime and English/Chinese models.
/// Only dependency bytes go over the network. PDFs and recognized text stay on the device.
/// No administrator rights, shell, system package manager or global environment changes.</summary>
public sealed class PdfOcrProvisioner : IDisposable
{
    private static readonly HttpClient Downloads = new() { Timeout = Timeout.InfiniteTimeSpan };
    private readonly string _directory;
    private readonly string? _bundledDirectory;
    private readonly PdfOcrManifest _manifest;
    private readonly PdfOcrRuntimeAsset? _runtime;
    private readonly string _manifestHash;
    private readonly HttpClient _http;
    private readonly Func<string, IReadOnlyList<string>, string, CancellationToken,
        IReadOnlyDictionary<string, string>?, Task<PortablePdfProcessResult>> _run;
    private readonly object _gate = new();
    private Task<PdfOcrOptions>? _pending;
    private CancellationTokenSource? _operation;
    private int _waiters;
    private bool _disposed;
    private PdfOcrOptions? _prepared;
    private PortablePdfPageExtractor.FileStamp[] _readyFiles = [];
    private PdfOcrSetupStatus _status = new("not_started", "PDF OCR will be prepared on first use.");
    private long _downloaded;
    private long _lastProgress;

    public PdfOcrProvisioner(string directory, string? bundledPackagesDirectory = null)
        : this(directory, ReadManifest(), CurrentRuntimeId, Downloads, bundledPackagesDirectory) { }

    internal PdfOcrProvisioner(string directory, PdfOcrManifest manifest, string? rid, HttpClient http,
        string? bundledPackagesDirectory = null,
        Func<string, IReadOnlyList<string>, string, CancellationToken,
            IReadOnlyDictionary<string, string>?, Task<PortablePdfProcessResult>>? run = null)
    {
        _directory = Path.GetFullPath(directory);
        _bundledDirectory = Path.GetFullPath(bundledPackagesDirectory ?? Path.Combine(AppContext.BaseDirectory, "ocr-assets"));
        _manifest = manifest;
        ValidateManifest(manifest);
        _manifestHash = Convert.ToHexStringLower(SHA256.HashData(
            JsonSerializer.SerializeToUtf8Bytes(manifest, PdfOcrAssetJson.Default.PdfOcrManifest)));
        _runtime = manifest.Runtimes.SingleOrDefault(r => r.Rid == rid);
        _http = http;
        _run = run ?? ((executable, arguments, cwd, ct, environment) =>
            PortablePdfProcessRunner.RunAsync(executable, arguments, cwd, 1024 * 1024, ct, environment: environment));
        if (_runtime == null) _status = new("unsupported", "Automatic PDF OCR is unavailable for this operating system/architecture.");
    }

    internal static string? CurrentRuntimeId => RuntimeInformation.RuntimeIdentifier.StartsWith("linux-musl-", StringComparison.Ordinal) ? null : RuntimeId(
        OperatingSystem.IsWindows() ? "windows" : OperatingSystem.IsLinux() ? "linux" : "other",
        RuntimeInformation.OSArchitecture, OperatingSystem.IsWindowsVersionAtLeast(10, 0, 22000));

    internal static string? RuntimeId(string platform, Architecture architecture, bool windows11) => (platform, architecture) switch
    {
        ("linux", Architecture.X64) => "linux-x64",
        ("linux", Architecture.Arm64) => "linux-arm64",
        ("windows", Architecture.X64) => "win-x64",
        // Windows 11 supplies x64 emulation on ARM64. The locked tools themselves are x64.
        ("windows", Architecture.Arm64) when windows11 => "win-x64",
        _ => null
    };

    public static bool IsSupported => CurrentRuntimeId != null;

    /// <summary>Explicit operator installations and disabled/native Apple providers take precedence.</summary>
    public static bool ShouldManage(PdfOcrOptions options) => IsSupported &&
        (options.Provider?.Trim().ToLowerInvariant() ?? "auto") is "auto" or "tesseract" &&
        string.IsNullOrWhiteSpace(options.PdfToPpmExecutable) &&
        string.IsNullOrWhiteSpace(options.TesseractExecutable) &&
        string.IsNullOrWhiteSpace(options.TessdataDirectory);

    public PdfOcrSetupStatus Status => Volatile.Read(ref _status);
    public event Action<PdfOcrSetupStatus>? StatusChanged;

    /// <summary>Concurrent callers share preparation; cancelling one caller does not cancel another.
    /// When every caller leaves, the transfer/installer is cancelled and verified downloads are retained.</summary>
    public async Task<PdfOcrOptions> PrepareAsync(CancellationToken cancellationToken = default)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Task<PdfOcrOptions> task;
            bool draining;
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_runtime == null) throw new PlatformNotSupportedException(Status.Stage);
                if (_prepared != null && _readyFiles.All(file => file.IsCurrent())) return _prepared;
                draining = _pending is { IsCompleted: false } && _operation!.IsCancellationRequested;
                if (_pending == null || _pending.IsCompleted)
                {
                    _operation?.Dispose();
                    _operation = new CancellationTokenSource(TimeSpan.FromMinutes(30));
                    _pending = Task.Run(() => PrepareCoreAsync(_operation.Token));
                }
                task = _pending;
                if (!draining) _waiters++;
            }
            if (draining)
            {
                try { await task.WaitAsync(cancellationToken).ConfigureAwait(false); }
                catch (Exception) when (!cancellationToken.IsCancellationRequested) { }
                continue;
            }
            try { return await task.WaitAsync(cancellationToken).ConfigureAwait(false); }
            finally
            {
                lock (_gate)
                {
                    if (--_waiters == 0 && _pending is { IsCompleted: false }) _operation!.Cancel();
                }
            }
        }
    }

    private async Task<PdfOcrOptions> PrepareCoreAsync(CancellationToken ct)
    {
        string? incompletePrefix = null;
        try
        {
            Directory.CreateDirectory(_directory);
            if (!OperatingSystem.IsWindows()) File.SetUnixFileMode(_directory,
                UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            SetStatus(new("checking", "Checking the private PDF OCR installation."));
            await using var installationLock = await AcquireLockAsync(ct).ConfigureAwait(false);
            string markerPath = Path.Combine(_directory, "ready.json");
            var ready = await ReadReadyAsync(markerPath, ct).ConfigureAwait(false);
            if (ready != null)
            {
                var existing = MakeOptions(InstallationPath(ready.Directory));
                if (await IsHealthyAsync(existing, ready, ct).ConfigureAwait(false)) return Complete(existing);
            }

            var assets = _runtime!.Packages.Prepend(_runtime.Bootstrap).Concat(_manifest.Languages).ToArray();
            long total = assets.Sum(asset => asset.Bytes);
            Interlocked.Exchange(ref _downloaded, 0);
            SetStatus(new("downloading", "Downloading verified OCR tools and English/Chinese language packs.", 0, total));
            // Bound simultaneous connections, memory and disk pressure; no solver or channel query is used.
            await Parallel.ForEachAsync(assets, new ParallelOptions { MaxDegreeOfParallelism = 4, CancellationToken = ct },
                async (asset, token) => await EnsureAssetAsync(asset, token).ConfigureAwait(false)).ConfigureAwait(false);

            string name = _manifest.Id + "-" + _runtime.Rid + "-" + Guid.NewGuid().ToString("N");
            incompletePrefix = InstallationPath(name);
            Directory.CreateDirectory(Path.GetDirectoryName(incompletePrefix)!);
            SetStatus(new("installing", "Installing the private OCR runtime.", total, total));
            string bootstrap = AssetPath(_runtime.Bootstrap);
            if (!OperatingSystem.IsWindows()) File.SetUnixFileMode(bootstrap,
                UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            string explicitFile = Path.Combine(_directory, "packages.explicit.txt");
            // Micromamba installs already-verified local archives offline. Do not rename the prefix
            // afterwards: conda's relocation embeds its absolute path into binaries/configuration.
            await File.WriteAllLinesAsync(explicitFile, new[] { "@EXPLICIT" }.Concat(_runtime.Packages
                .Select(asset => new Uri(AssetPath(asset)).AbsoluteUri)), ct).ConfigureAwait(false);
            var arguments = new[] { "create", "--yes", "--no-rc", "--no-env", "--offline", "--always-copy",
                "--skip-run-link-scripts", "--root-prefix", Path.Combine(_directory, "mamba"),
                "--prefix", incompletePrefix, "--file", explicitFile };
            var installed = await _run(bootstrap, arguments, _directory, ct, null).ConfigureAwait(false);
            if (installed.ExitCode != 0) throw new IOException("Private OCR installation failed: " + Diagnostic(installed));
            string data = Path.Combine(incompletePrefix, "tessdata");
            Directory.CreateDirectory(data);
            foreach (var language in _manifest.Languages)
                File.Copy(AssetPath(language), Path.Combine(data, language.FileName), overwrite: true);
            // Preserve the complete lock and upstream provenance beside the installed package license files.
            await File.WriteAllTextAsync(Path.Combine(incompletePrefix, "tensorsharp-ocr-lock.json"),
                JsonSerializer.Serialize(_manifest, PdfOcrAssetJson.Default.PdfOcrManifest), ct).ConfigureAwait(false);
            foreach (string resource in new[] { "NOTICE.md", "package-notices.json" })
            {
                await using var notice = typeof(PdfOcrProvisioner).Assembly.GetManifestResourceStream("TensorSharp.Chat.Knowledge.OcrAssets." + resource)
                    ?? throw new InvalidDataException("The PDF OCR license notices are missing.");
                await using var destination = File.Create(Path.Combine(incompletePrefix, resource));
                await notice.CopyToAsync(destination, ct).ConfigureAwait(false);
            }
            var options = MakeOptions(incompletePrefix);
            ready = new PdfOcrReady { ManifestHash = _manifestHash, Directory = name };
            foreach (string path in new[] { options.PdfToPpmExecutable, options.TesseractExecutable })
                ready.Binaries.Add(Path.GetRelativePath(incompletePrefix, path), await HashAsync(path, ct).ConfigureAwait(false));
            if (!await IsHealthyAsync(options, ready, ct).ConfigureAwait(false))
                throw new IOException("The downloaded OCR tools or language packs failed their startup check.");
            string temporaryMarker = markerPath + ".tmp";
            await File.WriteAllTextAsync(temporaryMarker, JsonSerializer.Serialize(ready, PdfOcrAssetJson.Default.PdfOcrReady), ct).ConfigureAwait(false);
            ct.ThrowIfCancellationRequested();
            File.Move(temporaryMarker, markerPath, overwrite: true);
            incompletePrefix = null; // Publication is atomic; other processes only see a healthy installation.
            return Complete(options);
        }
        catch (OperationCanceledException)
        {
            SetStatus(new("not_started", "OCR setup was interrupted. Retry to reuse verified downloads.",
                Interlocked.Read(ref _downloaded), Status.TotalBytes, Retryable: true));
            throw;
        }
        catch (Exception error)
        {
            SetStatus(new("failed", "PDF OCR setup failed. Retry after checking connectivity and free disk space.",
                Interlocked.Read(ref _downloaded), Status.TotalBytes, error.Message, true));
            throw new PdfOcrSetupException("PDF OCR setup failed: " + error.Message, error);
        }
        finally
        {
            if (incompletePrefix != null)
            {
                try { Directory.Delete(incompletePrefix, recursive: true); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }
    }

    private PdfOcrOptions Complete(PdfOcrOptions options)
    {
        _readyFiles = new[] { options.PdfToPpmExecutable, options.TesseractExecutable }.Concat(
            _manifest.Languages.Select(asset => Path.Combine(options.TessdataDirectory, asset.FileName)))
            .Select(PortablePdfPageExtractor.FileStamp.Read).ToArray();
        lock (_gate) _prepared = options;
        SetStatus(new("ready", "PDF OCR is ready. Tools and English/Chinese language packs are stored on this device.",
            Status.TotalBytes, Status.TotalBytes));
        return options;
    }

    private async Task<FileStream> AcquireLockAsync(CancellationToken ct)
    {
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            try { return new FileStream(Path.Combine(_directory, "setup.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None); }
            catch (IOException) { await Task.Delay(150, ct).ConfigureAwait(false); }
        }
    }

    private async Task EnsureAssetAsync(PdfOcrDownloadAsset asset, CancellationToken ct)
    {
        string path = AssetPath(asset);
        if (await VerifiedAsync(path, asset, ct).ConfigureAwait(false)) { Progress(asset.Bytes); return; }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        string temporary = path + ".download";
        long counted = 0;
        try
        {
            string? bundled = _bundledDirectory == null ? null : Path.Combine(_bundledDirectory, asset.Sha256, asset.FileName);
            if (bundled != null && await VerifiedAsync(bundled, asset, ct).ConfigureAwait(false))
            {
                await using var source = File.OpenRead(bundled);
                await using var target = File.Create(temporary);
                await source.CopyToAsync(target, ct).ConfigureAwait(false);
                Progress(asset.Bytes);
                counted = asset.Bytes;
            }
            else
            {
                using var response = await _http.GetAsync(asset.Url, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
                response.EnsureSuccessStatusCode();
                if (response.Content.Headers.ContentLength is long length && length != asset.Bytes)
                    throw new InvalidDataException("Unexpected download size for " + asset.FileName + ".");
                await using var source = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
                await using var target = File.Create(temporary);
                byte[] buffer = new byte[128 * 1024];
                int count;
                while ((count = await source.ReadAsync(buffer, ct).ConfigureAwait(false)) != 0)
                {
                    if (count > asset.Bytes - counted) throw new InvalidDataException("Download exceeded its pinned size: " + asset.FileName + ".");
                    await target.WriteAsync(buffer.AsMemory(0, count), ct).ConfigureAwait(false);
                    counted += count;
                    Progress(count);
                }
            }
            if (!await VerifiedAsync(temporary, asset, ct).ConfigureAwait(false))
                throw new InvalidDataException("SHA-256 verification failed for " + asset.FileName + ".");
            File.Move(temporary, path, overwrite: true);
        }
        catch { Progress(-counted); throw; }
        finally { File.Delete(temporary); }
    }

    private void Progress(long bytes)
    {
        long total = Interlocked.Add(ref _downloaded, bytes);
        long now = Environment.TickCount64, previous = Interlocked.Read(ref _lastProgress);
        if (now - previous < 200 || Interlocked.CompareExchange(ref _lastProgress, now, previous) != previous) return;
        var status = Status;
        SetStatus(status with { DownloadedBytes = total });
    }

    private static async Task<bool> VerifiedAsync(string path, PdfOcrDownloadAsset asset, CancellationToken ct)
        => File.Exists(path) && new FileInfo(path).Length == asset.Bytes &&
            string.Equals(await HashAsync(path, ct).ConfigureAwait(false), asset.Sha256, StringComparison.Ordinal);

    private static async Task<string> HashAsync(string path, CancellationToken ct)
    {
        await using var input = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(input, ct).ConfigureAwait(false));
    }

    private async Task<PdfOcrReady?> ReadReadyAsync(string path, CancellationToken ct)
    {
        if (!File.Exists(path)) return null;
        try
        {
            if (new FileInfo(path).Length > 16384) return null;
            var marker = JsonSerializer.Deserialize(await File.ReadAllTextAsync(path, ct).ConfigureAwait(false), PdfOcrAssetJson.Default.PdfOcrReady);
            return marker?.ManifestHash == _manifestHash && SafeName(marker.Directory) && marker.Binaries != null ? marker : null;
        }
        catch (JsonException) { return null; }
        catch (IOException) { return null; }
    }

    private async Task<bool> IsHealthyAsync(PdfOcrOptions options, PdfOcrReady marker, CancellationToken ct)
    {
        try
        {
            string prefix = InstallationPath(marker.Directory);
            foreach (string binary in new[] { options.PdfToPpmExecutable, options.TesseractExecutable })
            {
                if (!marker.Binaries.TryGetValue(Path.GetRelativePath(prefix, binary), out string? expected) ||
                    !File.Exists(binary) || await HashAsync(binary, ct).ConfigureAwait(false) != expected) return false;
            }
            foreach (var language in _manifest.Languages)
                if (!await VerifiedAsync(Path.Combine(options.TessdataDirectory, language.FileName), language, ct).ConfigureAwait(false)) return false;
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
            timeout.CancelAfter(TimeSpan.FromSeconds(30));
            var renderer = await _run(options.PdfToPpmExecutable, ["-v"], prefix, timeout.Token, options.ProcessEnvironment).ConfigureAwait(false);
            var ocr = await _run(options.TesseractExecutable, ["--version"], prefix, timeout.Token, options.ProcessEnvironment).ConfigureAwait(false);
            return renderer.ExitCode == 0 && ocr.ExitCode == 0 &&
                PortablePdfPageExtractor.IsSupportedVersion((Encoding.UTF8.GetString(ocr.StandardOutput) + ocr.StandardError).Split('\n')[0].Trim());
        }
        catch (Exception error) when (!ct.IsCancellationRequested && error is IOException or System.ComponentModel.Win32Exception or OperationCanceledException)
        { return false; }
    }

    private PdfOcrOptions MakeOptions(string prefix)
    {
        var environment = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var pair in _runtime!.Environment)
        {
            string value = WithinPrefix(prefix, pair.Value);
            // Preserve OS utility lookup for child DLL dependencies while giving the private directory priority.
            if (pair.Key == "PATH") value += Path.PathSeparator + (System.Environment.GetEnvironmentVariable("PATH") ?? "");
            environment.Add(pair.Key, value);
        }
        return new PdfOcrOptions { Provider = "tesseract", PdfToPpmExecutable = WithinPrefix(prefix, _runtime.RendererPath),
            TesseractExecutable = WithinPrefix(prefix, _runtime.OcrPath), TessdataDirectory = Path.Combine(prefix, "tessdata"),
            // Combining Simplified and Traditional models can normalize correctly recognized
            // Simplified glyphs into Traditional characters. Keep a predictable default; all
            // five packs are installed, and callers can select eng+chi_tra without another download.
            Languages = "eng+chi_sim", ProcessEnvironment = environment };
    }

    private string AssetPath(PdfOcrDownloadAsset asset) => Path.Combine(_directory, "downloads", asset.Sha256, asset.FileName);
    private string InstallationPath(string name) => Path.Combine(_directory, "runtimes", name);
    private static string WithinPrefix(string prefix, string relative)
    {
        string path = Path.GetFullPath(Path.Combine(prefix, relative));
        if (Path.IsPathRooted(relative) || !path.StartsWith(prefix + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            throw new InvalidDataException("Invalid OCR asset path.");
        return path;
    }
    private static string Diagnostic(PortablePdfProcessResult result) =>
        (result.StandardError + " " + Encoding.UTF8.GetString(result.StandardOutput)).Trim() is var text
            ? text[..Math.Min(text.Length, 2000)] : "Unknown error";
    private void SetStatus(PdfOcrSetupStatus status)
    {
        Volatile.Write(ref _status, status);
        if (StatusChanged is not { } handlers) return;
        foreach (Action<PdfOcrSetupStatus> handler in handlers.GetInvocationList())
        {
            try { handler(status); }
            catch (Exception) { } // An observer must not interrupt a verified installation.
        }
    }

    internal static PdfOcrManifest ReadManifest()
    {
        using var input = typeof(PdfOcrProvisioner).Assembly.GetManifestResourceStream("TensorSharp.Chat.Knowledge.OcrAssets.manifest.json")
            ?? throw new InvalidDataException("The PDF OCR dependency manifest is missing.");
        return JsonSerializer.Deserialize(input, PdfOcrAssetJson.Default.PdfOcrManifest)
            ?? throw new InvalidDataException("The PDF OCR dependency manifest is empty.");
    }

    private static bool SafeName(string value) => !string.IsNullOrEmpty(value) && value is not "." and not ".." &&
        value.All(c => char.IsAsciiLetterOrDigit(c) || c is '.' or '_' or '-');
    private static void ValidateManifest(PdfOcrManifest manifest)
    {
        if (manifest.Version != 1 || !SafeName(manifest.Id) || manifest.Runtimes.Length == 0 || manifest.Languages.Length == 0)
            throw new InvalidDataException("Invalid PDF OCR dependency manifest.");
        foreach (var runtime in manifest.Runtimes)
        {
            if (!SafeName(runtime.Rid) || runtime.Packages.Length == 0) throw new InvalidDataException("Invalid OCR platform.");
            foreach (var asset in runtime.Packages.Prepend(runtime.Bootstrap).Concat(manifest.Languages))
                if (!SafeName(asset.FileName) || asset.Bytes <= 0 || asset.Sha256.Length != 64 ||
                    asset.Sha256.Any(c => !char.IsAsciiHexDigit(c) || char.IsUpper(c)) ||
                    !Uri.TryCreate(asset.Url, UriKind.Absolute, out var uri) || uri.Scheme != "https")
                    throw new InvalidDataException("Invalid pinned OCR download.");
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            _operation?.Cancel();
        }
    }
}

internal sealed class PdfOcrManifest
{
    public int Version { get; init; }
    public string Id { get; init; } = "";
    public PdfOcrRuntimeAsset[] Runtimes { get; init; } = [];
    public PdfOcrDownloadAsset[] Languages { get; init; } = [];
}

internal sealed class PdfOcrRuntimeAsset
{
    public string Rid { get; init; } = "";
    public PdfOcrDownloadAsset Bootstrap { get; init; } = new();
    public PdfOcrDownloadAsset[] Packages { get; init; } = [];
    public string RendererPath { get; init; } = "";
    public string OcrPath { get; init; } = "";
    public Dictionary<string, string> Environment { get; init; } = [];
}

internal sealed class PdfOcrDownloadAsset
{
    public string Url { get; init; } = "";
    public string Sha256 { get; init; } = "";
    public long Bytes { get; init; }
    public string FileName { get; init; } = "";
    // Retain upstream provenance and license metadata without coupling installation to it.
    [JsonExtensionData] public Dictionary<string, JsonElement>? Metadata { get; set; }
}

internal sealed class PdfOcrReady
{
    public string ManifestHash { get; init; } = "";
    public string Directory { get; init; } = "";
    public Dictionary<string, string> Binaries { get; init; } = [];
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(PdfOcrManifest))]
[JsonSerializable(typeof(PdfOcrReady))]
internal partial class PdfOcrAssetJson : JsonSerializerContext { }
