// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using TensorAgent.Core.Localization;

namespace TensorAgent.Core.Downloads;

/// <summary>Progress of one file transfer.</summary>
/// <param name="BytesReceived">Bytes on disk so far, including a resumed prefix.</param>
/// <param name="TotalBytes">Expected size when known, else null.</param>
/// <param name="BytesPerSecond">Recent transfer rate.</param>
/// <param name="Phase">"downloading" or "verifying".</param>
public readonly record struct DownloadProgress(long BytesReceived, long? TotalBytes, double BytesPerSecond, string Phase)
{
    public double? Fraction => TotalBytes is > 0 ? Math.Min(1.0, (double)BytesReceived / TotalBytes.Value) : null;
    public TimeSpan? Eta => TotalBytes is > 0 && BytesPerSecond > 1
        ? TimeSpan.FromSeconds((TotalBytes.Value - BytesReceived) / BytesPerSecond)
        : null;
}

/// <summary>Thrown when the bytes on disk do not match the catalog's SHA-256.</summary>
public sealed class DownloadIntegrityException(string path, string expected, string actual)
    : IOException(Loc.T("host.download.verificationFailed",
        ("file", Path.GetFileName(path)), ("expected", expected), ("actual", actual)))
{
    public string ExpectedSha256 { get; } = expected;
    public string ActualSha256 { get; } = actual;
}

/// <summary>
/// Downloads a multi-gigabyte file to a phone: resumable, cancellable, verified.
///
/// <para>
/// Why not TensorSharp.Runtime's <c>ModelDownloader</c>: it is synchronous, restarts from
/// byte zero on any failure and reports progress as console text. An app that gets
/// suspended, loses Wi-Fi or is cancelled mid-transfer needs <c>Range</c> resumption of the
/// kept <c>.part</c> file and a structured progress callback. The contract is otherwise the
/// same: temp file beside the destination, atomic rename, optional SHA-256.
/// </para>
/// <para>
/// Verification hashes the file while it streams; on a resume the already-present prefix
/// is hashed first so the check still covers every byte without a second pass at the end.
/// </para>
/// </summary>
public sealed class ResumableDownloader
{
    private readonly HttpClient _http;
    private readonly int _maxAttempts;

    public ResumableDownloader(HttpClient? http = null, int maxAttempts = 5)
    {
        _http = http ?? CreateDefaultClient();
        _maxAttempts = Math.Max(1, maxAttempts);
    }

    public static HttpClient CreateDefaultClient()
    {
        var handler = new SocketsHttpHandler
        {
            AllowAutoRedirect = true,           // Hugging Face 302s to a signed CDN URL
            MaxAutomaticRedirections = 10,
            ConnectTimeout = TimeSpan.FromSeconds(30),
            PooledConnectionLifetime = TimeSpan.FromMinutes(5),
        };
        var client = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("TensorAgent/1.0 (TensorSharp)");
        return client;
    }

    /// <summary>Path of the partial file kept between attempts.</summary>
    public static string PartPath(string destinationPath) => destinationPath + ".part";

    /// <summary>
    /// Download <paramref name="url"/> to <paramref name="destinationPath"/>. An existing
    /// complete file is returned as-is when <paramref name="expectedBytes"/> matches (the
    /// hash is not re-checked: that costs a full read of a 7 GB file on every launch and the
    /// store verifies once at download time). A <c>.part</c> file is resumed.
    /// </summary>
    public async Task DownloadAsync(
        string url,
        string destinationPath,
        long? expectedBytes,
        string? expectedSha256,
        IProgress<DownloadProgress>? progress,
        CancellationToken ct)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(url);
        ArgumentException.ThrowIfNullOrWhiteSpace(destinationPath);

        if (File.Exists(destinationPath))
        {
            long existing = new FileInfo(destinationPath).Length;
            if (expectedBytes is null || existing == expectedBytes)
            {
                progress?.Report(new DownloadProgress(existing, expectedBytes ?? existing, 0, "downloading"));
                return;
            }
            // A wrong-sized file is not a cache hit; it is a broken download.
            File.Delete(destinationPath);
        }

        Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);
        string part = PartPath(destinationPath);
        Exception? last = null;

        for (int attempt = 1; attempt <= _maxAttempts; attempt++)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                await TransferAsync(url, part, expectedBytes, expectedSha256, progress, ct).ConfigureAwait(false);
                File.Move(part, destinationPath, overwrite: true);
                return;
            }
            catch (OperationCanceledException)
            {
                throw;                          // the .part stays for the next run
            }
            catch (DownloadIntegrityException)
            {
                TryDelete(part);
                throw;
            }
            catch (Exception ex) when (attempt < _maxAttempts)
            {
                last = ex;
                // Transient network trouble: keep the .part, back off, and resume.
                await Task.Delay(TimeSpan.FromSeconds(Math.Min(30, 2 * attempt)), ct).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                last = ex;
            }
        }

        throw new IOException(Loc.T("host.download.failedAfterAttempts",
            ("url", url), ("attempts", _maxAttempts), ("reason", last?.Message)), last);
    }

    private async Task TransferAsync(
        string url, string part, long? expectedBytes, string? expectedSha256,
        IProgress<DownloadProgress>? progress, CancellationToken ct)
    {
        long offset = File.Exists(part) ? new FileInfo(part).Length : 0;
        if (expectedBytes is not null && offset > expectedBytes)
        {
            // Cannot be a prefix of the file we want.
            File.Delete(part);
            offset = 0;
        }

        using var hash = expectedSha256 is null ? null : IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        if (hash is not null && offset > 0)
        {
            progress?.Report(new DownloadProgress(0, expectedBytes, 0, "verifying"));
            await using var existing = new FileStream(part, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
            byte[] buf = new byte[1 << 20];
            int n;
            while ((n = await existing.ReadAsync(buf, ct).ConfigureAwait(false)) > 0)
                hash.AppendData(buf, 0, n);
        }

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        TensorSharp.Runtime.HuggingFaceAuth.Apply(request);
        if (offset > 0)
            request.Headers.Range = new RangeHeaderValue(offset, null);

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);

        if (offset > 0 && response.StatusCode == HttpStatusCode.OK)
        {
            // The server ignored the range and is sending the whole file: start the
            // part over rather than appending a second copy, and hash from scratch. The
            // response body we already have IS the full file, so no extra request.
            offset = 0;
            hash?.GetHashAndReset();
            File.Delete(part);
        }
        if (offset > 0 && response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            // Either the .part is already complete or the remote changed; let the size check decide.
            long size = new FileInfo(part).Length;
            if (expectedBytes is not null && size == expectedBytes)
            {
                await VerifyWholeFileAsync(part, expectedSha256, progress, ct).ConfigureAwait(false);
                return;
            }
            File.Delete(part);
            throw new IOException(Loc.T("host.download.partialMismatch"));
        }
        response.EnsureSuccessStatusCode();

        long? total = expectedBytes;
        if (total is null && response.Content.Headers.ContentLength is long len)
            total = offset + len;
        if (response.Content.Headers.ContentLength is long contentLength && expectedBytes is not null
            && offset + contentLength != expectedBytes)
        {
            throw new IOException(Loc.T("host.download.sizeMismatch",
                ("reported", (offset + contentLength).ToString("N0", Loc.Culture)),
                ("expected", expectedBytes.Value.ToString("N0", Loc.Culture))));
        }

        await using (var body = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false))
        await using (var file = new FileStream(part, FileMode.Append, FileAccess.Write, FileShare.None, 1 << 20, useAsync: true))
        {
            byte[] buf = new byte[1 << 20];
            long received = offset;
            var sw = Stopwatch.StartNew();
            long windowStart = received;
            double rate = 0;
            var lastReport = Stopwatch.StartNew();
            int n;
            while ((n = await body.ReadAsync(buf, ct).ConfigureAwait(false)) > 0)
            {
                await file.WriteAsync(buf.AsMemory(0, n), ct).ConfigureAwait(false);
                hash?.AppendData(buf, 0, n);
                received += n;
                if (sw.Elapsed.TotalSeconds >= 1)
                {
                    rate = (received - windowStart) / sw.Elapsed.TotalSeconds;
                    windowStart = received;
                    sw.Restart();
                }
                if (lastReport.ElapsedMilliseconds >= 250)
                {
                    progress?.Report(new DownloadProgress(received, total, rate, "downloading"));
                    lastReport.Restart();
                }
            }
            await file.FlushAsync(ct).ConfigureAwait(false);
            progress?.Report(new DownloadProgress(received, total, rate, "downloading"));

            if (expectedBytes is not null && received != expectedBytes)
                throw new IOException(Loc.T("host.download.connectionClosed",
                    ("received", received.ToString("N0", Loc.Culture)),
                    ("expected", expectedBytes.Value.ToString("N0", Loc.Culture))));
        }

        if (hash is not null)
        {
            string actual = Convert.ToHexStringLower(hash.GetHashAndReset());
            if (!string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
                throw new DownloadIntegrityException(part, expectedSha256!, actual);
        }
    }

    private static async Task VerifyWholeFileAsync(string path, string? expectedSha256, IProgress<DownloadProgress>? progress, CancellationToken ct)
    {
        if (expectedSha256 is null)
            return;
        progress?.Report(new DownloadProgress(0, new FileInfo(path).Length, 0, "verifying"));
        string actual = await ComputeSha256Async(path, ct).ConfigureAwait(false);
        if (!string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
        {
            TryDelete(path);
            throw new DownloadIntegrityException(path, expectedSha256, actual);
        }
    }

    public static async Task<string> ComputeSha256Async(string path, CancellationToken ct)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        using var sha = SHA256.Create();
        byte[] digest = await sha.ComputeHashAsync(stream, ct).ConfigureAwait(false);
        return Convert.ToHexStringLower(digest);
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { /* best effort */ }
    }
}
