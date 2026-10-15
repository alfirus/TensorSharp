// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Security.Cryptography;
using System.Threading;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Downloads a model (or any companion file) referenced by a configuration
    /// file when it is not already present on disk. Each file may list several
    /// mirror URLs; they are tried in order until one succeeds, and the bytes are
    /// streamed to a <c>.part</c> temp file that is atomically renamed into place
    /// only after a complete, integrity-checked transfer — so an interrupted run
    /// never leaves a truncated file that a later run would mistake for a good
    /// download. Progress is written to the supplied <see cref="TextWriter"/>
    /// (the hosts pass <see cref="Console.Error"/>) so operators can see exactly
    /// what is being fetched and how far along it is.
    /// </summary>
    public static class ModelDownloader
    {
        private const string LogPrefix = "[model-download]";

        // One shared client: connection pooling across the mirror attempts, auto
        // redirect (HuggingFace and most CDNs 302 to a signed URL), and no overall
        // timeout because model files are large — the connect timeout below is what
        // makes a dead mirror fail fast enough to fall through to the next one.
        private static readonly HttpClient Http = CreateClient();

        private static HttpClient CreateClient()
        {
            var handler = new SocketsHttpHandler
            {
                AllowAutoRedirect = true,
                MaxAutomaticRedirections = 10,
                ConnectTimeout = TimeSpan.FromSeconds(30),
                PooledConnectionLifetime = TimeSpan.FromMinutes(5),
            };
            var client = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
            client.DefaultRequestHeaders.UserAgent.ParseAdd("TensorSharp-ModelDownloader/1.0");
            return client;
        }

        /// <summary>
        /// Ensure <paramref name="destinationPath"/> exists by downloading it from
        /// the first working URL in <paramref name="urls"/>. Throws when every
        /// mirror fails. The caller is expected to have already checked that the
        /// file is missing.
        /// </summary>
        /// <param name="destinationPath">Absolute local path the file should end up at.</param>
        /// <param name="urls">Candidate download URLs, tried in order.</param>
        /// <param name="sha256">Optional lowercase hex SHA-256 to verify the transfer; null to skip.</param>
        /// <param name="label">Human-readable name used in the log lines (e.g. "model").</param>
        /// <param name="log">Where progress is written (never null).</param>
        /// <param name="interactiveProgress">
        /// True to overwrite a single status line with a carriage return (a TTY);
        /// false to emit periodic full lines (a redirected/log sink).
        /// </param>
        public static void Download(
            string destinationPath,
            IReadOnlyList<string> urls,
            string? sha256,
            string label,
            TextWriter log,
            bool interactiveProgress)
        {
            if (string.IsNullOrWhiteSpace(destinationPath)) throw new ArgumentException("destinationPath is required.", nameof(destinationPath));
            if (urls == null || urls.Count == 0) throw new ArgumentException("At least one URL is required.", nameof(urls));
            log ??= TextWriter.Null;

            string? directory = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);

            string tempPath = destinationPath + ".part";
            var failures = new List<string>();

            for (int i = 0; i < urls.Count; i++)
            {
                string url = urls[i];
                if (string.IsNullOrWhiteSpace(url))
                    continue;

                log.WriteLine($"{LogPrefix} {label}: downloading from {url}");
                log.WriteLine($"{LogPrefix} {label}: saving to {destinationPath}");
                try
                {
                    DownloadOne(url, tempPath, sha256, label, log, interactiveProgress);

                    // Atomic publish: only now does the fully-verified file take
                    // its real name, so a crash mid-transfer never leaves a file a
                    // later run would treat as already-downloaded.
                    File.Move(tempPath, destinationPath, overwrite: true);
                    log.WriteLine($"{LogPrefix} {label}: done — saved {FormatSize(new FileInfo(destinationPath).Length)} to {destinationPath}");
                    return;
                }
                catch (Exception ex)
                {
                    TryDelete(tempPath);
                    string detail = $"{url} -> {ex.Message}";
                    failures.Add(detail);
                    bool more = HasAnotherUrl(urls, i);
                    log.WriteLine($"{LogPrefix} {label}: source failed ({ex.Message}){(more ? "; trying next source" : string.Empty)}");
                }
            }

            throw new IOException(
                $"Failed to download {label} to '{destinationPath}' from all {urls.Count} source(s):" +
                Environment.NewLine + "  " + string.Join(Environment.NewLine + "  ", failures));
        }

        private static bool HasAnotherUrl(IReadOnlyList<string> urls, int current)
        {
            for (int j = current + 1; j < urls.Count; j++)
                if (!string.IsNullOrWhiteSpace(urls[j]))
                    return true;
            return false;
        }

        private static void DownloadOne(
            string url, string tempPath, string? sha256, string label, TextWriter log, bool interactiveProgress)
        {
            using var request = new HttpRequestMessage(HttpMethod.Get, url);
            HuggingFaceAuth.Apply(request);
            using HttpResponseMessage response = Http.Send(request, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            long? total = response.Content.Headers.ContentLength;
            using Stream network = response.Content.ReadAsStream();
            using var file = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None);
            using IncrementalHash? hasher = sha256 != null
                ? IncrementalHash.CreateHash(HashAlgorithmName.SHA256)
                : null;

            byte[] buffer = new byte[1 << 20]; // 1 MiB
            long downloaded = 0;
            var stopwatch = Stopwatch.StartNew();
            var progress = new ProgressReporter(log, label, interactiveProgress);

            int read;
            while ((read = network.Read(buffer, 0, buffer.Length)) > 0)
            {
                file.Write(buffer, 0, read);
                hasher?.AppendData(buffer, 0, read);
                downloaded += read;
                progress.Maybe(downloaded, total, stopwatch.Elapsed.TotalSeconds);
            }

            file.Flush();
            progress.Finish(downloaded, total, stopwatch.Elapsed.TotalSeconds);

            if (total.HasValue && total.Value != downloaded)
                throw new IOException($"incomplete transfer: expected {total.Value} bytes, received {downloaded}");

            if (sha256 != null && hasher != null)
            {
                string actual = Convert.ToHexString(hasher.GetHashAndReset()).ToLowerInvariant();
                string expected = sha256.Trim().ToLowerInvariant();
                if (!string.Equals(actual, expected, StringComparison.Ordinal))
                    throw new IOException($"SHA-256 mismatch: expected {expected}, got {actual}");
                log.WriteLine($"{LogPrefix} {label}: SHA-256 verified ({actual})");
            }
        }

        private static void TryDelete(string path)
        {
            try { if (File.Exists(path)) File.Delete(path); }
            catch { /* best effort */ }
        }

        internal static string FormatSize(long bytes)
        {
            const double KiB = 1024, MiB = KiB * 1024, GiB = MiB * 1024;
            if (bytes >= GiB) return (bytes / GiB).ToString("F2") + " GiB";
            if (bytes >= MiB) return (bytes / MiB).ToString("F1") + " MiB";
            if (bytes >= KiB) return (bytes / KiB).ToString("F1") + " KiB";
            return bytes + " B";
        }

        /// <summary>
        /// Throttles progress output: overwrites a single carriage-return line on a
        /// TTY, or emits a fresh line at coarse intervals when redirected to a log.
        /// </summary>
        private sealed class ProgressReporter
        {
            private readonly TextWriter _log;
            private readonly string _label;
            private readonly bool _interactive;
            private DateTime _lastAt = DateTime.MinValue;
            private int _lastBucket = -1;
            private int _lastLineLength;

            public ProgressReporter(TextWriter log, string label, bool interactive)
            {
                _log = log;
                _label = label;
                _interactive = interactive;
            }

            public void Maybe(long downloaded, long? total, double elapsedSeconds)
            {
                var now = DateTime.UtcNow;
                if (_interactive)
                {
                    // Refresh the in-place line a few times a second.
                    if ((now - _lastAt).TotalMilliseconds < 250) return;
                }
                else if (total.HasValue && total.Value > 0)
                {
                    // Redirected + known size: one line every 10%.
                    int bucket = (int)(10.0 * downloaded / total.Value);
                    if (bucket == _lastBucket && (now - _lastAt).TotalSeconds < 15) return;
                    _lastBucket = bucket;
                }
                else
                {
                    // Redirected + unknown size: one line every 10 s.
                    if ((now - _lastAt).TotalSeconds < 10) return;
                }

                _lastAt = now;
                Write(Compose(downloaded, total, elapsedSeconds), final: false);
            }

            public void Finish(long downloaded, long? total, double elapsedSeconds)
            {
                Write(Compose(downloaded, total, elapsedSeconds), final: true);
            }

            private string Compose(long downloaded, long? total, double elapsedSeconds)
            {
                double speed = elapsedSeconds > 0 ? downloaded / elapsedSeconds : 0; // bytes/s
                string rate = FormatSize((long)speed) + "/s";
                if (total.HasValue && total.Value > 0)
                {
                    double pct = 100.0 * downloaded / total.Value;
                    double remaining = speed > 0 ? (total.Value - downloaded) / speed : 0;
                    string eta = TimeSpan.FromSeconds(remaining < 0 ? 0 : remaining).ToString(@"hh\:mm\:ss");
                    return $"{LogPrefix} {_label}: {pct,5:F1}% ({FormatSize(downloaded)} / {FormatSize(total.Value)}) {rate} ETA {eta}";
                }
                return $"{LogPrefix} {_label}: {FormatSize(downloaded)} {rate}";
            }

            private void Write(string line, bool final)
            {
                if (_interactive)
                {
                    // Pad to erase any leftover characters from a longer previous line.
                    int pad = Math.Max(0, _lastLineLength - line.Length);
                    _log.Write('\r' + line + new string(' ', pad));
                    _lastLineLength = line.Length;
                    if (final)
                        _log.Write(Environment.NewLine);
                }
                else
                {
                    _log.WriteLine(line);
                }
            }
        }
    }
}
