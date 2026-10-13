using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Knowledge;

namespace TensorSharp.Server.Knowledge;

internal sealed record PortablePdfProcessResult(int ExitCode, byte[] StandardOutput,
    string StandardError, bool ErrorTruncated);

/// <summary>Bounded binary pipes for trusted, host-configured PDF utilities. This is process
/// isolation, not an operating-system memory or filesystem sandbox.</summary>
internal static class PortablePdfProcessRunner
{
    internal const int MaximumErrorBytes = 16 * 1024;

    internal static async Task<PortablePdfProcessResult> RunAsync(string executable,
        IReadOnlyList<string> arguments, string workingDirectory, int maxOutputBytes,
        CancellationToken ct, ReadOnlyMemory<byte> input = default,
        IReadOnlyDictionary<string, string>? environment = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(executable);
        ArgumentNullException.ThrowIfNull(arguments);
        ArgumentException.ThrowIfNullOrWhiteSpace(workingDirectory);
        ArgumentOutOfRangeException.ThrowIfNegative(maxOutputBytes);
        ct.ThrowIfCancellationRequested();

        var start = new ProcessStartInfo(executable)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = workingDirectory,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        foreach (string argument in arguments)
            start.ArgumentList.Add(argument);
        if (environment is not null)
            foreach (var item in environment)
                start.Environment[item.Key] = item.Value;

        using var process = new Process { StartInfo = start };
        // Preserve Win32Exception from Start so the provider can distinguish a missing
        // dependency from a renderer rejecting a particular document.
        if (!process.Start())
            throw new IOException("The PDF utility could not be started.");
        using var failure = new CancellationTokenSource();
        using var operation = CancellationTokenSource.CreateLinkedTokenSource(ct, failure.Token);
        using var cancel = operation.Token.Register(() => Kill(process));

        Task<byte[]> outputTask = ReadOutputAsync(process.StandardOutput.BaseStream,
            maxOutputBytes, failure, operation.Token);
        Task<(string Text, bool Truncated)> errorTask = ReadErrorAsync(
            process.StandardError.BaseStream, failure, operation.Token);
        Task inputTask = WriteInputAsync(process.StandardInput.BaseStream, input, operation.Token);
        Task exitTask = process.WaitForExitAsync(operation.Token);
        try
        {
            // Read both streams while writing stdin: OCR utilities may fill stderr or
            // stdout before consuming the rest of a large raster supplied on stdin.
            await Task.WhenAll(outputTask, errorTask, inputTask, exitTask).ConfigureAwait(false);
            ct.ThrowIfCancellationRequested();
            var error = await errorTask.ConfigureAwait(false);
            return new PortablePdfProcessResult(process.ExitCode,
                await outputTask.ConfigureAwait(false), error.Text, error.Truncated);
        }
        catch
        {
            Kill(process);
            // Reap the process before its caller removes the private working directory.
            // The wait is bounded even if the OS refuses a termination request.
            using var reap = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            try { await process.WaitForExitAsync(reap.Token).ConfigureAwait(false); }
            catch (OperationCanceledException) { }
            // Caller cancellation takes precedence over secondary broken-pipe failures.
            ct.ThrowIfCancellationRequested();
            throw;
        }
    }

    private static async Task<byte[]> ReadOutputAsync(Stream stream, int maximum,
        CancellationTokenSource failure, CancellationToken ct)
    {
        try
        {
            using var output = new MemoryStream(Math.Min(maximum, 64 * 1024));
            byte[] buffer = new byte[16 * 1024];
            while (true)
            {
                int count = await stream.ReadAsync(buffer.AsMemory(), ct).ConfigureAwait(false);
                if (count == 0) return output.ToArray();
                if (count > maximum - output.Length)
                    throw new KnowledgeLimitException("PDF utility output exceeds the configured extraction limit.");
                output.Write(buffer, 0, count);
            }
        }
        catch
        {
            failure.Cancel();
            throw;
        }
    }

    private static async Task<(string, bool)> ReadErrorAsync(Stream stream,
        CancellationTokenSource failure, CancellationToken ct)
    {
        try
        {
            using var error = new MemoryStream(MaximumErrorBytes);
            byte[] buffer = new byte[4096];
            bool truncated = false;
            while (true)
            {
                int count = await stream.ReadAsync(buffer.AsMemory(), ct).ConfigureAwait(false);
                if (count == 0) return (Encoding.UTF8.GetString(error.GetBuffer(), 0, (int)error.Length), truncated);
                int retained = Math.Min(count, MaximumErrorBytes - (int)error.Length);
                error.Write(buffer, 0, retained);
                truncated |= retained != count;
                // Continue draining after the cap so verbose utilities cannot deadlock.
            }
        }
        catch
        {
            failure.Cancel();
            throw;
        }
    }

    private static async Task WriteInputAsync(Stream stream, ReadOnlyMemory<byte> input,
        CancellationToken ct)
    {
        try
        {
            if (!input.IsEmpty)
                await stream.WriteAsync(input, ct).ConfigureAwait(false);
            await stream.FlushAsync(ct).ConfigureAwait(false);
        }
        catch (IOException) when (!ct.IsCancellationRequested)
        {
            // A utility may reject the request and close stdin before reading it all.
            // Preserve its exit status and stderr instead of masking them with EPIPE.
        }
        finally
        {
            stream.Dispose(); // EOF is required even when there is no input.
        }
    }

    private static void Kill(Process process)
    {
        try { process.Kill(entireProcessTree: true); }
        catch (InvalidOperationException) { } // Already exited.
        catch (System.ComponentModel.Win32Exception) { } // Exit raced the signal.
    }
}
