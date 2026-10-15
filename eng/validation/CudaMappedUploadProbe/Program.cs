using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Win32.SafeHandles;
using TensorSharp.Cuda;

// Isolated exact-byte loader diagnostic. It changes no product loader, does not
// execute a model, and deliberately excludes setup, optional file-cache warmup and GPU
// readback/hash verification from each timed host-to-device upload.
internal static class Program
{
    private static unsafe int Main(string[] args)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i += 2)
        {
            if (i + 1 >= args.Length || !args[i].StartsWith("--", StringComparison.Ordinal))
                throw new ArgumentException("Use --file PATH [--offset BYTES] [--bytes BYTES] [--chunk-bytes BYTES] [--rounds N] [--residency resident|fresh_nonresident] [--output PATH].");
            if (!options.TryAdd(args[i], args[i + 1])) throw new ArgumentException("Duplicate option " + args[i]);
        }
        if (!options.TryGetValue("--file", out string? path)) throw new ArgumentException("--file is required.");
        long Number(string key, long fallback) => options.TryGetValue(key, out var value) ? long.Parse(value, System.Globalization.CultureInfo.InvariantCulture) : fallback;
        long offset = Number("--offset", 0), bytes = Number("--bytes", 1L << 30);
        int chunk = checked((int)Number("--chunk-bytes", 8 << 20)), rounds = checked((int)Number("--rounds", 3));
        string residency = options.GetValueOrDefault("--residency", "resident");
        if (residency is not ("resident" or "fresh_nonresident"))
            throw new ArgumentException("--residency must be resident or fresh_nonresident.");
        bool residentLane = residency == "resident";
        long fileBytes = new FileInfo(path).Length;
        if (offset < 0 || bytes < 1 || offset > fileBytes - bytes || chunk is < (1 << 20) or > (128 << 20) || rounds is < 1 or > 12)
            throw new ArgumentOutOfRangeException(nameof(args), "The selected range must fit the file; chunks are 1–128MiB and rounds 1–12.");
        using var file = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        var warmup = Stopwatch.StartNew();
        string expectedHash = FileHash(file, offset, bytes, chunk);
        double fileWarmupMilliseconds = warmup.Elapsed.TotalMilliseconds;
        var setup = Stopwatch.StartNew();
        using var context = CudaContext.Create(checked((int)Number("--device", 0)));
        context.MakeCurrent();
        using var stream = CudaStream.Create();
        IntPtr device = IntPtr.Zero;
        var buffers = new IntPtr[2]; var events = new IntPtr[2];
        var results = new List<object>();
        try
        {
            Driver.Check(Driver.MemAlloc(out device, (nuint)bytes), "allocating device range");
            for (int i = 0; i < 2; ++i)
            {
                Driver.Check(Driver.MemHostAlloc(out buffers[i], (nuint)chunk, 0), "allocating bounded pinned buffer");
                Driver.Check(Driver.EventCreate(out events[i], 2), "allocating copy event");
            }
            double setupMilliseconds = setup.Elapsed.TotalMilliseconds;
            // Some network/container filesystems bypass the mmap page cache
            // for positional reads. Warm a disposable mapping as well, verify
            // its bytes, then close it so every trial still starts with fresh PTEs.
            using var heldMapping = residentLane
                ? MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read) : null;
            double mappedWarmupMilliseconds = 0;
            if (heldMapping != null)
            {
                var mappedWarmup = Stopwatch.StartNew();
                if (MappedHash(heldMapping, offset, bytes, chunk) != expectedHash)
                    throw new InvalidDataException("Mapped file bytes differ from the positional source.");
                mappedWarmupMilliseconds = mappedWarmup.Elapsed.TotalMilliseconds;
            }
            var attempts = new List<object>();
            string[][] orders = [["pageable_mmap", "pinned_mmap", "pinned_pread"],
                ["pinned_pread", "pinned_mmap", "pageable_mmap"], ["pinned_mmap", "pageable_mmap", "pinned_pread"]];
            for (int round = 0; round < rounds; ++round)
                foreach (string variant in orders[round % orders.Length])
                {
                    // The resident lane holds one MMF and replaces views. The
                    // nonresident lane reopens the path for each fresh mapping;
                    // mincore must independently prove the selected starting state.
                    using var freshMapping = heldMapping == null
                        ? MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read) : null;
                    var mapping = heldMapping ?? freshMapping!;
                    using var view = mapping.CreateViewAccessor(offset, bytes, MemoryMappedFileAccess.Read);
                    byte* acquired = null; view.SafeMemoryMappedViewHandle.AcquirePointer(ref acquired);
                    try
                    {
                        byte* source = acquired + view.PointerOffset;
                        double? residentBefore = ResidentFraction((IntPtr)source, bytes);
                        attempts.Add(new { round, variant, residentBefore });
                        bool accepted = residentBefore.HasValue && (residentLane ? residentBefore >= .999 : residentBefore <= .001);
                        if (!accepted)
                        {
                            string reason = $"The {residency} lane requires {(residentLane ? ">=99.9%" : "<=0.1%")} residency before every trial; observed {residentBefore:P3}. No cache eviction or model-file writes are performed.";
                            var unavailable = new { status = "unavailable", reason, residency, file = Path.GetFullPath(path), fileBytes,
                                offset, bytes, chunkBytes = chunk, pinnedBytes = chunk * 2L, rounds,
                                fileWarmupMilliseconds, mappedWarmupMilliseconds, setupMilliseconds, expectedHash, attempts, results };
                            WriteReport(unavailable, options);
                            throw new InvalidOperationException(reason);
                        }
                        stream.Synchronize();
                        double readMilliseconds = 0, copyWaitMilliseconds = 0, submitMilliseconds = 0;
                        var elapsed = Stopwatch.StartNew();
                        if (variant == "pageable_mmap")
                        {
                            Driver.Check(Driver.MemcpyHtoD(device, (IntPtr)source, (nuint)bytes), "pageable mapped upload");
                            // Pageable synchronous H2D may return after staging,
                            // before its DMA completes on the legacy default stream.
                            Driver.Check(Driver.ContextSynchronize(), "finishing pageable upload");
                            submitMilliseconds = elapsed.Elapsed.TotalMilliseconds;
                        }
                        else
                        {
                            var pending = new bool[2]; int slot = 0;
                            for (long done = 0; done < bytes;)
                            {
                                int count = checked((int)Math.Min(chunk, bytes - done));
                                if (pending[slot])
                                {
                                    long waitStart = Stopwatch.GetTimestamp();
                                    Driver.Check(Driver.EventSynchronize(events[slot]), "waiting before pinned buffer reuse");
                                    copyWaitMilliseconds += Stopwatch.GetElapsedTime(waitStart).TotalMilliseconds;
                                }
                                long readStart = Stopwatch.GetTimestamp();
                                if (variant == "pinned_pread") ReadExactly(file, offset + done, buffers[slot], count);
                                else Buffer.MemoryCopy(source + done, (void*)buffers[slot], chunk, count);
                                readMilliseconds += Stopwatch.GetElapsedTime(readStart).TotalMilliseconds;
                                long submitStart = Stopwatch.GetTimestamp();
                                Driver.Check(Driver.MemcpyHtoDAsync(device + checked((nint)done), buffers[slot], (nuint)count, stream.Handle), "submitting pinned upload");
                                Driver.Check(Driver.EventRecord(events[slot], stream.Handle), "recording copy completion");
                                submitMilliseconds += Stopwatch.GetElapsedTime(submitStart).TotalMilliseconds;
                                pending[slot] = true; slot ^= 1; done += count;
                            }
                            long finalWait = Stopwatch.GetTimestamp(); stream.Synchronize();
                            copyWaitMilliseconds += Stopwatch.GetElapsedTime(finalWait).TotalMilliseconds;
                        }
                        double milliseconds = elapsed.Elapsed.TotalMilliseconds;
                        var verification = Stopwatch.StartNew();
                        string actualHash = DeviceHash(device, bytes, chunk);
                        if (actualHash != expectedHash) throw new InvalidDataException("GPU bytes do not match the selected file range.");
                        var row = new { round, variant, bytes, milliseconds, gibPerSecond = bytes / (1024.0 * 1024 * 1024) / (milliseconds / 1000),
                            readMilliseconds, copyWaitMilliseconds, submitMilliseconds, residentBefore,
                            verificationMilliseconds = verification.Elapsed.TotalMilliseconds, sha256 = actualHash };
                        results.Add(row); Console.WriteLine(JsonSerializer.Serialize(row));
                    }
                    finally { view.SafeMemoryMappedViewHandle.ReleasePointer(); }
                }
            var report = new { status = "complete", residency, file = Path.GetFullPath(path), fileBytes, offset, bytes, chunkBytes = chunk, pinnedBytes = chunk * 2L,
                rounds, device = checked((int)Number("--device", 0)), fileWarmupMilliseconds, mappedWarmupMilliseconds, setupMilliseconds,
                scope = residency + " exact-byte upload microbenchmark; excludes setup/warmup/verification, no model execution or disk-cache claim", expectedHash, attempts, results };
            WriteReport(report, options);
            return 0;
        }
        finally
        {
            // An exception can leave DMA reading a staging buffer. Finish the
            // stream before freeing buffers, events or the destination.
            stream.Synchronize();
            for (int i = 0; i < 2; ++i)
            {
                if (events[i] != IntPtr.Zero) Driver.EventDestroy(events[i]);
                if (buffers[i] != IntPtr.Zero) Driver.MemFreeHost(buffers[i]);
            }
            if (device != IntPtr.Zero) Driver.MemFree(device);
        }
    }

    private static void WriteReport(object report, Dictionary<string, string> options)
    {
        string json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
        if (options.TryGetValue("--output", out string? output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!); File.WriteAllText(output, json + "\n");
        }
        else Console.WriteLine(json);
    }

    private static unsafe void ReadExactly(SafeFileHandle file, long offset, IntPtr destination, int count)
    {
        int done = 0;
        while (done < count)
        {
            int read = RandomAccess.Read(file, new Span<byte>((byte*)destination + done, count - done), checked(offset + done));
            if (read == 0) throw new EndOfStreamException("Selected model range changed during upload.");
            done += read;
        }
    }
    private static unsafe string FileHash(SafeFileHandle file, long offset, long bytes, int chunk)
    {
        byte[] buffer = new byte[chunk]; using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        fixed (byte* pointer = buffer)
            for (long done = 0; done < bytes;)
            {
                int count = checked((int)Math.Min(chunk, bytes - done)); ReadExactly(file, offset + done, (IntPtr)pointer, count);
                hash.AppendData(buffer.AsSpan(0, count)); done += count;
            }
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
    private static unsafe string DeviceHash(IntPtr device, long bytes, int chunk)
    {
        byte[] buffer = new byte[chunk]; using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        fixed (byte* pointer = buffer)
            for (long done = 0; done < bytes;)
            {
                int count = checked((int)Math.Min(chunk, bytes - done));
                Driver.Check(Driver.MemcpyDtoH((IntPtr)pointer, device + checked((nint)done), (nuint)count), "verifying device bytes");
                hash.AppendData(buffer.AsSpan(0, count)); done += count;
            }
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
    private static unsafe string MappedHash(MemoryMappedFile mapping, long offset, long bytes, int chunk)
    {
        using var view = mapping.CreateViewAccessor(offset, bytes, MemoryMappedFileAccess.Read);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] buffer = new byte[chunk]; byte* acquired = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref acquired);
        try
        {
            byte* source = acquired + view.PointerOffset;
            fixed (byte* destination = buffer)
                for (long done = 0; done < bytes;)
                {
                    int count = checked((int)Math.Min(chunk, bytes - done));
                    Buffer.MemoryCopy(source + done, destination, chunk, count);
                    hash.AppendData(buffer.AsSpan(0, count)); done += count;
                }
            return Convert.ToHexStringLower(hash.GetHashAndReset());
        }
        finally { view.SafeMemoryMappedViewHandle.ReleasePointer(); }
    }
    private static unsafe double? ResidentFraction(IntPtr pointer, long length)
    {
        if (!OperatingSystem.IsLinux()) return null;
        long address = pointer.ToInt64(), page = Environment.SystemPageSize, aligned = address - address % page;
        long bytes = checked(address - aligned + length); int pages = checked((int)((bytes + page - 1) / page));
        byte[] vector = new byte[pages];
        fixed (byte* result = vector)
        {
            if (Driver.Mincore((IntPtr)aligned, (nuint)bytes, result) != 0) throw new IOException("mincore failed: " + Marshal.GetLastPInvokeError());
        }
        return vector.Count(value => (value & 1) != 0) / (double)pages;
    }
}

internal static partial class Driver
{
    static Driver() => NativeLibrary.SetDllImportResolver(typeof(Driver).Assembly, (name, _, _) => name == "cuda"
        ? NativeLibrary.Load(OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1") : IntPtr.Zero);
    internal static void Check(int code, string operation) { if (code != 0) throw new InvalidOperationException($"CUDA error {code} while {operation}."); }
    [LibraryImport("cuda", EntryPoint = "cuMemAlloc_v2")] internal static partial int MemAlloc(out IntPtr pointer, nuint bytes);
    [LibraryImport("cuda", EntryPoint = "cuCtxSynchronize")] internal static partial int ContextSynchronize();
    [LibraryImport("cuda", EntryPoint = "cuMemFree_v2")] internal static partial int MemFree(IntPtr pointer);
    [LibraryImport("cuda", EntryPoint = "cuMemHostAlloc")] internal static partial int MemHostAlloc(out IntPtr pointer, nuint bytes, uint flags);
    [LibraryImport("cuda", EntryPoint = "cuMemFreeHost")] internal static partial int MemFreeHost(IntPtr pointer);
    [LibraryImport("cuda", EntryPoint = "cuMemcpyHtoD_v2")] internal static partial int MemcpyHtoD(IntPtr destination, IntPtr source, nuint bytes);
    [LibraryImport("cuda", EntryPoint = "cuMemcpyHtoDAsync_v2")] internal static partial int MemcpyHtoDAsync(IntPtr destination, IntPtr source, nuint bytes, IntPtr stream);
    [LibraryImport("cuda", EntryPoint = "cuMemcpyDtoH_v2")] internal static partial int MemcpyDtoH(IntPtr destination, IntPtr source, nuint bytes);
    [LibraryImport("cuda", EntryPoint = "cuEventCreate")] internal static partial int EventCreate(out IntPtr ev, uint flags);
    [LibraryImport("cuda", EntryPoint = "cuEventRecord")] internal static partial int EventRecord(IntPtr ev, IntPtr stream);
    [LibraryImport("cuda", EntryPoint = "cuEventSynchronize")] internal static partial int EventSynchronize(IntPtr ev);
    [LibraryImport("cuda", EntryPoint = "cuEventDestroy_v2")] internal static partial int EventDestroy(IntPtr ev);
    [LibraryImport("libc", EntryPoint = "mincore", SetLastError = true)] internal static unsafe partial int Mincore(IntPtr pointer, nuint bytes, byte* vector);
}
