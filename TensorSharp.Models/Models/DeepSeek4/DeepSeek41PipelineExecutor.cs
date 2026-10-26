// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using TensorSharp.GGML;

namespace TensorSharp.Models;

/// <summary>
/// Experimental exact, single-sequence V4.1 pipeline. A verified 1017 checkpoint
/// is divided at layer 20, whose attention caches are independent of layers 0..19.
/// Transport is private TCP (use an SSH tunnel between hosts). No draft, vision,
/// slot batching, prefix retention, or cache truncation is advertised.
/// </summary>
internal sealed class DeepSeek41PipelineExecutor : IDisposable
{
    internal const int Magic = 0x34504950;
    // v3 adds a separate exact greedy-token result (never synthetic logits).
    // A version handshake refuses old peers before any model state is reused.
    internal const int Version = 3;
    internal const int Cut = 20;
    internal const int Layers = 40;
    private const int ForwardOp = 1, ResetOp = 2, ForwardNoLogitsOp = 3, ForwardGreedyOp = 4;
    private IntPtr _head;
    private readonly string _peer, _identity;
    private readonly int _hidden, _hc, _vocab, _context, _timeout;
    private readonly int _headUbatch;
    private readonly bool _overlapPrefill;
    private readonly bool _greedyTransport;
    private readonly Func<int[], float[], float[], int> _forwardHead;
    private readonly Action _resetHead, _releaseHead;
    private readonly Func<Stream> _connectionFactory;
    private readonly object _operationLock = new();
    private Thread _replyReader;
    private int _disposeRequested, _nPast;
    private bool _disposed;
    private TcpClient _client;
    private Stream _stream;
    private BinaryReader _reader;
    private BinaryWriter _writer;
    private bool _healthy;
    public int UBatch { get; private set; }
    public int NPast { get => Volatile.Read(ref _nPast); private set => Volatile.Write(ref _nPast, value); }
    internal bool HasPendingReplyReader => Volatile.Read(ref _replyReader) != null;
    internal bool GreedyTransportEnabled => _greedyTransport;

    public DeepSeek41PipelineExecutor(string model, int gpuCount, int context,
        int ubatch, int threads, string peer, string manifest)
    {
        _peer = peer;
        _context = context;
        _timeout = TimeoutMilliseconds();
        _overlapPrefill = Environment.GetEnvironmentVariable("TS_DSV41_PIPELINE_PREFILL_OVERLAP") == "1";
        _greedyTransport = Environment.GetEnvironmentVariable("TS_DSV41_PIPELINE_GREEDY_TRANSPORT") == "1";
        _identity = CheckpointIdentity(model, manifest);
        _head = GgmlDeepSeek4Native.LoadStage(model, gpuCount, context,
            ubatch, threads, "CUDA", 0, Cut);
        if (_head == IntPtr.Zero) throw new InvalidOperationException("DeepSeek pipeline head load failed (see stderr).");
        _releaseHead = () =>
        {
            if (_head != IntPtr.Zero) { GgmlDeepSeek4Native.Free(_head); _head = IntPtr.Zero; }
        };
        try
        {
            if (!GgmlDeepSeek4Native.StageInfo(_head, out int first, out int end, out _hidden, out _hc)
                || first != 0 || end != Cut)
                throw new InvalidOperationException("Invalid pipeline head dimensions.");
            _vocab = GgmlDeepSeek4Native.VocabSize(_head);
            _headUbatch = GgmlDeepSeek4Native.UBatch(_head);
            _forwardHead = (tokens, hidden, pre) =>
            {
                if (!GgmlDeepSeek4Native.StageForward(_head, tokens, null, null, hidden, pre, null))
                    throw new InvalidOperationException("Pipeline head forward failed (see stderr).");
                return GgmlDeepSeek4Native.NPast(_head);
            };
            _resetHead = () =>
            {
                if (!GgmlDeepSeek4Native.ResetChecked(_head)) throw new InvalidOperationException("Pipeline head reset failed.");
            };
            Connect();
        }
        catch { Dispose(); throw; }
    }

    // The same transport/orchestration with injected stage operations permits
    // fragmented real-TCP tests without loading a native library or a model.
    internal DeepSeek41PipelineExecutor(int context, int ubatch, int hidden, int hc, int vocab,
        Func<int[], float[], float[], int> forwardHead, Action resetHead, Action releaseHead,
        Func<Stream> connectionFactory, bool overlapPrefill, int timeoutMilliseconds = 5000, bool greedyTransport = false)
    {
        if (context <= 0 || ubatch <= 0 || hidden <= 0 || hc <= 0 || vocab <= 0 || timeoutMilliseconds <= 0)
            throw new ArgumentOutOfRangeException(nameof(context));
        _peer = "injected transport"; _identity = new string('a', 64);
        _context = context; _headUbatch = ubatch; _hidden = hidden; _hc = hc; _vocab = vocab;
        _timeout = timeoutMilliseconds; _overlapPrefill = overlapPrefill;
        _greedyTransport = greedyTransport;
        _forwardHead = forwardHead ?? throw new ArgumentNullException(nameof(forwardHead));
        _resetHead = resetHead ?? throw new ArgumentNullException(nameof(resetHead));
        _releaseHead = releaseHead ?? throw new ArgumentNullException(nameof(releaseHead));
        _connectionFactory = connectionFactory ?? throw new ArgumentNullException(nameof(connectionFactory));
        try { Connect(); }
        catch { Dispose(); throw; }
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposeRequested) != 0, this);

    private void Connect()
    {
        ThrowIfDisposed();
        CloseConnection();
        if (_connectionFactory == null)
        {
            IPEndPoint endpoint = ParseEndpoint(_peer);
            _client = new TcpClient(endpoint.AddressFamily) { NoDelay = true,
                ReceiveTimeout = _timeout, SendTimeout = _timeout };
            _client.ConnectAsync(endpoint.Address, endpoint.Port).WaitAsync(TimeSpan.FromMilliseconds(_timeout)).GetAwaiter().GetResult();
            _stream = _client.GetStream();
        }
        else _stream = _connectionFactory();
        if (_stream.CanTimeout) { _stream.ReadTimeout = _timeout; _stream.WriteTimeout = _timeout; }
        ThrowIfDisposed();
        _reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: true);
        _writer = new BinaryWriter(_stream, Encoding.UTF8, leaveOpen: true);
        _writer.Write(Magic); _writer.Write(Version); _writer.Write(_identity);
        _writer.Write(_hidden); _writer.Write(_hc); _writer.Write(_vocab); _writer.Write(_context);
        _writer.Write(_headUbatch); _writer.Flush();
        ReadSuccess();
        int width = _reader.ReadInt32();
        if (width <= 0 || width > _headUbatch)
            throw new InvalidDataException("Pipeline peer negotiated an invalid microbatch width.");
        UBatch = width;
        ThrowIfDisposed();
        _resetHead();
        NPast = 0; _healthy = true;
        Console.Error.WriteLine($"[dsv41-pipeline] head [0,{Cut}), peer {_peer} tail [{Cut},{Layers}), " +
            $"ubatch={UBatch}, identity={_identity}; exact single-sequence text");
    }

    public void Forward(int[] tokens, float[] logits)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(logits);
        lock (_operationLock)
        {
            ThrowIfDisposed();
            if (!_healthy) throw new InvalidOperationException("Pipeline state is unusable; reset the request before forwarding.");
            if (tokens.Length == 0 || logits.Length < _vocab || (long)NPast + tokens.Length > _context)
                throw new ArgumentException("Pipeline token/logit buffer or context bounds are invalid.");
            int start = NPast;
            try
            {
                if (_overlapPrefill && tokens.Length > UBatch)
                {
                    float[] completed = ForwardOverlapped(tokens, start);
                    completed.CopyTo(logits, 0);
                }
                else ForwardSequential(tokens, logits, start);
                // The two native caches may have advanced while a request is
                // in flight. Publish its new logical head only after every
                // reply, full F32 row and reader join completed successfully.
                NPast = checked(start + tokens.Length);
            }
            catch
            {
                _healthy = false;
                CloseConnection(); // overlapped path has already joined its reader
                throw;
            }
        }
    }

    public int ForwardGreedy(int token)
    {
        lock (_operationLock)
        {
            ThrowIfDisposed();
            if (!_greedyTransport) throw new InvalidOperationException("Pipeline greedy transport is disabled.");
            if (!_healthy) throw new InvalidOperationException("Pipeline state is unusable; reset the request before forwarding.");
            if (token < 0 || token >= _vocab) throw new ArgumentOutOfRangeException(nameof(token));
            if (NPast >= _context) throw new ArgumentException("Pipeline context has no room for another token.");
            int start = NPast;
            try
            {
                int[] tokens = [token];
                float[] hidden = new float[checked(_hidden * _hc)], pre = new float[_hc];
                var headTimer = Stopwatch.StartNew();
                if (_forwardHead(tokens, hidden, pre) != start + 1)
                    throw new InvalidDataException("Pipeline head advanced to an unexpected position.");
                double headMs = headTimer.Elapsed.TotalMilliseconds;
                var peerTimer = Stopwatch.StartNew();
                WriteForwardFrame(_stream, start, tokens, hidden, pre, greedy: true);
                ReadSuccess();
                if (_reader.ReadInt32() != start + 1)
                    throw new InvalidDataException("Pipeline greedy peer advanced to an unexpected position.");
                int nextToken = _reader.ReadInt32();
                if (nextToken < 0 || nextToken >= _vocab)
                    throw new InvalidDataException("Pipeline greedy peer returned an invalid token ID.");
                NPast = checked(start + 1);
                if (Environment.GetEnvironmentVariable("TS_DSV41_PIPELINE_PERF") == "1")
                    Console.Error.WriteLine(FormattableString.Invariant(
                        $"[dsv41-pipeline-perf] position={start} tokens=1 head_ms={headMs:F4} peer_ms={peerTimer.Elapsed.TotalMilliseconds:F4} greedy_transport=1 reply_bytes=12"));
                return nextToken;
            }
            catch
            {
                _healthy = false;
                CloseConnection();
                throw;
            }
        }
    }

    private void ForwardSequential(int[] tokens, float[] logits, int start)
    {
        for (int offset = 0; offset < tokens.Length; offset += UBatch)
        {
            int count = Math.Min(UBatch, tokens.Length - offset), expected = checked(start + offset);
            int[] chunk = tokens.AsSpan(offset, count).ToArray();
            float[] hidden = new float[checked(count * _hidden * _hc)], pre = new float[checked(count * _hc)];
            var headTimer = Stopwatch.StartNew();
            if (_forwardHead(chunk, hidden, pre) != expected + count)
                throw new InvalidDataException("Pipeline head advanced to an unexpected position.");
            double headMs = headTimer.Elapsed.TotalMilliseconds;
            var peerTimer = Stopwatch.StartNew();
            if (_overlapPrefill)
                WriteForwardFrame(_stream, expected, chunk, hidden, pre);
            else
            {
                _writer.Write(ForwardOp); _writer.Write(expected); _writer.Write(count); _writer.Flush();
                WriteArray(_stream, chunk); WriteArray(_stream, hidden); WriteArray(_stream, pre);
            }
            ReadSuccess();
            if (_reader.ReadInt32() != expected + count)
                throw new InvalidDataException("Pipeline stage cache positions differ.");
            ReadArray(_stream, logits.AsSpan(0, _vocab));
            if (Environment.GetEnvironmentVariable("TS_DSV41_PIPELINE_PERF") == "1")
                Console.Error.WriteLine(FormattableString.Invariant(
                    $"[dsv41-pipeline-perf] position={expected} tokens={count} head_ms={headMs:F4} peer_ms={peerTimer.Elapsed.TotalMilliseconds:F4}"));
        }
    }

    private float[] ForwardOverlapped(int[] tokens, int start)
    {
        int width = UBatch, chunks = checked((tokens.Length - 1) / width + 1);
        Stream stream = _stream;
        BinaryReader reader = _reader;
        float[] received = new float[_vocab];
        var headMs = new double[chunks];
        var peerMs = new double[chunks];
        var submitted = new long[chunks];
        using var cancellation = new CancellationTokenSource();
        using var permits = new SemaphoreSlim(2, 2);
        ExceptionDispatchInfo failure = null;
        int aborted = 0;
        void Abort()
        {
            if (Interlocked.Exchange(ref aborted, 1) != 0) return;
            cancellation.Cancel();
            AbortConnection(); // breaks a synchronous read/write before joining
        }
        var replyThread = new Thread(() =>
        {
            try
            {
                for (int index = 0; index < chunks; ++index)
                {
                    int offset = checked(index * width), count = Math.Min(width, tokens.Length - offset);
                    ReadSuccess(reader);
                    if (reader.ReadInt32() != checked(start + offset + count))
                        throw new InvalidDataException("Pipeline peer advanced to an unexpected position.");
                    if (index == chunks - 1) ReadArray(stream, received.AsSpan());
                    peerMs[index] = Stopwatch.GetElapsedTime(Volatile.Read(ref submitted[index])).TotalMilliseconds;
                    permits.Release();
                }
            }
            catch (Exception error)
            {
                Interlocked.CompareExchange(ref failure, ExceptionDispatchInfo.Capture(error), null);
                Abort();
            }
        }) { IsBackground = true, Name = "DeepSeek pipeline prefill replies" };
        Volatile.Write(ref _replyReader, replyThread);
        try
        {
            replyThread.Start();
            try
            {
                for (int index = 0; index < chunks; ++index)
                {
                    permits.Wait(cancellation.Token);
                    cancellation.Token.ThrowIfCancellationRequested();
                    int offset = checked(index * width), count = Math.Min(width, tokens.Length - offset);
                    int expected = checked(start + offset);
                    int[] chunk = tokens.AsSpan(offset, count).ToArray();
                    float[] hidden = new float[checked(count * _hidden * _hc)], pre = new float[checked(count * _hc)];
                    var headTimer = Stopwatch.StartNew();
                    // Only this caller thread touches the head executor/CUDA.
                    if (_forwardHead(chunk, hidden, pre) != expected + count)
                        throw new InvalidDataException("Pipeline head advanced to an unexpected position.");
                    headMs[index] = headTimer.Elapsed.TotalMilliseconds;
                    cancellation.Token.ThrowIfCancellationRequested();
                    Volatile.Write(ref submitted[index], Stopwatch.GetTimestamp());
                    WriteForwardFrame(stream, expected, chunk, hidden, pre, requestLogits: index == chunks - 1);
                }
            }
            catch (Exception error)
            {
                Interlocked.CompareExchange(ref failure, ExceptionDispatchInfo.Capture(error), null);
                Abort();
            }
            finally
            {
                // Closing the NetworkStream/Socket interrupts blocked reads.
                // A reader may never outlive this call, reset or native free.
                replyThread.Join();
            }
        }
        finally { Volatile.Write(ref _replyReader, null); }
        failure?.Throw();
        if (Environment.GetEnvironmentVariable("TS_DSV41_PIPELINE_PERF") == "1")
            for (int index = 0; index < chunks; ++index)
                Console.Error.WriteLine(FormattableString.Invariant(
                    $"[dsv41-pipeline-perf] position={start + index * width} tokens={Math.Min(width, tokens.Length - index * width)} head_ms={headMs[index]:F4} peer_ms={peerMs[index]:F4} prefill_overlap=1 inflight_limit=2 return_logits={(index == chunks - 1 ? 1 : 0)}"));
        return received;
    }

    public void Reset()
    {
        lock (_operationLock)
        {
            ThrowIfDisposed();
            try
            {
                if (!_healthy || _stream == null) { Connect(); return; }
                _writer.Write(ResetOp); _writer.Flush(); ReadSuccess();
                _resetHead();
                NPast = 0;
            }
            catch { _healthy = false; CloseConnection(); throw; }
        }
    }

    private void ReadSuccess() => ReadSuccess(_reader);

    private static void ReadSuccess(BinaryReader reader)
    {
        int status = reader.ReadInt32();
        if (status == 0) return;
        if (status != 1) throw new InvalidDataException("Pipeline peer returned an invalid status.");
        int bytes = reader.Read7BitEncodedInt();
        if (bytes < 0 || bytes > 4096) throw new InvalidDataException("Pipeline peer error text exceeds its bound.");
        byte[] message = reader.ReadBytes(bytes);
        if (message.Length != bytes) throw new EndOfStreamException("Incomplete pipeline peer error text.");
        throw new InvalidDataException("Pipeline peer refused the operation: " + Encoding.UTF8.GetString(message));
    }

    private void AbortConnection()
    {
        try { _client?.Client?.Shutdown(SocketShutdown.Both); } catch (SocketException) { } catch (ObjectDisposedException) { }
        try { _stream?.Dispose(); } catch (IOException) { } catch (ObjectDisposedException) { }
    }

    private void CloseConnection()
    {
        // A failed/background operation may already have closed the socket.
        // BinaryWriter.Dispose flushes even with leaveOpen, so cleanup must
        // continue past its expected closed-stream exception.
        try { _reader?.Dispose(); } catch (IOException) { } catch (ObjectDisposedException) { }
        try { _writer?.Dispose(); } catch (IOException) { } catch (ObjectDisposedException) { }
        try { _stream?.Dispose(); } catch (IOException) { } catch (ObjectDisposedException) { }
        try { _client?.Dispose(); } catch (SocketException) { }
        _reader = null; _writer = null; _stream = null; _client = null;
    }

    public void Dispose()
    {
        Interlocked.Exchange(ref _disposeRequested, 1);
        AbortConnection();
        lock (_operationLock)
        {
            if (_disposed) return;
            _disposed = true; _healthy = false;
            CloseConnection();
            _releaseHead?.Invoke();
        }
    }

    internal static int TimeoutMilliseconds()
    {
        string value = Environment.GetEnvironmentVariable("TS_DSV41_PIPELINE_TIMEOUT_SECONDS");
        int seconds = string.IsNullOrEmpty(value) ? 1800 : int.Parse(value);
        if (seconds < 1 || seconds > 86400) throw new ArgumentOutOfRangeException(nameof(value));
        return checked(seconds * 1000);
    }

    internal static IPEndPoint ParseEndpoint(string value)
    {
        if (!IPEndPoint.TryParse(value, out IPEndPoint endpoint) || endpoint.Port == 0)
            throw new ArgumentException("Pipeline endpoint must be a numeric IP address and nonzero port.");
        return endpoint;
    }

    // A completed hash manifest is deliberately required: matching just a GGUF
    // header is insufficient, since different quantizations can share that header.
    internal static string CheckpointIdentity(string model, string manifest)
    {
        if (string.IsNullOrWhiteSpace(manifest))
            throw new ArgumentException("Set TS_DSV41_PIPELINE_MANIFEST to a completed download-deepseek1017.py manifest.");
        try { return CheckpointIdentityCore(model, manifest); }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or InvalidOperationException
            or ArgumentOutOfRangeException or OverflowException)
        {
            throw new InvalidDataException("Checkpoint manifest is malformed or incomplete.", ex);
        }
    }

    private static string CheckpointIdentityCore(string model, string manifest)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifest));
        JsonElement root = document.RootElement;
        if (!root.GetProperty("complete").GetBoolean()) throw new InvalidDataException("Checkpoint hashes are not verified.");
        string quant = root.GetProperty("quant").GetString();
        if (string.IsNullOrWhiteSpace(quant) || quant.IndexOfAny(new[] { '/', '\\', '\r', '\n' }) >= 0)
            throw new InvalidDataException("Manifest quant must be a single directory name.");
        string repo = root.GetProperty("repo").GetString(), revision = root.GetProperty("revision").GetString();
        if (string.IsNullOrWhiteSpace(repo) || repo.IndexOfAny(new[] { '\r', '\n' }) >= 0
            || revision?.Length != 40 || !revision.All(Uri.IsHexDigit))
            throw new InvalidDataException("Manifest needs a repository and pinned 40-character revision.");
        double finishedUnix = root.GetProperty("finished_unix").GetDouble();
        if (!double.IsFinite(finishedUnix) || finishedUnix < 0 || finishedUnix > 253402300799)
            throw new InvalidDataException("Invalid manifest verification timestamp.");
        DateTime verifiedUtc = DateTime.UnixEpoch.AddTicks(checked((long)(finishedUnix * TimeSpan.TicksPerSecond)));
        var files = root.GetProperty("files").EnumerateArray()
            .Select(file => file.ValueKind == JsonValueKind.Object
                && file.TryGetProperty("path", out JsonElement path) && path.ValueKind == JsonValueKind.String
                && !string.IsNullOrEmpty(path.GetString()) ? file
                : throw new InvalidDataException("Every manifest record needs a nonempty path."))
            .Where(f => f.GetProperty("path").GetString().StartsWith(quant + "/", StringComparison.Ordinal))
            .OrderBy(f => f.GetProperty("path").GetString(), StringComparer.Ordinal).ToArray();
        if (files.Length == 0) throw new InvalidDataException("Manifest contains no model shards.");
        string parent = Path.GetDirectoryName(Path.GetFullPath(model));
        string fullModel = Path.GetFullPath(model);
        Match modelName = Regex.Match(Path.GetFileName(model), @"^(?<base>.+)-(?<index>\d{5})-of-(?<total>\d{5})\.gguf$",
            RegexOptions.CultureInvariant);
        if (!modelName.Success || !int.TryParse(modelName.Groups["total"].Value, out int shardCount)
            || shardCount < 1 || int.Parse(modelName.Groups["index"].Value) < 1
            || int.Parse(modelName.Groups["index"].Value) > shardCount || files.Length != shardCount)
            throw new InvalidDataException("Manifest must contain the complete numbered GGUF shard set.");
        var expectedNames = Enumerable.Range(1, shardCount).Select(index =>
            $"{modelName.Groups["base"].Value}-{index:D5}-of-{shardCount:D5}.gguf").ToHashSet(StringComparer.Ordinal);
        bool containsModel = false;
        var canonical = new StringBuilder(repo).Append('\n').Append(revision).Append('\n');
        foreach (JsonElement file in files)
        {
            string name = Path.GetFileName(file.GetProperty("path").GetString());
            if (file.GetProperty("path").GetString() != quant + "/" + name || !expectedNames.Remove(name))
                throw new InvalidDataException("Manifest has an unexpected or duplicate model shard: " + name);
            string path = Path.Combine(parent, name);
            string expected = file.GetProperty("sha256").GetString();
            if (!file.GetProperty("verified").GetBoolean() || expected?.Length != 64
                || !expected.All(Uri.IsHexDigit)
                || file.GetProperty("actual_sha256").GetString() != expected
                || !File.Exists(path) || new FileInfo(path).Length != file.GetProperty("size").GetInt64())
                throw new InvalidDataException($"Checkpoint shard lacks a valid hash record or has changed size: {name}");
            // Reject checkpoints modified since full-file verification, without
            // rereading hundreds of GB into page cache before each benchmark.
            if (File.GetLastWriteTimeUtc(path) > verifiedUtc)
                throw new InvalidDataException($"Checkpoint changed after verification: {name}");
            containsModel |= string.Equals(path, fullModel, OperatingSystem.IsWindows()
                ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
            canonical.Append(name).Append(':').Append(file.GetProperty("size").GetInt64())
                .Append(':').Append(expected).Append('\n');
        }
        if (!containsModel) throw new InvalidDataException("Loaded model is absent from the verified manifest.");
        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(canonical.ToString())));
    }

    internal static string ReadHandshakeIdentity(BinaryReader reader)
    {
        if (reader.ReadInt32() != Magic || reader.ReadInt32() != Version)
            throw new InvalidDataException("Pipeline protocol version mismatch.");
        if (reader.Read7BitEncodedInt() != 64)
            throw new InvalidDataException("Pipeline checkpoint identity must be a 64-character SHA-256.");
        Span<byte> bytes = stackalloc byte[64];
        reader.BaseStream.ReadExactly(bytes);
        foreach (byte value in bytes)
            if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f')))
                throw new InvalidDataException("Pipeline checkpoint identity is not a lowercase SHA-256.");
        return Encoding.ASCII.GetString(bytes);
    }

    internal static void WriteArray<T>(Stream stream, ReadOnlySpan<T> values) where T : unmanaged
    {
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Pipeline protocol requires little endian.");
        stream.Write(MemoryMarshal.AsBytes(values));
    }
    internal static void WriteArray<T>(Stream stream, T[] values) where T : unmanaged => WriteArray<T>(stream, values.AsSpan());
    internal static void ReadArray<T>(Stream stream, Span<T> values) where T : unmanaged
    {
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Pipeline protocol requires little endian.");
        stream.ReadExactly(MemoryMarshal.AsBytes(values));
    }

    internal static void WriteForwardFrame(Stream stream, int position, int[] tokens, float[] hidden, float[] pre,
        bool requestLogits = true, bool greedy = false)
    {
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Pipeline protocol requires little endian.");
        if (greedy && tokens.Length != 1) throw new ArgumentException("Greedy pipeline forward requires exactly one token.");
        int tokensBytes = checked(tokens.Length * sizeof(int)), hiddenBytes = checked(hidden.Length * sizeof(float));
        byte[] frame = new byte[checked(12 + tokensBytes + hiddenBytes + pre.Length * sizeof(float))];
        BinaryPrimitives.WriteInt32LittleEndian(frame.AsSpan(0, 4), greedy ? ForwardGreedyOp : requestLogits ? ForwardOp : ForwardNoLogitsOp);
        BinaryPrimitives.WriteInt32LittleEndian(frame.AsSpan(4, 4), position);
        BinaryPrimitives.WriteInt32LittleEndian(frame.AsSpan(8, 4), tokens.Length);
        MemoryMarshal.AsBytes(tokens.AsSpan()).CopyTo(frame.AsSpan(12, tokensBytes));
        MemoryMarshal.AsBytes(hidden.AsSpan()).CopyTo(frame.AsSpan(12 + tokensBytes, hiddenBytes));
        MemoryMarshal.AsBytes(pre.AsSpan()).CopyTo(frame.AsSpan(12 + tokensBytes + hiddenBytes));
        stream.Write(frame);
    }

    internal static void WriteForwardReply(Stream stream, int position, float[] logits)
    {
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Pipeline protocol requires little endian.");
        byte[] frame = new byte[checked(8 + (logits?.Length ?? 0) * sizeof(float))];
        BinaryPrimitives.WriteInt32LittleEndian(frame.AsSpan(0, 4), 0);
        BinaryPrimitives.WriteInt32LittleEndian(frame.AsSpan(4, 4), position);
        if (logits != null) MemoryMarshal.AsBytes(logits.AsSpan()).CopyTo(frame.AsSpan(8));
        stream.Write(frame);
    }

    internal static int ArgmaxFirstFinite(ReadOnlySpan<float> logits)
    {
        if (logits.Length == 0) throw new ArgumentException("Greedy pipeline logits must contain a complete nonempty row.");
        int best = 0;
        for (int index = 0; index < logits.Length; ++index)
        {
            if (!float.IsFinite(logits[index])) throw new InvalidDataException("Greedy pipeline logits contain a nonfinite value.");
            if (logits[index] > logits[best]) best = index;
        }
        return best;
    }

    internal static void WriteGreedyReply(Stream stream, int position, int token)
    {
        Span<byte> frame = stackalloc byte[12];
        BinaryPrimitives.WriteInt32LittleEndian(frame[..4], 0);
        BinaryPrimitives.WriteInt32LittleEndian(frame.Slice(4, 4), position);
        BinaryPrimitives.WriteInt32LittleEndian(frame.Slice(8, 4), token);
        stream.Write(frame);
    }

    internal static void ServeConnection(TcpClient client, IntPtr tail, string identity, int context)
    {
        if (!GgmlDeepSeek4Native.StageInfo(tail, out int first, out int end, out int hidden, out int hc)
            || first != Cut || end != Layers)
            throw new InvalidOperationException("Invalid pipeline tail stage dimensions/range.");
        ServeConnection(client, identity, context, hidden, hc, GgmlDeepSeek4Native.VocabSize(tail),
            GgmlDeepSeek4Native.UBatch(tail), () => GgmlDeepSeek4Native.NPast(tail),
            () => { if (!GgmlDeepSeek4Native.ResetChecked(tail)) throw new InvalidOperationException("Tail reset failed."); },
            (tokens, hiddenRows, pre, logits) =>
            {
                if (!GgmlDeepSeek4Native.StageForward(tail, tokens, hiddenRows, pre, null, null, logits))
                    throw new InvalidOperationException("Pipeline tail forward failed (see stderr).");
            });
    }

    // Shared production wire loop with injected stage operations for bounded
    // fragmented TCP tests; the real worker supplies the unchanged native ABI.
    internal static void ServeConnection(TcpClient client, string identity, int context, int hidden, int hc,
        int vocab, int ubatch, Func<int> position, Action reset, Action<int[], float[], float[], float[]> forward)
    {
        client.NoDelay = true;
        client.ReceiveTimeout = client.SendTimeout = TimeoutMilliseconds();
        using NetworkStream stream = client.GetStream();
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        bool coalesce = Environment.GetEnvironmentVariable("TS_DSV41_PIPELINE_PREFILL_OVERLAP") == "1";
        try
        {
            string peerIdentity = ReadHandshakeIdentity(reader);
            if (peerIdentity != identity) throw new InvalidDataException("Checkpoint identity mismatch.");
            if (reader.ReadInt32() != hidden || reader.ReadInt32() != hc
                || reader.ReadInt32() != vocab || reader.ReadInt32() != context)
                throw new InvalidDataException("Pipeline dimensions/context mismatch.");
            int peerWidth = reader.ReadInt32();
            if (peerWidth <= 0) throw new InvalidDataException("Invalid peer microbatch width.");
            int width = Math.Min(peerWidth, ubatch);
            reset();
            writer.Write(0); writer.Write(width); writer.Flush();
            while (true)
            {
                int op = reader.ReadInt32();
                if (op == ResetOp)
                {
                    reset();
                    writer.Write(0); writer.Flush(); continue;
                }
                bool requestLogits = op == ForwardOp;
                bool greedy = op == ForwardGreedyOp;
                if (!requestLogits && !greedy && op != ForwardNoLogitsOp) throw new InvalidDataException("Unknown pipeline operation.");
                int expected = reader.ReadInt32(), count = reader.ReadInt32();
                if (count <= 0 || count > width || (greedy && count != 1) || expected != position()
                    || expected < 0 || (long)expected + count > context)
                    throw new InvalidDataException("Pipeline cache position or batch bounds mismatch.");
                int[] tokens = new int[count];
                float[] hiddenRows = new float[checked(count * hidden * hc)], pre = new float[checked(count * hc)];
                ReadArray(stream, tokens.AsSpan()); ReadArray(stream, hiddenRows.AsSpan()); ReadArray(stream, pre.AsSpan());
                foreach (int token in tokens)
                    if (token < 0 || token >= vocab) throw new InvalidDataException("Pipeline input token ID is outside the vocabulary.");
                float[] logits = requestLogits || greedy ? new float[vocab] : null;
                var tailTimer = Stopwatch.StartNew();
                forward(tokens, hiddenRows, pre, logits);
                double tailMs = tailTimer.Elapsed.TotalMilliseconds;
                if (position() != expected + count)
                    throw new InvalidDataException("Pipeline tail advanced to an unexpected position.");
                if (greedy) WriteGreedyReply(stream, expected + count, ArgmaxFirstFinite(logits));
                else if (coalesce) WriteForwardReply(stream, expected + count, logits);
                else
                {
                    writer.Write(0); writer.Write(expected + count); writer.Flush();
                    if (requestLogits) WriteArray(stream, logits);
                }
                if (Environment.GetEnvironmentVariable("TS_DSV41_PIPELINE_PERF") == "1")
                    Console.Error.WriteLine(FormattableString.Invariant(
                        $"[dsv41-pipeline-tail-perf] position={expected} tokens={count} tail_ms={tailMs:F4} return_logits={(requestLogits ? 1 : 0)} greedy_reply={(greedy ? 1 : 0)}"));
            }
        }
        catch (EndOfStreamException) { }
        catch (IOException ex) { Console.Error.WriteLine("[dsv41-pipeline] connection ended: " + ex.Message); }
        catch (Exception ex)
        {
            Console.Error.WriteLine("[dsv41-pipeline] peer refused: " + ex.Message);
            try { writer.Write(1); writer.Write(ex.Message); writer.Flush(); } catch (IOException) { }
        }
        finally
        {
            try { reset(); }
            catch (Exception error) { Console.Error.WriteLine("[dsv41-pipeline] tail reset failed: " + error.Message); }
        }
    }
}

/// <summary>Tail worker for the experimental two-node exact text pipeline.</summary>
public static class DeepSeek41PipelineWorker
{
    public static void Run(string model, string endpoint, string manifest, int gpuCount = 2,
        int context = 4096, int ubatch = 256, int threads = 16)
    {
        IPEndPoint address = DeepSeek41PipelineExecutor.ParseEndpoint(endpoint);
        if (!IPAddress.IsLoopback(address.Address))
            throw new ArgumentException("Bind the pipeline worker to loopback and use an SSH tunnel.");
        string identity = DeepSeek41PipelineExecutor.CheckpointIdentity(model, manifest);
        IntPtr tail = GgmlDeepSeek4Native.LoadStage(model, gpuCount, context, ubatch, threads, "CUDA",
            DeepSeek41PipelineExecutor.Cut, DeepSeek41PipelineExecutor.Layers);
        if (tail == IntPtr.Zero) throw new InvalidOperationException("DeepSeek pipeline tail load failed (see stderr).");
        var listener = new TcpListener(address);
        try
        {
            listener.Start(1);
            Console.Error.WriteLine($"[dsv41-pipeline] READY tail [20,40) {endpoint}, identity={identity}");
            while (true)
            {
                using TcpClient client = listener.AcceptTcpClient();
                DeepSeek41PipelineExecutor.ServeConnection(client, tail, identity, context);
            }
        }
        finally { listener.Stop(); GgmlDeepSeek4Native.Free(tail); }
    }
}
