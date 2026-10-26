// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>Exact compact-token production wire loop over fragmented TCP; no native/model/GPU calls.</summary>
public sealed class DeepSeek41PipelineGreedyTests
{
    [Fact]
    public void FirstMaximumIsExactAndEveryVocabularyCoordinateMustBeFinite()
    {
        Assert.Equal(1, DeepSeek41PipelineExecutor.ArgmaxFirstFinite([-7f, -2f, -2f, -5f]));
        Assert.Equal(0, DeepSeek41PipelineExecutor.ArgmaxFirstFinite([0f, -0f]));
        var full = Enumerable.Repeat(-float.MaxValue, 129280).ToArray();
        full[^1] = float.MaxValue;
        Assert.Equal(129279, DeepSeek41PipelineExecutor.ArgmaxFirstFinite(full));
        foreach (float value in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            full[129278] = value;
            Assert.Throws<InvalidDataException>(() => DeepSeek41PipelineExecutor.ArgmaxFirstFinite(full));
        }
        Assert.Throws<ArgumentException>(() => DeepSeek41PipelineExecutor.ArgmaxFirstFinite([]));
    }

    [Fact]
    public void CompactReplyIsOneTwelveByteWriteAndGreedyFramesRequireOneToken()
    {
        using var stream = new CountingStream();
        DeepSeek41PipelineExecutor.WriteGreedyReply(stream, 43, 129279);
        Assert.Equal(1, stream.Writes); Assert.Equal(12, stream.Length);
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(stream.ToArray()));
        Assert.Equal(43, BinaryPrimitives.ReadInt32LittleEndian(stream.ToArray().AsSpan(4)));
        Assert.Equal(129279, BinaryPrimitives.ReadInt32LittleEndian(stream.ToArray().AsSpan(8)));
        Assert.Throws<ArgumentException>(() => DeepSeek41PipelineExecutor.WriteForwardFrame(stream, 0, [1, 2], [], [], greedy: true));
        Assert.Equal(1, stream.Writes);
    }

    [Fact]
    public async Task FullFiniteTailRowReturnsFirstTieUsingOnlyTwelveReplyBytesAndAtomicPosition()
    {
        using var entered = new ManualResetEventSlim(); using var release = new ManualResetEventSlim();
        var tail = new Tail { Before = () => { entered.Set(); Wait(release); } };
        var head = new Head(); using var peer = new Peer(tail.Serve);
        using var executor = Create(peer, head);
        long beforeRead = peer.Stream!.BytesRead, beforeWrite = peer.Stream.BytesWritten;
        Task<int> work = Task.Run(() => executor.ForwardGreedy(3));
        Wait(entered); Assert.Equal(0, executor.NPast); Assert.Equal(1, head.Position);
        release.Set(); Assert.Equal(2, await work.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal(1, executor.NPast); Assert.False(executor.HasPendingReplyReader);
        Assert.Equal(12, peer.Stream.BytesRead - beforeRead);
        Assert.Equal(12 + 4 + 6 * 4 + 2 * 4, peer.Stream.BytesWritten - beforeWrite);
        Assert.Equal(7, tail.LastLogitsLength); Assert.Equal(1, tail.Calls);
    }

    [Fact]
    public void RegularFullLogitsGreedyAndResetShareExactCachePositions()
    {
        var tail = new Tail(); var head = new Head(); using var peer = new Peer(tail.Serve);
        using var executor = Create(peer, head);
        var logits = new float[7]; executor.Forward([1], logits);
        Assert.Equal(Tail.Row(), logits); Assert.Equal(1, executor.NPast);
        Assert.Equal(2, executor.ForwardGreedy(2)); Assert.Equal(2, executor.NPast);
        executor.Reset(); Assert.Equal(0, executor.NPast); Assert.Equal(0, head.Position);
        Assert.Equal(2, executor.ForwardGreedy(3)); Assert.Equal(1, executor.NPast);
        Assert.Equal(2, head.Resets);
    }

    [Fact]
    public void ModelAdapterDeclinesWithoutMutationWhenFlagOffOrPipelineAbsent()
    {
        var head = new Head(); using var peer = new Peer(new Tail().Serve);
        using var executor = Create(peer, head, enabled: false);
        IGreedyForwardModel model = Model(executor);
        long written = peer.Stream!.BytesWritten;
        Assert.False(model.TryForwardGreedyToken(3, out int token)); Assert.Equal(0, token);
        Assert.Equal(0, head.Calls); Assert.Equal(0, executor.NPast); Assert.Equal(written, peer.Stream.BytesWritten);
        Assert.False(Model(null).TryForwardGreedyToken(3, out _));
        Assert.Throws<InvalidOperationException>(() => executor.ForwardGreedy(3));
        Assert.Equal(0, head.Calls);
    }

    [Fact]
    public void EnabledModelAdapterUsesSeparateExactTokenResult()
    {
        var head = new Head(); using var peer = new Peer(new Tail().Serve); using var executor = Create(peer, head);
        Assert.True(Model(executor).TryForwardGreedyToken(3, out int token));
        Assert.Equal(2, token); Assert.Equal(1, executor.NPast); Assert.Equal(1, head.Calls);
    }

    [Theory]
    [InlineData(-1)] [InlineData(7)] [InlineData(int.MaxValue)]
    public void InvalidInputTokenIsRejectedWithoutHeadOrPeerMutation(int token)
    {
        var head = new Head(); using var peer = new Peer(new Tail().Serve); using var executor = Create(peer, head);
        long written = peer.Stream!.BytesWritten;
        Assert.Throws<ArgumentOutOfRangeException>(() => executor.ForwardGreedy(token));
        Assert.Equal(0, executor.NPast); Assert.Equal(0, head.Calls); Assert.Equal(written, peer.Stream.BytesWritten);
        Assert.Equal(2, executor.ForwardGreedy(3));
    }

    [Fact]
    public void MaximumContextAllowsLastTokenThenRejectsBeforeComputing()
    {
        var head = new Head(); var tail = new Tail { Context = 1 }; using var peer = new Peer(tail.Serve);
        using var executor = Create(peer, head, context: 1);
        Assert.Equal(2, executor.ForwardGreedy(3)); long written = peer.Stream!.BytesWritten;
        Assert.Throws<ArgumentException>(() => executor.ForwardGreedy(3));
        Assert.Equal(1, executor.NPast); Assert.Equal(1, head.Calls); Assert.Equal(written, peer.Stream.BytesWritten);
    }

    [Theory]
    [InlineData(0, 2, 12)] [InlineData(2, 2, 12)]
    [InlineData(1, -1, 12)] [InlineData(1, 7, 12)]
    [InlineData(1, 2, 4)] [InlineData(1, 2, 8)] [InlineData(1, 2, 11)]
    public async Task MalformedOrTruncatedCompactReplyPoisonsWithoutCommit(int position, int token, int length)
    {
        var head = new Head(); using var peer = new Peer(client =>
        {
            using var stream = client.GetStream(); Handshake(stream); ReceiveGreedy(stream);
            using var reply = new MemoryStream(); DeepSeek41PipelineExecutor.WriteGreedyReply(reply, position, token);
            Fragment(stream, reply.ToArray()[..length]);
        });
        using var executor = Create(peer, head);
        Exception? error = await Record.ExceptionAsync(() => Task.Run(() => executor.ForwardGreedy(3)));
        if (length == 12) Assert.IsType<InvalidDataException>(error);
        else Assert.IsType<EndOfStreamException>(error);
        Assert.Equal(0, executor.NPast); Assert.False(executor.HasPendingReplyReader);
        Assert.Throws<InvalidOperationException>(() => executor.ForwardGreedy(3));
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(5));
    }

    [Theory]
    [InlineData(2)] [InlineData(1)]
    public void InvalidStatusAndOversizedErrorAreBoundedAndPoisonState(int status)
    {
        var head = new Head(); using var peer = new Peer(client =>
        {
            using var stream = client.GetStream(); Handshake(stream); ReceiveGreedy(stream);
            using var reply = new MemoryStream(); using (var writer = new BinaryWriter(reply, Encoding.UTF8, true))
            { writer.Write(status); if (status == 1) writer.Write7BitEncodedInt(4097); }
            Fragment(stream, reply.ToArray());
        });
        using var executor = Create(peer, head);
        Assert.Throws<InvalidDataException>(() => executor.ForwardGreedy(3)); Assert.Equal(0, executor.NPast);
        Assert.Throws<InvalidOperationException>(() => executor.ForwardGreedy(3));
    }

    [Fact]
    public void ResetAfterFailedReplyReconnectsAndResetsBothStagesBeforeReuse()
    {
        var head = new Head(); using var peer = new Peer(client =>
        {
            using var stream = client.GetStream(); Handshake(stream); ReceiveGreedy(stream);
            DeepSeek41PipelineExecutor.WriteGreedyReply(stream, 99, 2);
        }, new Tail().Serve);
        using var executor = Create(peer, head);
        Assert.Throws<InvalidDataException>(() => executor.ForwardGreedy(3));
        Assert.Equal(0, executor.NPast); Assert.Equal(1, head.Position);
        executor.Reset(); Assert.Equal(0, head.Position); Assert.Equal(0, executor.NPast);
        Assert.Equal(2, executor.ForwardGreedy(3)); Assert.Equal(1, executor.NPast); Assert.Equal(2, head.Resets);
    }

    [Theory]
    [InlineData(float.NaN)] [InlineData(float.PositiveInfinity)] [InlineData(float.NegativeInfinity)]
    public async Task ProductionTailRejectsNonfiniteCoordinateEvenBelowTheWinner(float value)
    {
        var tail = new Tail { Nonfinite = value }; var head = new Head(); using var peer = new Peer(tail.Serve);
        using var executor = Create(peer, head);
        Assert.Throws<InvalidDataException>(() => executor.ForwardGreedy(3));
        Assert.Equal(0, executor.NPast); Assert.Equal(1, tail.Calls);
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(0, tail.Position); Assert.True(tail.Resets >= 2);
    }

    [Theory]
    [InlineData(2, 0, 3)] [InlineData(0, 0, 3)] [InlineData(int.MaxValue, 0, 3)]
    [InlineData(1, -1, 3)] [InlineData(1, 1, 3)] [InlineData(1, 0, 7)]
    public async Task ProductionTailRejectsGreedyCountPositionAndTokenBeforeStageCompute(int count, int position, int token)
    {
        var tail = new Tail(); using var peer = new Peer(tail.Serve); using var stream = peer.Connect();
        ClientHandshake(stream, 64);
        using var frame = new MemoryStream(); using (var writer = new BinaryWriter(frame, Encoding.UTF8, true))
        { writer.Write(4); writer.Write(position); writer.Write(count); }
        if (count == 1 && position == 0)
        {
            DeepSeek41PipelineExecutor.WriteArray(frame, new[] { token });
            DeepSeek41PipelineExecutor.WriteArray(frame, new float[6]); DeepSeek41PipelineExecutor.WriteArray(frame, new float[2]);
        }
        Fragment(stream, frame.ToArray()); using var reader = new BinaryReader(stream, Encoding.UTF8, true);
        Assert.Equal(1, reader.ReadInt32()); Assert.InRange(reader.ReadString().Length, 1, 4096);
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(5)); Assert.Equal(0, tail.Calls); Assert.Equal(0, tail.Position);
    }

    [Fact]
    public async Task DisposeInterruptsBlockedGreedyBeforeFreeAndIsIdempotent()
    {
        using var waiting = new ManualResetEventSlim(); var head = new Head();
        using var peer = new Peer(client =>
        {
            using var stream = client.GetStream(); Handshake(stream); ReceiveGreedy(stream); waiting.Set();
            try { Assert.Equal(-1, stream.ReadByte()); } catch (IOException) { }
        });
        var executor = Create(peer, head); Task<int> work = Task.Run(() => executor.ForwardGreedy(3)); Wait(waiting);
        await Task.Run(executor.Dispose).WaitAsync(TimeSpan.FromSeconds(3));
        Assert.NotNull(await Record.ExceptionAsync(() => work)); Assert.Equal(0, executor.NPast);
        Assert.Equal(1, head.Releases); executor.Dispose(); Assert.Equal(1, head.Releases);
        Assert.Throws<ObjectDisposedException>(() => executor.ForwardGreedy(3));
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(5));
    }

    [Fact]
    public void IncorrectNativeHeadAdvanceFailsBeforeSendingAForwardFrame()
    {
        var head = new Head { ExtraAdvance = 1 };
        using var peer = new Peer(client =>
        {
            using var stream = client.GetStream(); Handshake(stream);
            try { Assert.Equal(-1, stream.ReadByte()); } catch (IOException) { }
        });
        using var executor = Create(peer, head);
        long written = peer.Stream!.BytesWritten;
        Assert.Throws<InvalidDataException>(() => executor.ForwardGreedy(3));
        Assert.Equal(written, peer.Stream.BytesWritten); Assert.Equal(0, executor.NPast);
        Assert.Throws<InvalidOperationException>(() => executor.ForwardGreedy(3));
    }

    private static DeepSeek4Model Model(DeepSeek41PipelineExecutor? executor)
    {
        var model = (DeepSeek4Model)RuntimeHelpers.GetUninitializedObject(typeof(MockModel));
        typeof(DeepSeek4Model).GetField("_sync", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, new object());
        typeof(DeepSeek4Model).GetField("_pipelineExec", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, executor);
        return model;
    }
    private sealed class MockModel : DeepSeek4Model
    {
        private MockModel() : base("unused", BackendType.Cpu) { }
        internal override bool TryGetBackendFailure(out string detail) { detail = null!; return false; }
    }
    private static DeepSeek41PipelineExecutor Create(Peer peer, Head head, bool enabled = true, int context = 64)
        => new(context, 2, 3, 2, 7, head.Forward, head.Reset, head.Release, peer.Connect, false, greedyTransport: enabled);
    private static void Wait(ManualResetEventSlim signal) => Assert.True(signal.Wait(TimeSpan.FromSeconds(5)), "Peer synchronization timeout");
    private static void Fragment(Stream stream, byte[] bytes)
    { for (int offset = 0; offset < bytes.Length; offset += 3) stream.Write(bytes, offset, Math.Min(3, bytes.Length - offset)); }
    private static void Handshake(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, true);
        Assert.Equal(new string('a', 64), DeepSeek41PipelineExecutor.ReadHandshakeIdentity(reader));
        Assert.Equal(3, reader.ReadInt32()); Assert.Equal(2, reader.ReadInt32()); Assert.Equal(7, reader.ReadInt32());
        Assert.Equal(64, reader.ReadInt32()); Assert.Equal(2, reader.ReadInt32());
        using var reply = new MemoryStream(); using (var writer = new BinaryWriter(reply, Encoding.UTF8, true)) { writer.Write(0); writer.Write(2); }
        Fragment(stream, reply.ToArray());
    }
    private static void ClientHandshake(Stream stream, int context)
    {
        using var frame = new MemoryStream(); using (var writer = new BinaryWriter(frame, Encoding.UTF8, true))
        { writer.Write(DeepSeek41PipelineExecutor.Magic); writer.Write(3); writer.Write(new string('a', 64)); writer.Write(3); writer.Write(2); writer.Write(7); writer.Write(context); writer.Write(2); }
        Fragment(stream, frame.ToArray()); using var reader = new BinaryReader(stream, Encoding.UTF8, true);
        Assert.Equal(0, reader.ReadInt32()); Assert.Equal(2, reader.ReadInt32());
    }
    private static void ReceiveGreedy(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, true);
        Assert.Equal(4, reader.ReadInt32()); Assert.Equal(0, reader.ReadInt32()); Assert.Equal(1, reader.ReadInt32());
        var tokens = new int[1]; var hidden = new float[6]; var pre = new float[2];
        DeepSeek41PipelineExecutor.ReadArray(stream, tokens.AsSpan()); DeepSeek41PipelineExecutor.ReadArray(stream, hidden.AsSpan()); DeepSeek41PipelineExecutor.ReadArray(stream, pre.AsSpan());
        Assert.Equal(3, tokens[0]); Assert.All(hidden, value => Assert.Equal(1.25f, value)); Assert.All(pre, value => Assert.Equal(-0.5f, value));
    }
    private sealed class Head
    {
        public int Position, Calls, Resets, Releases, ExtraAdvance;
        public int Forward(int[] tokens, float[] hidden, float[] pre)
        { ++Calls; Array.Fill(hidden, 1.25f); Array.Fill(pre, -0.5f); return Position += tokens.Length + ExtraAdvance; }
        public void Reset() { Position = 0; ++Resets; }
        public void Release() => ++Releases;
    }
    private sealed class Tail
    {
        public int Context = 64, Position, Calls, Resets, LastLogitsLength;
        public float? Nonfinite; public Action? Before;
        public static float[] Row() => [-8f, -2f, 3.5f, 0f, 3.5f, -3f, -9f];
        public void Serve(TcpClient client)
            => DeepSeek41PipelineExecutor.ServeConnection(client, new string('a', 64), Context, 3, 2, 7, 2,
                () => Position, () => { Position = 0; ++Resets; }, (tokens, hidden, pre, logits) =>
                {
                    ++Calls; LastLogitsLength = logits?.Length ?? 0; Before?.Invoke();
                    Assert.Equal(6 * tokens.Length, hidden.Length); Assert.Equal(2 * tokens.Length, pre.Length);
                    if (logits != null) { Row().CopyTo(logits, 0); if (Nonfinite.HasValue) logits[^1] = Nonfinite.Value; }
                    Position += tokens.Length;
                });
    }
    private sealed class Peer : IDisposable
    {
        private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
        public Task Completion { get; } public MeteredStream? Stream;
        public Peer(params Action<TcpClient>[] scripts)
        {
            _listener.Start(); Completion = Task.Run(() =>
            { foreach (var script in scripts) { using var client = _listener.AcceptTcpClient(); client.NoDelay = true; client.ReceiveTimeout = client.SendTimeout = 5000; script(client); } });
        }
        public Stream Connect()
        {
            var socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp) { NoDelay = true };
            socket.Connect((IPEndPoint)_listener.LocalEndpoint); return Stream = new MeteredStream(new NetworkStream(socket, true));
        }
        public void Dispose() => _listener.Stop();
    }
    private sealed class MeteredStream(Stream inner) : Stream
    {
        public long BytesRead, BytesWritten;
        public override bool CanRead => inner.CanRead; public override bool CanSeek => false; public override bool CanWrite => inner.CanWrite;
        public override bool CanTimeout => inner.CanTimeout; public override int ReadTimeout { get => inner.ReadTimeout; set => inner.ReadTimeout = value; }
        public override int WriteTimeout { get => inner.WriteTimeout; set => inner.WriteTimeout = value; }
        public override long Length => throw new NotSupportedException(); public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() => inner.Flush();
        public override int Read(byte[] buffer, int offset, int count) { int n = inner.Read(buffer, offset, Math.Min(count, 2)); BytesRead += n; return n; }
        public override int Read(Span<byte> buffer) { int n = inner.Read(buffer[..Math.Min(buffer.Length, 2)]); BytesRead += n; return n; }
        public override void Write(byte[] buffer, int offset, int count) { Fragment(inner, buffer.AsSpan(offset, count).ToArray()); BytesWritten += count; }
        public override void Write(ReadOnlySpan<byte> buffer) { Fragment(inner, buffer.ToArray()); BytesWritten += buffer.Length; }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException(); public override void SetLength(long value) => throw new NotSupportedException();
        protected override void Dispose(bool disposing) { if (disposing) inner.Dispose(); base.Dispose(disposing); }
    }
    private sealed class CountingStream : MemoryStream
    { public int Writes; public override void Write(ReadOnlySpan<byte> buffer) { ++Writes; base.Write(buffer); } }
}
