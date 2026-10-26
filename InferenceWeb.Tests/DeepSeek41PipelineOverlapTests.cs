// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Production pipeline orchestration against fragmented real loopback TCP; no native/GPU calls.</summary>
public sealed class DeepSeek41PipelineOverlapTests
{
    [Fact]
    public async Task PrefillOverlapsTwoChunksBoundsBackpressureAndCommitsOnlyAfterEveryReply()
    {
        using var twoReceived = new ManualResetEventSlim();
        using var releaseFirst = new ManualResetEventSlim();
        using var beforeLast = new ManualResetEventSlim();
        using var releaseLast = new ManualResetEventSlim();
        var head = new Head();
        using var peer = new Peer(session =>
        {
            var first = session.Receive(); var second = session.Receive();
            twoReceived.Set(); Wait(releaseFirst);
            session.Reply(first); var third = session.Receive();
            session.Reply(second); var fourth = session.Receive();
            session.Reply(third); beforeLast.Set(); Wait(releaseLast); session.Reply(fourth);
            Assert.Equal(new[] { 2, 2, 2, 1 }, new[] { first.Tokens.Length, second.Tokens.Length, third.Tokens.Length, fourth.Tokens.Length });
            Assert.Equal(Enumerable.Range(1, 7), first.Tokens.Concat(second.Tokens).Concat(third.Tokens).Concat(fourth.Tokens));
        });
        using var executor = Create(peer, head, overlap: true);
        head.Joined = () => !executor.HasPendingReplyReader;
        int caller = 0;
        float[] logits = new float[7];
        Task forward = Task.Run(() => { caller = Environment.CurrentManagedThreadId; executor.Forward(Enumerable.Range(1, 7).ToArray(), logits); });
        Wait(twoReceived); // peer refuses its first reply until the second frame arrives: causal proof of overlap
        Assert.True(executor.HasPendingReplyReader);
        Assert.Equal(0, executor.NPast);
        Assert.False(head.ThirdStarted.Wait(150)); // two unreplied chunks exhaust the bounded producer window
        Assert.Equal(2, head.Calls);
        releaseFirst.Set(); Wait(beforeLast);
        Assert.Equal(0, executor.NPast);
        releaseLast.Set();
        await forward.WaitAsync(TimeSpan.FromSeconds(10));
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(7, executor.NPast);
        Assert.Equal(Head.ExpectedLogits(7), logits);
        Assert.False(executor.HasPendingReplyReader);
        Assert.All(head.Threads, thread => Assert.Equal(caller, thread));
    }

    [Fact]
    public async Task SingleChunkDecodeKeepsTheCallerThreadAndNeedsNoReplyThread()
    {
        var head = new Head();
        using var peer = new Peer(session => { var chunk = session.Receive(); Assert.True(chunk.RequestLogits); session.Reply(chunk); });
        using var executor = Create(peer, head, overlap: true);
        head.Before = _ => Assert.False(executor.HasPendingReplyReader);
        int caller = Environment.CurrentManagedThreadId;
        var logits = new float[7];
        executor.Forward([19], logits);
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(1, executor.NPast);
        Assert.Equal(Head.ExpectedLogits(1), logits);
        Assert.Equal(caller, Assert.Single(head.Threads));
    }

    [Fact]
    public async Task IntermediatePrefillChunksAcknowledgeOnlyAndOnlyFinalChunkReturnsVocabulary()
    {
        var head = new Head();
        var seen = new List<Chunk>();
        long replyBytes = 0;
        using var peer = new Peer(session =>
        {
            for (int index = 0; index < 9; ++index)
            {
                Chunk chunk = session.Receive(); seen.Add(chunk);
                using var bytes = new MemoryStream();
                DeepSeek41PipelineExecutor.WriteForwardReply(bytes, chunk.End,
                    chunk.RequestLogits ? Head.ExpectedLogits(chunk.End) : null);
                replyBytes += bytes.Length;
                session.WriteFragmented(bytes.ToArray());
            }
        });
        using var executor = Create(peer, head, overlap: true);
        float[] logits = new float[7];
        await Task.Run(() => executor.Forward(Enumerable.Range(1, 17).ToArray(), logits)).WaitAsync(TimeSpan.FromSeconds(10));
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(Enumerable.Range(0, 9).Select(i => i * 2), seen.Select(chunk => chunk.Position));
        Assert.All(seen.Take(8), chunk => Assert.False(chunk.RequestLogits));
        Assert.True(seen[^1].RequestLogits);
        Assert.Equal(9 * 8 + 7 * sizeof(float), replyBytes);
        Assert.Equal(9, head.Calls);
        Assert.Equal(17, executor.NPast);
        Assert.Equal(Head.ExpectedLogits(17), logits);
    }

    [Fact]
    public async Task InterruptedIntermediateAcknowledgementPoisonsStateAndJoinsWithoutAFalseCommit()
    {
        var head = new Head();
        using var peer = new Peer(session =>
        {
            Chunk first = session.Receive(), middle = session.Receive();
            Assert.False(first.RequestLogits); Assert.False(middle.RequestLogits);
            session.Reply(first);
            Assert.True(session.Receive().RequestLogits); // finish accepting the bounded producer's final frame
            using var bytes = new MemoryStream();
            DeepSeek41PipelineExecutor.WriteForwardReply(bytes, middle.End, null);
            session.WriteFragmented(bytes.ToArray()[..^1]);
        });
        using var executor = Create(peer, head, overlap: true);
        var logits = Enumerable.Repeat(-123f, 7).ToArray();
        var error = await Record.ExceptionAsync(() => Task.Run(() => executor.Forward([1, 2, 3, 4, 5, 6], logits)));
        Assert.IsAssignableFrom<IOException>(error);
        Assert.False(executor.HasPendingReplyReader);
        Assert.Equal(0, executor.NPast);
        Assert.All(logits, value => Assert.Equal(-123, value));
        Assert.Throws<InvalidOperationException>(() => executor.Forward([7], logits));
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public async Task WrongPeerPositionDoesNotCommitAndResetReconnectsAfterTheReaderHasJoined()
    {
        var head = new Head();
        using var peer = new Peer(
            session =>
            {
                session.Reply(session.Receive()); // establish an already committed one-token history
                var first = session.Receive(); var second = session.Receive();
                session.Reply(first); session.Reply(second, 999);
            },
            session => session.Reply(session.Receive()));
        using var executor = Create(peer, head, overlap: true);
        head.Joined = () => !executor.HasPendingReplyReader;
        executor.Forward([1], new float[7]);
        float[] logits = Enumerable.Repeat(-123f, 7).ToArray();
        var error = await Record.ExceptionAsync(() => Task.Run(() => executor.Forward([2, 3, 4, 5], logits)));
        Assert.IsType<InvalidDataException>(error);
        Assert.Equal(1, executor.NPast);
        Assert.All(logits, value => Assert.Equal(-123, value));
        Assert.False(executor.HasPendingReplyReader);
        Assert.Throws<InvalidOperationException>(() => executor.Forward([6], logits));
        executor.Reset();
        Assert.Equal(0, executor.NPast);
        executor.Forward([6], logits);
        Assert.Equal(1, executor.NPast);
        Assert.Equal(2, head.Resets);
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public async Task HeadFailureCancelsAndJoinsTheBlockedReaderBeforeResourceRelease()
    {
        var head = new Head { Before = call => { if (call == 2) throw new InvalidOperationException("injected head failure"); } };
        using var peer = new Peer(session => { session.Receive(); session.ExpectClosed(); });
        var executor = Create(peer, head, overlap: true);
        head.Joined = () => !executor.HasPendingReplyReader;
        var error = await Record.ExceptionAsync(() => Task.Run(() => executor.Forward([1, 2, 3, 4], new float[7])));
        Assert.IsType<InvalidOperationException>(error);
        Assert.Contains("injected head failure", error.Message);
        Assert.Equal(0, executor.NPast);
        Assert.False(executor.HasPendingReplyReader);
        executor.Dispose();
        Assert.Equal(1, head.Releases);
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public async Task TruncatedFullVocabularyReplyFailsWithoutPublishingPartialState()
    {
        var head = new Head();
        using var peer = new Peer(session =>
        {
            var first = session.Receive(); var second = session.Receive();
            session.Reply(first);
            using var bytes = new MemoryStream();
            DeepSeek41PipelineExecutor.WriteForwardReply(bytes, second.End, Head.ExpectedLogits(second.End));
            session.WriteFragmented(bytes.ToArray()[..^1]);
        });
        using var executor = Create(peer, head, overlap: true);
        var logits = Enumerable.Repeat(-321f, 7).ToArray();
        var error = await Record.ExceptionAsync(() => Task.Run(() => executor.Forward([1, 2, 3, 4], logits)));
        Assert.IsType<EndOfStreamException>(error);
        Assert.Equal(0, executor.NPast);
        Assert.All(logits, value => Assert.Equal(-321, value));
        Assert.False(executor.HasPendingReplyReader);
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public async Task DisposeInterruptsBlockedPrefillAndJoinsBeforeFreeingTheStage()
    {
        using var waiting = new ManualResetEventSlim();
        var head = new Head();
        using var peer = new Peer(session => { session.Receive(); session.Receive(); waiting.Set(); session.ExpectClosed(); });
        var executor = Create(peer, head, overlap: true);
        head.Joined = () => !executor.HasPendingReplyReader;
        Task forward = Task.Run(() => executor.Forward([1, 2, 3, 4, 5, 6], new float[7]));
        Wait(waiting);
        await Task.Run(executor.Dispose).WaitAsync(TimeSpan.FromSeconds(3));
        Assert.NotNull(await Record.ExceptionAsync(() => forward));
        Assert.False(executor.HasPendingReplyReader);
        Assert.Equal(0, executor.NPast);
        Assert.Equal(1, head.Releases);
        Assert.Throws<ObjectDisposedException>(() => executor.Reset());
        Assert.Throws<ObjectDisposedException>(() => executor.Forward([1], new float[7]));
        executor.Dispose();
        Assert.Equal(1, head.Releases);
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public async Task ResetWaitsForAllRepliesAndReaderJoinBeforeResettingEitherStage()
    {
        using var pending = new ManualResetEventSlim();
        using var finish = new ManualResetEventSlim();
        var head = new Head();
        using var peer = new Peer(session =>
        {
            var first = session.Receive(); var second = session.Receive();
            session.Reply(first); pending.Set(); Wait(finish); session.Reply(second);
            Assert.Equal(2, session.Reader.ReadInt32());
            session.WriteFragmented(BitConverter.GetBytes(0));
        });
        using var executor = Create(peer, head, overlap: true);
        head.Joined = () => !executor.HasPendingReplyReader;
        Task forward = Task.Run(() => executor.Forward([1, 2, 3, 4], new float[7]));
        Wait(pending);
        Task reset = Task.Run(executor.Reset);
        Assert.NotSame(reset, await Task.WhenAny(reset, Task.Delay(100)));
        Assert.Equal(1, head.Resets);
        finish.Set();
        await Task.WhenAll(forward, reset).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(0, executor.NPast);
        Assert.Equal(2, head.Resets);
        Assert.False(executor.HasPendingReplyReader);
        await peer.Completion.WaitAsync(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public void CoalescedFramesPreserveLegacyWireBytesAndEveryVocabularyFloatBit()
    {
        int[] tokens = [17, 19];
        float[] hidden = [0, -0f, BitConverter.Int32BitsToSingle(unchecked((int)0x7fc00001)), 1.25f];
        float[] pre = [float.NegativeInfinity, -3.5f];
        using var legacy = new MemoryStream();
        using (var writer = new BinaryWriter(legacy, Encoding.UTF8, leaveOpen: true))
        { writer.Write(1); writer.Write(12); writer.Write(tokens.Length); writer.Flush(); }
        DeepSeek41PipelineExecutor.WriteArray(legacy, tokens);
        DeepSeek41PipelineExecutor.WriteArray(legacy, hidden);
        DeepSeek41PipelineExecutor.WriteArray(legacy, pre);
        using var frame = new CountingStream();
        DeepSeek41PipelineExecutor.WriteForwardFrame(frame, 12, tokens, hidden, pre);
        Assert.Equal(1, frame.Writes);
        Assert.Equal(legacy.ToArray(), frame.ToArray());
        float[] logits = Enumerable.Range(0, 129280).Select(i => BitConverter.Int32BitsToSingle(i * 7919)).ToArray();
        using var reply = new CountingStream();
        DeepSeek41PipelineExecutor.WriteForwardReply(reply, 14, logits);
        Assert.Equal(1, reply.Writes);
        Assert.Equal(8 + 129280 * sizeof(float), reply.Length);
        Assert.Equal(MemoryMarshal.AsBytes(logits.AsSpan()).ToArray(), reply.ToArray()[8..]);
    }

    private static DeepSeek41PipelineExecutor Create(Peer peer, Head head, bool overlap)
        => new(64, 2, 3, 2, 7, head.Forward, head.Reset, head.Release, peer.Connect, overlap);

    private static void Wait(ManualResetEventSlim signal)
        => Assert.True(signal.Wait(TimeSpan.FromSeconds(5)), "Fake peer synchronization timed out");

    private sealed class Head
    {
        private int _position, _calls, _resets, _releases;
        public int Calls => Volatile.Read(ref _calls);
        public int Resets => Volatile.Read(ref _resets);
        public int Releases => Volatile.Read(ref _releases);
        public Action<int>? Before;
        public Func<bool>? Joined;
        public ManualResetEventSlim ThirdStarted { get; } = new();
        public ConcurrentBag<int> Threads { get; } = new();
        public int Forward(int[] tokens, float[] hidden, float[] pre)
        {
            int call = Interlocked.Increment(ref _calls);
            if (call == 3) ThirdStarted.Set();
            Threads.Add(Environment.CurrentManagedThreadId);
            Before?.Invoke(call);
            for (int row = 0; row < tokens.Length; ++row)
            {
                for (int lane = 0; lane < 6; ++lane) hidden[row * 6 + lane] = tokens[row] + lane * 0.125f;
                for (int lane = 0; lane < 2; ++lane) pre[row * 2 + lane] = tokens[row] * 0.25f + lane * 0.125f;
            }
            return _position += tokens.Length;
        }
        public void Reset()
        {
            Assert.True(Joined?.Invoke() ?? true, "Head reset preceded reply-reader join");
            _position = 0; Interlocked.Increment(ref _resets);
        }
        public void Release()
        {
            Assert.True(Joined?.Invoke() ?? true, "Head free preceded reply-reader join");
            Interlocked.Increment(ref _releases);
        }
        public static float[] ExpectedLogits(int position)
            => Enumerable.Range(0, 7).Select(i => position + i * 0.125f).ToArray();
    }

    private sealed record Chunk(int Position, int[] Tokens, bool RequestLogits) { public int End => Position + Tokens.Length; }

    private sealed class Session : IDisposable
    {
        private readonly NetworkStream _stream;
        private readonly int _hidden, _hc, _vocab, _width;
        public BinaryReader Reader { get; }
        public Session(TcpClient client)
        {
            client.NoDelay = true; client.ReceiveTimeout = client.SendTimeout = 5000;
            _stream = client.GetStream(); Reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: true);
            Assert.Equal(new string('a', 64), DeepSeek41PipelineExecutor.ReadHandshakeIdentity(Reader));
            _hidden = Reader.ReadInt32(); _hc = Reader.ReadInt32(); _vocab = Reader.ReadInt32();
            Assert.Equal(64, Reader.ReadInt32()); _width = Reader.ReadInt32();
            using var reply = new MemoryStream();
            using (var writer = new BinaryWriter(reply, Encoding.UTF8, true)) { writer.Write(0); writer.Write(_width); }
            WriteFragmented(reply.ToArray());
        }
        public Chunk Receive()
        {
            int op = Reader.ReadInt32();
            Assert.Contains(op, new[] { 1, 3 });
            int position = Reader.ReadInt32(), count = Reader.ReadInt32();
            Assert.InRange(count, 1, _width);
            var tokens = new int[count]; var hidden = new float[count * _hidden * _hc]; var pre = new float[count * _hc];
            DeepSeek41PipelineExecutor.ReadArray(_stream, tokens.AsSpan());
            DeepSeek41PipelineExecutor.ReadArray(_stream, hidden.AsSpan());
            DeepSeek41PipelineExecutor.ReadArray(_stream, pre.AsSpan());
            for (int row = 0; row < count; ++row)
            {
                for (int lane = 0; lane < _hidden * _hc; ++lane)
                    Assert.Equal(tokens[row] + lane * 0.125f, hidden[row * _hidden * _hc + lane]);
                for (int lane = 0; lane < _hc; ++lane)
                    Assert.Equal(tokens[row] * 0.25f + lane * 0.125f, pre[row * _hc + lane]);
            }
            return new Chunk(position, tokens, op == 1);
        }
        public void Reply(Chunk chunk, int? position = null)
        {
            Assert.Equal(7, _vocab);
            using var bytes = new MemoryStream();
            DeepSeek41PipelineExecutor.WriteForwardReply(bytes, position ?? chunk.End,
                chunk.RequestLogits ? Head.ExpectedLogits(chunk.End) : null);
            WriteFragmented(bytes.ToArray());
        }
        public void WriteFragmented(byte[] bytes)
        {
            for (int offset = 0; offset < bytes.Length; offset += 3)
                _stream.Write(bytes, offset, Math.Min(3, bytes.Length - offset));
        }
        public void ExpectClosed()
        {
            try { Assert.Equal(0, _stream.ReadByte() + 1); }
            catch (IOException) { } // abortive socket close is also a valid cancellation
        }
        public void Dispose() { Reader.Dispose(); _stream.Dispose(); }
    }

    private sealed class Peer : IDisposable
    {
        private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
        public Task Completion { get; }
        public Peer(params Action<Session>[] scripts)
        {
            _listener.Start();
            Completion = Task.Run(() =>
            {
                foreach (Action<Session> script in scripts)
                {
                    using TcpClient client = _listener.AcceptTcpClient();
                    using var session = new Session(client);
                    try { script(session); }
                    catch (IOException) { } // production worker also accepts a fresh connection after a client abort
                }
            });
        }
        public Stream Connect()
        {
            var socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp) { NoDelay = true };
            socket.Connect((IPEndPoint)_listener.LocalEndpoint);
            return new FragmentedStream(new NetworkStream(socket, ownsSocket: true));
        }
        public void Dispose() => _listener.Stop();
    }

    private sealed class FragmentedStream(Stream inner) : Stream
    {
        public override bool CanRead => inner.CanRead;
        public override bool CanSeek => false;
        public override bool CanWrite => inner.CanWrite;
        public override bool CanTimeout => inner.CanTimeout;
        public override int ReadTimeout { get => inner.ReadTimeout; set => inner.ReadTimeout = value; }
        public override int WriteTimeout { get => inner.WriteTimeout; set => inner.WriteTimeout = value; }
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() => inner.Flush();
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, Math.Min(count, 3));
        public override int Read(Span<byte> buffer) => inner.Read(buffer[..Math.Min(buffer.Length, 3)]);
        public override void Write(byte[] buffer, int offset, int count) => inner.Write(buffer, offset, count);
        public override void Write(ReadOnlySpan<byte> buffer) => inner.Write(buffer);
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        protected override void Dispose(bool disposing) { if (disposing) inner.Dispose(); base.Dispose(disposing); }
    }

    private sealed class CountingStream : MemoryStream
    {
        public int Writes;
        public override void Write(ReadOnlySpan<byte> buffer) { ++Writes; base.Write(buffer); }
    }
}
