// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

public sealed class WebUiSteeringTests
{
    [Fact]
    public async Task SessionDeletedWhileWaitingCannotStartTheQueuedRequest()
    {
        using var model = new ModelService();
        var sessions = new SessionManager();
        var session = sessions.CreateSession();
        var first = await session.WebUiTurns.EnterAsync(false, CancellationToken.None);
        string root = Path.GetTempPath();
        var options = ServerOptionsBuilder.Build(new[] { "--model", "steering.gguf", "--no-skills" }, root);
        var service = new WebUiChatService(model, sessions, options, new UploadStoragePolicy(root),
            new SkillRegistry(new SkillRegistryOptions()), null, null, null, NullLoggerFactory.Instance);
        bool prepared = false;
        service.AcquireChatRequestLease = _ => { prepared = true; return null; };
        using var request = JsonDocument.Parse(JsonSerializer.Serialize(new
        {
            sessionId = session.Id,
            messages = new[] { new { role = "user", content = "Queued prompt" } },
        }));
        await using var stream = service.ChatStreamAsync(request.RootElement, CancellationToken.None).GetAsyncEnumerator();
        Task<bool> reading = stream.MoveNextAsync().AsTask();
        try
        {
            Assert.False(reading.IsCompleted);
            await service.DisposeSessionAsync(session.Id, CancellationToken.None);
        }
        finally { first.Dispose(); }

        var exception = await Assert.ThrowsAsync<WebUiRequestRejectedException>(() => reading);
        Assert.Equal(404, exception.StatusCode);
        Assert.False(prepared);
    }

    [Fact]
    public async Task ReplacementCannotCancelTheSharedDefaultSession()
    {
        using var model = new ModelService();
        var sessions = new SessionManager();
        string root = Path.GetTempPath();
        var options = ServerOptionsBuilder.Build(new[] { "--model", "steering.gguf", "--no-skills" }, root);
        var service = new WebUiChatService(model, sessions, options, new UploadStoragePolicy(root),
            new SkillRegistry(new SkillRegistryOptions()), null, null, null, NullLoggerFactory.Instance);
        using var request = JsonDocument.Parse("""{"replaceActive":true,"messages":[{"role":"user","content":"Steer"}]}""");

        var exception = await Assert.ThrowsAsync<WebUiRequestRejectedException>(async () =>
        {
            await foreach (var _ in service.ChatStreamAsync(request.RootElement, CancellationToken.None)) { }
        });
        Assert.Equal(400, exception.StatusCode);
        Assert.Contains("dedicated sessionId", exception.Message);
    }

    [Fact]
    public async Task ReplacementCancelsPriorTurnButWaitsForItsTranscriptCleanup()
    {
        var turns = new WebUiTurnCoordinator();
        var first = await turns.EnterAsync(false, CancellationToken.None);
        Task<WebUiTurnCoordinator.Lease> replacement = turns.EnterAsync(true, CancellationToken.None);

        Assert.True(first.Token.IsCancellationRequested);
        Assert.False(replacement.IsCompleted);
        first.Dispose();
        using var next = await replacement.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.False(next.Token.IsCancellationRequested);
    }

    [Fact]
    public async Task RapidReplacementCancelsWaitersWithoutOvertakingTheRunningTurn()
    {
        var turns = new WebUiTurnCoordinator();
        var first = await turns.EnterAsync(false, CancellationToken.None);
        Task<WebUiTurnCoordinator.Lease> second = turns.EnterAsync(true, CancellationToken.None);
        Task<WebUiTurnCoordinator.Lease> third = turns.EnterAsync(true, CancellationToken.None);

        Assert.False(second.IsCompleted);
        Assert.False(third.IsCompleted);
        first.Dispose();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => second);
        using var last = await third.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.False(last.Token.IsCancellationRequested);
    }

    [Fact]
    public async Task CancelledQueuedRequestDoesNotReleaseItsPredecessorEarly()
    {
        var turns = new WebUiTurnCoordinator();
        var first = await turns.EnterAsync(false, CancellationToken.None);
        using var cancel = new CancellationTokenSource();
        Task<WebUiTurnCoordinator.Lease> second = turns.EnterAsync(false, cancel.Token);
        cancel.Cancel();
        Task<WebUiTurnCoordinator.Lease> third = turns.EnterAsync(false, CancellationToken.None);

        Assert.False(first.Token.IsCancellationRequested);
        Assert.False(third.IsCompleted);
        first.Dispose();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => second);
        using var last = await third.WaitAsync(TimeSpan.FromSeconds(5));
    }

    [Fact]
    public async Task ReplacementInAnotherSessionDoesNotCancelThisConversation()
    {
        using var firstSession = new ChatSession();
        using var secondSession = new ChatSession();
        using var first = await firstSession.WebUiTurns.EnterAsync(false, CancellationToken.None);
        using var second = await secondSession.WebUiTurns.EnterAsync(true, CancellationToken.None);
        Assert.False(first.Token.IsCancellationRequested);
    }

    [Fact]
    public async Task AlreadyAbortedRequestCannotInterruptAnActiveTurn()
    {
        var turns = new WebUiTurnCoordinator();
        using var first = await turns.EnterAsync(false, CancellationToken.None);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            turns.EnterAsync(true, new CancellationToken(canceled: true)));
        Assert.False(first.Token.IsCancellationRequested);
    }

    [Fact]
    public void InterruptedAssistantReasoningSurvivesWebUiHistoryParsing()
    {
        using var json = JsonDocument.Parse("""
            [{"role":"user","content":"Explain","thinking":"untrusted channel"},
             {"role":"assistant","content":"","thinking":"Considering alternatives"},
             {"role":"user","content":"Focus on cost"}]
            """);
        var messages = ChatMessageParser.ParseWebUi(json.RootElement);
        Assert.Null(messages[0].Thinking);
        Assert.Equal("Considering alternatives", messages[1].Thinking);
        Assert.Equal("Focus on cost", messages[2].Content);
    }

    [Fact]
    public async Task DisconnectedStreamIsCancelledAndDrainedForPartialTranscriptRecording()
    {
        var context = new DefaultHttpContext();
        context.Response.Body = new DisconnectedStream();
        using var cancel = new CancellationTokenSource();
        bool recorded = false;
        bool disposed = false;

        async IAsyncEnumerable<object> Frames([EnumeratorCancellation] CancellationToken ct)
        {
            try
            {
                yield return new { token = "Partial answer" };
                Assert.True(ct.IsCancellationRequested);
                // Represents the producer's post-cancellation engine completion and
                // transcript write; DisposeAsync alone cannot execute this statement.
                await Task.Yield();
                recorded = true;
                yield return new { done = true, aborted = true };
            }
            finally { disposed = true; }
        }

        await WebUiAdapter.WriteChatFramesAsync(context, Frames(cancel.Token), cancel,
            NullLogger.Instance).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(cancel.IsCancellationRequested);
        Assert.True(recorded);
        Assert.True(disposed);
    }

    private sealed class DisconnectedStream : MemoryStream
    {
        public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken ct = default)
            => ValueTask.FromException(new IOException("Client disconnected"));
        public override Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken ct)
            => Task.FromException(new IOException("Client disconnected"));
    }
}
