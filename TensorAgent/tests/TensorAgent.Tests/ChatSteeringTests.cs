// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System.Runtime.CompilerServices;
using System.Text.Json;
using TensorAgent.Core.Sessions;

namespace TensorAgent.Tests;

public sealed partial class ChatTurnTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SteeringPreservesAnUnacceptedQuestionAndOnlyOmitsAlreadyAcceptedShares(bool shareAccepted)
    {
        string root = Path.Combine(Path.GetTempPath(), "steering-request-" + Guid.NewGuid().ToString("N"));
        try
        {
            var store = new ConversationStore(root);
            var recorder = new ConversationRecorder(store);
            recorder.Bind("session", "new");
            JsonElement original = JsonSerializer.Deserialize<JsonElement>("""
                {"shareIds":["shared-draft"],"messages":[{"role":"user","content":"Earlier question"},{"role":"assistant","content":"Earlier answer"}]}
                """);
            recorder.Record("session", original, accepted: shareAccepted);
            JsonElement steering = JsonSerializer.Deserialize<JsonElement>("""
                {"shareIds":["shared-draft"],"messages":[{"role":"user","content":"Earlier question"},{"role":"assistant","content":"Earlier answer"},
                {"role":"user","content":"Question still in preflight","imagePaths":["photo.png"]},{"role":"user","content":"New guidance"}]}
                """);
            JsonElement merged = recorder.SteeringRequest("session", steering);
            Assert.Equal(new[] { "Earlier question", "Earlier answer", "Question still in preflight", "New guidance" },
                merged.GetProperty("messages").EnumerateArray().Select(message => message.GetProperty("content").GetString()));
            Assert.Equal("photo.png", merged.GetProperty("messages")[2].GetProperty("imagePaths")[0].GetString());
            Assert.Equal(shareAccepted ? 0 : 1, merged.GetProperty("shareIds").GetArrayLength());
            Assert.Equal(1, steering.GetProperty("shareIds").GetArrayLength());
        }
        finally { try { Directory.Delete(root, true); } catch (IOException) { } }
    }

    [Fact]
    public async Task OrdinaryReplacementsCannotBypassAPendingSteeringCleanup()
    {
        using var turns = new ChatTurnManager();
        using var release = new SemaphoreSlim(0);
        var unwound = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var finalStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        string original = turns.Start("conversation", _ => Stuck(release, unwound, "Partial", "session"));
        await WaitFor(() => turns.StatusOfId(original)!.FrameCount > 0);
        turns.StartSteered("conversation", _ => One());
        turns.Start("conversation", _ => One());
        string final = turns.Start("conversation", _ => Final());
        Assert.False(finalStarted.Task.IsCompleted);
        release.Release();
        await finalStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        await WaitFor(() => !turns.StatusOfId(final)!.IsRunning);
        Assert.True(unwound.Task.IsCompleted);

        async IAsyncEnumerable<object> Final()
        {
            finalStarted.TrySetResult();
            await Task.Yield();
            yield return new { token = "Final" };
        }
    }

    [Fact]
    public async Task RapidSteeringWaitsForCleanupAndPreservesPartialOutputAndEveryGuidanceMessage()
    {
        string root = Path.Combine(Path.GetTempPath(), "turn-steering-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            using var harness = new RouteHarness(root);
            using var release = new SemaphoreSlim(0);
            var unwound = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var continued = new TaskCompletionSource<JsonElement>(TaskCreationOptions.RunContinuationsAsynchronously);
            harness.RequestFrames = Response;

            using HttpResponseMessage original = await harness.SendChatAsync("Explain the plan");
            string originalId = harness.Turns.StatusOfKey(harness.ConversationId)!.Id;
            await WaitFor(() => harness.Turns.StatusOfId(originalId)!.FrameCount == 2);
            Task<HttpResponseMessage> first = harness.SendChatAsync("Use fewer words", steer: true);
            await WaitFor(() => harness.Turns.StatusOfKey(harness.ConversationId)!.Id != originalId);
            string firstId = harness.Turns.StatusOfKey(harness.ConversationId)!.Id;
            Task<HttpResponseMessage> second = harness.SendChatAsync("Also give an example", steer: true);
            await WaitFor(() => harness.Turns.StatusOfKey(harness.ConversationId)!.Id != firstId);

            Assert.False(continued.Task.IsCompleted, "a successor ran before the interrupted producer unwound");
            release.Release();
            using HttpResponseMessage firstResponse = await first.WaitAsync(TimeSpan.FromSeconds(10));
            using HttpResponseMessage secondResponse = await second.WaitAsync(TimeSpan.FromSeconds(10));
            string frames = await secondResponse.Content.ReadAsStringAsync();
            Assert.Contains("steeringMessages", frames, StringComparison.Ordinal);
            Assert.DoesNotContain("late stale token", frames, StringComparison.Ordinal);
            JsonElement body = await continued.Task.WaitAsync(TimeSpan.FromSeconds(10));
            JsonElement[] messages = body.GetProperty("messages").EnumerateArray().ToArray();
            Assert.Equal(new[] { "Explain the plan", "Partial answer", "Use fewer words", "Also give an example" },
                messages.Select(message => message.GetProperty("content").GetString()));
            Assert.Equal("Useful reasoning", messages[1].GetProperty("thinking").GetString());
            await WaitFor(() => !harness.Turns.IsBusy);
            Assert.Equal(ChatTurnState.Cancelled, harness.Turns.StatusOfId(originalId)!.State);
            Assert.Equal(ChatTurnState.Cancelled, harness.Turns.StatusOfId(firstId)!.State);
            Conversation saved = new ConversationStore(harness.ConversationRoot).Load(harness.ConversationId)!;
            Assert.Equal(new[] { "Explain the plan", "Partial answer", "Use fewer words", "Also give an example", "Short answer with example" },
                saved.Messages.Select(message => message.Content));
            Assert.Equal("Useful reasoning", saved.Messages[1].Thinking);

            async IAsyncEnumerable<object> Response(JsonElement body, [EnumeratorCancellation] CancellationToken token)
            {
                string prompt = body.GetProperty("messages").EnumerateArray().Last().GetProperty("content").GetString()!;
                if (prompt == "Explain the plan")
                {
                    harness.Recorder.Record(harness.SessionId, body);
                    try
                    {
                        // Neither frame names the session. Cancellation during reasoning
                        // must retain it from the request instead of waiting for done.
                        yield return new { thinking = "Useful reasoning" };
                        yield return new { token = "Partial answer" };
                        await release.WaitAsync(); // a native/tool step finishing after cancellation
                        yield return new { token = "late stale token" };
                    }
                    finally { unwound.TrySetResult(); }
                    yield break;
                }
                Assert.True(unwound.Task.IsCompleted);
                continued.TrySetResult(body.Clone());
                token.ThrowIfCancellationRequested();
                yield return new { token = "Short answer with example" };
                yield return new { done = true, sessionId = harness.SessionId };
            }
        }
        finally { try { Directory.Delete(root, true); } catch (IOException) { } }
    }

    [Fact]
    public async Task SteeringDuringPrefillNeedsNoTurnHeaderAndKeepsTheOriginalQuestion()
    {
        string root = Path.Combine(Path.GetTempPath(), "turn-prefill-steering-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            using var harness = new RouteHarness(root);
            var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var unwound = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var continued = new TaskCompletionSource<JsonElement>(TaskCreationOptions.RunContinuationsAsynchronously);
            harness.RequestFrames = Response;
            Task<HttpResponseMessage> original = harness.SendChatAsync("Original question");
            await started.Task.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.Equal(0, harness.Turns.FramesFor(harness.ConversationId));
            using HttpResponseMessage redirected = await harness.SendChatAsync("Focus on the conclusion", steer: true);
            await redirected.Content.ReadAsStringAsync();
            using HttpResponseMessage originalResponse = await original.WaitAsync(TimeSpan.FromSeconds(10));
            JsonElement body = await continued.Task.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.Equal(new[] { "Original question", "Focus on the conclusion" },
                body.GetProperty("messages").EnumerateArray().Select(message => message.GetProperty("content").GetString()));
            await WaitFor(() => !harness.Turns.IsBusy);

            async IAsyncEnumerable<object> Response(JsonElement body, [EnumeratorCancellation] CancellationToken token)
            {
                string prompt = body.GetProperty("messages").EnumerateArray().Last().GetProperty("content").GetString()!;
                if (prompt == "Original question")
                {
                    harness.Recorder.Record(harness.SessionId, body);
                    started.TrySetResult();
                    try { await Task.Delay(Timeout.Infinite, token); }
                    finally { unwound.TrySetResult(); }
                }
                Assert.True(unwound.Task.IsCompleted);
                continued.TrySetResult(body.Clone());
                yield return new { token = "Conclusion" };
                yield return new { done = true, sessionId = harness.SessionId };
            }
        }
        finally { try { Directory.Delete(root, true); } catch (IOException) { } }
    }
}
