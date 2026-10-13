// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections.Concurrent;
using System.Threading.Channels;
using TensorSharp.Cli;

namespace InferenceWeb.Tests;

public sealed class InteractiveChatSteeringTests
{
    private static TaskCompletionSource Signal() => new(TaskCreationOptions.RunContinuationsAsynchronously);
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(10);

    [Fact]
    public async Task SteeringDrainsThePreviousResponseAndAcceptsMultiplePrompts()
    {
        var input = Channel.CreateUnbounded<string>();
        var started = new[] { Signal(), Signal(), Signal() };
        var cancelled = Signal();
        using var drain = new ManualResetEventSlim();
        var prompts = new ConcurrentQueue<string>();
        var order = new ConcurrentQueue<string>();
        int active = 0;
        var loop = new InteractiveChatLoop();
        Task running = loop.RunAsync(() => input.Reader.ReadAsync().AsTask(), (prompt, token) =>
        {
            Assert.Equal(1, Interlocked.Increment(ref active));
            int index = prompts.Count;
            prompts.Enqueue(prompt);
            started[index].TrySetResult();
            Assert.True(token.WaitHandle.WaitOne(Timeout), "turn was not interrupted");
            if (index == 0)
            {
                cancelled.TrySetResult();
                Assert.True(drain.Wait(Timeout), "engine drain was not released");
            }
            order.Enqueue(prompt + " drained");
            Interlocked.Decrement(ref active);
        }, command => { order.Enqueue(command); return command == "/exit"; }, () => { }, allowSteering: true);

        try
        {
            await input.Writer.WriteAsync("original");
            await started[0].Task.WaitAsync(Timeout);
            await input.Writer.WriteAsync("make it concise");
            await cancelled.Task.WaitAsync(Timeout);
            Assert.False(started[1].Task.IsCompleted);
            drain.Set();
            await started[1].Task.WaitAsync(Timeout);
            await input.Writer.WriteAsync("use bullet points too");
            await started[2].Task.WaitAsync(Timeout);
            await input.Writer.WriteAsync("/exit");
            await running.WaitAsync(Timeout);
            Assert.Equal(new[] { "original", "make it concise", "use bullet points too" }, prompts);
            Assert.Equal(new[] { "original drained", "make it concise drained", "use bullet points too drained", "/exit" }, order);
            Assert.Equal(0, active);
        }
        finally
        {
            drain.Set();
            loop.InterruptOrExit();
            input.Writer.TryWrite("/exit");
            await running.WaitAsync(Timeout);
        }
    }

    [Fact]
    public async Task BlankLinesDoNotInterruptAndCommandsRunAfterCancellation()
    {
        var input = Channel.CreateUnbounded<string>();
        var started = Signal();
        var readAfterBlank = Signal();
        int reads = 0;
        bool drained = false;
        string commandSeen = null;
        var loop = new InteractiveChatLoop();
        Task running = loop.RunAsync(() =>
        {
            if (Interlocked.Increment(ref reads) == 3)
                readAfterBlank.TrySetResult();
            return input.Reader.ReadAsync().AsTask();
        }, (_, token) =>
        {
            started.TrySetResult();
            Assert.True(token.WaitHandle.WaitOne(Timeout));
            drained = true;
        }, command =>
        {
            Assert.True(drained);
            commandSeen = command;
            return true;
        }, () => { }, allowSteering: true);

        await input.Writer.WriteAsync("first prompt");
        await started.Task.WaitAsync(Timeout);
        await input.Writer.WriteAsync("  ");
        await readAfterBlank.Task.WaitAsync(Timeout);
        Assert.False(drained);
        await input.Writer.WriteAsync("  /reset  ");
        await running.WaitAsync(Timeout);
        Assert.Equal("/reset", commandSeen);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task EndOfInputEndsTheLoopAndCancelsAnyActiveResponse(bool active)
    {
        var input = Channel.CreateUnbounded<string>();
        var started = Signal();
        bool drained = false;
        Task running = new InteractiveChatLoop().RunAsync(async () =>
        {
            if (await input.Reader.WaitToReadAsync())
                return await input.Reader.ReadAsync();
            return null;
        }, (_, token) =>
        {
            started.TrySetResult();
            Assert.True(token.WaitHandle.WaitOne(Timeout));
            drained = true;
        }, _ => throw new InvalidOperationException("EOF became a command"), () => { }, allowSteering: true);

        if (active)
        {
            await input.Writer.WriteAsync("prompt");
            await started.Task.WaitAsync(Timeout);
        }
        input.Writer.Complete();
        await running.WaitAsync(Timeout);
        Assert.Equal(active, drained);
    }

    [Fact]
    public async Task IdleCtrlCExitsWithoutWaitingForTheConsoleRead()
    {
        var pendingRead = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        var loop = new InteractiveChatLoop();
        Task running = loop.RunAsync(() => pendingRead.Task,
            (_, _) => throw new InvalidOperationException("no prompt was submitted"), _ => false, () => { }, allowSteering: true);

        Assert.False(loop.InterruptOrExit());
        await running.WaitAsync(Timeout);
        Assert.False(pendingRead.Task.IsCompleted);
        pendingRead.SetResult(null);
    }

    [Fact]
    public async Task ActiveCtrlCStopsOnlyTheResponseAndTheNextPromptStillRuns()
    {
        var input = Channel.CreateUnbounded<string>();
        var started = Signal();
        var idle = Signal();
        var secondCompleted = Signal();
        var prompts = new ConcurrentQueue<string>();
        var loop = new InteractiveChatLoop();
        Task running = loop.RunAsync(() => input.Reader.ReadAsync().AsTask(), (prompt, token) =>
        {
            prompts.Enqueue(prompt);
            if (prompt == "first")
            {
                started.TrySetResult();
                Assert.True(token.WaitHandle.WaitOne(Timeout));
            }
            else
            {
                Assert.False(token.IsCancellationRequested);
                secondCompleted.TrySetResult();
            }
        }, command => command == "/exit", () =>
        {
            if (!prompts.IsEmpty)
                idle.TrySetResult();
        }, allowSteering: true);

        await input.Writer.WriteAsync("first");
        await started.Task.WaitAsync(Timeout);
        Assert.True(loop.InterruptOrExit());
        await idle.Task.WaitAsync(Timeout);
        await input.Writer.WriteAsync("second");
        // Queue the exit only after the complete second response, so it cannot
        // cancel the very response whose clean token this test checks.
        await secondCompleted.Task.WaitAsync(Timeout);
        await input.Writer.WriteAsync("/exit");
        await running.WaitAsync(Timeout);
        Assert.Equal(new[] { "first", "second" }, prompts);
    }

    [Fact]
    public async Task RedirectedTranscriptsWaitForEachResponseBeforeReadingAhead()
    {
        var inputs = new Queue<string>(new[] { "first", "second", null });
        var started = Signal();
        using var release = new ManualResetEventSlim();
        int reads = 0;
        var prompts = new ConcurrentQueue<string>();
        Task running = new InteractiveChatLoop().RunAsync(() =>
        {
            Interlocked.Increment(ref reads);
            return Task.FromResult(inputs.Dequeue());
        }, (prompt, token) =>
        {
            prompts.Enqueue(prompt);
            started.TrySetResult();
            Assert.True(release.Wait(Timeout));
            Assert.False(token.IsCancellationRequested);
        }, _ => false, () => { }, allowSteering: false);

        try
        {
            await started.Task.WaitAsync(Timeout);
            Assert.Equal(1, Volatile.Read(ref reads));
            release.Set();
            await running.WaitAsync(Timeout);
            Assert.Equal(new[] { "first", "second" }, prompts);
            Assert.Equal(3, reads);
        }
        finally
        {
            release.Set();
        }
    }
}
