// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Cli;

/// <summary>
/// Keeps one console read outstanding while a response runs. All turns and commands
/// are serialized: an interrupt must finish draining the engine before a replacement
/// turn can touch conversation history, media, or the model.
/// </summary>
internal sealed class InteractiveChatLoop
{
    private readonly object _gate = new();
    private readonly TaskCompletionSource _exit = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private CancellationTokenSource _generation;

    /// <returns>True when an active response was stopped; false when exit was requested.</returns>
    public bool InterruptOrExit()
    {
        lock (_gate)
        {
            if (_generation != null)
            {
                _generation.Cancel();
                return true;
            }
            _exit.TrySetResult();
            return false;
        }
    }

    public async Task RunAsync(
        Func<Task<string>> readInput,
        Action<string, CancellationToken> runTurn,
        Func<string, bool> handleCommand,
        Action showPrompt,
        bool allowSteering)
    {
        Task<string> input = null;
        Task turn = null;
        string next = null;
        bool hasNext = false;
        bool promptShown = false;
        try
        {
            while (!_exit.Task.IsCompleted)
            {
                if (turn != null)
                {
                    if (!allowSteering)
                    {
                        await turn.ConfigureAwait(false);
                        FinishGeneration();
                        turn = null;
                        continue;
                    }

                    input ??= readInput();
                    // Favor a submitted line when both tasks finish together. That line
                    // still belongs to this conversation and must never be discarded.
                    await Task.WhenAny(input, turn, _exit.Task).ConfigureAwait(false);
                    if (!input.IsCompleted)
                    {
                        if (_exit.Task.IsCompleted)
                            break;
                        await turn.ConfigureAwait(false);
                        FinishGeneration();
                        turn = null;
                        continue;
                    }

                    next = await input.ConfigureAwait(false);
                    input = null;
                    if (next != null && string.IsNullOrWhiteSpace(next))
                        continue;

                    CancelGeneration();
                    await turn.ConfigureAwait(false);
                    FinishGeneration();
                    turn = null;
                    hasNext = true;
                    promptShown = false;
                }

                if (!hasNext)
                {
                    if (!promptShown)
                    {
                        showPrompt();
                        promptShown = true;
                    }
                    input ??= readInput();
                    await Task.WhenAny(input, _exit.Task).ConfigureAwait(false);
                    if (_exit.Task.IsCompleted)
                        break;
                    next = await input.ConfigureAwait(false);
                    input = null;
                }
                hasNext = false;
                promptShown = false;
                if (next == null)
                    break;
                string trimmed = next.Trim();
                if (trimmed.Length == 0)
                    continue;
                if (trimmed.StartsWith("/", StringComparison.Ordinal))
                {
                    if (handleCommand(trimmed))
                        break;
                    continue;
                }

                CancellationToken token;
                lock (_gate)
                {
                    if (_exit.Task.IsCompleted)
                        break;
                    _generation = new CancellationTokenSource();
                    token = _generation.Token;
                }
                string prompt = next;
                // No cancellation token on Task.Run: even an early interruption must
                // let the turn account for its submitted user message and attachments.
                turn = Task.Run(() => runTurn(prompt, token));
            }
        }
        finally
        {
            CancelGeneration();
            try
            {
                if (turn != null)
                    await turn.ConfigureAwait(false);
            }
            finally
            {
                FinishGeneration();
            }
        }
    }

    private void CancelGeneration()
    {
        lock (_gate)
            _generation?.Cancel();
    }

    private void FinishGeneration()
    {
        lock (_gate)
        {
            _generation?.Dispose();
            _generation = null;
        }
    }
}
