// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Chat;

/// <summary>
/// Serializes a dedicated chat's requests through cancellation and transcript cleanup.
/// A replacement also cancels requests that are still waiting to start.
/// </summary>
internal sealed class WebUiTurnCoordinator
{
    private readonly object _gate = new();
    private readonly List<Lease> _pending = new();
    private Task _tail = Task.CompletedTask;

    public async Task<Lease> EnterAsync(bool replaceActive, CancellationToken cancellationToken)
    {
        Task predecessor;
        Lease lease;
        Lease[] interrupted;
        lock (_gate)
        {
            cancellationToken.ThrowIfCancellationRequested();
            interrupted = replaceActive ? _pending.ToArray() : Array.Empty<Lease>();
            predecessor = _tail;
            lease = new Lease(this, cancellationToken);
            _pending.Add(lease);
            _tail = lease.Completion;
        }

        foreach (Lease previous in interrupted)
            previous.Cancel();

        try
        {
            // Even a cancelled waiter keeps its place until its predecessor has
            // finished. Otherwise a third prompt could overtake unfinished cleanup.
            await predecessor.ConfigureAwait(false);
            lease.Token.ThrowIfCancellationRequested();
            return lease;
        }
        catch
        {
            lease.Dispose();
            throw;
        }
    }

    private void Exit(Lease lease)
    {
        lock (_gate)
            _pending.Remove(lease);
    }

    internal sealed class Lease : IDisposable
    {
        private readonly WebUiTurnCoordinator _owner;
        private readonly CancellationTokenSource _cancel;
        private readonly TaskCompletionSource _finished = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private int _disposed;

        internal Lease(WebUiTurnCoordinator owner, CancellationToken cancellationToken)
        {
            _owner = owner;
            _cancel = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            Token = _cancel.Token;
        }

        public CancellationToken Token { get; }
        internal Task Completion => _finished.Task;

        internal void Cancel()
        {
            // A finishing predecessor may dispose between our snapshot and cancel.
            try { _cancel.Cancel(); }
            catch (ObjectDisposedException) { }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            _owner.Exit(this);
            _cancel.Dispose();
            _finished.TrySetResult();
        }
    }
}
