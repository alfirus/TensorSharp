using System.Collections.Concurrent;
using System.Runtime.InteropServices;

internal sealed class PersistentWorker : IDisposable
{
    private sealed record Work(Func<object?> Callback, TaskCompletionSource<object?> Completion);
    private readonly BlockingCollection<Work> queue = new();
    private readonly object state = new();
    private readonly object disposal = new();
    private readonly Thread thread;
    private bool disposed;
    private int active;
    internal int ManagedThreadId { get; private set; }
    internal int NativeThreadId { get; private set; }
    internal int ActiveCallbacks => Volatile.Read(ref active);
    internal bool Joined => disposed && !thread.IsAlive;
    [DllImport("libc", EntryPoint = "gettid")] private static extern int GetTid();

    internal PersistentWorker(string name)
    {
        var ready = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        thread = new Thread(() =>
        {
            try
            {
                ManagedThreadId = Environment.CurrentManagedThreadId;
                NativeThreadId = OperatingSystem.IsLinux() ? GetTid() : 0;
                ready.SetResult(true);
            }
            catch (Exception error) { ready.SetException(error); return; }
            foreach (var work in queue.GetConsumingEnumerable())
            {
                object? result = null; Exception? error = null;
                Interlocked.Increment(ref active);
                try { result = work.Callback(); }
                catch (Exception caught) { error = caught; }
                finally { Interlocked.Decrement(ref active); }
                // Completion is published only after the callback becomes idle.
                if (error != null) work.Completion.SetException(error);
                else work.Completion.SetResult(result);
            }
        }) { Name = name, IsBackground = false };
        thread.Start();
        try { ready.Task.GetAwaiter().GetResult(); }
        catch { thread.Join(); queue.Dispose(); throw; }
    }
    internal Task<T> Submit<T>(Func<T> callback)
    {
        ArgumentNullException.ThrowIfNull(callback);
        var completion = new TaskCompletionSource<object?>(TaskCreationOptions.RunContinuationsAsynchronously);
        lock (state)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            queue.Add(new Work(() => callback(), completion));
        }
        return ConvertResult<T>(completion.Task);
    }
    private static async Task<T> ConvertResult<T>(Task<object?> task) => (T)(await task.ConfigureAwait(false))!;
    internal T Invoke<T>(Func<T> callback) => Submit(callback).GetAwaiter().GetResult();
    public void Dispose()
    {
        if (Thread.CurrentThread == thread) throw new InvalidOperationException("A worker cannot join itself.");
        lock (disposal)
        {
            lock (state)
            {
                if (disposed) return;
                disposed = true; queue.CompleteAdding();
            }
            thread.Join(); queue.Dispose();
            if (ActiveCallbacks != 0) throw new InvalidOperationException("Worker joined with an active callback.");
        }
    }
}
