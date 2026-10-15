using System.Runtime.InteropServices;

internal enum SchedulingMode { SameSplit, ThreadPoolResident, PublicResident }

internal static class SchedulingModes
{
    internal static SchedulingMode Parse(string? value, string? prewarmImage)
    {
        var mode = value switch
        {
            null or "same-split" => SchedulingMode.SameSplit,
            "threadpool-resident" => SchedulingMode.ThreadPoolResident,
            "public-resident" => SchedulingMode.PublicResident,
            _ => throw new ArgumentException("Mode must be same-split, threadpool-resident or public-resident.")
        };
        if (prewarmImage != null && (mode == SchedulingMode.SameSplit || string.IsNullOrWhiteSpace(prewarmImage)))
            throw new ArgumentException("A nonempty prewarm image requires threadpool-resident or public-resident mode.");
        return mode;
    }

    internal static (string Name, bool Route)[] Phases(SchedulingMode mode) => mode switch
    {
        SchedulingMode.SameSplit => [("same-a", false), ("split-a", true), ("split-b", true), ("same-b", false)],
        SchedulingMode.ThreadPoolResident => [("whole-taskrun-a", true), ("resident-a", false), ("resident-b", false), ("whole-taskrun-b", true)],
        SchedulingMode.PublicResident => [("public-a", true), ("resident-a", false), ("resident-b", false), ("public-b", true)],
        _ => throw new ArgumentOutOfRangeException(nameof(mode))
    };
}

internal sealed record CallerIdentity(int ManagedThreadId, int NativeThreadId, bool IsThreadPoolThread)
{
    [DllImport("libc", EntryPoint = "gettid")] private static extern int GetTid();
    internal static CallerIdentity Capture() => new(Environment.CurrentManagedThreadId,
        OperatingSystem.IsLinux() ? GetTid() : 0, Thread.CurrentThread.IsThreadPoolThread);
}

internal static class SchedulingDispatch
{
    // Both branches invoke the exact same synchronous callback and wait until
    // completion before returning or propagating its exception. No pool tuning,
    // artificial worker churn, asynchronous native work or cancellation policy.
    internal static T Run<T>(bool threadPool, PersistentWorker owner, Func<T> callback)
    {
        ArgumentNullException.ThrowIfNull(owner);
        ArgumentNullException.ThrowIfNull(callback);
        return threadPool ? Task.Run(callback, CancellationToken.None).GetAwaiter().GetResult() : owner.Invoke(callback);
    }
}
