using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.Server.Hosting;

/// <summary>Completed scheduled requests, not a claim that a prefix was retained.
/// The actual reuse count belongs to each subsequent request's telemetry.</summary>
public sealed class StartupWarmupStatus
{
    public sealed record Stage(bool Thinking, bool Completed, double ElapsedMilliseconds);
    public sealed record JointStage(int Width, bool RequestsCompleted, bool NativeBatchAccepted,
        bool CleanupCompleted, ulong NativeAcceptedExecutionId, int NativeNodeCount, double ElapsedMilliseconds, string Detail,
        bool CleanupPending = false)
    {
        public bool Completed => Width is 2 or 3 && RequestsCompleted && NativeBatchAccepted
            && CleanupCompleted && !CleanupPending && NativeAcceptedExecutionId > 0 && NativeNodeCount > 0;
    }

    public StartupWarmupStatus(IEnumerable<Stage> stages, bool prefixRetentionRequested,
        IEnumerable<JointStage>? jointStages = null, bool jointWarmupRequested = false, string? jointWarmupSkipReason = null)
        : this(stages, prefixRetentionRequested, jointStages, jointWarmupRequested, jointWarmupSkipReason, false) { }

    public StartupWarmupStatus(IEnumerable<Stage> stages, bool prefixRetentionRequested,
        IEnumerable<JointStage>? jointStages, bool jointWarmupRequested, string? jointWarmupSkipReason,
        bool kernelInitializationOnly)
    {
        var snapshot = stages?.ToArray() ?? throw new ArgumentNullException(nameof(stages));
        Stages = Array.AsReadOnly(snapshot);
        PrefixRetentionRequested = prefixRetentionRequested;
        Scope = kernelInitializationOnly ? "kernel-initialization" : "default-chat-prefix";
        Completed = snapshot.Length == 2 && snapshot.All(s => s.Completed)
            && snapshot.Any(s => s.Thinking) && snapshot.Any(s => !s.Thinking);
        ElapsedMilliseconds = snapshot.Sum(s => s.ElapsedMilliseconds);
        var joint = jointStages?.ToArray() ?? Array.Empty<JointStage>();
        JointStages = Array.AsReadOnly(joint);
        JointWarmupRequested = jointWarmupRequested;
        JointWarmupSkipReason = jointWarmupSkipReason;
        JointCompleted = joint.Length == 2 && joint.All(s => s.Completed)
            && joint.Any(s => s.Width == 2) && joint.Any(s => s.Width == 3);
        JointCleanupPending = joint.Any(s => s.CleanupPending);
        JointElapsedMilliseconds = joint.Sum(s => s.ElapsedMilliseconds);
    }

    public string Mode => "scheduled";
    public string Scope { get; }
    public bool Completed { get; }
    public bool PrefixRetentionRequested { get; }
    public double ElapsedMilliseconds { get; }
    public IReadOnlyList<Stage> Stages { get; }
    public bool JointWarmupRequested { get; }
    public string? JointWarmupSkipReason { get; }
    public bool JointCompleted { get; }
    public bool JointCleanupPending { get; }
    public double JointElapsedMilliseconds { get; }
    public IReadOnlyList<JointStage> JointStages { get; }
}

/// <summary>Never carry a startup result across unload/reload, or retain the old model.</summary>
internal sealed class StartupWarmupStatusStore
{
    private readonly object _gate = new();
    private WeakReference<object> _model;
    private StartupWarmupStatus _status;

    public void Set(object model, StartupWarmupStatus status)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        lock (_gate)
        {
            _model = new WeakReference<object>(model);
            _status = status ?? throw new ArgumentNullException(nameof(status));
        }
    }

    public StartupWarmupStatus Get(object model)
    {
        lock (_gate)
            return model != null && _model != null && _model.TryGetTarget(out var target)
                && ReferenceEquals(model, target) ? _status : null;
    }
}
