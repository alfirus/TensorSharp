using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.Server.Hosting;

/// <summary>Completed scheduled requests, not a claim that a prefix was retained.
/// The actual reuse count belongs to each subsequent request's telemetry.</summary>
public sealed class StartupWarmupStatus
{
    public sealed record Stage(bool Thinking, bool Completed, double ElapsedMilliseconds);

    public StartupWarmupStatus(IEnumerable<Stage> stages, bool prefixRetentionRequested)
    {
        var snapshot = stages?.ToArray() ?? throw new ArgumentNullException(nameof(stages));
        Stages = Array.AsReadOnly(snapshot);
        PrefixRetentionRequested = prefixRetentionRequested;
        Completed = snapshot.Length == 2 && snapshot.All(s => s.Completed)
            && snapshot.Any(s => s.Thinking) && snapshot.Any(s => !s.Thinking);
        ElapsedMilliseconds = snapshot.Sum(s => s.ElapsedMilliseconds);
    }

    public string Mode => "scheduled";
    public bool Completed { get; }
    public bool PrefixRetentionRequested { get; }
    public double ElapsedMilliseconds { get; }
    public IReadOnlyList<Stage> Stages { get; }
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
