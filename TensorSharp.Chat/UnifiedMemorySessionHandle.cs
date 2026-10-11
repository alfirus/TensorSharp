// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Server;

/// <summary>A lifecycle-owned session, with a CPU-only test seam for release failures.</summary>
internal sealed class UnifiedMemorySessionHandle
{
    private readonly Func<SchedulerConfig, SchedulerConfig> _configure;
    private readonly Func<string> _featureReport;
    private readonly Action _dispose;

    internal UnifiedMemorySessionHandle(ModelBase model,
        Func<SchedulerConfig, SchedulerConfig> configure, Func<string> featureReport, Action dispose)
    {
        Model = model;
        _configure = configure ?? throw new ArgumentNullException(nameof(configure));
        _featureReport = featureReport ?? throw new ArgumentNullException(nameof(featureReport));
        _dispose = dispose ?? throw new ArgumentNullException(nameof(dispose));
    }

    internal static UnifiedMemorySessionHandle Wrap(UnifiedMemoryModelSession session)
        => new(session.Model, session.ConfigureScheduler, () => session.FeatureReport, session.Dispose);

    internal static UnifiedMemorySessionHandle CreateOwned(string path, string projector, BackendType backend)
    {
        var session = UnifiedMemoryModelSession.Create(path, projector, backend);
        try { return Wrap(session); }
        catch (Exception creation)
        {
            // Creating this managed adapter happens after the model owns real
            // buffers. Even failure here must not orphan the original session.
            try { session.Dispose(); }
            catch (Exception cleanup)
            { throw new UnifiedMemoryModelLoadException(session, new AggregateException(creation, cleanup)); }
            throw;
        }
    }

    internal ModelBase Model { get; }
    internal SchedulerConfig ConfigureScheduler(SchedulerConfig config) => _configure(config);
    internal string FeatureReport => _featureReport();
    internal void Dispose() => _dispose();
}
