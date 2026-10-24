// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using TensorSharp.Memory.Planning;

namespace TensorSharp.Models;

/// <summary>Only changed admission shapes/ownership are printed. Target and
/// draft histories are separate so their normal alternation is not log spam.</summary>
internal sealed class GlmUnifiedMtpDiagnostics
{
    private readonly Dictionary<string, GlmUnifiedMtpMemorySample> _last = new(StringComparer.Ordinal);

    internal static bool IsEnabled(string? value) => value == "1";

    internal string? Record(string boundary, GlmUnifiedMtpMemorySample sample, int targetPosition, bool force = false)
    {
        if (!force && _last.TryGetValue(boundary, out var previous) && previous == sample) return null;
        _last[boundary] = sample;
        return FormattableString.Invariant($"[GLM-MTP-MEMORY] boundary={boundary} request={sample.RequestId} context={sample.Context} rows={sample.Rows} output_rows={sample.OutputRows} target_position={targetPosition} kind5_owned_host={sample.TargetOwned.Host} kind5_owned_device={sample.TargetOwned.Device} kind5_live_host={sample.TargetLive.Host} kind5_live_device={sample.TargetLive.Device} kind7_owned_host={sample.DraftOwned.Host} kind7_owned_device={sample.DraftOwned.Device} kind7_live_host={sample.DraftLive.Host} kind7_live_device={sample.DraftLive.Device} snapshot_forecast_device={sample.SnapshotBytes} snapshot_live_device={sample.SnapshotLiveBytes} snapshot_missing_device={Math.Max(0, sample.SnapshotBytes - sample.SnapshotLiveBytes)} expert_budget_device={sample.ExpertBytes} unit=bytes");
    }
}

internal readonly record struct GlmUnifiedMtpMemorySample(string RequestId, int Context, int Rows, int OutputRows,
    InferenceMemoryBytes TargetOwned, InferenceMemoryBytes TargetLive,
    InferenceMemoryBytes DraftOwned, InferenceMemoryBytes DraftLive, long SnapshotBytes, long ExpertBytes,
    long SnapshotLiveBytes = 0);

public sealed partial class GlmUnifiedMemoryModelSession
{
    private readonly GlmUnifiedMtpDiagnostics? _mtpDiagnostics = GlmUnifiedMtpDiagnostics.IsEnabled(
        Environment.GetEnvironmentVariable("TS_GLM_FLASH_MTP_DIAGNOSTICS")) ? new() : null;

    private void LogMtpMemory(string boundary, int context, int rows, int outputRows, bool force = false, string? requestId = null)
    {
        if (_mtpDiagnostics == null || !_options.EnableGlmFlashMtp || _speculativeEnvelope == null) return;
        var sample = new GlmUnifiedMtpMemorySample(requestId ?? _executingSequence?.RequestId ?? "-", context, rows, outputRows,
            Bytes(_joint!.OwnedCharges), Bytes(_joint.LiveChildCharges),
            Bytes(_speculativeEnvelope.OwnedCharges), Bytes(_speculativeEnvelope.LiveChildCharges),
            SpeculativeSnapshotBytes, _experts?.BudgetBytes ?? 0, _model!.UnifiedSnapshotBytes());
        string? line = _mtpDiagnostics.Record(boundary, sample, _model?.CacheSeqLen ?? 0, force);
        if (line != null) Console.WriteLine(line);
    }
}
