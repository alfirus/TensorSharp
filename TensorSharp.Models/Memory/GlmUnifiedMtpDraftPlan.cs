// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using TensorSharp.Memory.Planning;

namespace TensorSharp.Models;

internal enum GlmUnifiedMtpGraphPurpose { DraftStep, CatchUp, Snapshot }

/// <summary>A live snapshot is already part of the speculative envelope's
/// children. Reserve only its missing forecast; never infer snapshot residency
/// from aggregate graph/snapshot ownership or from snapshot validity.</summary>
internal static class GlmUnifiedMtpDraftPlan
{
    internal static bool ResolveShapePlan(string? option, bool nativeCapable)
        => option != "1" ? false : nativeCapable ? true
            : throw new NotSupportedException("TS_GLM_FLASH_MTP_SHAPE_PLAN=1 requires the matching shape-plan native capability.");

    internal static bool ResolvePrefixCommit(string? option, bool nativeCapable)
        => option != "1" ? false : nativeCapable ? true
            : throw new NotSupportedException("TS_GLM_FLASH_MTP_PREFIX_COMMIT=1 requires the matching native prefix-state capability.");

    internal static long SnapshotForecast(GlmUnifiedMemoryGeometry geometry)
        => checked((geometry.Mtp?.PrefixStateCommit == true ? 2L : 1L) *
            (geometry.FixedStateBytes + (64L << 10) * geometry.RecurrentLayers));

    internal static InferenceMemoryBytes Workspace(GlmUnifiedMemoryGeometry geometry, int context, int rows,
        GlmUnifiedMtpGraphPurpose purpose)
    {
        if (purpose is not (GlmUnifiedMtpGraphPurpose.DraftStep or GlmUnifiedMtpGraphPurpose.CatchUp or GlmUnifiedMtpGraphPurpose.Snapshot))
            throw new ArgumentOutOfRangeException(nameof(purpose));
        if (purpose != GlmUnifiedMtpGraphPurpose.CatchUp && rows != 1)
            throw new ArgumentOutOfRangeException(nameof(rows));
        // Native kind-3 file staging stays on the executing request even when
        // the selected experts belong to a draft graph. Its optional expansion
        // is already in that request's forecast, not this kind-7 owner.
        var draft = geometry with { Layers = 1, RecurrentLayers = 0, HyperConnections = 1,
            Mtp = null, ExpandedDecodeStaging = false, ExpandedPrefillRingStaging = false };
        var workspace = draft.Workspace(context, rows, 1);
        if (geometry.Mtp?.ShapeSpecificPlan != true || purpose != GlmUnifiedMtpGraphPurpose.CatchUp) return workspace;
        // The capability certifies that catch-up returns after attention KV:
        // no routed/shared FFN, output hidden or head. Keep the existing broad
        // attention/embedding forecast, metadata allowance and live children.
        long absent = checked(4L * rows * 8L * geometry.ExpertFf * geometry.UsedExperts + 4L * geometry.Vocab);
        return new(checked(workspace.Host - absent), checked(workspace.Device - absent));
    }

    internal static InferenceMemoryBytes Target(InferenceMemoryBytes workspace, InferenceMemoryBytes live,
        long snapshotForecast, long snapshotLive)
    {
        if (workspace.Host < 0 || workspace.Device < 0 || live.Host < 0 || live.Device < 0
            || snapshotForecast < 0 || snapshotLive < 0 || snapshotLive > live.Device)
            throw new InvalidOperationException("Flash MTP snapshot accounting is inconsistent with its live speculative owner.");
        long missing = Math.Max(0, snapshotForecast - snapshotLive);
        return new(checked(live.Host + workspace.Host), checked(live.Device + missing + workspace.Device));
    }
}
