// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models;

public sealed partial class GlmUnifiedMemoryModelSession
{
    internal static void ValidateFlashMtpLoadOptions(UnifiedMemoryModelOptions options,
        SpeculationOptions speculation, string? projector, string? optIn)
    {
        _ = ResolveFlashMtpSpeculation(speculation);
        if (!options.EnableGlmFlashMtp || optIn != "1" || !options.EnableGlmSelectedExperts
            || !options.EnableAdaptiveGlmPrefill || options.MaximumRunningRequests != 1
            || options.PrefixCheckpointCount != 0 || options.RetainedConversationCount != 0
            || options.ContextTokens > 2048
            || (!string.IsNullOrWhiteSpace(projector) && projector != "none"))
            throw new NotSupportedException("Experimental Flash MTP requires its explicit opt-in, one primary request, selected experts, adaptive admission, bounded context, no retained prefixes and no image tower.");
    }

    internal static SpeculationOptions ResolveFlashMtpSpeculation(SpeculationOptions requested)
    {
        if (!requested.Enabled || (requested.SpeculatorName != SpeculatorRegistry.Auto
            && requested.SpeculatorName != SpeculatorRegistry.DraftHead)
            || (requested.MaxDraftTokensExplicit && requested.MaxDraftTokens is < 1 or > 2))
            throw new NotSupportedException("Experimental unified Flash MTP requires --spec with auto/draft-head and at most two explicitly requested draft tokens.");
        return requested with { SpeculatorName = SpeculatorRegistry.DraftHead,
            MaxDraftTokens = requested.MaxDraftTokensExplicit ? requested.MaxDraftTokens : 2,
            MaxDraftTokensExplicit = true };
    }

    private SchedulerConfig ConfigureFlashMtpScheduler(SchedulerConfig requested)
    {
        if (requested.EnablePrefixCaching)
            throw new NotSupportedException("Experimental unified Flash MTP requires --no-prefix-cache.");
        var speculation = ResolveFlashMtpSpeculation(requested.Speculation);
        int block = requested.BlockSize > 0 ? requested.BlockSize : SchedulerConfig.DefaultBlockSize;
        return new SchedulerConfig
        {
            BlockSize = block, NumBlocks = checked((_context + block - 1) / block),
            MaxNumRunningSequences = 1, MaxNumBatchedTokens = _forwardTokens,
            MaxPrefillChunkSize = _forwardTokens, SoloPrefillChunkSize = _forwardTokens,
            PrefillChunkTokenLimit = _forwardTokens, EnablePrefixCaching = false,
            StopRepetition = requested.StopRepetition, DecodeQuantumTokens = requested.DecodeQuantumTokens,
            Speculation = speculation,
            ExecutionOptionsOverride = new ExecutionOptions
            {
                BatchedPathDisabled = true, PerSeqFusedEnabled = false, BatchedFusedDecodeEnabled = false,
                RetainedFusedCacheBudget = 0, PrefixCheckpointBudget = 0, KvHolderPoolMax = 0
            },
            MemoryAdmission = new RequestMemoryAdmission(Budget, EstimateRequest)
            {
                ReestimateAfterPhysicalCheck = true, AllowSpeculativeSerialExecution = true,
                BeforeAdmissionWithRunning = PrepareAdmission, AfterPrefill = CompletePrefill,
                PrefillTokenLimit = ForwardPrefillLimit, AfterSequenceMemoryRelease = RecoverReleasedCapacity,
                EnterSerialExecution = EnterSerial
            }
        };
    }

    /// <summary>Before each draft graph or snapshot dispatch, preserve every
    /// live kind-7 child and one replacement graph plus any missing KDA copy.
    /// No estimated refund is credited until native owners have released.</summary>
    private void PrepareDraftGraph(int context, int rows, GlmUnifiedMtpGraphPurpose purpose = GlmUnifiedMtpGraphPurpose.DraftStep)
    {
        if (_native!.CallbackError is { } callback)
            throw new InvalidOperationException("Flash MTP draft admission is disabled after an accounting failure.", callback);
        if (_speculativeEnvelope == null || !_options.EnableGlmFlashMtp || context <= 0 || context > _context
            || rows <= 0 || rows > _prefill)
            throw new MemoryPressureException("Flash MTP exceeds its admitted draft geometry.");
        var workspace = GlmUnifiedMtpDraftPlan.Workspace(_geometry, PaddedContext(context), rows, purpose);
        InferenceMemoryBytes Target()
        {
            var live = Bytes(_speculativeEnvelope.LiveChildCharges);
            return GlmUnifiedMtpDraftPlan.Target(workspace, live, SpeculativeSnapshotBytes,
                _model!.UnifiedSnapshotBytes());
        }
        var target = Target();
        bool Admit()
        {
            var owned = Bytes(_speculativeEnvelope.OwnedCharges);
            var growth = Charges(new(checked(target.Host - owned.Host), checked(target.Device - owned.Device)));
            var physical = RepayHostExpertCapacity(ObservePhysicalConstraints(), growth);
            physical = RepayExpertCapacity(physical, growth, _expertMinimumBytes);
            return _speculativeEnvelope.TryResizeTotal(Charges(target), physical.Limits);
        }
        if (!Admit())
        {
            _model!.TrimUnifiedGraphs(); _lastGraphShape = null;
            target = Target();
            if (!Admit()) throw new MemoryPressureException("Flash MTP draft/snapshot cannot fit without violating live target or expert ownership.");
        }
        bool kvOnly = _geometry.Mtp?.ShapeSpecificPlan == true && purpose == GlmUnifiedMtpGraphPurpose.CatchUp;
        LogMtpMemory(kvOnly ? "draft-kv-admitted" : "draft-admitted", PaddedContext(context), rows, kvOnly ? 0 : 1);
    }
}
