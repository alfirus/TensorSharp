// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>Actual session forecasts and ledger transactions with synthetic
/// model metadata. No model execution, native backend or GPU qualification.</summary>
public sealed class UnifiedMemoryProgressiveReservationTests
{
    private const string Host = UnifiedMemoryModelSession.HostPool;
    private const string Device = UnifiedMemoryModelSession.DevicePool;

    [Fact]
    public void MaterializationForecastOwnsCompleteStateWithoutAssumingPrefixOrWidePrefill()
    {
        var model = MetadataModel(); var session = UninitializedSession(model);
        var request = new SequenceState("cold", Enumerable.Repeat(1, 1479).ToArray(), 20000, 256, SamplingConfig.Greedy);
        var materialization = (InferenceMemoryBytes)typeof(UnifiedMemoryModelSession)
            .GetMethod("MaterializationPeak", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(session, [request])!;
        Assert.Equal(model.GetUnifiedMemoryRequestPeak(1479, 1, checkpointCopies: 7,
            independentHolder: true, requestId: "cold"), materialization);
        Assert.True(materialization.Device < Peak(session, request).Device);
        Assert.True(materialization.Host < Peak(session, request).Host);
        Assert.Equal(0, request.NumComputedTokens); Assert.Equal(0, request.PrefixCacheReusedTokens);
        Assert.Equal((1479, 512), Shape(session, request));
    }

    [Fact]
    public void TwentyThousandOutputAllowanceDoesNotReserveUnusedGenerationUpfront()
    {
        var model = MetadataModel();
        var session = UninitializedSession(model);
        var prompt = Enumerable.Repeat(1, 1479).ToArray();
        var shortReply = new SequenceState("short", prompt, 160, 256, SamplingConfig.Greedy);
        var longReply = new SequenceState("long", prompt, 20000, 256, SamplingConfig.Greedy);
        Assert.Equal(Peak(session, shortReply), Peak(session, longReply));
        Assert.Equal((1479, 512), Shape(session, longReply));
        // This is the exact full-output forecast previously used for admission.
        var allOutput = model.GetUnifiedMemoryRequestPeak(1479 + 20000, 512,
            checkpointCopies: 7, independentHolder: true, requestId: "long");
        Assert.True(allOutput.Host > Peak(session, longReply).Host);
        Assert.True(allOutput.Device > Peak(session, longReply).Device);
    }

    [Fact]
    public void ForecastUsesRetainedHolderCapacityAndActiveFieldsInsteadOfFreshInitialCapacity()
    {
        var model = MetadataModel();
        AddHolder(model, "retained", 8192);
        AddHolder(model, "active", 8192);
        Set(model, "_activeFusedKey", "active");
        Assert.Equal(1479, model.GetUnifiedMemoryCapacity(1479, requestId: "new"));
        Assert.Equal(8192, model.GetUnifiedMemoryCapacity(1479, requestId: "retained"));
        Assert.Equal(32768, model.GetUnifiedMemoryCapacity(1479, requestId: "active"));
        Assert.True(model.GetUnifiedMemoryRequestPeak(1479, 1, independentHolder: true, requestId: "retained").Device
            > model.GetUnifiedMemoryRequestPeak(1479, 1, independentHolder: true, requestId: "new").Device);
    }

    [Fact]
    public void ForecastGrowsBeforeTheNextBoundaryTokenAndReflectsActualChunkGrowth()
    {
        var model = MetadataModel();
        AddHolder(model, "growing", 1024);
        Assert.Equal(2048, model.GetUnifiedMemoryCapacity(1479, requestId: "growing"));
        AddHolder(model, "growing", 2048);
        var session = UninitializedSession(model);
        var request = new SequenceState("growing", Enumerable.Repeat(1, 1479).ToArray(), 20000, 256, SamplingConfig.Greedy);
        typeof(SequenceState).GetProperty(nameof(SequenceState.NumComputedTokens))!.SetValue(request, 2048);
        Assert.Equal((4096, 1), Shape(session, request));
        Assert.Equal(model.GetUnifiedMemoryRequestPeak(2049, 1, checkpointCopies: 0,
            requestId: "growing"), Peak(session, request));
    }

    [Fact]
    public void DecodeCapacityGrowthDoesNotGrowSevenImmutablePromptCheckpoints()
    {
        var model = MetadataModel(); var session = UninitializedSession(model);
        AddHolder(model, "growing", 1536);
        var request = new SequenceState("growing", Enumerable.Repeat(1, 1479).ToArray(), 20000, 256, SamplingConfig.Greedy);
        SetComputed(request, 1478);
        Assert.Equal(model.GetUnifiedMemoryRequestPeak(1479, 1, checkpointCopies: 7,
            requestId: request.RequestId), Peak(session, request)); // Prompt-end publication still needs its copy.
        SetComputed(request, 1479);
        Assert.Equal(model.GetUnifiedMemoryRequestPeak(1480, 1, checkpointCopies: 0,
            requestId: request.RequestId), Peak(session, request)); // Publication has completed.
        SetComputed(request, 1536);
        var actual = Peak(session, request);
        var manufactured = model.GetUnifiedMemoryRequestPeak(1537, 1, checkpointCopies: 7, requestId: request.RequestId);
        Assert.True(manufactured.Host > actual.Host);
        Assert.Equal(manufactured.Device, actual.Device); // Device growth allowance remains unchanged.
        Assert.Equal((3072, 1), Shape(session, request));
    }

    [Fact]
    public void CompletedCheckpointChildrenRemainChargedWhileOnlyLiveStateGrowthIsReserved()
    {
        var model = MetadataModel(); var session = UninitializedSession(model);
        AddHolder(model, "growing", 1536);
        var request = new SequenceState("growing", Enumerable.Repeat(1, 1479).ToArray(), 20000, 256, SamplingConfig.Greedy);
        SetComputed(request, 1478);
        var beforePublication = Peak(session, request);
        SetComputed(request, 1479);
        var beforeGrowth = Peak(session, request);
        SetComputed(request, 1536);
        var afterGrowth = Peak(session, request);
        long fixedCopies = beforePublication.Host - beforeGrowth.Host;
        Assert.True(fixedCopies > 0);
        var budget = new MemoryBudget([new(Host, afterGrowth.Host + fixedCopies), new(Device, afterGrowth.Device)]);
        using var joint = budget.Reserve([new(Host, 0), new(Device, 0)]);
        using var envelope = budget.Reserve([new(Host, beforePublication.Host), new(Device, beforeGrowth.Device)]);
        using var checkpoint = envelope.TryTake([new(Host, fixedCopies)])!; checkpoint.Commit();
        using var nativeState = envelope.TryTake([new(Device, 32)])!; nativeState.Commit();
        request.MemoryEnvelope = envelope;
        var protectedGrowth = Peak(session, request);
        Assert.Equal(afterGrowth.Host + fixedCopies, protectedGrowth.Host);
        Assert.Equal(afterGrowth.Device, protectedGrowth.Device);
        // The prompt's real copy children retain lifetime ownership under this
        // root. Closing unused reservation does not free these physical bytes.
        Assert.True(envelope.TryResizeTotal([new(Host, beforeGrowth.Host), new(Device, beforeGrowth.Device)]));
        var transaction = TransactionSession(budget, joint, new(0, 0), 1536);
        var obsolete = model.GetUnifiedMemoryRequestPeak(1537, 1, checkpointCopies: 7, requestId: request.RequestId);
        var before = budget.Snapshot().ToArray();
        Assert.Throws<MemoryPressureException>(() => transaction.ResizeExecutionEnvelopes(envelope,
            [new(Host, obsolete.Host), new(Device, obsolete.Device)], new(0, 0), 3072));
        Assert.Equal(before, budget.Snapshot());
        transaction.ResizeExecutionEnvelopes(envelope,
            [new(Host, protectedGrowth.Host), new(Device, protectedGrowth.Device)], new(0, 0), 3072);
        Assert.Equal(fixedCopies, envelope.LiveChildCharges.Single(c => c.Pool == Host).Bytes);
        Assert.Equal(fixedCopies, budget.Snapshot().Single(c => c.Pool == Host).Committed);
        Assert.Equal(32, budget.Snapshot().Single(c => c.Pool == Device).Committed);
        envelope.Dispose();
        Assert.Equal(fixedCopies, budget.Snapshot().Single(c => c.Pool == Host).Committed);
        Assert.Equal(0, budget.Snapshot().Single(c => c.Pool == Host).Reserved);
        checkpoint.Dispose(); nativeState.Dispose();
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Committed + p.Reserved));
    }

    [Fact]
    public void FixedChildrenLargerThanDecodePeakStillLeaveFullCreditForTheNextPhysicalGrowth()
    {
        var model = MetadataModel(); var session = UninitializedSession(model);
        AddHolder(model, "large-copies", 1536);
        var request = new SequenceState("large-copies", Enumerable.Repeat(1, 1479).ToArray(), 20000, 256, SamplingConfig.Greedy);
        SetComputed(request, 1536);
        var live = Peak(session, request);
        long fixedBytes = checked(4 * live.Host);
        var budget = new MemoryBudget([new(Host, fixedBytes + live.Host), new(Device, live.Device)]);
        using var joint = budget.Reserve([new(Host, 0), new(Device, 0)]);
        using var envelope = budget.Reserve([new(Host, fixedBytes), new(Device, 0)]);
        request.MemoryEnvelope = envelope;
        using var copies = envelope.TryTake([new(Host, fixedBytes)])!; copies.Commit();
        var target = Peak(session, request);
        Assert.Equal(live.Host + fixedBytes, target.Host);
        var transaction = TransactionSession(budget, joint, new(0, 0), 1536);
        transaction.ResizeExecutionEnvelopes(envelope,
            [new(Host, target.Host), new(Device, target.Device)], new(0, 0), 3072);
        Assert.Equal(live.Host, envelope.Charges.Single(c => c.Pool == Host).Bytes);
        using var next = envelope.TryTake([new(Host, live.Host)])!;
        Assert.NotNull(next); next.Commit();
        envelope.Dispose();
        Assert.Equal(fixedBytes + live.Host, budget.Snapshot().Single(c => c.Pool == Host).Committed);
        next.Dispose(); copies.Dispose();
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    private static void SetComputed(SequenceState request, int computed)
        => typeof(SequenceState).GetProperty(nameof(SequenceState.NumComputedTokens))!.SetValue(request, computed);

    [Fact]
    public void RefusedCombinedGrowthPreservesBothOwnersAndJointMetadata()
    {
        var budget = new MemoryBudget([new(Host, 100), new(Device, 100)]);
        using var joint = budget.Reserve([new(Host, 20), new(Device, 20)]);
        using var request = budget.Reserve([new(Host, 20), new(Device, 20)]);
        using var state = request.TryTake([new(Host, 10), new(Device, 10)])!;
        state.Commit();
        var session = TransactionSession(budget, joint, new(20, 20), 512);
        Assert.Throws<MemoryPressureException>(() => session.ResizeExecutionEnvelopes(request,
            [new(Host, 80), new(Device, 80)], new(50, 50), 4096));
        Assert.Equal(new InferenceMemoryBytes(20, 20), Get<InferenceMemoryBytes>(session, "_jointTarget"));
        Assert.Equal(512, Get<int>(session, "_jointContextHighWater"));
        Assert.Equal(20, joint.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(20, request.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(10, state.OwnedCharges.Single(c => c.Pool == Device).Bytes);
    }

    [Fact]
    public void PhaseCreditCanFundJointGrowthWithoutRechargingWeightsOrHostMargin()
    {
        var budget = new MemoryBudget([new(Host, 100), new(Device, 100)]);
        using var joint = budget.Reserve([new(Host, 30), new(Device, 30)]);
        using var request = budget.Reserve([new(Host, 70), new(Device, 70)]);
        using var state = request.TryTake([new(Host, 50), new(Device, 50)])!;
        state.Commit();
        MemoryCharge[] peak = [new(Host, 40), new(Device, 40)];
        Assert.Equal(new InferenceMemoryBytes(0, 0),
            UnifiedMemoryModelSession.ExecutionEnvelopeGrowth(request, peak, joint, new(50, 50)));
        var session = TransactionSession(budget, joint, new(30, 30), 512);
        session.ResizeExecutionEnvelopes(request, peak, new(50, 50), 4096);
        Assert.Equal(new InferenceMemoryBytes(50, 50), Get<InferenceMemoryBytes>(session, "_jointTarget"));
        Assert.Equal(4096, Get<int>(session, "_jointContextHighWater"));
        Assert.Equal(50, state.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(50, request.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(50, joint.OwnedCharges.Single(c => c.Pool == Device).Bytes);
    }

    private static Qwen4ExpModel MetadataModel()
    {
        var model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        Set(model, "_maxContextLength", 32768);
        Set(model, "_initialKvCacheCapacity", 512);
        Set(model, "_kvCacheCapacity", 32768); // A different bound primary.
        var dtype = Field(model, "_kvCacheDtype");
        dtype.SetValue(model, Enum.Parse(dtype.FieldType, "F16"));
        Set(model, "_unifiedGeometry", new Qwen4ExpUnifiedGeometry
        {
            Layers = 1, AttentionLayers = 1, QsaLayers = 1, Hidden = 32, HyperHidden = 128,
            Heads = 1, KvHeads = 1, HeadDim = 32, Vocab = 16, ConvDim = 64, ConvKernel = 2,
            StateDim = 4, ValueHeads = 1, ExpertFf = 32, SharedFf = 32, UsedExperts = 1,
            ExpertCount = 2, IndexerHeads = 1, IndexerDim = 8,
            ExpertProjectionBytes = [[256, 256, 256]], ExpertProjectionCudaTailBytes = [[0, 0, 0]]
        });
        return model;
    }

    private static void AddHolder(Qwen4ExpModel model, string key, int capacity)
    {
        var field = Field(model, "_fusedHolders");
        var holders = (IDictionary)(field.GetValue(model) ?? Activator.CreateInstance(field.FieldType)!);
        var holderType = field.FieldType.GenericTypeArguments[1];
        var holder = RuntimeHelpers.GetUninitializedObject(holderType);
        holderType.GetField("KvCapacity")!.SetValue(holder, capacity);
        holders[key] = holder; field.SetValue(model, holders);
    }

    private static UnifiedMemoryModelSession UninitializedSession(Qwen4ExpModel model)
    {
        var session = (UnifiedMemoryModelSession)RuntimeHelpers.GetUninitializedObject(typeof(UnifiedMemoryModelSession));
        Set(session, "_model", model); Set(session, "_prefillTokens", 512); Set(session, "_checkpointCopies", 7);
        return session;
    }
    private static UnifiedMemoryModelSession TransactionSession(MemoryBudget budget, BudgetReservation joint,
        InferenceMemoryBytes target, int context)
    {
        var session = (UnifiedMemoryModelSession)RuntimeHelpers.GetUninitializedObject(typeof(UnifiedMemoryModelSession));
        Set(session, "<Budget>k__BackingField", budget); Set(session, "_jointEnvelope", joint);
        Set(session, "_jointTarget", target); Set(session, "_jointContextHighWater", context);
        return session;
    }
    private static InferenceMemoryBytes Peak(UnifiedMemoryModelSession session, SequenceState request)
        => (InferenceMemoryBytes)typeof(UnifiedMemoryModelSession).GetMethod("RequestPeak", BindingFlags.NonPublic | BindingFlags.Instance)!
            .Invoke(session, [request])!;
    private static (int, int) Shape(UnifiedMemoryModelSession session, SequenceState request)
        => ((int, int))typeof(UnifiedMemoryModelSession).GetMethod("RequestShape", BindingFlags.NonPublic | BindingFlags.Instance)!
            .Invoke(session, [request])!;
    private static FieldInfo Field(object instance, string name) => instance.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!;
    private static void Set(object instance, string name, object value) => Field(instance, name).SetValue(instance, value);
    private static T Get<T>(object instance, string name) => (T)Field(instance, name).GetValue(instance)!;
}
