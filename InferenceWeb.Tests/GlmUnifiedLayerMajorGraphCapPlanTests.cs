// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace InferenceWeb.Tests;

/// <summary>CPU policy/ownership tests. Graph fit, kernels, canonical logits and
/// model performance require separate native and full-model qualification.</summary>
public sealed class GlmUnifiedLayerMajorGraphCapPlanTests
{
    private const long MiB = 1L << 20;

    [Theory]
    [InlineData(null, 0)]
    [InlineData("0", 0)]
    [InlineData("512", 512)]
    [InlineData("768", 768)]
    public void ExactCapValuesAreTheEntireOperatorContract(string? value, long expectedMiB)
        => Assert.Equal(expectedMiB * MiB, GlmUnifiedLayerMajorGraphCapPlan.Resolve(value));

    [Theory]
    [InlineData("")]
    [InlineData("1")]
    [InlineData("0512")]
    [InlineData("512 ")]
    [InlineData(" 768")]
    [InlineData("512.0")]
    [InlineData("1024")]
    [InlineData("-512")]
    public void InvalidOptInFailsBeforeNativeConstruction(string value)
        => Assert.Throws<ArgumentException>(() => GlmUnifiedLayerMajorGraphCapPlan.Resolve(value));

    [Fact]
    public void DisabledModeDoesNotProbeOrConfigureTheNewAbi()
        => Assert.Equal(0, GlmUnifiedLayerMajorGraphCapPlan.Certify(0,
            () => throw new EntryPointNotFoundException(), (_, _) => throw new EntryPointNotFoundException()));

    [Theory]
    [InlineData(512)]
    [InlineData(768)]
    public void ActualHandleConfigureMustFollowMatchingVersionAndAcceptExactBytes(long capMiB)
    {
        var calls = new List<string>();
        long cap = capMiB * MiB;
        Assert.Equal(cap, GlmUnifiedLayerMajorGraphCapPlan.Certify(cap,
            () => { calls.Add("version"); return 1; },
            (version, bytes) => { Assert.Equal(1, version); Assert.Equal(cap, bytes); calls.Add("configure actual handle"); return true; }));
        Assert.Equal(["version", "configure actual handle"], calls);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    public void IncompatibleVersionCannotConfigureOrGrantCredit(int version)
    {
        int configured = 0;
        Assert.Throws<NotSupportedException>(() => GlmUnifiedLayerMajorGraphCapPlan.Certify(512 * MiB,
            () => version, (_, _) => { ++configured; return true; }));
        Assert.Equal(0, configured);
    }

    [Fact]
    public void MissingCapabilityAndActualHandleRefusalCannotCertify()
    {
        Assert.Throws<EntryPointNotFoundException>(() => GlmUnifiedLayerMajorGraphCapPlan.Certify(512 * MiB,
            () => throw new EntryPointNotFoundException(), (_, _) => true));
        Assert.Throws<NotSupportedException>(() => GlmUnifiedLayerMajorGraphCapPlan.Certify(512 * MiB,
            () => 1, (_, _) => false));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedLayerMajorGraphCapPlan.Certify(513 * MiB,
            () => 1, (_, _) => true));
    }

    [Fact]
    public void InitialProfileRequiresEveryDeclaredGeometryAndNoDraft()
    {
        var g = Geometry();
        Assert.True(GlmUnifiedLayerMajorGraphCapPlan.ProfileGeometry(g, 2048));
        Assert.True(GlmUnifiedLayerMajorGraphCapPlan.ProfileGeometry(g with { Architecture = "glm5-next" }, 2048));
        Assert.False(GlmUnifiedLayerMajorGraphCapPlan.ProfileGeometry(g, 1024));
        foreach (var invalid in new[]
        {
            g with { Architecture = "glm-dsa" }, g with { Layers = 44 }, g with { RecurrentLayers = 33 },
            g with { Hidden = 4095 }, g with { Heads = 63 }, g with { Vocab = 154879 },
            g with { DeclaredContextTokens = 2047 }, g with { KvRow = 511 }, g with { IndexerDim = 127 },
            g with { KdaDim = 127 }, g with { ConvKernel = 3 }, g with { HyperConnections = 3 },
            g with { ExpertFf = 2047 }, g with { UsedExperts = 7 }, g with { ExpertCount = 287 },
            g with { Mtp = new GlmUnifiedMtpGeometry(0, 0, null!, 2048) }
        }) Assert.False(GlmUnifiedLayerMajorGraphCapPlan.ProfileGeometry(invalid, 2048));
    }

    [Fact]
    public void ProfileRejectsOtherRequestsAndRetainedPrefixMediaOrDraftOwners()
    {
        var options = Options();
        Assert.True(GlmUnifiedLayerMajorGraphCapPlan.ProfileOptions(options, false));
        Assert.False(GlmUnifiedLayerMajorGraphCapPlan.ProfileOptions(options, true));
        foreach (var invalid in new[]
        {
            options with { EnableGlmSelectedExperts = false }, options with { EnableGlmFlashMtp = true },
            options with { MaximumRunningRequests = 2 }, options with { RetainedConversationCount = 1 },
            options with { PrefixCheckpointCount = 1 }, options with { PrefillChunkTokens = 256 },
            options with { LayerMajorPrefillTokens = 512 }, options with { LayerMajorPrefillTokens = 0 },
            options with { EnableAdaptiveGlmPrefill = true, AdaptiveGlmPrefillChunkTokens = 1024 }
        }) Assert.False(GlmUnifiedLayerMajorGraphCapPlan.ProfileOptions(invalid, false));
    }

    [Theory]
    [InlineData(521, 0)]
    [InlineData(1024, 0)]
    [InlineData(1033, 100)]
    [InlineData(1872, 0)]
    [InlineData(2048, 0)]
    public void ActualSubmissionSelectsStrictStrategyBeforeAdmission(int submitted, int nPast)
    {
        Assert.True(Select(submitted, nPast));
        Assert.True(Eligible(new(nPast + submitted, 512, false, SubmittedTokens: submitted, StrictLayerMajor: true)));
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 0)]
    [InlineData(512, 0)]
    [InlineData(513, 0)]
    [InlineData(520, 0)]
    [InlineData(1025, 0)]
    [InlineData(1032, 0)]
    [InlineData(2048, 1)]
    [InlineData(1872, -1)]
    public void ShortTailAndInvalidHeadSelectBroadBeforeAnyStrictAdmission(int submitted, int nPast)
        => Assert.False(Select(submitted, nPast));

    [Fact]
    public void NativePathChunkContextOverridesAndTracingMustAllQualify()
    {
        Assert.False(GlmUnifiedLayerMajorGraphCapPlan.Select(1872, 256, 2048, 0, true, 1, false, false));
        Assert.False(GlmUnifiedLayerMajorGraphCapPlan.Select(1872, 512, 4096, 0, true, 1, false, false));
        Assert.False(GlmUnifiedLayerMajorGraphCapPlan.Select(1872, 512, 2048, 0, false, 1, false, false));
        Assert.False(GlmUnifiedLayerMajorGraphCapPlan.Select(1872, 512, 2048, 0, true, 2, false, false));
        Assert.False(GlmUnifiedLayerMajorGraphCapPlan.Select(1872, 512, 2048, 0, true, 1, true, false));
        Assert.False(GlmUnifiedLayerMajorGraphCapPlan.Select(1872, 512, 2048, 0, true, 1, false, true));
        var shape = Shape();
        foreach (var invalid in new[]
        {
            shape with { StrictLayerMajor = false }, shape with { SingleNativeGraph = true },
            shape with { Batched = true }, shape with { OutputRows = 2 }, shape with { Rows = 256 },
            shape with { SubmittedTokens = 513 }, shape with { ContextTokens = 2049 }
        }) Assert.False(Eligible(invalid));
        Assert.False(GlmUnifiedLayerMajorGraphCapPlan.Eligible(shape, 0, 2048, true));
        Assert.False(GlmUnifiedLayerMajorGraphCapPlan.Eligible(shape, 512 * MiB, 2048, false));
    }

    [Fact]
    public void ActualStrictIdentityIsIndependentOfDecodeOptInAndEveryStrictCallRechecksLiveChildren()
    {
        var strict = Shape();
        var broad = strict with { StrictLayerMajor = false };
        Assert.NotEqual(strict, broad);
        Assert.False(GlmUnifiedDecodeGraphPlan.Eligible(strict, 512, 1, true, false));
        Assert.True(GlmUnifiedLayerMajorGraphCapPlan.MustPrepare(strict, strict, true, true));
        Assert.True(GlmUnifiedLayerMajorGraphCapPlan.MustPrepare(broad, broad, false, true));
        Assert.True(GlmUnifiedLayerMajorGraphCapPlan.MustPrepare(broad, strict, false, true));
        Assert.False(GlmUnifiedLayerMajorGraphCapPlan.MustPrepare(broad, broad, false, false));
    }

    [Theory]
    [InlineData(512, 1152)]
    [InlineData(768, 1664)]
    public void TwoGenerationsKeepSeparateAllowanceWithoutChangingHostOrRawCudaHeadroom(long capMiB, long targetMiB)
    {
        var broad = Geometry().GraphPeak(2048, 512, false);
        var target = GlmUnifiedLayerMajorGraphCapPlan.Target(broad, new(32 * MiB, 400 * MiB), capMiB * MiB);
        Assert.Equal(broad.Host, target.Host);
        Assert.Equal(targetMiB * MiB, target.Device);
        Assert.True(target.Device < broad.Device);
        Assert.Equal(3068.306640625, broad.Device / (double)MiB);
        Assert.Equal((capMiB + 64 + 2000) * MiB,
            GlmUnifiedLayerMajorGraphCapPlan.Target(broad, new(32 * MiB, 2000 * MiB), capMiB * MiB).Device);
        Assert.Equal(broad.Host + 1,
            GlmUnifiedLayerMajorGraphCapPlan.Target(broad, new(broad.Host + 1, 0), capMiB * MiB).Host);
    }

    [Fact]
    public void OnlyExecutingDevicePromiseIsReplacedAndEveryOtherHostAndDevicePromiseSurvives()
    {
        var target = GlmUnifiedLayerMajorGraphCapPlan.Target(new(100, 4000 * MiB), default, 512 * MiB);
        var active = GlmUnifiedLayerMajorGraphCapPlan.PreservePromise(target, new(200, 4000 * MiB), true, true);
        Assert.Equal(new InferenceMemoryBytes(200, 1152 * MiB), active);
        var sibling = GlmUnifiedLayerMajorGraphCapPlan.PreservePromise(active, new(150, 3000 * MiB), false, true);
        Assert.Equal(new InferenceMemoryBytes(200, 3000 * MiB), sibling);
        Assert.Equal(new InferenceMemoryBytes(200, 4000 * MiB),
            GlmUnifiedLayerMajorGraphCapPlan.PreservePromise(target, new(200, 4000 * MiB), true, false));
    }

    [Fact]
    public void RealSessionDispatchReplacesOnlyActivePromiseAndRechecksLiveChildrenOnIdenticalStrictShape()
    {
        // Invoke the real session admission method using ledger owners and
        // metadata only. Every resize shrinks already owned credit, so this
        // test performs no hardware observation, native registration or I/O.
        const string host = UnifiedMemoryModelSession.HostPool, device = UnifiedMemoryModelSession.DevicePool;
        var budget = new MemoryBudget([new(host, 5000 * MiB), new(device, 10000 * MiB)]);
        using var joint = budget.Reserve([new(host, 1500 * MiB), new(device, 3500 * MiB)]);
        using var request = budget.Reserve([new(host, 10 * MiB), new(device, 0)]);
        using var peerOwner = budget.Reserve([new(host, 10 * MiB), new(device, 0)]);
        var active = new SequenceState("active", [1, 2, 3], 1, 16, SamplingConfig.Greedy) { MemoryEnvelope = request };
        var peer = new SequenceState("queued", [1, 2, 3], 1, 16, SamplingConfig.Greedy) { MemoryEnvelope = peerOwner };
        var sessionType = typeof(GlmUnifiedMemoryModelSession);
        var session = (GlmUnifiedMemoryModelSession)RuntimeHelpers.GetUninitializedObject(sessionType);
        void Set(string name, object value) => sessionType.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(session, value);
        Set("_options", Options() with { EnableAdaptiveGlmPrefill = true });
        Set("_geometry", Geometry()); Set("_context", 2048); Set("_prefill", 512);
        Set("_layerMajorGraphCapBytes", 512 * MiB); Set("_joint", joint);
        Set("<Budget>k__BackingField", budget); Set("_executingRequest", request); Set("_executingSequence", active);
        var callbackScope = (GgmlCacheBudgetScope)RuntimeHelpers.GetUninitializedObject(typeof(GgmlCacheBudgetScope));
        typeof(GgmlCacheBudgetScope).GetField("_gate", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(callbackScope, new object());
        Set("_native", callbackScope);
        var tableField = sessionType.GetField("_requestPrefill", BindingFlags.Instance | BindingFlags.NonPublic)!;
        object table = Activator.CreateInstance(tableField.FieldType)!;
        Set("_requestPrefill", table);
        Type stateType = sessionType.GetNestedType("PrefillState", BindingFlags.NonPublic)!;
        object AddState(SequenceState sequence, InferenceMemoryBytes peak)
        {
            object state = Activator.CreateInstance(stateType, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                null, [512], null)!;
            stateType.GetField("Peak", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(state, peak);
            tableField.FieldType.GetMethod("Add")!.Invoke(table, [sequence, state]);
            return state;
        }
        AddState(active, new(1000 * MiB, 3068 * MiB));
        object peerState = AddState(peer, new(400 * MiB, 2500 * MiB));
        using var retainedHost = joint.TryTake([new(host, 700 * MiB)])!;
        retainedHost.Commit();
        var prepare = sessionType.GetMethod("PrepareGraph", BindingFlags.Instance | BindingFlags.NonPublic)!;
        prepare.Invoke(session, [Shape()]);
        Assert.Equal(2500 * MiB, joint.OwnedCharges.Single(c => c.Pool == device).Bytes);
        Assert.Equal(1200 * MiB, joint.OwnedCharges.Single(c => c.Pool == host).Bytes); // Live700 + one500 replacement.

        // The same managed shape can replace a different native graph key.
        // A changed live floor and expired peer promise must be read again.
        stateType.GetField("Peak", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(peerState, default(InferenceMemoryBytes));
        Assert.True(joint.TryResizeTotal([new(host, 1200 * MiB), new(device, 3000 * MiB)]));
        using var live = joint.TryTake([new(device, 2000 * MiB)])!;
        live.Commit();
        prepare.Invoke(session, [Shape()]);
        Assert.Equal(2576 * MiB, joint.OwnedCharges.Single(c => c.Pool == device).Bytes);
        Assert.Equal(2000 * MiB, joint.LiveChildCharges.Single(c => c.Pool == device).Bytes);
        Assert.Equal(1200 * MiB, joint.OwnedCharges.Single(c => c.Pool == host).Bytes);
    }

    [Fact]
    public void PendingAndCommittedNativeChildrenCannotBeLentAwayAndPhysicalLimitsStillApply()
    {
        var budget = new MemoryBudget([new("host", 1000 * MiB), new("device", 4000 * MiB)]);
        using var owner = budget.Reserve([new("host", 200 * MiB), new("device", 3000 * MiB)]);
        using var old = owner.TryTake([new("device", 900 * MiB)])!;
        old.Commit();
        using var pending = owner.TryTake([new("device", 100 * MiB)])!;
        var target = GlmUnifiedLayerMajorGraphCapPlan.Target(new(200 * MiB, 3000 * MiB), new(0, 1000 * MiB), 512 * MiB);
        Assert.Equal(1576 * MiB, target.Device);
        Assert.True(owner.TryResizeTotal(Charges(target)));
        Assert.False(owner.TryResizeTotal(Charges(target with { Device = target.Device + 1 }),
            [new("host", 1000 * MiB), new("device", target.Device)]));
        Assert.Equal(target.Device, owner.OwnedCharges.Single(c => c.Pool == "device").Bytes);
        Assert.Equal(1000 * MiB, owner.LiveChildCharges.Single(c => c.Pool == "device").Bytes);
        using var replacement = owner.TryTake([new("device", 576 * MiB)])!;
        Assert.NotNull(replacement);
        Assert.Null(owner.TryTake([new("device", 1)]));
        replacement.Dispose();
        pending.Dispose();
        Assert.Equal(900 * MiB, owner.LiveChildCharges.Single(c => c.Pool == "device").Bytes);
        Assert.Equal(1476 * MiB, GlmUnifiedLayerMajorGraphCapPlan.Target(new(200 * MiB, 3000 * MiB),
            new(0, 900 * MiB), 512 * MiB).Device);
    }

    [Fact]
    public void TargetsRejectInvalidOrOverflowingOwnership()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedLayerMajorGraphCapPlan.Target(default, default, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedLayerMajorGraphCapPlan.Target(new(-1, 0), default, 512 * MiB));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedLayerMajorGraphCapPlan.Target(default, new(0, -1), 512 * MiB));
        Assert.Throws<OverflowException>(() => GlmUnifiedLayerMajorGraphCapPlan.Target(default, new(0, long.MaxValue), 512 * MiB));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(-2)]
    public void StrictHandleNeverExecutesLegacyFallbackAfterAnyNegativeNativeResult(int result)
    {
        int fallback = 0;
        Assert.Throws<InvalidOperationException>(() => GlmUnifiedLayerMajorGraphCapPlan.Complete(result, true,
            () => { ++fallback; return true; }));
        Assert.Equal(0, fallback);
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(1, true)]
    [InlineData(2, false)]
    public void NativeSuccessAndFailurePublishTheirOwnResultWithoutRetry(int result, bool expected)
    {
        Assert.Equal(expected, GlmUnifiedLayerMajorGraphCapPlan.Complete(result, true,
            () => throw new InvalidOperationException("unexpected fallback")));
    }

    [Fact]
    public void OrdinaryLegacyFallbackRemainsAvailableWhenProfileIsOff()
    {
        int fallback = 0;
        Assert.True(GlmUnifiedLayerMajorGraphCapPlan.Complete(-1, false, () => { ++fallback; return true; }));
        Assert.Equal(1, fallback);
    }

    private static GlmUnifiedGraphShape Shape() => new(1872, 512, false, SubmittedTokens: 1872, StrictLayerMajor: true);
    private static bool Eligible(GlmUnifiedGraphShape shape)
        => GlmUnifiedLayerMajorGraphCapPlan.Eligible(shape, 512 * MiB, 2048, true);
    private static bool Select(int submitted, int nPast)
        => GlmUnifiedLayerMajorGraphCapPlan.Select(submitted, 512, 2048, nPast, true, 1, false, false);
    private static MemoryCharge[] Charges(InferenceMemoryBytes bytes) => [new("host", bytes.Host), new("device", bytes.Device)];
    private static UnifiedMemoryModelOptions Options() => new()
    {
        ContextTokens = 2048, MaximumRunningRequests = 1, PrefixCheckpointCount = 0,
        RetainedConversationCount = 0, EnableGlmSelectedExperts = true
    };
    private static GlmUnifiedMemoryGeometry Geometry() => new()
    {
        Architecture = "glm5next", Layers = 45, DeclaredContextTokens = 32768, RecurrentLayers = 34,
        Hidden = 4096, Heads = 64, Vocab = 154880, KvRow = 512, IndexerDim = 128,
        KdaDim = 128, ConvKernel = 4, HyperConnections = 4, ExpertFf = 2048, UsedExperts = 8, ExpertCount = 288
    };
}
