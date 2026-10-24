// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;
using TensorSharp.Server;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Real ledger and exact planning/capability contracts. No device,
/// full-model math, capture or performance qualification is implied.</summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class GlmUnifiedLayerMajorDeviceBatonPlanTests
{
    private const long MiB = 1L << 20;
    private const string Host = UnifiedMemoryModelSession.HostPool, Device = UnifiedMemoryModelSession.DevicePool;
    private static MemoryCharge[] Charges(long host,long device) => [new(Host,host),new(Device,device)];
    private static GlmUnifiedMemoryGeometry Geometry(bool enabled = true) => new() {
        Architecture="glm5next", Layers=45,RecurrentLayers=34,DeclaredContextTokens=2048,Hidden=4096,
        HyperConnections=4,Heads=64,Vocab=154880,KvRow=512,IndexerDim=128,KdaDim=128,ConvKernel=4,
        ExpertFf=2048,UsedExperts=8,ExpertCount=288,HostTokenEmbedding=true,LayerMajorDeviceBaton=enabled
    };
    private static UnifiedMemoryModelOptions Options() => new() {
        ContextTokens=2048,PrefillChunkTokens=512,AdaptiveGlmPrefillChunkTokens=512,LayerMajorPrefillTokens=8192,
        MaximumRunningRequests=1,RetainedConversationCount=0,PrefixCheckpointCount=0,
        EnableGlmSelectedExperts=true,EnableAdaptiveGlmPrefill=true,EnableExpertCacheCapacityGrowth=true
    };
    [Theory]
    [InlineData("2048")] [InlineData("02048")]
    public void AutomaticContextUsesTheExistingConfiguredIntegerParser(string configured)
    {
        var automatic = Options() with { ContextTokens = 0 };
        var resolved = GlmUnifiedLayerMajorDeviceBatonPlan.ResolveContextOptions(automatic, configured);
        Assert.Equal(automatic with { ContextTokens = 2048 }, resolved);
        Assert.Equal(0, automatic.ContextTokens);
        Assert.True(GlmUnifiedLayerMajorDeviceBatonPlan.ProfileOptions(resolved, false));
    }
    [Theory]
    [InlineData(null)] [InlineData("")] [InlineData("4096")] [InlineData("malformed")] [InlineData("-1")]
    public void ExplicitContextTakesPrecedenceWithoutParsingTheEnvironment(string? configured)
    {
        var options = Options();
        Assert.Same(options, GlmUnifiedLayerMajorDeviceBatonPlan.ResolveContextOptions(options, configured));
    }
    [Theory]
    [InlineData(null)] [InlineData("")] [InlineData(" ")] [InlineData("malformed")]
    [InlineData("4096")] [InlineData("0")] [InlineData("-1")] [InlineData("+2048")]
    [InlineData("2048 ")] [InlineData("1.0")] [InlineData("2147483648")]
    public void AutomaticContextRefusesUnconfiguredMalformedOrDifferentProfiles(string? configured)
    {
        var automatic = Options() with { ContextTokens = 0 };
        Assert.Throws<NotSupportedException>(() =>
            GlmUnifiedLayerMajorDeviceBatonPlan.ResolveContextOptions(automatic, configured));
        Assert.Equal(0, automatic.ContextTokens);
    }
    [Theory]
    [InlineData(-1)] [InlineData(4096)]
    public void AnExplicitDifferentContextCannotBeOverriddenToAdmitTheProfile(int context)
        => Assert.Throws<NotSupportedException>(() =>
            GlmUnifiedLayerMajorDeviceBatonPlan.ResolveContextOptions(Options() with { ContextTokens = context }, "2048"));
    [Fact]
    public void PublicHostAutomaticContextPassesTheEarlyProfileBeforeItsCapabilityHandshake()
    {
        using var environment = new EnvScope();
        environment.ClearSpeculationVars();
        foreach (var (name,value) in new[] {
            ("TS_GLM_LAYER_MAJOR_DEVICE_BATON","1"), ("TS_GLM_HOST_TOKEN_EMBEDDING","1"),
            ("MAX_CONTEXT","2048"), ("TS_SCHED_PREFILL_CHUNK","512"),
            ("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB","0"), ("TS_UNIFIED_LAYER_MAJOR_TOKENS","8192"),
            ("TS_UNIFIED_MAX_RUNNING_REQUESTS","1"), ("TS_UNIFIED_EXPERT_CACHE_GROWTH","1"),
            ("TS_HOST_MOE_RAM_CACHE_MB","0"), ("TS_GLM_FLASH_MTP","0"),
            ("TS_SCHED_PREFIX_CACHE","0"), ("TS_PREFIX_CHECKPOINTS_MAX","0"),
            ("TS_PER_SEQ_FUSED","1"), ("TS_BATCHED_FUSED_DECODE","1") })
            environment.Set(name,value);
        var automatic = UnifiedMemorySessionHandle.ResolveHostOptions();
        Assert.Equal(0, automatic.ContextTokens); // The real public Host source.
        var resolved = GlmUnifiedLayerMajorDeviceBatonPlan.ResolveContextOptions(automatic,
            Environment.GetEnvironmentVariable("MAX_CONTEXT"));
        GlmUnifiedMemoryModelSession.ValidateDeviceBatonOptions(resolved, BackendType.GgmlCuda, true, 0, false,
            flag => flag.StartsWith("TS_GLM_TRACE") ? null : "0");
        int calls = 0;
        GgmlGlmNative.CompleteLayerMajorDeviceBatonHandshake(true, () => { ++calls; return 1; },
            () => { ++calls; return 1; }, () => { ++calls; return 0; }, () => { ++calls; return 1; });
        Assert.Equal(4, calls);
        Assert.Equal(automatic with { ContextTokens = 2048 }, resolved);
        Assert.Equal(0, automatic.ContextTokens);
        Assert.Equal("2048", Environment.GetEnvironmentVariable("MAX_CONTEXT"));
        environment.Set("TS_GLM_LAYER_MAJOR_DEVICE_BATON","0");
        Assert.Equal(automatic, UnifiedMemorySessionHandle.ResolveHostOptions());
    }
    [Theory]
    [InlineData(null,false)] [InlineData("0",false)] [InlineData("1",true)]
    public void ExactModeAndImmutableSnapshot(string? value,bool expected)
    {
        Assert.Equal(expected,GlmUnifiedLayerMajorDeviceBatonPlan.Resolve(value));
        GlmUnifiedLayerMajorDeviceBatonPlan.ValidateMode(expected,value);
        Assert.Throws<InvalidOperationException>(() => GlmUnifiedLayerMajorDeviceBatonPlan.ValidateMode(!expected,value));
    }
    [Theory]
    [InlineData("")] [InlineData("01")] [InlineData("true")] [InlineData(" 1")]
    public void InvalidFlagCannotAdmitAProfile(string value)
        => Assert.Throws<ArgumentException>(() => GlmUnifiedLayerMajorDeviceBatonPlan.Resolve(value));
    [Fact]
    public void ProductionHandshakeIsOffFreeAndStrictlyPrecedesAnyLoad()
    {
        int calls=0;
        GgmlGlmNative.CompleteLayerMajorDeviceBatonHandshake(false,() => throw new Exception(),
            () => throw new Exception(),() => throw new Exception(),() => throw new Exception());
        GgmlGlmNative.CompleteLayerMajorDeviceBatonHandshake(true,() => {Assert.Equal(0,calls++);return 1;},
            () => {Assert.Equal(1,calls++);return 1;},() => {Assert.Equal(2,calls++);return 0;},
            () => {Assert.Equal(3,calls++);return 1;});
        Assert.Equal(4,calls);
        Assert.Throws<NotSupportedException>(() => GgmlGlmNative.CompleteLayerMajorDeviceBatonHandshake(true,
            () => throw new EntryPointNotFoundException(),() => 1,() => 0,() => 1));
        Assert.Throws<NotSupportedException>(() => GgmlGlmNative.CompleteLayerMajorDeviceBatonHandshake(true,() => 2,() => 1,() => 0,() => 1));
        Assert.Throws<NotSupportedException>(() => GgmlGlmNative.CompleteLayerMajorDeviceBatonHandshake(true,() => 1,() => 0,() => 0,() => 1));
        Assert.Throws<NotSupportedException>(() => GgmlGlmNative.CompleteLayerMajorDeviceBatonHandshake(true,() => 1,() => 1,() => 1,() => 1));
        Assert.Throws<NotSupportedException>(() => GgmlGlmNative.CompleteLayerMajorDeviceBatonHandshake(true,() => 1,() => 1,() => 0,() => 0));
    }
    [Theory]
    [InlineData(0)] [InlineData(-1)] [InlineData(2)]
    public void RefusedCheckedCloseStopsBeforeFollowingOwnerRetirement(int status)
    {
        int laterOwners=0;
        Assert.Throws<InvalidOperationException>(() => {
            GgmlGlmNative.CompleteLayerMajorDeviceBatonClose(() => status);
            ++laterOwners;
        });
        Assert.Equal(0,laterOwners);
        GgmlGlmNative.CompleteLayerMajorDeviceBatonClose(() => 1);
    }
    [Fact]
    public void OnlyBoundedPrimaryTextProfileIsAdmitted()
    {
        var options=Options();
        GlmUnifiedMemoryModelSession.ValidateDeviceBatonOptions(options,BackendType.GgmlCuda,true,0,false,
            f => f.StartsWith("TS_GLM_TRACE") ? null : "0");
        foreach(var invalid in new[] {options with {ContextTokens=4096},options with {MaximumRunningRequests=2},
            options with {RetainedConversationCount=1},options with {PrefixCheckpointCount=1},
            options with {EnableGlmFlashMtp=true},options with {EnableExpertCacheCapacityGrowth=false},
            options with {AdaptiveGlmPrefillChunkTokens=128},options with {MaximumHostExpertCacheBytes=MiB}})
            Assert.False(GlmUnifiedLayerMajorDeviceBatonPlan.ProfileOptions(invalid,false));
        Assert.False(GlmUnifiedLayerMajorDeviceBatonPlan.ProfileOptions(options,true));
        Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ValidateDeviceBatonOptions(options,
            BackendType.GgmlCuda,true,0,false,f => f=="TS_HOST_MOE_VMM_WEIGHTS" ? "1"
                : f.StartsWith("TS_GLM_TRACE") ? null : "0"));
    }
    [Fact]
    public void WholeCallBytesAreIndependentOfGraphRowsAndGenerationCount()
    {
        var g=Geometry();
        long bank=GlmUnifiedLayerMajorDeviceBatonPlan.Bytes(g,1872,512,true);
        Assert.Equal(117*MiB,bank);
        Assert.Equal(128*MiB,GlmUnifiedLayerMajorDeviceBatonPlan.Bytes(g,2048,512,true));
        Assert.Equal(0,GlmUnifiedLayerMajorDeviceBatonPlan.Bytes(g,33,33,false));
        Assert.Equal(0,GlmUnifiedLayerMajorDeviceBatonPlan.Bytes(g,512,512,true));
        Assert.Equal(0,GlmUnifiedLayerMajorDeviceBatonPlan.Bytes(Geometry(false),1872,512,true));
        Assert.Throws<NotSupportedException>(() => GlmUnifiedLayerMajorDeviceBatonPlan.Bytes(g,1872,128,true));
        Assert.Throws<NotSupportedException>(() => GlmUnifiedLayerMajorDeviceBatonPlan.Bytes(g,2049,512,true));
        foreach(int generations in new[] {1,2,3}) {
            var pure=g.GraphPeak(2048,512,false,generations);
            var replaced=GlmUnifiedPrefillAdmission.PreserveReplacement(pure,new(1,1),generations);
            var target=GlmUnifiedLayerMajorDeviceBatonPlan.AddOnce(replaced,bank);
            Assert.Equal(replaced.Host,target.Host);Assert.Equal(replaced.Device+bank,target.Device);
            // Geometry and the legacy Host residual forecast remain pure.
            Assert.Equal(pure,Geometry(false).GraphPeak(2048,512,false,generations));
            Assert.Equal(bank,g.LayerMajorHostBytes(1872));
        }
    }
    [Fact]
    public void ActualLedgerRequiresOneAdditionalBankAndCannotRevokeItsLiveChild()
    {
        var pure=Geometry().GraphPeak(2048,512,false,2);long bank=117*MiB;
        var target=GlmUnifiedLayerMajorDeviceBatonPlan.AddOnce(pure,bank);
        var budget=new MemoryBudget(Charges(target.Host,target.Device));
        using var joint=budget.Reserve(Charges(target.Host,target.Device));
        using var graph=joint.TryTake(Charges(0,pure.Device-1))!;graph.Commit();
        var baton=joint.TryTake(Charges(0,bank))!;baton.Commit();
        Assert.Null(joint.TryTake(Charges(0,2)));
        var before=budget.Snapshot();
        // A smaller target can release unused credit, but clamps at the live
        // graph and bank owners instead of revoking either physical charge.
        Assert.True(joint.TryResizeTotal(Charges(pure.Host,pure.Device)));
        Assert.Equal(pure.Device-1+bank,joint.OwnedCharges.Single(c=>c.Pool==Device).Bytes);
        Assert.Equal(before.Single(p=>p.Pool==Device).Committed,
            budget.Snapshot().Single(p=>p.Pool==Device).Committed);
        Assert.Equal(0,joint.Charges.Single(c=>c.Pool==Device).Bytes);
        Assert.Null(joint.TryTake(Charges(0,1)));
        baton.Dispose();
        Assert.True(joint.TryResizeTotal(Charges(pure.Host,pure.Device)));
        Assert.Equal(pure.Device-1,joint.LiveChildCharges.Single(c=>c.Pool==Device).Bytes);
    }
    [Fact]
    public void Fixed512ExperimentRefusesANarrowerFitBeforePublishingAnyReservation()
    {
        var budget=new MemoryBudget(Charges(1000,1000));
        var ledger=budget.Snapshot();var limits=Charges(1000,1000);
        (InferenceMemoryBytes,InferenceMemoryBytes) Estimate(int rows) => (new(0,rows*3L),default);
        Assert.True(GlmUnifiedPrefillAdmission.TrySelect(ledger,limits,default,default,0,0,512,Estimate,out var ordinary));
        Assert.Equal(333,ordinary.Rows);
        Assert.False(GlmUnifiedPrefillAdmission.TrySelect(ledger,limits,default,default,0,0,512,Estimate,out _,512));
        Assert.Equal(ledger.ToArray(),budget.Snapshot().ToArray());
    }
    [Theory]
    [InlineData(-1)] [InlineData(-2)] [InlineData(int.MinValue)]
    public void ImmutableBankProfileNeverRunsOrdinaryFallback(int result)
    {
        int fallback=0;
        Assert.Throws<InvalidOperationException>(() => GlmUnifiedLayerMajorDeviceBatonPlan.Complete(result,false,true,
            () => {++fallback;return true;}));
        Assert.Equal(0,fallback);
        Assert.True(GlmUnifiedLayerMajorDeviceBatonPlan.Complete(result,false,false,() => {++fallback;return true;}));
        Assert.Equal(1,fallback);
    }
    private static long[] CompleteInfo(long call=1) => [1,1,0,0,0,call,117*MiB,44*117*MiB,44*117*MiB,176,176,1];
    [Fact]
    public void PendingPhysicalOrFailedAcknowledgmentCannotSupplyReleaseCredit()
    {
        foreach(int field in new[] {2,3,4}) {
            var info=CompleteInfo();info[field]=1;long observed=0;
            Assert.Throws<InvalidOperationException>(() => GlmUnifiedLayerMajorDeviceBatonPlan.CompletedRelease(info,ref observed));
            Assert.Equal(0,observed);
        }
        long seen=0;Assert.Equal(117*MiB,GlmUnifiedLayerMajorDeviceBatonPlan.CompletedRelease(CompleteInfo(),ref seen));
        Assert.Equal(1,seen);Assert.Equal(0,GlmUnifiedLayerMajorDeviceBatonPlan.CompletedRelease(CompleteInfo(),ref seen));
        var bad=CompleteInfo(2);bad[8]--;Assert.Throws<InvalidOperationException>(() =>
            GlmUnifiedLayerMajorDeviceBatonPlan.CompletedRelease(bad,ref seen));Assert.Equal(1,seen);
    }
    [Fact]
    public void CheckedBankCreditCanRestoreSmallExpertLoanOnlyFromFreshAllPoolCapacity()
    {
        long bank=117*MiB;
        var budget=new MemoryBudget(Charges(1000*MiB,2000*MiB));
        using var dense=budget.Reserve(Charges(0,1000*MiB));dense.Commit();
        using var joint=budget.Reserve(Charges(0,400*MiB+bank));
        using var experts=budget.Reserve(Charges(0,355*MiB));
        var actual=joint.TryTake(Charges(0,bank))!;actual.Commit();
        long observed=0;long checkedFree=GlmUnifiedLayerMajorDeviceBatonPlan.CompletedRelease(CompleteInfo(),ref observed);
        // Fixture models the physical free+checkedACK first, then promise clear.
        actual.Dispose();Assert.True(joint.TryResizeTotal(Charges(0,400*MiB)));
        var physical=new GlmUnifiedPhysicalObservation(default,budget.Snapshot(),Charges(1000*MiB,2000*MiB),0,0);
        Assert.Equal(355*MiB,GlmUnifiedExpertLending.RecoveryTarget(physical,355*MiB,472*MiB));
        long minimum=GlmUnifiedLayerMajorDeviceBatonPlan.RecoveryMinimum(bank,checkedFree);
        Assert.Equal(64*MiB,minimum);
        Assert.Equal(472*MiB,GlmUnifiedExpertLending.RecoveryTarget(physical,355*MiB,472*MiB,minimum));
        Assert.True(experts.TryResizeTotal(Charges(0,472*MiB),physical.Limits));
        Assert.Equal(128*MiB,budget.Snapshot().Single(p=>p.Pool==Device).Available);
        Assert.Equal(512*MiB,GlmUnifiedLayerMajorDeviceBatonPlan.RecoveryMinimum(bank,0));
        Assert.Equal(512*MiB,GlmUnifiedLayerMajorDeviceBatonPlan.RecoveryMinimum(63*MiB,bank));
    }
    [Fact]
    public void FreshPhysicalRefusalAndOriginalRepaymentMarginAreNeverGrantedAway()
    {
        long bank=117*MiB;
        var budget=new MemoryBudget(Charges(1000*MiB,2000*MiB));
        using var dense=budget.Reserve(Charges(0,1000*MiB));dense.Commit();
        using var joint=budget.Reserve(Charges(0,400*MiB));
        using var experts=budget.Reserve(Charges(0,600*MiB)); // Initially no optional margin.
        var physical=new GlmUnifiedPhysicalObservation(default,budget.Snapshot(),Charges(1000*MiB,2000*MiB),0,0);
        long repaid=GlmUnifiedExpertLending.RepaymentTarget(physical,Charges(0,bank),600*MiB,100*MiB);
        Assert.Equal((600-117-128)*MiB,repaid);
        Assert.True(experts.TryResizeTotal(Charges(0,repaid),physical.Limits));
        Assert.True(joint.TryResizeTotal(Charges(0,400*MiB+bank),physical.Limits));
        var child=joint.TryTake(Charges(0,bank))!;child.Commit();
        child.Dispose();Assert.True(joint.TryResizeTotal(Charges(0,400*MiB)));
        physical=physical with {Ledger=budget.Snapshot()};
        long minimum=GlmUnifiedLayerMajorDeviceBatonPlan.RecoveryMinimum(bank,bank);
        long target=GlmUnifiedExpertLending.RecoveryTarget(physical,repaid,600*MiB,minimum);
        Assert.Equal((600-128)*MiB,target); // Original128MiB safety margin remains.
        var constrained=physical with {Limits=Charges(1000*MiB,1872*MiB)};
        Assert.Equal(repaid,GlmUnifiedExpertLending.RecoveryTarget(constrained,repaid,600*MiB,minimum));
        Assert.True(experts.TryResizeTotal(Charges(0,target),physical.Limits));
        Assert.Equal(128*MiB,budget.Snapshot().Single(p=>p.Pool==Device).Available);
    }
}
