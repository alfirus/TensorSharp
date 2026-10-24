// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class GlmUnifiedCausalRouteNeighborPlanTests
{
    private const long Slab = 320 * 1024;
    private static UnifiedMemoryModelOptions Options() => new() {
        ContextTokens=2048,PrefillChunkTokens=512,AdaptiveGlmPrefillChunkTokens=512,LayerMajorPrefillTokens=8192,
        MaximumRunningRequests=1,RetainedConversationCount=0,PrefixCheckpointCount=0,
        EnableGlmSelectedExperts=true,EnableAdaptiveGlmPrefill=true,EnableExpertCacheCapacityGrowth=true
    };
    private static string? ProfileEnvironment(string name) => name == "TS_HOST_MOE_EXPERT_CACHE_LFU" ? "0" : null;
    private static long[] Plan() => [1,1,Slab,Slab,42,288,64,8];
    private static long[] Loaded() => [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
    private static long[] Active() => [1,1,1,0,0,0,Slab,Slab,Slab,0,0,0,0,0,0,0];
    private static long[] Closed() => [1,1,1,0,1,0,Slab,0,0,0,32,1344,1,0,1200,144];

    [Theory]
    [InlineData(null,false)] [InlineData("0",false)] [InlineData("1",true)]
    public void ExactFlagValues(string? raw,bool expected)
        => Assert.Equal(expected,GlmUnifiedCausalRouteNeighborPlan.Resolve(raw));
    [Theory]
    [InlineData("")] [InlineData(" ")] [InlineData("01")] [InlineData("true")] [InlineData("-1")]
    public void MalformedFlagsRefuse(string raw)
        => Assert.Throws<ArgumentException>(() => GlmUnifiedCausalRouteNeighborPlan.Resolve(raw));
    [Fact]
    public void ModeIsImmutable()
    {
        GlmUnifiedCausalRouteNeighborPlan.ValidateMode(true,"1");
        GlmUnifiedCausalRouteNeighborPlan.ValidateMode(false,null);
        Assert.Throws<InvalidOperationException>(() => GlmUnifiedCausalRouteNeighborPlan.ValidateMode(true,"0"));
        Assert.Throws<InvalidOperationException>(() => GlmUnifiedCausalRouteNeighborPlan.ValidateMode(false,"1"));
    }
    [Theory]
    [InlineData("2048")] [InlineData("02048")]
    public void AutomaticHostContextUsesExistingPrecedence(string configured)
    {
        var original=Options() with { ContextTokens=0 };
        var normalized=GlmUnifiedCausalRouteNeighborPlan.ResolveContextOptions(original,configured);
        Assert.Equal(0,original.ContextTokens);
        Assert.Equal(2048,normalized.ContextTokens);
        GlmUnifiedMemoryModelSession.ValidateCausalRouteNeighborOptions(normalized,BackendType.GgmlCuda,true,0,false,ProfileEnvironment);
    }
    [Theory]
    [InlineData(null)] [InlineData("")] [InlineData("4096")] [InlineData("bad")] [InlineData("+2048")] [InlineData("2048 ")]
    public void AutomaticContextRefusesUnqualifiedProfiles(string? configured)
        => Assert.Throws<NotSupportedException>(() =>
            GlmUnifiedCausalRouteNeighborPlan.ResolveContextOptions(Options() with { ContextTokens=0 },configured));
    [Fact]
    public void ExplicitContextDoesNotParseIgnoredEnvironment()
    {
        var options=Options();
        Assert.Same(options,GlmUnifiedCausalRouteNeighborPlan.ResolveContextOptions(options,"bad"));
        Assert.Throws<NotSupportedException>(() =>
            GlmUnifiedCausalRouteNeighborPlan.ResolveContextOptions(options with { ContextTokens=4096 },"2048"));
    }
    [Fact]
    public void PublicHostAutomaticOptionsNormalizeBeforeTheProfileGuard()
    {
        using var environment=new EnvScope();
        environment.ClearSpeculationVars();
        foreach(var (name,value) in new[] {
            ("TS_HOST_MOE_CAUSAL_ROUTE_NEIGHBOR","1"),("TS_GLM_HOST_TOKEN_EMBEDDING","1"),
            ("MAX_CONTEXT","2048"),("TS_SCHED_PREFILL_CHUNK","512"),
            ("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB","0"),("TS_UNIFIED_LAYER_MAJOR_TOKENS","8192"),
            ("TS_UNIFIED_MAX_RUNNING_REQUESTS","1"),("TS_UNIFIED_EXPERT_CACHE_GROWTH","1"),
            ("TS_HOST_MOE_RAM_CACHE_MB","0"),("TS_GLM_FLASH_MTP","0"),
            ("TS_SCHED_PREFIX_CACHE","0"),("TS_PREFIX_CHECKPOINTS_MAX","0"),
            ("TS_PER_SEQ_FUSED","1"),("TS_BATCHED_FUSED_DECODE","1") })
            environment.Set(name,value);
        // ResolveHostOptions is separately qualified with96 cases. This case
        // binds its automatic context to this feature's actual early guard.
        var options=UnifiedMemorySessionHandle.ResolveHostOptions();
        Assert.Equal(0,options.ContextTokens);
        var normalized=GlmUnifiedCausalRouteNeighborPlan.ResolveContextOptions(options,"2048");
        GlmUnifiedMemoryModelSession.ValidateCausalRouteNeighborOptions(normalized,BackendType.GgmlCuda,true,0,false,ProfileEnvironment);
        Assert.Equal(options with { ContextTokens=2048 },normalized);
        environment.Set("TS_HOST_MOE_CAUSAL_ROUTE_NEIGHBOR","0");
        Assert.Equal(options,UnifiedMemorySessionHandle.ResolveHostOptions());
    }
    [Fact]
    public void ProfileRejectsOtherArithmeticAndConcurrency()
    {
        var valid=Options();
        Assert.True(GlmUnifiedCausalRouteNeighborPlan.ProfileOptions(valid,false));
        foreach(var invalid in new[] {valid with {MaximumRunningRequests=2},valid with {RetainedConversationCount=1},
            valid with {PrefixCheckpointCount=1},valid with {EnableGlmFlashMtp=true},
            valid with {MaximumHostExpertCacheBytes=1},valid with {EnableGlmSelectedExperts=false},
            valid with {EnableAdaptiveGlmPrefill=false},valid with {EnableExpertCacheCapacityGrowth=false},
            valid with {AdaptiveGlmPrefillChunkTokens=256},valid with {LayerMajorPrefillTokens=512}})
            Assert.False(GlmUnifiedCausalRouteNeighborPlan.ProfileOptions(invalid,false));
        Assert.False(GlmUnifiedCausalRouteNeighborPlan.ProfileOptions(valid,true));
    }
    [Theory]
    [InlineData("TS_GLM_LAYER_MAJOR_DEVICE_BATON")] [InlineData("TS_HOST_MOE_VMM_WEIGHTS")]
    [InlineData("TS_HOST_MOE_FILE_SOURCE_VIEW")] [InlineData("TS_HOST_MOE_PREFILL_DOWN_OVERLAP")]
    [InlineData("TS_HOST_MOE_EXPERT_CACHE_LFU")] [InlineData("TS_HOST_MOE_FILE_READ_WORKERS")]
    [InlineData("TS_HOST_MOE_ROUTE_TRACE")] [InlineData("TS_DSV4_PERF")]
    public void ConflictingExperimentsRefuse(string conflict)
        => Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ValidateCausalRouteNeighborOptions(
            Options(),BackendType.GgmlCuda,true,0,false,name=>name==conflict?"1":ProfileEnvironment(name)));
    [Fact]
    public void ProfileRequiresHostEmbeddingAndExplicitLru()
    {
        Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ValidateCausalRouteNeighborOptions(
            Options(),BackendType.GgmlCuda,false,0,false,ProfileEnvironment));
        Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ValidateCausalRouteNeighborOptions(
            Options(),BackendType.GgmlCuda,true,0,false,_=>null));
    }
    [Fact]
    public void NativeHostCertificateIsExactAndBounded()
    {
        Assert.Equal(Slab,GlmUnifiedCausalRouteNeighborPlan.ValidateHostPlan(Plan()));
        foreach(int field in Enumerable.Range(0,8)) {
            var invalid=Plan();invalid[field]++;
            // Size fields must be identical; all others are versioned constants.
            Assert.Throws<NotSupportedException>(() => GlmUnifiedCausalRouteNeighborPlan.ValidateHostPlan(invalid));
        }
        var tooLarge=Plan();tooLarge[2]=tooLarge[3]=(1L<<20)+1;
        Assert.Throws<NotSupportedException>(()=>GlmUnifiedCausalRouteNeighborPlan.ValidateHostPlan(tooLarge));
        Assert.Throws<NotSupportedException>(()=>GlmUnifiedCausalRouteNeighborPlan.ValidateHostPlan([]));
    }
    [Fact]
    public void DisabledHandshakeDoesNotDependOnTheNewNativeAbi()
    {
        int Unavailable()=>throw new EntryPointNotFoundException();
        GgmlGlmNative.CompleteCausalRouteNeighborHandshake(false,Unavailable,Unavailable,Unavailable,Unavailable);
        Assert.Throws<NotSupportedException>(()=>GgmlGlmNative.CompleteCausalRouteNeighborHandshake(true,Unavailable,Unavailable,Unavailable,Unavailable));
    }
    [Theory]
    [InlineData(0,1,0,1)] [InlineData(1,0,0,1)] [InlineData(1,-1,0,1)]
    [InlineData(1,1,1,1)] [InlineData(1,1,0,0)] [InlineData(1,1,0,2)]
    public void HandshakeRequiresExactCapabilityAndNullLifecycle(int version,int enabled,int info,int close)
        => Assert.Throws<NotSupportedException>(()=>GgmlGlmNative.CompleteCausalRouteNeighborHandshake(true,
            ()=>version,()=>enabled,()=>info,()=>close));
    [Fact]
    public void OwnershipCertificatesDistinguishLoadActiveAndRetirement()
    {
        GlmUnifiedCausalRouteNeighborPlan.ValidateLoadedInfo(Loaded());
        GlmUnifiedCausalRouteNeighborPlan.ValidateActiveInfo(Active(),Slab);
        GlmUnifiedCausalRouteNeighborPlan.ValidateClosedInfo(Closed(),Slab);
        Assert.Throws<InvalidOperationException>(()=>GlmUnifiedCausalRouteNeighborPlan.ValidateLoadedInfo(Active()));
        Assert.Throws<InvalidOperationException>(()=>GlmUnifiedCausalRouteNeighborPlan.ValidateClosedInfo(Active(),Slab));
        var neverConfigured=Loaded();neverConfigured[4]=1;
        GlmUnifiedCausalRouteNeighborPlan.ValidateClosedInfo(neverConfigured,0);
    }
    [Theory]
    [InlineData(3,1)] [InlineData(4,1)] [InlineData(5,1)] [InlineData(7,0)] [InlineData(8,0)]
    [InlineData(9,1)] [InlineData(1,2)] [InlineData(15,-1)]
    public void ActiveOwnershipCannotBeInventedOrPartiallyRetired(int field,long value)
    {
        var info=Active();info[field]=value;
        Assert.Throws<InvalidOperationException>(()=>GlmUnifiedCausalRouteNeighborPlan.ValidateActiveInfo(info,Slab));
    }
    [Fact]
    public void FailedAcknowledgmentRetainsTheOwnerAfterLedgerRefund()
    {
        var budget=new MemoryBudget([new("Host",Slab),new("Device",0)]);
        using var envelope=budget.Reserve([new("Host",Slab)]);
        var child=envelope.TryTake([new("Host",Slab)]);
        Assert.NotNull(child);child!.Commit();
        Assert.Equal(Slab,budget.Snapshot().Single(p=>p.Pool=="Host").Committed);
        var pending=Active();pending[3]=1;pending[7]=0;
        child!.Dispose(); // Physical free/refund can precede failed native ACK.
        Assert.Equal(0,budget.Snapshot().Single(p=>p.Pool=="Host").Committed);
        Assert.Throws<InvalidOperationException>(()=>GlmUnifiedCausalRouteNeighborPlan.ValidateClosedInfo(pending,Slab));
        Assert.Throws<InvalidOperationException>(()=>GgmlGlmNative.CompleteCausalRouteNeighborClose(()=>0));
        GgmlGlmNative.CompleteCausalRouteNeighborClose(()=>1);
        GlmUnifiedCausalRouteNeighborPlan.ValidateClosedInfo(Closed(),Slab);
        Assert.Equal(Slab,envelope.OwnedCharges.Single().Bytes);
        Assert.Equal(0,budget.Snapshot().Single(p=>p.Pool=="Device").Committed);
    }
}
