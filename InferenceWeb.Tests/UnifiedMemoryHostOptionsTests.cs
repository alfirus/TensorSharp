// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Models;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>Host option resolution only: no model files, native allocator or GPU
/// execution. Process environment changes are isolated and restored.</summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class UnifiedMemoryHostOptionsTests : IDisposable
{
    private readonly Dictionary<string, string> _previous = new();
    public UnifiedMemoryHostOptionsTests()
    {
        foreach (string name in new[] { "TS_SCHED_PREFILL_CHUNK", "TS_PER_SEQ_FUSED", "TS_BATCHED_FUSED_DECODE",
            "TS_SCHED_PREFIX_CACHE", "TS_PREFIX_CHECKPOINTS_MAX", "TS_UNIFIED_EXPERT_CACHE_GROWTH",
            "TS_HOST_MOE_RAM_CACHE_MB", "TS_GLM_FLASH_MTP", "TS_UNIFIED_MAX_RUNNING_REQUESTS",
            "TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", "TS_UNIFIED_LAYER_MAJOR_TOKENS",
            "TS_GLM_HOST_TOKEN_EMBEDDING",
            "TS_SPEC", "TS_SPEC_TYPE", "TS_SPEC_DRAFT", "TS_SPEC_DRAFT_MODEL" })
        {
            _previous.Add(name, Environment.GetEnvironmentVariable(name));
            Environment.SetEnvironmentVariable(name, null);
        }
    }
    public void Dispose()
    {
        foreach (var pair in _previous) Environment.SetEnvironmentVariable(pair.Key, pair.Value);
    }

    [Fact]
    public void ExplicitUnifiedHostSelectsThePlannedGlmTierWithoutChangingApiDefaultsOrEnvironment()
    {
        var expected = new UnifiedMemoryModelOptions { EnableGlmSelectedExperts = true, EnableAdaptiveGlmPrefill = true,
            AdaptiveGlmPrefillChunkTokens = 1024, EnableExpertCacheCapacityGrowth = true };
        Assert.Equal(expected, UnifiedMemorySessionHandle.ResolveHostOptions());
        Assert.Equal(512, expected.PrefillChunkTokens);
        Assert.Equal(1024, expected.ResolveGlmPrefillChunkTokens());
        Assert.Null(new UnifiedMemoryModelOptions().AdaptiveGlmPrefillChunkTokens);
        Assert.Equal(0, expected.ContextTokens);
        Assert.Equal(3, expected.MaximumRunningRequests);
        Assert.True(expected.EnableGlmSelectedExperts);
        Assert.False(new UnifiedMemoryModelOptions().EnableGlmSelectedExperts);
        Assert.True(expected.EnableAdaptiveGlmPrefill);
        Assert.False(new UnifiedMemoryModelOptions().EnableAdaptiveGlmPrefill);
        Assert.False(expected.EnableAsyncExpertPrefill);
        Assert.True(expected.EnableExpertCacheCapacityGrowth);
        Assert.False(new UnifiedMemoryModelOptions().EnableExpertCacheCapacityGrowth);
        Assert.All(_previous.Keys, name => Assert.Null(Environment.GetEnvironmentVariable(name)));
    }

    [Theory]
    [InlineData("1", 1)]
    [InlineData("16", 16)]
    [InlineData("128", 128)]
    [InlineData("512", 512)]
    public void ExplicitPrefillCapConstrainsBothCommonAndAdaptiveGlmModelLoadOptions(string value, int expected)
    {
        Environment.SetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK", value);
        var options = UnifiedMemorySessionHandle.ResolveHostOptions();
        Assert.Equal(new UnifiedMemoryModelOptions { PrefillChunkTokens = expected,
            EnableGlmSelectedExperts = true, EnableAdaptiveGlmPrefill = true,
            AdaptiveGlmPrefillChunkTokens = expected, EnableExpertCacheCapacityGrowth = true }, options);
        Assert.Equal(expected, options.ResolveGlmPrefillChunkTokens());
        Assert.Equal(value, Environment.GetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK"));
        // A later load receives a fresh snapshot, rather than mutating a prior
        // session's immutable options or resetting the operator's environment.
        Environment.SetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK", "32");
        Assert.Equal(32, UnifiedMemorySessionHandle.ResolveHostOptions().PrefillChunkTokens);
        Assert.Equal(expected, options.PrefillChunkTokens);
        Assert.Equal(expected, options.AdaptiveGlmPrefillChunkTokens);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("513")]
    [InlineData("2147483648")]
    [InlineData("abc")]
    [InlineData("1.5")]
    [InlineData(" 16 ")]
    [InlineData("+16")]
    public void InvalidPrefillRefusesBeforeAnyArchitectureOrNativeLoad(string value)
    {
        Environment.SetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK", value);
        foreach (string architecture in new[] { "qwen4exp", "glm5next", "glm5-next" })
        {
            var error = Assert.Throws<ArgumentException>(() => UnifiedMemorySessionHandle.CreateOwned(
                Path.Combine(Path.GetTempPath(), "missing-" + architecture + ".gguf"), null, BackendType.GgmlCuda));
            Assert.Contains("TS_SCHED_PREFILL_CHUNK", error.Message);
            Assert.Contains("1 through 512", error.Message);
        }
        Assert.Equal(value, Environment.GetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK"));
    }

    [Theory]
    [InlineData("TS_PER_SEQ_FUSED", "0")]
    [InlineData("TS_PER_SEQ_FUSED", "false")]
    [InlineData("TS_PER_SEQ_FUSED", "FALSE")]
    [InlineData("TS_BATCHED_FUSED_DECODE", "0")]
    [InlineData("TS_BATCHED_FUSED_DECODE", "false")]
    [InlineData("TS_BATCHED_FUSED_DECODE", "FALSE")]
    public void UnsupportedFusedOptOutRefusesAtTheSharedPreloadGuard(string variable, string value)
    {
        Environment.SetEnvironmentVariable(variable, value);
        foreach (string architecture in new[] { "qwen4exp", "glm5next", "glm5-next" })
        {
            var error = Assert.Throws<NotSupportedException>(() => UnifiedMemorySessionHandle.CreateOwned(
                Path.Combine(Path.GetTempPath(), "missing-" + architecture + ".gguf"), null, BackendType.GgmlCuda));
            Assert.Contains(variable, error.Message);
            Assert.Contains("requires", error.Message);
        }
        Assert.Equal(value, Environment.GetEnvironmentVariable(variable));
    }

    [Theory]
    [InlineData("--prefill-chunk-size", "16", 16)]
    [InlineData("--prefill-chunk-size=128", null, 128)]
    [InlineData("--PREFILL-CHUNK-SIZE", "512", 512)]
    public void ActualCliPassOverridesInheritedCapAndReachesUnifiedLoadOptions(
        string flag, string? value, int expected)
    {
        Environment.SetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK", "513");
        string[] args = value == null
            ? ["--unified-memory", "--backend", "ggml_cuda", flag]
            : ["--unified-memory", "--backend", "ggml_cuda", flag, value];
        Assert.True(ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(args));
        Assert.True(ServerOptionsBuilder.Build(args, Path.GetTempPath()).UnifiedMemoryEnabled);
        Assert.Equal(expected, UnifiedMemorySessionHandle.ResolveHostOptions().PrefillChunkTokens);
        Assert.Equal(expected, UnifiedMemorySessionHandle.ResolveHostOptions().AdaptiveGlmPrefillChunkTokens);
    }

    [Fact]
    public void ActualCliCapBeyondUnifiedCeilingRefusesBeforeModelLoading()
    {
        string[] args = ["--unified-memory", "--backend", "ggml_cuda", "--prefill-chunk-size=513"];
        Assert.True(ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(args));
        Assert.True(ServerOptionsBuilder.Build(args, Path.GetTempPath()).UnifiedMemoryEnabled);
        var error = Assert.Throws<ArgumentException>(() => UnifiedMemorySessionHandle.CreateOwned(
            Path.Combine(Path.GetTempPath(), "missing-cli-input.gguf"), null, BackendType.GgmlCuda));
        Assert.Contains("1 through 512", error.Message);
    }

    [Theory]
    [InlineData("1")]
    [InlineData("true")]
    [InlineData("TRUE")]
    public void ExplicitEnabledFusedPathsPreserveDefaults(string value)
    {
        Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", value);
        Environment.SetEnvironmentVariable("TS_BATCHED_FUSED_DECODE", value);
        Assert.Equal(new UnifiedMemoryModelOptions { EnableGlmSelectedExperts = true, EnableAdaptiveGlmPrefill = true,
                AdaptiveGlmPrefillChunkTokens = 1024, EnableExpertCacheCapacityGrowth = true },
            UnifiedMemorySessionHandle.ResolveHostOptions());
    }

    [Theory]
    [InlineData("0", "4", 0)]
    [InlineData("false", "8", 0)]
    [InlineData("1", "0", 0)]
    [InlineData("true", "1", 1)]
    [InlineData("1", "4", 4)]
    [InlineData("1", "8", 8)]
    public void PrefixPlacementHonorsTheActualHostCacheAndOperatorCheckpointBudget(
        string enabled, string copies, int expected)
    {
        Environment.SetEnvironmentVariable("TS_SCHED_PREFIX_CACHE", enabled);
        Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS_MAX", copies);
        var options = UnifiedMemorySessionHandle.ResolveHostOptions();
        Assert.Equal(expected, options.PrefixCheckpointCount);
        Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS_MAX", "2");
        Assert.Equal(expected, options.PrefixCheckpointCount); // Immutable load forecast.
        Assert.Equal(enabled, Environment.GetEnvironmentVariable("TS_SCHED_PREFIX_CACHE"));
    }

    [Theory]
    [InlineData(null, true)]
    [InlineData("1", true)]
    [InlineData("0", false)]
    public void ExpertCacheGrowthDefaultsOnAndHonorsTheOperatorSwitch(string? value, bool expected)
    {
        Environment.SetEnvironmentVariable("TS_UNIFIED_EXPERT_CACHE_GROWTH", value);
        Assert.Equal(expected, UnifiedMemorySessionHandle.ResolveHostOptions().EnableExpertCacheCapacityGrowth);
    }

    [Theory]
    [InlineData("2")]
    [InlineData("true")]
    [InlineData(" 1")]
    public void InvalidExpertCacheGrowthSwitchRefuses(string value)
    {
        Environment.SetEnvironmentVariable("TS_UNIFIED_EXPERT_CACHE_GROWTH", value);
        var error = Assert.Throws<ArgumentException>(UnifiedMemorySessionHandle.ResolveHostOptions);
        Assert.Contains("TS_UNIFIED_EXPERT_CACHE_GROWTH", error.Message);
    }

    [Theory]
    [InlineData(null, 0)]
    [InlineData("0", 0)]
    [InlineData("8192", 8589934592)]
    public void RamExpertCacheIsOptInAndPreservesTheExplicitOperatorCeiling(string? value, long bytes)
    {
        Environment.SetEnvironmentVariable("TS_HOST_MOE_RAM_CACHE_MB", value);
        Assert.Equal(bytes, UnifiedMemorySessionHandle.ResolveHostOptions().MaximumHostExpertCacheBytes);
        Assert.Equal(0, new UnifiedMemoryModelOptions().MaximumHostExpertCacheBytes);
        Assert.Equal(value, Environment.GetEnvironmentVariable("TS_HOST_MOE_RAM_CACHE_MB"));
    }

    [Theory]
    [InlineData("-1")]
    [InlineData("+2")]
    [InlineData("1.5")]
    [InlineData(" 2")]
    [InlineData("8796093022208")]
    public void InvalidRamExpertCacheCapacityRefusesBeforeModelLoading(string value)
    {
        Environment.SetEnvironmentVariable("TS_HOST_MOE_RAM_CACHE_MB", value);
        var error = Assert.Throws<ArgumentException>(() => UnifiedMemorySessionHandle.CreateOwned(
            Path.Combine(Path.GetTempPath(), "missing-glm.gguf"), null, BackendType.GgmlCuda));
        Assert.Contains("TS_HOST_MOE_RAM_CACHE_MB", error.Message);
    }

    [Theory]
    [InlineData("1", 1)]
    [InlineData("2", 2)]
    [InlineData("3", 3)]
    public void ExplicitConcurrencyAppliesIndependentlyOfExperimentalMtp(string value, int count)
    {
        Environment.SetEnvironmentVariable("TS_UNIFIED_MAX_RUNNING_REQUESTS", value);
        var options = UnifiedMemorySessionHandle.ResolveHostOptions();
        Assert.Equal(count, options.MaximumRunningRequests);
        Assert.False(options.EnableGlmFlashMtp);
        Assert.Equal(3, new UnifiedMemoryModelOptions().MaximumRunningRequests);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("4")]
    [InlineData(" 1")]
    [InlineData("+1")]
    [InlineData("1.0")]
    public void InvalidConcurrencyRefusesBeforeModelLoading(string value)
    {
        Environment.SetEnvironmentVariable("TS_UNIFIED_MAX_RUNNING_REQUESTS", value);
        Assert.Contains("TS_UNIFIED_MAX_RUNNING_REQUESTS",
            Assert.Throws<ArgumentException>(UnifiedMemorySessionHandle.ResolveHostOptions).Message);
    }

    [Fact]
    public void FlashMtpExplicitOptInRequiresOnePrimaryAndRemovesRetainedPrefixCredit()
    {
        Assert.False(UnifiedMemorySessionHandle.ResolveHostOptions().EnableGlmFlashMtp);
        Environment.SetEnvironmentVariable("TS_GLM_FLASH_MTP", "1");
        var options = UnifiedMemorySessionHandle.ResolveHostOptions();
        Assert.True(options.EnableGlmFlashMtp);
        Assert.Equal(1, options.MaximumRunningRequests);
        Assert.Equal(0, options.PrefixCheckpointCount);
        Assert.Equal(0, options.RetainedConversationCount);
        Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", "0");
        Environment.SetEnvironmentVariable("TS_BATCHED_FUSED_DECODE", "0");
        Assert.True(UnifiedMemorySessionHandle.ResolveHostOptions().EnableGlmFlashMtp);
        Environment.SetEnvironmentVariable("TS_UNIFIED_MAX_RUNNING_REQUESTS", "2");
        Assert.Throws<ArgumentException>(UnifiedMemorySessionHandle.ResolveHostOptions);
        Environment.SetEnvironmentVariable("TS_UNIFIED_MAX_RUNNING_REQUESTS", null);
        Environment.SetEnvironmentVariable("TS_GLM_FLASH_MTP", "true");
        Assert.Throws<ArgumentException>(UnifiedMemorySessionHandle.ResolveHostOptions);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("0")]
    public void DisabledStrictGraphCapPreservesOrdinaryHostConcurrencyAndRetention(string? value)
    {
        Environment.SetEnvironmentVariable("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", value);
        var options = UnifiedMemorySessionHandle.ResolveHostOptions();
        Assert.Equal(3, options.MaximumRunningRequests);
        Assert.Equal(2, options.RetainedConversationCount);
        Assert.Equal(4, options.PrefixCheckpointCount);
        Assert.Equal(1024, options.AdaptiveGlmPrefillChunkTokens);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("0")]
    public void DisabledHostEmbeddingPreservesOrdinaryOptions(string? value)
    {
        var original = UnifiedMemorySessionHandle.ResolveHostOptions();
        Environment.SetEnvironmentVariable("TS_GLM_HOST_TOKEN_EMBEDDING", value);
        Assert.Equal(original, UnifiedMemorySessionHandle.ResolveHostOptions());
    }

    [Theory]
    [InlineData("0", "1")]
    [InlineData("0", "512")]
    [InlineData("512", "512")]
    [InlineData("768", "512")]
    public void HostEmbeddingChoosesOnePrimaryWithCapOffOrCompatibleCap(string cap, string chunk)
    {
        Environment.SetEnvironmentVariable("TS_GLM_HOST_TOKEN_EMBEDDING", "1");
        Environment.SetEnvironmentVariable("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", cap);
        Environment.SetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK", chunk);
        var options = UnifiedMemorySessionHandle.ResolveHostOptions();
        Assert.Equal(1, options.MaximumRunningRequests);
        Assert.Equal(0, options.RetainedConversationCount);
        Assert.Equal(0, options.PrefixCheckpointCount);
        Assert.Equal(int.Parse(chunk), options.ResolveGlmPrefillChunkTokens());
        Assert.True(GlmUnifiedHostTokenEmbeddingPlan.ProfileOptions(options, false));
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_GLM_HOST_TOKEN_EMBEDDING"));
    }

    [Theory]
    [InlineData("")]
    [InlineData("true")]
    [InlineData("01")]
    [InlineData("1 ")]
    [InlineData("2")]
    public void InvalidHostEmbeddingRefusesBeforeHeaderAndNativeLoading(string value)
    {
        Environment.SetEnvironmentVariable("TS_GLM_HOST_TOKEN_EMBEDDING", value);
        var error = Assert.Throws<ArgumentException>(() => UnifiedMemorySessionHandle.CreateOwned(
            Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".gguf"), null, BackendType.GgmlCuda));
        Assert.Contains("TS_GLM_HOST_TOKEN_EMBEDDING", error.Message);
    }

    [Theory]
    [InlineData("TS_GLM_FLASH_MTP", "1")]
    [InlineData("TS_UNIFIED_MAX_RUNNING_REQUESTS", "2")]
    [InlineData("TS_SCHED_PREFIX_CACHE", "1")]
    [InlineData("TS_SCHED_PREFIX_CACHE", "true")]
    public void HostEmbeddingCannotSilentlyEnableIncompatibleOwners(string variable, string value)
    {
        Environment.SetEnvironmentVariable("TS_GLM_HOST_TOKEN_EMBEDDING", "1");
        Environment.SetEnvironmentVariable(variable, value);
        Assert.Throws<ArgumentException>(() => UnifiedMemorySessionHandle.CreateOwned(
            Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".gguf"), null, BackendType.GgmlCuda));
    }

    [Fact]
    public void HostEmbeddingRequiresExplicitUnifiedLoadPolicyAndHonestImmutableMode()
    {
        Environment.SetEnvironmentVariable("TS_GLM_HOST_TOKEN_EMBEDDING", "1");
        var policy = new ModelMemoryPolicy(2048, 512) { OmitEmbeddedDraftWeights = true, NativeMoeCpuLayers = 45 };
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ValidateUnifiedMemoryPolicy(null, BackendType.Cpu, 1, 1, false));
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ValidateUnifiedMemoryPolicy(policy, BackendType.GgmlCuda, 1, 1, false));
        policy = policy with { EnableGlmHostTokenEmbedding = true };
        GlmDsaModel.ValidateUnifiedMemoryPolicy(policy, BackendType.GgmlCuda, 1, 1, false);
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ValidateUnifiedMemoryPolicy(policy, BackendType.GgmlCuda, 2, 1, false));
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ValidateUnifiedMemoryPolicy(policy with { EnableGlmFlashMtp = true }, BackendType.GgmlCuda, 1, 1, false));
        Environment.SetEnvironmentVariable("TS_GLM_HOST_TOKEN_EMBEDDING", "0");
        Assert.Throws<InvalidOperationException>(() => GlmDsaModel.ValidateUnifiedMemoryPolicy(policy, BackendType.GgmlCuda, 1, 1, false));
    }

    [Theory]
    [InlineData("512", null)]
    [InlineData("768", null)]
    [InlineData("512", "0")]
    [InlineData("768", "false")]
    public void ExplicitStrictGraphCapSelectsOnePrimaryWithoutRetainedOrPrefixOwners(string cap, string? prefix)
    {
        Environment.SetEnvironmentVariable("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", cap);
        Environment.SetEnvironmentVariable("TS_SCHED_PREFIX_CACHE", prefix);
        Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS_MAX", "8");
        var options = UnifiedMemorySessionHandle.ResolveHostOptions();
        Assert.Equal(1, options.MaximumRunningRequests);
        Assert.Equal(0, options.RetainedConversationCount);
        Assert.Equal(0, options.PrefixCheckpointCount);
        Assert.Equal(512, options.ResolveGlmPrefillChunkTokens());
        Assert.True(GlmUnifiedLayerMajorGraphCapPlan.ProfileOptions(options, media: false));
        Assert.Equal(cap, Environment.GetEnvironmentVariable("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB"));
        Assert.Equal(prefix, Environment.GetEnvironmentVariable("TS_SCHED_PREFIX_CACHE"));
        Assert.Equal("8", Environment.GetEnvironmentVariable("TS_PREFIX_CHECKPOINTS_MAX"));
        Environment.SetEnvironmentVariable("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", "0");
        Assert.Equal(0, options.RetainedConversationCount); // Prior immutable snapshot stays strict.
        Assert.Equal(2, UnifiedMemorySessionHandle.ResolveHostOptions().RetainedConversationCount);
    }

    [Theory]
    [InlineData("512", 0, 1)]
    [InlineData("768", 0, 1)]
    [InlineData("0", 2, 3)]
    public void ActualNoPrefixCliReachesCompatibleStrictLoadPolicyWithoutChangingOffDefaults(
        string cap, int retained, int running)
    {
        Environment.SetEnvironmentVariable("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", cap);
        Environment.SetEnvironmentVariable("TS_SCHED_PREFIX_CACHE", "1");
        string[] args = ["--unified-memory", "--backend", "ggml_cuda", "--no-prefix-cache", "--prefill-chunk-size=512"];
        Assert.True(ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(args));
        Assert.True(ServerOptionsBuilder.ApplyPrefixCacheCliFlag(args));
        Assert.True(ServerOptionsBuilder.Build(args, Path.GetTempPath()).UnifiedMemoryEnabled);
        var options = UnifiedMemorySessionHandle.ResolveHostOptions();
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_SCHED_PREFIX_CACHE"));
        Assert.Equal(0, options.PrefixCheckpointCount);
        Assert.Equal(retained, options.RetainedConversationCount);
        Assert.Equal(running, options.MaximumRunningRequests);
        Assert.Equal(512, options.ResolveGlmPrefillChunkTokens());
        if (cap != "0") Assert.True(GlmUnifiedLayerMajorGraphCapPlan.ProfileOptions(options, media: false));
    }

    [Theory]
    [InlineData("")]
    [InlineData("1")]
    [InlineData("0512")]
    [InlineData("512 ")]
    [InlineData("1024")]
    public void InvalidStrictGraphCapRefusesAtHostGuardBeforeHeaderOrNativeAccess(string cap)
    {
        Environment.SetEnvironmentVariable("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", cap);
        var error = Assert.Throws<ArgumentException>(() => UnifiedMemorySessionHandle.CreateOwned(
            Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".gguf"), null, BackendType.GgmlCuda));
        Assert.Contains("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", error.Message);
    }

    [Theory]
    [InlineData("TS_GLM_FLASH_MTP", "1")]
    [InlineData("TS_UNIFIED_MAX_RUNNING_REQUESTS", "2")]
    [InlineData("TS_UNIFIED_MAX_RUNNING_REQUESTS", "3")]
    [InlineData("TS_SCHED_PREFILL_CHUNK", "256")]
    [InlineData("TS_UNIFIED_LAYER_MAJOR_TOKENS", "0")]
    [InlineData("TS_UNIFIED_LAYER_MAJOR_TOKENS", "512")]
    [InlineData("TS_SCHED_PREFIX_CACHE", "1")]
    [InlineData("TS_SCHED_PREFIX_CACHE", "true")]
    public void ExplicitIncompatibleStrictGraphCapOptionsRefuseBeforeModelLoading(string variable, string value)
    {
        Environment.SetEnvironmentVariable("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", "768");
        Environment.SetEnvironmentVariable(variable, value);
        Assert.Throws<ArgumentException>(() => UnifiedMemorySessionHandle.CreateOwned(
            Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".gguf"), null, BackendType.GgmlCuda));
        Assert.Equal(value, Environment.GetEnvironmentVariable(variable));
    }

    [Fact]
    public void QualifiedFlashMtpPassesTheActualHostAndFactoryGuardsBeforeHeaderAccess()
    {
        Environment.SetEnvironmentVariable("TS_GLM_FLASH_MTP", "1");
        Environment.SetEnvironmentVariable("TS_SPEC", "1");
        Environment.SetEnvironmentVariable("TS_SPEC_TYPE", "draft-head");
        Environment.SetEnvironmentVariable("TS_SPEC_DRAFT", "2");
        // Missing input is deliberate: reaching header access proves the real
        // public factory passed its outer guards, without native/GPU loading.
        Assert.Throws<FileNotFoundException>(() => UnifiedMemorySessionHandle.CreateOwned(
            Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".gguf"), null, BackendType.GgmlCuda));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, true)]
    public void OrdinarySpeculationAndExternalDraftsStillRefuseBeforeHeaderAccess(bool optIn, bool externalDraft)
    {
        Environment.SetEnvironmentVariable("TS_SPEC", "1");
        if (optIn) Environment.SetEnvironmentVariable("TS_GLM_FLASH_MTP", "1");
        if (externalDraft) Environment.SetEnvironmentVariable("TS_SPEC_DRAFT_MODEL", "unqualified-draft.gguf");
        var error = Assert.Throws<NotSupportedException>(() => UnifiedMemorySessionHandle.CreateOwned(
            Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".gguf"), null, BackendType.GgmlCuda));
        Assert.Contains("speculative execution and draft loading", error.Message);
    }
}
