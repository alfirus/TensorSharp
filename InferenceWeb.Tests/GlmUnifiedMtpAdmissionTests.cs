// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

/// <summary>CPU admission and opt-in policy only; native exactness is tested separately.</summary>
public sealed class GlmUnifiedMtpAdmissionTests
{
    [Theory]
    [InlineData(null, 3)]
    [InlineData("", 3)]
    [InlineData("1", 1)]
    [InlineData("8", 8)]
    [InlineData("16", 16)]
    public void ExperimentalGraphRetentionHasAnExplicitBound(string? value, int expected)
        => Assert.Equal(expected, GlmDsaModel.ResolveFlashMtpGraphCacheCapacity(value));

    [Theory]
    [InlineData("0")]
    [InlineData("17")]
    [InlineData("-1")]
    [InlineData("+8")]
    [InlineData(" 8")]
    [InlineData("8 ")]
    [InlineData("99999999999999999999")]
    [InlineData("8.0")]
    public void InvalidExperimentalGraphRetentionDoesNotSilentlySelectAnotherPolicy(string value)
        => Assert.Throws<ArgumentException>(() => GlmDsaModel.ResolveFlashMtpGraphCacheCapacity(value));

    private static UnifiedMemoryModelOptions FlashOptions => new()
    {
        EnableGlmFlashMtp = true, EnableGlmSelectedExperts = true, EnableAdaptiveGlmPrefill = true,
        MaximumRunningRequests = 1, PrefixCheckpointCount = 0, RetainedConversationCount = 0, ContextTokens = 2048
    };

    [Theory]
    [InlineData("glm5next", true)]
    [InlineData("glm5-next", true)]
    [InlineData("glm-dsa", false)]
    [InlineData("qwen4exp", false)]
    [InlineData(null, false)]
    public void FactoryOptInCannotAuthorizeAnotherArchitecture(string? architecture, bool allowed)
    {
        if (allowed) UnifiedMemoryModelSession.ValidateSpeculativeArchitecture(architecture, FlashOptions);
        else Assert.Throws<NotSupportedException>(() => UnifiedMemoryModelSession.ValidateSpeculativeArchitecture(architecture, FlashOptions));
    }

    [Fact]
    public void FactoryQualificationRequiresTheWholeDeclaredLoadContract()
    {
        var speculation = new SpeculationOptions { Enabled = true };
        GlmUnifiedMemoryModelSession.ValidateFlashMtpLoadOptions(FlashOptions, speculation, null, "1");
        foreach (var invalid in new[]
        {
            FlashOptions with { EnableGlmFlashMtp = false },
            FlashOptions with { EnableGlmSelectedExperts = false },
            FlashOptions with { EnableAdaptiveGlmPrefill = false },
            FlashOptions with { MaximumRunningRequests = 2 },
            FlashOptions with { PrefixCheckpointCount = 1 },
            FlashOptions with { RetainedConversationCount = 1 },
            FlashOptions with { ContextTokens = 2049 }
        }) Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ValidateFlashMtpLoadOptions(invalid, speculation, null, "1"));
        Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ValidateFlashMtpLoadOptions(FlashOptions, speculation, null, null));
        Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ValidateFlashMtpLoadOptions(FlashOptions, speculation, "image.gguf", "1"));
    }

    private sealed class Scope : IDisposable { public void Dispose() { } }

    private static SchedulerConfig Config(RequestMemoryAdmission admission, int lanes = 1,
        bool prefix = false, bool fused = false, bool enabled = true) => new()
    {
        BlockSize = 8, NumBlocks = 32, MaxNumRunningSequences = lanes,
        EnablePrefixCaching = prefix, Speculation = new() { Enabled = enabled },
        MemoryAdmission = admission,
        ExecutionOptionsOverride = new() { BatchedPathDisabled = true,
            PerSeqFusedEnabled = fused, BatchedFusedDecodeEnabled = false }
    };

    [Theory]
    [InlineData(1, false, false, true, true)]
    [InlineData(2, false, false, true, false)]
    [InlineData(1, true, false, true, false)]
    [InlineData(1, false, true, true, false)]
    [InlineData(1, false, false, false, false)]
    public void DedicatedSpeculativeContractExcludesUnownedExecutionShapes(
        int lanes, bool prefix, bool fused, bool enabled, bool accepted)
    {
        var budget = new MemoryBudget([new("device", 1024)]);
        var admission = new RequestMemoryAdmission(budget, _ => [new("device", 128)])
        {
            AllowSpeculativeSerialExecution = true, ReestimateAfterPhysicalCheck = true,
            EnterSerialExecution = _ => new Scope(), BeforeAdmission = _ => [new("device", 1024)]
        };
        var config = Config(admission, lanes, prefix, fused, enabled);
        if (accepted) admission.ValidateConfiguration(config);
        else Assert.Throws<ArgumentException>(() => admission.ValidateConfiguration(config));
    }

    [Fact]
    public void SpeculativeContractRequiresPhysicalChecksAndExplicitExecutionScope()
    {
        var budget = new MemoryBudget([new("device", 1024)]);
        var admission = new RequestMemoryAdmission(budget, _ => []) { AllowSpeculativeSerialExecution = true };
        Assert.Throws<ArgumentException>(() => admission.ValidateConfiguration(Config(admission)));
    }

    [Theory]
    [InlineData(false, "auto", false, 8, false, 0)]
    [InlineData(true, "ngram", false, 8, false, 0)]
    [InlineData(true, "auto", false, 8, true, 2)]
    [InlineData(true, "draft-head", true, 1, true, 1)]
    [InlineData(true, "draft-head", true, 2, true, 2)]
    [InlineData(true, "draft-head", true, 3, false, 0)]
    public void DraftWindowNeverSilentlyOverridesAnUnsupportedExplicitRequest(
        bool enabled, string algorithm, bool explicitWindow, int window, bool accepted, int expected)
    {
        var requested = new SpeculationOptions { Enabled = enabled, SpeculatorName = algorithm,
            MaxDraftTokens = window, MaxDraftTokensExplicit = explicitWindow };
        if (!accepted)
        {
            Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ResolveFlashMtpSpeculation(requested));
            return;
        }
        var actual = GlmUnifiedMemoryModelSession.ResolveFlashMtpSpeculation(requested);
        Assert.Equal(SpeculatorRegistry.DraftHead, actual.SpeculatorName);
        Assert.True(actual.MaxDraftTokensExplicit);
        Assert.Equal(expected, actual.MaxDraftTokens);
        Assert.Equal(window, requested.MaxDraftTokens);
    }
}
