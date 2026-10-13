// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using TensorSharp.Server;
using TensorSharp.Server.ProtocolAdapters;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The cap that stops a reasoning model spending an entire token allowance inside its
/// thinking channel and returning nothing.
///
/// <para>
/// Measured on the algorithmic-art skill before this existed: 8000 tokens generated,
/// 100% of them thinking, 888 seconds, and an empty answer reported to the caller as a
/// bare <c>truncated: true</c>. The same run's pptx scenario spent 94.6% of its tokens
/// thinking and took 936 seconds to produce a two-slide deck.
/// </para>
/// </summary>
[Collection(EngineEnvironmentCollection.Name)]
public class ThinkingBudgetTests
{
    private static string? Saved;

    private static void WithEnv(string? value, Action body)
    {
        Saved = Environment.GetEnvironmentVariable("TS_THINKING_BUDGET");
        try
        {
            Environment.SetEnvironmentVariable("TS_THINKING_BUDGET", value);
            body();
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_THINKING_BUDGET", Saved);
        }
    }

    [Fact]
    public void ThinkingOff_MeansNoCap()
    {
        // Nothing to cap: a non-reasoning turn has no thinking channel to run away in.
        WithEnv(null, () =>
            Assert.Equal(0, ChatGenerationPipeline.ThinkingBudgetFor(8000, enableThinking: false)));
    }

    [Fact]
    public void ALargeAllowance_LeavesAQuarterForTheAnswer()
    {
        WithEnv(null, () =>
        {
            Assert.Equal(6000, ChatGenerationPipeline.ThinkingBudgetFor(8000, enableThinking: true));
            Assert.Equal(1500, ChatGenerationPipeline.ThinkingBudgetFor(2000, enableThinking: true));
        });
    }

    [Fact]
    public void KnowledgePlanningLimitKeepsLongAnswerBudgetWithoutThousandsOfReasoningTokens()
    {
        WithEnv(null, () =>
        {
            Assert.Equal(512, ChatGenerationPipeline.ThinkingBudgetFor(8192, true, turnLimit: 512));
            Assert.Equal(384, ChatGenerationPipeline.ThinkingBudgetFor(512, true, turnLimit: 512));
            Assert.Equal(6144, ChatGenerationPipeline.ThinkingBudgetFor(8192, true));
            Assert.Equal(0, ChatGenerationPipeline.ThinkingBudgetFor(8192, false, turnLimit: 512));
        });
        WithEnv("1200", () => Assert.Equal(1200,
            ChatGenerationPipeline.ThinkingBudgetFor(8192, true, turnLimit: 512)));
        WithEnv("0", () => Assert.Equal(0,
            ChatGenerationPipeline.ThinkingBudgetFor(8192, true, turnLimit: 512)));
    }

    [Fact]
    public void ASmallAllowance_IsNotCapped()
    {
        // Capping a 200-token turn at 150 would fire on ordinary short reasoning, where
        // there is no runaway to prevent and the cost of a false positive is a worse
        // answer.
        WithEnv(null, () =>
        {
            Assert.Equal(0, ChatGenerationPipeline.ThinkingBudgetFor(200, enableThinking: true));
            Assert.Equal(0, ChatGenerationPipeline.ThinkingBudgetFor(511, enableThinking: true));
            Assert.True(ChatGenerationPipeline.ThinkingBudgetFor(512, enableThinking: true) > 0);
        });
    }

    [Fact]
    public void TheEnvironmentVariableOverridesTheDefault()
    {
        WithEnv("1200", () =>
            Assert.Equal(1200, ChatGenerationPipeline.ThinkingBudgetFor(8000, enableThinking: true)));
    }

    [Fact]
    public void SettingItToZero_DisablesTheCapEntirely()
    {
        // For a deployment that would rather have unbounded reasoning than a guaranteed
        // answer. It has to be expressible, or the only way out is a code change.
        WithEnv("0", () =>
            Assert.Equal(0, ChatGenerationPipeline.ThinkingBudgetFor(8000, enableThinking: true)));
    }

    [Fact]
    public void ANonsenseOverride_FallsBackToTheDefault()
    {
        WithEnv("not-a-number", () =>
            Assert.Equal(6000, ChatGenerationPipeline.ThinkingBudgetFor(8000, enableThinking: true)));
    }

    [Fact]
    public void AThinkingBudgetStop_CountsAsTruncated()
    {
        // The client-visible finish reason is a length stop either way; the difference
        // is only that this one was caught early enough to explain.
        Assert.True(FinishReasonMapper.IsTruncated(FinishReasonMapper.PipelineThinkingBudget));
        Assert.True(FinishReasonMapper.IsTruncated(FinishReasonMapper.PipelineMaxTokens));
        Assert.False(FinishReasonMapper.IsTruncated("stop_sequence"));
        Assert.False(FinishReasonMapper.IsTruncated("eos"));
    }
}
