// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Server;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class ThinkingChannelBudgetTests
{
    [Fact]
    public void UnexpectedMuseThinkingIsCountedEvenWhenTheRequestDisabledIt()
    {
        var counter = new ThinkingBudgetChannelCounter("muse-glimmer", false, null, "<|start|>assistant");
        Assert.False(counter.Observe(" to=self<|message|>", 2));
        Assert.False(counter.Observe("Unrequested reasoning.", 2));
        Assert.True(counter.Observe("More unrequested reasoning.", 2));
    }

    [Fact]
    public void IncompleteUtf8TokensInsideReasoningStillConsumeBudget()
    {
        var counter = new ThinkingBudgetChannelCounter("muse-glimmer", true, null, "<|start|>assistant");
        Assert.False(counter.Observe(" to=self<|message|>", 3));
        Assert.False(counter.Observe("", 3));
        Assert.False(counter.Observe("中", 3));
        Assert.True(counter.Observe("", 3));
    }

    [Fact]
    public void MuseAnswerAfterBriefReasoningCanExceedTheThinkingBudget()
    {
        var counter = new ThinkingBudgetChannelCounter("muse-glimmer", true, null, "<|start|>assistant");
        Assert.False(counter.Observe(" to=self<|message|>", 3));
        Assert.False(counter.Observe("Compare the grades.", 3));
        Assert.False(counter.Observe("<|eom|><|start|>assistant to=user<|message|>", 3));
        for (int i = 0; i < 20; i++)
            Assert.False(counter.Observe("Visible answer. ", 3));
        Assert.Equal(1, counter.ThinkingTokens);
    }

    [Fact]
    public void MuseSplitHeadersAndRepeatedReasoningChannelsAreCounted()
    {
        var counter = new ThinkingBudgetChannelCounter("muse-glimmer", true, null, "<|start|>assistant");
        foreach (string piece in new[] { " to=", "self", "<|message", "|>" })
            Assert.False(counter.Observe(piece, 3));
        Assert.False(counter.Observe("First thought.", 3));
        Assert.False(counter.Observe("<|eom|>", 3));
        Assert.False(counter.Observe("<|start|>assistant to=self<|message|>", 3));
        Assert.False(counter.Observe("Second thought.", 3));
        Assert.True(counter.Observe("Third thought.", 3));
    }
}
