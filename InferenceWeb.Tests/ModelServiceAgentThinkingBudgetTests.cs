// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.AgentHost.Agents;
using TensorSharp.Runtime;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class ModelServiceAgentThinkingBudgetTests
{
    [Theory]
    [InlineData(512)]
    [InlineData(0)]
    [InlineData(null)]
    public void DescendantsRetainTheRequestsThinkingLimitWithoutInheritingPrivateTurnState(int? limit)
    {
        var rootProfiles = new[] { Profile("Root public instructions.") };
        var childProfiles = new[] { Profile("Shared child instructions.") };
        var parent = new ChatTurnContext
        {
            ThinkingTokenLimit = limit,
            PublicPrefixCandidates = rootProfiles,
            CacheScope = "parent-private-cache-scope",
            LoggedSharedPrefix = "parent-prefix-checkpoint",
            StagedAttachmentNames = new Dictionary<string, string> { ["private/source"] = "parent-upload.txt" },
        };
        var child = ModelService.CreateChildTurnContext(parent, childProfiles);
        var sibling = ModelService.CreateChildTurnContext(parent, childProfiles);
        var grandchild = ModelService.CreateChildTurnContext(child, childProfiles);
        foreach (var turn in new[] { child, sibling, grandchild })
        {
            Assert.Equal(limit, turn.ThinkingTokenLimit);
            Assert.Same(childProfiles, turn.PublicPrefixCandidates);
            Assert.Null(turn.CacheScope);
            Assert.Null(turn.LoggedSharedPrefix);
            Assert.Null(turn.StagedAttachmentNames);
        }
        Assert.NotSame(parent, child);
        Assert.NotSame(child, sibling);
        child.ThinkingTokenLimit = 128;
        child.CacheScope = "child-private-cache-scope";
        Assert.Equal(limit, parent.ThinkingTokenLimit);
        Assert.Equal(limit, sibling.ThinkingTokenLimit);
        Assert.Equal(limit, grandchild.ThinkingTokenLimit);
        Assert.Equal("parent-private-cache-scope", parent.CacheScope);
        Assert.Same(rootProfiles, parent.PublicPrefixCandidates);
        Assert.Null(sibling.CacheScope);
    }

    [Theory]
    [InlineData(512, 512)]
    [InlineData(0, 6144)]
    [InlineData(null, 6144)]
    public void ChildGenerationUsesTheInheritedLimitInsteadOfResettingToItsOutputAllowance(int? limit, int expected)
    {
        var child = ModelService.CreateChildTurnContext(new ChatTurnContext { ThinkingTokenLimit = limit }, Array.Empty<MultiAgentPromptProfile>());
        string? previous = Environment.GetEnvironmentVariable("TS_THINKING_BUDGET");
        try
        {
            Environment.SetEnvironmentVariable("TS_THINKING_BUDGET", null);
            Assert.Equal(expected, ChatGenerationPipeline.ThinkingBudgetFor(8192, true, child.ThinkingTokenLimit));
        }
        finally { Environment.SetEnvironmentVariable("TS_THINKING_BUDGET", previous); }
    }

    private static MultiAgentPromptProfile Profile(string instructions) => new(
        new[] { new ChatMessage { Role = "system", Content = instructions } }, Array.Empty<ToolFunction>());
}
