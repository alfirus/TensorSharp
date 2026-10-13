// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections.Generic;

namespace TensorSharp.Server;

/// <summary>Counts reasoning pieces for protocols whose channels do not use a
/// single thinking-close marker. Answer and tool-call tokens never consume this budget.</summary>
internal sealed class ThinkingBudgetChannelCounter
{
    private readonly IOutputParser _parser;
    internal int ThinkingTokens { get; private set; }

    internal ThinkingBudgetChannelCounter(string architecture, bool enableThinking,
        List<ToolFunction> tools, string generationSuffix)
    {
        _parser = OutputParserFactory.Create(architecture);
        _parser.Init(enableThinking, tools);
        _parser.SetGenerationPromptSuffix(generationSuffix);
    }

    internal bool Observe(string piece, int budget)
    {
        // The caller supplies one decoded token piece, including empty pieces from
        // incomplete UTF-8. Count these while the reasoning channel remains open.
        bool wasThinking = _parser.IsThinkingChannelOpen;
        var parsed = _parser.Add(piece, false);
        if (string.IsNullOrEmpty(parsed.Thinking) && !(wasThinking && _parser.IsThinkingChannelOpen))
            return false;
        ThinkingTokens++;
        return budget > 0 && ThinkingTokens >= budget;
    }
}
