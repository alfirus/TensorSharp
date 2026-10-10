// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.RegularExpressions;
using System.Security.Cryptography;
using System.Text;
using TensorSharp.Runtime;

namespace DirectCudaProbe;

internal sealed record QualitySelection(string RawText, string Content, string Thinking, ToolCall[] ToolCalls);

internal sealed record QualityPrompt(string Text, string Utf8Sha256)
{
    public static string? ParseReasoningEffort(string? value)
    {
        if (value is null) return null; // Preserve the historical family default.
        if (!ReasoningEffort.TryNormalize(value, out string? normalized) || normalized is null)
            throw new ArgumentException("quality-reasoning-effort must be low, medium or high when provided.");
        return normalized;
    }

    public static QualityPrompt Create(string template, string architecture, string text, string? reasoningEffort)
    {
        string rendered = new GgufPromptRenderer().Render(template, new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are a helpful assistant." },
            new() { Role = "user", Content = text },
        }, addGenerationPrompt: true, architecture: architecture, tools: null,
            enableThinking: false, reasoningEffort: reasoningEffort);
        return new(rendered, Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(rendered))).ToLowerInvariant());
    }
}

internal static class QualityOutput
{
    public static string[] SelectCases(string value)
    {
        string[] cases = value.Split(',', StringSplitOptions.TrimEntries);
        if (cases.Length == 0 || cases.Distinct(StringComparer.Ordinal).Count() != cases.Length ||
            cases.Any(name => name is not ("math" or "extraction" or "squares")))
            throw new ArgumentException("quality-cases must be a nonempty, duplicate-free subset of math,extraction,squares.");
        return cases;
    }

    public static QualitySelection Parse(string raw, string architecture, string prompt, string mode)
    {
        if (mode == "raw")
        {
            // Preserve the probe's historical default. Only an explicitly empty
            // thinking block is removed, never arbitrary reasoning or prose.
            return new(raw, Regex.Replace(raw, @"^\s*<think>\s*</think>\s*", "").Trim(), "", []);
        }
        if (mode != "production") throw new ArgumentException("quality-parser must be raw or production.");
        IOutputParser parser = OutputParserFactory.Create(architecture);
        parser.Init(enableThinking: false, tools: null);
        parser.SetGenerationPromptSuffix(prompt);
        ParsedOutput parsed = parser.Add(raw, done: true);
        return new(raw, parsed.Content, parsed.Thinking, parsed.ToolCalls?.ToArray() ?? []);
    }

    // Parsing a final fragment does not prove generation completed. The actual
    // checkpoint EOS must have been sampled, and these are answer-only tasks.
    public static bool CompletePlainAnswer(QualitySelection output, bool eos) => eos && output.ToolCalls.Length == 0;
}
