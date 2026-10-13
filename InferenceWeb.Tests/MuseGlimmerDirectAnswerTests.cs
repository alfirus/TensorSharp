// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

public sealed class MuseGlimmerDirectAnswerTests
{
    private const string AnswerSuffix = " to=user<|message|>";
    private static readonly string Template = ReadTemplate();
    private static readonly List<ToolFunction> Tools =
    [
        new() { Name = "knowledge_search", Description = "Search reports.",
            Parameters = new Dictionary<string, ToolParameter> { ["query"] = new() { Type = "string" } } },
    ];

    private static List<ChatMessage> Messages() =>
    [
        new() { Role = "system", Content = "Use the provided evidence." },
        new() { Role = "user", Content = "Compare Jonathan's grades and teacher comments." },
    ];

    [Theory]
    [InlineData(null, "high")]
    [InlineData("", "high")]
    [InlineData("low", "low")]
    [InlineData(" MEDIUM ", "medium")]
    [InlineData("high", "high")]
    public void ActualTemplateHonorsExplicitEffortAndPreservesItsOmittedDefault(string? effort, string expected)
    {
        foreach (string architecture in new[] { "muse-glimmer", "muse_glimmer" })
        {
            string prompt = ChatTemplate.RenderFromGgufTemplate(Template, Messages(),
                architecture: architecture, tools: Tools, enableThinking: true, reasoningEffort: effort);
            Assert.Contains($"Reasoning strength: {expected}.", prompt);
            Assert.EndsWith("<|start|>assistant", prompt);
            Assert.True(ChatProtocolRegistry.For(architecture)!.RendersReasoningEffort);
        }
        // A foreign template using the same variable keeps its own semantics.
        Assert.Contains("Reasoning strength: high.", ChatTemplate.RenderFromGgufTemplate(
            Template, Messages(), architecture: "unrelated", enableThinking: true, reasoningEffort: "low"));
    }

    [Fact]
    public void BoundedKnowledgePlanningUsesMediumEffortWithoutMutatingSharedSettingsOrDroppingGrammar()
    {
        var original = new SamplingConfig
        {
            Temperature = 0.37f, StopSequences = new() { "END" },
            Grammar = new GrammarConstraint(Grammar.JsonObject(), new ByteTokenizer()),
        };
        SamplingConfig adjusted = ModelService.SamplingForBoundedKnowledgePlanning(
            "muse-glimmer", original, hasKnowledge: true, enableThinking: true, thinkingTokenLimit: 512);
        Assert.NotSame(original, adjusted);
        Assert.Null(original.ReasoningEffort);
        Assert.Equal("medium", adjusted.ReasoningEffort);
        Assert.Equal(original.Temperature, adjusted.Temperature);
        Assert.Equal(original.StopSequences, adjusted.StopSequences);
        Assert.NotSame(original.StopSequences, adjusted.StopSequences);
        Assert.NotNull(adjusted.Grammar);
        Assert.NotSame(original.Grammar, adjusted.Grammar);
        Assert.Contains("Reasoning strength: medium.", ChatTemplate.RenderFromGgufTemplate(Template, Messages(),
            architecture: "muse-glimmer", tools: Tools, enableThinking: true, reasoningEffort: adjusted.ReasoningEffort));
        Assert.Equal("medium", ModelService.SamplingForBoundedKnowledgePlanning(
            "muse_glimmer", null, true, true, 512).ReasoningEffort);
    }

    [Theory]
    [InlineData("muse-glimmer", true, true, 512, "low")]
    [InlineData("muse-glimmer", true, true, 512, "medium")]
    [InlineData("muse-glimmer", true, true, 512, "high")]
    [InlineData("muse-glimmer", false, true, 512, null)]
    [InlineData("muse-glimmer", true, false, 512, null)]
    [InlineData("muse-glimmer", true, true, null, null)]
    [InlineData("muse-glimmer", true, true, 0, null)]
    [InlineData("gpt-oss", true, true, 512, null)]
    public void PlanningDefaultPreservesExplicitChoicesAndUnrelatedRequests(string architecture,
        bool hasKnowledge, bool thinking, int? limit, string? effort)
    {
        var original = new SamplingConfig { ReasoningEffort = effort };
        Assert.Same(original, ModelService.SamplingForBoundedKnowledgePlanning(
            architecture, original, hasKnowledge, thinking, limit));
    }

    [Theory]
    [InlineData(false, false, "muse-glimmer", AnswerSuffix)]
    [InlineData(false, false, "muse_glimmer", AnswerSuffix)]
    [InlineData(true, false, "muse-glimmer", "")]
    [InlineData(false, true, "muse-glimmer", "")]
    [InlineData(true, true, "muse-glimmer", "")]
    public void ActualTemplate_PrimesAnswerOnlyWhenThinkingAndToolsAreOff(
        bool thinking, bool hasTools, string architecture, string suffix)
    {
        string prompt = ChatTemplate.RenderFromGgufTemplate(Template, Messages(),
            architecture: architecture, tools: hasTools ? Tools : null, enableThinking: thinking);
        Assert.EndsWith("<|start|>assistant" + suffix, prompt);
        Assert.Contains("Compare Jonathan's grades and teacher comments.", prompt);
        Assert.Equal(hasTools, prompt.Contains("Search reports.", StringComparison.Ordinal));

        var tokenizer = new ByteTokenizer();
        string recorded = ChatGenerationPipeline.RecordedGenerationSuffix(
            tokenizer, tokenizer.Encode(prompt), architecture, thinking, Template);
        Assert.Equal(suffix, recorded);
        Assert.Equal(suffix.Length > 0,
            ChatGenerationPipeline.AnnouncesGenerationSuffix(architecture, Template, recorded));
        // Legacy raw rounds started mid-header and must retain their empty fallback.
        Assert.Empty(KVCachePromptRenderer.GetAssistantGenerationSuffix(architecture, thinking, Template));
    }

    [Fact]
    public void EmptyToolListAlsoPrimesAnswer_AndHistoryHasNoExtraGenerationHeader()
    {
        Assert.EndsWith("<|start|>assistant" + AnswerSuffix,
            ChatTemplate.RenderFromGgufTemplate(Template, Messages(), architecture: "muse-glimmer", tools: []));
        string history = ChatTemplate.RenderFromGgufTemplate(Template, Messages(),
            addGenerationPrompt: false, architecture: "muse-glimmer");
        Assert.EndsWith("comments.<|eot|>", history);
    }

    [Fact]
    public void ExistingTemplateRoutingAndUnrelatedArchitecturesAreUnchanged()
    {
        // A publisher that already chooses a channel owns that boundary; only a bare
        // known assistant generation header is eligible for completion.
        const string routedTemplate =
            "{% for m in messages %}{{ m.content }}{% endfor %}<|start|>assistant to=self<|message|>";
        Assert.EndsWith("<|start|>assistant to=self<|message|>",
            ChatTemplate.RenderFromGgufTemplate(routedTemplate, Messages(), architecture: "muse-glimmer"));
        Assert.EndsWith("<|start|>assistant",
            ChatTemplate.RenderFromGgufTemplate(Template, Messages(), architecture: "unrelated"));
    }

    [Theory]
    [InlineData("Jonathan's mathematics is strongest.")]
    [InlineData("Jonathan的数学成绩更好。")]
    [InlineData("{\"strongest\":\"mathematics\"}")]
    public void PrimedAnswerStreamsItsFirstCharacter_WithoutFramingOrThinking(string answer)
    {
        var parser = OutputParserFactory.Create("muse-glimmer");
        parser.Init(false, null);
        parser.SetGenerationPromptSuffix(AnswerSuffix);
        var content = new StringBuilder();
        foreach (char c in answer)
        {
            ParsedOutput parsed = parser.Add(c.ToString(), done: false);
            Assert.Equal(c.ToString(), parsed.Content);
            Assert.Empty(parsed.Thinking);
            Assert.Null(parsed.ToolCalls);
            content.Append(parsed.Content);
        }
        foreach (char c in "<|eot|>")
            content.Append(parser.Add(c.ToString(), done: false).Content);
        content.Append(parser.Add("", done: true).Content);
        Assert.Equal(answer, content.ToString());

        // Init resets the parser for a subsequent tool-capable or thinking round.
        parser.Init(true, Tools);
        parser.SetGenerationPromptSuffix("");
        ParsedOutput next = parser.Add(" to=self<|message|>check evidence<|eom|>", true);
        Assert.Empty(next.Content);
        Assert.Equal("check evidence", next.Thinking);
    }

    [Fact]
    public void StructuredGrammarAlsoReceivesThePrimedBoundary()
    {
        var sampling = new SamplingConfig
        {
            Grammar = new GrammarConstraint(Grammar.JsonObject(), new ByteTokenizer()),
        };
        Assert.True(OutputParserFactory.ConstrainsReplyStart(sampling));
        Assert.True(ChatGenerationPipeline.AnnouncesGenerationSuffix(
            "muse-glimmer", Template, AnswerSuffix, sampling));
        Assert.False(ChatGenerationPipeline.AnnouncesGenerationSuffix(
            "muse-glimmer", Template, "", sampling));
        const string json = "{\"strongest\":\"mathematics\"}";
        EmittedAssistantTurn emitted = ChatGenerationPipeline.BuildEmittedTurn(
            "muse-glimmer", json, false, null, AnswerSuffix, false);
        Assert.Equal(json, emitted.Content);
        Assert.True(string.IsNullOrEmpty(emitted.Thinking));
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, false)]
    public void SwitchingThinkingModePreservesEachRecordedAssistantBoundary(
        bool firstThinking, bool nextThinking)
    {
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var tokenizer = new ByteTokenizer();
        var first = Messages();
        List<int> prompt1 = renderer.RenderToTokens(tokenizer, Template, first, "muse-glimmer", true,
            out _, out string trailing, tools: null, enableThinking: firstThinking);
        string suffix = ChatGenerationPipeline.RecordedGenerationSuffix(
            tokenizer, prompt1, "muse-glimmer", firstThinking, Template);
        Assert.Equal(firstThinking ? "" : AnswerSuffix, suffix);
        List<int> raw = tokenizer.Encode(firstThinking
            ? " to=self<|message|>Compare grades.<|eom|><|start|>assistant to=user<|message|>Mathematics."
            : "Mathematics.", addSpecial: false);
        var next = new List<ChatMessage>(first)
        {
            new() { Role = "assistant", Content = "Mathematics.", RawOutputTokens = raw,
                RawGenerationSuffix = suffix, RawPromptTrailingWhitespace = trailing },
            new() { Role = "user", Content = "Which area needs improvement?" },
        };
        List<int> prompt2 = renderer.RenderToTokens(tokenizer, Template, next, "muse-glimmer", true,
            tools: null, enableThinking: nextThinking);
        var cached = new List<int>(prompt1);
        cached.AddRange(raw);
        cached.AddRange(tokenizer.Encode("<|eot|>", addSpecial: false));
        Assert.Equal(cached, prompt2.Take(cached.Count));
        Assert.EndsWith("<|start|>assistant" + (nextThinking ? "" : AnswerSuffix), tokenizer.Decode(prompt2));
    }

    private static string ReadTemplate()
    {
        using Stream stream = typeof(MuseGlimmerDirectAnswerTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.MuseGlimmer.chat_template.jinja")!;
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    private sealed class ByteTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 256).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => 256;
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public bool IsEos(int id) => false;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true)
            => Encoding.UTF8.GetBytes(text ?? string.Empty).Select(b => (int)b).ToList();
        public string Decode(List<int> tokens) => Encoding.UTF8.GetString(tokens.Select(t => (byte)t).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }
}
