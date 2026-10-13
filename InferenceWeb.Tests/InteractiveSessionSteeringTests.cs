// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Threading.Channels;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Cli;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class InteractiveSessionSteeringTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"ts-cli-steer-{Guid.NewGuid():N}.gguf");
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(10);

    public InteractiveSessionSteeringTests() => BackendFailureWarmupTests.WriteProbeGguf(_path);
    public void Dispose() => File.Delete(_path);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task SteeringRetainsThePartialAnswerOrReasoningInTheNextPrompt(bool thinking)
    {
        using var model = new StreamingModel(_path, thinking);
        var input = Channel.CreateUnbounded<string>();
        var captured = new TaskCompletionSource<List<ChatMessage>>(TaskCreationOptions.RunContinuationsAsynchronously);
        var renderer = new ProbeRenderer(thinking, messages =>
        {
            if (messages.LastOrDefault()?.Content == "make it concise")
                captured.TrySetResult(messages);
        });
        var session = CreateSession(model, renderer, thinking, maxTokens: 1024);
        TextWriter originalOutput = Console.Out;
        int printed = 0;
        using var output = new ObservingWriter(piece =>
        {
            if (piece == "x" && Interlocked.Increment(ref printed) == 5)
                input.Writer.TryWrite("make it concise");
        });
        Console.SetOut(output);
        Task running = Task.Run(() => session.RunWithInput(() => input.Reader.ReadAsync().AsTask(), allowSteering: true));
        try
        {
            await input.Writer.WriteAsync("explain everything");
            List<ChatMessage> history = await captured.Task.WaitAsync(Timeout);
            Assert.Equal(new[] { "user", "assistant", "user" }, history.Select(message => message.Role));
            Assert.Equal("explain everything", history[0].Content);
            Assert.Equal("make it concise", history[2].Content);
            ChatMessage partial = history[1];
            Assert.StartsWith("xxxxx", thinking ? partial.Thinking : partial.Content);
            Assert.Null(partial.RawOutputTokens);
            Assert.Null(partial.ToolCalls);
            await input.Writer.WriteAsync("/exit");
            await running.WaitAsync(Timeout);
            Assert.DoesNotContain("[error]", output.ToString());
        }
        finally
        {
            input.Writer.TryWrite("/exit");
            try { await running.WaitAsync(Timeout); }
            finally { Console.SetOut(originalOutput); }
        }
    }

    [Fact]
    public void MultilineCommandsChangeHowSubsequentPromptsAreRead()
    {
        using var model = new StreamingModel(_path, thinking: false);
        var prompts = new List<string>();
        var renderer = new ProbeRenderer(false, history =>
        {
            if (history.LastOrDefault()?.Role == "user")
                prompts.Add(history[^1].Content);
        });
        var session = CreateSession(model, renderer, thinking: false, maxTokens: 1);
        using var input = new StringReader("/multiline on\nfirst line\nsecond line\n.\n/multiline off\n.\nthird prompt\n/exit\n");
        TextWriter originalOutput = Console.Out;
        using var output = new StringWriter();
        Console.SetOut(output);
        try
        {
            session.RunWithInput(input, allowSteering: false);
            Assert.Equal(new[] { "first line\nsecond line", "third prompt" }, prompts);
            Assert.DoesNotContain("[error]", output.ToString());
        }
        finally
        {
            Console.SetOut(originalOutput);
        }
    }

    private InteractiveSession CreateSession(ModelBase model, IPromptRenderer renderer, bool thinking, int maxTokens)
        => new(model, _path, BackendType.Cpu, null, renderer, SamplingConfig.Greedy, null,
            thinking, maxTokens, NullLogger.Instance) { PrefixCacheEnabled = false };

    private sealed class ObservingWriter : StringWriter
    {
        private readonly Action<string> _observe;
        public ObservingWriter(Action<string> observe) => _observe = observe;
        public override void Write(string value)
        {
            base.Write(value);
            _observe(value);
        }
    }

    private sealed class ProbeRenderer : IPromptRenderer
    {
        private readonly bool _thinking;
        private readonly Action<List<ChatMessage>> _capture;
        public ProbeRenderer(bool thinking, Action<List<ChatMessage>> capture)
        {
            _thinking = thinking;
            _capture = capture;
        }
        public string Render(string template, List<ChatMessage> messages, bool addGenerationPrompt = true,
            string architecture = null, List<ToolFunction> tools = null, bool enableThinking = false)
        {
            _capture(messages);
            // Keep the raw-token placeholder the KV renderer substitutes for a
            // completed assistant, just as a real template renders message content.
            return "abcdefgh" + string.Join("\n", messages.Select(message => message.Content))
                + (_thinking ? "<|channel>thought\n" : "");
        }
    }

    private sealed class StreamingModel : ModelBase
    {
        public StreamingModel(string path, bool thinking) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = thinking ? "gemma4" : "probe", VocabSize = 128, NumLayers = 1 };
            Tokenizer = new CharacterTokenizer();
        }
        protected override float[] ForwardCore(int[] tokens)
        {
            // A paced fake makes interruption happen during live output even when
            // the inference worker runs ahead of the console consumer.
            Thread.Sleep(2);
            var logits = new float[128];
            logits['x'] = 10;
            return logits;
        }
        protected override void ResetKVCacheCore() { }
    }

    private sealed class CharacterTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 128).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => 128;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 1 };
        public bool IsEos(int id) => id == 1;
        public int LookupToken(string token) => token.Length == 1 ? token[0] : -1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> tokens) => new(tokens.Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.AddRange(Encoding.UTF8.GetBytes(((char)token).ToString()));
    }
}
