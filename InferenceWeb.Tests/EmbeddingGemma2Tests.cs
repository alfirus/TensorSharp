// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using TensorSharp.Models.Embeddings;
using TensorSharp.Models.Media;
using TensorSharp.Runtime;
using TensorSharp.Server.Knowledge;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class EmbeddingGemma2Tests(ITestOutputHelper output)
{
    private const string ModelVariable = "TENSORSHARP_EMBEDDINGGEMMA2_MODEL";

    [Fact]
    public void GemmaBpeMetadataPreservesSpacesMergeRanksAndTerminalEos()
    {
        string path = Path.Combine(Path.GetTempPath(), $"embeddinggemma2-tokenizer-{Guid.NewGuid():N}.gguf");
        try
        {
            using (var writer = new BinaryWriter(File.Create(path)))
            {
                writer.Write(Encoding.ASCII.GetBytes("GGUF"));
                writer.Write(3U); writer.Write(0UL); writer.Write(0UL); writer.Write(0UL);
            }
            using var file = new GgufFile(path);
            file.Metadata["general.architecture"] = "gemma-embedding2";
            file.Metadata["tokenizer.ggml.model"] = "gemma4";
            file.Metadata["tokenizer.ggml.pre"] = "gemma4";
            file.Metadata["tokenizer.ggml.tokens"] = new[] { "<unk>", "<eos>", "<bos>", "a", "b", "ab", "▁", "▁ab" };
            file.Metadata["tokenizer.ggml.token_type"] = new[] { 2, 3, 3, 1, 1, 1, 1, 1 };
            file.Metadata["tokenizer.ggml.merges"] = new[] { "a b", "▁ ab" };
            file.Metadata["tokenizer.ggml.unknown_token_id"] = 0U;
            file.Metadata["tokenizer.ggml.bos_token_id"] = 2U;
            file.Metadata["tokenizer.ggml.eos_token_id"] = 1U;
            file.Metadata["tokenizer.ggml.add_bos_token"] = true;
            file.Metadata["tokenizer.ggml.add_eos_token"] = true;
            var tokenizer = EmbeddingTokenizer.Create(file);
            Assert.Equal(new[] { 2, 7, 1 }, tokenizer.Encode(" ab"));
            Assert.Equal(new[] { 2, 5, 1 }, tokenizer.Encode("ab"));
            Assert.Equal(new[] { 7 }, tokenizer.Encode(" ab", addSpecial: false));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void KnowledgeRetrievalUsesGemmaTaskPromptsAndPreservesExplicitOverrides()
    {
        Assert.Equal(("task: search result | query: ", "title: none | text: "),
            GgufKnowledgeEmbeddingProvider.ResolvePrefixes("gemma-embedding2", null, null));
        Assert.Equal(("", ""), GgufKnowledgeEmbeddingProvider.ResolvePrefixes("gemma-embedding2", "", ""));
        Assert.Equal(("custom query: ", "title: none | text: "),
            GgufKnowledgeEmbeddingProvider.ResolvePrefixes("gemma-embedding2", "custom query: ", null));
        Assert.Equal(("", ""), GgufKnowledgeEmbeddingProvider.ResolvePrefixes("bert", null, null));
        Assert.Equal(("query: ", "passage: "),
            GgufKnowledgeEmbeddingProvider.ResolvePrefixes("nomic-bert", "query: ", "passage: "));
    }

    [Theory]
    [InlineData("describe <|image|>")]
    [InlineData("describe <|video|>")]
    [InlineData("listen <|audio|>")]
    [InlineData("<|image>")]
    [InlineData("<image|>")]
    [InlineData("<|audio>")]
    [InlineData("<audio|>")]
    public void TextMediaDelimitersAreDetectedBeforeUnmatchedSoftTokenInference(string text)
        => Assert.True(EmbeddingGemma2MultimodalEncoder.ContainsMediaDelimiter(text));

    [Theory]
    [InlineData("title: none | text: a photo of a cat")]
    [InlineData("task: sentence similarity | query: <img src='cat.png'>")]
    [InlineData("你好！ Café 😀")]
    [InlineData("<bos>hello<eos>")]
    public void OrdinaryTextDoesNotRequireMediaParts(string text)
        => Assert.False(EmbeddingGemma2MultimodalEncoder.ContainsMediaDelimiter(text));

    [Theory]
    [InlineData(24, 96, 1, 32, new[] { 0, 24, 48, 72 })]
    [InlineData(30, 45, 1, 32, new[] { 0 })]
    [InlineData(24, 240, 1, 3, new[] { 0, 96, 216 })]
    [InlineData(30, 6, 1, 32, new[] { 0 })]
    public void VideoSamplingUsesSourceTimeAndCoversClipsBeyondTheFrameBudget(double sourceFps, int frames,
        double requestedFps, int budget, int[] expected)
        => Assert.Equal(expected, EmbeddingGemma2MultimodalEncoder.SampleVideoFrames(new VideoInfo(sourceFps, frames, 192, 192), requestedFps, budget));

    [Fact]
    public void VideoSamplingRejectsInvalidRatesAndBudgets()
    {
        Assert.Throws<InvalidDataException>(() => EmbeddingGemma2MultimodalEncoder.SampleVideoFrames(new VideoInfo(0, 4, 192, 192), 1, 32));
        Assert.Throws<InvalidDataException>(() => EmbeddingGemma2MultimodalEncoder.SampleVideoFrames(new VideoInfo(1, 0, 192, 192), 1, 32));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2MultimodalEncoder.SampleVideoFrames(new VideoInfo(1, 4, 192, 192), double.NaN, 32));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2MultimodalEncoder.SampleVideoFrames(new VideoInfo(1, 4, 192, 192), 1, 0));
    }

    [ModelFact(ModelVariable)]
    public void RealTokenizerMatchesIndependentLlamaCppOracleForUnicodeCodeAndSpecialTokens()
    {
        using var file = new GgufFile(ModelPath);
        var tokenizer = EmbeddingTokenizer.Create(file);
        using var oracle = ReadOracle();
        Assert.Equal("gemma-embedding2", file.GetString("general.architecture"));
        Assert.Equal(oracle.RootElement.GetProperty("bos_token_id").GetInt32(), tokenizer.BosTokenId);
        Assert.Contains(oracle.RootElement.GetProperty("eos_token_id").GetInt32(), tokenizer.EosTokenIds);
        foreach (var item in oracle.RootElement.GetProperty("cases").EnumerateArray())
        {
            string text = item.GetProperty("text").GetString()!;
            int[] expected = item.GetProperty("tokens").EnumerateArray().Select(token => token.GetInt32()).ToArray();
            Assert.True(expected.SequenceEqual(tokenizer.Encode(text)), $"Tokenizer mismatch for {JsonSerializer.Serialize(text)}");
        }
    }

    [ModelFact(ModelVariable)]
    public void ManagedCpuLoadsGemmaWithoutAHiddenNativeBackendFallback()
    {
        using var model = EmbeddingModel.Load(ModelPath);
        Assert.True(model.IsManaged);
        Assert.Equal("CPU", model.Backend);
        Assert.Equal(768, model.Dimensions);
    }

    [ModelFact(ModelVariable)]
    public async Task ManagedCpuBatchesPreserveOrderDuplicatesAnd768DimensionalProjection()
        => await VerifyBatch("CPU");

    [EmbeddingGemma2MlxFact]
    public async Task MlxBatchesPreserveOrderDuplicatesAnd768DimensionalProjection()
        => await VerifyBatch("MLX");

    [EmbeddingGemma2MlxFact]
    public async Task MlxMixedShortSequencesCrossing1024TokensRemainIndependentAndOrdered()
    {
        using var model = EmbeddingModel.Load(ModelPath, new() { Backend = "MLX", Threads = 4 });
        int[] seed = model.Tokenize("title: add.py | text: def add(a, b): return a + b");
        int[] lengths = [17, 65, 129, 193, 257, 321, 33, 161];
        int[][] inputs = lengths.Select((length, variant) =>
        {
            var row = new int[length]; row[0] = seed[0]; row[^1] = seed[^1];
            for (int i = 1; i < length - 1; ++i) row[i] = seed[1 + (i - 1 + variant) % (seed.Length - 2)];
            return row;
        }).ToArray();
        Assert.InRange(inputs.Sum(row => row.Length), 1025, 1536);
        var batch = await model.EmbedTokensAsync(inputs);
        var reversed = await model.EmbedTokensAsync(inputs.Reverse().ToArray());
        Assert.Equal(inputs.Sum(row => row.Length), batch.PromptTokens);
        Assert.Equal(inputs.Length, batch.Embeddings.Length);
        for (int i = 0; i < inputs.Length; ++i)
        {
            float[] single = (await model.EmbedTokensAsync([inputs[i]])).Embeddings[0];
            AssertNormalized(batch.Embeddings[i]);
            double similarity = Cosine(single, batch.Embeddings[i]);
            Assert.True(similarity > .99999, $"Mixed-short MLX length {inputs[i].Length}: cosine {similarity:R}");
            Assert.True(Cosine(batch.Embeddings[i], reversed.Embeddings[inputs.Length - i - 1]) > .99999);
        }
    }

    [EmbeddingGemma2MlxFact]
    public async Task MlxPackedLongSequencesPreserveSlidingWindowPositionsAndInputOrder()
    {
        using var model = EmbeddingModel.Load(ModelPath, new() { Backend = "MLX", Threads = 4 });
        int[] seed = model.Tokenize("title: add.py | text: def add(a, b): return a + b");
        int[][] inputs = new[] { 31, 511, 513, 1025 }.Select(length =>
        {
            var row = new int[length]; row[0] = seed[0]; row[^1] = seed[^1];
            for (int i = 1; i < length - 1; ++i) row[i] = seed[1 + (i - 1) % (seed.Length - 2)];
            return row;
        }).ToArray();
        var batch = await model.EmbedTokensAsync(inputs);
        var reversed = await model.EmbedTokensAsync(inputs.Reverse().ToArray());
        Assert.Equal(inputs.Sum(row => row.Length), batch.PromptTokens);
        for (int i = 0; i < inputs.Length; ++i)
        {
            float[] single = (await model.EmbedTokensAsync([inputs[i]])).Embeddings[0];
            AssertNormalized(batch.Embeddings[i]);
            Assert.True(Cosine(single, batch.Embeddings[i]) > .99999, $"Packed MLX length {inputs[i].Length} differs from independent sequence inference.");
            Assert.True(Cosine(batch.Embeddings[i], reversed.Embeddings[inputs.Length - i - 1]) > .99999);
        }
    }

    [CudaFact(ModelVariable)]
    public async Task DirectCudaBatchesPreserveOrderDuplicatesAnd768DimensionalProjection()
        => await VerifyBatch("CUDA");

    [CudaFact(ModelVariable, GgmlBackend = BackendType.GgmlCuda)]
    public async Task NativeCudaBatchesPreserveOrderDuplicatesAnd768DimensionalProjection()
        => await VerifyBatch("GGML_CUDA");

    [ModelFact(ModelVariable, GgmlBackend = BackendType.GgmlCpu)]
    public async Task NativeCpuBatchesPreserveOrderDuplicatesAnd768DimensionalProjection()
        => await VerifyBatch("GGML_CPU");

    [ModelFact(ModelVariable, GgmlBackend = BackendType.GgmlMetal)]
    public async Task NativeMetalBatchesPreserveOrderDuplicatesAnd768DimensionalProjection()
        => await VerifyBatch("GGML_METAL");

    [ModelFact(ModelVariable, GgmlBackend = BackendType.GgmlCpu)]
    public async Task NativeCpuTokenLimitsPreserveEosAndValidationLeavesTheModelUsable()
    {
        using var model = EmbeddingModel.Load(ModelPath, new() { Backend = "GGML_CPU", Threads = 4, MaxTokens = 16 });
        Assert.Equal(768, model.Dimensions);
        Assert.Equal(16, model.MaxTokens);
        string longText = "title: none | text: " + string.Concat(Enumerable.Repeat("word ", 100));
        Assert.Throws<ArgumentException>(() => model.Tokenize(longText));
        int[] truncated = model.Tokenize(longText, truncate: true);
        using var oracle = ReadOracle();
        Assert.Equal(16, truncated.Length);
        Assert.Equal(oracle.RootElement.GetProperty("bos_token_id").GetInt32(), truncated[0]);
        Assert.Equal(oracle.RootElement.GetProperty("eos_token_id").GetInt32(), truncated[^1]);
        await Assert.ThrowsAsync<ArgumentException>(() => model.EmbedTokensAsync([Array.Empty<int>()]));
        await Assert.ThrowsAsync<ArgumentException>(() => model.EmbedTokensAsync([new[] { model.VocabularySize }]));
        await Assert.ThrowsAsync<ArgumentException>(() => model.EmbedTokensAsync([Enumerable.Repeat(truncated[1], 17).ToArray()]));
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => model.EmbedTokensAsync([truncated], cancelled.Token));
        var result = await model.EmbedTokensAsync([truncated]);
        Assert.Equal(16, result.PromptTokens);
        AssertNormalized(result.Embeddings[0]);
    }

    [ModelFact(ModelVariable, GgmlBackend = BackendType.GgmlCpu)]
    public async Task NativeCpuSymmetricSlidingWindowBoundariesKeepSequencesIndependent()
        => await VerifySlidingWindow("GGML_CPU");

    [ModelFact(ModelVariable, GgmlBackend = BackendType.GgmlMetal)]
    public async Task NativeMetalSymmetricSlidingWindowBoundariesKeepSequencesIndependent()
        => await VerifySlidingWindow("GGML_METAL");

    [ModelFact(ModelVariable, GgmlBackend = BackendType.GgmlCpu)]
    public async Task KnowledgeDefaultPromptsMatchDirectInferenceAndResolvedPrefixesDefineIdentity()
    {
        using var model = EmbeddingModel.Load(ModelPath, new() { Backend = "GGML_CPU", Threads = 4 });
        using var provider = new GgufKnowledgeEmbeddingProvider(ModelPath, "GGML_CPU");
        string[] texts = ["The northern lights are caused by charged particles from the sun.", "你好，检索文档。"];
        foreach (bool isQuery in new[] { false, true })
        {
            string prefix = isQuery ? "task: search result | query: " : "title: none | text: ";
            var expected = await model.EmbedTokensAsync(texts.Select(text => model.Tokenize(prefix + text)).ToArray());
            float[][] actual = await provider.EmbedAsync(texts, isQuery);
            for (int index = 0; index < texts.Length; ++index)
                Assert.True(Cosine(expected.Embeddings[index], actual[index]) > .99999);
        }
        using var explicitDefaults = new GgufKnowledgeEmbeddingProvider(ModelPath, "GGML_CPU",
            "task: search result | query: ", "title: none | text: ");
        Assert.Equal(provider.ModelId, explicitDefaults.ModelId);
        using var unprompted = new GgufKnowledgeEmbeddingProvider(ModelPath, "GGML_CPU", "", "");
        Assert.NotEqual(provider.ModelId, unprompted.ModelId);
    }

    private async Task VerifyBatch(string backend)
    {
        using var model = EmbeddingModel.Load(ModelPath, new() { Backend = backend, Threads = 4 });
        Assert.Equal("gemma-embedding2", model.Architecture);
        Assert.Equal(768, model.Dimensions);
        Assert.Equal(8192, model.MaxTokens);
        Assert.Equal(backend == "CPU", model.IsManaged);
        Assert.Equal(backend, model.Backend);
        string[] texts = ["task: code retrieval | query: add two numbers", "title: add.py | text: def add(a, b): return a + b",
            "title: none | text: 你好，检索文档。", "task: code retrieval | query: add two numbers"];
        int[][] inputs = texts.Select(text => model.Tokenize(text)).ToArray();
        var result = await model.EmbedTokensAsync(inputs);
        var reversed = await model.EmbedTokensAsync(inputs.Reverse().ToArray());
        Assert.Equal(inputs.Sum(row => row.Length), result.PromptTokens);
        Assert.Equal(texts.Length, result.Embeddings.Length);
        for (int index = 0; index < inputs.Length; ++index)
        {
            float[] single = (await model.EmbedTokensAsync([inputs[index]])).Embeddings[0];
            float[] raw = model.GetTokenEmbeddings(inputs[index]);
            Assert.Equal(model.HiddenSize * inputs[index].Length, raw.Length);
            Assert.All(raw, value => Assert.True(float.IsFinite(value)));
            float[] fromRaw = (await model.EmbedInputsAsync(raw, inputs[index].Length)).Embeddings[0];
            Assert.True(Cosine(single, fromRaw) > .99999, "Raw embedding inputs must preserve scaled token embedding semantics.");
            AssertNormalized(result.Embeddings[index]);
            double singleCosine = Cosine(result.Embeddings[index], single);
            output.WriteLine($"{backend} input {index}, batch/single cosine {singleCosine:R}");
            Assert.True(singleCosine > .9999);
            Assert.True(Cosine(result.Embeddings[index], reversed.Embeddings[inputs.Length - index - 1]) > .9999);
        }
        Assert.True(Cosine(result.Embeddings[0], result.Embeddings[3]) > .99999);
        await VerifyThirtyTwoInputBatches(model, backend);
    }

    private async Task VerifyThirtyTwoInputBatches(EmbeddingModel model, string backend)
    {
        int[] seed = model.Tokenize("title: add.py | text: def add(a, b): return a + b");
        int[] Row(int length, int variant)
        {
            var row = new int[length]; row[0] = seed[0]; row[^1] = seed[^1];
            for (int position = 1; position < length - 1; ++position)
                row[position] = seed[1 + (position - 1 + variant) % (seed.Length - 2)];
            return row;
        }
        double gate = backend is "MLX" or "CUDA" ? .99999 : .9999;
        foreach (bool mixedLengths in new[] { false, true })
        {
            int[] lengths = mixedLengths ? [4, 9, 16, 23] : [23, 23, 23, 23];
            int[][] unique = Enumerable.Range(0, 4).Select(index => Row(lengths[index], index)).ToArray();
            int[][] inputs = Enumerable.Range(0, 32).Select(index => unique[index % 4]).ToArray();
            var singles = new float[4][];
            for (int index = 0; index < unique.Length; ++index)
                singles[index] = (await model.EmbedTokensAsync([unique[index]])).Embeddings[0];
            var batch = await model.EmbedTokensAsync(inputs);
            var reversed = await model.EmbedTokensAsync(inputs.Reverse().ToArray());
            Assert.Equal(inputs.Sum(row => row.Length), batch.PromptTokens);
            Assert.Equal(32, batch.Embeddings.Length);
            for (int index = 0; index < inputs.Length; ++index)
            {
                AssertNormalized(batch.Embeddings[index]);
                double similarity = Cosine(batch.Embeddings[index], singles[index % 4]);
                Assert.True(similarity > gate, $"{backend} batch32 mixed={mixedLengths} row={index}: cosine {similarity:R}");
                Assert.True(Cosine(batch.Embeddings[index], reversed.Embeddings[31 - index]) > gate);
            }
        }
    }

    private async Task VerifySlidingWindow(string backend)
    {
        using var model = EmbeddingModel.Load(ModelPath, new() { Backend = backend, Threads = 4 });
        int[] seed = model.Tokenize("title: add.py | text: def add(a, b): return a + b");
        int[][] inputs = new[] { 31, 511, 512, 513, 1023, 1024, 1025 }.Select(length =>
        {
            var row = new int[length];
            row[0] = seed[0]; row[^1] = seed[^1];
            for (int index = 1; index < length - 1; ++index) row[index] = seed[1 + (index - 1) % (seed.Length - 2)];
            return row;
        }).ToArray();
        var mixed = await model.EmbedTokensAsync(inputs);
        Assert.Equal(inputs.Sum(row => row.Length), mixed.PromptTokens);
        for (int index = 0; index < inputs.Length; ++index)
        {
            float[] single = (await model.EmbedTokensAsync([inputs[index]])).Embeddings[0];
            AssertNormalized(single);
            AssertNormalized(mixed.Embeddings[index]);
            double similarity = Cosine(single, mixed.Embeddings[index]);
            output.WriteLine($"{backend} length {inputs[index].Length}, batch/single cosine {similarity:R}");
            Assert.True(similarity > .9999, $"Length {inputs[index].Length}: cosine {similarity:R}");
        }
    }

    private static string ModelPath => Environment.GetEnvironmentVariable(ModelVariable)!;
    private static JsonDocument ReadOracle()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory != null; directory = directory.Parent)
            if (File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
                return JsonDocument.Parse(File.ReadAllText(Path.Combine(directory.FullName, "InferenceWeb.Tests", "Fixtures",
                    "EmbeddingTokenizer", "embeddinggemma2-tokenization.json")));
        throw new DirectoryNotFoundException("TensorSharp source tree is required for the EmbeddingGemma 2 tokenizer oracle.");
    }

    private static double Cosine(float[] left, float[] right)
        => left.Zip(right, (a, b) => (double)a * b).Sum()
            / Math.Sqrt(left.Sum(value => (double)value * value) * right.Sum(value => (double)value * value));
    private static void AssertNormalized(float[] vector)
    {
        Assert.Equal(768, vector.Length);
        Assert.All(vector, value => Assert.True(float.IsFinite(value)));
        Assert.InRange(vector.Sum(value => (double)value * value), .99999, 1.00001);
    }
}
