// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using TensorSharp.Models.Embeddings;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class EmbeddingGemma2MetadataTests
{
    [Theory]
    [InlineData("attention.value_length")]
    [InlineData("attention.value_length_swa")]
    public void ManagedEncoderRejectsUnequalKeyAndValueWidthsBeforeLoadingWeights(string key)
    {
        using var fixture = new MetadataFixture();
        fixture.File.Metadata["gemma-embedding2." + key] = 16U;
        var error = Assert.Throws<InvalidDataException>(() => new ManagedEmbeddingGemma2Encoder(fixture.File, 1));
        Assert.Contains(key, error.Message);
    }

    [Theory]
    [InlineData("rope.dimension_count")]
    [InlineData("rope.dimension_count_swa")]
    public void ManagedEncoderRejectsPartialRotarySettingsBeforeLoadingWeights(string key)
    {
        using var fixture = new MetadataFixture();
        fixture.File.Metadata["gemma-embedding2." + key] = 16U;
        var error = Assert.Throws<NotSupportedException>(() => new ManagedEmbeddingGemma2Encoder(fixture.File, 1));
        Assert.Contains(key, error.Message);
    }

    [Fact]
    public void ManagedEncoderRejectsRotaryScalingBeforeLoadingWeights()
    {
        using var fixture = new MetadataFixture();
        fixture.File.Metadata["gemma-embedding2.rope.scaling.factor"] = 2f;
        var error = Assert.Throws<NotSupportedException>(() => new ManagedEmbeddingGemma2Encoder(fixture.File, 1));
        Assert.Contains("rotary scaling", error.Message);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void ManagedEncoderRejectsInvalidRotaryScaling(float factor)
    {
        using var fixture = new MetadataFixture();
        fixture.File.Metadata["gemma-embedding2.rope.scaling.factor"] = factor;
        Assert.Throws<InvalidDataException>(() => new ManagedEmbeddingGemma2Encoder(fixture.File, 1));
    }

    [Fact]
    public void TextContractRejectsCausalAttentionAndNonMeanPooling()
    {
        using var fixture = new MetadataFixture();
        fixture.File.Metadata["gemma-embedding2.attention.causal"] = true;
        Assert.Throws<InvalidDataException>(() => EmbeddingGemma2TextMetadata.Validate(fixture.File, 32, 32));
        fixture.File.Metadata["gemma-embedding2.attention.causal"] = false;
        fixture.File.Metadata["gemma-embedding2.pooling_type"] = 2U;
        Assert.Throws<InvalidDataException>(() => EmbeddingGemma2TextMetadata.Validate(fixture.File, 32, 32));
    }

    [Fact]
    public void TextContractAllowsDefaultsAndOddSymmetricWindowWidths()
    {
        using var fixture = new MetadataFixture();
        fixture.File.Metadata["gemma-embedding2.attention.sliding_window"] = 9U;
        fixture.File.Metadata.Remove("gemma-embedding2.pooling_type");
        // llama-hparams.h masks distance > floor(n_swa / 2); an odd GGUF
        // window does not change this contract or require rejecting the model.
        EmbeddingGemma2TextMetadata.Validate(fixture.File, 32, 32);
    }

    private sealed class MetadataFixture : IDisposable
    {
        private readonly string _path = Path.Combine(Path.GetTempPath(), $"gemma2-metadata-{Guid.NewGuid():N}.gguf");
        internal GgufFile File { get; }

        internal MetadataFixture()
        {
            using (var writer = new BinaryWriter(System.IO.File.Create(_path)))
            {
                writer.Write(Encoding.ASCII.GetBytes("GGUF"));
                writer.Write(3U); writer.Write(0UL); writer.Write(0UL);
            }
            File = new GgufFile(_path);
            foreach (string key in new[] { "embedding_length", "embedding_length_out", "embedding_length_per_layer_input",
                "feed_forward_length", "attention.key_length", "attention.key_length_swa" })
                File.Metadata["gemma-embedding2." + key] = 32U;
            foreach (string key in new[] { "block_count", "attention.head_count", "pooling_type" })
                File.Metadata["gemma-embedding2." + key] = 1U;
            File.Metadata["gemma-embedding2.attention.sliding_window"] = 8U;
        }

        public void Dispose() { File.Dispose(); System.IO.File.Delete(_path); }
    }
}
