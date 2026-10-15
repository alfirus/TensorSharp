using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

public sealed class Ltx25LatentUpsamplerTests
{
    private static Stream Resource(string name) => typeof(Ltx25LatentUpsamplerTests).Assembly.GetManifestResourceStream(
        "InferenceWeb.Tests.Fixtures.Ltx25." + name)!;
    private static float[] Values(JsonElement node) => node.EnumerateArray().Select(value => value.GetSingle()).ToArray();
    private static IAllocator Allocator() => Environment.GetEnvironmentVariable("TS_TEST_LTX25_VAE_BACKEND")?.ToLowerInvariant() switch
    {
        "cuda" => new CudaAllocator(), "mlx" => new MlxAllocator(),
        null or "" or "cpu" => new CpuAllocator(BlasEnum.DotNet),
        var name => throw new ArgumentException("Unknown video VAE test backend " + name),
    };

    [Fact]
    public void UpsamplingMatchesUnmodifiedOriginalWithBf16StoredWeights()
    {
        // generate-ltx25-latent-upsampler-oracle.py: 16 latent / 64 mid channels, one block per
        // stage (two channels per GroupNorm group) and three frames for both temporal edges.
        using var document = JsonDocument.Parse(Resource("latent-upsampler.json"));
        var fixture = document.RootElement;
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", fixture.GetProperty("source_revision").GetString());
        using var checkpoint = new Checkpoint();
        Assert.Equal(fixture.GetProperty("checkpoint_sha256").GetString(), checkpoint.Hash);
        int[] shape = fixture.GetProperty("input_shape").EnumerateArray().Select(value => value.GetInt32()).ToArray();
        float[] input = Values(fixture.GetProperty("input")), preserved = (float[])input.Clone();
        float[] expected = Values(fixture.GetProperty("output"));
        IAllocator allocator = Allocator();
        try
        {
            using var upsampler = new Ltx25LatentUpsampler(checkpoint.Path, allocator);
            Assert.Equal((16, 64, 1), (upsampler.LatentChannels, upsampler.MidChannels, upsampler.BlocksPerStage));
            float[] actual = upsampler.Upsample(input, shape[1], shape[2], shape[3],
                Values(fixture.GetProperty("mean")), Values(fixture.GetProperty("std")));
            Assert.Equal(preserved, input);
            Assert.Equal(expected.Length, actual.Length);
            Assert.Equal(16 * 3 * 6 * 8, actual.Length);
            for (int i = 0; i < actual.Length; ++i)
                Assert.True(float.IsFinite(actual[i]) && Math.Abs(actual[i] - expected[i]) <= 1e-4f + 1e-4f * Math.Abs(expected[i]),
                    $"output[{i}] expected {expected[i]:R}, actual {actual[i]:R}");
        }
        finally { (allocator as IDisposable)?.Dispose(); }
    }

    [Theory]
    [InlineData("temporal_upsample", "true")]
    [InlineData("rational_resampler", "true")]
    [InlineData("spatial_scale", "1.5")]
    [InlineData("dims", "2")]
    public void OtherUpsamplerVariantsAreRefused(string key, string value)
    {
        using var checkpoint = new Checkpoint(config => config[key] = JsonNode.Parse(value));
        Assert.Throws<NotSupportedException>(() => new Ltx25LatentUpsampler(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet)));
    }

    [Fact]
    public void LatentsMustMatchTheConfiguredChannelsAndStatistics()
    {
        using var checkpoint = new Checkpoint();
        using var upsampler = new Ltx25LatentUpsampler(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet));
        float[] statistics = new float[16];
        Assert.Throws<ArgumentException>(() => upsampler.Upsample(new float[17 * 4], 1, 2, 2, statistics, statistics));
        Assert.Throws<ArgumentException>(() => upsampler.Upsample(new float[16 * 4], 1, 2, 2, new float[15], statistics));
        Assert.Throws<ArgumentException>(() => upsampler.Upsample([float.NaN, .. new float[16 * 4 - 1]], 1, 2, 2, statistics, statistics));
    }

    private sealed class Checkpoint : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ltx25-upsampler-" + Guid.NewGuid().ToString("N") + ".safetensors");
        public string Hash { get; }
        public Checkpoint(Action<JsonObject>? changeConfig = null)
        {
            using var source = Resource("latent-upsampler-tiny.safetensors");
            using var memory = new MemoryStream(); source.CopyTo(memory);
            byte[] bytes = memory.ToArray();
            if (changeConfig == null) File.WriteAllBytes(Path, bytes);
            else
            {
                int headerSize = checked((int)BitConverter.ToUInt64(bytes, 0));
                var header = JsonNode.Parse(Encoding.UTF8.GetString(bytes, 8, headerSize))!.AsObject();
                var metadata = header["__metadata__"]!.AsObject();
                var config = JsonNode.Parse(metadata["config"]!.GetValue<string>())!.AsObject();
                changeConfig(config);
                metadata["config"] = config.ToJsonString();
                byte[] json = Encoding.UTF8.GetBytes(header.ToJsonString());
                using var target = File.Create(Path); target.Write(BitConverter.GetBytes((ulong)json.Length));
                target.Write(json); target.Write(bytes, 8 + headerSize, bytes.Length - 8 - headerSize);
            }
            using var file = File.OpenRead(Path); Hash = Convert.ToHexStringLower(SHA256.HashData(file));
        }
        public void Dispose() => File.Delete(Path);
    }
}
