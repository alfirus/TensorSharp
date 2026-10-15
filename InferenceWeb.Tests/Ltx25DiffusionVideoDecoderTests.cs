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

public sealed class Ltx25DiffusionVideoDecoderTests
{
    private static Stream Resource(string name) => typeof(Ltx25DiffusionVideoDecoderTests).Assembly.GetManifestResourceStream(
        "InferenceWeb.Tests.Fixtures.Ltx25." + name)!;
    private static IAllocator Allocator() => Environment.GetEnvironmentVariable("TS_TEST_LTX25_VAE_BACKEND")?.ToLowerInvariant() switch
    {
        "mlx" => new MlxAllocator(),
        null or "" or "cpu" => new CpuAllocator(BlasEnum.DotNet),
        var name => throw new ArgumentException("Unknown diffusion VAE test backend " + name),
    };

    private sealed class Fixture
    {
        public required JsonElement Root { get; init; }
        public required byte[] Bin { get; init; }
        public (float[] Values, int[] Shape) Array(string name)
        {
            var entry = Root.GetProperty("arrays").GetProperty(name);
            int[] shape = entry.GetProperty("shape").EnumerateArray().Select(value => value.GetInt32()).ToArray();
            var values = new float[shape.Aggregate(1, (a, b) => a * b)];
            Buffer.BlockCopy(Bin, entry.GetProperty("offset").GetInt32(), values, 0, values.Length * 4);
            return (values, shape);
        }
    }

    private static Fixture Load()
    {
        using var json = JsonDocument.Parse(Resource("diffusion-vae.json"));
        using var bin = new MemoryStream(); Resource("diffusion-vae.bin").CopyTo(bin);
        var fixture = new Fixture { Root = json.RootElement.Clone(), Bin = bin.ToArray() };
        Assert.Equal(fixture.Root.GetProperty("bin_sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(fixture.Bin)));
        return fixture;
    }

    private static void Close(float[] expected, float[] actual, string label, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = 0, reference = 0, worst = 0;
        for (int i = 0; i < actual.Length; ++i)
        {
            Assert.True(float.IsFinite(actual[i]), $"{label}[{i}] is not finite");
            double difference = actual[i] - expected[i];
            error += difference * difference; reference += (double)expected[i] * expected[i];
            worst = Math.Max(worst, Math.Abs(difference));
        }
        double relative = Math.Sqrt(error / Math.Max(reference, 1e-30));
        Assert.True(relative <= tolerance, $"{label}: relative L2 {relative:E3} (max abs {worst:E3}) exceeds {tolerance:E1}");
    }

    [Fact]
    public void DecodeMatchesUnmodifiedOriginalStageByStage()
    {
        // generate-ltx25-diffusion-vae-oracle.py: tiny NADiffusionDecoder through the original
        // configurator, edge-padded 2x1x2 latent, saved pixel noise, eager NATTEN-semantics attention.
        var fixture = Load();
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", fixture.Root.GetProperty("source_revision").GetString());
        using var checkpoint = new Checkpoint();
        Assert.Equal(fixture.Root.GetProperty("checkpoint_sha256").GetString(), checkpoint.Hash);
        var (latent, latentShape) = fixture.Array("latent");
        var (noise, noiseShape) = fixture.Array("noise");
        IAllocator allocator = Allocator();
        try
        {
            using var decoder = new Ltx25DiffusionVideoDecoder(checkpoint.Path, allocator);
            Assert.Equal(8, decoder.LatentChannels);
            Assert.Equal(noiseShape, decoder.NoiseShape(latentShape[1], latentShape[2], latentShape[3]));
            var traces = new Dictionary<string, (float[] Values, int[] Shape)>();
            float[] preserved = (float[])latent.Clone();
            float[] actual = decoder.DecodeRaw(latent, latentShape[1], latentShape[2], latentShape[3], noise,
                trace: (name, values, shape) => traces[name] = (values, shape));
            Assert.Equal(preserved, latent);
            foreach (string stage in new[] { "feat_stage4_input", "context" })
            {
                var expected = fixture.Array(stage);
                Assert.Equal(expected.Shape, traces[stage].Shape);
                Close(expected.Values, traces[stage].Values, stage, 2e-5f);
            }
            var output = fixture.Array("output");
            Assert.Equal(3 * 9 * 32 * 64, actual.Length);
            Close(output.Values, actual, "output", 2e-5f);
        }
        finally { (allocator as IDisposable)?.Dispose(); }
    }

    [Fact]
    public void SeededDecodeIsDeterministicFrameMajorRgb()
    {
        var fixture = Load();
        using var checkpoint = new Checkpoint();
        var (latent, shape) = fixture.Array("latent");
        IAllocator allocator = Allocator();
        try
        {
            using var decoder = new Ltx25DiffusionVideoDecoder(checkpoint.Path, allocator);
            float[] first = decoder.Decode(latent, shape[1], shape[2], shape[3], seed: 7);
            float[] again = decoder.Decode(latent, shape[1], shape[2], shape[3], seed: 7);
            float[] other = decoder.Decode(latent, shape[1], shape[2], shape[3], seed: 8);
            Assert.Equal(9 * 32 * 64 * 3, first.Length);
            Assert.Equal(first, again);
            Assert.NotEqual(first, other);
            Assert.All(first, value => Assert.InRange(value, 0f, 1f));
        }
        finally { (allocator as IDisposable)?.Dispose(); }
    }

    [Theory]
    [InlineData("model_output_type", "\"v\"")]
    [InlineData("decoder.default_num_inference_steps", "2")]
    [InlineData("decoder.patch_size", "2")]
    [InlineData("decoder.stage5_channels", "32")]
    public void UnsupportedDecoderVariantsAreRefused(string key, string value)
    {
        using var checkpoint = new Checkpoint(vae =>
        {
            var target = key.StartsWith("decoder.") ? vae["decoder"]!.AsObject() : vae;
            target[key.Split('.')[^1]] = JsonNode.Parse(value);
        });
        Assert.ThrowsAny<Exception>(() => new Ltx25DiffusionVideoDecoder(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet)));
    }

    [Fact]
    public void ConvolutionalCheckpointsAreNotDiffusionVaes()
    {
        using var diffusion = new Checkpoint();
        using (var file = new TensorSharp.Runtime.SafetensorsFile(diffusion.Path))
            Assert.True(Ltx25DiffusionVaeConfig.IsDiffusionVae(file.Metadata));
        using var conv = new Checkpoint(vae => vae["_class_name"] = "CausalVideoAutoencoder");
        using (var file = new TensorSharp.Runtime.SafetensorsFile(conv.Path))
            Assert.False(Ltx25DiffusionVaeConfig.IsDiffusionVae(file.Metadata));
        Assert.ThrowsAny<NotSupportedException>(() => new Ltx25DiffusionVideoDecoder(conv.Path, new CpuAllocator(BlasEnum.DotNet)));
    }

    private sealed class Checkpoint : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ltx25-diffusion-vae-" + Guid.NewGuid().ToString("N") + ".safetensors");
        public string Hash { get; }
        public Checkpoint(Action<JsonObject>? changeVae = null)
        {
            using var source = Resource("diffusion-vae-tiny.safetensors");
            using var memory = new MemoryStream(); source.CopyTo(memory);
            byte[] bytes = memory.ToArray();
            if (changeVae == null) File.WriteAllBytes(Path, bytes);
            else
            {
                int headerSize = checked((int)BitConverter.ToUInt64(bytes, 0));
                var header = JsonNode.Parse(Encoding.UTF8.GetString(bytes, 8, headerSize))!.AsObject();
                var metadata = header["__metadata__"]!.AsObject();
                var config = JsonNode.Parse(metadata["config"]!.GetValue<string>())!.AsObject();
                changeVae(config["vae"]!.AsObject());
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
