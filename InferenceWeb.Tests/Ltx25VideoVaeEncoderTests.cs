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

public sealed class Ltx25VideoVaeEncoderTests
{
    private static Stream Resource(string name) => typeof(Ltx25VideoVaeEncoderTests).Assembly.GetManifestResourceStream(
        "InferenceWeb.Tests.Fixtures.Ltx25." + name)!;
    private static float[] Values(JsonElement node) => node.EnumerateArray().Select(value => value.GetSingle()).ToArray();
    private static int[] Shape(JsonElement node) => node.EnumerateArray().Select(value => value.GetInt32()).ToArray();
    private static IAllocator Allocator() => Environment.GetEnvironmentVariable("TS_TEST_LTX25_VAE_BACKEND")?.ToLowerInvariant() switch
    {
        "cuda" => new CudaAllocator(), "mlx" => new MlxAllocator(),
        null or "" or "cpu" => new CpuAllocator(BlasEnum.DotNet),
        var name => throw new ArgumentException("Unknown video VAE test backend " + name),
    };
    private static void Close(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < actual.Length; ++i)
            Assert.True(float.IsFinite(actual[i]) && Math.Abs(actual[i] - expected[i]) <= 2e-5f + 2e-5f * Math.Abs(expected[i]),
                $"{label}[{i}] expected {expected[i]:R}, actual {actual[i]:R}");
    }

    [Theory]
    [InlineData("video-vae-encoder-single-frame.json")]
    [InlineData("video-vae-encoder-nine-frame.json")]
    public void EveryEncoderStageMatchesUnmodifiedOfficialWithBf16Weights(string fixture)
    {
        using var reference = JsonDocument.Parse(Resource(fixture));
        var expected = reference.RootElement;
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", expected.GetProperty("source_revision").GetString());
        using var checkpoint = new Checkpoint();
        Assert.Equal(expected.GetProperty("checkpoint_sha256").GetString(), checkpoint.Hash);
        IAllocator allocator = Allocator();
        try
        {
            using var encoder = new Ltx25VideoVaeEncoder(checkpoint.Path, allocator);
            Assert.Equal(2, encoder.LatentChannels); Assert.Equal(8, encoder.TemporalScale); Assert.Equal(16, encoder.SpatialScale);
            var stages = new Dictionary<string, (float[] Values, int[] Shape)>();
            int[] shape = Shape(expected.GetProperty("input_shape"));
            float[] input = Values(expected.GetProperty("input")), preserved = (float[])input.Clone();
            float[] actual = encoder.Encode(input, shape[0], shape[1], shape[2], trace: (name, values, dimensions) => stages.Add(name, (values, dimensions)));
            Assert.Equal(preserved, input);
            var oracleStages = expected.GetProperty("stages");
            Assert.Equal(oracleStages.EnumerateObject().Count(), stages.Count);
            foreach (var stage in oracleStages.EnumerateObject())
            {
                Assert.True(stages.TryGetValue(stage.Name, out var observed), stage.Name);
                Assert.Equal(Shape(stage.Value.GetProperty("shape")), observed.Shape);
                Close(Values(stage.Value.GetProperty("values")), observed.Values, stage.Name);
            }
            Close(Values(oracleStages.GetProperty("encoded").GetProperty("values")), actual, "output");
        }
        finally { (allocator as IDisposable)?.Dispose(); }
    }

    [Fact]
    public void Genuine121FrameStaticVideoEncodingMatchesOfficialTemporalEncoder()
    {
        using var reference = JsonDocument.Parse(Resource("video-vae-encoder-static121.json"));
        using var checkpoint = new Checkpoint();
        var root = reference.RootElement;
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", root.GetProperty("source_revision").GetString());
        Assert.True(root.GetProperty("static_rgb_video").GetBoolean());
        Assert.Equal(checkpoint.Hash, root.GetProperty("checkpoint_sha256").GetString());
        float[] frame = Values(root.GetProperty("static_frame_rgb"));
        float[] rgbVideo = new float[frame.Length * 121];
        for (int index = 0; index < 121; ++index) frame.CopyTo(rgbVideo, index * frame.Length);
        IAllocator allocator = Allocator();
        try
        {
            using var encoder = new Ltx25VideoVaeEncoder(checkpoint.Path, allocator);
            float[] actual = encoder.Encode(rgbVideo, 121, 16, 32);
            float[] oracle = Values(root.GetProperty("stages").GetProperty("encoded").GetProperty("values"));
            Close(oracle, actual, "static121-video");
            Assert.Equal(2 * 16 * 1 * 2, actual.Length);
            float[] preserved = (float[])frame.Clone();
            float[] optimized = encoder.EncodeStaticVideo(frame, 121, 16, 32);
            Close(oracle, optimized, "static121-optimized-vs-official");
            Close(actual, optimized, "static121-optimized-vs-full");
            Assert.Equal(preserved, frame);
            // Layout is channel-major; every temporal slice of each channel is
            // the same, including after a prior arbitrary-video invocation.
            for (int channel = 0; channel < 2; ++channel)
                for (int time = 1; time < 16; ++time)
                    Assert.Equal(optimized.AsSpan(channel * 32, 2).ToArray(), optimized.AsSpan(channel * 32 + time * 2, 2).ToArray());
            Assert.Equal(encoder.EncodeImage(frame, 16, 32), encoder.EncodeStaticVideo(frame, 1, 16, 32));
        }
        finally { (allocator as IDisposable)?.Dispose(); }
    }

    [Fact]
    public void ImageEntryPointIsCausalAndDoesNotRetainTemporalState()
    {
        using var reference = JsonDocument.Parse(Resource("video-vae-encoder-nine-frame.json"));
        using var checkpoint = new Checkpoint();
        IAllocator allocator = Allocator();
        try
        {
            using var encoder = new Ltx25VideoVaeEncoder(checkpoint.Path, allocator);
            float[] pixels = Values(reference.RootElement.GetProperty("input"));
            float[] image = pixels[..(16 * 32 * 3)];
            float[] first = encoder.EncodeImage(image, 16, 32);
            float[] video = encoder.Encode(pixels, 9, 16, 32);
            float[] firstVideoFrame = [video[0], video[1], video[4], video[5]];
            Close(first, firstVideoFrame, "causal-first-frame");
            encoder.ReleaseDeviceResidency();
            Assert.Equal(first, encoder.EncodeImage(image, 16, 32, trace: (_, values, _) => Array.Fill(values, 1000f)));
        }
        finally { (allocator as IDisposable)?.Dispose(); }
    }

    [Theory]
    [InlineData(0, 16, 32)]
    [InlineData(1, 0, 32)]
    [InlineData(1, 16, 31)]
    [InlineData(2, 16, 32)]
    public void InvalidGeometryFailsBeforeExecutingEncoder(int frames, int height, int width)
    {
        using var checkpoint = new Checkpoint();
        using var encoder = new Ltx25VideoVaeEncoder(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Encode([], frames, height, width));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.EncodeStaticVideo([], frames, height, width));
    }

    [Fact]
    public void InvalidPixelsAreRejectedAndCancellationAndDisposalPreventStages()
    {
        using var checkpoint = new Checkpoint();
        using var encoder = new Ltx25VideoVaeEncoder(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet));
        Assert.Throws<ArgumentException>(() => encoder.EncodeImage([1], 16, 16));
        Assert.Throws<ArgumentException>(() => encoder.EncodeStaticVideo([1], 121, 16, 16));
        var pixels = new float[16 * 16 * 3];
        foreach (float invalid in new[] { float.NaN, float.PositiveInfinity, -0.1f, 1.1f })
        { pixels[0] = invalid; Assert.Throws<ArgumentException>(() => encoder.EncodeImage(pixels, 16, 16)); }
        pixels[0] = .5f;
        int stages = 0;
        using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() => encoder.EncodeImage(pixels, 16, 16, cancellation.Token, (_, _, _) => stages++));
        Assert.Throws<OperationCanceledException>(() => encoder.EncodeStaticVideo(pixels, 121, 16, 16, cancellation.Token, (_, _, _) => stages++));
        encoder.Dispose();
        Assert.Throws<ObjectDisposedException>(() => encoder.EncodeImage(pixels, 16, 16));
        Assert.Throws<ObjectDisposedException>(() => encoder.EncodeStaticVideo(pixels, 121, 16, 16));
        Assert.Equal(0, stages);
    }

    [Theory]
    [InlineData("norm_layer", "group_norm")]
    [InlineData("spatial_padding_mode", "reflect")]
    [InlineData("latent_log_var", "none")]
    public void UnsupportedEncoderArchitecturesFailClosed(string field, string value)
    {
        using var checkpoint = new Checkpoint(header =>
        {
            var metadata = header["__metadata__"]!.AsObject();
            var config = JsonNode.Parse(metadata["config"]!.GetValue<string>())!;
            config["vae"]![field] = value; metadata["config"] = config.ToJsonString();
        });
        Assert.Throws<NotSupportedException>(() => new Ltx25VideoVaeEncoder(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet)));
    }

    [Fact]
    public void EncoderValidatesConvolutionShapesBeforeInference()
    {
        using var checkpoint = new Checkpoint(header => header["encoder.conv_in.conv.weight"]!["shape"] = new JsonArray(12, 2, 3, 3, 3));
        Assert.Throws<InvalidDataException>(() => new Ltx25VideoVaeEncoder(checkpoint.Path, new CpuAllocator(BlasEnum.DotNet)));
    }

    private sealed class Checkpoint : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ltx25-video-vae-encoder-" + Guid.NewGuid().ToString("N") + ".safetensors");
        public string Hash { get; }
        public Checkpoint(Action<JsonObject>? change = null)
        {
            using var source = Resource("video-vae-encoder-tiny.safetensors");
            using var memory = new MemoryStream(); source.CopyTo(memory);
            byte[] bytes = memory.ToArray();
            if (change == null) File.WriteAllBytes(Path, bytes);
            else
            {
                int headerSize = checked((int)BitConverter.ToUInt64(bytes, 0));
                var header = JsonNode.Parse(Encoding.UTF8.GetString(bytes, 8, headerSize))!.AsObject(); change(header);
                byte[] json = Encoding.UTF8.GetBytes(header.ToJsonString());
                using var target = File.Create(Path); target.Write(BitConverter.GetBytes((ulong)json.Length));
                target.Write(json); target.Write(bytes, 8 + headerSize, bytes.Length - 8 - headerSize);
            }
            using var file = File.OpenRead(Path); Hash = Convert.ToHexStringLower(SHA256.HashData(file));
        }
        public void Dispose() => File.Delete(Path);
    }
}
