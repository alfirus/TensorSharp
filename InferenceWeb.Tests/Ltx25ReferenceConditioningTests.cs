using System.Text.Json;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;

namespace InferenceWeb.Tests;

public sealed class Ltx25ReferenceConditioningTests
{
    private static float[] Values(JsonElement node) => node.EnumerateArray().Select(value => value.GetSingle()).ToArray();
    private static void Close(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i) Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, 4e-7f);
    }
    private static RgbImage Solid(int width, int height, float r, float g, float b)
    {
        var pixels = new float[width * height * 3];
        for (int i = 0; i < pixels.Length; i += 3) { pixels[i] = r; pixels[i + 1] = g; pixels[i + 2] = b; }
        return new(width, height, pixels);
    }

    [Fact]
    public void ReferenceVideoMasksPositionsNoiseAndEulerMatchUnchangedOfficialImplementation()
    {
        using var stream = typeof(Ltx25ReferenceConditioningTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Ltx25.reference-conditioning.json")!;
        using var document = JsonDocument.Parse(stream);
        var fixture = document.RootElement;
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", fixture.GetProperty("source_revision").GetString());
        var settings = Ltx25GenerationSettings.Resolve(new() { Mode = "ref", ReferenceImagePaths = ["identity.png"], Width = 64, Height = 32, Frames = 121, Seed = 42 }, true);
        float[] noise = Values(fixture.GetProperty("noise"));
        var input = new Ltx25DenoiserInput
        {
            Video = noise, VideoTokens = settings.VideoTokens * 2, Audio = [0], AudioTokens = 1,
            VideoContext = [0], VideoContextTokens = 1, AudioContext = [0], AudioContextTokens = 1,
            VideoPositions = Ltx25Geometry.VideoPositions(settings.LatentFrames, 1, 2, 24),
            AudioPositions = Ltx25Geometry.AudioPositions(1), VideoSigma = 1, AudioSigma = 1,
            FrequencyPrecision = Ltx25FrequencyPrecision.Float64,
        };
        float[] reference = Values(fixture.GetProperty("reference_latent")), preserved = (float[])reference.Clone();
        var initial = Ltx25ReferenceConditioning.Apply(input, settings, reference, 16, 1, 2);
        Close(Values(fixture.GetProperty("initial")), initial.Video);
        Close(Values(fixture.GetProperty("clean")), initial.CleanVideo);
        Close(Values(fixture.GetProperty("mask")), initial.VideoDenoiseMask);
        Close(Values(fixture.GetProperty("keyframe_mask")), initial.KeyframeMask);
        Close(Values(fixture.GetProperty("positions")), initial.VideoPositions.Boundaries);
        Assert.True(initial.UseReferenceLora); Assert.Same(input.Audio, initial.Audio);
        Assert.Equal(preserved, reference); Close(Values(fixture.GetProperty("noise")), input.Video);
        var steps = fixture.GetProperty("steps").EnumerateArray().ToArray();
        int step = 0;
        var result = Ltx25Sampling.RunEuler(current =>
        {
            Assert.True(current.UseReferenceLora);
            Close(step == 0 ? Values(fixture.GetProperty("initial")) : Values(steps[step - 1].GetProperty("output")), current.Video);
            Close(Values(steps[step].GetProperty("timesteps")), current.VideoTimesteps);
            return new(Values(steps[step++].GetProperty("velocity")), [0], 48);
        }, initial, Values(fixture.GetProperty("sigmas")));
        Close(Values(steps[^1].GetProperty("output")), result.Video);
        Assert.Equal(new[] { 1f, 1f }, initial.KeyframeMask.Take(2));
        Assert.All(initial.KeyframeMask.Skip(2), value => Assert.Equal(0, value));
        Close(initial.CleanVideo[(settings.VideoTokens * 128)..], result.Video[(settings.VideoTokens * 128)..]);
    }

    [Fact]
    public void IngredientsDefaultsRequireAnAvailableAdapterAndSources()
    {
        var request = new VideoGenerationParams { Mode = "ref", ReferenceImagePaths = ["identity.png"] };
        Assert.Throws<NotSupportedException>(() => Ltx25GenerationSettings.Resolve(request));
        var settings = Ltx25GenerationSettings.Resolve(request, true);
        Assert.Equal((768, 448, 121, 24), (settings.Width, settings.Height, settings.Frames, settings.Fps));
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new() { Mode = "ref" }, true));
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new() { Mode = "ref", ReferenceImagePaths = ["identity.png"], Frames = 113 }, true));
        Assert.Equal(129, Ltx25GenerationSettings.Resolve(new() { Mode = "ref", ReferenceImagePaths = ["identity.png"], Frames = 129 }, true).Frames);
    }

    [Fact]
    public void IdentitySourcesCannotBeSilentlyInterpretedAsTimedGuides()
    {
        Assert.Throws<ArgumentException>(() => Ltx25ReferenceConditioning.ResolveSources(new() { Mode = "ref", ImagePath = "first", ReferenceImagePaths = ["other"] }));
        Assert.Throws<ArgumentException>(() => Ltx25ReferenceConditioning.ResolveSources(new() { Mode = "ref", ImagePath = "first", EndImagePath = "last" }));
        Assert.Throws<ArgumentException>(() => Ltx25ReferenceConditioning.ResolveSources(new() { Mode = "ref", ImageConditions = [new() { ImagePath = "timed", FrameIndex = 8 }] }));
        Assert.Throws<ArgumentException>(() => Ltx25ReferenceConditioning.ResolveSources(new() { Mode = "ref", ReferenceImagePaths = [""] }));
        Assert.Throws<ArgumentException>(() => Ltx25ReferenceConditioning.ResolveSources(new() { Mode = "ref", ReferenceImagePaths = Enumerable.Repeat("identity", 17).ToArray() }));
        Assert.Equal(new[] { "one", "two" }, Ltx25ReferenceConditioning.ResolveSources(new() { Mode = "ref", ReferenceImagePaths = ["one", "two"] }).Select(source => source.ImagePath));
        Assert.Equal("single", Assert.Single(Ltx25ReferenceConditioning.ResolveSources(new() { Mode = "ref", ImagePath = "single" })).ImagePath);
    }

    [Fact]
    public void ReferenceSheetPreservesWholeSourcesAndLeavesUnusedGridCellsBlack()
    {
        var red = Solid(32, 32, 1, 0, 0); var green = Solid(32, 32, 0, 1, 0); var blue = Solid(32, 32, 0, 0, 1);
        var sheet = Ltx25ReferenceConditioning.BuildSheet([red, green, blue], 64, 64);
        Assert.Equal(new[] { 1f, 0, 0 }, sheet.Pixels[0..3]);
        Assert.Equal(new[] { 0f, 1, 0 }, sheet.Pixels[(32 * 3)..(33 * 3)]);
        Assert.Equal(new[] { 0f, 0, 1 }, sheet.Pixels[(32 * 64 * 3)..(32 * 64 * 3 + 3)]);
        Assert.Equal(new float[3], sheet.Pixels[^3..]);
        var landscape = Solid(32, 16, 1, .25f, 0);
        var letterboxed = Ltx25ReferenceConditioning.BuildSheet([landscape], 32, 32);
        Assert.All(letterboxed.Pixels[..(8 * 32 * 3)], value => Assert.Equal(0, value));
        Assert.Equal(landscape.Pixels, letterboxed.Pixels[(8 * 32 * 3)..(24 * 32 * 3)]);
        Assert.All(letterboxed.Pixels[(24 * 32 * 3)..], value => Assert.Equal(0, value));
        var translucent = new RgbImage(32, 32, red.Pixels, Enumerable.Repeat(.25f, 32 * 32).ToArray());
        var composited = Ltx25ReferenceConditioning.BuildSheet([translucent], 32, 32);
        Assert.Equal(new[] { .25f, 0f, 0f }, composited.Pixels[..3]);
        Assert.Equal(1, translucent.Pixels[0]);
    }

    [Fact]
    public void StaticReferenceIsRepeatedInRgbBeforeEncodingAndHonorsCancellation()
    {
        var sheet = Solid(32, 32, .25f, .5f, .75f);
        float[] video = Ltx25ReferenceConditioning.RepeatSheet(sheet, 121);
        Assert.Equal(sheet.Pixels.Length * 121, video.Length);
        for (int frame = 0; frame < 121; ++frame) Assert.Equal(sheet.Pixels, video[(frame * sheet.Pixels.Length)..((frame + 1) * sheet.Pixels.Length)]);
        Assert.Throws<ArgumentException>(() => Ltx25ReferenceConditioning.RepeatSheet(sheet, 17));
        using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() => Ltx25ReferenceConditioning.RepeatSheet(sheet, 121, cancellation.Token));
    }
}
