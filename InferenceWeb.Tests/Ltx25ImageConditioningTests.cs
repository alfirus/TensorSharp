using TensorSharp.Models.Ltx25;
using TensorSharp.Models.Video;
using System.Text.Json;

namespace InferenceWeb.Tests;

public sealed class Ltx25ImageConditioningTests
{
    [Fact]
    public void CombinedGuidesAndMaskedEulerMatchUnchangedOfficialImplementation()
    {
        using Stream stream = typeof(Ltx25ImageConditioningTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Ltx25.image-conditioning.json")!;
        using var document = JsonDocument.Parse(stream);
        var fixture = document.RootElement;
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", fixture.GetProperty("source_revision").GetString());
        static float[] Values(JsonElement value) => value.EnumerateArray().Select(item => item.GetSingle()).ToArray();
        static void Close(float[] expected, float[] actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; ++i)
                Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, 4e-7f);
        }
        var settings = Settings();
        var template = Input(settings, 2);
        var input = new Ltx25DenoiserInput
        {
            Video = Values(fixture.GetProperty("noise")), VideoTokens = 5,
            Audio = template.Audio, AudioTokens = template.AudioTokens,
            VideoContext = template.VideoContext, VideoContextTokens = 1,
            AudioContext = template.AudioContext, AudioContextTokens = 1,
            VideoPositions = template.VideoPositions, AudioPositions = template.AudioPositions,
            VideoSigma = 1, AudioSigma = 1, FrequencyPrecision = template.FrequencyPrecision,
        };
        var images = fixture.GetProperty("images").EnumerateArray().Select(image => new Ltx25EncodedImage(
            Values(image.GetProperty("latent")), image.GetProperty("frame_index").GetInt32(), image.GetProperty("strength").GetSingle())).ToArray();
        var state = Ltx25ImageConditioning.Apply(input, settings, images);
        Close(Values(fixture.GetProperty("initial")), state.Video);
        Close(Values(fixture.GetProperty("clean")), state.CleanVideo);
        Close(Values(fixture.GetProperty("mask")), state.VideoDenoiseMask);
        Close(Values(fixture.GetProperty("keyframe_mask")), state.KeyframeMask);
        Close(Values(fixture.GetProperty("positions")), state.VideoPositions.Boundaries);
        var steps = fixture.GetProperty("steps").EnumerateArray().ToArray();
        int step = 0;
        var result = Ltx25Sampling.RunEuler(current =>
        {
            Close(step == 0 ? Values(fixture.GetProperty("initial")) : Values(steps[step - 1].GetProperty("output")), current.Video);
            Close(Values(steps[step].GetProperty("timesteps")), current.VideoTimesteps);
            return new(Values(steps[step++].GetProperty("velocity")), new float[current.Audio.Length], 48);
        }, state, Values(fixture.GetProperty("sigmas")));
        Close(Values(steps[^1].GetProperty("output")), result.Video);
        Close(Values(fixture.GetProperty("noise")), input.Video);
    }

    // One 32x32 single-stage pass: the official fixture's 1x1 latent grid. Requests
    // resolve to two stages at 64-pixel multiples; the conditioning math is per pass.
    private static Ltx25GenerationSettings Settings() => new(32, 32, 17, 24, 1, 1, 3, 18, 42);

    private static Ltx25DenoiserInput Input(Ltx25GenerationSettings settings, int guides = 0)
    {
        var noise = settings.InitialNoise(guides);
        return new()
        {
            Video = noise.Video, VideoTokens = settings.VideoTokens + guides,
            Audio = noise.Audio, AudioTokens = settings.AudioTokens,
            VideoContext = [0], VideoContextTokens = 1, AudioContext = [0], AudioContextTokens = 1,
            VideoPositions = Ltx25Geometry.VideoPositions(settings.LatentFrames, 1, 1, settings.Fps),
            AudioPositions = Ltx25Geometry.AudioPositions(settings.AudioTokens), VideoSigma = 1, AudioSigma = 1,
            FrequencyPrecision = Ltx25FrequencyPrecision.Float64,
        };
    }

    [Fact]
    public void OrderedUploadsBecomeFirstInteriorAndLastPixelFrameGuides()
    {
        var images = Ltx25ImageConditioning.Resolve(new() { ReferenceImagePaths = ["a", "b", "c"] }, 17);
        Assert.Equal(new[] { 0, 8, 16 }, images.Select(image => image.FrameIndex));
        Assert.Equal(new[] { "a", "b", "c" }, images.Select(image => image.ImagePath));
        var last = Ltx25ImageConditioning.Resolve(new() { ImageConditions = [new() { ImagePath = "last", FrameIndex = -1 }] }, 17);
        Assert.Equal(16, Assert.Single(last).FrameIndex);
    }

    [Theory]
    [InlineData(-2, 1)]
    [InlineData(17, 1)]
    [InlineData(0, -0.1f)]
    [InlineData(0, 1.1f)]
    [InlineData(0, float.NaN)]
    public void InvalidGuidesFailBeforeImageOrNetworkLoading(int frame, float strength)
    {
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new()
            { ImageConditions = [new() { ImagePath = "does-not-exist", FrameIndex = frame, Strength = strength }] }));
    }

    [Fact]
    public void ConflictingOrAmbiguousInputsCannotSilentlyDropImages()
    {
        Assert.Throws<ArgumentException>(() => Ltx25ImageConditioning.Resolve(new()
            { ImagePath = "a", ImageConditions = [new() { ImagePath = "b", FrameIndex = 0 }] }, 17));
        Assert.Throws<ArgumentException>(() => Ltx25ImageConditioning.Resolve(new()
            { ImagePath = "a", ReferenceImagePaths = ["b"] }, 17));
        Assert.Throws<ArgumentException>(() => Ltx25ImageConditioning.Resolve(new()
            { Mode = "t2v", ImagePath = "a" }, 17));
        Assert.Throws<ArgumentException>(() => Ltx25ImageConditioning.Resolve(new()
            { Mode = "fl2v", ImagePath = "a" }, 17));
        Assert.Throws<ArgumentException>(() => Ltx25ImageConditioning.Resolve(new()
            { ReferenceImagePaths = Enumerable.Repeat("a", 17).ToArray() }, 17));
        Assert.Throws<NotSupportedException>(() => Ltx25GenerationSettings.Resolve(new()
            { Mode = "ref", ReferenceImagePaths = ["identity.png"] }));
    }

    [Fact]
    public void FirstFrameIsReplacedAndOtherImagesAreAppendedWithoutGeneratedKeyframeMarker()
    {
        var settings = Settings();
        var input = Input(settings, 2);
        float[] original = (float[])input.Video.Clone();
        var first = Enumerable.Repeat(2f, 128).ToArray();
        var middle = Enumerable.Repeat(3f, 128).ToArray();
        var last = Enumerable.Repeat(4f, 128).ToArray();
        var result = Ltx25ImageConditioning.Apply(input, settings,
            [new(first, 0), new(middle, 7, .5f), new(last, 16)]);
        Assert.Equal(5, result.VideoTokens);
        Assert.Equal(new[] { 0f, 1f, 1f, .5f, 0f }, result.VideoDenoiseMask);
        Assert.Equal(first, result.Video.Take(128));
        Assert.Equal(last, result.Video.Skip(4 * 128));
        Assert.Equal(original, input.Video);
        Assert.Same(input.Audio, result.Audio);
        Assert.Equal(new[] { 1f, 0f, 0f, 0f, 0f }, result.KeyframeMask);
        Assert.Equal(7 / 24f, result.VideoPositions.Boundaries[18]);
        Assert.Equal(8 / 24f, result.VideoPositions.Boundaries[19]);
        Assert.Equal(16 / 24f, result.VideoPositions.Boundaries[24]);
        Assert.Equal(17 / 24f, result.VideoPositions.Boundaries[25]);
        for (int i = 0; i < 128; ++i)
            Assert.Equal(3 + (original[3 * 128 + i] - 3) * .5f, result.Video[3 * 128 + i]);
    }

    [Fact]
    public void RefinementPassNoisesGuidesFromTheirCleanLatentLikeGaussianNoiser()
    {
        // Stage two: GaussianNoiser lerps toward noise by the first refinement sigma
        // after the conditionings, then lerps back toward the clean latent by the mask.
        var settings = Settings();
        var input = Input(settings, 1);
        float[] raw = (float[])input.Video.Clone();
        const float scale = Ltx25GenerationSettings.RefinementNoiseScale;
        float[] prior = Enumerable.Range(0, 3 * 128).Select(i => (i % 7) * .25f).ToArray();
        for (int i = 0; i < prior.Length; ++i) input.Video[i] = prior[i] + (raw[i] - prior[i]) * scale;
        float[] noised = (float[])input.Video.Clone();
        var first = Enumerable.Repeat(2f, 128).ToArray();
        var middle = Enumerable.Repeat(3f, 128).ToArray();
        var result = Ltx25ImageConditioning.Apply(input, settings, [new(first, 0, .75f), new(middle, 8, .5f)], raw, scale);
        Assert.Equal(new[] { .25f, 1f, 1f, .5f }, result.VideoDenoiseMask);
        for (int i = 0; i < 128; ++i)
        {
            Assert.Equal(2 + (raw[i] - 2) * (.25f * scale), result.Video[i]);
            Assert.Equal(3 + (raw[3 * 128 + i] - 3) * (.5f * scale), result.Video[3 * 128 + i]);
        }
        Assert.Equal(noised.Skip(128).Take(2 * 128), result.Video.Skip(128).Take(2 * 128));
        Assert.Throws<ArgumentException>(() => Ltx25ImageConditioning.Apply(input, settings, [new(first, 0)], raw, 1f));
        Assert.Throws<ArgumentException>(() => Ltx25ImageConditioning.Apply(input, settings, [new(first, 0)], null, scale));
    }

    [Fact]
    public void ConditionedEulerUsesPerTokenTimestepsAndBlendsX0BeforeGlobalSigmaStep()
    {
        var settings = Settings();
        var initial = Ltx25ImageConditioning.Apply(Input(settings), settings,
            [new(Enumerable.Repeat(2f, 128).ToArray(), 0, .5f)]);
        var before = (float[])initial.Video.Clone();
        var result = Ltx25Sampling.RunEuler(input =>
        {
            Assert.Equal(new[] { .5f, 1f, 1f }, input.VideoTimesteps);
            Assert.Equal(1, input.VideoSigma);
            Assert.Equal(new[] { 1f, 0f, 0f }, input.KeyframeMask);
            return new(Enumerable.Repeat(.25f, input.Video.Length).ToArray(), new float[input.Audio.Length], 48);
        }, initial, [1f, 0f]);
        for (int i = 0; i < result.Video.Length; ++i)
        {
            float expected = i < 128 ? (before[i] - .25f * .5f) * .5f + 2 * .5f : before[i] - .25f;
            Assert.InRange(Math.Abs(expected - result.Video[i]), 0, 3e-7f);
        }
        Assert.Equal(before, initial.Video);
    }

    [Fact]
    public void FullyConditionedFirstImageRemainsCleanForAllEightSteps()
    {
        var settings = Settings();
        float[] clean = Enumerable.Range(0, 128).Select(i => i / 128f).ToArray();
        var initial = Ltx25ImageConditioning.Apply(Input(settings), settings, [new(clean, 0)]);
        var result = Ltx25Sampling.RunEuler(input =>
        {
            Assert.Equal(clean, input.Video.Take(128));
            Assert.Equal(0, input.VideoTimesteps![0]);
            return new(Enumerable.Repeat(100f, input.Video.Length).ToArray(), new float[input.Audio.Length], 48);
        }, initial, Ltx25Scheduler.DistilledSigmas());
        Assert.Equal(clean, result.Video.Take(128));
    }
}
