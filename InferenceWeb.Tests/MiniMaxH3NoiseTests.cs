using System.Globalization;
using System.Text.Json;
using TensorSharp.Models.MiniMaxH3;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3NoiseTests
{
    private static readonly JsonElement[] Cases = Load();
    public static IEnumerable<object[]> ReferenceCases => Cases.Select((value, index) => new object[] { index });

    [Theory]
    [MemberData(nameof(ReferenceCases))]
    public void InitialPackedNoiseMatchesUnchangedStableDiffusionCpp(int caseIndex)
    {
        JsonElement reference = Cases[caseIndex];
        MiniMaxH3Shape shape = Shape(reference);
        long seed = Seed(reference);
        var (video, audio) = MiniMaxH3Noise.Initial(shape, seed);
        Assert.Equal(reference.GetProperty("packedLength").GetInt32(), shape.PackedLatentLength);
        Equal(reference.GetProperty("video"), video, "patchified video");
        Equal(reference.GetProperty("audio"), audio, "stereo audio");
        var repeated = MiniMaxH3Noise.Initial(shape, seed);
        Assert.Equal(video, repeated.Video);
        Assert.Equal(audio, repeated.Audio);
    }

    [Theory]
    [MemberData(nameof(ReferenceCases))]
    public void VisualConditionNoiseMatchesReferenceLayoutAndReseedsEachLatent(int caseIndex)
    {
        JsonElement reference = Cases[caseIndex];
        MiniMaxH3Shape shape = Shape(reference);
        float[] latent = Enumerable.Range(0, checked((int)shape.VideoLatentLength))
            .Select(index => (index % 19 - 9) * .125f).ToArray();
        float[] repeated = (float[])latent.Clone();
        MiniMaxH3Noise.AddVisual(latent, Seed(reference), shape.LatentFrames, shape.LatentHeight, shape.LatentWidth);
        Equal(reference.GetProperty("conditioned"), latent, "THWC visual conditioning");
        MiniMaxH3Noise.AddVisual(repeated, Seed(reference), shape.LatentFrames, shape.LatentHeight, shape.LatentWidth);
        Assert.Equal(latent, repeated);
    }

    private static long Seed(JsonElement reference) => unchecked((long)ulong.Parse(
        reference.GetProperty("seed").GetString()!, CultureInfo.InvariantCulture));

    private static MiniMaxH3Shape Shape(JsonElement reference)
    {
        int frames = reference.GetProperty("frames").GetInt32(), height = reference.GetProperty("height").GetInt32(),
            width = reference.GetProperty("width").GetInt32(), audio = reference.GetProperty("audioFrames").GetInt32();
        int volume = frames * height * width;
        return new MiniMaxH3Shape
        {
            LatentFrames = frames, LatentHeight = height, LatentWidth = width, AudioLatentFrames = audio,
            PackedExtraChannels = (audio * 2 * 32 + volume - 1) / volume,
        };
    }

    private static void Equal(JsonElement expected, float[] actual, string stage)
    {
        Assert.Equal(expected.GetArrayLength(), actual.Length);
        for (int i = 0; i < actual.Length; ++i)
        {
            Assert.True(float.IsFinite(actual[i]), $"{stage}[{i}] must be finite");
            double error = Math.Abs((double)actual[i] - expected[i].GetSingle());
            Assert.True(error <= 1e-6, $"{stage}[{i}] error {error:G9} exceeds independent reference tolerance 1e-6");
        }
    }

    private static JsonElement[] Load()
    {
        using Stream stream = typeof(MiniMaxH3NoiseTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.MiniMaxH3.noise-reference.json")!;
        using JsonDocument document = JsonDocument.Parse(stream);
        Assert.Equal("d637cfb935feb2254898a8f1524046c958f3df80", document.RootElement.GetProperty("reference").GetProperty("revision").GetString());
        return document.RootElement.GetProperty("cases").EnumerateArray().Select(value => value.Clone()).ToArray();
    }
}
