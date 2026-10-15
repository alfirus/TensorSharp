using System.Text.Json;
using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

public sealed class Ltx25HiddenStateFeaturesTests
{
    private static JsonDocument Fixture() => JsonDocument.Parse(typeof(Ltx25HiddenStateFeaturesTests).Assembly
        .GetManifestResourceStream("InferenceWeb.Tests.Fixtures.Ltx25.hidden-state-features.json")!);

    private static float[] Floats(JsonElement value) => value.EnumerateArray().Select(item => item.GetSingle()).ToArray();
    private static int[] Ints(JsonElement value) => value.EnumerateArray().Select(item => item.GetInt32()).ToArray();

    private static void Close(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i)
            Assert.True(float.IsFinite(actual[i]) && Math.Abs(expected[i] - actual[i]) <= 2e-6f,
                $"Element {i}: expected {expected[i]:R}, got {actual[i]:R}");
    }

    [Theory]
    [InlineData("left")]
    [InlineData("right")]
    [InlineData("scattered")]
    [InlineData("all_pad")]
    [InlineData("all_valid")]
    public void Float32FeaturesAndSharedTokenPermutationMatchPinnedOriginal(string name)
    {
        using JsonDocument fixture = Fixture();
        JsonElement root = fixture.RootElement;
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", root.GetProperty("source_revision").GetString());
        Assert.Equal("float32", root.GetProperty("dtype").GetString());
        JsonElement testCase = root.GetProperty("cases").EnumerateArray().Single(item => item.GetProperty("name").GetString() == name);
        float[][] states = root.GetProperty("hidden_states").EnumerateArray().Select(Floats).ToArray();
        float[][] originalStates = states.Select(state => (float[])state.Clone()).ToArray();
        int[] mask = Ints(testCase.GetProperty("attention_mask"));
        int[] originalMask = (int[])mask.Clone();
        int hiddenSize = root.GetProperty("hidden_size").GetInt32();

        float[] normalized = Ltx25HiddenStateFeatures.NormalizeAndPackFloat32(states, hiddenSize, mask);
        Close(Floats(testCase.GetProperty("normalized")), normalized);
        float[] videoScaled = Ltx25HiddenStateFeatures.RescaleForAggregateProjectionFloat32(normalized,
            hiddenSize, root.GetProperty("video_projection_width").GetInt32());
        float[] audioScaled = Ltx25HiddenStateFeatures.RescaleForAggregateProjectionFloat32(normalized,
            hiddenSize, root.GetProperty("audio_projection_width").GetInt32());
        Close(Floats(testCase.GetProperty("video_scaled")), videoScaled);
        Close(Floats(testCase.GetProperty("audio_scaled")), audioScaled);
        Close(Floats(testCase.GetProperty("normalized")), normalized);

        Ltx25RightPadOrder order = Ltx25HiddenStateFeatures.CreateRightPadOrder(mask);
        Assert.Equal(Ints(testCase.GetProperty("source_token_order")), order.SourceTokenIndices);
        Assert.Equal(Ints(testCase.GetProperty("right_padded_mask")), order.AttentionMask);
        // Separate projected feature widths share exactly one order. These rows
        // include nonzero padding values, as learned projection biases would.
        float[] videoRows = Floats(root.GetProperty("video_reorder_input"));
        float[] audioRows = Floats(root.GetProperty("audio_reorder_input"));
        float[] videoReordered = Ltx25HiddenStateFeatures.ApplyTokenOrderFloat32(videoRows,
            root.GetProperty("video_reorder_width").GetInt32(), order.SourceTokenIndices);
        float[] audioReordered = Ltx25HiddenStateFeatures.ApplyTokenOrderFloat32(audioRows,
            root.GetProperty("audio_reorder_width").GetInt32(), order.SourceTokenIndices);
        Assert.Equal(Floats(testCase.GetProperty("video_reordered")), videoReordered);
        Assert.Equal(Floats(testCase.GetProperty("audio_reordered")), audioReordered);
        Assert.Equal(Floats(root.GetProperty("video_reorder_input")), videoRows);
        Assert.Equal(Floats(root.GetProperty("audio_reorder_input")), audioRows);
        Assert.Equal(originalMask, mask);
        for (int layer = 0; layer < states.Length; ++layer) Assert.Equal(originalStates[layer], states[layer]);
        // Reordering an already right-padded mask must be idempotent.
        Assert.Equal(Enumerable.Range(0, mask.Length), Ltx25HiddenStateFeatures.CreateRightPadOrder(order.AttentionMask).SourceTokenIndices);
    }

    [Theory]
    [InlineData("no_states")]
    [InlineData("null_states")]
    [InlineData("null_state")]
    [InlineData("short_state")]
    [InlineData("hidden_zero")]
    [InlineData("empty_mask")]
    [InlineData("negative_mask")]
    [InlineData("nonbinary_mask")]
    [InlineData("nan_valid")]
    [InlineData("nan_padding")]
    [InlineData("rms_overflow")]
    public void MalformedHiddenStatesCannotProduceApparentlyValidFeatures(string kind)
    {
        float[][] states = [[1f, 2f, 3f, 4f]];
        int hidden = 2;
        int[] mask = [0, 1];
        switch (kind)
        {
            case "no_states": states = []; break;
            case "null_states": states = null!; break;
            case "null_state": states = [null!]; break;
            case "short_state": states = [[1f]]; break;
            case "hidden_zero": hidden = 0; break;
            case "empty_mask": mask = []; break;
            case "negative_mask": mask[0] = -1; break;
            case "nonbinary_mask": mask[0] = 2; break;
            case "nan_valid": states[0][3] = float.NaN; break;
            case "nan_padding": states[0][0] = float.NaN; break;
            case "rms_overflow": states[0][2] = float.MaxValue; break;
        }
        Assert.ThrowsAny<ArgumentException>(() => Ltx25HiddenStateFeatures.NormalizeAndPackFloat32(states, hidden, mask));
    }

    [Fact]
    public void OversizedFeatureShapeFailsBeforeAllocation()
    {
        Assert.Throws<OverflowException>(() => Ltx25HiddenStateFeatures.NormalizeAndPackFloat32([[1f]], int.MaxValue, [1, 1]));
        Assert.Throws<OverflowException>(() => Ltx25HiddenStateFeatures.ApplyTokenOrderFloat32([1f], int.MaxValue, [0, 1]));
    }

    [Theory]
    [InlineData("hidden_zero")]
    [InlineData("projection_zero")]
    [InlineData("empty")]
    [InlineData("nonfinite")]
    [InlineData("overflow")]
    public void InvalidProjectionPreparationFailsExplicitly(string kind)
    {
        float[] features = kind == "empty" ? [] : kind == "nonfinite" ? [float.NegativeInfinity]
            : kind == "overflow" ? [float.MaxValue] : [1f];
        Assert.ThrowsAny<ArgumentException>(() => Ltx25HiddenStateFeatures.RescaleForAggregateProjectionFloat32(features,
            kind == "hidden_zero" ? 0 : 1, kind == "projection_zero" ? 0 : 4));
    }

    [Theory]
    [InlineData("width_zero")]
    [InlineData("empty_order")]
    [InlineData("short_rows")]
    [InlineData("duplicate")]
    [InlineData("out_of_range")]
    [InlineData("negative")]
    [InlineData("nonfinite")]
    public void InvalidTokenOrderCannotDropOrRepeatFeatureRows(string kind)
    {
        float[] features = kind == "short_rows" ? [1f] : kind == "nonfinite" ? [1f, float.NaN] : [1f, 2f];
        int[] order = kind switch
        {
            "empty_order" => [], "duplicate" => [0, 0], "out_of_range" => [0, 2],
            "negative" => [-1, 0], _ => [1, 0],
        };
        Assert.ThrowsAny<ArgumentException>(() => Ltx25HiddenStateFeatures.ApplyTokenOrderFloat32(features,
            kind == "width_zero" ? 0 : 1, order));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(2)]
    public void StableRightPaddingRequiresANonemptyBinaryMask(int invalid)
    {
        int[] mask = invalid == 0 ? [] : [1, invalid];
        Assert.Throws<ArgumentException>(() => Ltx25HiddenStateFeatures.CreateRightPadOrder(mask));
    }
}
