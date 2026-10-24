// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Security.Cryptography;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class GlmTeacherForcingNumericsTests
{
    private static void Near(double expected, double actual, double tolerance = 1e-13)
        => Assert.True(Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}.");

    [Fact]
    public void SingletonHasUnitProbabilityAndNoAlternativeMargin()
    {
        var score = GlmTeacherForcingNumerics.Score([37.5f], 0);
        Assert.Equal(0, score.Target); Assert.Equal(0, score.Argmax); Assert.Equal(1, score.TargetRank);
        Assert.Equal(0, score.Nll); Assert.Equal(0, score.TargetLogProbability);
        Assert.Equal(37.5, score.LogNormalizer); Assert.Equal(0, score.TopMargin);
    }

    [Theory]
    [InlineData(2, 0)]
    [InlineData(2, 1)]
    [InlineData(4, 3)]
    [InlineData(17, 8)]
    public void ExplicitUniformProbabilityIsOneOverVocabulary(int size, int target)
    {
        var score = GlmTeacherForcingNumerics.Score(new float[size], target);
        Near(Math.Log(size), score.Nll); Near(-Math.Log(size), score.TargetLogProbability);
        Near(Math.Log(size), score.LogNormalizer);
        Assert.Equal(target + 1, score.TargetRank); Assert.Equal(0, score.Argmax); Assert.Equal(0, score.TopMargin);
    }

    [Fact]
    public void SmallNonuniformDistributionMatchesDirectKnownNormalizer()
    {
        // Independent unshifted formula is safe for these tiny exact logits.
        double z = 1 + Math.E + Math.E * Math.E;
        var score = GlmTeacherForcingNumerics.Score([0f, 1f, 2f], 1);
        Near(Math.Log(z), score.LogNormalizer);
        Near(1 - Math.Log(z), score.TargetLogProbability); Near(Math.Log(z) - 1, score.Nll);
        Assert.Equal(2, score.Argmax); Assert.Equal(2, score.TargetRank); Assert.Equal(1, score.TopMargin);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(1048576f)]
    [InlineData(-1048576f)]
    public void ExactlyRepresentableCommonShiftPreservesProbabilities(float offset)
    {
        var original = GlmTeacherForcingNumerics.Score([0f, 1f, 2f, -3f], 1);
        var shifted = GlmTeacherForcingNumerics.Score([offset, offset + 1, offset + 2, offset - 3], 1);
        Assert.Equal(original.Nll, shifted.Nll);
        Assert.Equal(original.TargetLogProbability, shifted.TargetLogProbability);
        Assert.Equal(original.Argmax, shifted.Argmax); Assert.Equal(original.TargetRank, shifted.TargetRank);
        Assert.Equal(original.TopMargin, shifted.TopMargin);
    }

    [Theory]
    [InlineData(float.MaxValue)]
    [InlineData(-float.MaxValue)]
    public void HugeEqualOffsetCannotRoundAwayLogTwo(float value)
    {
        var score = GlmTeacherForcingNumerics.Score([value, value], 1);
        Near(Math.Log(2), score.Nll); Near(-Math.Log(2), score.TargetLogProbability);
        Assert.Equal((double)value, score.LogNormalizer);
        Assert.Equal(0d, score.LogNormalizer - (double)value); // Naive subtraction would lose NLL.
        Assert.Equal(0, score.Argmax); Assert.Equal(2, score.TargetRank); Assert.Equal(0, score.TopMargin);
    }

    [Fact]
    public void OppositeFloatExtremesSubtractOnlyAfterF64Conversion()
    {
        double expected = 2d * float.MaxValue;
        var score = GlmTeacherForcingNumerics.Score([float.MaxValue, -float.MaxValue], 1);
        Assert.True(double.IsFinite(score.Nll));
        Assert.Equal(expected, score.Nll); Assert.Equal(-expected, score.TargetLogProbability);
        Assert.Equal(expected, score.TopMargin); Assert.Equal(0, score.Argmax); Assert.Equal(2, score.TargetRank);
    }

    [Fact]
    public void UnderflowedUnselectedExponentialsDoNotMakeTargetLossInfinite()
    {
        var score = GlmTeacherForcingNumerics.Score([0f, -1000f, -2000f], 2);
        Assert.Equal(2000, score.Nll); Assert.Equal(-2000, score.TargetLogProbability);
        Assert.Equal(0, score.LogNormalizer); Assert.Equal(1000, score.TopMargin);
    }

    [Theory]
    [InlineData(0, 4)]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 5)]
    [InlineData(4, 3)]
    public void EqualLogitsUseLowestTokenIdForArgmaxAndRank(int target, int rank)
    {
        var score = GlmTeacherForcingNumerics.Score([2f, 3f, 3f, 1f, 3f], target);
        Assert.Equal(1, score.Argmax); Assert.Equal(rank, score.TargetRank); Assert.Equal(0, score.TopMargin);
    }

    [Fact]
    public void AllTargetProbabilitiesOfSmallDistributionConserveOne()
    {
        float[] logits = [-2f, 0f, 1f, 4f];
        double total = 0;
        for (int target = 0; target < logits.Length; ++target)
            total += Math.Exp(GlmTeacherForcingNumerics.Score(logits, target).TargetLogProbability);
        Near(1, total);
    }

    [Fact]
    public void ChecksumCoversExactNativeF32BytesAndScoringDoesNotMutateInput()
    {
        float[] logits = [BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)), 0f, 1f, -2.5f];
        int[] before = logits.Select(BitConverter.SingleToInt32Bits).ToArray();
        byte[] expectedBytes = [0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 32, 192];
        if (!BitConverter.IsLittleEndian)
            for (int first = 0; first < expectedBytes.Length; first += 4) Array.Reverse(expectedBytes, first, 4);
        string expected = Convert.ToHexString(SHA256.HashData(expectedBytes)).ToLowerInvariant();
        var firstScore = GlmTeacherForcingNumerics.Score(logits, 0);
        var otherScore = GlmTeacherForcingNumerics.Score(logits, 3);
        Assert.Equal(expected, firstScore.LogitsSha256); Assert.Equal(expected, otherScore.LogitsSha256);
        Assert.Equal(before, logits.Select(BitConverter.SingleToInt32Bits).ToArray());
        Assert.Matches("^[0-9a-f]{64}$", firstScore.LogitsSha256);
    }

    [Fact]
    public void SignedZeroHasEqualProbabilityButDifferentExactVectorHash()
    {
        var plus = GlmTeacherForcingNumerics.Score([0f, 0f], 1);
        var minus = GlmTeacherForcingNumerics.Score([0f, BitConverter.Int32BitsToSingle(unchecked((int)0x80000000))], 1);
        Assert.Equal(plus.Nll, minus.Nll); Assert.Equal(plus.Argmax, minus.Argmax);
        Assert.Equal(plus.TargetRank, minus.TargetRank); Assert.NotEqual(plus.LogitsSha256, minus.LogitsSha256);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(2)]
    [InlineData(int.MinValue)]
    [InlineData(int.MaxValue)]
    public void InvalidTeacherIdIsRejected(int target)
        => Assert.Throws<ArgumentOutOfRangeException>(() => GlmTeacherForcingNumerics.Score([0f, 1f], target));

    [Fact]
    public void EmptyVectorIsRejected()
        => Assert.Throws<ArgumentException>(() => GlmTeacherForcingNumerics.Score([], 0));

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void AnyNonfiniteLogitIsRejectedEvenWhenNotTarget(float invalid)
    {
        Assert.Throws<ArgumentException>(() => GlmTeacherForcingNumerics.Score([0f, invalid], 0));
        Assert.Throws<ArgumentException>(() => GlmTeacherForcingNumerics.Score([invalid, 0f], 0));
    }

    [Fact]
    public void ExactMaximumVocabularyIsValidAndOneExtraEntryIsRejected()
    {
        int maximum = GlmTeacherForcingNumerics.MaximumLogits;
        var score = GlmTeacherForcingNumerics.Score(new float[maximum], maximum - 1);
        Near(Math.Log(maximum), score.Nll); Assert.Equal(maximum, score.TargetRank); Assert.Equal(0, score.Argmax);
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmTeacherForcingNumerics.Score(new float[maximum + 1], 0));
    }
}
