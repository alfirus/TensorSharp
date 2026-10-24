// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace InferenceWeb.Tests;

internal readonly record struct GlmTeacherForcingScore(
    int Target, int Argmax, int TargetRank, double TargetLogProbability,
    double Nll, double LogNormalizer, double TopMargin, string LogitsSha256);

/// <summary>Scores one already-produced full F32 logit vector. This does not
/// select/inject a teacher, tokenize text, or qualify a model's quality. The
/// checksum covers the exact unchanged in-process F32 bytes (native endian).
/// The span is borrowed and must remain stable for this synchronous call.</summary>
internal static class GlmTeacherForcingNumerics
{
    internal const int MaximumLogits = 1_000_000;

    internal static GlmTeacherForcingScore Score(ReadOnlySpan<float> logits, int target)
    {
        if (logits.IsEmpty)
            throw new ArgumentException("A full nonempty logit vector is required.", nameof(logits));
        if (logits.Length > MaximumLogits)
            throw new ArgumentOutOfRangeException(nameof(logits), "Logit vector exceeds the diagnostic bound.");
        if ((uint)target >= (uint)logits.Length)
            throw new ArgumentOutOfRangeException(nameof(target));

        double maximum = double.NegativeInfinity, runnerUp = double.NegativeInfinity;
        int argmax = 0;
        for (int i = 0; i < logits.Length; ++i)
        {
            double value = logits[i]; // Exact F32 -> F64 conversion before subtraction.
            if (!double.IsFinite(value))
                throw new ArgumentException("Every logit must be finite.", nameof(logits));
            if (value > maximum)
            {
                runnerUp = maximum;
                maximum = value;
                argmax = i;
            }
            else if (value > runnerUp)
                runnerUp = value;
        }

        double sumExp = 0, targetValue = logits[target];
        int rank = 1;
        for (int i = 0; i < logits.Length; ++i)
        {
            double value = logits[i];
            sumExp += Math.Exp(value - maximum);
            // Stable total ordering: higher value first, then lower token ID.
            if (value > targetValue || (value == targetValue && i < target))
                ++rank;
        }
        double logSumExp = Math.Log(sumExp);
        // Do NOT compute targetValue - (maximum + logSumExp). With a huge
        // common offset the addition rounds away logSumExp, losing the NLL.
        double logProbability = (targetValue - maximum) - logSumExp;
        if (logProbability == 0) logProbability = 0; // Canonical positive zero.
        double nll = logProbability == 0 ? 0 : -logProbability;
        double margin = logits.Length == 1 ? 0 : maximum - runnerUp;
        double logNormalizer = maximum + logSumExp;
        if (!double.IsFinite(logProbability) || !double.IsFinite(nll)
            || !double.IsFinite(logNormalizer) || !double.IsFinite(margin)
            || logProbability > 0 || nll < 0 || margin < 0)
            throw new InvalidOperationException("Finite bounded logits produced invalid F64 scoring metrics.");

        Span<byte> digest = stackalloc byte[SHA256.HashSizeInBytes];
        SHA256.HashData(MemoryMarshal.AsBytes(logits), digest);
        return new(target, argmax, rank, logProbability, nll, logNormalizer, margin,
            Convert.ToHexString(digest).ToLowerInvariant());
    }
}
