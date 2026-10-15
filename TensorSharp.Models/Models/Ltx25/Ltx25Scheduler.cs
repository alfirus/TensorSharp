// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Models.Ltx25;

/// <summary>LTX flow-matching schedules and Euler stepping. DiT consumes sigma
/// directly; H3's inverted/dual-shift carrier rules do not apply.</summary>
public static class Ltx25Scheduler
{
    public static float[] DistilledSigmas() => [1f, .99375f, .9875f, .98125f, .975f, .909375f, .725f, .421875f, 0f];
    public static float[] RefinementSigmas() => [.909375f, .725f, .421875f, 0f];
    public static float[] BuildSigmas(int steps, int tokens = 4096, double baseShift = .95, double maxShift = 2.05,
        bool stretch = true, float terminal = .1f)
    {
        if (steps <= 0 || steps > 100000 || tokens <= 0 || !double.IsFinite(baseShift) || !double.IsFinite(maxShift) ||
            !float.IsFinite(terminal) || terminal <= 0 || terminal >= 1) throw new ArgumentOutOfRangeException(nameof(steps));
        // The original LTX2Scheduler stretches its sole nonzero sigma by 0/0
        // at steps=1. Define the bounded single Euler step explicitly instead.
        // This guard is deliberate, not a claim of exact NaN-source parity.
        if (steps == 1) return [1f, 0f];
        double slope = (maxShift - baseShift) / (4096 - 1024);
        float shift = (float)Math.Exp(tokens * slope + baseShift - slope * 1024);
        if (!float.IsFinite(shift) || shift <= 0) throw new ArgumentException("LTX sigma shift exceeds Float32 range.");
        var sigmas = new float[steps + 1];
        for (int i = 0; i < steps; ++i)
        {
            float sigma = 1f - (float)i / steps;
            sigmas[i] = shift / (shift + (1f / sigma - 1f));
        }
        if (stretch)
        {
            float scale = (1f - sigmas[steps - 1]) / (1f - terminal);
            if (!(scale > 0) || !float.IsFinite(scale)) throw new ArgumentException("LTX schedule cannot be stretched at this geometry.");
            for (int i = 0; i < steps; ++i) sigmas[i] = 1f - (1f - sigmas[i]) / scale;
        }
        return sigmas;
    }
    public static void EulerStep(ReadOnlySpan<float> sample, ReadOnlySpan<float> velocity, float sigma, float nextSigma, Span<float> output)
    {
        if (sample.Length != velocity.Length || sample.Length != output.Length || !float.IsFinite(sigma) || !float.IsFinite(nextSigma) ||
            sigma <= 0 || sigma > 1 || nextSigma < 0 || nextSigma > sigma) throw new ArgumentException("Invalid LTX Euler step.");
        if (sample.Overlaps(output, out int sampleOffset) && sampleOffset != 0 ||
            velocity.Overlaps(output, out int velocityOffset) && velocityOffset != 0)
            throw new ArgumentException("LTX Euler output may alias a complete input, not a shifted input slice.");
        float delta = nextSigma - sigma;
        for (int i = 0; i < sample.Length; ++i)
        {
            if (!float.IsFinite(sample[i]) || !float.IsFinite(velocity[i])) throw new ArgumentException("LTX Euler inputs must be finite.");
            float value = sample[i] + velocity[i] * delta;
            if (!float.IsFinite(value)) throw new ArgumentException("LTX Euler step exceeds Float32 range.");
            output[i] = value;
        }
    }
}
