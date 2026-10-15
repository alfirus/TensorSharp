// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;

namespace TensorSharp.Models;

/// <summary>Time-aligned, bandlimited PCM resampling for the Gemma audio frontend.</summary>
internal static class Gemma4AudioResampler
{
    // The Hann-windowed sinc algorithm follows torchaudio's polyphase resampler.
    // A wider filter than its six-crossing default suppresses aliases near the
    // destination Nyquist frequency. Original 16 kHz inputs are left unchanged.
    internal const int FilterWidth = 64;
    internal const double Rolloff = .99;
    private sealed record Kernel(int First, float[] Weights);
    private sealed record Plan(int Original, int Target, double Base, double Support, Kernel[] Kernels);
    private static readonly object CacheLock = new();
    private static readonly Dictionary<(int, int), Plan> Cache = new();

    internal static float[] Resample(float[] samples, int originalRate, int targetRate)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (originalRate <= 0 || targetRate <= 0) throw new ArgumentOutOfRangeException(nameof(originalRate));
        foreach (float value in samples)
            if (!float.IsFinite(value)) throw new InvalidDataException("Audio contains a non-finite sample.");
        if (originalRate == targetRate || samples.Length == 0) return samples;
        int length = checked((int)(((long)samples.Length * targetRate + originalRate - 1) / originalRate));
        Plan plan = GetPlan(originalRate, targetRate);
        var result = new float[length];
        for (int index = 0; index < length; ++index)
        {
            int phase = index % plan.Target;
            long block = (long)(index / plan.Target) * plan.Original;
            double sum = 0;
            if (plan.Kernels != null)
            {
                Kernel kernel = plan.Kernels[phase];
                long start = block + kernel.First;
                int first = (int)Math.Max(0, -start);
                int last = (int)Math.Min(kernel.Weights.Length, samples.LongLength - start);
                for (int k = first; k < last; ++k) sum += (double)samples[start + k] * kernel.Weights[k];
            }
            else
            {
                // Unusual coprime rates must not create unbounded phase caches.
                // Evaluate only the available input support, including for tiny
                // waveforms and very large declared source rates.
                double fraction = (float)phase / plan.Target;
                double center = block + fraction * plan.Original;
                int first = (int)Math.Max(0, Math.Ceiling(center - plan.Support));
                int last = (int)Math.Min(samples.LongLength - 1, Math.Floor(center + plan.Support));
                for (int source = first; source <= last; ++source)
                    sum += (double)samples[source] * Weight((source - block) / (double)plan.Original - fraction, plan);
            }
            result[index] = (float)sum;
        }
        return result;
    }

    private static Plan GetPlan(int originalRate, int targetRate)
    {
        lock (CacheLock) if (Cache.TryGetValue((originalRate, targetRate), out var cached)) return cached;
        int a = originalRate, b = targetRate;
        while (b != 0) { int remainder = a % b; a = b; b = remainder; }
        int original = originalRate / a, target = targetRate / a;
        double baseFrequency = Math.Min(original, target) * Rolloff;
        double support = FilterWidth * original / baseFrequency;
        bool cacheable = target <= 2048 && support <= 4096 && target * (2 * support + 3) <= 1_048_576;
        var kernels = cacheable ? new Kernel[target] : null;
        var plan = new Plan(original, target, baseFrequency, support, kernels);
        if (!cacheable) return plan;
        for (int phase = 0; phase < target; ++phase)
        {
            // torchaudio's phase grid is float32 before its float64 kernel
            // construction. Retain that rounding for independent comparisons.
            double fraction = (float)phase / target;
            int first = (int)Math.Ceiling(fraction * original - support);
            int last = (int)Math.Floor(fraction * original + support);
            var weights = new float[last - first + 1];
            for (int k = 0; k < weights.Length; ++k) weights[k] = Weight((first + k) / (double)original - fraction, plan);
            kernels[phase] = new Kernel(first, weights);
        }
        lock (CacheLock)
        {
            if (Cache.TryGetValue((originalRate, targetRate), out var cached)) return cached;
            if (Cache.Count == 8)
            {
                using var keys = Cache.Keys.GetEnumerator(); keys.MoveNext(); Cache.Remove(keys.Current);
            }
            Cache.Add((originalRate, targetRate), plan);
        }
        return plan;
    }

    private static float Weight(double position, Plan plan)
    {
        double t = position * plan.Base;
        if (Math.Abs(t) >= FilterWidth) return 0;
        double window = Math.Cos(t * Math.PI / FilterWidth / 2);
        double angle = t * Math.PI;
        return (float)((angle == 0 ? 1 : Math.Sin(angle) / angle) * window * window * plan.Base / plan.Original);
    }
}
