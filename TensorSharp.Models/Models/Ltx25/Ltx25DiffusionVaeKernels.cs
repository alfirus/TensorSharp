// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Threading.Tasks;
using TensorSharp.Cpu;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;

namespace TensorSharp.Models.Ltx25;

/// <summary>Neighborhood attention and absolute rotary embedding for the LTX
/// diffusion video decoder, on the MLX device or the CPU. Both follow the
/// original's NATTEN semantics: on each axis a query at index i attends to the
/// window starting at clamp(i - k/2, 0, L - k), k = min(kernel, L), so windows
/// shift inward at the volume's edges rather than being masked.</summary>
internal static class Ltx25DiffusionVaeKernels
{
    /// <summary>The start of index <paramref name="index"/>'s window on an axis of <paramref name="length"/>.</summary>
    internal static int WindowStart(int index, int kernel, int length)
    {
        int k = Math.Min(kernel, length);
        return Math.Clamp(index - k / 2, 0, length - k);
    }

    /// <summary>output, q: [TQ*H*W, heads*headDim] for frames qT0..qT0+TQ; k, v:
    /// [TK*H*(W+2*padW), heads*headDim] for frames kT0..; T is the whole volume's length.
    /// With <paramref name="padW"/> = 0 every axis has NATTEN windows. A positive pad is
    /// the original W-slab pathway's edge: keys extend padW edge-replicated columns on
    /// each side and a column's window is centered on it.</summary>
    internal static void Attention(DirectContext context, Tensor output, Tensor q, Tensor k, Tensor v,
        int frames, int height, int width, int queryFrame, int keyFrame, int heads, int headDim, int[] kernel, int padW = 0)
    {
        int keyWidth = width + 2 * padW;
        int queryFrames = checked((int)(q.Sizes[0] / ((long)height * width)));
        int keyFrames = checked((int)(k.Sizes[0] / ((long)height * keyWidth)));
        int kt = Math.Min(kernel[0], frames);
        if (WindowStart(queryFrame, kt, frames) < keyFrame ||
            WindowStart(queryFrame + queryFrames - 1, kt, frames) + kt > keyFrame + keyFrames)
            throw new ArgumentException("LTX neighborhood attention keys do not cover the query windows.");
        if (context.IsMlx)
        {
            using var parameters = context.FromFloats([frames, height, width, queryFrame, keyFrame, queryFrames, keyFrames, keyWidth, padW], 9);
            if (!MlxLtx25DiffusionVaeOps.TryNeighborhoodAttention(output, q, k, v, parameters, heads, headDim, kernel[0], kernel[1], kernel[2], height, width))
                throw new NotSupportedException("MLX LTX neighborhood attention is unavailable.");
            return;
        }
        if (context.IsCuda) throw new NotSupportedException("The LTX diffusion video decoder has no CUDA neighborhood attention.");
        unsafe
        {
            float* qp = (float*)CpuNativeHelpers.GetBufferStart(q), kp = (float*)CpuNativeHelpers.GetBufferStart(k);
            float* vp = (float*)CpuNativeHelpers.GetBufferStart(v), op = (float*)CpuNativeHelpers.GetBufferStart(output);
            int channels = heads * headDim;
            int kh = Math.Min(kernel[1], height), kw = Math.Min(kernel[2], keyWidth);
            Parallel.For(0, queryFrames * height * width, token =>
            {
                int w = token % width, h = token / width % height, t = queryFrame + token / (width * height);
                int st = WindowStart(t, kt, frames) - keyFrame, sh = WindowStart(h, kh, height);
                int sw = Math.Clamp(w + padW - kw / 2, 0, keyWidth - kw);
                Span<float> accumulator = stackalloc float[headDim];
                for (int head = 0; head < heads; ++head)
                {
                    float* query = qp + (long)token * channels + head * headDim;
                    accumulator.Clear();
                    float best = float.NegativeInfinity, total = 0;
                    for (int dt = 0; dt < kt; ++dt)
                    for (int dh = 0; dh < kh; ++dh)
                    for (int dw = 0; dw < kw; ++dw)
                    {
                        long row = ((long)(st + dt) * height + sh + dh) * keyWidth + sw + dw;
                        float* key = kp + row * channels + head * headDim, value = vp + row * channels + head * headDim;
                        float score = 0;
                        for (int d = 0; d < headDim; ++d) score += query[d] * key[d];
                        if (score > best)
                        {
                            float correction = MathF.Exp(best - score);
                            total *= correction;
                            for (int d = 0; d < headDim; ++d) accumulator[d] *= correction;
                            best = score;
                        }
                        float weight = MathF.Exp(score - best);
                        total += weight;
                        for (int d = 0; d < headDim; ++d) accumulator[d] += weight * value[d];
                    }
                    float* target = op + (long)token * channels + head * headDim;
                    for (int d = 0; d < headDim; ++d) target[d] = accumulator[d] / total;
                }
            });
        }
    }

    /// <summary>Returns rotated input [n*H*W, heads*headDim] (frames from t0 on an HxW
    /// grid), scaled, as [n*H*(W+2*padW), heads*headDim]: a positive pad edge-replicates
    /// columns whose rotary positions continue past the edge (-padW.., W..), as the
    /// original W-slab halos do. Zero frequencies make it a plain replication.</summary>
    internal static Tensor Rope(DirectContext context, Tensor input, Tensor frequencies, float[] hostFrequencies,
        int height, int width, int firstFrame, int heads, int headDim, (int T, int H, int W) split, float scale, int padW = 0)
    {
        int outWidth = width + 2 * padW;
        long frames = input.Sizes[0] / ((long)height * width);
        var output = context.NewF32(frames * height * outWidth, input.Sizes[1]);
        try
        {
            if (context.IsMlx)
            {
                using var parameters = context.FromFloats([height, width, firstFrame, scale, padW], 5);
                if (!MlxLtx25DiffusionVaeOps.TryAbsoluteRope(output, input, frequencies, parameters, heads, headDim, split.T, split.H))
                    throw new NotSupportedException("MLX LTX absolute RoPE is unavailable.");
                return output;
            }
            if (context.IsCuda) throw new NotSupportedException("The LTX diffusion video decoder has no CUDA rotary kernel.");
            unsafe
            {
                float* x = (float*)CpuNativeHelpers.GetBufferStart(input), y = (float*)CpuNativeHelpers.GetBufferStart(output);
                int pairs = headDim / 2, tPairs = split.T / 2, hPairs = (split.T + split.H) / 2;
                Parallel.For(0, checked((int)(frames * height * outWidth)), token =>
                {
                    int column = token % outWidth, row = token / outWidth;
                    int t = firstFrame + row / height, h = row % height;
                    long source = (long)row * width + Math.Clamp(column - padW, 0, width - 1);
                    for (int head = 0; head < heads; ++head)
                        for (int pair = 0; pair < pairs; ++pair)
                        {
                            int position = pair < tPairs ? t : pair < hPairs ? h : column - padW;
                            float angle = position * hostFrequencies[pair];
                            (float sin, float cos) = MathF.SinCos(angle);
                            long input = (source * heads + head) * headDim + 2 * pair;
                            long offset = ((long)token * heads + head) * headDim + 2 * pair;
                            float even = x[input], odd = x[input + 1];
                            y[offset] = (even * cos - odd * sin) * scale;
                            y[offset + 1] = (even * sin + odd * cos) * scale;
                        }
                });
            }
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    /// <summary>The original inverse frequencies 1/base^(i/d) for even i, computed in
    /// Float64 and rounded, per axis, flattened in the head's (t, h, w) pair order.</summary>
    internal static float[] RopeFrequencies((int T, int H, int W) split, float ropeBase)
    {
        var result = new float[(split.T + split.H + split.W) / 2];
        int index = 0;
        foreach (int dim in new[] { split.T, split.H, split.W })
            for (int i = 0; i < dim; i += 2) result[index++] = (float)(1.0 / Math.Pow(ropeBase, (double)i / dim));
        return result;
    }
}
