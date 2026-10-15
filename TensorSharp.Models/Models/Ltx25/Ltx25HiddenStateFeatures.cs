// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models.Ltx25;

/// <summary>A stable token permutation and the resulting binary attention mask.
/// SourceTokenIndices can be reused for both video and audio features.</summary>
public sealed record Ltx25RightPadOrder(int[] SourceTokenIndices, int[] AttentionMask);

/// <summary>Batch-one Float32 preparation of caller-supplied Gemma hidden states
/// for the LTX 2.5 V2 feature extractor. This does not extract hidden states, run
/// learned aggregate projections or connectors, or produce denoiser contexts.
/// The authentic encoder uses BF16; its dtype-specific rounding is not provided
/// by these explicitly Float32 operations.</summary>
public static class Ltx25HiddenStateFeatures
{
    /// <summary>Each state is token-major [tokens, hiddenSize], in the original
    /// encoder's hidden-state tuple order (including its embedding state). RMS
    /// normalization is independent per token and state, over hiddenSize only.
    /// Output is [tokens, hiddenSize, stateCount] flattened with state varying
    /// fastest, as in torch.stack(states, dim=-1).reshape(B,T,D*L). Padding rows
    /// are zero. Inputs are unchanged; all states must contain finite Float32.</summary>
    public static float[] NormalizeAndPackFloat32(IReadOnlyList<float[]> hiddenStates,
        int hiddenSize, ReadOnlySpan<int> attentionMask)
    {
        ArgumentNullException.ThrowIfNull(hiddenStates);
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        ValidateMask(attentionMask);
        if (hiddenStates.Count == 0) throw new ArgumentException("At least one precomputed hidden state is required.", nameof(hiddenStates));
        int stateLength = checked(attentionMask.Length * hiddenSize);
        int featureWidth = checked(hiddenSize * hiddenStates.Count);
        int outputLength = checked(attentionMask.Length * featureWidth);
        foreach (float[] state in hiddenStates)
        {
            if (state == null || state.Length != stateLength)
                throw new ArgumentException("Each hidden state must be token-major [mask.Length, hiddenSize].", nameof(hiddenStates));
            ValidateFinite(state, nameof(hiddenStates));
        }
        var output = new float[outputLength];
        for (int token = 0; token < attentionMask.Length; ++token)
        {
            if (attentionMask[token] == 0) continue;
            int sourceOffset = token * hiddenSize, outputOffset = token * featureWidth;
            for (int layer = 0; layer < hiddenStates.Count; ++layer)
            {
                float[] state = hiddenStates[layer];
                float squares = 0;
                for (int hidden = 0; hidden < hiddenSize; ++hidden)
                {
                    float value = state[sourceOffset + hidden];
                    squares += value * value;
                }
                if (!float.IsFinite(squares))
                    throw new ArgumentException("Hidden-state RMS overflows Float32.", nameof(hiddenStates));
                float inverseRms = 1f / MathF.Sqrt(squares / hiddenSize + 1e-6f);
                for (int hidden = 0; hidden < hiddenSize; ++hidden)
                    output[outputOffset + hidden * hiddenStates.Count + layer] = state[sourceOffset + hidden] * inverseRms;
            }
        }
        return output;
    }

    /// <summary>Prepare a separate input for a learned aggregate projection by
    /// multiplying normalized packed features by sqrt(projectionWidth/hiddenSize).
    /// The divisor is the encoder hidden size, not the flattened state width.
    /// This does not execute the projection or its bias. Inputs are unchanged.</summary>
    public static float[] RescaleForAggregateProjectionFloat32(ReadOnlySpan<float> normalizedFeatures,
        int hiddenSize, int projectionWidth)
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (projectionWidth <= 0) throw new ArgumentOutOfRangeException(nameof(projectionWidth));
        if (normalizedFeatures.Length == 0) throw new ArgumentException("Features must be nonempty.", nameof(normalizedFeatures));
        ValidateFinite(normalizedFeatures, nameof(normalizedFeatures));
        float scale = (float)Math.Sqrt((double)projectionWidth / hiddenSize);
        var output = new float[normalizedFeatures.Length];
        for (int i = 0; i < output.Length; ++i)
        {
            output[i] = normalizedFeatures[i] * scale;
            if (!float.IsFinite(output[i]))
                throw new ArgumentException("Feature rescaling overflows Float32.", nameof(normalizedFeatures));
        }
        return output;
    }

    /// <summary>Place valid tokens before padding, preserving the original order
    /// within both groups. Compute once and reuse the source indices across the
    /// learned video and audio projection outputs before their connectors.</summary>
    public static Ltx25RightPadOrder CreateRightPadOrder(ReadOnlySpan<int> attentionMask)
    {
        ValidateMask(attentionMask);
        var order = new int[attentionMask.Length];
        var mask = new int[attentionMask.Length];
        int next = 0;
        for (int token = 0; token < attentionMask.Length; ++token)
            if (attentionMask[token] == 1) { order[next] = token; mask[next++] = 1; }
        for (int token = 0; token < attentionMask.Length; ++token)
            if (attentionMask[token] == 0) order[next++] = token;
        return new(order, mask);
    }

    /// <summary>Copy token-major [tokens, featureWidth] rows according to a full
    /// permutation of token indices. Padding values are retained: aggregate
    /// projection biases can make them nonzero before register replacement.</summary>
    public static float[] ApplyTokenOrderFloat32(ReadOnlySpan<float> features, int featureWidth,
        ReadOnlySpan<int> sourceTokenIndices)
    {
        if (featureWidth <= 0) throw new ArgumentOutOfRangeException(nameof(featureWidth));
        if (sourceTokenIndices.Length == 0 || features.Length != checked(sourceTokenIndices.Length * featureWidth))
            throw new ArgumentException("Features must have one complete row per token index.", nameof(features));
        var seen = new bool[sourceTokenIndices.Length];
        foreach (int source in sourceTokenIndices)
        {
            if ((uint)source >= (uint)seen.Length || seen[source])
                throw new ArgumentException("Source token indices must be a full permutation.", nameof(sourceTokenIndices));
            seen[source] = true;
        }
        ValidateFinite(features, nameof(features));
        var output = new float[features.Length];
        for (int token = 0; token < sourceTokenIndices.Length; ++token)
            features.Slice(sourceTokenIndices[token] * featureWidth, featureWidth)
                .CopyTo(output.AsSpan(token * featureWidth, featureWidth));
        return output;
    }

    private static void ValidateMask(ReadOnlySpan<int> mask)
    {
        if (mask.Length == 0) throw new ArgumentException("A nonempty binary attention mask is required.", nameof(mask));
        foreach (int value in mask)
            if (value != 0 && value != 1) throw new ArgumentException("Attention mask values must be zero or one.", nameof(mask));
    }

    private static void ValidateFinite(ReadOnlySpan<float> values, string parameter)
    {
        foreach (float value in values)
            if (!float.IsFinite(value)) throw new ArgumentException("Hidden states and features must be finite Float32.", parameter);
    }
}
