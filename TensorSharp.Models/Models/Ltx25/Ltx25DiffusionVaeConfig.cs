// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace TensorSharp.Models.Ltx25;

/// <summary>One deterministic-stage upsample: a Linear, then a channels-last pixel
/// shuffle by <see cref="Stride"/> (t,h,w) that divides channels by <see cref="Reduction"/>.</summary>
internal sealed record Ltx25DiffusionVaeUpsample(int[] Stride, int Reduction);

/// <summary>Architecture of the publisher's diffusion video decoder
/// (<c>CausalDiffusionVAE</c> / <c>NADiffusionDecoder</c>), read as the original
/// <c>_build_diffusion_video_decoder</c> reads it: stage ladders from
/// <c>vae.decoder</c>, <c>model_output_type</c> from <c>vae</c>.</summary>
internal sealed record Ltx25DiffusionVaeConfig(int LatentChannels, int OutputChannels, int PatchSize, int HeadDim,
    int TimestepEmbedDim, int[] StageChannels, int[] StageDepths, int[][] StageKernels,
    Ltx25DiffusionVaeUpsample[] Upsamples, int[] Stage5Kernel, int Stage5Channels, float TimestepScale)
{
    internal const float NormEpsilon = 1e-6f;
    internal const float RopeBase = 10000f;

    /// <summary>Stage-4 output channels, the diffusion stage's context width.</summary>
    internal int ContextChannels => StageChannels[^1];
    internal int DiffusionBlocks => StageDepths[^1];
    internal int PixelChannels => OutputChannels * PatchSize * PatchSize;

    /// <summary>The original default split of a head across (t, h, w) rotary axes.</summary>
    internal (int T, int H, int W) RopeSplit
    {
        get
        {
            int t = HeadDim / 4 / 2 * 2, hw = (HeadDim - t) / 2;
            if (hw % 2 != 0) { t -= 2; hw = (HeadDim - t) / 2; }
            return (t, hw, hw);
        }
    }

    /// <summary>True for a checkpoint whose embedded config declares the diffusion decoder.</summary>
    internal static bool IsDiffusionVae(IReadOnlyDictionary<string, string> metadata)
    {
        if (metadata == null || !metadata.TryGetValue("config", out string json)) return false;
        using var doc = JsonDocument.Parse(json);
        return doc.RootElement.TryGetProperty("vae", out var vae) && vae.TryGetProperty("_class_name", out var name) &&
            name.GetString() == "CausalDiffusionVAE";
    }

    internal static Ltx25DiffusionVaeConfig Parse(IReadOnlyDictionary<string, string> metadata)
    {
        if (!metadata.TryGetValue("config", out string json))
            throw new InvalidDataException("The LTX diffusion video VAE requires embedded architecture metadata.");
        using var doc = JsonDocument.Parse(json);
        var vae = doc.RootElement.GetProperty("vae");
        if (!vae.TryGetProperty("_class_name", out var className) || className.GetString() != "CausalDiffusionVAE")
            throw new NotSupportedException("The LTX diffusion decoder requires a CausalDiffusionVAE checkpoint.");
        var decoder = vae.TryGetProperty("decoder", out var nested) ? nested : vae;
        int Number(JsonElement element, string key, int fallback) => element.TryGetProperty(key, out var value) ? value.GetInt32() : fallback;
        int[] Ints(JsonElement element) => element.EnumerateArray().Select(value => value.GetInt32()).ToArray();
        string outputType = vae.TryGetProperty("model_output_type", out var type) ? type.GetString() : "v";
        int steps = Number(decoder, "default_num_inference_steps", 2);
        // The published checkpoint predicts x0 in one step from pure noise at t = 1.
        // Multi-step or velocity decoders would need the original Euler loop.
        if (outputType != "x0" || steps != 1)
            throw new NotSupportedException("The LTX diffusion decoder supports the published one-step x0 checkpoint only.");
        if (Number(decoder, "out_channels", 3) != 3)
            throw new NotSupportedException("The LTX diffusion decoder produces RGB only.");
        int[] channels = decoder.TryGetProperty("stage_channels", out var c) ? Ints(c) : [1024, 512, 256, 256, 128];
        int[] depths = decoder.TryGetProperty("stage_depths", out var d) ? Ints(d) : [4, 6, 4, 2, 8];
        int[][] kernels = decoder.TryGetProperty("stage_kernels", out var k)
            ? k.EnumerateArray().Select(Ints).ToArray() : [[3, 7, 7], [3, 7, 7], [3, 5, 5], [3, 5, 5], [3, 3, 3]];
        var upsamples = decoder.TryGetProperty("upsamples", out var u)
            ? u.EnumerateArray().Select(entry => new Ltx25DiffusionVaeUpsample(Ints(entry[0]), entry[1].GetInt32())).ToArray()
            : [new([1, 2, 2], 2), new([2, 1, 1], 2), new([2, 2, 2], 1), new([2, 2, 2], 2)];
        int[] stage5 = decoder.TryGetProperty("stage5_kernel", out var s5) ? Ints(s5) : [3, 7, 7];
        var config = new Ltx25DiffusionVaeConfig(
            Number(decoder, "in_channels", Number(vae, "latent_channels", 128)), 3,
            Number(decoder, "patch_size", 4), Number(decoder, "head_dim", Number(decoder, "na_head_dim", 64)),
            Number(decoder, "t_emb_dim", 384), channels, depths, kernels, upsamples, stage5,
            Number(decoder, "stage5_channels", channels.Length == 0 ? 0 : channels[^1]),
            decoder.TryGetProperty("timestep_scale_multiplier", out var scale) ? scale.GetSingle() : 1f);
        config.Validate();
        return config;
    }

    private void Validate()
    {
        if (LatentChannels <= 0 || PatchSize <= 0 || HeadDim <= 0 || HeadDim % 4 != 0 || TimestepEmbedDim <= 0 ||
            !float.IsFinite(TimestepScale) || StageChannels.Length != 5 || StageDepths.Length != 5 ||
            StageKernels.Length != 5 || Upsamples.Length != 4 || Stage5Kernel.Length != 3)
            throw new InvalidDataException("Invalid LTX diffusion decoder dimensions.");
        if (Stage5Channels != ContextChannels)
            throw new NotSupportedException("A diffusion stage wider than its context is not supported.");
        var (t, h, w) = RopeSplit;
        if (t <= 0 || h <= 0 || t % 2 != 0 || h % 2 != 0 || t + h + w != HeadDim)
            throw new InvalidDataException("The LTX diffusion decoder head does not split across three rotary axes.");
        for (int stage = 0; stage < 5; ++stage)
        {
            if (StageChannels[stage] <= 0 || StageChannels[stage] % HeadDim != 0 || StageDepths[stage] <= 0 ||
                StageKernels[stage].Length != 3 || StageKernels[stage].Any(value => value <= 0 || value % 2 == 0))
                throw new InvalidDataException("LTX diffusion decoder stages need positive head-aligned widths and odd 3D windows.");
            if (stage < 4)
            {
                var upsample = Upsamples[stage];
                if (upsample.Stride.Length != 3 || upsample.Stride.Any(value => value is not (1 or 2)) || upsample.Reduction <= 0 ||
                    StageChannels[stage] % upsample.Reduction != 0 || StageChannels[stage] / upsample.Reduction != StageChannels[stage + 1])
                    throw new InvalidDataException("LTX diffusion decoder upsamples do not connect its stage widths.");
            }
        }
        if (Stage5Kernel.Any(value => value <= 0 || value % 2 == 0))
            throw new InvalidDataException("The LTX diffusion stage needs an odd 3D window.");
        if (Enumerable.Range(0, 4).Aggregate(1, (product, stage) => product * Upsamples[stage].Stride[0]) != 8 ||
            Enumerable.Range(0, 4).Aggregate(1, (product, stage) => product * Upsamples[stage].Stride[1]) * PatchSize != 32 ||
            Enumerable.Range(0, 4).Aggregate(1, (product, stage) => product * Upsamples[stage].Stride[2]) * PatchSize != 32)
            throw new NotSupportedException("The LTX diffusion decoder must expand latents 8x in time and 32x in space.");
    }

    /// <summary>Per-axis latent floor so every stage's window fits (original
    /// <c>all_stages_min_tile_size</c>); smaller latents are edge padded.</summary>
    internal int[] MinimumLatent()
    {
        int[] minimum = [1, 1, 1], cumulative = [1, 1, 1];
        for (int stage = 0; stage < 4; ++stage)
        {
            for (int axis = 0; axis < 3; ++axis)
                minimum[axis] = Math.Max(minimum[axis], (StageKernels[stage][axis] + cumulative[axis] - 1) / cumulative[axis]);
            for (int axis = 0; axis < 3; ++axis) cumulative[axis] *= Upsamples[stage].Stride[axis];
        }
        for (int axis = 0; axis < 3; ++axis)
            minimum[axis] = Math.Max(minimum[axis], (Stage5Kernel[axis] + cumulative[axis] - 1) / cumulative[axis]);
        return minimum;
    }
}
