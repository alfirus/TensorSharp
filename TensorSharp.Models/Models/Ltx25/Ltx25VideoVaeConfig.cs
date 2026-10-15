// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace TensorSharp.Models.Ltx25;

internal sealed record Ltx25VideoVaeBlock(string Kind, int Layers, int Multiplier);

/// <summary>The deterministic pixel-normalized convolutional decoder architecture.</summary>
internal sealed record Ltx25VideoVaeConfig(int LatentChannels, int BaseChannels, int PatchSize,
    Ltx25VideoVaeBlock[] Blocks)
{
    internal int BottleneckChannels
    {
        get
        {
            int result = BaseChannels;
            foreach (var block in Blocks) result = checked(result * block.Multiplier);
            return result;
        }
    }
    internal int TemporalScale => Scale(true);
    internal int SpatialScale => checked(PatchSize * Scale(false));
    private int Scale(bool temporal)
    {
        int result = 1;
        foreach (var block in Blocks)
            if (block.Kind == "compress_all" || block.Kind == (temporal ? "compress_time" : "compress_space"))
                result = checked(result * 2);
        return result;
    }

    internal static Ltx25VideoVaeConfig Parse(IReadOnlyDictionary<string, string> metadata)
    {
        if (!metadata.TryGetValue("config", out string json))
            throw new InvalidDataException("The LTX convolutional video VAE requires embedded architecture metadata.");
        using var doc = JsonDocument.Parse(json);
        var vae = doc.RootElement.GetProperty("vae");
        string Kind(string key, string fallback) => vae.TryGetProperty(key, out var value) ? value.GetString() : fallback;
        int Number(string key, int fallback) => vae.TryGetProperty(key, out var value) ? value.GetInt32() : fallback;
        bool Flag(string key, bool fallback) => vae.TryGetProperty(key, out var value) ? value.GetBoolean() : fallback;
        if (Kind("_class_name", "") != "CausalVideoAutoencoder" || Number("dims", 3) != 3 ||
            Number("out_channels", 3) != 3 || Kind("norm_layer", "pixel_norm") != "pixel_norm" ||
            Kind("spatial_padding_mode", "reflect") != "zeros" || Flag("causal_decoder", false) ||
            Flag("timestep_conditioning", true) || Flag("use_quant_conv", false))
            throw new NotSupportedException("LTX video decode requires a non-causal, zero-spatial-padded convolutional VAE with pixel normalization and no timestep/quant-conv conditioning.");
        int channels = Number("latent_channels", 128), baseChannels = Number("decoder_base_channels", 128), patch = Number("patch_size", 4);
        if (channels <= 0 || baseChannels <= 0 || patch <= 0 || patch > 16)
            throw new InvalidDataException("Invalid LTX VAE channel or patch dimensions.");
        var blocks = new List<Ltx25VideoVaeBlock>();
        foreach (var entry in vae.GetProperty("decoder_blocks").EnumerateArray())
        {
            string kind = entry[0].GetString();
            var parameters = entry[1];
            int layers = parameters.ValueKind == JsonValueKind.Number ? parameters.GetInt32()
                : parameters.TryGetProperty("num_layers", out var n) ? n.GetInt32() : 0;
            int multiplier = parameters.ValueKind == JsonValueKind.Object && parameters.TryGetProperty("multiplier", out var m)
                ? m.GetInt32() : 1;
            if (kind is not ("res_x" or "compress_time" or "compress_space" or "compress_all") ||
                multiplier <= 0 || (kind == "res_x" && (layers <= 0 || multiplier != 1)) ||
                (parameters.ValueKind == JsonValueKind.Object &&
                    ((parameters.TryGetProperty("inject_noise", out var noise) && noise.GetBoolean()) ||
                     (parameters.TryGetProperty("residual", out var residual) && residual.GetBoolean()))))
                throw new NotSupportedException("Unsupported or stochastic LTX convolutional video VAE block.");
            blocks.Add(new(kind, layers, multiplier));
        }
        if (blocks.Count == 0) throw new InvalidDataException("LTX VAE decoder blocks are empty.");
        var result = new Ltx25VideoVaeConfig(channels, baseChannels, patch, blocks.ToArray());
        _ = result.BottleneckChannels; _ = result.TemporalScale; _ = result.SpatialScale;
        return result;
    }
}
