// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Linq;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>Checkpoint-derived placement for the opt-in, dense-attention Flash
/// draft. Header geometry is not a proof of numerical or throughput parity.</summary>
internal sealed record GlmUnifiedMtpGeometry(long DenseDeviceBytes, long ExpertSourceBytes,
    GlmUnifiedExpertLayout ExpertLayout, int MaximumContext)
{
    // Set only after the explicit opt-in and matching native capability check.
    internal bool ShapeSpecificPlan { get; init; }
    internal bool PrefixStateCommit { get; init; }
    internal static GlmUnifiedMtpGeometry Read(GgufFile file, GlmUnifiedMemoryGeometry trunk)
    {
        string a = trunk.Architecture;
        uint next = file.GetUint32(a + ".nextn_predict_layers");
        int topk = checked((int)file.GetUint32(a + ".attention.indexer.top_k"));
        uint[]? layout = file.GetUint32Array(a + ".attention.head_count_kv");
        if (next != 1 || topk <= 0 || layout == null || layout.Length != trunk.Layers + 1 || layout[^1] != 1)
            throw new NotSupportedException("Flash MTP requires one complete MLA NextN block and an explicit dense-context bound.");
        string prefix = $"blk.{trunk.Layers}.";
        void Require(string name, params ulong[] shape)
        {
            if (!file.Tensors.TryGetValue(prefix + name, out var tensor) || !tensor.Shape.SequenceEqual(shape))
                throw new NotSupportedException("Incomplete Flash MTP checkpoint geometry: " + prefix + name);
        }
        ulong h = (ulong)trunk.Hidden;
        Require("nextn.eh_proj.weight", 2 * h, h);
        Require("nextn.enorm.weight", h); Require("nextn.hnorm.weight", h);
        Require("nextn.shared_head_norm.weight", h);
        foreach (string name in new[] { "attn_norm.weight", "ffn_norm.weight", "attn_q_a.weight",
            "attn_q_a_norm.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
            "attn_k_b.weight", "attn_v_b.weight", "attn_output.weight", "ffn_gate_inp.weight",
            "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight" })
            if (!file.Tensors.ContainsKey(prefix + name)) throw new NotSupportedException("Incomplete Flash MTP checkpoint: " + prefix + name);
        long dense = 0, experts = 0;
        GlmUnifiedExpertProjection Projection(string part, bool down)
        {
            string name = prefix + "ffn_" + part + "_exps.weight";
            if (!file.Tensors.TryGetValue(name, out var tensor)) throw new NotSupportedException("Incomplete Flash MTP expert triple: " + name);
            long width = down ? trunk.ExpertFf : trunk.Hidden, rows = down ? trunk.Hidden : trunk.ExpertFf;
            if (!tensor.Shape.SequenceEqual(new[] { (ulong)width, (ulong)rows, (ulong)trunk.ExpertCount }))
                throw new NotSupportedException("Flash MTP expert shape disagrees with the trunk: " + name);
            long block = GgufFile.GetBlockSize(tensor.Type), size = GgufFile.GetTypeSize(tensor.Type);
            long bytes = file.GetTensorFileRegion(name).ByteLength;
            long padding = (512 - width % 512) % 512;
            bool supported = block > 1 && (uint)tensor.Type <= (uint)GgmlTensorType.Q2_0 && padding % block == 0;
            if (width % block != 0 || checked(width / block * size * rows * trunk.ExpertCount) != bytes || !supported)
                throw new NotSupportedException("Flash MTP requires supported exact quantized expert payloads: " + name);
            experts = checked(experts + bytes);
            return new(tensor.Type, bytes / trunk.ExpertCount, padding / block * size, true);
        }
        var expertLayout = new GlmUnifiedExpertLayout(Projection("gate", false), Projection("up", false), Projection("down", true));
        foreach (var tensor in file.Tensors.Values.Where(t => t.Name.StartsWith(prefix, StringComparison.Ordinal)))
            if (!MoeCpuOffloadConfig.IsRoutedExpertWeightName(tensor.Name))
                dense = checked(dense + file.GetTensorFileRegion(tensor.Name).ByteLength + (64L << 10));
        return new(dense, experts, expertLayout, Math.Min(2048, topk));
    }
}
