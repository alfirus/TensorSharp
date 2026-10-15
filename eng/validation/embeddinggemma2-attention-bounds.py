#!/usr/bin/env python3
"""Independently audit normalized attention and QKV/gate-up projection F16 range bounds.

Requires NumPy and the unchanged llama.cpp gguf-py reader/dequantizer. This reads
model weights only and performs no model inference or performance measurement.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reader = GGUFReader(str(args.model), mode="r")
    if reader.fields["general.architecture"].contents() != "gemma-embedding2":
        raise ValueError("Expected an EmbeddingGemma 2 GGUF")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    def norm(name):
        tensor = tensors[name]
        value = np.asarray(dequantize(tensor.data.reshape(1, -1), tensor.tensor_type), dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(value)):
            raise ValueError("Norm weight must be finite: " + name)
        return value
    rows = []
    projections = []
    hidden = int(reader.fields["gemma-embedding2.embedding_length"].contents())
    for layer in range(int(reader.fields["gemma-embedding2.block_count"].contents())):
        q = norm(f"blk.{layer}.attn_q_norm.weight")
        k = norm(f"blk.{layer}.attn_k_norm.weight")
        if len(q) != len(k) or not len(q):
            raise ValueError("Query and key head norm lengths must match")
        head = len(q)
        qmax, kmax = float(np.max(np.abs(q))), float(np.max(np.abs(k)))
        dot = head * qmax * kmax
        component = max(qmax, kmax) * math.sqrt(head)
        rows.append(dict(layer=layer, head=head, q_norm_max=qmax, k_norm_max=kmax,
                         dot_bound=dot, component_bound=component,
                         half_safe=dot <= 16384 and component <= 16384))
        for kind, norm_name, names in [("qkv", "attn_norm", ["attn_q", "attn_k", "attn_v"]),
                                      ("gate_up", "ffn_norm", ["ffn_gate", "ffn_up"])]:
            gains = norm(f"blk.{layer}.{norm_name}.weight").astype(np.float64)
            input_bound = 2 * math.sqrt(hidden) * float(np.max(np.abs(gains)))
            output_bound = 0.0
            weight_max = 0.0
            for name in names:
                tensor = tensors[f"blk.{layer}.{name}.weight"]
                weight = np.asarray(dequantize(tensor.data.reshape(int(tensor.shape[1]), -1), tensor.tensor_type), dtype=np.float32)
                if not np.all(np.isfinite(weight)):
                    raise ValueError("Projection weight must be finite: " + name)
                weight_max = max(weight_max, float(np.max(np.abs(weight))))
                half_weight = weight.astype(np.float16).astype(np.float64)
                weighted = half_weight * gains
                bounds = 2 * np.sqrt(hidden * np.sum(weighted * weighted, axis=1)) + np.sum(np.abs(half_weight), axis=1) * 2**-24
                output_bound = max(output_bound, float(np.max(bounds)))
            projections.append(dict(layer=layer, projection=kind, input_component_bound=input_bound,
                                    partial_or_full_dot_bound=output_bound, weight_max=weight_max,
                                    half_safe=input_bound <= 16384 and output_bound <= 16384 and weight_max <= 16384))
    digest = hashlib.sha256()
    with args.model.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    report = dict(model=str(args.model.resolve()), model_sha256=digest.hexdigest(),
                  proof="RMS gives head L2 <= sqrt(head)*maxnormweight; Cauchy-Schwarz bounds every partial/full QK dot.",
                  operand_and_score_margin=16384, layers=rows,
                  all_half_safe=all(row["half_safe"] for row in rows),
                  maximum_dot_bound=max(row["dot_bound"] for row in rows),
                  maximum_component_bound=max(row["component_bound"] for row in rows),
                  projection_proof="RMS row L2<=sqrt(hidden); Cauchy-Schwarz with exact F16-rounded weights bounds full and partial dots. Factor2 covers norm/input rounding; add absolute F16 subnormal rounding bound.",
                  projections=projections, all_half_projection_safe=all(row["half_safe"] for row in projections),
                  maximum_projection_dot_bound=max(row["partial_or_full_dot_bound"] for row in projections))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("all_half_safe", "maximum_dot_bound", "maximum_component_bound", "all_half_projection_safe", "maximum_projection_dot_bound")}))


if __name__ == "__main__":
    main()
