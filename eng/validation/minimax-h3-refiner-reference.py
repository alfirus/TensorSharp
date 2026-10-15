#!/usr/bin/env python3
"""Arbitrate matched H3 text-refiner boundaries with independent PyTorch F32.

Reads exact text input from the unchanged SD CLI public eval callback. Reuses
the corrected independent fixture oracle, with no refiner positional embedding.
This diagnostic is excluded from generation/performance timings.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def metrics(reference, candidate):
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return {"elements": a.size, "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
            "cosine": float(np.dot(a.ravel(), b.ravel()) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
            "relative_l2": float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-30)),
            "max_abs": float(np.max(np.abs(a-b)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing oracle report")
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    root = Path(__file__).resolve().parents[2]
    source = root / "InferenceWeb.Tests/tools/minimax_h3_dit_oracle.py"
    spec = importlib.util.spec_from_file_location("h3_dit_oracle", source)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    metadata = [json.loads(path.read_text()) for path in args.reference.glob("*.f32.json")]
    projection = [item for item in metadata if item["file"].startswith("projection-") and
                  any("condition_proj.weight" in src["name"] for src in item["sources"])]
    if len(projection) != 1:
        raise ValueError("Require exactly one condition projection")
    input_id = projection[0]["sources"][1]["id"]
    input_item = next(item for item in metadata if item["id"] == input_id)
    input_path = args.reference / input_item["file"]
    text = torch.from_numpy(np.fromfile(input_path, dtype="<f4").reshape(input_item["ne"][1], input_item["ne"][0]))
    weights = oracle.Weights(str(args.gguf))
    head_dim = weights.get("token_refiner.blocks.0.attn.q_norm.weight").numel()
    heads = weights.get("token_refiner.blocks.0.attn.qkv_proj.weight").shape[0] // (3*head_dim)
    cfg = {"heads": heads, "head_dim": head_dim}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = args.output.parent / (args.output.stem + "-snapshots")
    snapshots.mkdir(exist_ok=False)
    result = {"scope": "Independent PyTorch CPU F32 with exact SD callback text input; diagnostic only",
              "torch": torch.__version__, "threads": args.threads, "text_tokens": text.shape[0],
              "input_file": str(input_path), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "oracle_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "boundaries": {}}
    def record(name, value):
        independent = value.detach().numpy().astype("<f4", copy=False)
        independent.tofile(snapshots / (name + ".f32"))
        sd = [item for item in metadata if item["file"].startswith(name + "-")]
        if len(sd) != 1:
            raise ValueError(f"Require exactly one {name} boundary")
        candidate = args.candidate / (name + ".f32")
        result["boundaries"][name] = {
            "stable_diffusion_cpp": metrics(independent, np.fromfile(args.reference/sd[0]["file"], dtype="<f4").reshape(independent.shape)),
            "tensorsharp": metrics(independent, np.fromfile(candidate, dtype="<f4").reshape(independent.shape))}
    ctx = F.linear(text, weights.get("condition_proj.weight"), weights.get("condition_proj.bias"))
    record("condition-proj", ctx)
    index = 0
    while f"token_refiner.blocks.{index}.norm1.weight" in weights:
        prefix = f"token_refiner.blocks.{index}."
        h = oracle.rms_norm(ctx, weights.get(prefix+"norm1.weight"), 1e-5)
        attention = {key: weights.get(prefix+suffix) for key, suffix in
                     [("qkv", "attn.qkv_proj.weight"), ("out", "attn.out_proj.weight"),
                      ("q_norm", "attn.q_norm.weight"), ("k_norm", "attn.k_norm.weight")]}
        ctx = ctx + oracle.attention(h, attention, cfg, None, None, 1e-5)
        h = oracle.rms_norm(ctx, weights.get(prefix+"norm2.weight"), 1e-5)
        ctx = ctx + oracle.mlp(h, {"fc1": weights.get(prefix+"mlp.fc1.weight"),
                                 "fc2": weights.get(prefix+"mlp.fc2.weight")})
        record("refiner-"+str(index), ctx)
        index += 1
    refined = oracle.rms_norm(ctx, weights.get("token_refiner.final_norm.weight"), 1e-5).numpy()
    independent = refined.astype("<f4", copy=False)
    independent.tofile(snapshots / "refined-text.f32")
    count, width = independent.shape
    packed = [item for item in metadata if item["file"].startswith("packed-in-") and item["ne"][0] == width and item["ne"][1] >= count]
    if not packed:
        raise ValueError("Missing packed sequence")
    first = min(packed, key=lambda item: item["id"])
    result["boundaries"]["refined-text"] = {
        "stable_diffusion_cpp": metrics(independent, np.fromfile(args.reference/first["file"], dtype="<f4").reshape(-1, width)[:count]),
        "tensorsharp": metrics(independent, np.fromfile(args.candidate/"packed-in.f32", dtype="<f4").reshape(-1, width)[:count])}
    result["completed"] = True
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
