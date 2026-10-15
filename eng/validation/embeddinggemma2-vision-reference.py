#!/usr/bin/env python3
"""Independent HF Gemma4 vision math for the EmbeddingGemma 2 GGUF projector.

gguf-py only reads/dequantizes weights. The HF-derived NumPy operations in
eng/diffusiongemma-vision-oracle.py define axial rotation, GELU, RMSNorm and
pooling; this adapter supplies this tower's geometry and converted tensor names.
Input is exact channel-first [0,1] F32 pixels, avoiding resize differences.
Evidence belongs under ignored docs/validation/ or artifacts/.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


class VisionEncoder:
    def __init__(self, model):
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.reader = GGUFReader(str(model), mode="r")
        self.dequantize = dequantize
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        self.cache = {}
        self.hidden = int(self.metadata("clip.vision.embedding_length", 768))
        self.heads = int(self.metadata("clip.vision.attention.head_count", 12))
        self.layers = int(self.metadata("clip.vision.block_count", 16))
        self.patch = int(self.metadata("clip.vision.patch_size", 16))
        self.merge = int(self.metadata("clip.vision.projector_scale_factor",
            self.metadata("clip.vision.projector.scale_factor", 3)))
        self.epsilon = float(self.metadata("clip.vision.attention.layer_norm_epsilon", 1e-6))
        if self.hidden % self.heads or (self.hidden // self.heads) % 4:
            raise ValueError("Invalid axial head geometry")
        spec = importlib.util.spec_from_file_location("hf_vision_reference",
            Path(__file__).resolve().parents[1] / "diffusiongemma-vision-oracle.py")
        self.hf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.hf)

    def metadata(self, name, default):
        field = self.reader.fields.get(name)
        return default if field is None else field.contents()

    def weight(self, name):
        if name not in self.cache:
            tensor = self.tensors[name]
            self.cache[name] = np.asarray(self.dequantize(tensor.data, tensor.tensor_type), dtype=np.float32).reshape(tuple(reversed(tensor.shape)))
        return self.cache[name]

    def linear(self, x, name):
        for suffix in ("input_min", "input_max"):
            if name + "." + suffix in self.tensors:
                limit = float(self.weight(name + "." + suffix)[0])
                x = np.maximum(x, limit) if suffix == "input_min" else np.minimum(x, limit)
        result = x @ self.weight(name + ".weight").T
        if name + ".bias" in self.tensors: result += self.weight(name + ".bias")
        for suffix in ("output_min", "output_max"):
            if name + "." + suffix in self.tensors:
                limit = float(self.weight(name + "." + suffix)[0])
                result = np.maximum(result, limit) if suffix == "output_min" else np.minimum(result, limit)
        return result.astype(np.float32)

    def encode(self, pixels):
        _, height, width = pixels.shape
        rows, columns = height // self.patch, width // self.patch
        if height % self.patch or width % self.patch or rows % self.merge or columns % self.merge:
            raise ValueError("Pixels must form a complete pooling grid")
        patches = self.hf.convert_image_to_patches(pixels, self.patch)
        projection = self.weight("v.patch_embd.weight").transpose(0, 2, 3, 1).reshape(self.hidden, -1)
        x = np.float32(2) * (patches - np.float32(.5))
        x = x @ projection.T
        if "v.patch_embd.bias" in self.tensors: x += self.weight("v.patch_embd.bias")
        gx, gy = np.meshgrid(np.arange(columns), np.arange(rows), indexing="xy")
        positions = np.stack([gx, gy], axis=-1).reshape(-1, 2).astype(np.int32)
        table = self.weight("v.position_embd.weight")
        x += table[0, positions[:, 0]] + table[1, positions[:, 1]]
        head = self.hidden // self.heads
        cos, sin = self.hf.build_rope(positions, head, float(self.metadata("clip.vision.rope.freq_base", 100)))
        norm = self.hf.rms_norm
        for layer in range(self.layers):
            prefix = f"v.blk.{layer}"
            h = norm(x, self.weight(prefix + ".ln1.weight"), self.epsilon)
            q = norm(self.linear(h, prefix + ".attn_q").reshape(-1, self.heads, head), self.weight(prefix + ".attn_q_norm.weight"), self.epsilon)
            k = norm(self.linear(h, prefix + ".attn_k").reshape(-1, self.heads, head), self.weight(prefix + ".attn_k_norm.weight"), self.epsilon)
            v = norm(self.linear(h, prefix + ".attn_v").reshape(-1, self.heads, head), None, self.epsilon)
            q = self.hf.apply_axial_rope(q, cos, sin); k = self.hf.apply_axial_rope(k, cos, sin)
            attended = np.empty_like(v)
            for index in range(self.heads):
                scores = q[:, index] @ k[:, index].T  # HF vision scaling is exactly 1.
                scores -= scores.max(axis=-1, keepdims=True)
                np.exp(scores, out=scores); scores /= scores.sum(axis=-1, keepdims=True)
                attended[:, index] = scores @ v[:, index]
            x = x + norm(self.linear(attended.reshape(-1, self.hidden), prefix + ".attn_out"), self.weight(prefix + ".attn_post_norm.weight"), self.epsilon)
            h = norm(x, self.weight(prefix + ".ln2.weight"), self.epsilon)
            gate, up = self.linear(h, prefix + ".ffn_gate"), self.linear(h, prefix + ".ffn_up")
            x = x + norm(self.linear(self.hf.ACT["gelu_tanh"](gate) * up, prefix + ".ffn_down"), self.weight(prefix + ".ffn_post_norm.weight"), self.epsilon)
            if prefix + ".out_scale.weight" in self.tensors: x *= self.weight(prefix + ".out_scale.weight")
            if not np.isfinite(x).all(): raise ValueError(f"Nonfinite layer {layer}")
        weights = dict(std_bias=self.weight("v.std_bias") if "v.std_bias" in self.tensors else np.zeros(self.hidden, dtype=np.float32),
            std_scale=self.weight("v.std_scale") if "v.std_scale" in self.tensors else np.ones(self.hidden, dtype=np.float32),
            embedding_projection=self.weight("mm.input_projection.weight"))
        cfg = SimpleNamespace(pooling_kernel_size=self.merge, hidden_size=self.hidden, rms_norm_eps=self.epsilon,
            text_hidden_size=weights["embedding_projection"].shape[0])
        return self.hf.pool_and_project(weights, x, rows, columns, positions, cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--pixels", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int); parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", action="append", default=[], type=Path)
    parser.add_argument("--min-cosine", type=float, default=.9999)
    parser.add_argument("--max-absolute-error", type=float, default=.01)
    args = parser.parse_args()
    pixels = np.fromfile(args.pixels, dtype="<f4").reshape(3, args.height, args.width)
    if not np.isfinite(pixels).all() or np.min(pixels) < 0 or np.max(pixels) > 1:
        raise ValueError("Expected finite [0,1] channel-first pixels")
    encoder = VisionEncoder(args.model)
    with np.errstate(all="ignore"): expected = encoder.encode(pixels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected.astype("<f4").tofile(args.output.with_suffix(".f32"))
    comparisons = []
    for path in args.candidate:
        actual = np.fromfile(path, dtype="<f4").reshape(expected.shape)
        a, b = expected.astype(np.float64), actual.astype(np.float64)
        cos = np.sum(a*b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)
        error = float(np.max(np.abs(a-b)))
        comparisons.append(dict(candidate=str(path.resolve()), candidate_sha256=sha(path),
            min_row_cosine=float(cos.min()), max_absolute_error=error,
            relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-30)),
            passed=bool(np.isfinite(actual).all() and cos.min() >= args.min_cosine and error <= args.max_absolute_error)))
    report = dict(model=str(args.model.resolve()), model_sha256=sha(args.model), pixels=str(args.pixels.resolve()),
        pixels_sha256=sha(args.pixels), shape=list(expected.shape), layers=encoder.layers, heads=encoder.heads,
        hf_source="transformers Gemma4 vision math in eng/diffusiongemma-vision-oracle.py",
        shared_reference_sha256=sha(Path(__file__).resolve().parents[1]/"diffusiongemma-vision-oracle.py"),
        tool_sha256=sha(__file__), thresholds=dict(min_row_cosine=args.min_cosine, max_absolute_error=args.max_absolute_error),
        comparisons=comparisons, passed=all(item["passed"] for item in comparisons) if comparisons else None,
        limitations="Exact pixels and dequantized projector; no preprocessing, text pooling or performance claim.")
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(shape=report["shape"], comparisons=comparisons, passed=report["passed"])), flush=True)
    return 0 if report["passed"] is not False else 1


if __name__ == "__main__": raise SystemExit(main())
