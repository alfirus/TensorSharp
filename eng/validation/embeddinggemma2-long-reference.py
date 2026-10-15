#!/usr/bin/env python3
"""Independent F32 NumPy encoder oracle using bounded query attention blocks.

Reuses the GGUF reader and arithmetic helpers from embeddinggemma2-reference.py
without editing it. Each query still attends to every allowed key in its full
sequence; query blocking only bounds temporary scores, and local layers avoid
computing keys outside the symmetric window.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

spec = importlib.util.spec_from_file_location("text_reference", Path(__file__).parents[1] / "embeddinggemma2-reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class BlockedEncoder(reference.Encoder):
    def attention(self, q, k, v, window, block):
        length, heads, dim = q.shape
        output = np.empty_like(q)
        for first in range(0, length, block):
            last = min(length, first + block)
            begin = 0 if window is None else max(0, first - window)
            end = length if window is None else min(length, last + window)
            for head in range(heads):
                scores = q[first:last, head] @ k[begin:end, head].T
                if window is not None:
                    distance = np.arange(first, last)[:, None] - np.arange(begin, end)[None, :]
                    scores[np.abs(distance) > window] = -np.inf
                probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
                probabilities /= probabilities.sum(axis=-1, keepdims=True)
                output[first:last, head] = probabilities @ v[begin:end, head]
        return output.reshape(length, heads * dim)

    def encode_blocked(self, tokens, block=256):
        length = len(tokens)
        tensor = self.tensors["token_embd.weight"]
        raw = np.ascontiguousarray(tensor.data.reshape(int(tensor.shape[1]), -1)[tokens])
        x = np.asarray(self.dequantize(raw, tensor.tensor_type), dtype=np.float32) * np.float32(math.sqrt(self.dim))
        ple_dim = int(self.metadata(self.prefix + "embedding_length_per_layer_input"))
        ple = self.linear(x, "per_layer_model_proj") / np.float32(math.sqrt(self.dim))
        ple = self.norm(ple.reshape(length, self.layers, ple_dim), "per_layer_proj_norm")
        heads = int(self.metadata(self.prefix + "attention.head_count"))
        kv_heads = self.metadata(self.prefix + "attention.head_count_kv")
        pattern = self.metadata(self.prefix + "attention.sliding_window_pattern")
        window = int(self.metadata(self.prefix + "attention.sliding_window")) // 2
        for layer in range(self.layers):
            prefix = f"blk.{layer}."
            local = bool(pattern[layer])
            head = int(self.metadata(self.prefix + "attention.key_length" + ("_swa" if local else "")))
            kv = int(kv_heads[layer] if isinstance(kv_heads, list) else kv_heads)
            normalized = self.norm(x, prefix + "attn_norm")
            q = self.norm(self.linear(normalized, prefix + "attn_q").reshape(length, heads, head), prefix + "attn_q_norm")
            k = self.norm(self.linear(normalized, prefix + "attn_k").reshape(length, kv, head), prefix + "attn_k_norm")
            v = self.norm(self.linear(normalized, prefix + "attn_v").reshape(length, kv, head))
            base = float(self.metadata(self.prefix + "rope.freq_base" + ("_swa" if local else "")))
            q, k = reference.rope(q, base), reference.rope(k, base)
            k, v = np.repeat(k, heads // kv, axis=1), np.repeat(v, heads // kv, axis=1)
            attended = self.attention(q, k, v, window if local else None, block)
            x = x + self.norm(self.linear(attended, prefix + "attn_output"), prefix + "post_attention_norm")
            normalized = self.norm(x, prefix + "ffn_norm")
            ffn = reference.gelu(self.linear(normalized, prefix + "ffn_gate")) * self.linear(normalized, prefix + "ffn_up")
            x = x + self.norm(self.linear(ffn, prefix + "ffn_down"), prefix + "post_ffw_norm")
            layer_input = reference.gelu(self.linear(x, prefix + "inp_gate")) * ple[:, layer, :]
            x = x + self.norm(self.linear(layer_input, prefix + "proj"), prefix + "post_norm")
            x *= self.matrix(prefix + "layer_output_scale.weight").reshape(-1)
            print(f"{length} tokens, layer {layer + 1}/{self.layers}", flush=True)
        rows = self.linear(self.norm(x, "output_norm"), "output")
        pooled = rows.mean(axis=0, dtype=np.float32)
        return pooled / np.float32(np.linalg.norm(pooled.astype(np.float64)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--dense-reference", type=Path, help="Optional precomputed dense oracle for validating blocking")
    args = parser.parse_args()
    if args.block <= 0:
        parser.error("block must be positive")
    inputs = json.loads(args.tokens.read_text())["cases"]
    model = BlockedEncoder(args.model)
    results = []
    for index, case in enumerate(inputs):
        started = time.perf_counter()
        output = model.encode_blocked(case["tokens"], args.block)
        if not np.isfinite(output).all() or abs(float(np.linalg.norm(output)) - 1) > 1e-5:
            raise ValueError("Oracle returned invalid normalized output")
        results.append(dict(index=index, text=case.get("text"), tokens=case["tokens"], embedding=output.tolist(), seconds=time.perf_counter() - started))
    digest = hashlib.sha256()
    with args.model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    report = dict(model=str(args.model), sha256=digest.hexdigest(), numpy=np.__version__, platform=platform.platform(),
                  method="Independent NumPy F32 complete-sequence attention with bounded query blocks", query_block=args.block,
                  coverage="Numerical architecture oracle only; no device or performance claim", cases=results)
    if args.dense_reference:
        dense = json.loads(args.dense_reference.read_text())
        if dense["sha256"] != report["sha256"] or len(dense["cases"]) != len(results):
            raise ValueError("Dense reference model/case count differs")
        comparisons = []
        for left, right in zip(dense["cases"], results):
            if left["tokens"] != right["tokens"]:
                raise ValueError("Dense and blocked oracle inputs differ")
            a, b = np.array(left["embedding"], dtype=np.float64), np.array(right["embedding"], dtype=np.float64)
            cosine = float(a @ b / np.sqrt((a @ a) * (b @ b)))
            error = float(np.max(np.abs(a - b)))
            comparisons.append(dict(index=right["index"], cosine=cosine, maximum_absolute_error=error, passed=cosine > .9999999 and error < 1e-4))
        report["dense_validation"] = comparisons
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all(row["passed"] for row in report.get("dense_validation", [])):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
