#!/usr/bin/env python3
"""Independent float32 NumPy EmbeddingGemma 2 GGUF encoder oracle.

Uses gguf-py only for reading and dequantizing weights. Runs each complete
sequence separately with dense bidirectional attention, symmetric sliding
windows, per-layer inputs, and the learned output projection. This checks
architecture/numerics, not MTEB quality or runtime performance.

PYTHONPATH=/path/to/llama.cpp/gguf-py python eng/embeddinggemma2-reference.py \
  --model /path/to/embeddinggemma-2-BF16.gguf --tokens oracle.json \
  --output docs/validation/embeddinggemma2/numpy.json

Token JSON accepts {"cases": [{"text": "...", "tokens": [2, ..., 1]}]}.
Requires NumPy and gguf-py dependencies. The tanh GELU matches the published
model configuration; all arithmetic and activations use float32.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np


class Encoder:
    def __init__(self, path):
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.reader = GGUFReader(str(path), mode="r")
        self.dequantize = dequantize
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.cache = {}
        self.prefix = "gemma-embedding2."
        assert self.metadata("general.architecture") == "gemma-embedding2"
        self.dim = int(self.metadata(self.prefix + "embedding_length"))
        self.layers = int(self.metadata(self.prefix + "block_count"))
        self.eps = np.float32(self.metadata(self.prefix + "attention.layer_norm_rms_epsilon"))

    def metadata(self, key):
        return self.reader.fields[key].contents()

    def matrix(self, name):
        if name not in self.cache:
            t = self.tensors[name]
            rows = int(np.prod(t.shape[1:])) or 1
            self.cache[name] = np.asarray(self.dequantize(t.data.reshape(rows, -1), t.tensor_type), dtype=np.float32)
        return self.cache[name]

    def linear(self, x, name):
        return x @ self.matrix(name + ".weight").T

    def norm(self, x, name=None):
        out = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32) + self.eps)
        return out if name is None else out * self.matrix(name + ".weight").reshape(-1)

    def token_rows(self, tokens):
        """Exact vocabulary rows with Gemma's sqrt(hidden) token scaling."""
        t = self.tensors["token_embd.weight"]
        raw = np.ascontiguousarray(t.data.reshape(int(t.shape[1]), -1)[tokens])
        return np.asarray(self.dequantize(raw, t.tensor_type), dtype=np.float32) * np.float32(math.sqrt(self.dim))

    def encode(self, tokens):
        return self.encode_inputs(self.token_rows(tokens))

    def encode_inputs(self, inputs):
        """Encode full F32 input rows: scaled vocabulary rows and unscaled soft media rows."""
        x = np.asarray(inputs, dtype=np.float32).copy()
        if x.ndim != 2 or x.shape[1] != self.dim or not 1 <= x.shape[0] <= 8192 or not np.all(np.isfinite(x)):
            raise ValueError("Require finite [1..8192, hidden] input rows")
        tokens = x.shape[0]
        ple_dim = int(self.metadata(self.prefix + "embedding_length_per_layer_input"))
        ple = self.linear(x, "per_layer_model_proj") / np.float32(math.sqrt(self.dim))
        ple = self.norm(ple.reshape(tokens, self.layers, ple_dim), "per_layer_proj_norm")
        heads = int(self.metadata(self.prefix + "attention.head_count"))
        kv_heads = self.metadata(self.prefix + "attention.head_count_kv")
        pattern = self.metadata(self.prefix + "attention.sliding_window_pattern")
        window = int(self.metadata(self.prefix + "attention.sliding_window")) // 2
        for layer in range(self.layers):
            p = f"blk.{layer}."
            local = bool(pattern[layer])
            head_dim = int(self.metadata(self.prefix + "attention.key_length" + ("_swa" if local else "")))
            kv = int(kv_heads[layer] if isinstance(kv_heads, list) else kv_heads)
            cur = self.norm(x, p + "attn_norm")
            q = self.norm(self.linear(cur, p + "attn_q").reshape(tokens, heads, head_dim), p + "attn_q_norm")
            k = self.norm(self.linear(cur, p + "attn_k").reshape(tokens, kv, head_dim), p + "attn_k_norm")
            v = self.norm(self.linear(cur, p + "attn_v").reshape(tokens, kv, head_dim))
            base = float(self.metadata(self.prefix + "rope.freq_base" + ("_swa" if local else "")))
            q = rope(q, base)
            k = rope(k, base)
            k = np.repeat(k, heads // kv, axis=1)
            v = np.repeat(v, heads // kv, axis=1)
            scores = np.einsum("qhd,khd->hqk", q, k)
            if local:
                positions = np.arange(tokens)
                scores[:, np.abs(positions[:, None] - positions[None, :]) > window] = -np.inf
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            attention = np.einsum("hqk,khd->qhd", probabilities, v).reshape(tokens, heads * head_dim)
            x = x + self.norm(self.linear(attention, p + "attn_output"), p + "post_attention_norm")
            cur = self.norm(x, p + "ffn_norm")
            cur = gelu(self.linear(cur, p + "ffn_gate")) * self.linear(cur, p + "ffn_up")
            x = x + self.norm(self.linear(cur, p + "ffn_down"), p + "post_ffw_norm")
            cur = gelu(self.linear(x, p + "inp_gate")) * ple[:, layer, :]
            x = x + self.norm(self.linear(cur, p + "proj"), p + "post_norm")
            x *= self.matrix(p + "layer_output_scale.weight").reshape(-1)
        rows = self.linear(self.norm(x, "output_norm"), "output")
        pooled = rows.mean(axis=0, dtype=np.float32)
        return pooled / np.float32(np.linalg.norm(pooled.astype(np.float64)))


def gelu(x):
    return np.float32(.5) * x * (np.float32(1) + np.tanh(np.float32(math.sqrt(2 / math.pi)) *
        (x + np.float32(.044715) * x * x * x)))


def rope(x, base):
    half = x.shape[-1] // 2
    angles = np.arange(x.shape[0], dtype=np.float32)[:, None] * np.power(
        np.float32(base), -np.arange(half, dtype=np.float32)[None, :] * np.float32(2 / x.shape[-1]))
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return np.concatenate((a * cos - b * sin, b * cos + a * sin), axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--indices", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.tokens.read_text())
    cases = source["cases"]
    selected = [int(i) for i in args.indices.split(",") if i] or list(range(len(cases)))
    model = Encoder(args.model)
    results = []
    for index in selected:
        case = cases[index]
        start = time.perf_counter()
        vector = model.encode(case["tokens"])
        assert np.isfinite(vector).all() and abs(float(np.linalg.norm(vector)) - 1) < 1e-5
        results.append(dict(index=index, text=case.get("text"), tokens=case["tokens"], embedding=vector.tolist(), seconds=time.perf_counter()-start))
        print(index, len(case["tokens"]), "tokens", round(results[-1]["seconds"], 3), "s", flush=True)
    digest = hashlib.sha256()
    with args.model.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(model=str(args.model), sha256=digest.hexdigest(), numpy=np.__version__,
        platform=platform.platform(), method="Independent NumPy F32 dense attention, tanh GELU, complete sequences",
        coverage="Numerical architecture oracle only; no MTEB, device or performance claim", cases=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
