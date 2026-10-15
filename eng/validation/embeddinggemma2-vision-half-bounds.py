#!/usr/bin/env python3
"""Explore bounded normalized vision projections without changing the F32 oracle.

Only Q/K/V following ln1 and gate/up following ln2 are candidates. This tool
proves a conservative bound on every partial dot using the original RMS gain
and rounded weight, then emulates F16 input/weight/output with F32 accumulation.
This is a precision experiment, not a device test or a new quality reference.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "pixels", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--projections", choices=("all", "qkv", "gate_up"), default="all")
    parser.add_argument("--result-type", choices=("float16", "float32"), default="float16")
    parser.add_argument("--operand-type", choices=("float16", "single_tf32", "split_tf32"), default="float16")
    args = parser.parse_args()
    reference_path = Path(__file__).with_name("embeddinggemma2-vision-reference.py")
    spec = importlib.util.spec_from_file_location("vision_f32_reference", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    def round_tf32(value):
        bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
        rounded = ((bits + np.uint32(0xfff) + ((bits >> 13) & np.uint32(1))) & np.uint32(0xffffe000)).view(np.float32)
        return np.where((bits & np.uint32(0x7fffffff)) >= np.uint32(0x7f7ff000), value, rounded).astype(np.float32)

    class Candidate(reference.VisionEncoder):
        def __init__(self, model):
            super().__init__(model)
            self.bounds = {}
            for block in range(self.layers):
                prefix = f"v.blk.{block}"
                for norm, suffixes in (("ln1", ("attn_q", "attn_k", "attn_v")),
                                       ("ln2", ("ffn_gate", "ffn_up"))):
                    if (args.projections == "qkv" and norm == "ln2" or
                            args.projections == "gate_up" and norm == "ln1"):
                        continue
                    gains = self.weight(prefix + "." + norm + ".weight").astype(np.float64)
                    for suffix in suffixes:
                        name = prefix + "." + suffix
                        weights = self.weight(name + ".weight")
                        rounded = weights.astype(np.float16).astype(np.float64)
                        size = len(gains)
                        # Weighted RMS rows have L2 <= sqrt(size); factor two
                        # covers input narrowing and norm/reduction rounding.
                        dots = 2 * np.sqrt(size * np.sum((rounded * gains) ** 2, axis=1))
                        dots += np.ldexp(np.sum(np.abs(rounded), axis=1), -24)
                        component = 2 * np.sqrt(size) * np.max(np.abs(gains))
                        clamp_contains_zero = True
                        for boundary, valid in (("input_min", lambda value: value <= 0),
                                                ("input_max", lambda value: value >= 0)):
                            key = name + "." + boundary
                            if key in self.tensors:
                                clamp_contains_zero &= valid(float(self.weight(key)[0]))
                        safe = bool(np.isfinite(gains).all() and np.isfinite(weights).all()
                                    and np.isfinite(rounded).all() and component <= 16384
                                    and np.max(np.abs(weights)) <= 16384
                                    and dots.max() <= 16384 and clamp_contains_zero)
                        self.bounds[name] = dict(safe=safe, input_component_bound=float(component),
                            partial_dot_bound=float(dots.max()), input=int(size),
                            output=int(weights.shape[0]), clamp_contains_zero=bool(clamp_contains_zero))

        def linear(self, x, name):
            if not self.bounds.get(name, {}).get("safe", False):
                return super().linear(x, name)
            for suffix in ("input_min", "input_max"):
                if name + "." + suffix in self.tensors:
                    limit = float(self.weight(name + "." + suffix)[0])
                    x = np.maximum(x, limit) if suffix == "input_min" else np.minimum(x, limit)
            weights = self.weight(name + ".weight")
            if args.operand_type == "split_tf32":
                high_x, high_w = round_tf32(x), round_tf32(weights)
                low_x, low_w = round_tf32(x - high_x), round_tf32(weights - high_w)
                result = (high_x @ high_w.T) + ((high_x @ low_w.T) + (low_x @ high_w.T))
            elif args.operand_type == "single_tf32":
                result = round_tf32(x) @ round_tf32(weights).T
            else:
                result = x.astype(np.float16).astype(np.float32) @ weights.astype(np.float16).astype(np.float32).T
            if args.result_type == "float16": result = result.astype(np.float16).astype(np.float32)
            if name + ".bias" in self.tensors: result += self.weight(name + ".bias")
            for suffix in ("output_min", "output_max"):
                if name + "." + suffix in self.tensors:
                    limit = float(self.weight(name + "." + suffix)[0])
                    result = np.maximum(result, limit) if suffix == "output_min" else np.minimum(result, limit)
            return result

    encoder = Candidate(args.model)
    pixels = np.fromfile(args.pixels, dtype="<f4").reshape(3, args.height, args.width)
    if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("Expected exact finite channel-first [0,1] pixels")
    # Accelerate-backed NumPy may leave stale floating-point exception flags
    # after GEMM; inspect each layer and the returned values explicitly.
    with np.errstate(all="ignore"): actual = encoder.encode(pixels)
    expected = np.fromfile(args.reference, dtype="<f4").reshape(actual.shape)
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    error = float(np.max(np.abs(a - b)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    actual.astype("<f4").tofile(args.output.with_suffix(".f32"))
    report = dict(completed=True, model=str(args.model.resolve()), model_sha256=reference.sha(args.model),
        pixels=str(args.pixels.resolve()), pixels_sha256=reference.sha(args.pixels),
        reference=str(args.reference.resolve()), reference_sha256=reference.sha(args.reference),
        tool_sha256=reference.sha(__file__), source_reference_sha256=reference.sha(reference_path),
        width=args.width, height=args.height, shape=list(actual.shape), bounds=encoder.bounds,
        numpy_version=np.__version__, projections=args.projections, result_type=args.result_type, operand_type=args.operand_type,
        qualified_projections=sum(value["safe"] for value in encoder.bounds.values()),
        min_row_cosine=float(cosine.min()), max_absolute_error=error,
        relative_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
        thresholds=dict(min_row_cosine=.9999, max_absolute_error=.01),
        passed=bool(np.isfinite(actual).all() and cosine.min() >= .9999 and error <= .01),
        limitations="NumPy operand narrowing/split with F32 accumulation emulation; no device or performance claim.")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("shape", "qualified_projections", "min_row_cosine",
                                                 "max_absolute_error", "relative_l2", "passed")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
