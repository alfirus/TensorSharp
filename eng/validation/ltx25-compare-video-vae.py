#!/usr/bin/env python3
"""Fail-closed stage comparison of native LTX VAE encoding/decoding and original F32."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_verified(directory, entry):
    path = directory / entry["file"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError("Stage hash mismatch: " + str(path))
    values = np.frombuffer(data, dtype="<f4")
    if values.size != int(np.prod(entry["shape"])) or not np.isfinite(values).all():
        raise ValueError("Invalid stage dimensions or non-finite values: " + str(path))
    return values.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-absolute", type=float, default=0.002)
    parser.add_argument("--relative-l2", type=float, default=0.0001)
    parser.add_argument("--max-normalized-absolute", type=float,
                        help="Optional scale-aware stage gate: max absolute error / max absolute reference value; replaces the unscaled absolute gate")
    parser.add_argument("--stages", nargs="+", help="Explicit stage subset; omitted requires every original stage")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to replace an existing quality report")
    if not 0 < args.max_absolute < 1 or not 0 < args.relative_l2 < 1:
        raise SystemExit("Invalid error bounds")
    if args.max_normalized_absolute is not None and not 0 < args.max_normalized_absolute < 1:
        raise SystemExit("Invalid normalized absolute error bound")
    probe = json.loads((args.probe / "probe.json").read_text())
    original = json.loads((args.reference / "reference.json").read_text())
    if not probe.get("completed") or not original.get("completed"):
        raise SystemExit("Both component runs must have completed")
    if original["source_revision"] != "be6d6d0a69d11a0a399c462ed92606692e5f8697" or original["dtype"] != "float32":
        raise SystemExit("A pinned original Float32 VAE reference is required")
    checkpoint_hash = probe.get("model_sha256", probe.get("checkpoint_sha256"))
    if checkpoint_hash != original["checkpoint_sha256"] or probe["input_sha256"] != original["input_sha256"]:
        raise SystemExit("Original and native VAE must use identical checkpoint and input bytes")
    if probe["input_shape"] != original["input_shape"]:
        raise SystemExit("Input dimensions differ")
    terminal = "encoded" if "encoded" in original["stages"] else "decoded"
    if args.stages == [terminal] and not probe["stages"]:
        probe["stages"][terminal] = dict(shape=probe["output_shape"], file="output.f32", sha256=probe["output_sha256"])
    selected = set(args.stages) if args.stages else set(original["stages"])
    if not selected or not selected.issubset(original["stages"]) or not selected.issubset(probe["stages"]):
        raise SystemExit("Requested VAE stages are missing")
    if not args.stages and set(probe["stages"]) != set(original["stages"]):
        raise SystemExit("VAE stage sets differ")
    results = {}
    for name, expected_stage in original["stages"].items():
        if name not in selected:
            continue
        actual_stage = probe["stages"][name]
        if actual_stage["shape"] != expected_stage["shape"]:
            raise SystemExit("Stage shapes differ: " + name)
        expected = read_verified(args.reference, expected_stage)
        actual = read_verified(args.probe, actual_stage)
        error = actual - expected
        maximum = float(np.abs(error).max(initial=0))
        reference_maximum = float(np.abs(expected).max(initial=0))
        normalized_maximum = maximum / reference_maximum if reference_maximum else (0.0 if maximum == 0 else float("inf"))
        norm = float(np.linalg.norm(expected))
        relative = float(np.linalg.norm(error) / norm) if norm else (0.0 if not np.any(error) else float("inf"))
        results[name] = dict(shape=expected_stage["shape"], max_absolute_error=maximum,
                             max_absolute_reference=reference_maximum, max_normalized_absolute_error=normalized_maximum,
                             relative_l2_error=relative, rms_error=float(np.sqrt(np.mean(error * error))),
                             passed=(normalized_maximum <= args.max_normalized_absolute if args.max_normalized_absolute is not None
                                     else maximum <= args.max_absolute) and relative <= args.relative_l2)
    report = dict(completed=True, passed=all(value["passed"] for value in results.values()),
                  scope="Convolutional video VAE numerical stage comparison only; Float32 compute with identical checkpoint and input bytes. No perceptual, complete generation or performance proof.",
                  source_revision=original["source_revision"], checkpoint_sha256=checkpoint_hash,
                  input_sha256=probe["input_sha256"], input_shape=probe["input_shape"], backend=probe["backend"],
                  compared_stage_names=sorted(selected),
                  reference_stages_not_compared=sorted(set(original["stages"]) - selected),
                  acceptance=dict(max_absolute_error=args.max_absolute, relative_l2_error=args.relative_l2,
                                  max_normalized_absolute_error=args.max_normalized_absolute,
                                  rule="Every finite stage must pass relative L2 and the explicitly selected absolute bound (normalized by its global reference maximum when requested)"),
                  probe_path=str(args.probe / "probe.json"), reference_path=str(args.reference / "reference.json"),
                  probe_report_sha256=hashlib.sha256((args.probe / "probe.json").read_bytes()).hexdigest(),
                  reference_report_sha256=hashlib.sha256((args.reference / "reference.json").read_bytes()).hexdigest(),
                  stages=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], stages=len(results), **{terminal: results.get(terminal)})))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
