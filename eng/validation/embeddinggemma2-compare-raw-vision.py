#!/usr/bin/env python3
"""Compare unprefixed VisionProbe rows with a saved independent vision oracle.

Unlike compare-vision-embeddings.py, these files have no length prefix. The
oracle's recorded shape and thresholds are authoritative; do not loosen them.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VISION_OVERRIDE_FIELDS = (
    "vision_device", "vision_cuda_device", "vision_cpu_q16", "vision_metal_precise", "vision_metal_mps_mm",
    "vision_metal_split_mm", "vision_metal_mixed_mps", "vision_spatial_device",
    "vision_split_projection", "vision_simd_projection", "vision_split_attention",
    "embedding_linux_blas", "embedding_blas_library", "embedding_cpu_attention", "embedding_cpu_profile",
    "embedding_cpu_attention_profile", "embedding_cpu_attention_simd", "vision_stage", "vision_trace", "vision_profile", "cupti_activity_output",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(reference_path, candidate_path, require_default, require_embedding_default=False):
    reference = json.loads(reference_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    shape = tuple(reference["shape"])
    if shape != (candidate["rows"], candidate["dimensions"]) or len(shape) != 2:
        raise ValueError("Raw vision shapes differ")
    if not candidate["completed"]:
        raise ValueError("Candidate did not complete")
    for reference_key, candidate_key in (("model_sha256", "projector_sha256"), ("pixels_sha256", "pixels_sha256")):
        if reference[reference_key] != candidate[candidate_key]:
            raise ValueError("Candidate model/pixels differ from the saved oracle")
    for path_key, hash_key in (("projector", "projector_sha256"), ("pixels", "pixels_sha256"), ("output", "output_sha256")):
        if sha(candidate[path_key]) != candidate[hash_key]:
            raise ValueError("Candidate input/output no longer matches its report")
    for artifact in candidate["binary_artifacts"]:
        if sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Candidate binary no longer matches its report")
    if require_default and (candidate["backend"] != "MLX" or not candidate.get("require_mlx_default") or any(
            candidate.get(name) is not None for name in VISION_OVERRIDE_FIELDS)):
        raise ValueError("Default proof requires MLX with all vision override variables absent")
    guard_version = candidate.get("embedding_default_guard_version", 0)
    if require_embedding_default and (not candidate.get("require_embedding_default") or
            not isinstance(guard_version, int) or guard_version < 4 or "cupti_activity_output" not in candidate or any(
            candidate.get(name) is not None for name in VISION_OVERRIDE_FIELDS)):
        raise ValueError("Embedding default proof requires a current guard and all vision/CPU override variables absent")
    if require_embedding_default and candidate["backend"] == "GGML_CUDA" and candidate.get("require_validated_float32_precision") is not True:
        raise ValueError("CUDA embedding default proof requires the validated embedding constructor")
    reference_rows = reference_path.with_suffix(".f32")
    expected = np.fromfile(reference_rows, dtype="<f4").reshape(shape).astype(np.float64)
    actual = np.fromfile(candidate["output"], dtype="<f4").reshape(shape).astype(np.float64)
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Nonfinite vision rows")
    cosine = np.sum(expected * actual, axis=1) / np.maximum(
        np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1), 1e-30)
    error = float(np.max(np.abs(expected - actual)))
    thresholds = reference["thresholds"]
    return dict(reference=str(reference_path.resolve()), reference_sha256=sha(reference_path),
                reference_rows_sha256=sha(reference_rows), candidate=str(candidate_path.resolve()),
                candidate_sha256=sha(candidate_path), shape=list(shape),
                model_sha256=reference["model_sha256"], pixels_sha256=reference["pixels_sha256"],
                min_row_cosine=float(cosine.min()), max_absolute_error=error,
                relative_l2=float(np.linalg.norm(expected-actual) / max(np.linalg.norm(expected), 1e-30)),
                thresholds=thresholds, default_route_required=require_default,
                embedding_default_route_required=require_embedding_default,
                passed=bool(cosine.min() >= thresholds["min_row_cosine"] and error <= thresholds["max_absolute_error"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--require-mlx-default", action="store_true")
    parser.add_argument("--require-embedding-default", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.reference) != len(args.candidate):
        raise ValueError("Pass equally many reference and candidate reports in matching order")
    comparisons = [compare(reference, candidate, args.require_mlx_default, args.require_embedding_default)
                   for reference, candidate in zip(args.reference, args.candidate)]
    report = dict(comparisons=comparisons, executed=len(comparisons),
                  passed=sum(item["passed"] for item in comparisons), skipped=0,
                  tool_sha256=sha(__file__),
                  limitations="Saved independent full-tower oracle and exact pixels; no preprocessing, text pooling, protocol, or performance claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] == report["executed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
