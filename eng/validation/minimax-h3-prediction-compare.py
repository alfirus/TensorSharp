#!/usr/bin/env python3
"""Check a full50 H3 probe's returned velocities against saved independent F32 truth.

The saved oracle must use exactly the probe's projection inputs. This checks the
sigma-one denoiser endpoint; it does not establish multi-step perceptual quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def metrics(reference, actual):
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Prediction shape or finiteness mismatch")
    return {"elements": a.size,
            "relative_l2": float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-30)),
            "cosine": float(np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-30)),
            "max_abs": float(np.max(np.abs(a-b)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-report", required=True, type=Path)
    parser.add_argument("--oracle-report", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-relative-l2", type=float, default=.001)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite evidence")
    probe, oracle = [json.loads(path.read_text()) for path in (args.probe_report, args.oracle_report)]
    if not (probe["completed"] and oracle["completed"] and probe["blocks"] == oracle["actual_blocks"] == 50
            and probe["sigma"] == 1 and not oracle["tf32"]):
        raise ValueError("Require completed full50 sigma-one probe and IEEE F32 oracle")
    identities = lambda report: {Path(item["file"]).name: item["sha256"] for item in report["inputs"]}
    if identities(probe) != identities(oracle):
        raise ValueError("Oracle and probe inputs differ")
    candidates = Path(probe["trace"])
    video_path = candidates / "video-velocity.f32"
    audio_path = candidates / "carrier-audio-velocity.f32"
    digest = hashlib.sha256(video_path.read_bytes()+audio_path.read_bytes()).hexdigest()
    if digest != probe["velocitySha256"]:
        raise ValueError("Saved velocities differ from the probe report")
    snapshots = args.oracle_report.parent / (args.oracle_report.stem + "-snapshots")
    read = lambda path: np.fromfile(path, dtype="<f4")
    metadata = [json.loads(path.read_text()) for path in args.reference.glob("*.f32.json")]
    projection = [item for item in metadata if item["file"].startswith("projection-") and
                  any("audio_patch_proj.weight" in source["name"] for source in item["sources"])]
    if len(projection) != 1:
        raise ValueError("Require one exact physical audio input")
    identity = projection[0]["sources"][1]["id"]
    physical_path = args.reference / next(item["file"] for item in metadata if item["id"] == identity)
    if hashlib.sha256(physical_path.read_bytes()).hexdigest() != identities(probe).get(physical_path.name):
        raise ValueError("Physical audio input differs from the frozen probe")
    physical = read(physical_path).astype(np.float64)
    # Forward returns -rawVideo and -4*rawAudio at sigma1; the probe converts
    # audio to carrier velocity -3*physicalAudio -4*rawAudio.
    audio = -(read(audio_path).astype(np.float64) + 3*physical) / 4
    video = -read(video_path)
    boundaries = {name: metrics(read(snapshots / ("raw-"+name+"-output.f32")), value)
                  for name, value in [("audio", audio), ("video", video)]}
    result = {"scope": "Same-input full50 sigma-one raw predictions; audio recovered from returned carrier velocity",
              "probe_report": str(args.probe_report.resolve()), "oracle_report": str(args.oracle_report.resolve()),
              "inputs": identities(probe), "boundaries": boundaries,
              "maximum_relative_l2": args.maximum_relative_l2,
              "passed": all(row["relative_l2"] <= args.maximum_relative_l2 for row in boundaries.values()),
              "limitations": ["Saved oracle is reused; unchanged input hashes checked",
                              "This is not an oracle for the text encoder, multi-step sampler or perceptual quality"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
