#!/usr/bin/env python3
"""Bounded eight-case phase diagnostic of an already qualified frozen CPU graph.

No build, long sweep or performance claim. Every runtime artifact in the prior
quality provenance must still match. Native fallback cannot establish a pass.
"""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-quality", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quality = module("owned_quality", args.root / "eng/validation/embeddinggemma2-owned-attention-quality.py")
    phase = module("phase", args.root / "eng/validation/embeddinggemma2-profile.py")
    prior = json.loads(args.prior_quality.read_text())
    if not prior["completed"]:
        raise ValueError("Frozen graph must already have completed strict model qualification")
    for item in prior["binary_artifacts"]:
        if quality.sha(item["path"]) != item["sha256"]:
            raise ValueError("A frozen runtime artifact changed")
    if quality.sha(args.library) != prior["library"]["sha256"]:
        raise ValueError("Frozen native library changed")
    if len(json.loads(args.cases.read_text())["cases"]) != 8:
        raise ValueError("Run exactly the eight qualified inputs")
    args.output.mkdir(parents=True, exist_ok=True)
    flags = dict(prior["flags"], TENSORSHARP_EMBEDDING_CPU_PROFILE="1")
    env = dict(os.environ, **flags)
    env.pop("TS_CPU_DISABLE_AVX512", None)
    candidate, log, comparison = [args.output / name for name in ("eight.json", "eight.log", "eight-comparison.json")]
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), flags=flags,
                  frozen_quality_sha256=quality.sha(args.prior_quality), frozen_binaries=prior["binary_artifacts"],
                  library=prior["library"], completed=False,
                  limitations="Bounded buffered diagnostic includes warmup and profiling overhead; no formal timing or whole HTTP decomposition claim. No binary/model source changes. Derived parent timing totals overlap leaf phases.")
    try:
        argv = ["dotnet", str(args.probe), "--model", str(args.model), "--cases", str(args.cases), "--output", str(candidate),
                "--backend", "CPU", "--threads", "8", "--warmup", "1", "--repeats", "1"]
        report["command"] = argv
        with log.open("w") as stream:
            report["exit_code"] = subprocess.run(argv, env=env, cwd=args.root, stdout=stream, stderr=subprocess.STDOUT).returncode
        if report["exit_code"]:
            return 1
        report["actual_math"] = quality.native_evidence(log, args.library, require_q16=True)
        argv = [sys.executable, str(args.root / "eng/validation/embeddinggemma2-compare.py"), "--reference", str(args.reference),
                "--candidate", str(candidate), "--case-manifest", str(args.cases), "--output", str(comparison),
                "--min-cosine", ".99999", "--max-absolute-error", ".0005"]
        with (args.output / "eight-comparison.log").open("w") as stream:
            report["comparison_exit_code"] = subprocess.run(argv, cwd=args.root, stdout=stream, stderr=subprocess.STDOUT).returncode
        numerical = json.loads(comparison.read_text())
        report.update(executed=len(numerical["cases"]), passed=sum(item["passed"] for item in numerical["cases"]), skipped=0,
                      min_cosine=numerical["min_cosine"], max_absolute_error=numerical["max_absolute_error"])
        if report["comparison_exit_code"] or report["executed"] != 8 or report["passed"] != 8:
            return 1
        summary = phase.summarize(log)
        (args.output / "phase-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        report["completed"] = True
        print(json.dumps(dict(completed=True, compute_ms=summary["compute_elapsed_ms"],
                             projection_percent=summary["projection_compute_percent"],
                             phases={name:round(value["compute_percent"], 2) for name,value in summary["phases"].items()},
                             projections={name:round(value["compute_percent"], 2) for name,value in summary["groups"].items()}), indent=2))
        return 0
    finally:
        report["report_hashes"] = {path.name: quality.sha(path) for path in args.output.iterdir() if path.is_file()}
        (args.output / "phase-provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
