#!/usr/bin/env python3
"""Compare EmbeddingGemma2Probe vectors with the independent NumPy oracle.

The probe case names must be oracle_<index>, and its JSON case manifest must
contain exactly the oracle's texts or token IDs. Both reports must describe the same model.
Writes a complete comparison on success or numerical failure; never forwards.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path,
                        help="Local copy of a remote probe's manifest; its recorded SHA256 must still match")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=.9999)
    parser.add_argument("--max-absolute-error", type=float, default=.002)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    if not candidate["completed"] or reference["sha256"] != candidate["model_sha256"]:
        raise ValueError("Require completed probe output and the same model SHA256 as the oracle")
    manifest_path = args.case_manifest or Path(candidate["case_manifest"])
    if sha256(manifest_path) != candidate["case_manifest_sha256"]:
        raise ValueError("Probe case manifest has changed since inference")
    manifest = json.loads(manifest_path.read_text())
    manifest_cases = {item["name"]: item for item in manifest["cases"]}
    candidate_cases = {item["name"]: item for item in candidate["cases"]}
    rows = []
    for item in reference["cases"]:
        name = "oracle_" + str(item["index"])
        actual = candidate_cases[name]
        inputs = manifest_cases[name]
        if "tokens" in inputs:
            if inputs["tokens"] != item["tokens"] or any(part["kind"] != "text" for part in inputs.get("parts", [])):
                raise ValueError("Probe token input differs from the oracle: " + name)
        else:
            parts = inputs["parts"]
            if any(part["kind"] != "text" for part in parts) or "".join(part["value"] for part in parts) != item["text"]:
                raise ValueError("Probe input text differs from the oracle: " + name)
        left, right = item["embedding"], actual["embedding"]
        if len(left) != 768 or len(right) != 768 or any(not math.isfinite(value) for value in left + right):
            raise ValueError("Require finite 768-dimensional vectors")
        if actual["prompt_tokens"] != len(item["tokens"]):
            raise ValueError("Token accounting differs: " + name)
        cosine = sum(a*b for a, b in zip(left, right)) / math.sqrt(sum(a*a for a in left)*sum(b*b for b in right))
        error = max(abs(a-b) for a, b in zip(left, right))
        rows.append(dict(name=name, tokens=actual["prompt_tokens"], cosine=cosine, max_absolute_error=error,
                         passed=cosine >= args.min_cosine and error <= args.max_absolute_error))
    report = dict(reference=str(args.reference.resolve()), reference_sha256=sha256(args.reference),
                  candidate=str(args.candidate.resolve()), candidate_sha256=sha256(args.candidate),
                  case_manifest=str(manifest_path.resolve()), case_manifest_sha256=sha256(manifest_path),
                  captured_case_manifest=candidate["case_manifest"],
                  backend=candidate["backend"], model_sha256=candidate["model_sha256"], cases=rows,
                  min_cosine=min(row["cosine"] for row in rows),
                  max_absolute_error=max(row["max_absolute_error"] for row in rows),
                  thresholds=dict(min_cosine=args.min_cosine, max_absolute_error=args.max_absolute_error),
                  passed=all(row["passed"] for row in rows),
                  limitations="Numerical parity for the recorded exact text/token inputs; no retrieval leaderboard or performance claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "min_cosine", "max_absolute_error", "backend")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
