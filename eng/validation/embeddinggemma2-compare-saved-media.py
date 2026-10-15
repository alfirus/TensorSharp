#!/usr/bin/env python3
"""Numerically compare a completed media probe with previously captured llama vectors.

Requires the original comparison and HTTP fixture-provenance report. Checks the
model, manifest, captured candidate, and media bytes; saved timings are excluded.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--media-provenance", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=.999)
    args = parser.parse_args()
    reference, provenance, candidate = [json.loads(path.read_text()) for path in
                                       [args.reference, args.media_provenance, args.candidate]]
    original_probe = Path(reference["candidate_probe"])
    if digest(original_probe) != reference["candidate_probe_sha256"]:
        raise ValueError("Original reference candidate changed")
    original = json.loads(original_probe.read_text())
    if not all(report["completed"] for report in [reference, original, candidate, provenance]):
        raise ValueError("All source processes must have completed")
    if original["model_sha256"] != candidate["model_sha256"] or original["projector_sha256"] != candidate["projector_sha256"]:
        raise ValueError("Candidate model/projector differs from the reference")
    if Path(reference["models"]["llama"]).resolve() != Path(original["model"]).resolve():
        raise ValueError("Reference model path differs from its captured model provenance")
    manifest_sha = reference["manifest_sha256"]
    if candidate["case_manifest_sha256"] != manifest_sha or provenance["manifest"]["sha256"] != manifest_sha:
        raise ValueError("Media manifest differs")
    if digest(Path(candidate["case_manifest"])) != manifest_sha:
        raise ValueError("Media manifest changed after inference")
    for media in provenance["media"]:
        if digest(Path(media["path"])) != media["sha256"]:
            raise ValueError("Media bytes changed: " + media["path"])
    captured = {case["name"]: case for case in candidate["cases"]}
    results = []
    for row in reference["cases"]:
        actual, expected = captured[row["name"]], row["llama"]
        left, right = actual["embedding"], expected["embedding"]
        if len(left) != 768 or len(right) != 768 or not all(math.isfinite(value) for value in left + right):
            raise ValueError("Require finite768-dimensional vectors")
        cosine = sum(a*b for a, b in zip(left, right)) / math.sqrt(sum(a*a for a in left) * sum(b*b for b in right))
        equal = actual["prompt_tokens"] == expected["prompt_tokens"]
        results.append(dict(name=row["name"], cosine=cosine, same_token_count=equal,
                            candidate_tokens=actual["prompt_tokens"], reference_tokens=expected["prompt_tokens"],
                            passed=equal and cosine >= args.min_cosine))
    report = dict(reference=str(args.reference.resolve()), reference_sha256=digest(args.reference),
                  candidate=str(args.candidate.resolve()), candidate_sha256=digest(args.candidate),
                  media_provenance=str(args.media_provenance.resolve()), media_provenance_sha256=digest(args.media_provenance),
                  backend=candidate["backend"], binary_artifacts=candidate["binary_artifacts"],
                  min_cosine=args.min_cosine, cases=results, passed=all(row["passed"] for row in results),
                  limitations="Recorded unchanged synthetic model/projector/media/manifest numerical comparison; no saved latency reuse, speedup or retrieval-quality claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["backend", "passed", "cases"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
