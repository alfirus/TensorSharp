#!/usr/bin/env python3
"""Check complete source-rate audio embeddings against independent resampled PCM.

The model is shared for source/reference comparisons. A separate preserved full
HF NumPy audio+text golden qualifies the canonical16kHz waveform. No saved timing
is treated as a fresh paired benchmark.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(left, right):
    if len(left) != 768 or len(right) != 768 or not all(math.isfinite(value) for value in left + right):
        raise ValueError("Require finite768-dimensional embeddings")
    cosine = sum(a * b for a, b in zip(left, right)) / math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    error = max(abs(a - b) for a, b in zip(left, right))
    return dict(cosine=cosine, max_component_error=error, passed=cosine >= .9999 and error <= .002)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "canonical-golden", "media-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    canonical = json.loads(args.canonical_golden.read_text())
    media = json.loads(args.media_manifest.read_text())
    if not candidate["completed"] or candidate["model_sha256"] != canonical["model_sha256"] or candidate["projector_sha256"] != canonical["projector_sha256"]:
        raise ValueError("Require a completed actual candidate with the independent model/projector identity")
    if candidate.get("audio_log_policy") != "HuggingFaceAddFloor":
        raise ValueError("This comparison requires the default HF log-add policy")
    if digest(candidate["case_manifest"]) != candidate["case_manifest_sha256"]:
        raise ValueError("Candidate input manifest changed after inference")
    rows = {row["name"]: row for row in candidate["cases"]}
    comparisons = []
    for item in media["files"]:
        actual, expected = rows[Path(item["source"]).stem], rows[Path(item["reference"]).stem]
        for key, row in (("source", actual), ("reference", expected)):
            if digest(item[key]) != item[key + "_sha256"] or not any(record["sha256"] == item[key + "_sha256"] for record in row["media_files"]):
                raise ValueError("Actual source or independent reference waveform differs")
        result = compare(actual["embedding"], expected["embedding"])
        result.update(source=item["source"], reference=item["reference"], same_token_count=actual["prompt_tokens"] == expected["prompt_tokens"])
        result["passed"] &= result["same_token_count"]
        comparisons.append(result)
    audio = rows["audio"]
    if not any(item["sha256"] == canonical["waveform_sha256"] for item in audio["media_files"]) or audio["prompt_tokens"] != canonical["total_input_rows"]:
        raise ValueError("Canonical waveform/input rows differ from the preserved independent golden")
    canonical_result = compare(audio["embedding"], canonical["embedding"])
    report = dict(passed=all(item["passed"] for item in comparisons) and canonical_result["passed"],
        candidate=str(args.candidate.resolve()), candidate_sha256=digest(args.candidate),
        canonical_golden=str(args.canonical_golden.resolve()), canonical_golden_sha256=digest(args.canonical_golden),
        media_manifest=str(args.media_manifest.resolve()), media_manifest_sha256=digest(args.media_manifest),
        comparisons=comparisons, canonical_independent_audio=canonical_result,
        thresholds=dict(min_cosine=.9999, max_component_error=.002),
        limitations="Source-rate versus independently resampled PCM shares the actual model. Canonical16kHz audio uses a preserved complete independent HF NumPy audio+text golden; no fresh model math or performance parity is claimed by this comparison.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
