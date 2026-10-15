#!/usr/bin/env python3
"""Compare raw audio soft rows with the independent complete HF NumPy oracle.

This consumes saved evidence without executing either implementation. The
projector and exact independently generated mel input must match. Raw rows are
unnormalized; both raw errors and normalized row errors are recorded.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--oracle-mel", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=.9999)
    parser.add_argument("--max-absolute-error", type=float, default=.01)
    args = parser.parse_args()
    if not math.isfinite(args.min_cosine) or not -1 <= args.min_cosine <= 1:
        parser.error("min-cosine must be finite and between -1 and 1")
    if not math.isfinite(args.max_absolute_error) or args.max_absolute_error < 0:
        parser.error("max-absolute-error must be finite and nonnegative")
    oracle = json.loads(args.oracle.read_text())
    candidate = json.loads(args.candidate_report.read_text())
    if not candidate.get("completed"):
        raise ValueError("Candidate lifecycle must have completed; also verify process exit status")
    if candidate["projector_sha256"] != oracle["projector_sha256"]:
        raise ValueError("Projectors differ")
    if candidate["mel_sha256"] != digest(args.oracle_mel):
        raise ValueError("Candidate mel differs from the independent oracle input")
    rows, dimensions = oracle["shape"]
    if (candidate["rows"], candidate["dimensions"], candidate["frames"]) != (rows, dimensions, oracle["input_mel_frames"]):
        raise ValueError("Audio input/output dimensions differ")
    path = Path(candidate["output"])
    data = path.read_bytes()
    if digest(path) != candidate["output_sha256"] or len(data) != rows * dimensions * 4:
        raise ValueError("Candidate raw rows were changed or truncated")
    values = struct.unpack("<" + str(rows * dimensions) + "f", data)
    comparisons = []
    for index, expected in enumerate(oracle["soft_rows"]):
        actual = values[index * dimensions:(index + 1) * dimensions]
        if len(expected) != dimensions or not all(math.isfinite(v) for v in (*expected, *actual)):
            raise ValueError("Invalid or nonfinite audio row")
        left = math.sqrt(sum(v * v for v in expected))
        right = math.sqrt(sum(v * v for v in actual))
        if min(left, right) == 0:
            raise ValueError("Zero audio row")
        comparisons.append(dict(row=index,
            cosine=sum(a * b for a, b in zip(expected, actual)) / (left * right),
            max_absolute_error=max(abs(a - b) for a, b in zip(expected, actual)),
            max_normalized_error=max(abs(a / left - b / right) for a, b in zip(expected, actual)),
            expected_norm=left, candidate_norm=right))
    minimum = min(row["cosine"] for row in comparisons)
    maximum = max(row["max_absolute_error"] for row in comparisons)
    passed = minimum >= args.min_cosine and maximum <= args.max_absolute_error
    report = dict(passed=passed, backend=candidate["backend"], min_row_cosine=minimum,
        max_absolute_error=maximum, max_normalized_error=max(row["max_normalized_error"] for row in comparisons),
        gates=dict(min_row_cosine=args.min_cosine, max_absolute_error=args.max_absolute_error), rows=comparisons,
        oracle=dict(path=str(args.oracle.resolve()), sha256=digest(args.oracle), method=oracle["method"],
                    source_url=oracle["source_url"], source_revision=oracle["source_revision"]),
        oracle_mel=dict(path=str(args.oracle_mel.resolve()), sha256=digest(args.oracle_mel)),
        candidate_report=dict(path=str(args.candidate_report.resolve()), sha256=digest(args.candidate_report)),
        projector_sha256=oracle["projector_sha256"], binary_artifacts=candidate["binary_artifacts"],
        limitations="Unnormalized audio soft rows for this waveform/projector; no speech retrieval leaderboard or end-to-end latency claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "backend", "min_row_cosine", "max_absolute_error", "max_normalized_error")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
