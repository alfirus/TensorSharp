#!/usr/bin/env python3
"""Diagnose actual audio stages against independent HF NumPy states.

This reports numerical differences without relaxing or replacing the final raw
or pooled quality gates. Generated reports belong in ignored validation paths.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-states", type=Path, required=True)
    parser.add_argument("--audio-oracle", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    oracle = json.loads(args.audio_oracle.read_text())
    candidate = json.loads(args.candidate_report.read_text())
    if not candidate.get("completed") or candidate["projector_sha256"] != oracle["projector_sha256"]:
        raise ValueError("Require a completed actual process and identical projector bytes")
    if oracle.get("past_distance_inclusive") != 11:
        raise ValueError("Use the corrected pinned-HF exclusive-left-window oracle")
    traces = {row["name"]: row for row in candidate.get("traces", [])}
    if not traces:
        raise ValueError("Candidate report contains no actual stage traces")
    rows = []
    with np.load(args.oracle_states, allow_pickle=False) as states:
        if hashlib.sha256(states["mel"].astype("<f4").tobytes()).hexdigest() != candidate["mel_sha256"]:
            raise ValueError("Oracle states and actual process use different mel bytes")
        for name, trace in traces.items():
            if name not in states:
                raise ValueError("Unknown oracle stage " + name)
            path = Path(trace["path"])
            if digest(path) != trace["sha256"]:
                raise ValueError("Actual trace changed after inference: " + name)
            expected = states[name].astype(np.float64).reshape(states[name].shape[0], -1)
            actual = np.fromfile(path, dtype="<f4").astype(np.float64).reshape(trace["rows"], trace["columns"])
            if actual.shape != expected.shape or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
                raise ValueError("Shape or finite-value mismatch: " + name)
            left_norm = np.linalg.norm(expected, axis=1)
            right_norm = np.linalg.norm(actual, axis=1)
            denominator = left_norm * right_norm
            cosine = np.divide(np.sum(expected * actual, axis=1), denominator,
                out=np.where((left_norm == 0) & (right_norm == 0), 1., 0.), where=denominator != 0)
            normalized_left = np.divide(expected, left_norm[:, None], out=np.zeros_like(expected), where=left_norm[:, None] != 0)
            normalized_right = np.divide(actual, right_norm[:, None], out=np.zeros_like(actual), where=right_norm[:, None] != 0)
            difference = np.abs(expected - actual)
            index = tuple(int(i) for i in np.unravel_index(np.argmax(difference), difference.shape))
            rows.append(dict(name=name, shape=list(expected.shape), min_row_cosine=float(np.min(cosine)),
                max_absolute_error=float(np.max(difference)), max_normalized_error=float(np.max(np.abs(normalized_left-normalized_right))),
                rms_error=float(np.sqrt(np.mean(difference * difference))), max_expected_magnitude=float(np.max(np.abs(expected))),
                maximum_error_index=list(index), expected_at_maximum=float(expected[index]), actual_at_maximum=float(actual[index])))
        missing = [name for name in states.files if name != "mel" and name not in traces]
    result = dict(completed=True, diagnostic_only=True, backend=candidate["backend"],
        audio_oracle=str(args.audio_oracle.resolve()), audio_oracle_sha256=digest(args.audio_oracle),
        oracle_states=str(args.oracle_states.resolve()), oracle_states_sha256=digest(args.oracle_states),
        candidate=str(args.candidate_report.resolve()), candidate_sha256=digest(args.candidate_report),
        stages=rows, unobserved_stages=missing,
        limitations="Intermediate diagnostics on one waveform/model/device, without quality or performance pass claims. Original final raw and complete pooled gates remain separate and unchanged.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
