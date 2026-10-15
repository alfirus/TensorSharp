#!/usr/bin/env python3
"""Summarize diagnostic split-graph native CPU profiles with checked accounting.

Enable TSG_EMBEDDING_CPU_PROFILE=1 before loading ggml_cpu. Splitting a graph
changes fusion and adds scheduling overhead; these timings cannot establish
paired latency parity. Use an unprofiled HTTP run for performance claims.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

CATEGORIES = {"projection", "attention", "other"}
PREFIX = "EMBEDDING_GGML_CPU_PROFILE "


def check(record):
    if record.get("diagnostic") is not True or record.get("split_graph") is not True:
        raise ValueError("Require explicit diagnostic split-graph marker")
    if any(type(record[key]) is not int or record[key] <= 0 for key in
           ("step", "length", "batch", "sequences", "nodes", "segments")):
        raise ValueError("Require positive integer geometry and node counts")
    for key in ("category_ms", "category_nodes", "category_segments"):
        if set(record[key]) != CATEGORIES:
            raise ValueError("Require all three unique categories")
    for key, total in (("category_nodes", "nodes"), ("category_segments", "segments")):
        if any(type(value) is not int or value < 0 for value in record[key].values()):
            raise ValueError("Invalid category counts")
        if sum(record[key].values()) != record[total]:
            raise ValueError("Category counts do not cover every graph node/segment")
    times = [record["wall_ms"], *record["category_ms"].values()]
    if any(type(value) not in (float, int) or not math.isfinite(value) or value < 0 for value in times):
        raise ValueError("Require finite nonnegative intervals")
    if record["wall_ms"] <= 0 or sum(record["category_ms"].values()) <= 0:
        raise ValueError("Require completed positive-duration CPU work")
    if sum(record["category_ms"].values()) > record["wall_ms"] + 1e-5:
        raise ValueError("Category intervals exceed enclosing wall interval")
    for category in CATEGORIES:
        if record["category_segments"][category] > record["category_nodes"][category]:
            raise ValueError("More segments than graph nodes")
        if record["category_segments"][category] == 0 and record["category_ms"][category] != 0:
            raise ValueError("Unexecuted category contains a measured interval")


def summarize(logs, discard_first=0):
    if type(discard_first) is not int or discard_first < 0:
        raise ValueError("Warmup discard must be a nonnegative integer")
    groups, inputs = {}, []
    for log in logs:
        inputs.append(dict(path=str(log.resolve()), sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        for line_number, line in enumerate(log.read_text().splitlines(), 1):
            if line.startswith("EMBEDDING_GGML_CPU_PROFILE_ABORT "):
                raise ValueError(f"Aborted profile in {log}:{line_number}")
            if not line.startswith(PREFIX):
                continue
            record = json.loads(line[len(PREFIX):])
            check(record)
            # Keep graph counts in the key: different input modes/policies can
            # have the same length while executing different graph structures.
            geometry = tuple(record[key] for key in ("length", "batch", "sequences", "nodes", "segments"))
            groups.setdefault(geometry, []).append(dict(log=str(log.resolve()), line=line_number, profile=record))
    if not groups:
        raise ValueError("No completed native CPU phase records")
    output = []
    for geometry, records in groups.items():
        selected = records[discard_first:]
        if not selected:
            raise ValueError("Warmup discard removed every record for a geometry")
        profiles = [record["profile"] for record in selected]
        total = sum(sum(record["category_ms"].values()) for record in profiles)
        categories = {}
        for category in sorted(CATEGORIES):
            elapsed = sum(record["category_ms"][category] for record in profiles)
            categories[category] = dict(median_ms=statistics.median(record["category_ms"][category] for record in profiles),
                total_ms=elapsed, fraction=elapsed / total,
                node_counts=sorted({record["category_nodes"][category] for record in profiles}),
                segment_counts=sorted({record["category_segments"][category] for record in profiles}))
        output.append(dict(geometry=dict(zip(("length", "batch", "sequences", "nodes", "segments"), geometry)),
            available_steps=len(records), discarded_steps=discard_first, selected_steps=len(selected),
            wall_median_ms=statistics.median(record["wall_ms"] for record in profiles), categories=categories, records=records))
    return dict(completed=True, accounting_passed=True, diagnostic=True, inputs=inputs, groups=output,
        limitations="Split graph plans disable fusion across category boundaries and add scheduling overhead. "
        "Allocation, input preparation, tokenization and final normalization are excluded. Classification uses explicit learned-weight/attention-node identity; other includes pooling and arithmetic. "
        "Percentages use accumulated intervals; category medians need not sum to median wall time. This report is neither a quality gate nor a paired performance benchmark.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--discard-first-per-geometry", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.log, args.discard_first_per_geometry)
    result["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
