#!/usr/bin/env python3
"""Summarize opt-in CUDA embedding event profiles with checked phase accounting.

Enable TS_EMBEDDING_CUDA_PHASES=1 before creating the direct CUDA encoder. These
GPU stream intervals may include host issue idle time. They diagnose phase cost;
they do not establish model quality or replace a trace-free paired HTTP run.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


PHASES = (
    "initial_ple", "norm_qkv", "split_norm_rope", "local_attention",
    "global_attention", "attention_output_residual", "feed_forward",
    "ple_residual", "final_pool_head",
)
LAYER_PHASES = ("norm_qkv", "split_norm_rope", "attention_output_residual", "feed_forward", "ple_residual")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(profile):
    lengths = profile["sequence_lengths"]
    integers = (profile["step"], profile["tokens"], profile["batches"], profile["hidden"], profile["heads"], profile["layers"], *lengths)
    if any(type(value) is not int or value <= 0 for value in integers):
        raise ValueError("Require positive integer geometry and step")
    if len(lengths) != profile["batches"] or sum(lengths) != profile["tokens"] or max(lengths) > 8192:
        raise ValueError("Packed sequence lengths do not match profile geometry")
    for key in ("per_stage_ms", "stage_calls", "host_issue_stage_ms"):
        if set(profile[key]) != set(PHASES):
            raise ValueError("Require all nine unique embedding phases")
    calls = profile["stage_calls"]
    if any(type(value) is not int or value < 0 for value in calls.values()):
        raise ValueError("Invalid phase call count")
    if calls["initial_ple"] != 1 or calls["final_pool_head"] != 1:
        raise ValueError("Require one PLE preparation and one final pool/head")
    if any(calls[name] != profile["layers"] for name in LAYER_PHASES):
        raise ValueError("Layer phase counts differ from the encoder depth")
    if calls["local_attention"] + calls["global_attention"] != profile["layers"]:
        raise ValueError("Attention phases do not cover every layer exactly once")
    times = [profile[key] for key in ("wall_ms", "host_issue_ms", "gpu_span_total_ms")]
    times += list(profile["per_stage_ms"].values()) + list(profile["host_issue_stage_ms"].values())
    if any(not math.isfinite(value) or value < 0 for value in times):
        raise ValueError("Require finite nonnegative timing records")
    if profile["wall_ms"] <= 0 or profile["gpu_span_total_ms"] <= 0:
        raise ValueError("Require completed positive-duration GPU work")
    if profile["host_issue_ms"] > profile["wall_ms"]:
        raise ValueError("Host issue interval exceeds enclosing wall time")
    for parts, total in (("per_stage_ms", "gpu_span_total_ms"), ("host_issue_stage_ms", "host_issue_ms")):
        if not math.isclose(sum(profile[parts].values()), profile[total], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("Phase intervals differ from their recorded total")
    for name in PHASES:
        if calls[name] == 0 and (profile["per_stage_ms"][name] != 0 or profile["host_issue_stage_ms"][name] != 0):
            raise ValueError("Uncalled phase contains a measured interval")


def summarize(logs, discard_first):
    groups, inputs = {}, []
    for log in logs:
        inputs.append({"path": str(log.resolve()), "sha256": sha(log)})
        for line_number, line in enumerate(log.read_text().splitlines(), 1):
            if line.startswith("EMBEDDING_CUDA_PHASES_ABORT "):
                raise ValueError(f"Aborted step in {log}:{line_number}")
            if not line.startswith("EMBEDDING_CUDA_PHASES "):
                continue
            record = json.loads(line.removeprefix("EMBEDDING_CUDA_PHASES "))
            check(record)
            geometry = (tuple(record["sequence_lengths"]), record["hidden"], record["heads"], record["layers"])
            groups.setdefault(geometry, []).append({"log": str(log.resolve()), "line": line_number, "profile": record})
    if not groups:
        raise ValueError("No completed CUDA embedding phase records found")
    summaries = []
    for geometry, records in groups.items():
        chosen = records[discard_first:]
        if not chosen:
            raise ValueError("Warmup discard removed every record for a geometry")
        profiles = [record["profile"] for record in chosen]
        gpu_total = sum(profile["gpu_span_total_ms"] for profile in profiles)
        stages = {}
        for name in PHASES:
            stage_total = sum(profile["per_stage_ms"][name] for profile in profiles)
            stages[name] = {
                "gpu_interval_median_ms": statistics.median(profile["per_stage_ms"][name] for profile in profiles),
                "gpu_interval_total_ms": stage_total,
                "gpu_interval_percent": stage_total / gpu_total * 100,
                "host_issue_median_ms": statistics.median(profile["host_issue_stage_ms"][name] for profile in profiles),
                "calls_per_step": sorted({profile["stage_calls"][name] for profile in profiles}),
            }
        summaries.append({
            "geometry": {"sequence_lengths": list(geometry[0]), "hidden": geometry[1], "heads": geometry[2], "layers": geometry[3]},
            "available_steps": len(records), "discarded_first_steps": discard_first, "selected_steps": len(chosen),
            "wall_median_ms": statistics.median(profile["wall_ms"] for profile in profiles),
            "host_issue_median_ms": statistics.median(profile["host_issue_ms"] for profile in profiles),
            "gpu_span_median_ms": statistics.median(profile["gpu_span_total_ms"] for profile in profiles),
            "stages": dict(sorted(stages.items(), key=lambda item: item[1]["gpu_interval_total_ms"], reverse=True)),
            "records": records,
        })
    return {"completed": True, "phase_accounting_passed": True, "inputs": inputs, "groups": summaries,
            "limitations": "Diagnostic GPU event intervals include stream idle time from host issue/allocation. Host issue and wall overlap these intervals and must not be added. "
            "Initial model load, tokenization, pre-Begin input preparation and final CPU embedding normalization are excluded. "
            "Median phase intervals need not sum to the median total; the percentages use accumulated intervals. "
            "No numerical model gate, runtime binary provenance or formal throughput comparison is supplied by this parser."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--discard-first-per-geometry", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.discard_first_per_geometry < 0:
        parser.error("Warmup discard must be nonnegative")
    report = summarize(args.log, args.discard_first_per_geometry)
    report["tool_sha256"] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed": True, "phase_accounting_passed": True,
                      "groups": [{key: value for key, value in group.items() if key != "records"} for group in report["groups"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
