#!/usr/bin/env python3
"""Summarize an opt-in EmbeddingGemma2 CPU projection and phase diagnostic log.

Enable TENSORSHARP_EMBEDDING_CPU_PROFILE=1 before loading the CPU model. The
encoder emits one JSON record on disposal. These nested elapsed times include
warmup and scheduling; they are not a replacement for paired HTTP benchmarks.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def category(name):
    if name == "per_layer_model_proj":
        return "PLE"
    if name == "output":
        return "Output"
    if ".attn_q+" in name:
        return "QKV"
    if name.endswith(".attn_output"):
        return "Attention output"
    if ".ffn_gate+" in name:
        return "FFN gate/up"
    if name.endswith(".ffn_down"):
        return "FFN down"
    if name.endswith(".inp_gate"):
        return "Input gate"
    if name.endswith(".proj"):
        return "Layer projection"
    return name


def summarize(log):
    records = [json.loads(line.removeprefix("EMBEDDING_CPU_PROFILE "))
               for line in log.read_text().splitlines()
               if line.startswith("EMBEDDING_CPU_PROFILE ")]
    if len(records) != 1:
        raise ValueError("Require exactly one completed encoder profile record")
    profile = records[0]
    compute_ms = profile["compute_elapsed_ms"]
    if not math.isfinite(compute_ms) or compute_ms <= 0 or profile["successful_compute_calls"] <= 0:
        raise ValueError("Require successful compute calls and a finite positive compute duration")
    groups = {}
    for projection in profile["projections"]:
        ms = projection["elapsed_ms"]
        if not math.isfinite(ms) or ms < 0 or projection["successful_calls"] <= 0:
            raise ValueError("Require finite successful projection measurements")
        key = category(projection["name"])
        group = groups.setdefault(key, dict(matrices=0, elapsed_ms=0.0, shapes=set(), math=set(), source_formats=set()))
        group["matrices"] += 1
        group["elapsed_ms"] += ms
        group["shapes"].add((projection["input"], projection["output"]))
        group["math"].add(projection["math"])
        group["source_formats"].add(projection["source_format"])
    projection_ms = sum(group["elapsed_ms"] for group in groups.values())
    if not math.isclose(projection_ms, profile["projection_elapsed_ms"], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("Projection total differs from its recorded component measurements")
    if projection_ms > compute_ms:
        raise ValueError("Projection times exceed successful enclosing compute time")
    for group in groups.values():
        group["compute_percent"] = group["elapsed_ms"] / compute_ms * 100
        for key in ("shapes", "math", "source_formats"):
            group[key] = sorted(group[key])
    result = dict(log=str(log.resolve()), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                successful_compute_calls=profile["successful_compute_calls"], input_rows=profile["input_rows"],
                compute_elapsed_ms=compute_ms, projection_elapsed_ms=projection_ms,
                projection_compute_percent=projection_ms / compute_ms * 100,
                other_compute_ms=compute_ms - projection_ms,
                other_compute_percent=(compute_ms - projection_ms) / compute_ms * 100,
                groups=dict(sorted(groups.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
                profile=profile,
                limitations=profile["limitations"] + " Other compute is an unmeasured remainder; it must not be attributed to a specific operation without further profiling.")
    if "phases" in profile:
        phases = {}
        for phase in profile["phases"]:
            name = phase["name"]
            if name in phases or phase["parent"] != "compute" or phase["accounting"] != "exclusive-leaf":
                raise ValueError("Require unique disjoint compute leaf phase records")
            ms = phase["elapsed_ms"]
            if not math.isfinite(ms) or ms < 0 or phase["successful_calls"] < 0:
                raise ValueError("Invalid phase measurement")
            phases[name] = dict(phase, compute_percent=ms / compute_ms * 100)
        phase_ms = sum(phase["elapsed_ms"] for phase in phases.values())
        accounted_ms = projection_ms + phase_ms
        if accounted_ms > compute_ms:
            raise ValueError("Disjoint phase and projection times exceed enclosing compute time")

        def rollup(phase_names, projection_names=()):
            ms = sum(phases[name]["elapsed_ms"] for name in phase_names)
            ms += sum(groups[name]["elapsed_ms"] for name in projection_names)
            return dict(accounting="inclusive-derived-overlaps-leaves", phases=list(phase_names),
                        projection_categories=list(projection_names), elapsed_ms=ms,
                        compute_percent=ms / compute_ms * 100)

        result.update(phases=dict(sorted(phases.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)),
                      exclusive_phase_elapsed_ms=phase_ms, accounted_compute_ms=accounted_ms,
                      unattributed_compute_ms=compute_ms - accounted_ms,
                      unattributed_compute_percent=(compute_ms - accounted_ms) / compute_ms * 100,
                      derived_parents={
                          "Attention core including RoPE": rollup(("RopeTables", "Rope", "AttentionBuffers", "AttentionClear", "AttentionPacking", "AttentionCompute")),
                          "PLE preparation including projection": rollup(("PleScaling", "PleNormalization"), ("PLE",)),
                          "Normalization including output L2": rollup(("RmsNormalization", "PleNormalization", "FinalNormalization", "OutputNormalization")),
                          "Pooling and output": rollup(("FinalNormalization", "Pooling", "OutputNormalization"), ("Output",)),
                      },
                      backend=profile["backend"], graph=profile["graph"], native_math=profile["native_math"],
                      input_preparation=profile["input_preparation"], token_row_export=profile["token_row_export"])
        result["limitations"] += " Phase leaf records are disjoint. Derived parents overlap leaves and each other; never add parent totals to the disjoint accounting sum. Input preparation and TokenRows are outside compute."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "profile"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
