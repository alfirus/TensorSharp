#!/usr/bin/env python3
"""Audit both HTTP engines against an independent EmbeddingGemma 2 oracle.

Matches oracle text cases to the shared correctness corpus and raw-token cases
to exact recorded benchmark inputs. Hashes and comparisons bind the evidence;
this command performs no inference and can run after all timed work finishes.
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
    parser.add_argument("--http-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=.9999)
    parser.add_argument("--max-absolute-error", type=float, default=.002)
    args = parser.parse_args()
    oracle = json.loads(args.reference.read_text())
    results = json.loads(args.http_results.read_text())
    if oracle["sha256"] != results["model_sha256"]:
        raise ValueError("The HTTP report and oracle must describe the same model SHA256")
    comparisons = {}
    for engine_name in ("tensorsharp", "llama"):
        engine = results["engines"][engine_name]
        rows = []
        for case in oracle["cases"]:
            vector = None
            source = None
            if case["text"] in results["inputs"]:
                index = results["inputs"].index(case["text"])
                vector = engine["correctness"]["vectors"][index]
                source = "correctness corpus row " + str(index)
            else:
                for name, inputs in results["benchmark_inputs"].items():
                    if case["tokens"] in inputs and "vectors" in engine["benchmarks"][name]:
                        index = inputs.index(case["tokens"])
                        vector = engine["benchmarks"][name]["vectors"][index]
                        source = name + " row " + str(index)
                        break
            if vector is None:
                raise ValueError("No exact recorded text or raw-token input for oracle case " + str(case["index"]))
            reference = case["embedding"]
            if len(vector) != 768 or len(reference) != 768 or any(not math.isfinite(value) for value in vector + reference):
                raise ValueError("Require finite 768-dimensional vectors")
            cosine = sum(a*b for a, b in zip(vector, reference)) / math.sqrt(sum(a*a for a in vector)*sum(b*b for b in reference))
            error = max(abs(a-b) for a, b in zip(vector, reference))
            rows.append(dict(index=case["index"], text=case["text"], tokens=len(case["tokens"]), source=source,
                             cosine=cosine, max_absolute_error=error,
                             passed=cosine >= args.min_cosine and error <= args.max_absolute_error))
        comparisons[engine_name] = dict(cases=rows, passed=all(row["passed"] for row in rows),
            min_cosine=min(row["cosine"] for row in rows), max_absolute_error=max(row["max_absolute_error"] for row in rows))
    report = dict(reference=str(args.reference.resolve()), reference_sha256=sha256(args.reference),
        http_results=str(args.http_results.resolve()), http_results_sha256=sha256(args.http_results),
        model_sha256=oracle["sha256"], engines=comparisons,
        thresholds=dict(min_cosine=args.min_cosine, max_absolute_error=args.max_absolute_error),
        tensorsharp_passed=comparisons["tensorsharp"]["passed"],
        all_engines_passed=all(engine["passed"] for engine in comparisons.values()),
        limitations="Architecture parity for recorded text/token cases; no quality leaderboard or performance claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({name: {key: engine[key] for key in ("passed", "min_cosine", "max_absolute_error")}
                      for name, engine in comparisons.items()}, indent=2))
    return 0 if report["tensorsharp_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
