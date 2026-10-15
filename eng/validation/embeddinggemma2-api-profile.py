#!/usr/bin/env python3
"""Bounded exact-HTTP-input API diagnostic of an already frozen CPU graph.

Build the new small probe in a COPY of the frozen model probe directory with
TensorSharpApiProbeSnapshot pointing to the original immutable directory. This
build references assemblies only; this tool rejects changed runtime binaries.
The retained HTTP report provides inputs; this diagnostic makes no HTTP claims.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess


STANDARD = ("single_short", "single_medium", "single_long", "batch8_short",
            "batch8_medium", "batch32_short", "batch_mixed")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def qualified_graph(prior):
    if not prior["completed"] or prior["quality_thresholds"] != {"min_cosine": .99999, "max_absolute_error": .0005}:
        raise ValueError("Require completed qualification with the unchanged strict NumPy thresholds")
    runs = prior["runs"]
    if len(runs) != 2 or {run["name"] for run in runs} != {"eight", "8192"}:
        raise ValueError("Require both independent eight-case and 8192-token model gates")
    for run in runs:
        expected = 8 if run["name"] == "eight" else 1
        if (run["exit_code"], run["comparison_exit_code"], run["executed"], run["passed"], run["skipped"]) != (0, 0, expected, expected, 0):
            raise ValueError("Unavailable, skipped or failed model gates cannot qualify the frozen graph")
        if not math.isfinite(run["min_cosine"]) or not math.isfinite(run["max_absolute_error"]) or run["min_cosine"] < .99999 or run["max_absolute_error"] > .0005:
            raise ValueError("The frozen graph failed the unchanged strict NumPy thresholds")


def decomposition(candidate, profile, http):
    if not candidate["completed"] or set(http["benchmark_inputs"]) != set(STANDARD):
        raise ValueError("Require a completed diagnostic and exactly the seven preserved HTTP inputs")
    cases = candidate["cases"]
    if len(cases) != 7 or [item["name"] for item in cases] != list(STANDARD):
        raise ValueError("Require every standard case exactly once")
    if any(not item["api_direct_bit_identical"] or len(item["string_api_samples_ms"]) != 2 or
           len(item["direct_token_api_samples_ms"]) != 2 for item in cases):
        raise ValueError("Require both AB/BA pairs and actual API/direct bit identity")
    expected_calls = expected_rows = 0
    for item in cases:
        lengths = item["sequence_lengths"]
        if len(lengths) != len(http["benchmark_inputs"][item["name"]]) or lengths != [len(row) for row in item["tokens"]]:
            raise ValueError("The diagnostic input/token geometry differs")
        if any(length <= 0 or length > 8192 for length in lengths):
            raise ValueError("Invalid recorded sequence geometry")
        hashes = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in http["benchmark_inputs"][item["name"]]]
        if item["text_utf8_sha256"] != hashes:
            raise ValueError("The diagnostic text differs from the preserved HTTP inputs")
        token_bytes = b"".join(struct.pack("<i", token) for row in item["tokens"] for token in row)
        if hashlib.sha256(token_bytes).hexdigest() != item["tokens_sha256"]:
            raise ValueError("Recorded token bytes do not match their hash")
        sorted_lengths = sorted(lengths)
        first = batches = 0
        while first < len(sorted_lengths):
            next_row, count = first + 1, sorted_lengths[first]
            while next_row < len(sorted_lengths) and next_row - first < 64 and count + sorted_lengths[next_row] <= 4096:
                count += sorted_lengths[next_row]
                next_row += 1
            first = next_row
            batches += 1
        if item["compact_compute_calls_per_api"] != batches or item["prompt_tokens"] != sum(lengths):
            raise ValueError("Recorded compact batch geometry differs from the API contract")
        actual_http = http["engines"]["tensorsharp"]["benchmarks"][item["name"]]
        if actual_http["prompt_tokens"] != sum(lengths) or actual_http["texts"] != len(lengths):
            raise ValueError("Token/text counts differ from the preserved actual HTTP response")
        expected_calls += batches * 5
        expected_rows += sum(lengths) * 5
    if (candidate["measured_string_calls"], candidate["measured_direct_token_calls"], candidate["warmup_calls"]) != (14, 14, 7):
        raise ValueError("Require the bounded seven warmups and fourteen calls per measured API")
    if (candidate["expected_successful_compute_calls"], candidate["expected_compute_rows"]) != (expected_calls, expected_rows):
        raise ValueError("Recorded Compute coverage differs from the actual input geometry")
    if (profile["successful_compute_calls"] != candidate["expected_successful_compute_calls"] or
            profile["input_rows"] != candidate["expected_compute_rows"]):
        raise ValueError("Buffered successful Compute counters do not cover the actual API work")
    total = candidate["total_enclosing_embedding_api_ms"]
    expected = sum(candidate[key] for key in ("measured_string_api_ms", "measured_direct_token_api_ms", "warmup_api_ms"))
    if not math.isfinite(total) or total <= 0 or not math.isclose(total, expected, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("Invalid enclosing API duration")
    for field, sample_field in (("measured_string_api_ms", "string_api_samples_ms"),
                                ("measured_direct_token_api_ms", "direct_token_api_samples_ms"),
                                ("separate_tokenization_ms", "separate_tokenization_samples_ms")):
        if not math.isclose(candidate[field], sum(sum(item[sample_field]) for item in cases), rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError("Recorded aggregate duration differs from its actual samples")
    if not math.isclose(candidate["warmup_api_ms"], sum(item["warmup_ms"] for item in cases), rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("Recorded warmup duration differs from its actual samples")
    compute = profile["compute_elapsed_ms"]
    preparation = profile["input_preparation"]["elapsed_ms"]
    if not math.isfinite(preparation) or preparation < 0 or compute + preparation > total:
        raise ValueError("Successful nested graph times exceed their enclosing measured API calls")
    if any(not math.isfinite(sample) or sample <= 0 for item in cases for key in
           ("separate_tokenization_samples_ms", "string_api_samples_ms", "direct_token_api_samples_ms") for sample in item[key]):
        raise ValueError("Invalid recorded API/tokenization samples")
    return dict(successful_cases=7, successful_compute_calls=profile["successful_compute_calls"],
                compute_rows=profile["input_rows"], total_enclosing_embedding_api_ms=total,
                compute_ms=compute, input_preparation_ms=preparation,
                compute_percent_of_enclosing_api=compute / total * 100,
                input_preparation_percent_of_enclosing_api=preparation / total * 100,
                api_remainder_ms=total - compute - preparation,
                api_remainder_percent=(total - compute - preparation) / total * 100,
                separate_tokenization_ms=candidate["separate_tokenization_ms"],
                limitations="Compute/input preparation are nested successful totals including all warmup and AB/BA calls. The remainder includes string tokenization, cloning, sorting, scheduling, allocation and profiler overhead; no HTTP request was measured and no HTTP-overhead attribution is justified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "prior-quality", "http-report", "probe", "library", "model", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    quality = module("owned_quality", args.root / "eng/validation/embeddinggemma2-owned-attention-quality.py")
    phase = module("phase", args.root / "eng/validation/embeddinggemma2-profile.py")
    prior = json.loads(args.prior_quality.read_text())
    http = json.loads(args.http_report.read_text())
    qualified_graph(prior)
    if sha(args.library) != prior["library"]["sha256"]:
        raise ValueError("Require the exact previously qualified graph/library")
    if set(http["benchmark_inputs"]) != set(STANDARD) or sha(args.model) != http["model_sha256"]:
        raise ValueError("Require the preserved standard7 inputs and exact model")
    copied = []
    for item in prior["binary_artifacts"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("An original frozen runtime artifact changed")
        name = Path(item["path"]).name
        if name == "EmbeddingGemma2Probe.dll":
            continue
        path = args.probe.parent / name
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError("The diagnostic runtime differs from the frozen graph: " + name)
        copied.append(dict(path=str(path), sha256=item["sha256"]))
    flags = dict(prior["flags"], TENSORSHARP_EMBEDDING_CPU_PROFILE="1")
    env = dict(os.environ, **flags)
    env.pop("TS_CPU_DISABLE_AVX512", None)
    args.output.mkdir(parents=True, exist_ok=True)
    candidate, log = [args.output / name for name in ("api.json", "api.log")]
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), completed=False,
                  prior_quality_sha256=sha(args.prior_quality), http_report_sha256=sha(args.http_report),
                  flags=flags, frozen_runtime=copied, diagnostic_probe=dict(path=str(args.probe), sha256=sha(args.probe)),
                  library=prior["library"], limitations="Bounded diagnostic only. No model/backend/default change, no formal performance or HTTP decomposition claim.")
    try:
        argv = ["dotnet", str(args.probe), "--model", str(args.model), "--http-report", str(args.http_report),
                "--output", str(candidate), "--threads", "8"]
        report["command"] = argv
        with log.open("w") as stream:
            report["exit_code"] = subprocess.run(argv, env=env, cwd=args.root, stdout=stream, stderr=subprocess.STDOUT).returncode
        if report["exit_code"]:
            return 1
        report["actual_math"] = quality.native_evidence(log, args.library, require_q16=True,
                                                       require_fused=flags.get("TENSORSHARP_EMBEDDING_FUSED_GATE_GELU") == "1")
        summary = phase.summarize(log)
        analysis = decomposition(json.loads(candidate.read_text()), summary, http)
        (args.output / "phase-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output / "api-decomposition.json").write_text(json.dumps(analysis, indent=2) + "\n")
        report.update(completed=True, executed=7, passed=7, skipped=0, decomposition=analysis)
        print(json.dumps(analysis, indent=2))
        return 0
    finally:
        # Hash only this tool's closed outputs. A surrounding runner may keep
        # driver.log open while this process writes stdout into it.
        names = ("api.json", "api.log", "phase-summary.json", "api-decomposition.json")
        report["report_hashes"] = {name: sha(args.output / name) for name in names if (args.output / name).is_file()}
        (args.output / "api-provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
