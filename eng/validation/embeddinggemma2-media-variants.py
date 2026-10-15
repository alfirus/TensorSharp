#!/usr/bin/env python3
"""Matched ABBA HTTP measurements of one TensorSharp environment flag.

Uses identical binaries, models, prepared media, and frontend policies in four
sequential owned server processes. Model loading and client payload preparation
are excluded. Generated evidence belongs in ignored artifacts/ or docs/validation/.
This measures the selected case only, rather than cross-engine performance.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from embeddinggemma2_evidence import command_artifacts


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", type=json.loads, required=True)
    parser.add_argument("--flag", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18380")
    parser.add_argument("--expected-backend", required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case", default="audio")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--prewarm-seconds", type=float, default=10)
    parser.add_argument("--minimum-measure-seconds", type=float, default=2)
    parser.add_argument("--maximum-ratio", type=float, default=.98)
    parser.add_argument("--min-cosine", type=float, default=.99999)
    parser.add_argument("--max-absolute-error", type=float, default=.0005)
    args = parser.parse_args()
    if not isinstance(args.command, list) or not args.command or not all(isinstance(x, str) for x in args.command):
        parser.error("Command must be a nonempty JSON string array")
    if not re.fullmatch(r"[A-Z_][A-Z_0-9]*", args.flag) or any(x.startswith(args.flag + "=") for x in args.command):
        parser.error("Flag must be an environment variable absent from the command")
    if args.warmup < 0 or args.repeats < 1 or any(not math.isfinite(x) or x < 0 for x in
            (args.prewarm_seconds, args.minimum_measure_seconds, args.max_absolute_error)):
        parser.error("Invalid warmup, repeats, duration or error limit")
    if not math.isfinite(args.maximum_ratio) or args.maximum_ratio <= 0 or not -1 <= args.min_cosine <= 1:
        parser.error("Invalid quality/performance thresholds")
    bench = module("embedding_bench", ROOT / "benchmarks/EmbeddingBench/embedding_bench.py")
    media = module("media_requests", Path(__file__).with_name("compare-embeddinggemma2-media.py"))
    bench.KEEP_ALIVE = True
    matching = [c for c in json.loads(args.cases.read_text())["cases"] if c["name"] == args.case]
    if len(matching) != 1:
        raise ValueError("Require exactly one matching case")
    case = matching[0]
    files = {(args.cases.parent / p["value"]).resolve() for p in case["parts"] if p["kind"] != "text"}
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(completed=False, passed=False, command=args.command, flag=args.flag, case=case,
        model=fingerprint(args.model_file), projector=fingerprint(args.mmproj), manifest=fingerprint(args.cases),
        media=[fingerprint(p) for p in sorted(files)], runner=fingerprint(__file__),
        binary_artifacts=command_artifacts(bench.command_artifacts, args.command),
        ggml_revision=subprocess.check_output(["git", "-C", str(ROOT / "ExternalProjects/ggml"), "rev-parse", "HEAD"], text=True).strip(),
        ggml_tracked_changes=subprocess.check_output(["git", "-C", str(ROOT / "ExternalProjects/ggml"), "status",
                                                     "--porcelain", "--untracked-files=no"], text=True).strip(),
        expected_backend=args.expected_backend, order=["0", "1", "1", "0"],
        warmup=args.warmup, repeats=args.repeats, prewarm_seconds=args.prewarm_seconds,
        minimum_measure_seconds=args.minimum_measure_seconds,
        thresholds=dict(maximum_ratio=args.maximum_ratio, min_cosine=args.min_cosine,
                        max_absolute_error=args.max_absolute_error), runs=[],
        limitations=["One selected synthetic case; no model loading, retrieval leaderboard or cross-engine parity claim.",
                     "Numerical comparisons are against the same-binary baseline; independent full-model oracles remain separate.",
                     "The variant must have separate evidence of actual feature execution; this flag alone is insufficient."])
    destination = args.output / "results.json"

    def save():
        destination.write_text(json.dumps(result, indent=2) + "\n")

    for index, value in enumerate(result["order"]):
        record = dict(index=index, flag_value=value, completed=False)
        result["runs"].append(record)
        environment = os.environ.copy()
        environment[args.flag] = value
        with (args.output / f"run-{index}-{value}.log").open("w") as log:
            process = subprocess.Popen(args.command, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                started = time.monotonic()
                while time.monotonic() - started < 180:
                    if process.poll() is not None:
                        raise RuntimeError("Server exited at startup")
                    try:
                        status, models = bench.request(args.url, "/v1/models")
                        if status == 200:
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(.25)
                else:
                    raise TimeoutError("Server startup")
                status, discovery = bench.request(args.url, "/api/models")
                if status != 200 or discovery["loadedBackend"] != args.expected_backend:
                    raise ValueError("Unexpected actual backend: " + repr(discovery))
                record["discovery"] = discovery
                model_id = models["data"][0]["id"]
                prepared = json.loads(media.payload(case, args.cases.parent, model_id))["input"]

                def embed():
                    vectors, usage = bench.embedding(args.url, model_id, prepared)
                    bench.assert_vectors(vectors, 768)
                    if len(vectors) != 1 or usage["prompt_tokens"] <= 0:
                        raise ValueError("Unexpected vector/token count")
                    return vectors[0], usage["prompt_tokens"]

                started = time.monotonic()
                prewarm_calls = 0
                while time.monotonic() - started < args.prewarm_seconds:
                    embed()
                    prewarm_calls += 1
                record["prewarm_calls"] = prewarm_calls
                for _ in range(args.warmup):
                    embed()
                samples = []
                started = time.monotonic()
                while len(samples) < args.repeats or time.monotonic() - started < args.minimum_measure_seconds:
                    tick = time.perf_counter()
                    vector, tokens = embed()
                    samples.append((time.perf_counter() - tick) * 1000)
                record.update(completed=True, embedding=vector, prompt_tokens=tokens,
                              samples_ms=samples, median_ms=statistics.median(samples),
                              sample_count=len(samples))
                print(index, value, round(record["median_ms"], 3), "ms", tokens, "tokens", flush=True)
            finally:
                for connection in getattr(bench.CONNECTIONS, "connections", {}).values():
                    connection.close()
                bench.CONNECTIONS.connections = {}
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                record["server_exit_code"] = process.returncode
                save()
                if process.returncode != 0:
                    raise RuntimeError("Server failed graceful shutdown: " + str(process.returncode))
    comparisons = []
    for off, on in ((result["runs"][0], result["runs"][1]), (result["runs"][3], result["runs"][2])):
        cosine = bench.cosine(off["embedding"], on["embedding"])
        error = max(abs(a - b) for a, b in zip(off["embedding"], on["embedding"]))
        ratio = on["median_ms"] / off["median_ms"]
        comparisons.append(dict(off_index=off["index"], on_index=on["index"], cosine=cosine,
            max_absolute_error=error, same_tokens=off["prompt_tokens"] == on["prompt_tokens"],
            latency_ratio=ratio, quality_pass=cosine >= args.min_cosine and error <= args.max_absolute_error,
            performance_pass=ratio <= args.maximum_ratio))
    result.update(completed=True, comparisons=comparisons,
                  passed=all(c["quality_pass"] and c["performance_pass"] and c["same_tokens"] for c in comparisons))
    save()
    print(json.dumps(comparisons, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
