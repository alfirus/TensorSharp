#!/usr/bin/env python3
"""Sequential, owned-server EmbeddingGemma 2 media HTTP latency comparison.

The controlled option inserts llama.cpp's automatic ``Video:`` text into the
TensorSharp request. An explicitly selected direct TensorSharp reference can
receive the same prefix for a matched ablation. Model loading, client file/base64 preparation, and
startup are excluded; server decoding, preprocessing, inference, and JSON are
included. Reports and logs belong in ignored docs/validation/ or artifacts/.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from embeddinggemma2_evidence import command_artifacts
from embeddinggemma2_http_environment import audit_child_environment


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def controlled_parts(parts, engine, candidate_prefix, reference_prefix=False):
    """Keep both direct TensorSharp roles identical in an explicit ablation."""
    result = []
    for part in parts:
        if part["kind"] == "video" and (candidate_prefix if engine == "tensorsharp" else reference_prefix):
            result.append(dict(kind="text", value="Video:"))
        result.append(part)
    return result


def validate_reference_prefix(candidate_prefix, reference_prefix, command, inherited):
    if reference_prefix and (not candidate_prefix or not
            audit_child_environment(command, inherited).resolved_server_environment):
        raise ValueError("Reference video prefix requires the candidate prefix and a direct TensorSharp reference host.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorsharp-command", type=json.loads, required=True)
    parser.add_argument("--llama-command", type=json.loads, required=True)
    parser.add_argument("--tensorsharp-url", default="http://127.0.0.1:18380")
    parser.add_argument("--llama-url", default="http://127.0.0.1:18082")
    parser.add_argument("--expected-backend", required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controlled-video-prefix", action="store_true")
    parser.add_argument("--reference-controlled-video-prefix", action="store_true",
                        help="Insert the same Video: prefix for a direct TensorSharp reference in an ablation; never use for llama.cpp's automatic prefix.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--prewarm-seconds", type=float, default=10)
    parser.add_argument("--minimum-measure-seconds", type=float, default=1)
    parser.add_argument("--max-latency-ratio", type=float, default=1.05)
    parser.add_argument("--min-cosine", type=float, default=.9999)
    parser.add_argument("--tensorsharp-first", action="store_true")
    parser.add_argument("--environment-note", action="append", default=[])
    parser.add_argument("--capture-boundary-telemetry", action="store_true",
                        help="Record read-only GPU/cgroup snapshots before launch and after shutdown, outside timing.")
    args = parser.parse_args()
    validate_reference_prefix(args.controlled_video_prefix, args.reference_controlled_video_prefix,
                              args.llama_command, os.environ)
    if args.warmup < 0 or args.repeats < 1 or any(not math.isfinite(x) or x < 0 for x in
            (args.prewarm_seconds, args.minimum_measure_seconds)):
        parser.error("Invalid warmup/repeats/durations")
    if not math.isfinite(args.max_latency_ratio) or args.max_latency_ratio <= 0 or not -1 <= args.min_cosine <= 1:
        parser.error("Invalid quality/performance thresholds")
    spec = importlib.util.spec_from_file_location("embedding_bench", ROOT / "benchmarks/EmbeddingBench/embedding_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    bench.KEEP_ALIVE = True
    spec = importlib.util.spec_from_file_location("media_requests", Path(__file__).with_name("compare-embeddinggemma2-media.py"))
    media = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(media)
    cases = json.loads(args.cases.read_text())["cases"]
    if not cases or len({case["name"] for case in cases}) != len(cases):
        raise ValueError("Require nonempty unique cases")
    files = {((args.cases.parent / part["value"]).resolve()) for case in cases for part in case["parts"] if part["kind"] != "text"}
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(completed=False, passed=False, platform=platform.platform(), machine=platform.machine(),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), runner_arguments=sys.argv[1:],
        model=fingerprint(args.model_file), projector=fingerprint(args.mmproj), manifest=fingerprint(args.cases),
        media=[fingerprint(path) for path in sorted(files)], inputs=cases,
        controlled_video_prefix=args.controlled_video_prefix, warmup=args.warmup, repeats=args.repeats,
        reference_controlled_video_prefix=args.reference_controlled_video_prefix,
        prewarm_seconds=args.prewarm_seconds, minimum_measure_seconds=args.minimum_measure_seconds,
        connection_mode="persistent", expected_backend=args.expected_backend, environment_notes=args.environment_note,
        thresholds=dict(min_cosine=args.min_cosine, maximum_latency_ratio=args.max_latency_ratio), engines={},
        binary_artifacts={name: command_artifacts(bench.command_artifacts, command) for name, command in
            (("tensorsharp", args.tensorsharp_command), ("llama", args.llama_command))},
        dependency_revisions={name: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            for name, path in (("ggml", ROOT / "ExternalProjects/ggml"), ("llama", ROOT.parent / "llama.cpp"))},
        dependency_tracked_changes={name: subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
            for name, path in (("ggml", ROOT / "ExternalProjects/ggml"), ("llama", ROOT.parent / "llama.cpp"))},
        limitations=["Synthetic inputs, not a retrieval leaderboard; recorded device/models/modalities only.",
            "HF default image sizing and audio log(mel+.001) differ from llama.cpp; controlled image fixtures match pixels/token geometry.",
            "The explicit candidate Video: prefix reproduces llama.cpp's automatic prefix when requested; a direct TensorSharp reference receives it only through the separate reference flag. Default video policy remains separate.",
            "Audio cross-engine quality can fail from frontend policy differences; independent HF audio oracle is separate.",
            "Sequential owned servers avoid concurrent engine execution; every graceful process exit must be zero."])
    destination = args.output / "results.json"

    def save():
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    def run(name, command, url):
        record = result["engines"][name] = dict(command=command, url=url, benchmarks={}, completed=False, memory_snapshots=[])
        if args.capture_boundary_telemetry:
            from embeddinggemma2_boundary_telemetry import capture_boundary
            record["environment_audit"] = audit_child_environment(command, os.environ).metadata()
            record["boundary_telemetry_before"] = capture_boundary()
        with (args.output / (name + ".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                started = time.monotonic()
                while time.monotonic() - started < 180:
                    if process.poll() is not None:
                        raise RuntimeError(name + " exited at startup")
                    try:
                        status, models = bench.request(url, "/v1/models")
                        if status == 200:
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(.25)
                else:
                    raise TimeoutError(name + " startup")
                record["memory_snapshots"].append(dict(phase="ready", **bench.process_memory(process.pid)))
                model = models["data"][0]["id"]
                record["model_id"] = model
                if name == "tensorsharp":
                    status, discovery = bench.request(url, "/api/models")
                    assert status == 200 and discovery["loadedBackend"] == args.expected_backend, discovery
                    record["discovery"] = discovery
                prepared = []
                for case in cases:
                    parts = controlled_parts(case["parts"], name, args.controlled_video_prefix,
                                             args.reference_controlled_video_prefix)
                    request = json.loads(media.payload(dict(parts=parts), args.cases.parent, model))
                    prepared.append((case["name"], request["input"], parts))

                def embed(inputs):
                    vectors, usage = bench.embedding(url, model, inputs)
                    bench.assert_vectors(vectors, 768)
                    assert len(vectors) == 1 and usage["prompt_tokens"] > 0
                    return vectors[0], usage["prompt_tokens"]

                started = time.monotonic()
                prewarm_calls = 0
                while time.monotonic() - started < args.prewarm_seconds:
                    embed(prepared[0][1])
                    prewarm_calls += 1
                record["prewarm"] = dict(calls=prewarm_calls, seconds=time.monotonic() - started)
                record["memory_snapshots"].append(dict(phase="after-prewarm", **bench.process_memory(process.pid)))
                print(name, "ready", flush=True)
                for label, inputs, parts in prepared:
                    for _ in range(args.warmup):
                        embed(inputs)
                    samples = []
                    started = time.monotonic()
                    while len(samples) < args.repeats or time.monotonic() - started < args.minimum_measure_seconds:
                        sample_start = time.perf_counter()
                        vector, tokens = embed(inputs)
                        samples.append((time.perf_counter() - sample_start) * 1000)
                    record["benchmarks"][label] = dict(parts=parts, embedding=vector, prompt_tokens=tokens,
                        samples_ms=samples, sample_count=len(samples), median_ms=statistics.median(samples),
                        p95_ms=sorted(samples)[min(len(samples) - 1, math.ceil(.95 * len(samples)) - 1)])
                    record["memory_snapshots"].append(dict(phase="after-" + label, **bench.process_memory(process.pid)))
                    print(name, label, round(statistics.median(samples), 3), "ms", tokens, "tokens", flush=True)
                    save()
                record["completed"] = True
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
                if args.capture_boundary_telemetry:
                    record["boundary_telemetry_after"] = capture_boundary()
                save()
                if process.returncode != 0:
                    raise RuntimeError(name + " failed graceful shutdown: " + str(process.returncode))

    try:
        engines = [("llama", args.llama_command, args.llama_url), ("tensorsharp", args.tensorsharp_command, args.tensorsharp_url)]
        if args.tensorsharp_first:
            engines.reverse()
        for engine in engines:
            run(*engine)
        comparisons = []
        for case in cases:
            a = result["engines"]["tensorsharp"]["benchmarks"][case["name"]]
            b = result["engines"]["llama"]["benchmarks"][case["name"]]
            similarity = bench.cosine(a["embedding"], b["embedding"])
            ratio = a["median_ms"] / b["median_ms"]
            comparisons.append(dict(name=case["name"], cosine=similarity,
                same_token_count=a["prompt_tokens"] == b["prompt_tokens"], latency_ratio_tensorsharp_over_llama=ratio,
                performance_pass=ratio <= args.max_latency_ratio, quality_pass=similarity >= args.min_cosine))
        result["comparison"] = comparisons
        result["completed"] = True
        result["performance_pass"] = all(case["performance_pass"] and case["same_token_count"] for case in comparisons)
        result["quality_pass"] = all(case["quality_pass"] and case["same_token_count"] for case in comparisons)
        result["passed"] = result["performance_pass"] and result["quality_pass"]
        print(json.dumps(comparisons, indent=2), flush=True)
    finally:
        # Bind every comparison to the exact immutable runtime it measured.
        # Capture after both owned servers exit, outside all timed intervals.
        result["binary_artifacts_after"] = {name: command_artifacts(bench.command_artifacts, command) for name, command in
            (("tensorsharp", args.tensorsharp_command), ("llama", args.llama_command))}
        result["binary_artifacts_unchanged"] = result["binary_artifacts"] == result["binary_artifacts_after"]
        result["dependency_tracked_changes_after"] = {name: subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
            for name, path in (("ggml", ROOT / "ExternalProjects/ggml"), ("llama", ROOT.parent / "llama.cpp"))}
        result["dependency_revisions_after"] = {name: subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            for name, path in (("ggml", ROOT / "ExternalProjects/ggml"), ("llama", ROOT.parent / "llama.cpp"))}
        result["dependencies_unchanged_and_clean"] = result["dependency_revisions"] == result["dependency_revisions_after"] and all(
            not value for value in (*result["dependency_tracked_changes"].values(), *result["dependency_tracked_changes_after"].values()))
        if not result["binary_artifacts_unchanged"] or not result["dependencies_unchanged_and_clean"]:
            result["passed"] = False
            result["provenance_error"] = "Runtime artifacts changed or an upstream dependency has tracked changes"
        save()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
