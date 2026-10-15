#!/usr/bin/env python3
"""Compare sequential TensorSharp/llama HTTP embedding requests from an ordered media case manifest."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request


def payload(case, directory, model=None):
    content = []
    for part in case["parts"]:
        kind, value = part["kind"], part["value"]
        if kind == "text":
            content.append({"type": "text", "text": value})
            continue
        path = (directory / value).resolve()
        encoded = base64.b64encode(path.read_bytes()).decode()
        if kind == "image":
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        elif kind == "audio":
            content.append({"type": "input_audio", "input_audio": {"data": encoded, "format": path.suffix.lstrip(".")}})
        elif kind == "video":
            content.append({"type": "input_video", "input_video": {"url": f"data:video/mp4;base64,{encoded}"}})
        else:
            raise ValueError(f"Unsupported media kind {kind}")
    data = {"input": [{"content": content}], "encoding_format": "float"}
    if model is not None: data["model"] = model
    return json.dumps(data).encode()


def discover_model(url):
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=30) as response:
        return json.load(response)["data"][0]["id"]


def request(url, body):
    started = time.perf_counter()
    req = urllib.request.Request(url.rstrip("/") + "/v1/embeddings", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url}: HTTP{exc.code}: {exc.read().decode()}") from exc
    elapsed = (time.perf_counter() - started) * 1000
    vector = data["data"][0]["embedding"]
    norm = math.sqrt(sum(x * x for x in vector))
    if len(vector) != 768 or not all(map(math.isfinite, vector)) or abs(norm - 1) > 1e-5:
        raise ValueError(f"{url}: invalid768-dimensional normalized embedding, norm={norm}")
    return {"embedding": vector, "prompt_tokens": data["usage"]["prompt_tokens"], "norm": norm, "elapsed_ms": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    candidate_group = parser.add_mutually_exclusive_group()
    candidate_group.add_argument("--tensorsharp-url")
    candidate_group.add_argument("--tensorsharp-probe", type=Path)
    parser.add_argument("--llama-url", required=True)
    parser.add_argument("--tensorsharp-model")
    parser.add_argument("--llama-model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="media-comparison")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-cosine", type=float, default=0)
    parser.add_argument("--require-same-tokens", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be nonnegative and repeats must be positive")
    manifest = json.loads(args.cases.read_text())
    probe = json.loads(args.tensorsharp_probe.read_text()) if args.tensorsharp_probe else None
    if probe is not None and not probe.get("completed"):
        raise ValueError("Candidate probe must have completed successfully, including model cleanup")
    captured = {case["name"]: case for case in probe["cases"]} if probe else {}
    urls = {"llama": args.llama_url}
    if args.tensorsharp_url: urls = {"tensorsharp": args.tensorsharp_url, **urls}
    models = {name: (args.tensorsharp_model if name == "tensorsharp" else args.llama_model) or discover_model(url)
              for name, url in urls.items()}
    report = {"label": args.label, "manifest": str(args.cases.resolve()),
              "manifest_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
              "fixture_flavor": manifest.get("fixture_flavor"), "warmup": args.warmup, "repeats": args.repeats,
              "tensorsharp_url": args.tensorsharp_url, "llama_url": args.llama_url,
              "candidate_probe": str(args.tensorsharp_probe.resolve()) if args.tensorsharp_probe else None,
              "candidate_probe_sha256": hashlib.sha256(args.tensorsharp_probe.read_bytes()).hexdigest() if args.tensorsharp_probe else None,
              "models": models, "cases": [],
              "limitations": ["Synthetic media are numerical/behavior regression checks, not a retrieval benchmark.",
                               "Default image/video frontend policies differ; record token counts and server flags when interpreting timings.",
                               "Requests execute sequentially. Server decoding/preprocessing/inference/serialization are timed; client file reads/base64 construction are excluded."]}
    failures = []
    try:
        for case in manifest["cases"]:
            bodies = {name: payload(case, args.cases.parent, models[name]) for name in urls}
            for _ in range(args.warmup):
                for name, url in urls.items():
                    request(url, bodies[name])
            results, samples = {}, {name: [] for name in urls}
            if probe:
                candidate = captured[case["name"]]
                results["tensorsharp"] = {key: candidate[key] for key in ("embedding", "prompt_tokens", "norm")}
            for repeat in range(args.repeats):
                names = list(urls) if repeat % 2 == 0 else list(reversed(urls))
                for name in names:
                    results[name] = request(urls[name], bodies[name])
                    samples[name].append(results[name]["elapsed_ms"])
            cosine = equal_tokens = None
            if "tensorsharp" in results:
                left, right = results["tensorsharp"]["embedding"], results["llama"]["embedding"]
                cosine = sum(a * b for a, b in zip(left, right)) / (results["tensorsharp"]["norm"] * results["llama"]["norm"])
                equal_tokens = results["tensorsharp"]["prompt_tokens"] == results["llama"]["prompt_tokens"]
            for name in urls:
                results[name].update(samples_ms=samples[name], median_ms=statistics.median(samples[name]))
            entry = {"name": case["name"], "cosine": cosine, "same_token_count": equal_tokens,
                     "llama_over_tensorsharp_time": results["llama"]["median_ms"] / results["tensorsharp"]["median_ms"] if args.tensorsharp_url else None, **results}
            report["cases"].append(entry)
            if cosine is not None and (cosine < args.min_cosine or (args.require_same_tokens and not equal_tokens)):
                failures.append(case["name"])
            print(f"{case['name']}: cosine={cosine}, llama_tokens={results['llama']['prompt_tokens']}, same_tokens={equal_tokens}, HTTP_speedup={entry['llama_over_tensorsharp_time']}", flush=True)
    finally:
        report["failures"] = failures
        report["completed"] = len(report["cases"]) == len(manifest["cases"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit("Quality/token comparison failed for: " + ", ".join(failures))


if __name__ == "__main__":
    main()
