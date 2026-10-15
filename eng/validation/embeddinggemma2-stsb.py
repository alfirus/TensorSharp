#!/usr/bin/env python3
"""STS-B development quality comparison with original human gold scores.

Starts HTTP engines sequentially using JSON command arrays. Uses the Python
standard library, records original GLUE archive/TSV hashes and all inputs/scores,
and reports Spearman correlation at 768/512/256/128 dimensions. Supports the
complete 1500-pair development split or an ordered subset. This is a narrow
English similarity regression, not a test-split, MTEB, or multimodal score.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddinggemma2_evidence import command_artifacts
DATA_URL = "https://dl.fbaipublicfiles.com/glue/data/STS-B.zip"
ARCHIVE_SHA256 = "e60a6393de5a8b5b9bac5020a1554b54e3691f9d600b775bd131e613ac179c85"
TSV_SHA256 = "fe7dc0632a5033562f3cadff991f6e8d9b9b6311861660e78c3c82eb7db67f59"
PREFIX = "task: sentence similarity | query: "
DEVELOPMENT_ROWS = 1500
DEVELOPMENT_COLUMNS = ["index", "genre", "filename", "year", "old_index", "source1", "source2",
                       "sentence1", "sentence2", "score"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    first = 0
    while first < len(order):
        last = first + 1
        while last < len(order) and values[order[last]] == values[order[first]]:
            last += 1
        rank = (first + 1 + last) / 2
        for index in order[first:last]:
            result[index] = rank
        first = last
    return result


def spearman(left, right):
    left, right = ranks(left), ranks(right)
    lmean, rmean = statistics.mean(left), statistics.mean(right)
    return sum((a-lmean)*(b-rmean) for a, b in zip(left, right)) / math.sqrt(
        sum((a-lmean)**2 for a in left) * sum((b-rmean)**2 for b in right))


def read_development_pairs(tsv, expected_rows=DEVELOPMENT_ROWS):
    # GLUE's original TSV uses literal quotation marks inside sentence fields,
    # including an unmatched opening quote at development index 764. CSV quote
    # handling would absorb later physical rows into that sentence and lose gold
    # scores. Tabs and physical lines are the source's only field/row delimiters.
    reader = csv.DictReader(io.StringIO(tsv.decode("utf-8"), newline=""),
                            delimiter="\t", quoting=csv.QUOTE_NONE)
    if reader.fieldnames != DEVELOPMENT_COLUMNS:
        raise ValueError("STS-B development TSV has an unexpected header")
    pairs = []
    for row in reader:
        line = reader.line_num
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed STS-B development TSV line {line}: expected {len(DEVELOPMENT_COLUMNS)} tab-separated fields")
        try:
            score = float(row["score"])
        except ValueError as error:
            raise ValueError(f"Malformed STS-B development TSV line {line}: invalid gold score {row['score']!r}") from error
        if not math.isfinite(score) or not 0 <= score <= 5:
            raise ValueError(f"Malformed STS-B development TSV line {line}: gold score must be finite and in 0..5")
        if not row["sentence1"] or not row["sentence2"]:
            raise ValueError(f"Malformed STS-B development TSV line {line}: both sentences are required")
        pairs.append(row)
    if len(pairs) != expected_rows:
        raise ValueError(f"STS-B development TSV has {len(pairs)} complete rows; expected {expected_rows}")
    return pairs


def selection_description(selected, total):
    if selected == total:
        return f"complete development split ({total} pairs)"
    return f"first {selected} rows in original file order"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorsharp-command", type=json.loads, required=True)
    parser.add_argument("--llama-command", type=json.loads, required=True)
    parser.add_argument("--tensorsharp-url", default="http://127.0.0.1:18380")
    parser.add_argument("--llama-url", default="http://127.0.0.1:18381")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-spearman-drop", type=float, default=.001)
    parser.add_argument("--tensorsharp-first", action="store_true")
    parser.add_argument("--expected-tensorsharp-backend")
    parser.add_argument("--environment-note", action="append", default=[])
    args = parser.parse_args()
    if not 2 <= args.pairs <= 1500 or not 1 <= args.batch_size <= 32 or args.max_spearman_drop < 0:
        parser.error("pairs must be 2..1500; batch-size 1..32; max-spearman-drop nonnegative")
    if not args.archive.is_file():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            args.archive.write_bytes(response.read())
    if sha256(args.archive) != ARCHIVE_SHA256:
        raise ValueError("STS-B archive differs from the recorded official GLUE source")
    with zipfile.ZipFile(args.archive) as archive:
        tsv = archive.read("STS-B/dev.tsv")
    if hashlib.sha256(tsv).hexdigest() != TSV_SHA256:
        raise ValueError("STS-B development TSV differs from the recorded source")
    all_pairs = read_development_pairs(tsv)
    pairs = all_pairs[:args.pairs]
    gold = [float(item["score"]) for item in pairs]
    texts = [PREFIX + item[key] for item in pairs for key in ("sentence1", "sentence2")]
    spec = importlib.util.spec_from_file_location("embedding_bench", ROOT / "benchmarks/EmbeddingBench/embedding_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    bench.KEEP_ALIVE = True
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "quality.json"
    complete_development = len(pairs) == len(all_pairs)
    result = dict(dataset="GLUE STS-B", split="development", selection=selection_description(len(pairs), len(all_pairs)),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_url=DATA_URL, archive_sha256=ARCHIVE_SHA256, dev_tsv_sha256=TSV_SHA256,
        total_development_pairs=len(all_pairs), selected_development_pairs=len(pairs),
        development_split_complete=complete_development, pairs=pairs, prompt_prefix=PREFIX,
        model_file=str(args.model_file.resolve()), model_sha256=sha256(args.model_file),
        max_spearman_drop=args.max_spearman_drop, batch_size=args.batch_size, engines={}, server_exit_codes={},
        environment_notes=args.environment_note,
        binary_artifacts={"tensorsharp": command_artifacts(bench.command_artifacts, args.tensorsharp_command),
                          "llama": command_artifacts(bench.command_artifacts, args.llama_command)},
        limitations=["Complete original English development split; the STS-B test split and MTEB were not evaluated."
                     if complete_development else "Only the original first selected English development pairs; no random sampling or full development/test split claim.",
                     "MRL dimensions are computed from the same full vectors by slicing and cosine renormalization.",
                     "No multimodal, multilingual, code retrieval, or model/device outside this report was evaluated."])

    def save():
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    engines = [("llama", args.llama_command, args.llama_url), ("tensorsharp", args.tensorsharp_command, args.tensorsharp_url)]
    if args.tensorsharp_first:
        engines.reverse()
    try:
        for name, command, url in engines:
            with (args.output / (name + ".log")).open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    started = time.monotonic()
                    while time.monotonic() - started < 180:
                        if process.poll() is not None:
                            raise RuntimeError(name + " exited; see its log")
                        try:
                            if bench.request(url, "/v1/models")[0] == 200:
                                break
                        except (OSError, ValueError):
                            pass
                        time.sleep(.25)
                    else:
                        raise TimeoutError(name + " startup")
                    actual_backend = None
                    if name == "tensorsharp":
                        status, information = bench.request(url, "/api/models")
                        if status != 200:
                            raise ValueError("TensorSharp backend discovery failed")
                        actual_backend = information["loadedBackend"]
                        if args.expected_tensorsharp_backend and actual_backend != args.expected_tensorsharp_backend:
                            raise ValueError("The server loaded a different backend implementation: " + str(actual_backend))
                    vectors, tokens = [], 0
                    for first in range(0, len(texts), args.batch_size):
                        rows, usage = bench.embedding(url, args.model, texts[first:first+args.batch_size])
                        bench.assert_vectors(rows, 768)
                        vectors.extend(rows)
                        tokens += usage["prompt_tokens"]
                    if len(vectors) != len(texts):
                        raise ValueError("The engine returned the wrong number of vectors")
                    scores = {str(dim): [bench.cosine(vectors[2*i][:dim], vectors[2*i+1][:dim]) for i in range(len(pairs))]
                              for dim in (768, 512, 256, 128)}
                    result["engines"][name] = dict(command=command, url=url, actual_backend=actual_backend, prompt_tokens=tokens, vectors=vectors,
                        scores=scores, spearman={dim: spearman(gold, values) for dim, values in scores.items()})
                    print(name, result["engines"][name]["spearman"], flush=True)
                    save()
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
                    result["server_exit_codes"][name] = process.returncode
        ts, llama = result["engines"]["tensorsharp"], result["engines"]["llama"]
        if ts["prompt_tokens"] != llama["prompt_tokens"]:
            raise ValueError("Engine token accounting differs")
        result["comparison"] = {dim: dict(tensorsharp=ts["spearman"][dim], llama=llama["spearman"][dim],
            difference=ts["spearman"][dim]-llama["spearman"][dim],
            passed=ts["spearman"][dim] >= llama["spearman"][dim]-args.max_spearman_drop)
            for dim in ts["spearman"]}
        result["binary_artifacts_after"] = {
            "tensorsharp": command_artifacts(bench.command_artifacts, args.tensorsharp_command),
            "llama": command_artifacts(bench.command_artifacts, args.llama_command)}
        result["binary_artifacts_unchanged"] = result["binary_artifacts_after"] == result["binary_artifacts"]
        result["servers_exited_cleanly"] = all(code == 0 for code in result["server_exit_codes"].values())
        result["passed"] = (all(row["passed"] for row in result["comparison"].values())
                            and result["binary_artifacts_unchanged"] and result["servers_exited_cleanly"])
        print(json.dumps(result["comparison"], indent=2), flush=True)
        return 0 if result["passed"] else 1
    finally:
        save()


if __name__ == "__main__":
    raise SystemExit(main())
