#!/usr/bin/env python3
"""EmbeddingGemma 2 text HTTP parity, retrieval regression, and latency gates.

Uses the existing sequential EmbeddingBench runner and only the standard library.
All runner arguments are supported; model-specific options are documented by
--help. Generated reports belong in ignored docs/validation/ or artifacts/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddinggemma2_evidence import command_artifacts
TEXTS = [
    "task: code retrieval | query: find the function that adds two numbers",
    "title: add.py | text: def add(a, b):\n    return a + b",
    "title: read_file.py | text: def read_file(path):\n    with open(path) as f:\n        return f.read()",
    "task: code retrieval | query: read a text file from disk",
    "task: sentence similarity | query: The cat sits on the mat.",
    "task: sentence similarity | query: A kitten rests on a rug.",
    "task: sentence similarity | query: The stock exchange closed lower today.",
    "task: search result | query: 你好，世界！查找读取文件的函数。",
    "title: Unicode | text: Café naïve résumé — Straße ﬁ ﬃ ＡＢＣ e\u0301 😀",
    "title: whitespace | text: \t  white\nspace\r\n normalization  ",
    "task: question answering | query: What is the Snowflake Data Cloud?",
    "title: Snowflake | text: Snowflake is a cloud data platform for storing, analyzing, and sharing data.",
    "title: Mexico | text: Mexico City is the capital city of Mexico.",
]


def git_snapshot(path: Path):
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(path), *arguments], text=True).strip()
    return dict(path=str(path.resolve()), revision=git("rev-parse", "HEAD"),
                tracked_changes=git("status", "--porcelain", "--untracked-files=no"))


def build_options(path: Path):
    if not path.is_file():
        return {"path": str(path), "available": False}
    options = {}
    for line in path.read_text().splitlines():
        if line.startswith(("GGML_", "TENSORSHARP_GGML_", "CMAKE_BUILD_TYPE:")) and "=" in line:
            name, value = line.split("=", 1)
            options[name] = value
    return {"path": str(path.resolve()), "available": True, "options": options}


def add_model_correctness(bench, result, url, model, is_tensorsharp, token_cases,
                          measure_reference_after_quality_failure=False, expected_backend=None):
    """Extend strict checks; explicit reference diagnostics still retain failure."""
    assert result["dimensions"] == 768, result["dimensions"]
    raw_checks = []
    for item in token_cases:
        if not item["text"]:
            continue  # OpenAI explicitly rejects an empty text input.
        text_vector, text_usage = bench.embedding(url, model, item["text"])
        raw_vector, raw_usage = bench.embedding(url, model, item["tokens"])
        similarity = bench.cosine(text_vector[0], raw_vector[0])
        bench.numerical_gate(result["numerical_checks"], "raw_token_" + str(len(raw_checks)), similarity > .99999,
                             dict(text=item["text"], cosine=similarity, threshold=.99999), (item["text"], similarity),
                             is_tensorsharp, measure_reference_after_quality_failure)
        assert text_usage["prompt_tokens"] == raw_usage["prompt_tokens"] == len(item["tokens"]), item
        raw_checks.append(dict(text=item["text"], tokens=len(item["tokens"]), cosine=similarity,
                               threshold=.99999, passed=similarity > .99999))
    result["raw_token_parity"] = raw_checks
    result["quality_pass"] = all(check["passed"] for check in result["numerical_checks"].values())
    if is_tensorsharp:
        status, information = bench.request(url, "/api/models")
        assert status == 200, information
        result["actual_backend"] = information["loadedBackend"]
        if expected_backend:
            assert result["actual_backend"] == expected_backend, information
        result["matryoshka_dimensions"] = []
        for dimension in (128, 256, 512, 768):
            vector, _ = bench.embedding(url, model, TEXTS[0], dimensions=dimension)
            bench.assert_vectors(vector, dimension)
            similarity = bench.cosine(vector[0], result["single_vectors"][0][:dimension])
            assert similarity > .99999, (dimension, similarity)
            result["matryoshka_dimensions"].append(dict(dimensions=dimension, cosine=similarity))
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ggml-checkout", type=Path, default=ROOT / "ExternalProjects/ggml")
    parser.add_argument("--llama-checkout", type=Path, default=ROOT.parent / "llama.cpp")
    parser.add_argument("--llama-build-dir", type=Path,
                        help="Directory containing llama.cpp's actual CMakeCache.txt")
    parser.add_argument("--tensorsharp-build-dir", type=Path,
                        help="Directory containing the actual native library's CMakeCache.txt when built outside this source tree")
    parser.add_argument("--token-fixture", type=Path,
                        default=ROOT / "InferenceWeb.Tests/Fixtures/EmbeddingTokenizer/embeddinggemma2-tokenization.json")
    parser.add_argument("--include-full-context", action="store_true",
                        help="Add an exact 8192-token raw-input parity and performance case")
    parser.add_argument("--environment-note", action="append", default=[],
                        help="Record material local benchmark conditions, such as resident services")
    parser.add_argument("--expected-tensorsharp-backend",
                        help="Require /api/models loadedBackend to match this actual implementation")
    parser.add_argument("--help", action="store_true")
    local, forwarded = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location("embedding_bench", ROOT / "benchmarks/EmbeddingBench/embedding_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    original_artifacts = bench.command_artifacts
    bench.command_artifacts = lambda command: command_artifacts(original_artifacts, command)
    if local.help:
        parser.description = __doc__
        parser.print_help()
        print("\nShared runner options:")
        sys.argv = [sys.argv[0], "--help"]
        bench.main()
        return
    fixture = json.loads(local.token_fixture.read_text())
    if fixture["architecture"] != "gemma-embedding2":
        parser.error("token fixture must describe gemma-embedding2")
    token_cases = fixture["cases"]
    if not token_cases or any(not item.get("tokens") or not isinstance(item.get("text"), str) for item in token_cases):
        parser.error("token fixture must contain nonempty text/token cases")
    seed = next(item["tokens"] for item in token_cases if item["text"] == TEXTS[1])
    bos, eos = fixture["bos_token_id"], fixture["eos_token_id"]
    content = seed[1:-1] if seed[0] == bos and seed[-1] == eos else seed
    if not content:
        parser.error("token fixture seed contains no content tokens")

    def raw_input(length):
        return [bos] + [content[i % len(content)] for i in range(length - 2)] + [eos]

    lengths = [511, 512, 513, 1023, 1024, 1025, 2048]
    if local.include_full_context:
        lengths.append(8192)
    extra = {f"tokens_{length}": [raw_input(length)] for length in lengths}
    extra["batch_mixed_swa"] = [raw_input(length) for length in (31, 511, 513, 1025)]
    bench.TEXTS = TEXTS
    bench.EXTRA_SCENARIOS = extra
    original_correctness = bench.correctness

    def correctness(url, model, is_tensorsharp, reference_min_batch_cosine, reference_min_duplicate_cosine=.99999,
                    measure_reference_after_quality_failure=False):
        result = original_correctness(url, model, is_tensorsharp, reference_min_batch_cosine, reference_min_duplicate_cosine,
                                      measure_reference_after_quality_failure)
        return add_model_correctness(bench, result, url, model, is_tensorsharp, token_cases,
                                     measure_reference_after_quality_failure, local.expected_tensorsharp_backend)

    bench.correctness = correctness
    before = {"ggml": git_snapshot(local.ggml_checkout), "llama": git_snapshot(local.llama_checkout)}
    if before["ggml"]["tracked_changes"]:
        raise RuntimeError("TensorSharp ggml dependency has tracked changes; native validation requires unchanged upstream sources")
    options = {
        "tensorsharp": build_options((local.tensorsharp_build_dir or ROOT / "TensorSharp.GGML.Native/build") / "CMakeCache.txt"),
        "llama": build_options((local.llama_build_dir or local.llama_checkout / "build") / "CMakeCache.txt"),
    }
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    selected, _ = output_parser.parse_known_args(forwarded)
    destination = selected.output / "results.json"
    sys.argv = [sys.argv[0], *forwarded]
    try:
        bench.main()
    finally:
        after = {"ggml": git_snapshot(local.ggml_checkout), "llama": git_snapshot(local.llama_checkout)}
        if destination.is_file():
            report = json.loads(destination.read_text())
            report["dependencies"] = {"before": before, "after": after,
                                      "unchanged_during_validation": before == after, "build_options": options}
            report["coverage"] = {
                "model_family": "EmbeddingGemma 2", "modalities_tested": ["text"],
                "modalities_not_tested": ["image", "audio", "video"],
                "exact_raw_token_cases_available": lengths,
                "fixture": str(local.token_fixture.resolve()),
                "environment_notes": local.environment_note,
                "limitations": ["Small task-prefixed retrieval regression corpus; no MTEB or other quality leaderboard was run.",
                                "Only the model file, backend, device, and benchmark cases recorded in this report were tested.",
                                "HTTP steady-state timings include tokenization and serialization; startup is excluded.",
                                "Available cases omitted by --cases are untested performance scenarios."],
            }
            destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if before != after:
            raise RuntimeError("Dependency sources changed during validation; report cannot establish a stable reference")


if __name__ == "__main__":
    main()
