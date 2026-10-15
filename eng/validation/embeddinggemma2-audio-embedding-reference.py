#!/usr/bin/env python3
"""Independent HF audio soft rows through the complete NumPy text encoder.

Combines the exact vocabulary BOS/audio-start/audio-end/EOS rows, optionally
an already verified tokenizer fixture prefix, and unscaled HF NumPy audio
rows. Saved actual API/probe vectors can be checked without executing them.
All generated evidence belongs under ignored docs/validation/ or artifacts/.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-fixture", type=Path)
    parser.add_argument("--prefix-case", type=int, default=0)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--case", default="audio")
    parser.add_argument("--min-cosine", type=float, default=.9999)
    parser.add_argument("--max-absolute-error", type=float, default=.002)
    args = parser.parse_args()
    encoder_path = Path(__file__).resolve().parents[1] / "embeddinggemma2-reference.py"
    spec = importlib.util.spec_from_file_location("embedding_reference", encoder_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    model = reference.Encoder(args.model)
    model_hash = digest(args.model)
    audio = json.loads(args.audio_oracle.read_text())
    if audio.get("past_distance_inclusive") != 11 or audio.get("audio_log_policy", "huggingface_add") != "huggingface_add":
        raise ValueError("The pinned HF source uses exclusive-left window12 (maximum distance11) and the log-add default; earlier inclusive12 or clamp diagnostics are not the canonical HF oracle")
    soft = np.asarray(audio["soft_rows"], dtype=np.float32)
    if list(soft.shape) != audio["shape"] or soft.ndim != 2 or soft.shape[1] != model.dim or not np.all(np.isfinite(soft)):
        raise ValueError("Require complete finite independent audio soft rows")
    vocabulary = model.metadata("tokenizer.ggml.tokens")
    bos = int(model.metadata("tokenizer.ggml.bos_token_id"))
    eos = int(model.metadata("tokenizer.ggml.eos_token_id"))
    audio_start, audio_end = vocabulary.index("<|audio>"), vocabulary.index("<audio|>")
    prefix, prefix_text, prefix_hash = [], "", None
    if args.prefix_fixture:
        fixture = json.loads(args.prefix_fixture.read_text())
        if fixture["architecture"] != "gemma-embedding2" or fixture["model_sha256"] != model_hash:
            raise ValueError("Verified prefix fixture must match this exact text model")
        case = fixture["cases"][args.prefix_case]
        prefix = case["tokens"]
        if prefix[0] != bos or prefix[-1] != eos:
            raise ValueError("Verified prefix fixture must carry the model BOS and EOS")
        prefix, prefix_text, prefix_hash = prefix[1:-1], case["text"], digest(args.prefix_fixture)
    leading = [bos, *prefix, audio_start]
    trailing = [audio_end, eos]
    inputs = np.concatenate((model.token_rows(leading), soft, model.token_rows(trailing)), axis=0)
    expected = model.encode_inputs(inputs)
    norm = math.sqrt(sum(float(value) ** 2 for value in expected))
    if not np.all(np.isfinite(expected)) or abs(norm - 1) > 1e-5:
        raise ValueError("Expected a finite unit-normalized full audio embedding")
    comparisons = []
    for path in args.candidate:
        candidate = json.loads(path.read_text())
        if not candidate.get("completed"):
            raise ValueError("Actual candidate process/lifecycle must have completed")
        if candidate.get("audio_log_policy") == "LlamaClampFloor" or "TS_EMBEDDING_AUDIO_LOG_POLICY=llama_clamp" in candidate.get("command", []):
            raise ValueError("The default HF audio oracle requires the HuggingFaceAddFloor policy; controlled llama-clamp reports remain separate")
        candidate_model = candidate.get("model_sha256") or candidate.get("model", {}).get("sha256")
        if candidate_model != model_hash:
            raise ValueError("Candidate text model differs from this exact oracle")
        candidate_projector = candidate.get("projector_sha256") or candidate.get("projector", {}).get("sha256")
        if candidate_projector != audio["projector_sha256"]:
            raise ValueError("Candidate audio projector differs from this exact oracle")
        row = next(case for case in candidate["cases"] if case["name"] == args.case)
        if "case_manifest" in candidate:
            manifest = Path(candidate["case_manifest"])
            if digest(manifest) != candidate["case_manifest_sha256"]:
                raise ValueError("Candidate input manifest changed after inference")
            parts = next(case for case in json.loads(manifest.read_text())["cases"] if case["name"] == args.case)["parts"]
            files = row["media_files"]
        else:
            parts, files = row["parts"], candidate["media"]
        if "".join(part["value"] for part in parts if part["kind"] == "text") != prefix_text or any(part["kind"] not in ("text", "audio") for part in parts):
            raise ValueError("Candidate prefix or media sequence differs from the full HF input")
        waves = [part["value"] for part in parts if part["kind"] == "audio"]
        if len(waves) != 1 or not any(Path(entry["path"]).name == Path(waves[0]).name and entry["sha256"] == audio["waveform_sha256"] for entry in files):
            raise ValueError("Candidate waveform differs from the full HF input")
        actual = row["embedding"]
        if len(actual) != len(expected) or not all(math.isfinite(value) for value in actual):
            raise ValueError("Candidate vector dimensions or values differ")
        count = row.get("prompt_tokens", row.get("tokens"))
        if count != inputs.shape[0]:
            raise ValueError("Candidate input row count differs from the full HF sequence")
        actual_norm = math.sqrt(sum(value * value for value in actual))
        cosine = sum(float(a) * b for a, b in zip(expected, actual)) / (norm * actual_norm)
        error = max(abs(float(a) - b) for a, b in zip(expected, actual))
        comparisons.append(dict(candidate=str(path.resolve()), candidate_sha256=digest(path),
            backend=candidate.get("backend", candidate.get("expected_backend")), case=args.case,
            cosine=cosine, max_absolute_error=error,
            passed=cosine >= args.min_cosine and error <= args.max_absolute_error))
    report = dict(model=str(args.model.resolve()), model_sha256=model_hash,
        encoder_source_sha256=digest(encoder_path), audio_oracle=str(args.audio_oracle.resolve()),
        audio_oracle_sha256=digest(args.audio_oracle), audio_source_revision=audio["source_revision"],
        projector_sha256=audio["projector_sha256"], waveform_sha256=audio["waveform_sha256"],
        prefix_text=prefix_text, prefix_fixture_sha256=prefix_hash,
        leading_token_ids=leading, trailing_token_ids=trailing, audio_rows=soft.shape[0],
        total_input_rows=inputs.shape[0], input_rows_sha256=hashlib.sha256(inputs.astype("<f4").tobytes()).hexdigest(),
        method="Independent HF NumPy audio frontend/full tower plus complete F32 NumPy bidirectional text encoder and learned projection",
        embedding=expected.tolist(), comparisons=comparisons, thresholds=dict(min_cosine=args.min_cosine, max_absolute_error=args.max_absolute_error),
        passed=all(row["passed"] for row in comparisons),
        limitations="This waveform/model/projector numerical end-to-end oracle; no speech retrieval leaderboard or latency claim. Earlier raw soft-row absolute error reports remain separate.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], total_input_rows=inputs.shape[0], comparisons=comparisons), indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
