#!/usr/bin/env python3
"""Recheck a frozen candidate against complete, hash-bound default-HF vectors.

No learned model math is executed. The original seven inputs, preprocessing
assemblies or fresh identical observations, numerical limits and runtime guards
remain mandatory. Original HTTP and oracle reports are never modified.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


reference = module("default_hf_saved_reference", Path(__file__).with_name("embeddinggemma2-default-hf-composite-reference.py"))
controlled = module("default_hf_saved_controlled_recheck", Path(__file__).with_name("embeddinggemma2-controlled-composite-recheck.py"))
sha = reference.sha


def validate_oracle(oracle):
    if (oracle.get("completed") is not True or oracle.get("passed") is not True or
            oracle.get("input_source_hashes_unchanged") is not True or
            oracle.get("before_hashes") != oracle.get("after_hashes") or not oracle.get("before_hashes") or
            oracle.get("fresh_unique_vision_inputs") != 6 or oracle.get("fresh_complete_text_cases") != 7 or
            oracle.get("thresholds") != dict(min_cosine=.9999, max_absolute_error=.002) or
            oracle.get("pixel_max_absolute_error") != 2e-6 or
            tuple(row["name"] for row in oracle.get("cases", [])) != reference.CASE_NAMES or
            set(oracle.get("source_hashes", {})) != set(reference.SOURCES) | {
                "eng/validation/embeddinggemma2-default-hf-composite-reference.py"}):
        raise ValueError("Require the successful complete original default-HF oracle and unchanged limits")
    for case in oracle["cases"]:
        expected = np.asarray(case["embedding"], dtype=np.float64)
        if (expected.shape != (768,) or not np.all(np.isfinite(expected)) or
                abs(float(np.linalg.norm(expected)) - 1) > 1e-5 or
                type(case["total_input_rows"]) is not int or case["total_input_rows"] <= 0 or
                not case.get("comparisons")):
            raise ValueError("Invalid preserved complete default-HF vector")
        for comparison in case["comparisons"]:
            if (set(comparison["protocols"]) != {"openai", "ollama"} or
                    any(row.get("passed") is not True for row in comparison["protocols"].values())):
                raise ValueError("Original complete protocol comparisons did not pass")


def matching_http_inputs(original, current):
    if (original["media"] != current["media"] or
            original["manifest"]["sha256"] != current["manifest"]["sha256"] or
            [(row["name"], row["parts"]) for row in original["cases"]] !=
            [(row["name"], row["parts"]) for row in current["cases"]]):
        raise ValueError("Candidate default-HF media, manifest or ordered parts differ")


def validate_fresh_proof(original, current):
    reference.validate_proof(current)
    controlled.equivalent_proof(original, current)
    for key in ("source_report_kind", "audio_log_policy", "default_options_observed", "manifest_sha256"):
        if original[key] != current[key]:
            raise ValueError("Fresh default-HF preprocessing policy differs")


def check_proof_buffers(proof, parent):
    # Default HF resizes its inputs. The controlled identity-size checker is
    # deliberately inapplicable; bind the original resized buffers instead.
    for image in proof["images"] + [frame for video in proof["videos"] for frame in video["frames"]]:
        sw, sh = image["source_width"], image["source_height"]
        pw, ph = image["processed_width"], image["processed_height"]
        if (min(sw, sh, pw, ph) <= 0 or pw % 48 or ph % 48 or
                image["soft_tokens"] != pw // 48 * (ph // 48)):
            raise ValueError("Invalid observed complete pooling canvas")
        rgba = reference.base.artifact(parent, image["rgba"])
        pixels = reference.base.artifact(parent, image["pixels"])
        if (sha(rgba) != image["rgba_sha256"] or sha(pixels) != image["pixels_sha256"] or
                rgba.stat().st_size != sw * sh * 4 or pixels.stat().st_size != pw * ph * 3 * 4 or
                not np.all(np.isfinite(np.fromfile(pixels, dtype="<f4")))):
            raise ValueError("Observed default-HF resized buffers differ")
    for video in proof["videos"]:
        reference.base.validate_video_indices(video)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("oracle", "model", "projector", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--oracle-sha256", required=True)
    parser.add_argument("--input-proof", type=Path,
        help="Required when preprocessing assemblies change; observations must equal the original")
    parser.add_argument("--http", type=Path, action="append", required=True)
    parser.add_argument("--runtime-guard", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior evidence; choose a new report")
    observed = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError("Hash-bound evidence changed: " + str(path))
        if str(path) in observed and observed[str(path)] != digest:
            raise ValueError("Evidence changed during validation: " + str(path))
        observed[str(path)] = digest
        return digest

    def read(path, expected=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("Parsed evidence differs from expected hash: " + str(path))
        if str(path) in observed and observed[str(path)] != digest:
            raise ValueError("Parsed evidence changed: " + str(path))
        observed[str(path)] = digest
        return json.loads(raw)

    oracle = read(args.oracle, args.oracle_sha256)
    validate_oracle(oracle)
    # This also retains hashes of the original successful runtime/proof/media
    # evidence. No changed source or stale parsed file is silently accepted.
    for path, digest in oracle["before_hashes"].items():
        bind(path, digest)
    for name, digest in oracle["source_hashes"].items():
        bind(ROOT / name, digest)
    model_hash, projector_hash = bind(args.model), bind(args.projector)
    if (model_hash, projector_hash) != (oracle["model_sha256"], oracle["projector_sha256"]):
        raise ValueError("Candidate weights differ from the complete default-HF oracle")
    record = oracle["inputs"]["proof"]
    original_proof_path = Path(record["path"])
    original_proof = read(original_proof_path, record["sha256"])
    reference.validate_proof(original_proof)
    check_proof_buffers(original_proof, original_proof_path.parent)
    proof_path = args.input_proof or original_proof_path
    proof = original_proof
    if args.input_proof:
        proof = read(proof_path)
        validate_fresh_proof(original_proof, proof)
        check_proof_buffers(proof, proof_path.parent)
    for image in proof["images"] + [frame for video in proof["videos"] for frame in video["frames"]]:
        for key in ("rgba", "pixels"):
            bind(reference.base.artifact(proof_path.parent, image[key]), image[key + "_sha256"])
    original_reports = {}
    for case in oracle["cases"]:
        for comparison in case["comparisons"]:
            path, digest = comparison["http"], comparison["sha256"]
            original = read(path, digest)
            reference.validate_http(original, original_proof, model_hash, projector_hash)
            if path in original_reports and original_reports[path] != original:
                raise ValueError("Inconsistent original HTTP evidence")
            original_reports[path] = original
    original = next(iter(original_reports.values()))
    if original_proof["source_report_sha256"] not in {
            observed[str(Path(path).resolve())] for path in original_reports}:
        raise ValueError("Original input proof is not bound to original HTTP")
    reports = [(path, read(path)) for path in args.http]
    guards = [(path, read(path)) for path in args.runtime_guard]
    if (len({str(path.resolve()) for path, _ in reports}) != len(reports) or
            len({value["expected_backend"] for _, value in reports}) != len(reports)):
        raise ValueError("Duplicate candidate HTTP/backend does not add coverage")
    if args.input_proof and not any(
            proof["source_report_sha256"] == observed[str(path.resolve())] and
            Path(proof["source_report_path"]).name == path.name and
            proof["manifest_path"] == value["manifest"]["path"]
            for path, value in reports):
        raise ValueError("Fresh input proof does not bind an actual candidate HTTP report")
    for _, value in reports:
        reference.validate_http(value, proof, model_hash, projector_hash)
        reference.validate_runtime_guards(value, [guard for _, guard in guards])
        matching_http_inputs(original, value)
    bind(__file__)
    bind(Path(__file__).with_name("embeddinggemma2-controlled-composite-recheck.py"))
    bind(Path(__file__).with_name("embeddinggemma2_http_environment.py"))
    before = dict(observed)
    for path, digest in before.items():
        bind(path, digest)
    result = dict(completed=False, passed=False,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method="Hash-bound reuse of complete independent default-HF seven-case oracle",
        fresh_model_math=0, thresholds=oracle["thresholds"],
        oracle=dict(path=str(args.oracle.resolve()), sha256=args.oracle_sha256),
        proof=dict(path=str(proof_path.resolve()), sha256=observed[str(proof_path.resolve())],
            fresh_preprocessing_observations=args.input_proof is not None),
        runtime_guards=[dict(path=str(path.resolve()), sha256=observed[str(path.resolve())]) for path, _ in guards],
        model_sha256=model_hash, projector_sha256=projector_hash,
        before_hashes=before, cases=[],
        limitations=["Saved seven opaque synthetic default-HF cases only; no fresh learned model computation or service timing.",
            "Preprocessing assembly changes require fresh identical observations; native changes still require complete runtime guards.",
            "Original tokenizer/resize/audio reference scope and failed experiments remain unchanged."])
    for path, current in reports:
        values = []
        for case, row in zip(oracle["cases"], current["cases"]):
            if case["name"] != row["name"]:
                raise ValueError("Complete case ordering differs")
            values.append(dict(name=case["name"], protocols=reference.compare(
                np.asarray(case["embedding"], dtype=np.float64), row, case["total_input_rows"])))
        result["cases"].append(dict(http=str(path.resolve()), sha256=observed[str(path.resolve())],
            backend=current["expected_backend"], comparisons=values))
    result["after_hashes"] = {path: sha(path) for path in before}
    result["input_source_hashes_unchanged"] = result["after_hashes"] == before
    result["passed"] = result["input_source_hashes_unchanged"] and all(
        row["passed"] for report in result["cases"] for case in report["comparisons"] for row in case["protocols"].values())
    result["completed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
