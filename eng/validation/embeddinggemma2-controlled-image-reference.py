#!/usr/bin/env python3
"""Independent full pooled IMAGE-only oracle for preserved controlled HTTP pairs.

Uses actual frozen production preprocessing evidence, independently verifies PNG
bytes/geometry/pixels, then runs the existing HF-derived NumPy full vision tower
and complete F32 text encoder. No audio/video or overall multimodal qualification.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
from PIL import Image


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_pair(pair, case_name, image_hash, model_hash, projector_hash):
    if not pair.get("completed") or pair["model"]["sha256"] != model_hash or pair["projector"]["sha256"] != projector_hash:
        raise ValueError("Require completed actual engines with the exact text/projector bytes")
    case = next(item for item in pair["inputs"] if item["name"] == case_name)
    if len(case["parts"]) != 1 or case["parts"][0]["kind"] != "image":
        raise ValueError("This tool requires an unprefixed IMAGE-only sequence")
    path = case["parts"][0]["value"]
    if not any(item["path"] == path and item["sha256"] == image_hash for item in pair["media"]):
        raise ValueError("Actual HTTP image identity differs from the independent input")
    for name in ("tensorsharp", "llama"):
        engine = pair["engines"][name]
        if not engine.get("completed") or engine.get("server_exit_code") != 0:
            raise ValueError(name + " did not complete with a zero graceful exit")
        if engine["benchmarks"][case_name]["parts"] != case["parts"]:
            raise ValueError(name + " actual image sequence differs from the manifest")
    return case


def preprocessor_for_pair(pair, proofs):
    models = [item["sha256"] for item in pair["binary_artifacts"]["tensorsharp"]
              if Path(item["path"]).name == "TensorSharp.Models.dll"]
    if len(models) != 1:
        raise ValueError("Require exactly one recorded Models assembly identity per pair")
    matches = [proof for proof in proofs if proof.get("completed") and proof["models_assembly_sha256"] == models[0]]
    if len(matches) != 1:
        raise ValueError("Each distinct actual Models assembly requires its own observed preprocessing proof")
    return matches[0]


def reused_pixel_hash(case):
    # Current reports bind one proof per actual Models assembly. Retain the
    # original single-proof schema for immutable earlier numerical oracles.
    proofs = case.get("preprocessing_proofs")
    if proofs is None:
        proof = case.get("preprocessing_proof")
        proofs = [proof] if proof is not None else []
    if not isinstance(proofs, list) or not proofs:
        raise ValueError("Reused numerical oracle has no observed pixel proof")
    hashes = [proof.get("actual_processed_pixels_sha256") for proof in proofs]
    if any(not isinstance(value, str) or not value for value in hashes) or len(set(hashes)) != 1:
        raise ValueError("Reused numerical oracle pixel proofs are missing or disagree")
    return hashes[0]


def evidence_resolver(mappings):
    """Explicitly relocate immutable evidence paths without rewriting proofs."""
    bindings = {}
    for mapping in mappings:
        source, separator, target = mapping.partition("=")
        source, target = Path(source), Path(target).resolve()
        if (not separator or not source.is_absolute() or ".." in source.parts or
                not target.is_dir() or source in bindings):
            raise ValueError("Require unique absolute evidence-prefix=existing-local-directory bindings")
        bindings[source] = target

    def resolve(recorded):
        path = Path(recorded)
        if not bindings:
            return path
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Rebound evidence must have absolute traversal-free recorded paths")
        matches = [prefix for prefix in bindings if path.is_relative_to(prefix)]
        if not matches:
            return path
        prefix = max(matches, key=lambda item: len(item.parts))
        target = bindings[prefix]
        result = (target / path.relative_to(prefix)).resolve()
        if not result.is_relative_to(target):
            raise ValueError("Rebound evidence alias escapes its explicitly bound directory")
        return result

    return resolve


def pixel_proof(record, hf, resolve_path=Path):
    path, rgba_path, pixels_path = [resolve_path(record[key]) for key in ("image", "rgba", "pixels")]
    if digest(path) != record["image_sha256"] or digest(rgba_path) != record["rgba_sha256"] or digest(pixels_path) != record["pixels_sha256"]:
        raise ValueError("Actual image preprocessing evidence changed")
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("This exact invariant-input proof is restricted to PNG fixtures")
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    height, width, _ = rgba.shape
    actual_rgba = np.fromfile(rgba_path, dtype=np.uint8).reshape(height, width, 4)
    if not np.array_equal(rgba, actual_rgba):
        raise ValueError("Production decoder and independent PIL decoded pixels differ")
    target_h, target_w = hf.get_aspect_ratio_preserving_size(height, width, 16, 280 * 9, 3)
    if ((width, height) != (record["source_width"], record["source_height"]) or
        (target_w, target_h) != (record["processed_width"], record["processed_height"]) or
        (target_w, target_h) != (width, height)):
        raise ValueError("Require actual and independently proved unchanged HF-policy geometry")
    if not np.all(rgba[:, :, 3] == 255):
        raise ValueError("This no-resampling input proof requires fully opaque fixtures")
    # At identity size, cubic weights at integer offsets are 0 except the unit
    # center. TS uses F32 division; preserve its exact all-pixel byte proof.
    rgb = rgba[:, :, :3].transpose(2, 0, 1).astype(np.float32)
    expected_ts = rgb / np.float32(255)
    canonical_hf = rgb * np.float32(1 / 255)
    pixels = np.fromfile(pixels_path, dtype="<f4").reshape(3, height, width)
    if not np.array_equal(pixels.view(np.uint32), expected_ts.view(np.uint32)):
        raise ValueError("The actual processor changed identity-size decoded pixel values")
    if record["soft_tokens"] != width // 48 * (height // 48):
        raise ValueError("Reported pooling geometry differs from actual image dimensions")
    return pixels, dict(width=width, height=height, actual_processed_width=record["processed_width"],
        actual_processed_height=record["processed_height"], hf_policy_width=target_w, hf_policy_height=target_h,
        decoded_rgba_equal=True, opaque=True, all_processed_f32_pixels_bit_equal_to_no_resize=True,
        image_sha256=record["image_sha256"], decoded_rgba_sha256=record["rgba_sha256"],
        actual_processed_pixels_sha256=record["pixels_sha256"],
        canonical_hf_pixels_sha256=hashlib.sha256(canonical_hf.astype("<f4").tobytes()).hexdigest(),
        hf_rescale_vs_ts_divide_max_absolute_error=float(np.max(np.abs(canonical_hf - pixels))),
        hf_rescale_vs_ts_divide_changed_components=int(np.count_nonzero(canonical_hf.view(np.uint32) != pixels.view(np.uint32))),
        soft_tokens=record["soft_tokens"], normalization="Identity statistics. TS F32 division by255 versus HF F32 multiplication by(1/255) recorded separately.")


def compare(expected, actual, count, expected_count, minimum, maximum):
    actual = np.asarray(actual, dtype=np.float64)
    left = np.asarray(expected, dtype=np.float64)
    if actual.shape != left.shape or not np.all(np.isfinite(actual)) or count != expected_count:
        raise ValueError("Actual image embedding dimension, finite values or input rows differ")
    if abs(float(np.linalg.norm(actual)) - 1) > 1e-5:
        raise ValueError("Actual image embedding is not normalized")
    cosine = float(left @ actual / (np.linalg.norm(left) * np.linalg.norm(actual)))
    error = float(np.max(np.abs(left - actual)))
    return dict(cosine=cosine, max_absolute_error=error, passed=cosine >= minimum and error <= maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "projector", "hf-source-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, action="append", required=True)
    parser.add_argument("--existing-oracle", type=Path, help="Reuse preserved completed full-model math only when graph sources, original weights, complete inputs and pixels still match exactly")
    parser.add_argument("--image", action="append", required=True, help="Saved HTTP case name=local PNG path")
    parser.add_argument("--pair", action="append", type=Path, required=True)
    parser.add_argument("--evidence-prefix", action="append", default=[],
                        help="Explicit recorded absolute prefix=local directory for copied PNG/RGBA/pixel evidence; original proofs and SHA gates stay intact")
    parser.add_argument("--min-cosine", type=float, default=.9999)
    parser.add_argument("--max-absolute-error", type=float, default=.002)
    args = parser.parse_args()
    resolve_path = evidence_resolver(args.evidence_prefix)
    if args.min_cosine != .9999 or args.max_absolute_error != .002:
        parser.error("Preserve the original .9999 cosine/.002 component gates")
    names = [item.split("=", 1)[0] for item in args.image]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        parser.error("Require distinct simple image case names")
    if args.existing_oracle and args.existing_oracle.resolve() == args.output.resolve():
        parser.error("Preserve a reused numerical oracle in a separate immutable report")
    root = Path(__file__).resolve().parents[2]
    vision_path, text_path = root / "eng/validation/embeddinggemma2-vision-reference.py", root / "eng/embeddinggemma2-reference.py"
    vision_module = load_module("vision_reference", vision_path)
    text_module = load_module("text_reference", text_path)
    proofs = [json.loads(path.read_text()) for path in args.preprocessing]
    if any(not proof["completed"] for proof in proofs):
        raise ValueError("Actual preprocessing must have completed")
    sources = json.loads(args.hf_source_manifest.read_text())
    for source in sources["sources"]:
        if digest(args.hf_source_manifest.parent / source["path"]) != source["sha256"]:
            raise ValueError("Pinned official HF source changed")
    pairs = [(path, json.loads(path.read_text())) for path in args.pair]
    model_hash, projector_hash = digest(args.model), digest(args.projector)
    for _, pair in pairs:
        preprocessor_for_pair(pair, proofs)
    previous = json.loads(args.existing_oracle.read_text()) if args.existing_oracle else None
    if previous is not None:
        if not previous["completed"] or previous["model_sha256"] != model_hash or previous["projector_sha256"] != projector_hash:
            raise ValueError("Reused complete numerical oracle differs from the original weights")
        for path in (vision_path, text_path, root/"eng/diffusiongemma-vision-oracle.py"):
            if previous["source_hashes"][str(path.relative_to(root))] != digest(path):
                raise ValueError("The reused complete graph arithmetic source changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(completed=False, passed=False, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method="Exact published identity-size pixel proof; independent HF-derived complete F32 vision tower plus complete F32 bidirectional text encoder and learned projection",
        model_sha256=model_hash, projector_sha256=projector_hash,
        preprocessing_artifacts=[dict(path=str(path.resolve()),sha256=digest(path)) for path in args.preprocessing],
        production_preprocessing=proofs, official_hf_sources=sources, numpy=np.__version__, python=sys.version,
        reused_numerical_oracle=None if previous is None else dict(path=str(args.existing_oracle.resolve()),sha256=digest(args.existing_oracle)),
        full_model_cases_executed_this_run=0,
        platform=platform.platform(), thresholds=dict(min_cosine=.9999, max_absolute_error=.002),
        source_hashes={str(path.relative_to(root)):digest(path) for path in (Path(__file__), vision_path, text_path, root/"eng/diffusiongemma-vision-oracle.py")},
        cases=[], evidence_prefix_bindings=args.evidence_prefix,
        limitations=["Only the recorded unprefixed IMAGE-only PNG fixtures/model/projector/backends are checked; no audio/video or overall multimodal quality claim.",
            "NumPy vision/text arithmetic is independent; gguf-py only reads/dequantizes the original GGUF tensors.",
            "The oracle uses exact observed TS pixels. HF rescale-vs-TS-divide rounding is recorded separately; resizing/alpha changes are rejected by the invariant-pixel proof.",
            "Saved pair failures remain failed. This diagnostic does not replace cross-engine or latency gates."])
    def write():
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        vision, text = vision_module.VisionEncoder(args.projector), text_module.Encoder(args.model)
        vocabulary = text.metadata("tokenizer.ggml.tokens")
        leading = [int(text.metadata("tokenizer.ggml.bos_token_id")), vocabulary.index("<|image>")]
        trailing = [vocabulary.index("<image|>"), int(text.metadata("tokenizer.ggml.eos_token_id"))]
        for item in args.image:
            name, path = item.split("=", 1); image = Path(path).resolve(); image_hash = digest(image)
            for _, pair in pairs:
                validate_pair(pair, name, image_hash, model_hash, projector_hash)
            observed = []
            for proof in proofs:
                record = next(row for row in proof["cases"] if resolve_path(row["image"]).resolve() == image and row["image_sha256"] == image_hash)
                current, pixel_metadata = pixel_proof(record, vision.hf, resolve_path)
                if observed and not np.array_equal(current.view(np.uint32), pixels.view(np.uint32)):
                    raise ValueError("Distinct actual assemblies produced different pixel inputs")
                pixels = current
                observed.append(dict(models_assembly_sha256=proof["models_assembly_sha256"], **pixel_metadata))
            print("starting full independent image", name, "geometry", pixels.shape, flush=True)
            start = time.perf_counter()
            prior_case = next(case for case in previous["cases"] if case["name"] == name) if previous else None
            if prior_case:
                if reused_pixel_hash(prior_case) != observed[0]["actual_processed_pixels_sha256"]:
                    raise ValueError("Reused numerical oracle pixel input changed")
                soft_source = args.existing_oracle.parent / (name + "-soft.f32")
                if digest(soft_source) != prior_case["soft_rows_sha256"]:
                    raise ValueError("Preserved full vision rows changed")
                soft = np.fromfile(soft_source,dtype="<f4").reshape(prior_case["soft_shape"])
            else:
                soft = vision.encode(pixels)
            if soft.ndim != 2 or soft.shape != (record["soft_tokens"], text.dim) or not np.all(np.isfinite(soft)):
                raise ValueError("Independent full vision rows differ from proved pooling geometry")
            inputs = np.concatenate((text.token_rows(leading), soft, text.token_rows(trailing)), axis=0)
            if prior_case:
                if (leading != prior_case["leading_token_ids"] or trailing != prior_case["trailing_token_ids"] or
                    hashlib.sha256(inputs.astype("<f4").tobytes()).hexdigest() != prior_case["input_rows_sha256"]):
                    raise ValueError("Reused numerical oracle full token/media inputs changed")
                expected = np.asarray(prior_case["embedding"],dtype=np.float32)
            else:
                expected = text.encode_inputs(inputs)
                report["full_model_cases_executed_this_run"] += 1
            if expected.shape != (768,) or not np.all(np.isfinite(expected)) or abs(float(np.linalg.norm(expected)) - 1) > 1e-5:
                raise ValueError("Independent full pooled image vector is invalid")
            soft_path, input_path = [args.output.parent / (name + suffix) for suffix in ("-soft.f32", "-inputs.f32")]
            soft.astype("<f4").tofile(soft_path); inputs.astype("<f4").tofile(input_path)
            comparisons = []
            for pair_path, pair in pairs:
                engines = {}
                for engine in ("tensorsharp", "llama"):
                    row = pair["engines"][engine]["benchmarks"][name]
                    engines[engine] = compare(expected, row["embedding"], row["prompt_tokens"], inputs.shape[0], .9999, .002)
                comparisons.append(dict(pair=str(pair_path.resolve()), pair_sha256=digest(pair_path),
                    backend=pair["expected_backend"], engines=engines, original_pair_passed=pair["passed"],
                    observed_models_assembly_sha256=preprocessor_for_pair(pair,proofs)["models_assembly_sha256"],
                    original_pair_quality_pass=pair["quality_pass"], server_exit_codes={engine:pair["engines"][engine]["server_exit_code"] for engine in engines},
                    dependency_revisions=pair["dependency_revisions"], dependency_tracked_changes=pair["dependency_tracked_changes"],
                    binary_artifacts=pair["binary_artifacts"]))
            report["cases"].append(dict(name=name, image=str(image), preprocessing_proofs=observed,
                vision_layers=vision.layers, vision_heads=vision.heads, soft_shape=list(soft.shape),
                soft_rows_sha256=digest(soft_path), input_rows_sha256=digest(input_path), total_input_rows=int(inputs.shape[0]),
                leading_token_ids=leading, trailing_token_ids=trailing, embedding=expected.tolist(),
                oracle_seconds=time.perf_counter()-start, complete_graph_math_reused=prior_case is not None,
                original_full_graph_seconds=None if prior_case is None else prior_case["oracle_seconds"], comparisons=comparisons))
            write()
            print(json.dumps(dict(case=name, total_input_rows=int(inputs.shape[0]), comparisons=[dict(backend=x["backend"],engines=x["engines"]) for x in comparisons])),flush=True)
        report["completed"] = True
        report["passed"] = all(check["passed"] for case in report["cases"] for pair in case["comparisons"] for check in pair["engines"].values())
        return 0 if report["passed"] else 1
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        write()


if __name__ == "__main__":
    raise SystemExit(main())
