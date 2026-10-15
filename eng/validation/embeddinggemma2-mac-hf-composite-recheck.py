#!/usr/bin/env python3
"""Recheck one frozen Mac candidate against the complete Mac default-HF oracle.

No learned model math is executed. The original seven inputs, preprocessing
assemblies or fresh identical observations, numerical limits and runtime guards
remain mandatory. The same Apple codec is independently applied by a second
application; original failed FFmpeg equivalence remains unqualified. Original
HTTP, oracle and decoder reports are never modified.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE_RECHECK_SHA256 = "d675695a17b2d504b5c207e5c88230ccc074226e202a77bb9a1add1539813ede"
CODEC_REFERENCE_KIND = "mac_avfoundation_same_codec_separate_application"
EXPECTED_SOURCE_HASHES = {
    "eng/validation/embeddinggemma2-controlled-composite-reference.py": "d36770172d1011070354a44588e0b6ffc12658782cc1673ceba01de4d580e3f2",
    "eng/validation/embeddinggemma2-vision-reference.py": "895ad5e8fd916c70fb936a97d52d95bf357aae5991c12241c7594316fb7ba3ff",
    "eng/embeddinggemma2-reference.py": "fd7c30ff9faef986cf5ee47b2f2c2ea5faf1e945e9672ea65fae1874ee09bf47",
    "eng/diffusiongemma-vision-oracle.py": "49adff6b7cf3433aa55669010fe50882a99a11931ccaea1b87dddcebb2999ce8",
    "eng/validation/embeddinggemma2-default-hf-composite-reference.py": "d83acc0e55848abbc1498d6a9b3f8312d1c172b68d313a4d09d8637d868e769e",
    "eng/validation/embeddinggemma2-mac-hf-composite-reference.py": "e1cdfe47ab7142fa6d29f826366f5b15073dade6f0cc223036588b684939c93f",
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


reference_path = Path(__file__).with_name("embeddinggemma2-mac-hf-composite-reference.py")
if hashlib.sha256(reference_path.read_bytes()).hexdigest() != "e1cdfe47ab7142fa6d29f826366f5b15073dade6f0cc223036588b684939c93f":
    raise ValueError("Reviewed Mac reference source changed")
reference = module("mac_default_hf_saved_reference", reference_path)
controlled = module("default_hf_saved_controlled_recheck", Path(__file__).with_name("embeddinggemma2-controlled-composite-recheck.py"))
sha = reference.sha


def validate_report_selection(paths):
    if len(paths) != 1:
        raise ValueError("Mac saved recheck requires exactly one candidate HTTP report per invocation")


def validate_decoder_record(oracle):
    record = oracle.get("decoder_proof", {})
    path, digest = record.get("path"), record.get("sha256")
    if (not isinstance(path, str) or not path or not isinstance(digest, str) or len(digest) != 64 or
            any(c not in "0123456789abcdef" for c in digest) or
            oracle["before_hashes"].get(str(Path(path).resolve())) != digest):
        raise ValueError("Original Mac decoder proof is not bound by the original oracle guard")
    observed = oracle.get("mac_video_decoder", {})
    if (observed.get("proof_sha256") != digest or observed.get("exact_sampled_rgba") is not True or
            observed.get("same_platform_codec") is not True):
        raise ValueError("Original exact same-codec sampled RGBA proof is missing")


def validate_original_decoder(oracle, proof, proof_path, read, bind):
    """Revalidate original bytes/digests; never capture replacement decoder guards."""
    validate_decoder_record(oracle)
    original = oracle["before_hashes"]
    record = oracle["decoder_proof"]
    decoder_path = Path(record["path"])
    decoder = read(decoder_path, record["sha256"])

    def read_original(path):
        expected = original.get(str(Path(path).resolve()))
        if expected is None:
            raise ValueError("Decoder metadata missing from the original oracle guard: " + str(path))
        return read(path, expected)

    _, bound = reference.validate_decoder(decoder, decoder_path.parent, proof, proof_path, read_original)
    for path, digest in bound.items():
        expected = original.get(str(Path(path).resolve()))
        if expected is None or expected != digest:
            raise ValueError("Decoder returned a replaced or unbound original digest: " + str(path))
        bind(path, expected)


def validate_codec_runtime(report, proof, guards, bind):
    """HTTP's short binary list omits codecs; require them in the same full guard."""
    matching = [guard for guard in guards if guard.get("before") and
        guard.get("artifacts_unchanged") is True and guard["before"] == guard.get("after") and
        all(guard["before"].get(row["path"]) == row["sha256"] for row in report["binary_artifacts"])]
    if not matching:
        raise ValueError("No complete candidate runtime guard covers the HTTP binaries")
    models = [Path(row["path"]) for row in report["binary_artifacts"]
        if Path(row["path"]).name == "TensorSharp.Models.dll"]
    if len(models) != 1:
        raise ValueError("Require one actual candidate Models assembly path")
    host = models[0].parent
    codecs = proof.get("codec_assemblies", [])
    native = proof.get("loaded_native_codecs", [])
    if (not codecs or not native or
            len({Path(row["path"]).name for row in codecs}) != len(codecs) or
            sum(row["path"] == proof["video_codec_path"] and row["sha256"] == proof["video_codec_sha256"]
                for row in native) != 1):
        raise ValueError("Original codec assembly/native observations are incomplete")
    expected = {host / Path(row["path"]).name: row["sha256"] for row in codecs}
    for row in native:
        parts = Path(row["path"]).parts
        if parts.count("runtimes") != 1:
            raise ValueError("Mac native codec must have an observed bundled runtime path")
        relative = Path(*parts[parts.index("runtimes"):])
        candidate = host / relative
        if candidate in expected and expected[candidate] != row["sha256"]:
            raise ValueError("Conflicting observed codec identity")
        expected[candidate] = row["sha256"]
    if not any(all(guard["before"].get(str(path)) == digest for path, digest in expected.items())
            for guard in matching):
        raise ValueError("Candidate codec differs or is missing from the same complete runtime guard")
    for path, digest in expected.items():
        bind(path, digest)


def validate_mrl_if_present(original, current):
    old, new = original.get("base64_mrl"), current.get("base64_mrl")
    if old is not None and new is None:
        raise ValueError("Candidate omitted the original saved HTTP MRL check")
    for value in (old, new):
        if value is None:
            continue
        if (not isinstance(value, dict) or type(value.get("dimensions")) is not int or
                value["dimensions"] not in (128, 256, 512) or type(value.get("cosine")) not in (int, float) or
                not np.isfinite(value["cosine"]) or value["cosine"] < .99999):
            raise ValueError("Saved HTTP MRL dimension/cosine gate failed")
    if old is not None and old["dimensions"] != new["dimensions"]:
        raise ValueError("Candidate changed the original saved HTTP MRL dimension")


def validate_oracle(oracle):
    if (oracle.get("completed") is not True or oracle.get("passed") is not True or
            oracle.get("input_source_hashes_unchanged") is not True or
            oracle.get("before_hashes") != oracle.get("after_hashes") or not oracle.get("before_hashes") or
            oracle.get("fresh_unique_vision_inputs") != 6 or oracle.get("fresh_complete_text_cases") != 7 or
            oracle.get("thresholds") != dict(min_cosine=.9999, max_absolute_error=.002) or
            oracle.get("pixel_max_absolute_error") != 2e-6 or
            tuple(row["name"] for row in oracle.get("cases", [])) != reference.CASE_NAMES or
            oracle.get("source_hashes") != EXPECTED_SOURCE_HASHES or
            oracle.get("codec_reference_kind") != CODEC_REFERENCE_KIND or
            oracle.get("ffmpeg_equivalence_qualified") is not False or
            oracle.get("independent_codec_implementation") is not False):
        raise ValueError("Require the successful complete original Mac default-HF oracle, six exact sources, codec scope and unchanged limits")
    validate_decoder_record(oracle)
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
    validate_report_selection(args.http)
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
    validate_original_decoder(oracle, original_proof, original_proof_path, read, bind)
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
        validate_codec_runtime(value, proof, [guard for _, guard in guards], bind)
        validate_mrl_if_present(original, value)
    bind(__file__)
    bind(Path(__file__).with_name("embeddinggemma2-default-hf-composite-recheck.py"), BASE_RECHECK_SHA256)
    bind(Path(__file__).with_name("embeddinggemma2-controlled-composite-recheck.py"))
    bind(Path(__file__).with_name("embeddinggemma2_http_environment.py"))
    before = dict(observed)
    for path, digest in before.items():
        bind(path, digest)
    result = dict(completed=False, passed=False,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method="Hash-bound reuse of complete Mac default-HF seven-case oracle; same Apple codec separately applied",
        codec_reference_kind=CODEC_REFERENCE_KIND, ffmpeg_equivalence_qualified=False,
        independent_codec_implementation=False, decoder_proof=oracle["decoder_proof"],
        original_decoder_guards_revalidated=True, reference_source_hashes=EXPECTED_SOURCE_HASHES,
        fresh_model_math=0, thresholds=oracle["thresholds"],
        oracle=dict(path=str(args.oracle.resolve()), sha256=args.oracle_sha256),
        proof=dict(path=str(proof_path.resolve()), sha256=observed[str(proof_path.resolve())],
            fresh_preprocessing_observations=args.input_proof is not None),
        runtime_guards=[dict(path=str(path.resolve()), sha256=observed[str(path.resolve())]) for path, _ in guards],
        model_sha256=model_hash, projector_sha256=projector_hash,
        before_hashes=before, cases=[],
        limitations=["Saved seven opaque synthetic default-HF cases only; no fresh learned model computation or service timing.",
            "Preprocessing assembly changes require fresh identical observations; native changes still require complete runtime guards.",
            "Original tokenizer/resize/audio reference scope and failed experiments remain unchanged.",
            "Mac video codec is a separate application of the same Apple codec, not an independent codec implementation or restored FFmpeg equivalence.",
            "Saved HTTP MRL summary, when present, is rechecked for its original dimension and gate; no fresh independent prefix computation."])
    for path, current in reports:
        values = []
        for case, row in zip(oracle["cases"], current["cases"]):
            if case["name"] != row["name"]:
                raise ValueError("Complete case ordering differs")
            values.append(dict(name=case["name"], protocols=reference.compare(
                np.asarray(case["embedding"], dtype=np.float64), row, case["total_input_rows"])))
        result["cases"].append(dict(http=str(path.resolve()), sha256=observed[str(path.resolve())],
            backend=current["expected_backend"], comparisons=values,
            saved_http_mrl=current.get("base64_mrl")))
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
