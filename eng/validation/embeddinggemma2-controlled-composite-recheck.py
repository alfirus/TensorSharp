#!/usr/bin/env python3
"""Compare new frozen pairs with an explicitly hash-bound complete oracle.

This executes no new model math. It requires the original input policy, media,
observed preprocessing, oracle source, and weight hashes to match.
Original failed pair reports are never changed.
"""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np


def load_reference_tool():
    path = Path(__file__).with_name("embeddinggemma2-controlled-composite-reference.py")
    spec = importlib.util.spec_from_file_location("composite_reference_recheck", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


tool = load_reference_tool()
sys.path.insert(0, str(Path(__file__).parent))
from embeddinggemma2_http_environment import audit_child_environment
SOURCE_FILES = ("eng/validation/embeddinggemma2-controlled-composite-reference.py",
    "eng/validation/embeddinggemma2-vision-reference.py", "eng/embeddinggemma2-reference.py",
    "eng/diffusiongemma-vision-oracle.py")


def controlled_audio_command(pair):
    command = pair["engines"]["tensorsharp"]["command"]
    if not command or Path(command[0]).name != "env":
        raise ValueError("Require the recorded direct env command for controlled audio")
    assignments = []
    for index, argument in enumerate(command[1:], 1):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=.*", argument):
            break
        assignments.append(argument)
    else:
        raise ValueError("Missing executable after controlled environment assignments")
    key = "TS_EMBEDDING_AUDIO_LOG_POLICY="
    policies = [argument for argument in assignments if argument.startswith(key)]
    if (policies != [key + "llama_clamp"] or command[index].startswith("-") or
            any(argument.startswith(key) for argument in command[index:])):
        raise ValueError("Require one unambiguous leading controlled audio policy assignment")


def matching_inputs(original, current):
    """Require exactly the original composite sequences and all media identities."""
    if original["inputs"] != current["inputs"] or original["media"] != current["media"]:
        raise ValueError("New pair media or ordered input sequences differ from the complete oracle")


def bind_tensorsharp_reference(pair, reference_proof, model_hash, projector_hash):
    """An explicit ablation has two TensorSharp roles, with equal video parts."""
    if (pair.get("reference_controlled_video_prefix") is not True or
            pair.get("controlled_video_prefix") is not True or
            pair.get("binary_artifacts") != pair.get("binary_artifacts_after") or
            not audit_child_environment(pair["engines"]["llama"]["command"], {}).resolved_server_environment):
        raise ValueError("Require an explicit direct TensorSharp reference ablation with identical prefixes and stable binaries")
    reference_pair = {**pair,
        "engines": {**pair["engines"], "tensorsharp": pair["engines"]["llama"]},
        "binary_artifacts": {"tensorsharp": pair["binary_artifacts"]["llama"]}}
    controlled_audio_command(reference_pair)
    # Each frozen side needs its own actual assembly identity, with observations
    # equivalent to the original. Native-only candidates can share one proof.
    tool.bind_pair(reference_pair, reference_proof, model_hash, projector_hash)


def equivalent_proof(original, current):
    """A rebuilt assembly needs fresh actual input evidence, not a hash waiver."""
    keys = ("completed", "model_sha256", "tokenizer_type", "token_ids", "texts", "case_text_runs",
        "controlled_video_prefix", "video_prefix", "video_prefix_source", "include_video_timestamps",
        "video_soft_tokens", "video_frames_per_second", "max_video_frames", "videos",
        "image_soft_tokens", "image_reference_sizing", "images", "audios", "video_decoder",
        "video_codec_sha256", "dotnet_runtime", "os", "architecture", "execution_scope")
    if any(original[key] != current[key] for key in keys):
        raise ValueError("Fresh preprocessing observations differ from the complete oracle inputs")
    for key in ("codec_assemblies", "loaded_native_codecs"):
        identities = lambda proof: sorted((Path(row["path"]).name, row["sha256"]) for row in proof[key])
        if identities(original) != identities(current):
            raise ValueError("Fresh proof loaded different codec binaries")


def check_proof_buffers(proof, parent):
    for record in proof["images"] + [frame for video in proof["videos"] for frame in video["frames"]]:
        tool.identity_pixels(record, parent)
    for video in proof["videos"]:
        tool.validate_video_indices(video)


def validate_oracle(oracle):
    if (not oracle.get("completed") or oracle.get("fresh_unique_video_frames") != 4 or
            oracle.get("fresh_complete_text_cases") != 3 or
            oracle.get("thresholds") != dict(min_cosine=.9999, max_absolute_error=.002) or
            [case["name"] for case in oracle["cases"]] != ["video", "mixed", "all-modalities"]):
        raise ValueError("Require the complete original four-frame, three-case numerical oracle")
    if set(oracle.get("source_hashes", {})) != set(SOURCE_FILES):
        raise ValueError("Require every original independent arithmetic/checker source identity")
    for case in oracle["cases"]:
        value = np.asarray(case["embedding"], dtype=np.float64)
        if (value.shape != (768,) or not np.all(np.isfinite(value)) or
                abs(float(np.linalg.norm(value)) - 1) > 1e-5 or case["total_input_rows"] <= 0):
            raise ValueError("Preserved complete embedding is invalid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("oracle", "model", "projector", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--oracle-sha256", required=True)
    parser.add_argument("--input-proof", type=Path,
        help="Fresh frozen-assembly proof; every observed input and codec identity must equal the original")
    parser.add_argument("--pair", type=Path, action="append", required=True)
    parser.add_argument("--tensorsharp-reference", action="store_true",
        help="Explicit TS-vs-TS ablation; each side requires matching frozen preprocessing evidence")
    parser.add_argument("--reference-input-proof", type=Path,
        help="Frozen TensorSharp baseline observations; requires --tensorsharp-reference and original-equivalent inputs")
    args = parser.parse_args()
    if args.reference_input_proof and not args.tensorsharp_reference:
        raise ValueError("Reference preprocessing evidence requires explicit --tensorsharp-reference")
    if args.output.exists():
        raise ValueError("Preserve existing evidence; choose a new report")
    if tool.sha(args.oracle) != args.oracle_sha256:
        raise ValueError("Preserved oracle does not match the explicitly supplied hash")
    oracle = json.loads(args.oracle.read_text())
    validate_oracle(oracle)
    model_hash, projector_hash = tool.sha(args.model), tool.sha(args.projector)
    if (oracle["model_sha256"], oracle["projector_sha256"]) != (model_hash, projector_hash):
        raise ValueError("Original numerical oracle weights differ")
    root = Path(__file__).resolve().parents[2]
    for name, expected in oracle["source_hashes"].items():
        if tool.sha(tool.artifact(root, name)) != expected:
            raise ValueError("Independent model arithmetic or preprocessing checker changed")
    for record in oracle["inputs"].values():
        if tool.sha(record["path"]) != record["sha256"]:
            raise ValueError("Preserved oracle input evidence changed")
    proof_path = Path(oracle["inputs"]["proof"]["path"])
    original_proof = json.loads(proof_path.read_text())
    check_proof_buffers(original_proof, proof_path.parent)
    proof = original_proof
    if args.input_proof:
        proof = json.loads(args.input_proof.read_text())
        equivalent_proof(original_proof, proof)
        check_proof_buffers(proof, args.input_proof.parent)
    reference_proof = original_proof
    if args.reference_input_proof:
        reference_proof = json.loads(args.reference_input_proof.read_text())
        equivalent_proof(original_proof, reference_proof)
        check_proof_buffers(reference_proof, args.reference_input_proof.parent)
    # Original pairs supply the exact input sequences that produced the stored
    # independent complete embeddings, with their immutable hashes checked.
    originals = {}
    for case in oracle["cases"]:
        for comparison in case["comparisons"]:
            path, expected = comparison["pair"], comparison["pair_sha256"]
            if path not in originals:
                if tool.sha(path) != expected:
                    raise ValueError("Original pair input evidence changed")
                originals[path] = json.loads(Path(path).read_text())
                controlled_audio_command(originals[path])
                tool.bind_pair(originals[path], original_proof, model_hash, projector_hash)
            elif tool.sha(path) != expected:
                raise ValueError("Inconsistent original pair identity")
    if not originals:
        raise ValueError("Missing original pair evidence")
    original = next(iter(originals.values()))
    report = dict(completed=False, passed=False, tensorsharp_passed=False,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method="Hash-bound reuse of complete independent F32 oracle; no fresh model math",
        fresh_model_math=0, thresholds=oracle["thresholds"],
        oracle=dict(path=str(args.oracle.resolve()), sha256=args.oracle_sha256),
        input_proof=dict(path=str((args.input_proof or proof_path).resolve()),
            sha256=tool.sha(args.input_proof or proof_path), fresh_input_evidence=args.input_proof is not None),
        tool_sha256=tool.sha(__file__), model_sha256=model_hash, projector_sha256=projector_hash,
        environment_checker_sha256=tool.sha(Path(__file__).with_name("embeddinggemma2_http_environment.py")),
        cases=[], limitations=["Only the original controlled inputs; rebuilt managed assemblies require fresh identical preprocessing observations.",
            "This does not turn original cross-engine or performance failures into passes.",
            "The complete numerical computation is reused, not counted as a new model-math execution."])
    report["tensorsharp_reference_ablation"] = args.tensorsharp_reference
    if args.tensorsharp_reference:
        report["role_mapping"] = dict(tensorsharp="TensorSharp candidate", llama="Frozen TensorSharp baseline; llama.cpp not executed")
        reference_path = args.reference_input_proof or proof_path
        report["reference_input_proof"] = dict(path=str(reference_path.resolve()), sha256=tool.sha(reference_path))
    for path in args.pair:
        pair = json.loads(path.read_text())
        controlled_audio_command(pair)
        tool.bind_pair(pair, proof, model_hash, projector_hash)
        if args.tensorsharp_reference:
            bind_tensorsharp_reference(pair, reference_proof, model_hash, projector_hash)
        elif pair.get("reference_controlled_video_prefix"):
            raise ValueError("TensorSharp reference prefix requires explicit --tensorsharp-reference; do not treat it as llama.cpp")
        matching_inputs(original, pair)
        comparisons = []
        for case in oracle["cases"]:
            actual = next(row for row in pair["inputs"] if row["name"] == case["name"])
            parts = tool.controlled_parts(actual["parts"], proof["video_prefix"])
            engines = {engine:tool.compare(case["embedding"], pair["engines"][engine]["benchmarks"][case["name"]],
                parts if engine == "tensorsharp" or args.tensorsharp_reference else actual["parts"], case["total_input_rows"])
                for engine in ("tensorsharp", "llama")}
            comparisons.append(dict(name=case["name"], engines=engines))
        report["cases"].append(dict(pair=str(path.resolve()), pair_sha256=tool.sha(path),
            backend=pair["expected_backend"], original_pair_passed=pair["passed"],
            original_quality_pass=pair["quality_pass"], original_performance_pass=pair["performance_pass"],
            comparisons=comparisons))
    report["tensorsharp_passed"] = all(case["engines"]["tensorsharp"]["passed"] for row in report["cases"] for case in row["comparisons"])
    report["passed"] = all(result["passed"] for row in report["cases"] for case in row["comparisons"] for result in case["engines"].values())
    report["completed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
