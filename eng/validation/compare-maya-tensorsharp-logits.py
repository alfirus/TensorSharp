#!/usr/bin/env python3
"""Prepare and compare bounded full-vocabulary Maya / TensorSharp captures.

Requires numpy. This only reads artifacts; it never runs a model. Quality-only
captures cannot supply throughput evidence. Numerical metrics are descriptive,
without a quality threshold or a release-qualification decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEACHER_LAYOUT = "row i predicts output_token_ids[i] after prompt + output[:i]"
LIMITATIONS = [
    "This bounded capture comparison does not establish semantic correctness, broad model quality, or release qualification.",
    "Perplexity is measured on the teacher-generated continuation, conditioned on the identical prompt and prior teacher tokens; it is not held-out corpus perplexity.",
    "Every observed prediction is included, including a final EOS when emitted by the teacher.",
    "Capture timings include logit readback and artifact writes and are excluded from performance conclusions.",
    "Source/environment fields are retained as declared by the capture runners. This tool does not attest runtime binaries, checkpoint bytes, devices, or participating nodes.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path):
    raw = Path(path).read_bytes()

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid), sha(raw)


def hash_value(value):
    require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value), "Missing lowercase SHA-256 identity")
    return value


def positive_integer(value, name):
    require(type(value) is int and value > 0, name + " must be a positive integer")
    return value


def integer_equal(value, expected):
    return type(value) is int and value == expected


def token_ids(value, vocab, name):
    require(isinstance(value, list) and value and all(type(token) is int and 0 <= token < vocab for token in value),
            name + " must be nonempty in-vocabulary integer token IDs")
    return value


def token_sha(ids):
    return sha(np.asarray(ids, dtype="<i4").tobytes())


def capture_path(recorded, report_path, root=None):
    require(isinstance(recorded, str) and recorded, "Capture is missing its path")
    if root is not None:
        # Remote captures commonly retain absolute Linux paths after being copied
        # to a local evidence directory. Only the recorded basename is remapped.
        name = recorded.replace("\\", "/").rsplit("/", 1)[-1]
        require(name not in ("", ".", ".."), "Invalid capture filename")
        resolved = (Path(root).resolve() / name).resolve()
        require(resolved.is_relative_to(Path(root).resolve()), "Capture escapes its remapped root")
        return resolved
    path = Path(recorded)
    return path.resolve() if path.is_absolute() else (Path(report_path).resolve().parent / path).resolve()


def check_payload(path, rows, vocab, expected_sha):
    require(Path(path).stat().st_size == rows * vocab * 4, "Capture has truncated, extra, or missing full-vocabulary rows: " + str(path))
    require(file_sha(path) == hash_value(expected_sha), "Capture SHA-256 mismatch: " + str(path))


def indexed_cases(document):
    cases = document.get("cases")
    require(isinstance(cases, list) and cases, "Case file has no cases")
    indexed = {}
    for case in cases:
        require(isinstance(case, dict), "Case must be an object")
        name = case.get("name")
        require(isinstance(name, str) and name and name not in indexed, "Case names must be unique nonempty strings")
        indexed[name] = case
    return indexed


def teacher_rows(report, repeat):
    require(integer_equal(report.get("schema"), 1) and report.get("status") == "completed", "Teacher report did not complete")
    require(report.get("mode") == "quality_only" and report.get("capture_excluded_from_performance") is True,
            "Teacher capture must be explicitly quality-only and excluded from performance")
    require(isinstance(report.get("results"), list), "Teacher report has no results array")
    require(type(repeat) is int and repeat >= 0, "Invalid measured teacher repeat")
    selected = [row for row in report["results"] if integer_equal(row.get("repeat"), repeat) and row.get("warmup") is False]
    require(selected, "Teacher report has no measured results for the selected repeat")
    names = [row.get("name") for row in selected]
    require(len(set(names)) == len(names), "Duplicate teacher case at the selected repeat")
    return {row["name"]: row for row in selected}


def validate_teacher_case(case, result, vocab, eos, report_path, root=None):
    ids = token_ids(case.get("ids"), vocab, "Prompt")
    require("forced_ids" not in case, "Source teacher cases must be free generation, not forced replay")
    require(token_ids(result.get("input_ids"), vocab, "Teacher prompt") == ids
            and integer_equal(result.get("prompt_tokens"), len(ids)), "Teacher prompt token identity differs")
    outputs = token_ids(result.get("output_token_ids"), vocab, "Teacher continuation")
    maximum = positive_integer(case.get("max_new"), "Source max_new")
    require(integer_equal(result.get("max_new"), maximum) and integer_equal(result.get("generated_tokens"), len(outputs))
            and len(outputs) <= maximum, "Teacher output count/cap differs")
    require(integer_equal(result.get("draft_offered"), 0) and integer_equal(result.get("draft_accepted"), 0), "Teacher used speculative tokens")
    require(integer_equal(result.get("reused_prompt_tokens"), 0), "Teacher quality capture reused prompt state")
    require(not any(token in eos for token in outputs[:-1]), "Teacher continuation contains EOS before its final token")
    if result.get("finish_reason") == "stop":
        require(outputs[-1] in eos, "Teacher stop does not end in a declared EOS token")
    else:
        require(result.get("finish_reason") == "length" and len(outputs) == maximum and outputs[-1] not in eos,
                "Teacher generation is incomplete or has inconsistent finish/EOS metadata")
    capture = result.get("logits_capture")
    require(isinstance(capture, dict) and capture.get("capture_excluded_from_performance") is True,
            "Teacher result has no quality-only full-vocabulary capture")
    require(capture.get("dtype") == "float32" and capture.get("byte_order") == "little", "Teacher capture must be little-endian F32")
    require(capture.get("layout") == TEACHER_LAYOUT, "Teacher capture has an unknown row/token layout")
    request_index = positive_integer(result.get("request_index"), "Teacher request_index")
    require(integer_equal(capture.get("request_index"), request_index) and integer_equal(capture.get("row_count"), len(outputs))
            and integer_equal(capture.get("vocab"), vocab), "Teacher capture row/request/vocabulary metadata differs")
    path = capture_path(capture.get("path"), report_path, root)
    check_payload(path, len(outputs), vocab, capture.get("sha256"))
    return {"request_index": request_index, "repeat": result["repeat"], "finish_reason": result["finish_reason"],
            "prompt_i32_sha256": token_sha(ids), "continuation_i32_sha256": token_sha(outputs),
            "capture": {"path": str(path), "rows": len(outputs), "vocab": vocab, "dtype": "float32-le",
                        "sha256": capture["sha256"], "first_prediction_position": len(ids)},
            "output_ids": outputs}


def prepare_replay(teacher, teacher_hash, original, original_hash, teacher_path, repeat=0, teacher_root=None):
    require(type(repeat) is int and repeat >= 0, "Select a nonnegative measured teacher repeat")
    vocab = positive_integer(original.get("vocab_size"), "Vocabulary size")
    require(vocab <= 2**31 - 1, "Token vocabulary exceeds int32 identity encoding")
    eos = token_ids(original.get("eos_ids"), vocab, "Declared EOS IDs")
    require(len(set(eos)) == len(eos), "Duplicate EOS IDs")
    require(original.get("temperature") == 0 and original.get("thinking") is False,
            "Original cases must declare the matched greedy non-thinking workload")
    require(teacher.get("case_file_sha256") == original_hash, "Teacher report belongs to a different original case file")
    source_cases = indexed_cases(original)
    results = teacher_rows(teacher, repeat)
    require(set(results) == set(source_cases), "Teacher results omit or add cases; partial intersection is forbidden")
    replay_cases = []
    indices = set()
    for name, case in source_cases.items():
        metadata = validate_teacher_case(case, results[name], vocab, eos, teacher_path, teacher_root)
        require(metadata["request_index"] not in indices, "Teacher results reuse a request dump index")
        indices.add(metadata["request_index"])
        replay_cases.append({**case, "max_new": len(metadata["output_ids"]), "forced_ids": metadata.pop("output_ids"),
                             "teacher": metadata})
    return {"schema": 1, "kind": "maya-teacher-forced-replay", "thinking": False, "temperature": 0,
            "vocab_size": vocab, "eos_ids": eos, "tokenizer_model": original.get("tokenizer_model"),
            "original_cases_sha256": original_hash, "teacher_report_sha256": teacher_hash,
            "teacher_repeat": repeat, "teacher_source_revision": teacher.get("source_revision"),
            "capture_excluded_from_performance": True,
            "coverage": "all original cases at the explicitly selected measured teacher repeat",
            "cases": replay_cases}


def log_softmax(values):
    shifted = values.astype(np.float64) - float(np.max(values))
    return shifted - math.log(float(np.exp(shifted).sum(dtype=np.float64)))


def compare_vectors(teacher, candidate, forced):
    require(teacher.ndim == candidate.ndim == 1 and teacher.shape == candidate.shape and teacher.size > 1,
            "Full-vocabulary row shapes differ")
    require(type(forced) is int and 0 <= forced < teacher.size, "Forced token is outside the vocabulary")
    finite_teacher, finite_candidate = np.isfinite(teacher), np.isfinite(candidate)
    if not finite_teacher.all() or not finite_candidate.all():
        return {"finite": False, "teacher_nonfinite_count": int((~finite_teacher).sum()),
                "candidate_nonfinite_count": int((~finite_candidate).sum()),
                "teacher_nonfinite_coordinates_first32": np.flatnonzero(~finite_teacher)[:32].tolist(),
                "candidate_nonfinite_coordinates_first32": np.flatnonzero(~finite_candidate)[:32].tolist()}
    a, b = teacher.astype(np.float64), candidate.astype(np.float64)
    log_a, log_b = log_softmax(a), log_softmax(b)
    difference = b - a
    teacher_top, candidate_top = int(np.argmax(a)), int(np.argmax(b))
    return {"finite": True, "forced_id": forced, "teacher_argmax": teacher_top, "candidate_argmax": candidate_top,
            "top1_agrees": teacher_top == candidate_top, "teacher_emitted_matches_argmax": forced == teacher_top,
            "candidate_argmax_matches_forced": forced == candidate_top,
            "kl_teacher_to_candidate": max(0.0, float(np.dot(np.exp(log_a), log_a - log_b))),
            "max_absolute_logit_difference": float(np.abs(difference).max()),
            "rms_logit_difference": float(np.sqrt(np.mean(difference * difference))),
            "mean_logit_shift": float(difference.mean()),
            "teacher_forced_nll": float(-log_a[forced]), "candidate_forced_nll": float(-log_b[forced])}


def summarize(rows):
    require(rows and all(row.get("finite") is True for row in rows), "Cannot summarize empty/nonfinite rows")
    teacher_nll = float(np.mean([row["teacher_forced_nll"] for row in rows]))
    candidate_nll = float(np.mean([row["candidate_forced_nll"] for row in rows]))

    def perplexity(nll):
        return math.exp(nll) if nll <= math.log(sys.float_info.max) else None

    return {"rows": len(rows), "mean_kl_teacher_to_candidate": float(np.mean([row["kl_teacher_to_candidate"] for row in rows])),
            "max_kl_teacher_to_candidate": max(row["kl_teacher_to_candidate"] for row in rows),
            "top1_agreement_count": sum(row["top1_agrees"] for row in rows),
            "top1_agreement_fraction": sum(row["top1_agrees"] for row in rows) / len(rows),
            "teacher_emitted_argmax_agreement_fraction": sum(row["teacher_emitted_matches_argmax"] for row in rows) / len(rows),
            "candidate_forced_argmax_agreement_fraction": sum(row["candidate_argmax_matches_forced"] for row in rows) / len(rows),
            "teacher_mean_forced_nll": teacher_nll, "candidate_mean_forced_nll": candidate_nll,
            "teacher_continuation_perplexity": perplexity(teacher_nll),
            "candidate_continuation_perplexity": perplexity(candidate_nll),
            "teacher_perplexity_overflows_float64": perplexity(teacher_nll) is None,
            "candidate_perplexity_overflows_float64": perplexity(candidate_nll) is None,
            "max_absolute_logit_difference": max(row["max_absolute_logit_difference"] for row in rows),
            "max_rms_logit_difference": max(row["rms_logit_difference"] for row in rows)}


def declared_source(report, report_hash):
    return {"report_sha256": report_hash, "engine": report.get("engine"),
            "source_revision": report.get("source_revision"), "environment": report.get("environment"),
            "command": report.get("command"), "options": report.get("options"),
            "model": report.get("model"), "case_file_sha256": report.get("case_file_sha256"),
            "kv_fingerprint": report.get("kv_fingerprint"),
            "capture_excluded_from_performance": report.get("capture_excluded_from_performance")}


def compare_artifacts(replay, replay_hash, teacher, teacher_hash, candidate, candidate_hash,
                      replay_path, teacher_path, candidate_path, teacher_root=None, candidate_root=None, evidence=None):
    report = evidence if evidence is not None else {}
    report.update(schema=1, status="failed", quality_qualified=False, performance_qualified=False,
                  started_utc=datetime.now(timezone.utc).isoformat(), comparator_sha256=file_sha(__file__),
                  numpy_version=np.__version__, replay_cases_sha256=replay_hash, limitations=LIMITATIONS,
                  declared_sources={"teacher": declared_source(teacher, teacher_hash),
                                    "candidate": declared_source(candidate, candidate_hash)}, rows=[])
    try:
        require(integer_equal(replay.get("schema"), 1) and replay.get("kind") == "maya-teacher-forced-replay", "Wrong replay case schema")
        require(replay.get("teacher_report_sha256") == teacher_hash, "Teacher report changed after replay preparation")
        require(replay.get("capture_excluded_from_performance") is True, "Replay is missing quality-only capture declaration")
        require(integer_equal(candidate.get("schema"), 1) and candidate.get("status") == "completed", "Candidate report is incomplete or failed")
        require(candidate.get("capture_excluded_from_performance") is True, "Candidate capture is not excluded from performance")
        require(candidate.get("case_file_sha256") == replay_hash, "Candidate replay case file identity differs")
        require(candidate.get("temperature") == 0 and candidate.get("dspark") is False
                and candidate.get("prefix_reuse") is False, "Candidate is not the exact non-speculative fresh-state replay")
        vocab = positive_integer(replay.get("vocab_size"), "Replay vocabulary")
        require(vocab <= 2**31 - 1, "Replay vocabulary exceeds int32 token identity encoding")
        eos = token_ids(replay.get("eos_ids"), vocab, "Replay EOS IDs")
        cases = indexed_cases(replay)
        selected_teacher = teacher_rows(teacher, replay.get("teacher_repeat"))
        require(set(selected_teacher) == set(cases), "Teacher case set changed")
        measured = [row for row in candidate.get("results", []) if row.get("warmup") is False]
        require(len(measured) == len(cases) and len({row.get("name") for row in measured}) == len(cases),
                "Candidate must capture exactly one measured replay per case; no partial intersection or duplicate repeats")
        candidates = {row["name"]: row for row in measured}
        require(set(candidates) == set(cases), "Candidate case set differs")
        paths_seen = set()
        case_summaries = {}
        for name, case in cases.items():
            ids = token_ids(case.get("ids"), vocab, "Replay prompt")
            forced = token_ids(case.get("forced_ids"), vocab, "Replay forced continuation")
            require(integer_equal(case.get("max_new"), len(forced)), "Replay cap differs from forced continuation length")
            metadata = case.get("teacher")
            require(isinstance(metadata, dict), "Replay case is missing pinned teacher metadata")
            source = selected_teacher[name]
            require(token_ids(source.get("input_ids"), vocab, "Teacher prompt") == ids
                    and token_ids(source.get("output_token_ids"), vocab, "Teacher continuation") == forced
                    and integer_equal(source.get("prompt_tokens"), len(ids)) and integer_equal(source.get("generated_tokens"), len(forced)),
                    "Replay token identities differ from the teacher report")
            require(metadata.get("request_index") == source.get("request_index")
                    and metadata.get("prompt_i32_sha256") == token_sha(ids)
                    and metadata.get("continuation_i32_sha256") == token_sha(forced), "Pinned teacher input/request identities differ")
            source_capture = source.get("logits_capture", {})
            pinned = metadata.get("capture", {})
            require(integer_equal(pinned.get("rows"), len(forced)) and integer_equal(pinned.get("vocab"), vocab)
                    and integer_equal(pinned.get("first_prediction_position"), len(ids)) and pinned.get("dtype") == "float32-le"
                    and pinned.get("sha256") == source_capture.get("sha256"), "Pinned teacher row layout differs")
            teacher_file = capture_path(pinned.get("path"), replay_path, teacher_root)
            check_payload(teacher_file, len(forced), vocab, pinned.get("sha256"))
            current = candidates[name]
            require(current.get("status") == "completed" and current.get("teacher_forced") is True,
                    "Candidate case did not complete teacher-forced execution")
            require(integer_equal(current.get("prompt_tokens"), len(ids)) and integer_equal(current.get("max_new"), len(forced))
                    and integer_equal(current.get("generated_tokens"), len(forced))
                    and token_ids(current.get("generated_ids"), vocab, "Candidate continuation") == forced,
                    "Candidate prompt/forced continuation identity differs")
            require(integer_equal(current.get("cache_after_reset"), 0) and integer_equal(current.get("cache_after_prefill"), len(ids))
                    and integer_equal(current.get("cache_after_generation"), len(ids) + len(forced) - 1), "Candidate cache positions differ")
            require(current.get("finish_reason") == source.get("finish_reason"), "Candidate and teacher finish reasons differ")
            require(not any(token in eos for token in forced[:-1]), "Replay contains premature EOS")
            captures, argmax_ids = current.get("logit_captures"), current.get("argmax_ids")
            require(isinstance(captures, list) and isinstance(argmax_ids, list)
                    and len(captures) == len(argmax_ids) == len(forced), "Candidate row/argmax count differs")
            teacher_matrix = np.memmap(teacher_file, dtype="<f4", mode="r", shape=(len(forced), vocab))
            case_rows = []
            try:
                for index, (capture, token) in enumerate(zip(captures, forced)):
                    require(integer_equal(capture.get("position"), len(ids) + index) and integer_equal(capture.get("vocab"), vocab)
                            and integer_equal(capture.get("forced_id"), token) and type(argmax_ids[index]) is int
                            and integer_equal(capture.get("argmax"), argmax_ids[index]),
                            "Candidate row position/vocabulary/forced token metadata differs")
                    path = capture_path(capture.get("path"), candidate_path, candidate_root)
                    require(path not in paths_seen, "Candidate reused a capture path for multiple rows")
                    paths_seen.add(path)
                    check_payload(path, 1, vocab, capture.get("sha256"))
                    vector = np.fromfile(path, dtype="<f4")
                    metrics = compare_vectors(teacher_matrix[index], vector, token)
                    row = {"case": name, "row": index, "prediction_position": len(ids) + index,
                           "teacher_request_index": metadata["request_index"], "teacher_byte_offset": index * vocab * 4,
                           "teacher_file_sha256": pinned["sha256"], "candidate_file_sha256": capture["sha256"], **metrics}
                    report["rows"].append(row)
                    require(metrics["finite"], "Nonfinite full-vocabulary logits in " + name + " row " + str(index))
                    require(metrics["candidate_argmax"] == argmax_ids[index], "Candidate recorded argmax does not match captured logits")
                    case_rows.append(row)
            finally:
                del teacher_matrix
            case_summaries[name] = summarize(case_rows)
        report.update(status="compared", coverage="all rows of all replay cases", summary=summarize(report["rows"]),
                      summary_by_case=case_summaries, thresholds=None)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def output_path(value):
    path = Path(value).resolve()
    require(any(path.is_relative_to((ROOT / folder).resolve()) for folder in ("artifacts", "docs/validation")),
            "Generated evidence must stay in this checkout's ignored artifacts/ or docs/validation/")
    require(not path.exists(), "Output already exists; choose a fresh evidence file")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("prepare", help="Freeze the teacher continuation and exact dump mapping for TensorSharp replay")
    prepare.add_argument("--cases", type=Path, required=True)
    prepare.add_argument("--teacher-repeat", type=int, default=0)
    compare = modes.add_parser("compare", help="Compare every teacher-forced full-vocabulary row")
    compare.add_argument("--replay-cases", type=Path, required=True)
    compare.add_argument("--candidate-report", type=Path, required=True)
    compare.add_argument("--candidate-root", type=Path, help="Optional local directory containing copied candidate row files")
    compare.add_argument("--provenance", type=Path, help="Optional declared hardware/checkpoint/build evidence; retained without attestation")
    for command in (prepare, compare):
        command.add_argument("--teacher-report", type=Path, required=True)
        command.add_argument("--teacher-root", type=Path, help="Optional local directory containing copied Maya dump files")
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = output_path(args.output)
    teacher, teacher_hash = load_json(args.teacher_report)
    if args.mode == "prepare":
        original, original_hash = load_json(args.cases)
        replay = prepare_replay(teacher, teacher_hash, original, original_hash, args.teacher_report,
                                args.teacher_repeat, args.teacher_root)
        save_json(output, replay)
        print(json.dumps({"status": "replay_prepared", "cases": len(replay["cases"]), "output": str(output)}))
        return 0
    replay, replay_hash = load_json(args.replay_cases)
    candidate, candidate_hash = load_json(args.candidate_report)
    report = compare_artifacts(replay, replay_hash, teacher, teacher_hash, candidate, candidate_hash,
                               args.replay_cases, args.teacher_report, args.candidate_report,
                               args.teacher_root, args.candidate_root)
    if args.provenance:
        declared, declared_hash = load_json(args.provenance)
        report["declared_external_provenance"] = declared
        report["declared_external_provenance_sha256"] = declared_hash
    save_json(output, report)
    print(json.dumps({"status": report["status"], "rows": len(report["rows"]), "output": str(output),
                      "summary": report.get("summary"), "error": report.get("error")}))
    return 0 if report["status"] == "compared" else 1


if __name__ == "__main__":
    raise SystemExit(main())
