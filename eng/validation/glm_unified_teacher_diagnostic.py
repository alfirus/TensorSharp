"""Validate bounded teacher diagnostics from completed reports and native stderr.

This compares full-logit numerical observations, not a quality acceptance gate.
Native chunk widths are derived from actual cache calls/postcall timers. Binary
immutability and test-run completion/zero skips require the external launcher.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import struct

CASES = ("math", "extraction", "squares", "code")
TASKS = {
    "math": ("Compute 17 + 25. Reply with only the integer.", "42"),
    "extraction": ("Inventory: cedar=3, maple=7, pine=2. Return only the names whose quantity is at least 3, comma-separated in the original order.", "cedar,maple"),
    "squares": ("Return the squares of the integers 1 through 20, in order, as comma-separated integers. Return only the list.", "1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,400"),
    "code": ("Write a Python function square(x) that returns x*x. Return only the function, without Markdown.", "def square(x):\n    return x*x"),
}
ENV_ON = frozenset(("TS_GLM_NATIVE", "TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS", "TS_HOST_MOE_EXPERT_CACHE_TIMING",
    "TS_GLM_FLASH_MTP_FORWARD_TIMING", "TS_HOST_MOE_DECODE_REBALANCE", "TS_HOST_MOE_DECODE_STAGING_72MB", "TS_HOST_MOE_PREFILL_RING_72MB"))
ENV_CLEARED = frozenset(("TS_GLM_TRACE", "TS_GLM_TRACE_DUMP_DIR", "TS_GLM_NODES_PER_LAYER", "TS_GLM_MOE_MMAP", "TS_GLM_TP_SHARD", "TS_GLM_TP_OVERSUBSCRIBE"))
ENV_OFF = frozenset(("TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", "TS_GLM_FLASH_MTP", "TS_GLM_FLASH_MTP_SHAPE_PLAN", "TS_GLM_FLASH_MTP_PREFIX_COMMIT", "TS_GLM_DECODE_GRAPH_RESERVE", "TS_GLM_OP_OFFLOAD", "TS_GGML_STRICT_RECURRENT_PROJECTIONS", "TS_GGML_Q8_ORDERED_SMALL_BATCH", "TS_GGML_Q8_ORDERED_GATE_PREFILL", "TS_GGML_Q8_GATE_PREFILL_TILE16", "TS_GLM_SELECTED_EXPERT_TIMING", "TS_GLM_GRAPH_MEMORY_DIAGNOSTICS", "TS_DSV4_PERF", "TS_HOST_MOE_RAM_CACHE_MB", "TS_HOST_MOE_RAM_CACHE_MAPPED", "TS_HOST_MOE_RAM_CACHE_PINNED", "TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD", "TS_HOST_MOE_RAM_CACHE_VICTIM", "TS_HOST_MOE_RAM_CACHE_FREQUENCY", "TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY", "TS_HOST_MOE_LAYER_PREFETCH", "TS_HOST_MOE_DECODE_PREFETCH", "TS_HOST_MOE_PREFILL_RING_COALESCE16", "TS_HOST_MOE_PREFILL_DOWN_OVERLAP", "TS_HOST_MOE_PHASE_COLD_GROWTH", "TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER", "TS_HOST_MOE_EXPERT_CACHE_PREFETCH", "TS_HOST_MOE_EXPERT_CACHE_LFU", "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS", "TS_HOST_MOE_FILE_READ_WORKERS", "TS_HOST_MOE_FILE_RANDOM_ACCESS", "TS_HOST_MOE_FILE_READ_SORT", "TS_HOST_MOE_ROUTE_TRACE", "TS_HOST_MOE_FILE_SOURCE_VIEW", "TS_HOST_MOE_FILE_SOURCE_VIEW_PREFETCH", "TS_HOST_MOE_VMM_WEIGHTS", "TS_GLM_HOST_TOKEN_EMBEDDING", "TS_GLM_LAYER_MAJOR_DEVICE_BATON", "TS_HOST_MOE_CAUSAL_ROUTE_NEIGHBOR"))
ENV_FIXED = {"TS_HOST_MOE_EXPERT_CACHE_RING_READS": "2", "MAX_CONTEXT": "2048", "TS_UNIFIED_LAYER_MAJOR_TOKENS": "8192"}
LAYERS = tuple(range(3, 45))
PROMPT = 1872
MAX_BYTES = 64 << 20
U64 = (1 << 64) - 1
SCOPE = "bounded_teacher_numerical_diagnostic_not_general_quality_or_timing"
CACHE = "[HOSTMOE-CACHE-TIMING]"
FORWARD = "[GLM-MTP-FORWARD-TIMING]"
MARKER = re.compile(r"\[GLM-TEACHER-([A-Z0-9-]+)\]")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def integer(value, lower=0, upper=U64):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("Invalid bounded integer")
    return value


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Invalid finite number")
    return value


def sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("Invalid SHA256")
    return value


def token_hash(tokens):
    if not isinstance(tokens, list) or len(tokens) > 2048:
        raise ValueError("Invalid bounded token history")
    return hashlib.sha256(b"".join(struct.pack("<i", integer(x, 0, 154879)) for x in tokens)).hexdigest()


def exact_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    raw = Path(path).read_bytes()
    if not raw or len(raw) > 8 << 20:
        raise ValueError("JSON exceeds bounded diagnostic size")
    return json.loads(raw, object_pairs_hook=exact_pairs), hashlib.sha256(raw).hexdigest()


def read_log(path):
    path = Path(path)
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Log exceeds bounded diagnostic size")
    return path.read_text(encoding="utf-8-sig", errors="strict")


def pairs(line, tag):
    suffix = line.split(tag, 1)[1].strip()
    out = {}
    for item in suffix.split():
        if item.count("=") != 1:
            raise ValueError("Malformed diagnostic record")
        key, value = item.split("=")
        if key in out or not key or not value:
            raise ValueError("Duplicate/missing record field")
        out[key] = value
    return out


def uint(values, key):
    value = values[key]
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError("Invalid unsigned record field")
    return integer(int(value))


def validate_inputs(inputs):
    if (inputs["schema"] != 1 or inputs["selection"] !=
            "fixed-answers-before-first-forward; teacher suffix is full rendered-plus-teacher encoding minus exact prompt prefix; first32 suffix tokens"):
        raise ValueError("Wrong frozen input policy")
    for key in ("template_sha256", "source_catalog_sha256"):
        sha(inputs[key])
    integer(inputs["model_length"], 1)
    integer(inputs["model_modified_utc_ticks"], 1)
    if (not inputs["model_path"] or not inputs["architecture"] or not inputs["tokenizer_type"]
            or "BSD-3-Clause" not in inputs["case_provenance"]):
        raise ValueError("Missing source/template provenance")
    catalog = inputs["source_catalog"]
    if [x["layer"] for x in catalog] != list(LAYERS):
        raise ValueError("Expected all42 exact source layers")
    types = collections.Counter()
    for item in catalog:
        for key in ("gate_bytes", "up_bytes", "down_bytes"):
            integer(item[key], 1)
        triple = tuple(integer(item[key]) for key in ("gate_type", "up_type", "down_type"))
        types[triple] += 1
    if types != {(16, 16, 18): 11, (19, 19, 18): 28, (16, 16, 23): 3}:
        raise ValueError("Unexpected source quantization groups")
    # JsonSerializer's exact catalog representation is emitted without indentation.
    catalog_bytes = json.dumps(catalog, separators=(",", ":"), ensure_ascii=True).encode()
    if hashlib.sha256(catalog_bytes).hexdigest() != inputs["source_catalog_sha256"]:
        raise ValueError("Changed source catalog")
    if [x["name"] for x in inputs["cases"]] != list(CASES):
        raise ValueError("Wrong/duplicate teacher set")
    for case in inputs["cases"]:
        prompt, teachers = case["prompt_ids"], case["teacher_ids"]
        if len(prompt) != PROMPT or not 1 <= len(teachers) <= 32:
            raise ValueError("Wrong fixed teacher/prompt bounds")
        integer(case["full_teacher_tokens"], len(teachers))
        if len(teachers) != min(case["full_teacher_tokens"], 32):
            raise ValueError("Teacher prefix was not the predeclared first32 bound")
        if (type(case["teacher_truncated"]) is not bool
                or case["teacher_truncated"] != (case["full_teacher_tokens"] > len(teachers))):
            raise ValueError("Invalid teacher truncation label")
        if ((case["instruction"], case["teacher_text"]) != TASKS[case["name"]]
                or case["instruction"] not in case["rendered_prompt"]):
            raise ValueError("Missing natural final instruction")
        if (token_hash(prompt) != sha(case["prompt_sha256"])
                or token_hash(teachers) != sha(case["teacher_sha256"])
                or token_hash(prompt + teachers) != sha(case["full_history_sha256"])
                or hashlib.sha256(case["rendered_prompt"].encode()).hexdigest() != sha(case["rendered_sha256"])):
            raise ValueError("Changed frozen input/token history")


def validate_report(report, inputs, inputs_hash, profile):
    if (report["schema"] != 1 or report["profile"] != profile or report["scope"] != SCOPE
            or report["status"] != "numerical_diagnostic_unqualified_until_log_validation"
            or report["inputs_sha256"] != inputs_hash or report["model_path"] != inputs["model_path"]):
        raise ValueError("Wrong report/input/profile identity")
    for key in ("general_quality_qualified", "strict_canonical_api_qualified", "performance_qualified",
                "source_file_offsets_independently_captured"):
        if report[key] is not False:
            raise ValueError("Diagnostic must not grant broader quality/provenance credit")
    if report["checked_cache_release_attachment"] is not False:
        raise ValueError("Ordinary production budget attachment must remain unchanged")
    for key in ("checked_model_free_and_ordinary_scope_drained", "native_handle_zero", "binary_identity_requires_external_launcher_pins"):
        if report[key] is not True:
            raise ValueError("Incomplete checked lifecycle/provenance label")
    for key in ("active_allocations", "ledger_reserved", "ledger_committed", "host_embedding"):
        if integer(report[key]) != 0:
            raise ValueError("Unexpected owner/profile residue")
    if (integer(report["requested_context"]) != 2048 or integer(report["resolved_context"]) != 2048
            or integer(report["max_running"]) != 1 or integer(report["requested_prefill"]) != 512
            or integer(report["requested_adaptive_max"]) != (512 if profile == "pp512" else 1024)
            or report["scheduler_chunk_env"] != ("512" if profile == "pp512" else None)):
        raise ValueError("Wrong declared scheduling profile")
    integer(report["resolved_prefill"], 1, 512 if profile == "pp512" else 1024)
    integer(report["resolved_forward_tokens"], PROMPT, 2048)
    if not report["mapped_native_path"] or not report["models_mvid"] or not report["tests_mvid"]:
        raise ValueError("Missing binary labels; external pin validation still required")
    environment = report["environment"]
    if environment["TS_SCHED_PREFILL_CHUNK"] != report["scheduler_chunk_env"]:
        raise ValueError("Environment differs from declared chunk")
    expected_environment = {**dict.fromkeys(ENV_ON, "1"), **dict.fromkeys(ENV_CLEARED),
                            **dict.fromkeys(ENV_OFF, "0"), **ENV_FIXED,
                            "TS_SCHED_PREFILL_CHUNK": report["scheduler_chunk_env"]}
    if environment != expected_environment:
        raise ValueError("Incomplete or confounded diagnostic environment")
    if [x["name"] for x in report["cases"]] != list(CASES):
        raise ValueError("Missing/duplicate report case")
    count, total = 0, 0.0
    for case, source in zip(report["cases"], inputs["cases"]):
        if case["prefill_submissions"] != [PROMPT] or len(case["rows"]) != len(source["teacher_ids"]):
            raise ValueError("Partial prompt or teacher execution")
        subtotal = 0.0
        for index, (row, target) in enumerate(zip(case["rows"], source["teacher_ids"])):
            position = PROMPT + index
            for key, expected in (("index", index), ("position", position), ("target", target),
                                  ("sampled_token", target), ("tokens_forwarded", 1), ("head_after", position + 1)):
                if integer(row[key]) != expected:
                    raise ValueError("Forced input/head mismatch")
            if (row["history_sha256"] != token_hash(source["prompt_ids"] + source["teacher_ids"][:index])
                    or row["forwarded_history_sha256"] != token_hash(source["prompt_ids"] + source["teacher_ids"][:index + 1])):
                raise ValueError("Wrong teacher input chronology")
            score = row["score"]
            if integer(score["target"]) != target:
                raise ValueError("Score after wrong teacher selection")
            integer(score["argmax"], 0, 154879); integer(score["target_rank"], 1, 154880)
            sha(score["logits_sha256"])
            nll, logp = number(score["nll"]), number(score["target_log_probability"])
            number(score["log_normalizer"])
            if nll < 0 or logp > 0 or nll != -logp or number(score["top_margin"]) < 0:
                raise ValueError("Invalid stable full-logit metrics")
            subtotal += nll
        if (not math.isclose(number(case["total_nll"]), subtotal, rel_tol=1e-13, abs_tol=1e-10)
                or not math.isclose(number(case["mean_nll"]), subtotal / len(case["rows"]), rel_tol=1e-13, abs_tol=1e-10)):
            raise ValueError("Wrong per-case NLL aggregate")
        count += len(case["rows"]); total += subtotal
    if (integer(report["teacher_tokens"]) != count
            or not math.isclose(number(report["total_nll"]), total, rel_tol=1e-13, abs_tol=1e-10)
            or not math.isclose(number(report["mean_nll"]), total / count, rel_tol=1e-13, abs_tol=1e-10)):
        raise ValueError("Wrong token-weighted NLL aggregate")


def cache_record(values):
    fields = ("layer", "rows", "joint", "bulk", "file_bytes", "file_windows", "graph_submissions", "file_read_calls", "file_read_tasks", "file_read_completed_tasks",
              "file_read_failed_calls", "file_read_source_bytes", "read_pool_requested", "file_read_random_access", "file_sort_requested", "file_sort_calls",
              "file_sort_changed_calls", "file_sort_misses", "file_sort_bytes", "file_sort_inversions_before", "file_sort_inversions_after",
              "file_view_requested", "file_view_calls", "file_view_bytes", "file_view_fallbacks", "file_view_buffered_bytes", "file_prefetch_requested",
              "file_prefetch_calls", "file_prefetch_ranges", "file_prefetch_bytes", "file_prefetch_failures", "vmm_profile", "vmm_capacity",
              "vmm_physical_bytes", "vmm_pending_bytes", "vmm_quarantine_bytes", "vmm_certificate")
    out = {key: uint(values, key) for key in fields}
    if (values["outcome"] != "completed" or values["scope"] != "nested_host_wall_not_cuda_kernel_or_physical_io"
            or out["layer"] not in LAYERS or not 1 <= out["rows"] <= 1024
            or out["bulk"] != int(out["rows"] > 1) or out["joint"] != 0 or out["graph_submissions"] != 1):
        raise ValueError("Unsupported or failed source call")
    zeros = [key for key in fields if key.startswith(("file_sort_", "file_prefetch_", "vmm_"))]
    zeros += ["read_pool_requested", "file_read_random_access", "file_read_failed_calls", "file_view_requested", "file_view_calls", "file_view_bytes", "file_view_fallbacks"]
    if any(out[key] for key in zeros):
        raise ValueError("Confounded/failed file source path")
    size = out["file_bytes"]
    if (out["file_read_tasks"] != out["file_read_completed_tasks"]
            or out["file_read_source_bytes"] != size or out["file_view_buffered_bytes"] != size
            or (size > 0 and (not 1 <= out["file_read_calls"] <= out["file_read_tasks"] or not out["file_windows"]))
            or (size == 0 and any(out[key] for key in ("file_read_calls", "file_read_tasks", "file_windows")))):
        raise ValueError("Unjoined or nonconserving logical source reads")
    return out


def forward_record(values):
    out = {key: uint(values, key) for key in ("kind", "rows", "p0", "nout", "wanth", "logits")}
    if (out["kind"] != 0 or out["nout"] != 1 or out["wanth"] != 0 or out["logits"] not in (0, 1)
            or values["outcome"] != "completed" or values["scope"] != "host_wall_including_existing_sync_and_expert_callbacks"):
        raise ValueError("Failed or unsupported actual Forward")
    return out


def validate_log(text, report, inputs, inputs_hash):
    if len(text.encode()) > MAX_BYTES:
        raise ValueError("Log bound exceeded")
    profile = report["profile"]
    ordinal, current, mode, index, done = 0, None, None, 0, False
    frames, timers, proofs, pending = [], [], [], []
    for line in text.splitlines():
        marker = MARKER.search(line)
        if marker:
            kind = marker.group(1); values = pairs(line, marker.group())
            if kind == "CASE-BEGIN":
                if current is not None or done or ordinal >= 4 or values != {
                        "case": CASES[ordinal], "profile": profile, "prompt_sha": inputs["cases"][ordinal]["prompt_sha256"],
                        "teacher_sha": inputs["cases"][ordinal]["teacher_sha256"], "inputs_sha": inputs_hash}:
                    raise ValueError("Invalid/duplicate case begin")
                current, index = CASES[ordinal], 0
            elif kind == "PP-BEGIN":
                if current is None or mode is not None or index != 0 or values != {"case": current, "rows": str(PROMPT)}:
                    raise ValueError("Unexpected prefill begin")
                mode = "pp"; frames, timers, pending = [], [], []
            elif kind == "PP-END":
                if mode != "pp" or values != {"case": current, "head": str(PROMPT)} or pending:
                    raise ValueError("Incomplete prefill")
                sequence = [x["rows"] for x in frames if x["layer"] == 3]
                if (not sequence or any(not 9 <= x <= 1024 for x in sequence) or sum(sequence) != PROMPT
                        or (profile == "pp512" and sequence != [512, 512, 512, 336])
                        or (profile == "adaptive1024" and max(sequence) <= 512)):
                    raise ValueError("Requested width did not establish actual native profile")
                expected_frames = [(layer, width) for layer in LAYERS for width in sequence]
                if [(x["layer"], x["rows"]) for x in frames] != expected_frames:
                    raise ValueError("Not exactly all42 consistent native source chunks")
                positions, p0 = [], 0
                for width in sequence:
                    positions.append((width, p0)); p0 += width
                if [(x["rows"], x["p0"]) for x in timers] != positions * 45:
                    raise ValueError("Not exact45-layer actual Forward chronology")
                if [x["logits"] for x in timers] != [0] * (len(timers) - 1) + [1]:
                    raise ValueError("Wrong prefill last-logit publication")
                proofs.append({"case": current, "actual_pp_chunks": sequence, "selected_source_frames": len(frames),
                               "pp_logical_file_bytes": sum(x["file_bytes"] for x in frames), "n1_logical_file_bytes": 0, "n1_frames": 0})
                mode = "after_pp"
            elif kind == "N1-BEGIN":
                if mode != "after_pp" or current is None or index >= len(report["cases"][ordinal]["rows"]):
                    raise ValueError("Unexpected teacher decode begin")
                row = report["cases"][ordinal]["rows"][index]
                expected = {"case": current, "index": str(index), "p0": str(row["position"]), "target": str(row["target"]),
                            "history_sha": row["history_sha256"], "logits_sha": row["score"]["logits_sha256"]}
                if values != expected:
                    raise ValueError("Teacher score marker differs from frozen history")
                mode = "n1"; frames, timers, pending = [], [], []
            elif kind == "N1-END":
                row = report["cases"][ordinal]["rows"][index]
                if (mode != "n1" or pending or values != {"case": current, "index": str(index), "p0": str(row["position"]),
                    "target": str(row["target"]), "head": str(row["head_after"])}
                        or [(x["layer"], x["rows"]) for x in frames] != [(layer, 1) for layer in LAYERS]
                        or [(x["rows"], x["p0"], x["logits"]) for x in timers] != [(1, PROMPT + index, 1)]):
                    raise ValueError("Incomplete/nonconsecutive actual teacher Forward")
                proofs[-1]["n1_logical_file_bytes"] += sum(x["file_bytes"] for x in frames)
                proofs[-1]["n1_frames"] += len(frames)
                mode = "after_pp"; index += 1
            elif kind == "CASE-END":
                if (mode != "after_pp" or index != len(report["cases"][ordinal]["rows"])
                        or values != {"case": current, "profile": profile, "tokens": str(index), "outcome": "completed"}):
                    raise ValueError("Missing teacher steps/case completion")
                current, mode = None, None; ordinal += 1
            elif kind == "DONE":
                if (ordinal != 4 or current is not None or done
                        or values != {"profile": profile, "cases": "4", "checked_model_free": "1", "scope": SCOPE}):
                    raise ValueError("Invalid final completed marker")
                done = True
            else:
                raise ValueError("Unknown teacher marker")
        elif CACHE in line or FORWARD in line:
            if current is None:
                continue  # Loader/warm-up/teardown outside explicit case bounds.
            if mode not in ("pp", "n1"):
                raise ValueError("Native compute outside explicit scheduler step")
            if CACHE in line:
                frame = cache_record(pairs(line, CACHE)); frames.append(frame); pending.append(frame)
                if mode == "pp" and len(pending) != 1:
                    raise ValueError("Missing per-layer prefill postcall timer")
            else:
                timer = forward_record(pairs(line, FORWARD)); timers.append(timer)
                if mode == "pp":
                    if pending:
                        if pending[0]["rows"] != timer["rows"]:
                            raise ValueError("Prefill source/timer rows differ")
                        pending.clear()
                    elif frames:
                        raise ValueError("Unpaired extra timer among selected layers")
                else:
                    if [x["layer"] for x in pending] != list(LAYERS):
                        raise ValueError("Incomplete N1 postcall source block")
                    pending.clear()
    if not done or ordinal != 4 or current is not None:
        raise ValueError("No complete four-case checked diagnostic")
    return proofs


def compare(inputs, inputs_hash, fixed, fixed_log, adaptive, adaptive_log):
    validate_inputs(inputs)
    for report, profile in ((fixed, "pp512"), (adaptive, "adaptive1024")):
        validate_report(report, inputs, inputs_hash, profile)
    for key in ("mapped_native_path", "models_mvid", "tests_mvid", "model_path"):
        if fixed[key] != adaptive[key]:
            raise ValueError("Two fresh-process arms have different binary/model labels")
    fixed_env, adaptive_env = dict(fixed["environment"]), dict(adaptive["environment"])
    fixed_env.pop("TS_SCHED_PREFILL_CHUNK"); adaptive_env.pop("TS_SCHED_PREFILL_CHUNK")
    if fixed_env != adaptive_env:
        raise ValueError("Other experimental behavior differs")
    proofs = {"pp512": validate_log(fixed_log, fixed, inputs, inputs_hash),
              "adaptive1024": validate_log(adaptive_log, adaptive, inputs, inputs_hash)}
    observations = []
    for left, right in zip(fixed["cases"], adaptive["cases"]):
        rows = []
        for a, b in zip(left["rows"], right["rows"]):
            rows.append({"index": a["index"], "position": a["position"], "target": a["target"],
                         "pp512_nll": a["score"]["nll"], "adaptive_nll": b["score"]["nll"],
                         "delta_nll": b["score"]["nll"] - a["score"]["nll"],
                         "pp512_argmax": a["score"]["argmax"], "adaptive_argmax": b["score"]["argmax"],
                         "argmax_equal": a["score"]["argmax"] == b["score"]["argmax"],
                         "pp512_margin": a["score"]["top_margin"], "adaptive_margin": b["score"]["top_margin"],
                         "raw_logits_bit_hash_equal": a["score"]["logits_sha256"] == b["score"]["logits_sha256"]})
        observations.append({"case": left["name"], "teacher_tokens": len(rows), "rows": rows,
                             "pp512_total_nll": left["total_nll"], "adaptive_total_nll": right["total_nll"],
                             "delta_total_nll": right["total_nll"] - left["total_nll"]})
    return {"schema": 1, "diagnostic_complete": True, "scope": SCOPE, "inputs_sha256": inputs_hash,
            "binary_labels_equal": True, "binary_immutability_externally_qualified": False,
            "test_runner_zero_skips_externally_qualified": False,
            "actual_chunk_and_source_closure": proofs, "cases": observations,
            "teacher_tokens": fixed["teacher_tokens"], "pp512_mean_nll": fixed["mean_nll"],
            "adaptive_mean_nll": adaptive["mean_nll"], "delta_mean_nll": adaptive["mean_nll"] - fixed["mean_nll"],
            "general_quality_qualified": False, "strict_canonical_api_qualified": False, "performance_qualified": False,
            "physical_io_or_residency_qualified": False,
            "limits": ["Four padded authored cases and fixed teacher prefixes are a bounded numerical diagnostic.",
                       "No full vocabulary values are retained, only stable scalar scores and raw-vector hashes.",
                       "Joined source counters prove logical copies, not SSD traffic or kernel/CPU-exclusive time.",
                       "Native file offsets are not independently emitted; model/catalog and external immutable binary pins remain necessary."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "fixed-report", "fixed-log", "adaptive-report", "adaptive-log", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    inputs, inputs_hash = read_json(args.inputs)
    fixed, fixed_hash = read_json(args.fixed_report); adaptive, adaptive_hash = read_json(args.adaptive_report)
    result = compare(inputs, inputs_hash, fixed, read_log(args.fixed_log), adaptive, read_log(args.adaptive_log))
    result["report_sha256"] = {"pp512": fixed_hash, "adaptive1024": adaptive_hash}
    result["log_sha256"] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in
                            (("pp512", args.fixed_log), ("adaptive1024", args.adaptive_log))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")


if __name__ == "__main__":
    main()
