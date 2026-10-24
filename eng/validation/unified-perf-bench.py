#!/usr/bin/env python3
"""Sequential prefill/decode throughput bench for one owned TensorSharp server.

Starts TensorSharp.Server.Host (optionally with --unified-memory), sends a fixed
plan of requests to /api/chat/ollama one at a time, and records the server's own
attributed timings (prompt_eval_* / eval_*), cache hits, wall time, process
memory and board VRAM. Requests never overlap, so GPU work is never concurrent.

Prefill tok/s is only reported for requests with zero prompt-cache hits; decode
tok/s is eval_count / eval_duration as the server reports it (it includes the
final EOS forward). Results go to <output>/report.json; keep outputs under the
ignored artifacts/ directory.

Example (unified Qwen3.8 Flash Next, no prefix cache, fast startup):
  python eng/validation/unified-perf-bench.py --unified-memory \
      --model C:/Works/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf \
      --output artifacts/perf/qwen-baseline-v1
"""
import argparse
import hashlib
import http.client
import importlib.util
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

_decode_reserve_spec = importlib.util.spec_from_file_location(
    "glm_decode_graph_reserve", Path(__file__).with_name("glm_decode_graph_reserve.py"))
_decode_reserve_helper = importlib.util.module_from_spec(_decode_reserve_spec)
_decode_reserve_spec.loader.exec_module(_decode_reserve_helper)
DECODE_GRAPH_RESERVE_ENV = _decode_reserve_helper.DECODE_GRAPH_RESERVE_ENV
decode_graph_reserve_enabled = _decode_reserve_helper.decode_graph_reserve_enabled
decode_graph_reserve_records = _decode_reserve_helper.decode_graph_reserve_records
expert_recovery_records = _decode_reserve_helper.expert_recovery_records
decode_graph_reserve_qualification = _decode_reserve_helper.decode_graph_reserve_qualification

_file_workers_spec = importlib.util.spec_from_file_location(
    "glm_file_read_workers", Path(__file__).with_name("glm_file_read_workers.py"))
_file_workers_helper = importlib.util.module_from_spec(_file_workers_spec)
_file_workers_spec.loader.exec_module(_file_workers_helper)

_retain_peer_spec = importlib.util.spec_from_file_location(
    "glm_decode_retain_peer", Path(__file__).with_name("glm_decode_retain_peer.py"))
_retain_peer_helper = importlib.util.module_from_spec(_retain_peer_spec)
_retain_peer_spec.loader.exec_module(_retain_peer_helper)

_vmm_weights_spec = importlib.util.spec_from_file_location(
    "glm_vmm_weights", Path(__file__).with_name("glm_vmm_weights.py"))
_vmm_weights_helper = importlib.util.module_from_spec(_vmm_weights_spec)
_vmm_weights_spec.loader.exec_module(_vmm_weights_helper)

_file_sort_spec = importlib.util.spec_from_file_location(
    "glm_file_read_sort", Path(__file__).with_name("glm_file_read_sort.py"))
_file_sort_helper = importlib.util.module_from_spec(_file_sort_spec)
_file_sort_spec.loader.exec_module(_file_sort_helper)

_file_view_spec = importlib.util.spec_from_file_location(
    "glm_file_source_view", Path(__file__).with_name("glm_file_source_view.py"))
_file_view_helper = importlib.util.module_from_spec(_file_view_spec)
_file_view_spec.loader.exec_module(_file_view_helper)

_device_baton_spec = importlib.util.spec_from_file_location(
    "glm_layer_major_device_baton", Path(__file__).with_name("glm_layer_major_device_baton.py"))
_device_baton_helper = importlib.util.module_from_spec(_device_baton_spec)
_device_baton_spec.loader.exec_module(_device_baton_helper)

_causal_neighbor_spec = importlib.util.spec_from_file_location(
    "glm_causal_route_neighbor", Path(__file__).with_name("glm_causal_route_neighbor.py"))
_causal_neighbor_helper = importlib.util.module_from_spec(_causal_neighbor_spec)
_causal_neighbor_spec.loader.exec_module(_causal_neighbor_helper)

try:
    import psutil
except ImportError:  # memory sampling is optional
    psutil = None

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER = ROOT / "TensorSharp.Server.Host" / "bin" / "TensorSharp.Server.Host.exe"

# Distinct natural/technical text windows so each long prompt routes to a
# realistic, different expert mix (repeated synthetic text inflates hit rates).
SOURCES = {
    "corpus": ROOT / "benchmarks" / "engine_comparison" / "assets" / "prefill_corpus.txt",
    "paged": ROOT / "docs" / "PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md",
    "spec": ROOT / "docs" / "speculative_decoding.md",
}

PLANS = {
    # name, source, start char, length chars, instruction, max new tokens
    "default": [
        ("prefill-a", "corpus", 0, 9000, "Summarize the document above in five sentences.", 64),
        ("prefill-b", "paged", 2000, 9000, "Summarize the document above in five sentences.", 64),
        ("decode-a", None, 0, 0, "Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall.", 256),
        ("prefill-c", "spec", 4000, 9000, "Summarize the document above in five sentences.", 64),
        ("decode-b", None, 0, 0, "Explain in detail how a modern CPU executes instructions, covering pipelining, caches and branch prediction.", 256),
    ],
    "quick": [
        ("prefill-a", "corpus", 0, 9000, "Summarize the document above in five sentences.", 32),
        ("decode-a", None, 0, 0, "Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall.", 128),
    ],
    "prefill-a": [
        ("prefill-a", "corpus", 0, 9000, "Summarize the document above in five sentences.", 32),
    ],
    "prefill-c": [
        ("prefill-c", "spec", 4000, 9000, "Summarize the document above in five sentences.", 8),
    ],
    "decode": [
        ("decode-a", None, 0, 0, "Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall.", 256),
    ],
    "decode-short": [
        ("decode-a", None, 0, 0, "Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall.", 32),
    ],
    # Two different topics (different expert mixes); use with --concurrent.
    "decode-pair": [
        ("decode-a", None, 0, 0, "Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall.", 384),
        ("decode-b", None, 0, 0, "Explain in detail how a modern CPU executes instructions, covering pipelining, caches and branch prediction.", 384),
    ],
    "long": [
        ("prefill-4k", "spec", 0, 18000, "Summarize the document above in five sentences.", 64),
        ("decode-a", None, 0, 0, "Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall.", 256),
    ],
}

_semantic_spec = importlib.util.spec_from_file_location("fixed_task_semantic",
    Path(__file__).with_name("fixed-task-semantic.py"))
FIXED_TASKS = importlib.util.module_from_spec(_semantic_spec)
_semantic_spec.loader.exec_module(FIXED_TASKS)
PLANS["semantic"] = [(name, None, 0, 0, prompt, 256)
                     for name, prompt in FIXED_TASKS.PROMPTS.items()]


def semantic_result(name, prompt, response, request_log):
    """Known answer oracle plus exactly one matching natural-EOS log record.

    Call only with an isolated sequential request's newly appended log bytes.
    Protocol 'stop' alone cannot establish the engine's completion reason.
    """
    message = response.get("message") or {}
    answer = FIXED_TASKS.evaluate(name, prompt, message.get("content"))
    timings = [line for line in request_log.splitlines() if "generation.timings " in line]
    fields = dict(re.findall(r"(\w+)=([^\s]+)", timings[0])) if len(timings) == 1 else {}
    eos = (fields.get("finishReason") == "eos"
           and fields.get("engineCompletionAvailable") == "True"
           and fields.get("promptTokens") == str(response.get("prompt_eval_count"))
           and fields.get("generatedTokens") == str(response.get("eval_count")))
    complete = response.get("done") is True and response.get("done_reason") == "stop"
    return dict(answer=answer, engine_eos=eos, complete=complete,
                log_records=len(timings),
                passed=answer["passed"] and eos and complete and not message.get("tool_calls"))


def speculation_records(request_log):
    pattern = (r"Speculative decoding stats for (?P<request>[^:\r\n]+): "
               r"drafted=(?P<drafted>\d+) accepted=(?P<accepted>\d+) "
               r"acceptance=.*?verifySteps=(?P<verify_steps>\d+) "
               r"plainSteps=(?P<plain_steps>\d+) rollbacks=(?P<rollbacks>\d+)")
    records = []
    for line in request_log.splitlines():
        match = re.search(pattern, line)
        if match is None:
            continue
        record = {key: value if key == "request" else int(value)
                  for key, value in match.groupdict().items()}
        phases = re.search(r"\| phaseMs (.*?) \| governor ", line)
        if phases:
            values = dict(re.findall(r"(draft|verify|snapshot|rollback|catchUp|plain)=([0-9]+(?:\.[0-9]+)?)", phases[1]))
            if len(values) == 6:
                record["phase_ms"] = {key: float(value) for key, value in values.items()}
        records.append(record)
    return records


def cache_timing_summary(request_log):
    """Nested host wall observations, grouped by exact layer and row width.

    Stage totals overlap: they must not be added to infer request time, CUDA
    kernel time or physical SSD I/O. Keep failed/incomplete calls explicit.
    """
    fields = ("total_ms", "plan_ms", "allocation_ms", "wide_replace_ms", "file_pipeline_ms",
              "upload_enqueue_ms", "upload_wait_ms", "input_bridge_ms", "compute_ms", "output_bridge_ms")
    counters = ("file_bytes", "file_windows", "graph_submissions")
    groups, malformed = {}, 0
    for line in request_log.splitlines():
        if "[HOSTMOE-CACHE-TIMING]" not in line:
            continue
        values = dict(re.findall(r"(\w+)=([^\s]+)", line))
        try:
            key = (int(values["layer"]), int(values["rows"]), values["outcome"])
            stages = {field: float(values[field]) for field in fields}
            counts = {field: int(values[field]) for field in counters}
            if any(value < 0 or not value < float("inf") for value in (*stages.values(), *counts.values())):
                raise ValueError("invalid observation")
        except (KeyError, ValueError):
            malformed += 1
            continue
        group = groups.setdefault(key, dict(layer=key[0], rows=key[1], outcome=key[2], calls=0,
            **{field: 0 for field in fields + counters}))
        group["calls"] += 1
        for field, value in (stages | counts).items():
            group[field] += value
    return dict(scope="Nested host wall stages and logical file payload; not additive, CUDA kernel time or physical disk attribution",
                malformed_records=malformed, groups=[groups[key] for key in sorted(groups)])


def scalar_staging_summary(request_log):
    """Prove an actual scalar file transfer exceeded the ordinary arena."""
    calls, file_bytes = 0, 0
    for line in request_log.splitlines():
        if "[HOSTMOE-CACHE-TIMING]" not in line:
            continue
        values = dict(re.findall(r"(\w+)=([^\s]+)", line))
        try:
            if (values["outcome"] == "completed" and int(values["rows"]) == 1
                    and int(values["file_windows"]) == 1
                    and 32 * 2**20 < int(values["file_bytes"]) <= 72 * 2**20):
                calls += 1
                file_bytes += int(values["file_bytes"])
        except (KeyError, ValueError):
            continue  # Invalid records cannot provide activity credit.
    return dict(expanded_scalar_calls=calls, logical_file_bytes=file_bytes,
                scope="Completed scalar calls reading more than 32 MiB in one window; not physical SSD attribution")


def prefill_ring_summary(request_log):
    """Count completed wide fills using the actual two-36-MiB plan.

    A retained arena alone is not activity. Failed enclosing calls and the
    deliberately smaller fault-test plans cannot qualify an enabled run.
    """
    calls = expanded = file_bytes = windows = malformed = 0
    arena_peak = slot_peak = 0
    for line in request_log.splitlines():
        if "[HOSTMOE-CACHE-TIMING]" not in line:
            continue
        values = dict(re.findall(r"(\w+)=([^\s]+)", line))
        fields = ("ring_calls", "ring_expanded_calls", "ring_arena_peak_bytes", "ring_slot_peak_bytes")
        if not any(field in values for field in fields):
            continue  # Older binaries supply no activity evidence.
        try:
            rc, ec, arena, slot = (int(values[field]) for field in fields)
            payload, count = int(values["file_bytes"]), int(values["file_windows"])
            rows, bulk, joint = int(values["rows"]), int(values["bulk"]), int(values["joint"])
            if (min(rc, ec, arena, slot, payload, count) < 0 or ec > rc
                    or arena > 72 * 2**20 or (rc and (slot < 1 or arena - 2 * slot not in (0, 1)))
                    or (ec and (arena != 72 * 2**20 or slot != 36 * 2**20))):
                raise ValueError("invalid ring observation")
        except (KeyError, ValueError):
            malformed += 1
            continue
        if values.get("outcome") != "completed" or rows <= 1 or bulk != 1 or joint != 0:
            continue
        calls += rc
        arena_peak, slot_peak = max(arena_peak, arena), max(slot_peak, slot)
        if ec and payload > 32 * 2**20 and count > 0:
            expanded += ec
            file_bytes += payload
            windows += count
    return dict(completed_ring_calls=calls, expanded_ring_calls=expanded,
                logical_file_bytes_in_expanded_calls=file_bytes, windows_in_expanded_calls=windows,
                actual_arena_peak_bytes=arena_peak, actual_slot_peak_bytes=slot_peak,
                malformed_records=malformed,
                scope="Completed nonjoint wide calls with actual ring plans; logical payload, not physical SSD attribution")


def coalesced_ring_summary(request_log):
    """Require actual fewer source reads with the same complete logical payload."""
    fields = ("ring_coalesced_calls", "ring_read_tasks_before", "ring_read_tasks_after",
              "ring_coalesced_payload_bytes", "ring_coalesced_bytes", "ring_coalesced_read_cap_bytes")
    calls = before = after = payload_bytes = merged_bytes = malformed = 0
    for line in request_log.splitlines():
        if "[HOSTMOE-CACHE-TIMING]" not in line:
            continue
        values = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if not any(field in values for field in fields):
            continue
        try:
            cc, old, new, payload, merged, cap = (int(values[field]) for field in fields)
            rc, ec = int(values["ring_calls"]), int(values["ring_expanded_calls"])
            arena, slot = int(values["ring_arena_peak_bytes"]), int(values["ring_slot_peak_bytes"])
            rows, bulk, joint = int(values["rows"]), int(values["bulk"]), int(values["joint"])
            file_bytes = int(values["file_bytes"])
            if (min(cc, old, new, payload, merged, cap, rc, ec, arena, slot, file_bytes) < 0
                    or cc > ec or ec > rc or new > old
                    or (not cc and any((old, new, payload, merged, cap)))
                    or (cc and (old <= new or new == 0 or payload <= 72 * 2**20
                                or not 0 < merged <= payload or payload != file_bytes or cap != 16 * 2**20
                                or arena != 72 * 2**20 or slot != 36 * 2**20
                                or rows <= 1 or bulk != 1 or joint != 0))):
                raise ValueError("invalid coalesced read observation")
        except (KeyError, ValueError):
            malformed += 1
            continue
        if values.get("outcome") != "completed":
            continue
        calls += cc
        before += old
        after += new
        payload_bytes += payload
        merged_bytes += merged
    return dict(completed_coalesced_calls=calls, read_tasks_before=before, read_tasks_after=after,
                logical_payload_bytes=payload_bytes, logical_merged_bytes=merged_bytes,
                malformed_records=malformed,
                scope="Completed wide ring reads; actual tasks and exact logical payload, not physical SSD attribution")


STRICT_GRAPH_CAP_ENV = "TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB"
HOST_TOKEN_EMBEDDING_ENV = "TS_GLM_HOST_TOKEN_EMBEDDING"
HOST_TOKEN_EMBEDDING_SCOPE = "raw_rows_uploaded_not_compute_completion_or_physical_io"
STRICT_GRAPH_CAP_SCOPE = "wrapped_declared_payload_not_total_vram"
STRICT_GRAPH_CAP_NUMBERS = ("profile", "cap_bytes", "submitted", "chunk", "context",
                            "graph_builds", "graph_reuses", "peak_graph_device_bytes", "cap_refusals")


def strict_layer_major_graph_cap_bytes(value):
    if value not in (None, "0", "512", "768"):
        raise ValueError(STRICT_GRAPH_CAP_ENV + " must be unset, 0, 512 or 768")
    return int(value or "0") * 2**20


def strict_layer_major_graph_cap_records(request_log, requested_cap):
    """Parse only newly appended request log bytes; these events are call-local."""
    records, malformed = [], 0
    required = set(STRICT_GRAPH_CAP_NUMBERS) | {"outcome", "scope"}
    for line in request_log.splitlines():
        marker = "[GLM-STRICT-GRAPH-CAP]"
        if marker not in line:
            continue
        tokens = line.split(marker, 1)[1].split()
        pairs = [token.split("=", 1) for token in tokens]
        try:
            if (any(len(pair) != 2 for pair in pairs)
                    or len(pairs) != len(required) or {pair[0] for pair in pairs} != required):
                raise ValueError("missing, duplicate or unexpected strict cap field")
            values = dict(pairs)
            if (any(re.fullmatch(r"[0-9]+", values[key]) is None for key in STRICT_GRAPH_CAP_NUMBERS)
                    or values["outcome"] not in ("completed", "failed")
                    or values["scope"] != STRICT_GRAPH_CAP_SCOPE):
                raise ValueError("invalid strict cap field")
            record = {key: int(values[key]) for key in STRICT_GRAPH_CAP_NUMBERS}
            record.update(outcome=values["outcome"], scope=values["scope"])
            if (any(record[key] > 2**64 - 1 for key in STRICT_GRAPH_CAP_NUMBERS)
                    or record["graph_builds"] > 2**64 - 1 - record["graph_reuses"]):
                raise ValueError("overflowing strict cap field")
            if record["outcome"] == "completed" and (
                    record["profile"] != 1 or record["cap_bytes"] != requested_cap
                    or requested_cap not in (512 * 2**20, 768 * 2**20)
                    or record["chunk"] != 512 or record["context"] != 2048
                    or not 512 < record["submitted"] <= record["context"]
                    or 0 < record["submitted"] % 512 < 9
                    or not record["graph_builds"] + record["graph_reuses"]
                    or not 0 < record["peak_graph_device_bytes"] <= requested_cap
                    or record["cap_refusals"]):
                raise ValueError("unqualified completed strict cap call")
        except (KeyError, ValueError):
            malformed += 1
            continue
        records.append(record)
    return dict(records=records, malformed_records=malformed,
                scope="Call-local wrapped declared graph payload only; not total VRAM or physical free memory")


def strict_layer_major_graph_cap_qualification(rows, requested_cap):
    observations = [row.get("strict_layer_major_graph_cap", {}) for row in rows]
    records = [record for observation in observations for record in observation.get("records", [])]
    completed = [record for record in records if record["outcome"] == "completed"]
    failed = sum(record["outcome"] == "failed" for record in records)
    malformed = sum(observation.get("malformed_records", 0) for observation in observations)
    mismatches = sum(record["cap_bytes"] != requested_cap for record in completed)
    return dict(passed=bool(completed) and requested_cap in (512 * 2**20, 768 * 2**20)
                and not failed and not malformed and not mismatches,
                requested_cap_bytes=requested_cap, completed_calls=len(completed), failed_calls=failed,
                malformed_records=malformed, cap_mismatches=mismatches,
                peak_graph_device_bytes=max((record["peak_graph_device_bytes"] for record in completed), default=0),
                graph_builds=sum(record["graph_builds"] for record in completed),
                graph_reuses=sum(record["graph_reuses"] for record in completed),
                scope="Call-local wrapped declared graph payload only; not total VRAM or physical free memory")


def host_token_embedding_enabled(value):
    if value not in (None, "0", "1"):
        raise ValueError(HOST_TOKEN_EMBEDDING_ENV + " must be unset, 0 or 1")
    return value == "1"


def host_token_embedding_records(request_log):
    """Uploaded raw rows only; successful request/output checks remain separate."""
    numbers = ("version", "source_bytes", "row_bytes", "rows", "packed_bytes",
               "host_pack_bytes", "device_pack_max_bytes")
    required = set(numbers) | {"mode", "scope"}
    records, malformed = [], 0
    for line in request_log.splitlines():
        marker = "[GLM-HOST-TOKEN-EMBEDDING]"
        if marker not in line:
            continue
        try:
            pairs = [token.split("=", 1) for token in line.split(marker, 1)[1].split()]
            if (any(len(pair) != 2 for pair in pairs) or len(pairs) != len(required)
                    or {pair[0] for pair in pairs} != required):
                raise ValueError("missing, duplicate or unexpected raw embedding field")
            values = dict(pairs)
            if (any(re.fullmatch(r"[0-9]+", values[key]) is None for key in numbers)
                    or values["mode"] != "raw_q4k_cuda_get_rows"
                    or values["scope"] != HOST_TOKEN_EMBEDDING_SCOPE):
                raise ValueError("invalid raw embedding field")
            record = {key: int(values[key]) for key in numbers}
            if (any(value > 2**64 - 1 for value in record.values())
                    or record["version"] != 1 or record["source_bytes"] != 356843520
                    or record["row_bytes"] != 2304 or not 1 <= record["rows"] <= 512
                    or record["packed_bytes"] != record["rows"] * record["row_bytes"]
                    or record["host_pack_bytes"] != 1179648
                    or record["device_pack_max_bytes"] != 1179648):
                raise ValueError("unqualified raw embedding shape or footprint")
            records.append(record)
        except (KeyError, ValueError):
            malformed += 1
    return dict(records=records, malformed_records=malformed,
                scope="Raw packed rows uploaded; no compute completion, physical RAM residency or SSD attribution")


def host_token_embedding_qualification(rows):
    requests = []
    for row in rows:
        observation = row.get("host_token_embedding", {})
        records = observation.get("records", [])
        packed_rows = sum(record["rows"] for record in records)
        packed_bytes = sum(record["packed_bytes"] for record in records)
        prompt = row.get("prompt_tokens", 0)
        requests.append(dict(name=row.get("name"), packed_rows=packed_rows, packed_bytes=packed_bytes,
            upload_calls=len(records), malformed_records=observation.get("malformed_records", 0),
            passed=bool(records) and isinstance(prompt, int) and not isinstance(prompt, bool) and prompt > 0
                and row.get("cache_hits") == 0 and packed_rows >= prompt
                and packed_rows <= 2**64 - 1 and packed_bytes <= 2**64 - 1
                and observation.get("malformed_records", 0) == 0))
    return dict(passed=bool(requests) and all(request["passed"] for request in requests), requests=requests,
                scope="Per measured successful request raw uploads cover its uncached prompt; output quality checked separately")


PREFILL_DOWN_OVERLAP_ENV = "TS_HOST_MOE_PREFILL_DOWN_OVERLAP"
PREFILL_DOWN_OVERLAP_SCOPE = "exact_selected_payload_and_host_wall_not_kernel_overlap_or_physical_io"


def prefill_down_overlap_enabled(value):
    if value not in (None, "0", "1"):
        raise ValueError(PREFILL_DOWN_OVERLAP_ENV + " must be unset, 0 or 1")
    return value == "1"


def prefill_down_overlap_records(request_log):
    integers = ("profile", "layer", "rows", "selected_misses", "prefix_submissions", "suffix_submissions",
                "gate_up_bytes", "down_bytes", "gate_up_windows", "down_windows", "arena_bytes",
                "slot_bytes", "distinct_upload_backend")
    times = ("gate_up_file_ms", "prefix_submit_ms", "down_file_ms", "join_ms", "suffix_compute_ms")
    required = set(integers + times) | {"outcome", "scope"}
    records, malformed = [], 0
    for line in request_log.splitlines():
        marker = "[HOSTMOE-PREFILL-DOWN-OVERLAP]"
        if marker not in line:
            continue
        try:
            pairs = [token.split("=", 1) for token in line.split(marker, 1)[1].split()]
            if (any(len(pair) != 2 for pair in pairs) or len(pairs) != len(required)
                    or {pair[0] for pair in pairs} != required):
                raise ValueError("missing, duplicate or unexpected overlap field")
            values = dict(pairs)
            if (any(re.fullmatch(r"[0-9]+", values[key]) is None for key in integers)
                    or values["scope"] != PREFILL_DOWN_OVERLAP_SCOPE
                    or values["outcome"] not in ("completed", "failed")):
                raise ValueError("invalid overlap field")
            record = {key: int(values[key]) for key in integers}
            record.update({key: float(values[key]) for key in times})
            record["outcome"] = values["outcome"]
            if (any(record[key] > 2**64 - 1 for key in integers)
                    or any(not math.isfinite(record[key]) or record[key] < 0 for key in times)
                    or record["profile"] != 1 or not 0 <= record["layer"] <= 2**31 - 1
                    or not 9 <= record["rows"] <= 512
                    or not 1 <= record["selected_misses"] <= min(288, record["rows"] * 8)
                    or record["prefix_submissions"] not in (0, 1)
                    or record["suffix_submissions"] not in (0, 1)
                    or record["suffix_submissions"] > record["prefix_submissions"]
                    or record["distinct_upload_backend"] != 1
                    or not 2 <= record["arena_bytes"] <= 72 * 2**20
                    or record["arena_bytes"] % 2
                    or record["slot_bytes"] != record["arena_bytes"] // 2):
                raise ValueError("unqualified overlap shape or footprint")
            if record["outcome"] == "completed" and (
                    record["prefix_submissions"] != 1 or record["suffix_submissions"] != 1
                    or record["gate_up_bytes"] <= 0 or record["down_bytes"] <= 0
                    or record["gate_up_bytes"] + record["down_bytes"] > 2**64 - 1
                    or record["gate_up_windows"] != (record["gate_up_bytes"] + record["slot_bytes"] - 1) // record["slot_bytes"]
                    or record["down_windows"] != (record["down_bytes"] + record["slot_bytes"] - 1) // record["slot_bytes"]):
                raise ValueError("incomplete completed overlap event")
            records.append(record)
        except (KeyError, ValueError, OverflowError):
            malformed += 1
    return dict(records=records, malformed_records=malformed,
                scope="Exact selected logical payload and nested host intervals; no measured CUDA overlap or physical SSD attribution")


def prefill_down_overlap_qualification(rows):
    requests = []
    for row in rows:
        observation = row.get("prefill_down_overlap", {})
        records = observation.get("records", [])
        completed = [record for record in records if record["outcome"] == "completed"]
        failed = sum(record["outcome"] == "failed" for record in records)
        requests.append(dict(name=row.get("name"), completed_calls=len(completed), failed_calls=failed,
            gate_up_bytes=sum(record["gate_up_bytes"] for record in completed),
            down_bytes=sum(record["down_bytes"] for record in completed),
            malformed_records=observation.get("malformed_records", 0),
            passed=bool(completed) and not failed and observation.get("malformed_records", 0) == 0
                and row.get("cache_hits") == 0))
    return dict(passed=bool(requests) and all(request["passed"] for request in requests), requests=requests,
                scope="Completed measured-request split executions; output quality and actual CUDA overlap checked separately")


def phase_cold_growth_records(request_log):
    records, malformed = [], 0
    pattern = (r"\[HOSTMOE-PHASE-COLD-GROW\] layer=(\d+) slots=(\d+)->(\d+) "
               r"bytes=(\d+)->(\d+) lost_hot_slots=(\d+) reload=immutable_source "
               r"fallback=([01]) reserved=(\d+) budget=(\d+)(?:\s|$)")
    for line in request_log.splitlines():
        if "[HOSTMOE-PHASE-COLD-GROW]" not in line:
            continue
        match = re.search(pattern, line)
        if match is None:
            malformed += 1
            continue
        row = dict(zip(("layer", "old_slots", "new_slots", "old_bytes", "new_bytes",
                        "lost_hot_slots", "fallback", "reserved", "budget"), map(int, match.groups())))
        if (row["old_slots"] < 1 or row["new_slots"] < row["old_slots"]
                or row["new_bytes"] < row["old_bytes"] or row["reserved"] > row["budget"]
                or row["lost_hot_slots"] > row["old_slots"]):
            malformed += 1
            continue
        records.append(row)
    return dict(records=records, malformed_records=malformed)


def forward_timing_records(request_log):
    """Keep native positions/shapes and failures; compute includes expert callbacks.

    These stages partition one native call, not the enclosing API request.
    Never add nested HOSTMOE timings to the compute stage.
    """
    integer_fields = ("kind", "rows", "p0", "nout", "wanth", "logits", "reused")
    stages = ("acquire_ms", "input_ms", "compute_ms", "output_ms", "total_ms")
    records, malformed = [], 0
    for line in request_log.splitlines():
        if "[GLM-MTP-FORWARD-TIMING]" not in line:
            continue
        values = dict(re.findall(r"(\w+)=([^\s]+)", line))
        try:
            record = {field: int(values[field]) for field in integer_fields}
            record.update({field: float(values[field]) for field in stages})
            record["outcome"] = values["outcome"]
            if (record["kind"] not in (0, 1) or record["rows"] < 1 or record["p0"] < 0
                    or record["nout"] < 1 or record["nout"] > record["rows"]
                    or any(record[field] not in (0, 1) for field in ("wanth", "logits", "reused"))
                    or record["outcome"] not in ("completed", "failed")
                    or any(not 0 <= record[field] < float("inf") for field in stages)
                    or abs(sum(record[field] for field in stages[:-1]) - record["total_ms"]) > 0.001):
                raise ValueError("invalid native stage observation")
        except (KeyError, ValueError):
            malformed += 1
            continue
        records.append(record)
    return dict(scope="Native host wall including existing synchronization and nested expert callbacks; not CUDA kernel time",
                malformed_records=malformed, records=records)


def victim_cache_summary(before_log, request_log):
    """Require two complete snapshots; warm-up traffic is never request credit."""
    counters = ("candidates", "eligible", "admitted", "refusals", "d2h_bytes",
                "completed_bytes", "promoted_bytes", "failures")
    gauges = ("metadata_bytes", "payload_bytes", "configured_bytes", "history_events", "history_keys")
    storage_counters = ("allocation_attempts", "allocation_bytes", "allocation_ns", "pool_reuses",
                        "resident_reuses", "release_calls", "release_ns", "recycled_bytes")
    storage_gauges = ("free_bytes", "free_entries", "resident_bytes", "resident_entries")
    def snapshots(log):
        records, malformed = [], 0
        for line in log.splitlines():
            if "[HOSTMOE-RAM-VICTIM]" not in line:
                continue
            values = dict(re.findall(r"(\w+)=([^\s]+)", line))
            try:
                record = {key: int(values[key]) for key in counters + gauges}
                # V16 has no allocator telemetry. When a newer binary emits any
                # of it, require the whole snapshot instead of inventing zeros.
                if any(key in values for key in storage_counters + storage_gauges):
                    record.update({key: int(values[key]) for key in storage_counters + storage_gauges})
                if any(value < 0 for value in record.values()):
                    raise ValueError("negative victim counter")
            except (KeyError, ValueError):
                malformed += 1
                continue
            records.append(record)
        return records, malformed
    prior, prior_bad = snapshots(before_log)
    current, current_bad = snapshots(request_log)
    before, after = (prior[-1] if prior else None), (current[-1] if current else None)
    delta, storage_delta = None, None
    if before is not None and after is not None and not prior_bad and not current_bad:
        if all(after[key] >= before[key] for key in counters):
            delta = {key: after[key] - before[key] for key in counters}
        if (all(key in before and key in after for key in storage_counters)
                and all(after[key] >= before[key] for key in storage_counters)):
            storage_delta = {key: after[key] - before[key] for key in storage_counters}
    return dict(scope="Sequential request deltas of logical transfers, excluding startup/warm-up; not physical SSD attribution",
                before=before, after=after, delta=delta, storage_delta=storage_delta,
                malformed_records=prior_bad + current_bad)


def mapped_ram_summary(before_log, request_log):
    """Physical charges and VA locks, plus logical copy deltas; no SSD claim."""
    counters = ("hits", "misses", "hit_copy_bytes", "admitted_logical_bytes", "bypass_logical_bytes", "evictions", "cleanup_failures")
    gauges = ("capacity_charge_bytes", "logical_resident_bytes", "owned_charge_bytes", "locked_page_bytes",
              "metadata_charge_bytes", "resident_entries", "active_leases", "quarantine_charge_bytes",
              "pending_admissions", "quota_restore_pending", "quota_conflict")
    def snapshots(log):
        records, malformed = [], 0
        for line in log.splitlines():
            if "[HOSTMOE-RAM-MAPPED]" not in line:
                continue
            values = dict(re.findall(r"(\w+)=([^\s]+)", line))
            try:
                record = {key: int(values[key]) for key in counters + gauges}
                if (any(value < 0 for value in record.values())
                        or record["logical_resident_bytes"] > record["owned_charge_bytes"]
                        or record["locked_page_bytes"] + record["metadata_charge_bytes"] > record["owned_charge_bytes"]
                        or record["quarantine_charge_bytes"] > record["owned_charge_bytes"]
                        or record["quota_restore_pending"] not in (0, 1) or record["quota_conflict"] not in (0, 1)):
                    raise ValueError("invalid mapped RAM ownership")
            except (ValueError, KeyError):
                malformed += 1
                continue
            records.append(record)
        return records, malformed
    prior, prior_bad = snapshots(before_log)
    current, current_bad = snapshots(request_log)
    before, after = (prior[-1] if prior else None), (current[-1] if current else None)
    delta = None
    if before is not None and after is not None and not prior_bad and not current_bad:
        if all(after[key] >= before[key] for key in counters):
            delta = {key: after[key] - before[key] for key in counters}
    return dict(before=before, after=after, delta=delta, malformed_records=prior_bad + current_bad,
                scope="Charged owned bytes and Windows VA locks; logical hit copies exclude startup, not physical SSD bytes or unique shared pages")


MAPPED_RAM_COST_COUNTERS = (
    "plan_calls", "plan_nanoseconds", "admission_calls", "admission_successes", "admission_nanoseconds",
    "quota_claim_calls", "quota_claim_wait_nanoseconds", "quota_read_calls", "quota_read_nanoseconds",
    "quota_write_calls", "quota_write_nanoseconds", "quota_expand_writes", "quota_restore_writes", "quota_reuse_claims",
    "memory_query_calls", "memory_query_nanoseconds", "duplicate_calls", "duplicate_nanoseconds",
    "section_calls", "section_nanoseconds", "map_calls", "map_nanoseconds",
    "protection_query_calls", "protection_query_nanoseconds", "lock_calls", "lock_nanoseconds",
    "page_query_calls", "page_query_nanoseconds", "retire_calls", "retire_nanoseconds",
    "guarded_copy_calls", "guarded_copy_bytes", "guarded_copy_failures", "guarded_copy_nanoseconds",
    "probation_observations", "probation_firsts", "probation_repeats")
MAPPED_RAM_COST_GAUGES = ("peak_claim_bytes", "quota_min_high_water_bytes", "probation_keys", "probation_metadata_bytes")
MAPPED_RAM_COST_HIGH_WATERS = ("peak_claim_bytes", "quota_min_high_water_bytes")


def mapped_ram_cost_summary(before_log, request_log):
    """Require complete, ordered process snapshots; never hide an inner reset.

    High-water observations stay gauges in the report, but cannot decrease in
    one process. Current probation keys/metadata may shrink during cleanup.
    """
    fields = MAPPED_RAM_COST_COUNTERS + MAPPED_RAM_COST_GAUGES
    def snapshots(log):
        records, malformed = [], 0
        for line in log.splitlines():
            if "[HOSTMOE-RAM-MAPPED-COST]" not in line:
                continue
            pairs = re.findall(r"(\w+)=([^\s]+)", line)
            values = dict(pairs)
            try:
                if (any(sum(key == name for name, _ in pairs) != 1 for key in fields)
                        or any(re.fullmatch(r"[0-9]+", values[key]) is None for key in fields)):
                    raise ValueError("missing, duplicate or non-decimal mapped cost")
                r = {key: int(values[key]) for key in fields}
                if (any(v < 0 for v in r.values()) or r["admission_successes"] > r["admission_calls"]
                        or any(v > 2**64 - 1 for v in r.values())
                        or r["guarded_copy_failures"] > r["guarded_copy_calls"] or r["probation_keys"] > 4096
                        or r["probation_observations"] != r["probation_firsts"] + r["probation_repeats"]
                        or r["probation_metadata_bytes"] not in (0, 1 << 20)
                        or (r["probation_keys"] and not r["probation_metadata_bytes"])):
                    raise ValueError("invalid mapped costs/probation")
            except (KeyError, ValueError):
                malformed += 1
                continue
            records.append(r)
        return records, malformed
    prior, prior_bad = snapshots(before_log)
    current, current_bad = snapshots(request_log)
    before, after = (prior[-1] if prior else None), (current[-1] if current else None)
    ordered = prior + current
    monotonic = MAPPED_RAM_COST_COUNTERS + MAPPED_RAM_COST_HIGH_WATERS
    resets = sum(any(later[key] < earlier[key] for key in monotonic)
                 for earlier, later in zip(ordered, ordered[1:]))
    delta = None
    if before is not None and after is not None and not prior_bad and not current_bad and not resets:
        delta = {k: after[k] - before[k] for k in MAPPED_RAM_COST_COUNTERS}
    return dict(before=before, after=after, delta=delta, malformed_records=prior_bad + current_bad,
                reset_records=resets,
                scope="Process-lifetime host API deltas excluding startup; stages overlap, guarded bytes count successful complete copies separately from logical cache hits, not physical disk attribution")


def mapped_probation_qualification(records):
    """Measured first/repeat activity needs complete costs for every request."""
    observations = [row.get("mapped_ram_cost", {}) for row in records]
    deltas = [row.get("delta") for row in observations]
    q = dict(attributed=bool(deltas) and all(delta is not None for delta in deltas),
             observations=sum(d["probation_observations"] for d in deltas if d is not None),
             firsts=sum(d["probation_firsts"] for d in deltas if d is not None),
             repeats=sum(d["probation_repeats"] for d in deltas if d is not None),
             malformed_records=sum(row.get("malformed_records", 0) for row in observations),
             reset_records=sum(row.get("reset_records", 0) for row in observations))
    q["passed"] = (q["attributed"] and q["firsts"] > 0 and q["repeats"] > 0
                   and not q["malformed_records"] and not q["reset_records"]
                   and all(row.get("after") is not None
                           and row["after"]["probation_metadata_bytes"] == 1 << 20
                           and row["after"]["probation_keys"] > 0 for row in observations))
    return q


def prompt_text(source, start, length, instruction):
    if source is None:
        return instruction
    text = SOURCES[source].read_text(encoding="utf-8", errors="replace")[start:start + length]
    return f"Document:\n\n{text}\n\n{instruction}"


def response_identity(request, response):
    """Preserve all returned message fields, including reasoning and tool calls.

    Equality is a deterministic response regression check, not a semantic judge
    or raw-token/logit oracle. Timings and request-specific IDs are excluded.
    """
    canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":")).encode("utf-8")
    return dict(request_sha256=hashlib.sha256(canonical(request)).hexdigest(),
                message=response.get("message"), done=response.get("done"),
                done_reason=response.get("done_reason"),
                prompt_tokens=response.get("prompt_eval_count"),
                eval_tokens=response.get("eval_count"))


def disk_read_snapshot():
    """OS disk counters, machine-wide (including unrelated processes).

    Unlike expert payload counters these are device I/O, but they are neither
    per-model attribution nor a declaration that the OS page cache was cold.
    """
    if psutil is None:
        return None
    try:
        return {name: {"read_bytes": value.read_bytes, "read_count": value.read_count}
                for name, value in psutil.disk_io_counters(perdisk=True, nowrap=False).items()}
    except (OSError, RuntimeError, AttributeError):
        return None


def disk_read_delta(before, after):
    if not before or not after or before.keys() != after.keys():
        return None
    result = {}
    for name in before:
        delta = {key: after[name][key] - before[name][key] for key in ("read_bytes", "read_count")}
        if any(value < 0 for value in delta.values()):
            return None  # Reset/wrapped/missing observations are unavailable.
        result[name] = delta
    return result


def compare_responses(current, baseline):
    def indexed(report):
        result = {}
        for row in report["records"]:
            key = (row["round"], row["name"])
            identity = row.get("response_identity")
            required = {"request_sha256", "message", "done", "done_reason", "prompt_tokens", "eval_tokens"}
            if (key in result or not isinstance(identity, dict) or set(identity) != required
                    or not isinstance(identity["message"], dict) or identity["done"] is not True
                    or not identity["done_reason"] or not isinstance(identity["request_sha256"], str)
                    or len(identity["request_sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in identity["request_sha256"])
                    or any(type(identity[field]) is not int or identity[field] <= 0
                           for field in ("prompt_tokens", "eval_tokens"))):
                raise ValueError("Response comparison requires unique records with complete response identities")
            result[key] = identity
        if not result:
            raise ValueError("Response comparison requires executed requests")
        return result
    actual, expected = indexed(current), indexed(baseline)
    if actual.keys() != expected.keys():
        raise ValueError("Response comparison request sets differ")
    changed = [dict(round=key[0], name=key[1],
                    fields=[field for field in expected[key]
                            if actual[key].get(field) != expected[key][field]])
               for key in expected if actual[key] != expected[key]]
    return dict(passed=not changed, compared_requests=len(expected), differences=changed,
                scope="Exact request, returned message (including reasoning/tools), token counts and completion status; not raw token IDs, full logits or independent semantic quality.")


def http_json(port, method, path, body, timeout):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {raw[:400]}")
        return json.loads(raw)
    finally:
        connection.close()


def board_vram_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:
        return None


def benchmark_provenance(server):
    """Record the artifacts and driver before measured work begins."""
    provenance = {"binaries": []}
    for path in (server, *(server.parent / name for name in (
            "GgmlOps.dll", "TensorSharp.Models.dll", "TensorSharp.Runtime.dll"))):
        if path.is_file():
            with path.open("rb") as source:
                digest = hashlib.sha256()
                for block in iter(lambda: source.read(2**20), b""):
                    digest.update(block)
            provenance["binaries"].append(dict(path=str(path.resolve()), bytes=path.stat().st_size,
                                                sha256=digest.hexdigest()))
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        provenance["gpu_driver"] = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        provenance["gpu_driver"] = None
    for name, command in (("ggml_revision", ["git", "-C", str(ROOT / "ExternalProjects/ggml"), "rev-parse", "HEAD"]),
                          ("ggml_status", ["git", "-C", str(ROOT / "ExternalProjects/ggml"), "status", "--porcelain"])):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            provenance[name] = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            provenance[name] = None
    return provenance


class Sampler:
    """Samples process working set/private bytes every 0.5 s (peaks can be missed)."""

    def __init__(self):
        self.process, self.peak_rss, self.peak_private = None, 0, 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.wait(0.5):
            try:
                info = self.process.memory_info()
                self.peak_rss = max(self.peak_rss, info.rss)
                self.peak_private = max(self.peak_private, getattr(info, "private", 0) or 0)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", default="none")
    parser.add_argument("--output", type=Path, required=True)
    # An unbound port inside Windows' default client range can self-connect
    # during readiness polling and echo GET as a malformed HTTP status line.
    parser.add_argument("--port", type=int, default=42190)
    parser.add_argument("--plan", choices=sorted(PLANS), default="default")
    parser.add_argument("--unified-memory", action="store_true")
    parser.add_argument("--spec", action="store_true",
                        help="explicit speculative comparison; default disables speculation")
    parser.add_argument("--prefix-cache", action="store_true", help="keep the prefix cache (and its startup warmup) on")
    parser.add_argument("--require-mapped-probation", action="store_true",
                        help="require actual measured second-demand history and complete mapped API cost telemetry")
    parser.add_argument("--require-file-read-workers", action="store_true",
                        help="require completed measured exact-file task activity for the explicit worker setting")
    parser.add_argument("--require-file-random-access", action="store_true",
                        help="also require applied source-handle hint records on completed exact-file reads")
    parser.add_argument("--require-file-read-sort", action="store_true",
                        help="qualify requested sort0/1 on all measured PP/N1 calls; ON needs real changed permutations in both phases")
    parser.add_argument("--require-file-source-view", action="store_true",
                        help="qualify joined source bytes in both measured PP/N1 phases; ON requires actual mapped copies without fallback")
    parser.add_argument("--require-file-source-view-prefetch", action="store_true",
                        help="qualify source-view/prefetch 0/0 or 1/1; ON needs every joined miss range prefetched in both PP/N1 phases")
    parser.add_argument("--require-layer-major-device-baton", action="store_true",
                        help="qualify explicit bank0/1: one closed bank for PP1872, none for PP33/N1, with exact selected-source coverage")
    parser.add_argument("--require-causal-route-neighbor", action="store_true",
                        help="qualify explicit causal0/1: PP resets and every N1 layer commits, with a completed match in each decode request")
    parser.add_argument("--require-vmm-pressure", action="store_true",
                        help="require actual checked VMM pressure retirement in every measured request")
    parser.add_argument("--gpu-telemetry", action="store_true",
                        help="bounded read-only Windows GPU clock/power sampling; diagnostic only, never performance eligible")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--concurrent", action="store_true",
                        help="send every request of a plan round at once and report aggregate tokens/s")
    parser.add_argument("--repeat", type=int, default=1, help="run the whole plan N times in the same server")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--server-arg", action="append", default=[], help="extra server argument (repeatable)")
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE for the server process (repeatable)")
    parser.add_argument("--compare-with", type=Path,
                        help="require exact requests and complete response messages/counts versus this report")
    args = parser.parse_args()
    if (args.plan == "semantic" or args.spec) and args.concurrent:
        parser.error("Semantic/speculative qualification requires isolated sequential request logs")
    if args.gpu_telemetry and args.concurrent:
        parser.error("GPU clock diagnostics require isolated sequential request windows")

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit("Port is occupied; refusing to touch another server")

    env = dict(os.environ)
    settings = {}
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
        settings[key] = value
    try:
        strict_cap_bytes = strict_layer_major_graph_cap_bytes(env.get(STRICT_GRAPH_CAP_ENV))
        host_embedding = host_token_embedding_enabled(env.get(HOST_TOKEN_EMBEDDING_ENV))
        down_overlap = prefill_down_overlap_enabled(env.get(PREFILL_DOWN_OVERLAP_ENV))
        decode_reserve = decode_graph_reserve_enabled(env.get(DECODE_GRAPH_RESERVE_ENV))
        file_workers = _file_workers_helper.setting(env.get(_file_workers_helper.ENV))
        file_hint = _file_workers_helper.hint_setting(env.get(_file_workers_helper.HINT_ENV))
        file_sort = _file_sort_helper.setting(env.get(_file_sort_helper.ENV))
        file_view = _file_view_helper.setting(env.get(_file_view_helper.ENV))
        file_prefetch = _file_view_helper.prefetch_setting(env.get(_file_view_helper.PREFETCH_ENV))
        retain_peer = _retain_peer_helper.enabled(env.get(_retain_peer_helper.ENV))
        vmm_weights = _vmm_weights_helper.enabled(env.get(_vmm_weights_helper.ENV))
        device_baton = _device_baton_helper.setting(env.get(_device_baton_helper.ENV))
        causal_neighbor = _causal_neighbor_helper.setting(env.get(_causal_neighbor_helper.ENV))
    except ValueError as error:
        parser.error(str(error))
    causal_neighbor_mode = causal_neighbor == 1 or args.require_causal_route_neighbor
    if causal_neighbor_mode:
        try:
            _causal_neighbor_helper.validate_profile(args, settings, env)
        except ValueError as error:
            parser.error(str(error))
        settings[_causal_neighbor_helper.ENV] = env[_causal_neighbor_helper.ENV]
    device_baton_mode = device_baton == 1 or args.require_layer_major_device_baton
    if device_baton_mode:
        try:
            _device_baton_helper.validate_profile(args, settings, env)
        except ValueError as error:
            parser.error(str(error))
        settings[_device_baton_helper.ENV] = env[_device_baton_helper.ENV]
    if file_sort == 1 or args.require_file_read_sort:
        try:
            _file_sort_helper.validate_profile(args, settings)
        except ValueError as error:
            parser.error(str(error))
        settings[_file_sort_helper.ENV] = env[_file_sort_helper.ENV]
    prefetch_mode = file_prefetch == 1 or args.require_file_source_view_prefetch
    file_view_mode = file_view == 1 or args.require_file_source_view or prefetch_mode
    if file_view_mode:
        try:
            _file_view_helper.validate_profile(args, settings, prefetch=prefetch_mode)
        except ValueError as error:
            parser.error(str(error))
        settings[_file_view_helper.ENV] = env[_file_view_helper.ENV]
        settings[_file_view_helper.PREFETCH_ENV] = env[_file_view_helper.PREFETCH_ENV]
    if args.require_vmm_pressure and not vmm_weights:
        parser.error("--require-vmm-pressure requires the explicit VMM weights profile")
    if vmm_weights:
        try:
            _vmm_weights_helper.validate_profile(args, settings)
        except ValueError as error:
            parser.error(str(error))
        settings[_vmm_weights_helper.ENV] = env[_vmm_weights_helper.ENV]
    if file_workers not in (None, 0) and not args.require_file_read_workers:
        parser.error("A file reader override requires --require-file-read-workers activity qualification")
    if args.require_file_random_access and (not args.require_file_read_workers or file_hint is None):
        parser.error("Applied random-access hint qualification requires explicit hint 0/1 and joined file reader qualification")
    if args.require_file_read_workers:
        if (args.concurrent or not args.unified_memory or args.spec or args.prefix_cache or args.repeat < 1
                or file_workers is None or not host_embedding or strict_cap_bytes or down_overlap or decode_reserve or retain_peer
                or settings.get("TS_UNIFIED_MAX_RUNNING_REQUESTS") != "1"
                or settings.get("TS_SCHED_PREFILL_CHUNK") != "512" or settings.get("MAX_CONTEXT") != "2048"
                or settings.get("TS_HOST_MOE_EXPERT_CACHE_RING_READS") != "2"
                or settings.get("TS_HOST_MOE_EXPERT_CACHE_TIMING") != "1"
                or any(settings.get(key) != "0" for key in ("TS_HOST_MOE_RAM_CACHE_MB",
                    "TS_HOST_MOE_RAM_CACHE_MAPPED", "TS_HOST_MOE_RAM_CACHE_VICTIM",
                    "TS_HOST_MOE_LAYER_PREFETCH", "TS_HOST_MOE_EXPERT_CACHE_PREFETCH",
                    "TS_HOST_MOE_DECODE_PREFETCH", "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS",
                    "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS", "TS_HOST_MOE_PREFILL_RING_COALESCE16",
                    "TS_HOST_MOE_PHASE_COLD_GROWTH", "TS_HOST_MOE_DECODE_REBALANCE",
                    "TS_GLM_SELECTED_EXPERT_TIMING",
                    "TS_DSV4_PERF", "TS_HOST_MOE_ROUTE_TRACE", "TS_GGML_Q8_ORDERED_SMALL_BATCH",
                    "TS_GGML_Q8_ORDERED_GATE_PREFILL", "TS_GGML_Q8_GATE_PREFILL_TILE16",
                    "TS_GGML_STRICT_RECURRENT_PROJECTIONS", "TS_GLM_OP_OFFLOAD",
                    "TS_GLM_GRAPH_MEMORY_DIAGNOSTICS")) or args.server_arg
                or (not args.require_file_random_access and settings.get(_file_workers_helper.HINT_ENV) != "0")):
            parser.error("File reader qualification requires explicit isolated N1/context2048/PP512 Host embedding, "
                         "ring2/timing1 and disabled RAM/prefetch/direct/ordered/coalesced/random-access/graph experiments")
        settings[_file_workers_helper.ENV] = env[_file_workers_helper.ENV]
        if args.require_file_random_access:
            settings[_file_workers_helper.HINT_ENV] = env[_file_workers_helper.HINT_ENV]
    if down_overlap and (args.concurrent or not args.unified_memory or args.spec or args.prefix_cache or args.repeat < 1
            or env.get(HOST_TOKEN_EMBEDDING_ENV) != "1"
            or settings.get("TS_UNIFIED_MAX_RUNNING_REQUESTS") != "1"
            or settings.get("TS_SCHED_PREFILL_CHUNK") != "512" or settings.get("MAX_CONTEXT") != "2048"
            or any(settings.get(key) != "0" for key in ("TS_HOST_MOE_PREFILL_RING_COALESCE16",
                "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS"))
            or any("draft" in value.lower() or "spec" in value.lower() for value in args.server_arg)):
        parser.error("Prefill Down overlap qualification requires guarded Host embedding and explicit sequential N1/context2048/PP512 unified profile without speculation, prefix, coalesced/direct/ordered reads")
    if down_overlap:
        settings[PREFILL_DOWN_OVERLAP_ENV] = env[PREFILL_DOWN_OVERLAP_ENV]
    if retain_peer:
        if (args.concurrent or not args.unified_memory or args.spec or args.prefix_cache or args.repeat < 1
                or not host_embedding or strict_cap_bytes or down_overlap or decode_reserve or args.gpu_telemetry
                or env.get(_retain_peer_helper.PARENT_ENV) != "1" or args.server_arg
                or settings.get("TS_UNIFIED_MAX_RUNNING_REQUESTS") != "1"
                or settings.get("TS_SCHED_PREFILL_CHUNK") != "512" or settings.get("MAX_CONTEXT") != "2048"
                or settings.get("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS") != "1"
                or settings.get("TS_HOST_MOE_EXPERT_CACHE_RING_READS") != "2"
                or settings.get("TS_HOST_MOE_EXPERT_CACHE_TIMING") != "1"
                or any(settings.get(key) != "0" for key in ("TS_HOST_MOE_RAM_CACHE_MB",
                    "TS_HOST_MOE_RAM_CACHE_MAPPED", "TS_HOST_MOE_RAM_CACHE_VICTIM", "TS_HOST_MOE_LAYER_PREFETCH",
                    "TS_HOST_MOE_EXPERT_CACHE_PREFETCH", "TS_HOST_MOE_DECODE_PREFETCH",
                    "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS",
                    "TS_HOST_MOE_PREFILL_RING_COALESCE16", "TS_HOST_MOE_PHASE_COLD_GROWTH",
                    "TS_HOST_MOE_FILE_READ_WORKERS", "TS_HOST_MOE_FILE_RANDOM_ACCESS",
                    "TS_GLM_SELECTED_EXPERT_TIMING", "TS_DSV4_PERF", "TS_HOST_MOE_ROUTE_TRACE",
                    "TS_GGML_Q8_ORDERED_SMALL_BATCH", "TS_GGML_Q8_ORDERED_GATE_PREFILL", "TS_GGML_Q8_GATE_PREFILL_TILE16",
                    "TS_GGML_STRICT_RECURRENT_PROJECTIONS", "TS_GLM_OP_OFFLOAD", "TS_GLM_GRAPH_MEMORY_DIAGNOSTICS"))):
            parser.error("Retained-peer qualification requires isolated N1/context2048/PP512 Host embedding, "
                         "75% rebalance and expert diagnostics, with other cache/read/graph diagnostics disabled")
        settings[_retain_peer_helper.ENV] = env[_retain_peer_helper.ENV]
        settings[_retain_peer_helper.PARENT_ENV] = env[_retain_peer_helper.PARENT_ENV]
    if decode_reserve:
        if (args.concurrent or not args.unified_memory or args.spec or args.prefix_cache or args.repeat < 1
                or not host_embedding or strict_cap_bytes or down_overlap
                or settings.get("TS_UNIFIED_MAX_RUNNING_REQUESTS") != "1"
                or settings.get("TS_SCHED_PREFILL_CHUNK") != "512" or settings.get("MAX_CONTEXT") != "2048"
                or settings.get("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS") != "1"
                or any("draft" in value.lower() or "spec" in value.lower() for value in args.server_arg)):
            parser.error("Decode graph reserve qualification requires isolated N1/context2048/PP512 unified Host embedding, expert diagnostics, no graph cap, Down overlap, speculation or prefix cache")
        settings[DECODE_GRAPH_RESERVE_ENV] = env[DECODE_GRAPH_RESERVE_ENV]
    if strict_cap_bytes:
        if (args.concurrent or not args.unified_memory or args.spec or args.prefix_cache or args.repeat < 1
                or settings.get("TS_UNIFIED_MAX_RUNNING_REQUESTS") != "1"
                or settings.get("TS_SCHED_PREFILL_CHUNK") != "512" or settings.get("MAX_CONTEXT") != "2048"):
            parser.error("Strict layer-major graph cap qualification requires isolated sequential unified-memory "
                         "requests, no speculation/prefix cache, and explicit TS_UNIFIED_MAX_RUNNING_REQUESTS=1, "
                         "TS_SCHED_PREFILL_CHUNK=512 and MAX_CONTEXT=2048")
        if any(arg.split("=", 1)[0] in ("--spec", "--draft-model", "--prefix-cache") for arg in args.server_arg):
            parser.error("Strict layer-major graph cap qualification forbids speculative/prefix server overrides")
        # This known, non-sensitive requested value must also be recorded when
        # inherited. Never let an inherited enabled cap bypass qualification.
        settings[STRICT_GRAPH_CAP_ENV] = env[STRICT_GRAPH_CAP_ENV]
    if host_embedding:
        if (args.concurrent or not args.unified_memory or args.spec or args.prefix_cache or args.repeat < 1
                or settings.get("TS_UNIFIED_MAX_RUNNING_REQUESTS") != "1"
                or settings.get("TS_SCHED_PREFILL_CHUNK") != "512" or settings.get("MAX_CONTEXT") != "2048"
                or any(arg.split("=", 1)[0] in ("--spec", "--draft-model", "--prefix-cache") for arg in args.server_arg)):
            parser.error("Host token embedding qualification requires isolated sequential unified-memory requests, "
                         "no speculation/prefix cache, and explicit N1/chunk512/context2048")
        settings[HOST_TOKEN_EMBEDDING_ENV] = env[HOST_TOKEN_EMBEDDING_ENV]
    native_timing = any(settings.get(key) == "1" for key in
        ("TS_HOST_MOE_EXPERT_CACHE_TIMING", "TS_GLM_FLASH_MTP_FORWARD_TIMING", "TS_HOST_MOE_RAM_CACHE_VICTIM", "TS_HOST_MOE_RAM_CACHE_MAPPED"))
    victim_cache = settings.get("TS_HOST_MOE_RAM_CACHE_VICTIM") == "1"
    mapped_ram = settings.get("TS_HOST_MOE_RAM_CACHE_MAPPED") == "1"
    if args.require_mapped_probation and not mapped_ram:
        parser.error("Mapped probation qualification requires TS_HOST_MOE_RAM_CACHE_MAPPED=1")
    scalar_staging = settings.get("TS_HOST_MOE_DECODE_STAGING_72MB") == "1"
    prefill_ring = settings.get("TS_HOST_MOE_PREFILL_RING_72MB") == "1"
    coalesced_ring = settings.get("TS_HOST_MOE_PREFILL_RING_COALESCE16") == "1"
    cold_growth = settings.get("TS_HOST_MOE_PHASE_COLD_GROWTH") == "1"
    if cold_growth and (args.concurrent or settings.get("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS") != "1"):
        parser.error("Cold growth qualification requires sequential requests and expert diagnostics")
    if scalar_staging and settings.get("TS_HOST_MOE_EXPERT_CACHE_TIMING") != "1":
        parser.error("Expanded scalar staging qualification requires TS_HOST_MOE_EXPERT_CACHE_TIMING=1")
    if prefill_ring and (args.concurrent or settings.get("TS_HOST_MOE_EXPERT_CACHE_TIMING") != "1"):
        parser.error("Expanded prefill ring qualification requires sequential requests and expert cache timing")
    if coalesced_ring and (not prefill_ring or settings.get("TS_HOST_MOE_RAM_CACHE_MB") != "0"
                           or settings.get("TS_HOST_MOE_RAM_CACHE_VICTIM") != "0"):
        parser.error("Coalesced ring qualification requires prefill ring72 and explicit zero RAM/victim cache")
    if victim_cache and settings.get("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS") != "1":
        parser.error("Victim cache qualification requires TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS=1")
    if mapped_ram and (args.concurrent or not args.unified_memory
                       or settings.get("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS") != "1"):
        parser.error("Mapped RAM qualification requires sequential unified-memory requests and expert diagnostics")
    if args.concurrent and native_timing:
        parser.error("Native timing attribution requires isolated sequential request logs")
    command = [str(args.server), "--model", str(args.model), "--mmproj", args.mmproj, "--backend", "ggml_cuda",
               "--host", "127.0.0.1", "--port", str(args.port), "--spec" if args.spec else "--no-spec", "--no-multi-agent",
               "--temperature", "0", "--top-k", "0", "--top-p", "1", "--repeat-penalty", "1", "--seed", "17"]
    if not args.prefix_cache:
        command.append("--no-prefix-cache")
    if args.unified_memory:
        command.append("--unified-memory")
    command += args.server_arg

    report = dict(command=command, environment=settings, plan=args.plan, records=[], started_unix=time.time())
    report["provenance"] = benchmark_provenance(args.server)
    if args.gpu_telemetry:
        report["timing_eligible"] = False

    save_lock = threading.Lock()

    def save():
        with save_lock:
            (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sampler = Sampler()
    server = None
    gpu_telemetry = None
    try:
        if args.gpu_telemetry:
            telemetry_spec = importlib.util.spec_from_file_location(
                "glm_gpu_clock_telemetry", Path(__file__).with_name("glm_gpu_clock_telemetry.py"))
            telemetry_module = importlib.util.module_from_spec(telemetry_spec)
            telemetry_spec.loader.exec_module(telemetry_module)
            gpu_telemetry = telemetry_module.GpuClockTelemetry(out)
            report["gpu_telemetry"] = gpu_telemetry.report
            if not gpu_telemetry.start():
                raise RuntimeError("Requested GPU clock telemetry is unavailable; see gpu_telemetry")
        report["vram_before_mib"] = board_vram_mib()
        with (out / "server.log").open("w", encoding="utf-8", errors="replace") as log:
            started = time.monotonic()
            server = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            if psutil:
                sampler.process = psutil.Process(server.pid)
                sampler.thread.start()
            deadline = started + args.timeout
            model_id = None
            while model_id is None:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited during startup (code {server.returncode})")
                try:
                    models = http_json(args.port, "GET", "/v1/models", None, 5).get("data", [])
                    model_id = models[0]["id"] if models else None
                except (OSError, ValueError, RuntimeError):
                    pass
                if model_id is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError("startup deadline exceeded")
                    time.sleep(1)
            report["startup_s"] = round(time.monotonic() - started, 3)
            report["vram_after_load_mib"] = board_vram_mib()
            print(f"ready in {report['startup_s']} s", flush=True)
            save()
            def run_one(round_index, name, source, start, length, instruction, max_new):
                    body = dict(model=model_id, stream=False, think=args.think, skills=[], skills_discovery=False, multi_agent=False,
                                messages=[dict(role="user", content=prompt_text(source, start, length, instruction))],
                                options=dict(temperature=0, top_k=0, top_p=1, min_p=0, repeat_penalty=1, seed=17,
                                             num_predict=max_new, stop=[]))
                    row = dict(name=name, round=round_index)
                    scoped_log = args.plan == "semantic" or args.spec or native_timing or cold_growth or strict_cap_bytes > 0 or host_embedding or down_overlap or decode_reserve or retain_peer or vmm_weights or args.require_file_read_workers or file_sort == 1 or args.require_file_read_sort or file_view_mode or device_baton_mode or causal_neighbor_mode
                    log_offset = (out / "server.log").stat().st_size if scoped_log else 0
                    victim_before_log = ((out / "server.log").read_bytes()[:log_offset].decode("utf-8", errors="replace")
                                         if victim_cache or mapped_ram else "")
                    disk_before = disk_read_snapshot()
                    if args.gpu_telemetry:
                        request_start_unix_ns = time.time_ns()
                        request_start_monotonic_ns = time.monotonic_ns()
                    t0 = time.monotonic()
                    try:
                        response = http_json(args.port, "POST", "/api/chat/ollama", body, args.timeout)
                    except Exception as error:
                        if args.gpu_telemetry:
                            row["request_window"] = telemetry_module.request_window(request_start_unix_ns,
                                request_start_monotonic_ns, time.time_ns(), time.monotonic_ns())
                        row["error"] = f"{type(error).__name__}: {error}"
                        report["records"].append(row)
                        save()
                        raise
                    if args.gpu_telemetry:
                        row["request_window"] = telemetry_module.request_window(request_start_unix_ns,
                            request_start_monotonic_ns, time.time_ns(), time.monotonic_ns())
                    row["wall_s"] = round(time.monotonic() - t0, 3)
                    row["machine_disk_reads"] = disk_read_delta(disk_before, disk_read_snapshot())
                    prompt_tokens = response.get("prompt_eval_count") or 0
                    hits = response.get("prompt_cache_hit_tokens") or 0
                    prompt_ns = response.get("prompt_eval_duration") or 0
                    eval_tokens = response.get("eval_count") or 0
                    eval_ns = response.get("eval_duration") or 0
                    row.update(prompt_tokens=prompt_tokens, cache_hits=hits, prompt_s=round(prompt_ns / 1e9, 3),
                               eval_tokens=eval_tokens, eval_s=round(eval_ns / 1e9, 3),
                               done_reason=response.get("done_reason"),
                               content_head=(response.get("message") or {}).get("content", "")[:240],
                               content_sha1=hashlib.sha1(((response.get("message") or {}).get("content", ""))
                                                         .encode("utf-8")).hexdigest())
                    row["response_identity"] = response_identity(body, response)
                    if scoped_log:
                        request_log = (out / "server.log").read_bytes()[log_offset:].decode("utf-8", errors="replace")
                    if args.plan == "semantic":
                        row["semantic"] = semantic_result(name, instruction, response, request_log)
                    if args.spec:
                        row["speculation"] = speculation_records(request_log)
                    if settings.get("TS_HOST_MOE_EXPERT_CACHE_TIMING") == "1":
                        row["cache_timing"] = cache_timing_summary(request_log)
                    if file_sort == 1 or args.require_file_read_sort:
                        row["file_read_sort"] = _file_sort_helper.records(request_log)
                    if file_view_mode:
                        row["file_source_view"] = _file_view_helper.records(request_log)
                        row["file_source_view_prefetch"] = _file_view_helper.prefetch_records(request_log)
                    if device_baton_mode:
                        row["layer_major_device_baton"] = _device_baton_helper.records(request_log)
                    if causal_neighbor_mode:
                        row["causal_route_neighbor"] = _causal_neighbor_helper.records(request_log)
                    if args.require_file_read_workers:
                        row["file_read_workers"] = _file_workers_helper.records(request_log)
                    if retain_peer:
                        row["decode_retain_peer"] = _retain_peer_helper.records(request_log)
                    if vmm_weights:
                        row["vmm_weights"] = _vmm_weights_helper.records(request_log)
                    if scalar_staging:
                        row["scalar_staging"] = scalar_staging_summary(request_log)
                    if prefill_ring:
                        row["prefill_ring"] = prefill_ring_summary(request_log)
                    if coalesced_ring:
                        row["coalesced_ring"] = coalesced_ring_summary(request_log)
                    if strict_cap_bytes:
                        row["strict_layer_major_graph_cap"] = strict_layer_major_graph_cap_records(request_log, strict_cap_bytes)
                    if host_embedding:
                        row["host_token_embedding"] = host_token_embedding_records(request_log)
                    if down_overlap:
                        row["prefill_down_overlap"] = prefill_down_overlap_records(request_log)
                    if decode_reserve:
                        row["decode_graph_reserve"] = decode_graph_reserve_records(request_log)
                        row["expert_recovery"] = expert_recovery_records(request_log)
                    if cold_growth:
                        row["phase_cold_growth"] = phase_cold_growth_records(request_log)
                    if settings.get("TS_GLM_FLASH_MTP_FORWARD_TIMING") == "1":
                        row["forward_timing"] = forward_timing_records(request_log)
                    if victim_cache:
                        row["victim_cache"] = victim_cache_summary(victim_before_log, request_log)
                    if mapped_ram:
                        row["mapped_ram"] = mapped_ram_summary(victim_before_log, request_log)
                        row["mapped_ram_cost"] = mapped_ram_cost_summary(victim_before_log, request_log)
                    (out / f"round-{round_index}-{name}-response.json").write_text(
                        json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    if prompt_ns and hits == 0:
                        row["prefill_tps"] = round(prompt_tokens / (prompt_ns / 1e9), 2)
                    if eval_ns and eval_tokens:
                        row["decode_tps"] = round(eval_tokens / (eval_ns / 1e9), 2)
                    row["vram_mib"] = board_vram_mib()
                    report["records"].append(row)
                    save()
                    print(json.dumps({k: row.get(k) for k in ("name", "round", "prompt_tokens", "cache_hits", "prompt_s", "prefill_tps",
                                                              "eval_tokens", "eval_s", "decode_tps", "wall_s", "done_reason")}), flush=True)
                    return row

            for round_index in range(args.repeat):
                if not args.concurrent:
                    for entry in PLANS[args.plan]:
                        run_one(round_index, *entry)
                    continue
                # All entries of a round at once: aggregate throughput is total
                # generated tokens over the round's wall time.
                rows, errors = [None] * len(PLANS[args.plan]), []
                def worker(index, entry):
                    try:
                        rows[index] = run_one(round_index, *entry)
                    except Exception as error:
                        errors.append(error)
                threads = [threading.Thread(target=worker, args=(i, e)) for i, e in enumerate(PLANS[args.plan])]
                t0 = time.monotonic()
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                if errors:
                    raise errors[0]
                wall = time.monotonic() - t0
                tokens = sum(row.get("eval_tokens") or 0 for row in rows)
                batch = dict(round=round_index, requests=len(rows), wall_s=round(wall, 3), eval_tokens=tokens,
                             aggregate_tps=round(tokens / wall, 2) if wall > 0 else None)
                report.setdefault("concurrent_rounds", []).append(batch)
                save()
                print(json.dumps(batch), flush=True)
            if args.compare_with:
                report["response_comparison"] = compare_responses(report,
                    json.loads(args.compare_with.read_text(encoding="utf-8")))
                if not report["response_comparison"]["passed"]:
                    raise ValueError("Deterministic response comparison failed; see response_comparison")
            if args.plan == "semantic":
                report["semantic_passed"] = bool(report["records"]) and all(
                    row.get("semantic", {}).get("passed") is True for row in report["records"])
                if not report["semantic_passed"]:
                    raise ValueError("Semantic answer or natural-EOS qualification failed; see records")
            if args.spec:
                observed = [s for row in report["records"] for s in row.get("speculation", [])]
                report["speculation_qualification"] = dict(records=len(observed),
                    verify_steps=sum(s["verify_steps"] for s in observed),
                    drafted=sum(s["drafted"] for s in observed), accepted=sum(s["accepted"] for s in observed))
                if report["speculation_qualification"]["verify_steps"] <= 0:
                    raise ValueError("No speculative verification was observed; not a qualified speculative benchmark")
            if cold_growth:
                growth = [entry for row in report["records"] for entry in row["phase_cold_growth"]["records"]]
                report["cold_growth_qualification"] = dict(events=len(growth), successful_growths=sum(
                    entry["fallback"] == 0 and entry["new_slots"] > entry["old_slots"] for entry in growth),
                    malformed_records=sum(row["phase_cold_growth"]["malformed_records"] for row in report["records"]))
                if (report["cold_growth_qualification"]["successful_growths"] <= 0
                        or report["cold_growth_qualification"]["malformed_records"]):
                    raise ValueError("No qualified measured-request cold capacity growth; see phase_cold_growth records")
            if scalar_staging:
                report["scalar_staging_qualification"] = dict(expanded_scalar_calls=sum(
                    row.get("scalar_staging", {}).get("expanded_scalar_calls", 0) for row in report["records"]))
                if report["scalar_staging_qualification"]["expanded_scalar_calls"] <= 0:
                    raise ValueError("No actual expanded scalar file window was observed")
            if prefill_ring:
                observations = [row["prefill_ring"] for row in report["records"]]
                report["prefill_ring_qualification"] = dict(expanded_ring_calls=sum(
                    row["expanded_ring_calls"] for row in observations), malformed_records=sum(
                    row["malformed_records"] for row in observations))
                if (report["prefill_ring_qualification"]["expanded_ring_calls"] <= 0
                        or report["prefill_ring_qualification"]["malformed_records"]):
                    raise ValueError("No qualified completed expanded prefill ring activity was observed")
            if coalesced_ring:
                observations = [row["coalesced_ring"] for row in report["records"]]
                report["coalesced_ring_qualification"] = dict(completed_coalesced_calls=sum(
                    row["completed_coalesced_calls"] for row in observations), malformed_records=sum(
                    row["malformed_records"] for row in observations))
                if (report["coalesced_ring_qualification"]["completed_coalesced_calls"] <= 0
                        or report["coalesced_ring_qualification"]["malformed_records"]):
                    raise ValueError("No qualified completed coalesced prefill reads were observed")
            if strict_cap_bytes:
                q = report["strict_layer_major_graph_cap_qualification"] = strict_layer_major_graph_cap_qualification(
                    report["records"], strict_cap_bytes)
                if not q["passed"]:
                    raise ValueError("No qualified measured-request strict layer-major graph cap execution; see call-local records")
            if host_embedding:
                q = report["host_token_embedding_qualification"] = host_token_embedding_qualification(report["records"])
                if not q["passed"]:
                    raise ValueError("Raw host token embedding uploads did not cover every measured uncached prompt")
            if down_overlap:
                q = report["prefill_down_overlap_qualification"] = prefill_down_overlap_qualification(report["records"])
                if not q["passed"]:
                    raise ValueError("No qualified completed measured-request prefill Down overlap execution")
            if decode_reserve:
                q = report["decode_graph_reserve_qualification"] = decode_graph_reserve_qualification(
                    report["records"], host_token_embedding=host_embedding)
                if not q["passed"]:
                    raise ValueError("Actual first-N1 graph credit/recovery did not qualify every measured request")
            if file_sort == 1 or args.require_file_read_sort:
                q = report["file_read_sort_qualification"] = _file_sort_helper.qualification(
                    report["records"], file_sort)
                if not q["passed"]:
                    raise ValueError("Selected-miss sort activity did not qualify both measured PP and N1 phases")
            if file_view_mode:
                q = report["file_source_view_qualification"] = _file_view_helper.qualification(
                    report["records"], file_view)
                if not q["passed"]:
                    raise ValueError("Joined source-view byte conservation and activity did not qualify both measured PP and N1 phases")
                q = report["file_source_view_prefetch_qualification"] = _file_view_helper.prefetch_qualification(
                    report["records"], file_view, file_prefetch)
                if not q["passed"]:
                    raise ValueError("Joined exact-demand prefetch coverage did not qualify both measured PP and N1 phases")
            if device_baton_mode:
                q = report["layer_major_device_baton_qualification"] = _device_baton_helper.qualification(
                    report["records"], device_baton)
                if not q["passed"]:
                    raise ValueError("Closed layer-major bank and fixed quick-plan selected-source activity did not qualify")
            if causal_neighbor_mode:
                q = report["causal_route_neighbor_qualification"] = _causal_neighbor_helper.qualification(
                    report["records"], causal_neighbor)
                if not q["passed"]:
                    raise ValueError("Causal outer PP/N1 activity and fixed quick-plan selected-source coverage did not qualify")
            if args.require_file_read_workers:
                q = report["file_read_workers_qualification"] = _file_workers_helper.qualification(
                    report["records"], file_workers, file_hint if args.require_file_random_access else None)
                if not q["passed"]:
                    raise ValueError("Joined exact-file reader activity did not qualify every measured request")
            if retain_peer:
                q = report["decode_retain_peer_qualification"] = _retain_peer_helper.qualification(report["records"])
                if not q["passed"]:
                    raise ValueError("No actual retained hot expert payload qualified, or a measured request had invalid activity")
            if vmm_weights:
                q = report["vmm_weights_qualification"] = _vmm_weights_helper.qualification(
                    report["records"], require_pressure=args.require_vmm_pressure)
                if not q["passed"]:
                    raise ValueError("VMM use and checked redistribution did not qualify measured prefill/decode requests")
            if mapped_ram:
                observations = [row["mapped_ram"] for row in report["records"]]
                deltas = [row["delta"] for row in observations]
                valid = bool(deltas) and all(delta is not None for delta in deltas)
                report["mapped_ram_qualification"] = dict(attributed=valid,
                    hit_copy_bytes=sum(d["hit_copy_bytes"] for d in deltas if d is not None),
                    admitted_logical_bytes=sum(d["admitted_logical_bytes"] for d in deltas if d is not None),
                    malformed_records=sum(r["malformed_records"] for r in observations))
                q = report["mapped_ram_qualification"]
                if (not valid or q["hit_copy_bytes"] <= 0 or q["admitted_logical_bytes"] <= 0 or q["malformed_records"]
                        or any(d["cleanup_failures"] for d in deltas if d is not None)
                        or any(r["after"]["locked_page_bytes"] <= 0 or r["after"]["active_leases"]
                               or r["after"]["owned_charge_bytes"] > r["after"]["capacity_charge_bytes"]
                               or r["after"]["quarantine_charge_bytes"] or r["after"]["pending_admissions"]
                               or r["after"]["quota_conflict"] for r in observations if r["after"] is not None)):
                    raise ValueError("No qualified measured-request mapped RAM retention and hit copies; see snapshots")
            if args.require_mapped_probation:
                q = report["mapped_probation_qualification"] = mapped_probation_qualification(report["records"])
                if not q["passed"]:
                    raise ValueError("No qualified measured second-demand mapped RAM profile; see cost/probation records")
            if victim_cache:
                deltas = [row.get("victim_cache", {}).get("delta") for row in report["records"]]
                valid = bool(deltas) and all(delta is not None for delta in deltas)
                report["victim_cache_qualification"] = dict(attributed=valid,
                    completed_bytes=sum(delta["completed_bytes"] for delta in deltas if delta is not None),
                    promoted_bytes=sum(delta["promoted_bytes"] for delta in deltas if delta is not None),
                    failures=sum(delta["failures"] for delta in deltas if delta is not None))
                qualification = report["victim_cache_qualification"]
                if (not valid or qualification["completed_bytes"] <= 0 or qualification["promoted_bytes"] <= 0
                        or qualification["failures"] != 0):
                    raise ValueError("No qualified measured-request victim demotion/promotion activity; see victim_cache records")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], flush=True)
    finally:
        sampler.stop.set()
        if gpu_telemetry is not None:
            try:
                observation = report["gpu_telemetry"] = gpu_telemetry.finish(report["records"])
                if not observation["completed"] or not observation["clock_power_observed"]:
                    report.setdefault("error", "GPU clock telemetry did not complete with actual request-window observations")
            except Exception as error:
                report.setdefault("error", f"GPU clock telemetry cleanup/report failed: {type(error).__name__}: {error}")
        if server is not None:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
            report["server_exit"] = server.returncode
        report["peak_rss_gib"] = round(sampler.peak_rss / 2**30, 3)
        report["peak_private_gib"] = round(sampler.peak_private / 2**30, 3)
        report["finished_unix"] = time.time()
        save()
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    sys.exit(main())
