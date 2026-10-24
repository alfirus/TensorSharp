"""Qualify selected-miss permutations, separately from physical I/O and speed.

Input must be one successful request's newly appended native log bytes. Wide
coverage uses the submitted row shape; every N1 block is associated with its
following native postcall timer. Startup events cannot supply missing activity.
"""
import collections
import re

ENV = "TS_HOST_MOE_FILE_READ_SORT"
MARKER = "[HOSTMOE-CACHE-TIMING]"
FORWARD_MARKER = "[GLM-MTP-FORWARD-TIMING]"
SCOPE = "nested_host_wall_not_cuda_kernel_or_physical_io"
LAYERS = frozenset(range(3, 45))
U64_MAX = (1 << 64) - 1
SORT_COUNTERS = ("file_sort_calls", "file_sort_changed_calls", "file_sort_misses",
                 "file_sort_bytes", "file_sort_inversions_before", "file_sort_inversions_after")
NUMBERS = ("layer", "rows", "bulk", "file_bytes", "file_sort_requested") + SORT_COUNTERS
REQUIRED = frozenset(NUMBERS) | {"outcome", "scope"}
REQUIRED_ON = ("TS_GLM_HOST_TOKEN_EMBEDDING", "TS_HOST_MOE_DECODE_REBALANCE",
               "TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS", "TS_HOST_MOE_EXPERT_CACHE_TIMING",
               "TS_GLM_FLASH_MTP_FORWARD_TIMING", "TS_HOST_MOE_DECODE_STAGING_72MB",
               "TS_HOST_MOE_PREFILL_RING_72MB")
REQUIRED_OFF = (
    "TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", "TS_GLM_DECODE_GRAPH_RESERVE",
    "TS_HOST_MOE_VMM_WEIGHTS", "TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER",
    "TS_HOST_MOE_PHASE_COLD_GROWTH", "TS_HOST_MOE_PREFILL_DOWN_OVERLAP",
    "TS_HOST_MOE_PREFILL_RING_COALESCE16", "TS_HOST_MOE_FILE_READ_WORKERS",
    "TS_HOST_MOE_FILE_RANDOM_ACCESS", "TS_HOST_MOE_RAM_CACHE_MB",
    "TS_HOST_MOE_RAM_CACHE_MAPPED", "TS_HOST_MOE_RAM_CACHE_MAPPED_EXCLUSIVE_QUOTA",
    "TS_HOST_MOE_RAM_CACHE_MAPPED_QUOTA_MB", "TS_HOST_MOE_RAM_CACHE_VICTIM",
    "TS_HOST_MOE_RAM_CACHE_PINNED", "TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD",
    "TS_HOST_MOE_RAM_CACHE_FREQUENCY", "TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY",
    "TS_HOST_MOE_LAYER_PREFETCH", "TS_HOST_MOE_EXPERT_CACHE_PREFETCH",
    "TS_HOST_MOE_DECODE_PREFETCH", "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS",
    "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS", "TS_HOST_MOE_EXPERT_CACHE_LFU",
    "TS_GLM_SELECTED_EXPERT_TIMING", "TS_DSV4_PERF", "TS_HOST_MOE_ROUTE_TRACE",
    "TS_GGML_Q8_ORDERED_SMALL_BATCH", "TS_GGML_Q8_ORDERED_GATE_PREFILL",
    "TS_GGML_Q8_GATE_PREFILL_TILE16", "TS_GGML_STRICT_RECURRENT_PROJECTIONS",
    "TS_GLM_OP_OFFLOAD", "TS_GLM_GRAPH_MEMORY_DIAGNOSTICS", "TS_GLM_FLASH_MTP",
    "TS_GLM_FLASH_MTP_SHAPE_PLAN", "TS_GLM_FLASH_MTP_PREFIX_COMMIT")
REQUIRED_VALUES = {"TS_UNIFIED_MAX_RUNNING_REQUESTS": "1", "MAX_CONTEXT": "2048",
                   "TS_SCHED_PREFILL_CHUNK": "512", "TS_HOST_MOE_EXPERT_CACHE_RING_READS": "2"}


def setting(value):
    if value not in (None, "0", "1"):
        raise ValueError(ENV + " must be unset, 0 or 1")
    return None if value is None else int(value)


def validate_profile(args, settings):
    if (args.concurrent or not args.unified_memory or args.spec or args.prefix_cache
            or args.repeat != 1 or args.plan != "quick" or args.gpu_telemetry
            or args.server_arg or args.mmproj != "none" or args.think
            or args.require_file_read_workers or args.require_file_random_access
            or args.require_mapped_probation or args.require_vmm_pressure):
        raise ValueError("File-read sort qualification requires one isolated quick-plan N1 text run without other experiments")
    for key, value in ([(key, "1") for key in REQUIRED_ON]
                       + [(key, "0") for key in REQUIRED_OFF]
                       + list(REQUIRED_VALUES.items())):
        if settings.get(key) != value:
            raise ValueError(f"File-read sort qualification requires explicit {key}={value}")
    if settings.get(ENV) not in ("0", "1"):
        raise ValueError("File-read sort qualification requires an explicit requested flag")


def _uint(value):
    if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        value = int(value)
    if type(value) is not int or not 0 <= value <= U64_MAX:
        raise ValueError("Expected a canonical bounded unsigned integer")
    return value


def _pairs(line, marker):
    if line.count(marker) != 1:
        raise ValueError("Duplicate native marker")
    pairs = [token.split("=", 1) for token in line.split(marker, 1)[1].split()]
    if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
        raise ValueError("Malformed or duplicate native field")
    return dict(pairs)


def _event(values):
    if (not REQUIRED.issubset(values) or values["scope"] != SCOPE
            or any(key.startswith("file_sort_") and key not in NUMBERS for key in values)):
        raise ValueError("Missing sort activity fields or wrong scope")
    event = {key: _uint(values[key]) for key in NUMBERS}
    event.update(outcome=values["outcome"], scope=SCOPE)
    if (event["outcome"] not in ("completed", "exception", "not_completed")
            or event["layer"] not in LAYERS
            or not (event["rows"] == 1 or 9 <= event["rows"] <= 512)
            or event["bulk"] != int(event["rows"] > 1)
            or event["file_sort_requested"] not in (0, 1)):
        raise ValueError("Unsupported selected-call shape or status")
    calls, changed, misses, size, before, after = (event[key] for key in SORT_COUNTERS)
    if not calls:
        if any((changed, misses, size, before, after)):
            raise ValueError("Inactive sort call claimed work")
    elif (event["file_sort_requested"] != 1 or event["outcome"] != "completed"
            or not 0 <= changed <= calls <= misses <= min(288, event["rows"] * 8)
            or not size == event["file_bytes"] > 0 or after
            or bool(changed) != bool(before) or before < changed
            or before > misses * (misses - 1) // 2):
        raise ValueError("Inconsistent completed sort activity")
    return event


def records(text):
    events, malformed, pending = [], 0, []
    for line in text.splitlines():
        try:
            if MARKER in line:
                event = _event(_pairs(line, MARKER))
                events.append(event)
                if event["rows"] == 1:
                    pending.append(event)
            elif FORWARD_MARKER in line:
                values = _pairs(line, FORWARD_MARKER)
                kind, rows, position = (_uint(values[key]) for key in ("kind", "rows", "p0"))
                if kind == 0 and rows == 1:
                    if (values.get("outcome") != "completed" or len(pending) != 42
                            or [event["layer"] for event in pending] != sorted(LAYERS)):
                        raise ValueError("Incomplete N1 block at native postcall timer")
                    for event in pending:
                        event["forward_p0"] = position
                    pending = []
        except (KeyError, ValueError):
            malformed += 1
    if pending:
        malformed += 1
    return dict(records=events, malformed_records=malformed, scope=SCOPE)


def qualification(rows, requested):
    requests = []
    for row in rows:
        observed = row.get("file_read_sort", {})
        events = observed.get("records", [])
        event_error = None
        try:
            if (type(observed.get("malformed_records")) is not int
                    or observed["malformed_records"] or not isinstance(events, list)):
                raise ValueError("Malformed request activity")
            for event in events:
                if any(type(event[key]) is not int for key in NUMBERS):
                    raise ValueError("Decoded native counters must be integers")
                _event(event)
                if event["rows"] == 1:
                    if type(event.get("forward_p0")) is not int:
                        raise ValueError("Missing exact N1 association")
                    _uint(event["forward_p0"])
        except (KeyError, TypeError, ValueError) as error:
            event_error = str(error)
            events = []
        prompt, count = row.get("prompt_tokens"), row.get("eval_tokens")
        valid = (type(prompt) is int and 1 < prompt <= 2048 and type(count) is int
                 and 0 < count <= 2048 - prompt and type(row.get("cache_hits")) is int
                 and row["cache_hits"] == 0 and not row.get("error")
                 and not event_error and bool(events))
        prefill, decode = [], []
        if valid:
            prefill = [event for event in events if event["rows"] > 1]
            decode = [event for event in events if event["rows"] == 1]
            widths = [min(512, prompt - start) for start in range(0, prompt, 512)]
            # This experiment excludes scalar prefill tails so N1 means decode.
            valid = all(width > 1 for width in widths)
            valid = valid and all(event["outcome"] == "completed"
                and event["file_sort_requested"] == requested
                and (requested != 1 or not event["file_bytes"] or event["file_sort_calls"] > 0)
                for event in events)
            for layer in LAYERS:
                valid = valid and (collections.Counter(event["rows"] for event in prefill
                    if event["layer"] == layer) == collections.Counter(widths))
                positions = [event.get("forward_p0") for event in decode if event["layer"] == layer]
                valid = valid and positions == list(range(prompt, prompt + count))
            if requested == 0:
                valid = valid and not any(event[key] for event in events for key in SORT_COUNTERS)
        phases = {}
        for label, phase in (("prefill", prefill), ("decode", decode)):
            totals = {key: sum(event[key] for event in phase) for key in SORT_COUNTERS}
            phases[label] = dict(calls=len(phase), **totals,
                source_logical_bytes=sum(event["file_bytes"] for event in phase),
                changed_activity_observed=totals["file_sort_changed_calls"] > 0)
            valid = valid and all(value <= U64_MAX for value in totals.values())
            valid = valid and phases[label]["source_logical_bytes"] <= U64_MAX
            if requested == 1:
                valid = valid and phases[label]["changed_activity_observed"]
        requests.append(dict(name=row.get("name"), passed=bool(valid), phases=phases, error=event_error,
            malformed_records=observed.get("malformed_records", 0)))
    return dict(passed=type(requested) is int and requested in (0, 1) and bool(requests)
        and all(row["passed"] for row in requests), requested_sort=requested, requests=requests,
        scope="Completed selected-miss permutation and unchanged logical payload; physical issue order, numerical quality and speed are separate",
        physical_io_qualified=False, performance_qualified=False, numerical_quality_qualified=False)
