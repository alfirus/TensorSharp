"""Qualify past-only route-neighbor metadata activity on the fixed quick plan.

The source-call and position proof uses only each request's appended log bytes.
This records actual completed query activity, not saved I/O, policy quality,
canonical numerical quality, physical memory residence or speed.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_causal_source_activity",
    Path(__file__).with_name("glm_layer_major_device_baton.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_source = _base._source
ENV = "TS_HOST_MOE_CAUSAL_ROUTE_NEIGHBOR"
MARKER = "[GLM-CAUSAL-ROUTE-NEIGHBOR]"
SCOPE = "past_only_source_qualified_host_metadata_not_saved_io_or_speed"
MAX_HOST_BYTES = 1024 * 1024
NUMBERS = ("profile", "epoch", "p0", "rows", "queries", "committed", "matches",
    "no_matches", "compared_anchors", "min_completed_lag", "history_rows_max",
    "declared_host_bytes", "owned_slab_bytes", "held_ticket_bytes", "pending",
    "extra_device_bytes", "certificate")
REQUIRED = frozenset(NUMBERS) | {"outcome", "scope"}
ACTIVITY = ("queries", "committed", "matches", "no_matches", "compared_anchors",
            "min_completed_lag", "history_rows_max")
REQUIRED_ON = _base.REQUIRED_ON
REQUIRED_OFF = tuple(dict.fromkeys(_base.REQUIRED_OFF + (_base.ENV,)))
REQUIRED_VALUES = _base.REQUIRED_VALUES
QUICK_REQUESTS = _base.QUICK_REQUESTS
_uint, _pairs = _source._uint, _source._pairs


def setting(value):
    if value not in (None, "0", "1"):
        raise ValueError(ENV + " must be unset, 0 or 1")
    return None if value is None else int(value)


def validate_profile(args, settings, effective=None):
    if (args.concurrent or not args.unified_memory or args.spec or args.prefix_cache
            or args.repeat != 1 or args.plan != "quick" or args.gpu_telemetry
            or args.server_arg or args.mmproj != "none" or args.think
            or any(getattr(args, key, False) for key in (
                "require_file_read_workers", "require_file_random_access", "require_file_read_sort",
                "require_file_source_view", "require_file_source_view_prefetch",
                "require_layer_major_device_baton", "require_mapped_probation", "require_vmm_pressure"))):
        raise ValueError("Causal route neighbor requires an isolated fixed quick-plan N1 text run")
    for key, value in ([(key, "1") for key in REQUIRED_ON]
                       + [(key, "0") for key in REQUIRED_OFF] + list(REQUIRED_VALUES.items())):
        if settings.get(key) != value:
            raise ValueError(f"Causal route neighbor requires explicit {key}={value}")
    if settings.get(ENV) not in ("0", "1"):
        raise ValueError("Causal route neighbor requires explicit flag0/1")
    effective = settings if effective is None else effective
    if any(effective.get(key) not in (None, "") for key in ("TS_GLM_TRACE", "TS_GLM_TRACE_DUMP_DIR")):
        raise ValueError("Causal route neighbor requires tracing unset")


def _event(values):
    if set(values) != REQUIRED or values["scope"] != SCOPE:
        raise ValueError("Missing, unknown or unsupported causal event fields/scope")
    event = {key: _uint(values[key]) for key in NUMBERS}
    event.update(outcome=values["outcome"], scope=SCOPE)
    declared = event["declared_host_bytes"]
    if (event["profile"] != 1 or event["epoch"] == 0 or not 1 <= event["rows"] <= 2048
            or event["p0"] + event["rows"] > 2048
            or event["outcome"] not in ("completed", "failed")
            or not 0 < declared <= MAX_HOST_BYTES
            or event["owned_slab_bytes"] != declared or event["held_ticket_bytes"] != declared
            or event["extra_device_bytes"] != 0 or event["certificate"] != 1):
        raise ValueError("Unsupported causal profile, metadata ownership or position")
    if (event["queries"] > 42 or event["committed"] > 42 or event["matches"] > 42
            or event["no_matches"] > 42 or event["compared_anchors"] > 2352
            or event["min_completed_lag"] > 56 or event["history_rows_max"] > 64
            or event["pending"] > 42):
        raise ValueError("Causal counters exceed bounded fixed-layer history")
    if event["outcome"] == "completed":
        if event["pending"] != 0:
            raise ValueError("Completed call retains a pending query")
        if event["rows"] > 1:
            if any(event[key] for key in ACTIVITY):
                raise ValueError("Prefill cannot query or train decode history")
        elif (event["queries"] != 42 or event["committed"] != 42
                or event["matches"] + event["no_matches"] != 42
                or event["compared_anchors"] < event["matches"]
                or bool(event["min_completed_lag"]) != bool(event["matches"])
                or event["matches"] and event["history_rows_max"] < 9):
            raise ValueError("Incomplete per-layer query/commit or impossible completed match")
    return event


def records(text):
    events, malformed, wide_before, decode_before = [], 0, 0, 0
    for line in text.splitlines():
        try:
            if MARKER in line:
                event = _event(_pairs(line, MARKER))
                event.update(prefill_calls_before=wide_before, decode_calls_before=decode_before)
                events.append(event)
            elif _source.MARKER in line:
                width = _uint(_pairs(line, _source.MARKER)["rows"])
                wide_before += int(width > 1)
                decode_before += int(width == 1)
        except (KeyError, TypeError, ValueError):
            malformed += 1
    return dict(records=events, malformed_records=malformed,
                source_activity=_source.prefetch_records(text), scope=SCOPE)


def qualification(rows, requested):
    valid_request = type(requested) is int and requested in (0, 1)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        rows = []
    observations = [row.get("causal_route_neighbor", {}) for row in rows]
    observations = [value if isinstance(value, dict) else {} for value in observations]
    base_rows = []
    for row, observed in zip(rows, observations):
        activity = observed.get("source_activity", {})
        base_rows.append(dict(row, file_source_view_prefetch=activity if isinstance(activity, dict) else {}))
    source = _source.prefetch_qualification(base_rows, 0, 0)
    requests, declared_seen, last_epoch_seen = [], set(), 0
    exact_plan = len(rows) == len(QUICK_REQUESTS)
    for index, (row, source_request) in enumerate(zip(rows, source["requests"])):
        observed = observations[index]
        events, error = observed.get("records", []), None
        name, prompt, count = QUICK_REQUESTS[index] if index < len(QUICK_REQUESTS) else (None, 0, 0)
        valid = (valid_request and exact_plan and row.get("name") == name
                 and type(row.get("round")) is int and row["round"] == 0
                 and type(row.get("prompt_tokens")) is int and row["prompt_tokens"] == prompt
                 and type(row.get("eval_tokens")) is int and row["eval_tokens"] == count
                 and type(row.get("cache_hits")) is int and row["cache_hits"] == 0
                 and not row.get("error") and source_request["passed"])
        try:
            if (not isinstance(events, list) or type(observed.get("malformed_records")) is not int
                    or observed["malformed_records"] != 0 or observed.get("scope") != SCOPE):
                raise ValueError("Malformed measured causal records")
            for event in events:
                if not isinstance(event, dict) or set(event) != REQUIRED | {"prefill_calls_before", "decode_calls_before"}:
                    raise ValueError("Decoded causal event schema changed")
                if any(type(event[key]) is not int for key in NUMBERS):
                    raise ValueError("Decoded counters must be integers")
                _event({key: event[key] for key in REQUIRED})
                for key in ("prefill_calls_before", "decode_calls_before"):
                    if type(event.get(key)) is not int:
                        raise ValueError("Missing selected-call chronology")
                    _uint(event[key])
                if event["outcome"] != "completed":
                    raise ValueError("A failed outer call cannot qualify")
            widths = [min(512, prompt - start) for start in range(0, prompt, 512)]
            expected_order = [(layer, width) for layer in sorted(_source.LAYERS) for width in widths]
            expected_order += [(layer, 1) for _ in range(count) for layer in sorted(_source.LAYERS)]
            if [(event["layer"], event["rows"]) for event in observed["source_activity"]["records"]] != expected_order:
                raise ValueError("Selected PP/N1 chronology or chunk width is incomplete")
            if requested == 0:
                if events:
                    raise ValueError("OFF cannot emit causal policy events")
            else:
                if len(events) != count + 1:
                    raise ValueError("Every PP and N1 outer call requires exactly one event")
                if events[0]["epoch"] <= last_epoch_seen:
                    raise ValueError("Each outer prefill must reset to a fresh epoch")
                prefill_calls = 42 * len(widths)
                expected_calls = [(0, prompt, 0)] + [(prompt + offset, 1, 42 * (offset + 1)) for offset in range(count)]
                last_epoch = last_epoch_seen
                for event, (position, width, decode_calls) in zip(events, expected_calls):
                    if (event["p0"] != position or event["rows"] != width
                            or event["prefill_calls_before"] != prefill_calls
                            or event["decode_calls_before"] != decode_calls
                            or event["epoch"] < last_epoch):
                        raise ValueError("Outer event position/epoch differs from completed source calls")
                    if width == 1:
                        offset = position - prompt
                        if (event["history_rows_max"] > min(64, offset + 1)
                                or event["compared_anchors"] > 42 * max(0, min(64, offset) - 8)
                                or offset < 9 and event["matches"] != 0):
                            raise ValueError("Claimed history/matches exceed already completed request visits")
                    last_epoch = event["epoch"]
                    declared_seen.add(event["declared_host_bytes"])
                last_epoch_seen = last_epoch
                if not any(event["matches"] for event in events[1:]):
                    raise ValueError("Each measured decode request needs a real completed match")
                if len(declared_seen) != 1:
                    raise ValueError("Declared metadata changed across measured calls")
        except (KeyError, TypeError, ValueError) as problem:
            error, valid = str(problem), False
        complete = events if valid and requested else []
        requests.append(dict(name=row.get("name"), passed=bool(valid),
            completed_calls=len(complete), queries=sum(event["queries"] for event in complete),
            committed=sum(event["committed"] for event in complete),
            matches=sum(event["matches"] for event in complete),
            no_matches=sum(event["no_matches"] for event in complete),
            compared_anchors=sum(event["compared_anchors"] for event in complete),
            source_phases=source_request["phases"], error=error or source_request["error"],
            malformed_records=observed.get("malformed_records", 0)))
    return dict(passed=bool(valid_request and exact_plan and source["passed"] and requests
                           and all(row["passed"] for row in requests)), requested_causal_route_neighbor=requested,
        requests=requests, declared_host_bytes=next(iter(declared_seen)) if len(declared_seen) == 1 else None,
        scope=SCOPE, future_causality_qualified=False, victim_policy_qualified=False,
        saved_io_qualified=False, physical_io_qualified=False, performance_qualified=False,
        numerical_quality_qualified=False, all_process_metadata_qualified=False)
