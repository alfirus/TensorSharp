"""Qualify a closed full-call residual bank and exact selected-source activity.

Input is one successful measured request's newly appended log bytes. This
certifies declared bank payload and direct stock-CUDA copies, not other graph
owners, device residency, numerical quality, physical I/O or speed.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_baton_source_activity",
    Path(__file__).with_name("glm_file_source_view.py"))
_source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_source)

ENV = "TS_GLM_LAYER_MAJOR_DEVICE_BATON"
MARKER = "[GLM-LAYER-MAJOR-DEVICE-BATON]"
SCOPE = "declared_payload_direct_stock_cuda_d2d"
U64_MAX = _source.U64_MAX
ROW_BYTES, CHUNK, TRANSITIONS = 4 * 4096 * 4, 512, 44
NUMBERS = ("profile", "submitted", "chunk", "bank_bytes", "input_calls",
           "output_calls", "input_bytes", "output_bytes", "host_hidden_bytes",
           "live_bank_bytes", "held_ticket_bytes")
REQUIRED = frozenset(NUMBERS) | {"outcome", "scope"}
REQUIRED_ON = _source.REQUIRED_ON
REQUIRED_OFF = tuple(dict.fromkeys(_source.REQUIRED_OFF + (_source.ENV,)))
REQUIRED_VALUES = _source.REQUIRED_VALUES
QUICK_REQUESTS = (("prefill-a", 1872, 32), ("decode-a", 33, 128))
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
                "require_mapped_probation", "require_vmm_pressure"))):
        raise ValueError("Device baton requires one isolated quick-plan N1 text run without other experiments")
    for key, value in ([(key, "1") for key in REQUIRED_ON]
                       + [(key, "0") for key in REQUIRED_OFF]
                       + list(REQUIRED_VALUES.items())):
        if settings.get(key) != value:
            raise ValueError(f"Device baton requires explicit {key}={value}")
    if settings.get(ENV) not in ("0", "1"):
        raise ValueError("Device baton qualification requires explicit flag0/1")
    effective = settings if effective is None else effective
    if any(effective.get(key) not in (None, "")
           for key in ("TS_GLM_TRACE", "TS_GLM_TRACE_DUMP_DIR")):
        raise ValueError("Device baton requires tracing unset")


def _event(values):
    if set(values) != REQUIRED or values["scope"] != SCOPE:
        raise ValueError("Missing, unknown or unsupported bank event fields/scope")
    event = {key: _uint(values[key]) for key in NUMBERS}
    event.update(outcome=values["outcome"], scope=SCOPE)
    submitted = event["submitted"]
    if (event["profile"] != 1 or event["chunk"] != CHUNK
            or not CHUNK < submitted <= 2048
            or event["outcome"] not in ("completed", "failed")
            or event["bank_bytes"] != ROW_BYTES * submitted
            or event["host_hidden_bytes"] != 0):
        raise ValueError("Unsupported bank profile or inconsistent raw payload")
    calls = TRANSITIONS * ((submitted + CHUNK - 1) // CHUNK)
    size = TRANSITIONS * event["bank_bytes"]
    for side in ("input", "output"):
        if event[side + "_calls"] > calls or event[side + "_bytes"] > size:
            raise ValueError("Bank transfer counters exceed the fixed trunk geometry")
        if event["outcome"] == "completed" and (
                event[side + "_calls"] != calls or event[side + "_bytes"] != size):
            raise ValueError("Completed bank event lacks full transfer coverage")
    if (event["live_bank_bytes"] > event["bank_bytes"]
            or event["held_ticket_bytes"] > event["bank_bytes"]
            or event["outcome"] == "completed" and (
                event["live_bank_bytes"] or event["held_ticket_bytes"])):
        raise ValueError("Completed bank event still owns physical payload or a ticket")
    return event


def records(text):
    """Retain event order relative to actual selected PP/N1 cache completions."""
    events, malformed, wide_before, decode_before = [], 0, 0, 0
    for line in text.splitlines():
        try:
            if MARKER in line:
                event = _event(_pairs(line, MARKER))
                event.update(prefill_calls_before=wide_before, decode_calls_before=decode_before)
                events.append(event)
            elif _source.MARKER in line:
                rows = _uint(_pairs(line, _source.MARKER)["rows"])
                wide_before += int(rows > 1)
                decode_before += int(rows == 1)
        except (KeyError, TypeError, ValueError):
            malformed += 1
    return dict(records=events, malformed_records=malformed,
                source_activity=_source.prefetch_records(text), scope=SCOPE)


def qualification(rows, requested):
    """Fixed quick plan, no startup credit and no ordinary fallback evidence."""
    valid_request = type(requested) is int and requested in (0, 1)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        rows = []
    observations = [row.get("layer_major_device_baton", {}) for row in rows]
    observations = [observed if isinstance(observed, dict) else {} for observed in observations]
    base_rows = []
    for row, observed in zip(rows, observations):
        activity = observed.get("source_activity", {})
        base_rows.append(dict(row, file_source_view_prefetch=activity if isinstance(activity, dict) else {}))
    source = _source.prefetch_qualification(base_rows, 0, 0)
    requests = []
    exact_plan = len(rows) == len(QUICK_REQUESTS)
    for index, (row, source_request) in enumerate(zip(rows, source["requests"])):
        observed = observations[index]
        events, error = observed.get("records", []), None
        expected = QUICK_REQUESTS[index] if index < len(QUICK_REQUESTS) else (None, 0, 0)
        name, prompt, count = expected
        valid = (valid_request and exact_plan and row.get("name") == name
                 and type(row.get("round")) is int and row["round"] == 0
                 and type(row.get("prompt_tokens")) is int and row["prompt_tokens"] == prompt
                 and type(row.get("eval_tokens")) is int and row["eval_tokens"] == count
                 and type(row.get("cache_hits")) is int and row["cache_hits"] == 0
                 and not row.get("error") and source_request["passed"])
        try:
            if (not isinstance(events, list)
                    or type(observed.get("malformed_records")) is not int
                    or observed["malformed_records"] != 0 or observed.get("scope") != SCOPE):
                raise ValueError("Malformed measured bank records")
            for event in events:
                if not isinstance(event, dict) or set(event) != REQUIRED | {
                        "prefill_calls_before", "decode_calls_before"}:
                    raise ValueError("Decoded bank schema changed")
                if any(type(event[key]) is not int for key in NUMBERS):
                    raise ValueError("Decoded bank counters must be integers")
                fields = {key: event[key] for key in REQUIRED}
                _event(fields)
                for key in ("prefill_calls_before", "decode_calls_before"):
                    if type(event.get(key)) is not int:
                        raise ValueError("Missing selected-call chronology")
                    _uint(event[key])
            source_events = observed["source_activity"]["records"]
            widths = [min(CHUNK, prompt - start) for start in range(0, prompt, CHUNK)]
            expected_order = [(layer, width) for layer in sorted(_source.LAYERS) for width in widths]
            expected_order += [(layer, 1) for _ in range(count) for layer in sorted(_source.LAYERS)]
            if [(event["layer"], event["rows"]) for event in source_events] != expected_order:
                raise ValueError("Selected PP/N1 chronology or chunk width is incomplete")
            if requested == 1 and prompt > CHUNK:
                if (len(events) != 1 or events[0]["outcome"] != "completed"
                        or events[0]["submitted"] != prompt
                        or events[0]["prefill_calls_before"] != len(_source.LAYERS) * len(widths)
                        or events[0]["decode_calls_before"] != 0):
                    raise ValueError("Long PP lacks one completed closed bank before N1 decode")
            elif events:
                raise ValueError("OFF, short PP or N1 decode cannot supply a bank event")
        except (KeyError, TypeError, ValueError) as problem:
            error, valid = str(problem), False
        complete = [event for event in events if isinstance(event, dict)
                    and event.get("outcome") == "completed"] if isinstance(events, list) else []
        requests.append(dict(name=row.get("name"), passed=bool(valid),
            completed_calls=len(complete) if valid else 0,
            bank_bytes=sum(event["bank_bytes"] for event in complete) if valid else 0,
            input_bytes=sum(event["input_bytes"] for event in complete) if valid else 0,
            output_bytes=sum(event["output_bytes"] for event in complete) if valid else 0,
            source_phases=source_request["phases"], error=error or source_request["error"],
            malformed_records=observed.get("malformed_records", 0)))
    return dict(passed=bool(valid_request and exact_plan and source["passed"] and requests
                           and all(row["passed"] for row in requests)), requested_baton=requested,
        requests=requests, completed_calls=sum(row["completed_calls"] for row in requests),
        scope=SCOPE, all_kind5_owners_closed_qualified=False, residency_qualified=False,
        physical_io_qualified=False, performance_qualified=False, numerical_quality_qualified=False)
