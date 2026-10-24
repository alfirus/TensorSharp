"""Qualify joined immutable-view reads; CPU service and model speed are separate.

Only newly appended bytes for one successful request may be supplied. Every
measured PP layer and every N1 layer/postcall position must be present. These
counters describe complete logical copies, never physical RAM or SSD traffic.
Demand-prefetch counters qualify API requests for the same joined miss slices;
they do not certify OS completion or future residency.
"""
import collections
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_source_view_log_common",
    Path(__file__).with_name("glm_file_read_sort.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)

ENV = "TS_HOST_MOE_FILE_SOURCE_VIEW"
PREFETCH_ENV = "TS_HOST_MOE_FILE_SOURCE_VIEW_PREFETCH"
MARKER, FORWARD_MARKER, SCOPE = _common.MARKER, _common.FORWARD_MARKER, _common.SCOPE
LAYERS, U64_MAX = _common.LAYERS, _common.U64_MAX
VIEW_COUNTERS = ("file_view_calls", "file_view_bytes", "file_view_fallbacks",
                 "file_view_buffered_bytes")
NUMBERS = ("layer", "rows", "bulk", "file_bytes", "file_read_completed_tasks",
           "file_read_failed_calls", "file_read_source_bytes", "file_view_requested") + VIEW_COUNTERS
REQUIRED = frozenset(NUMBERS) | {"outcome", "scope"}
REQUIRED_ON = _common.REQUIRED_ON
REQUIRED_OFF = _common.REQUIRED_OFF + (_common.ENV,)
REQUIRED_OFF += (PREFETCH_ENV,)
REQUIRED_VALUES = _common.REQUIRED_VALUES
_uint, _pairs = _common._uint, _common._pairs


def setting(value):
    if value not in (None, "0", "1"):
        raise ValueError(ENV + " must be unset, 0 or 1")
    return None if value is None else int(value)


def prefetch_setting(value):
    if value not in (None, "0", "1"):
        raise ValueError(PREFETCH_ENV + " must be unset, 0 or 1")
    return None if value is None else int(value)


def validate_profile(args, settings, *, prefetch=False):
    if (args.concurrent or not args.unified_memory or args.spec or args.prefix_cache
            or args.repeat != 1 or args.plan != "quick" or args.gpu_telemetry
            or args.server_arg or args.mmproj != "none" or args.think
            or args.require_file_read_workers or args.require_file_random_access
            or args.require_file_read_sort or args.require_mapped_probation or args.require_vmm_pressure):
        raise ValueError("File-source view requires one isolated quick-plan N1 text run")
    if prefetch and args.require_file_source_view:
        raise ValueError("Pure source-view and combined demand-prefetch qualification are distinct")
    for key, value in ([(key, "1") for key in REQUIRED_ON]
                       + [(key, "0") for key in REQUIRED_OFF
                          if not (prefetch and key == PREFETCH_ENV)]
                       + list(REQUIRED_VALUES.items())):
        if settings.get(key) != value:
            raise ValueError(f"File-source view requires explicit {key}={value}")
    if settings.get(ENV) not in ("0", "1"):
        raise ValueError("File-source view requires an explicit requested flag")
    if prefetch and settings.get(PREFETCH_ENV) != settings[ENV]:
        raise ValueError("Demand prefetch requires explicit source-view/prefetch 0/0 or 1/1")


def _event(values):
    if (not REQUIRED.issubset(values) or values["scope"] != SCOPE
            or any(key.startswith("file_view_") and key not in NUMBERS for key in values)):
        raise ValueError("Missing view activity fields or wrong scope")
    event = {key: _uint(values[key]) for key in NUMBERS}
    event.update(outcome=values["outcome"], scope=SCOPE)
    if (event["outcome"] not in ("completed", "exception", "not_completed")
            or event["layer"] not in LAYERS
            or not (event["rows"] == 1 or 9 <= event["rows"] <= 512)
            or event["bulk"] != int(event["rows"] > 1)
            or event["file_view_requested"] not in (0, 1)):
        raise ValueError("Unsupported selected-call shape or status")
    calls, size, fallback, buffered = (event[key] for key in VIEW_COUNTERS)
    if fallback > calls or (size and not calls) or (not event["file_view_requested"] and any((calls, size, fallback))):
        raise ValueError("Inconsistent view activity")
    if event["outcome"] == "completed":
        if (event["file_read_failed_calls"] or calls > event["file_read_completed_tasks"]
                or size + buffered > U64_MAX
                or size + buffered != event["file_bytes"]
                or event["file_read_source_bytes"] != event["file_bytes"]
                or (not event["file_bytes"] and (calls or event["file_read_completed_tasks"]))
                or (event["file_bytes"] and not event["file_read_completed_tasks"])):
            raise ValueError("Unjoined or nonconserving source activity")
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
        observed = row.get("file_source_view", {})
        events, event_error = observed.get("records", []), None
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
            event_error, events = str(error), []
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
            valid = all(width > 1 for width in widths)
            valid = valid and all(event["outcome"] == "completed"
                and event["file_view_requested"] == requested
                and (requested != 1 or (event["file_view_fallbacks"] == 0
                    and event["file_view_bytes"] == event["file_bytes"])) for event in events)
            for layer in LAYERS:
                valid = valid and (collections.Counter(event["rows"] for event in prefill
                    if event["layer"] == layer) == collections.Counter(widths))
                positions = [event.get("forward_p0") for event in decode if event["layer"] == layer]
                valid = valid and positions == list(range(prompt, prompt + count))
        phases = {}
        for label, phase in (("prefill", prefill), ("decode", decode)):
            totals = {key: sum(event[key] for event in phase) for key in VIEW_COUNTERS}
            phases[label] = dict(calls=len(phase), **totals,
                source_logical_bytes=sum(event["file_bytes"] for event in phase),
                mapped_activity_observed=totals["file_view_bytes"] > 0)
            valid = valid and all(value <= U64_MAX for value in totals.values())
            valid = valid and phases[label]["source_logical_bytes"] <= U64_MAX
            if requested == 1:
                valid = valid and phases[label]["mapped_activity_observed"]
        requests.append(dict(name=row.get("name"), passed=bool(valid), phases=phases,
            error=event_error, malformed_records=observed.get("malformed_records", 0)))
    return dict(passed=type(requested) is int and requested in (0, 1) and bool(requests)
        and all(row["passed"] for row in requests), requested_view=requested, requests=requests,
        scope="Joined logical mapped or buffered source copies; numerical quality and speed are separate",
        physical_io_qualified=False, residency_qualified=False, performance_qualified=False,
        numerical_quality_qualified=False)


PREFETCH_COUNTERS = ("file_prefetch_calls", "file_prefetch_ranges",
                     "file_prefetch_bytes", "file_prefetch_failures")
PREFETCH_NUMBERS = ("file_prefetch_requested",) + PREFETCH_COUNTERS


def _prefetch_event(values):
    event = _event(values)
    if (not set(PREFETCH_NUMBERS).issubset(values)
            or any(key.startswith("file_prefetch_") and key not in PREFETCH_NUMBERS
                   for key in values)):
        raise ValueError("Missing or unknown demand-prefetch activity fields")
    event.update((key, _uint(values[key])) for key in PREFETCH_NUMBERS)
    requested = event["file_prefetch_requested"]
    calls, ranges, size, failures = (event[key] for key in PREFETCH_COUNTERS)
    if requested not in (0, 1) or (not requested and any((calls, ranges, size, failures))):
        raise ValueError("Disabled demand prefetch claimed activity")
    if not calls:
        if any((ranges, size, failures)):
            raise ValueError("Inactive demand prefetch claimed work")
    elif (event["outcome"] != "completed" or event["file_view_requested"] != 1
            or not calls <= ranges <= 64 * calls or not ranges <= size <= event["file_bytes"]
            or failures > calls or ranges > event["file_read_completed_tasks"]):
        raise ValueError("Inconsistent joined demand-prefetch activity")
    return event


def prefetch_records(text):
    """Reuse the view parser's exact following-postcall N1 association."""
    observed = records(text)
    lines = [line for line in text.splitlines() if MARKER in line]
    events, malformed = [], observed["malformed_records"]
    if len(lines) != len(observed["records"]):
        malformed += 1
    for index, line in enumerate(lines):
        try:
            event = _prefetch_event(_pairs(line, MARKER))
            if index < len(observed["records"]):
                forward = observed["records"][index].get("forward_p0")
                if forward is not None:
                    event["forward_p0"] = forward
            events.append(event)
        except (KeyError, TypeError, ValueError):
            malformed += 1
    return dict(records=events, malformed_records=malformed, scope=SCOPE)


def prefetch_qualification(rows, requested_view, requested_prefetch):
    """Joined API requests, not OS completion, RAM residency, or physical I/O."""
    view_rows = [dict(row, file_source_view=row.get("file_source_view_prefetch", {}))
                 for row in rows]
    base = qualification(view_rows, requested_view)
    requests = []
    pair_valid = (type(requested_view) is int and type(requested_prefetch) is int
                  and (requested_view, requested_prefetch) in ((0, 0), (1, 0), (1, 1)))
    for row, view_request in zip(rows, base["requests"]):
        observed = row.get("file_source_view_prefetch", {})
        events, event_error = observed.get("records", []), None
        valid = pair_valid and view_request["passed"]
        try:
            if not isinstance(events, list):
                raise ValueError("Demand-prefetch records must be a list")
            for event in events:
                if any(type(event[key]) is not int for key in NUMBERS + PREFETCH_NUMBERS):
                    raise ValueError("Decoded demand-prefetch counters must be integers")
                _prefetch_event(event)
                if event["file_prefetch_requested"] != requested_prefetch:
                    raise ValueError("Actual demand-prefetch request flag differs")
                if requested_prefetch and (event["file_prefetch_failures"]
                        or event["file_prefetch_bytes"] != event["file_bytes"]
                        or event["file_prefetch_bytes"] != event["file_view_bytes"]
                        or event["file_prefetch_ranges"] != event["file_read_completed_tasks"]
                        or (event["file_bytes"] and not event["file_prefetch_calls"])):
                    raise ValueError("Demand prefetch did not cover every joined miss slice")
        except (KeyError, TypeError, ValueError) as error:
            event_error, events, valid = str(error), [], False
        phases = {}
        for label, phase in (("prefill", [event for event in events if event["rows"] > 1]),
                             ("decode", [event for event in events if event["rows"] == 1])):
            totals = {key: sum(event[key] for event in phase) for key in PREFETCH_COUNTERS}
            phases[label] = dict(view_request["phases"][label], **totals,
                prefetch_activity_observed=totals["file_prefetch_bytes"] > 0)
            valid = valid and all(value <= U64_MAX for value in totals.values())
            if requested_prefetch:
                valid = valid and phases[label]["prefetch_activity_observed"]
        requests.append(dict(name=row.get("name"), passed=bool(valid), phases=phases,
            error=event_error or view_request["error"],
            malformed_records=observed.get("malformed_records", 0)))
    return dict(passed=bool(pair_valid and base["passed"] and requests
        and all(request["passed"] for request in requests)), requested_view=requested_view,
        requested_prefetch=requested_prefetch, requests=requests,
        scope="Joined exact-demand PrefetchVirtualMemory API ranges; OS completion, quality and speed are separate",
        physical_io_qualified=False, residency_qualified=False, performance_qualified=False,
        numerical_quality_qualified=False)
