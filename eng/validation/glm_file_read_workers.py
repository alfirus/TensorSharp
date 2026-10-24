"""Qualify joined exact-file task activity, separately from output and speed."""
import re

ENV = "TS_HOST_MOE_FILE_READ_WORKERS"
HINT_ENV = "TS_HOST_MOE_FILE_RANDOM_ACCESS"
HINT_FIELD = "file_read_random_access"
SCOPE = "nested_host_wall_not_cuda_kernel_or_physical_io"
MARKER = "[HOSTMOE-CACHE-TIMING]"
NUMBERS = ("read_pool_requested", "read_pool_workers", "read_pool_maximum", "read_pool_slots",
           "file_read_calls", "file_read_tasks", "file_read_completed_tasks",
           "file_read_failed_calls", "file_read_source_bytes", "file_bytes")
REQUIRED = set(NUMBERS) | {"rows", "outcome", "scope"}


def setting(value):
    if value is None:
        return None
    if value not in tuple(str(number) for number in range(17)):
        raise ValueError(ENV + " must be unset or canonical decimal 0..16")
    return int(value)


def hint_setting(value):
    if value not in (None, "0", "1"):
        raise ValueError(HINT_ENV + " must be unset, 0 or 1")
    return None if value is None else int(value)


def records(text):
    result, malformed = [], 0
    for line in text.splitlines():
        if MARKER not in line:
            continue
        try:
            pairs = [token.split("=", 1) for token in line.split(MARKER, 1)[1].split()]
            if (any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs)
                    or not REQUIRED.issubset({pair[0] for pair in pairs})):
                raise ValueError("Missing, duplicate or unexpected file-task field")
            values = dict(pairs)
            if (values["scope"] != SCOPE or values["outcome"] not in ("completed", "exception", "not_completed")
                    or any(re.fullmatch(r"[0-9]+", values[key]) is None for key in NUMBERS+("rows",))):
                raise ValueError("Invalid file-task field")
            item = {key: int(values[key]) for key in NUMBERS+("rows",)}
            if any(number > 2**64-1 for number in item.values()) or not item["rows"]:
                raise ValueError("File-task counter overflow")
            if HINT_FIELD in values and values[HINT_FIELD] not in ("0", "1"):
                raise ValueError("Invalid applied source hint")
            item[HINT_FIELD] = None if HINT_FIELD not in values else int(values[HINT_FIELD])
            if not any(item[key] for key in NUMBERS):
                if item[HINT_FIELD] not in (None, 0):
                    raise ValueError("Inactive call claimed a source hint")
                item["outcome"] = values["outcome"]
                result.append(item)
                continue
            requested, actual, maximum, slots = (item[key] for key in (
                "read_pool_requested", "read_pool_workers", "read_pool_maximum", "read_pool_slots"))
            if (not 0 <= requested <= 16 or not 1 <= maximum <= 15
                    or actual != (maximum if requested == 0 else min(requested, maximum))
                    or (actual > 1 and slots != actual+1)
                    or (actual == 1 and (slots not in (1, 2) if requested == 0 else slots != 1))
                    or not 1 <= item["file_read_calls"] <= item["file_read_tasks"]
                    or item["file_read_completed_tasks"] > item["file_read_tasks"]
                    or item["file_read_failed_calls"] > item["file_read_calls"]
                    or item["file_read_source_bytes"] > item["file_bytes"]):
                raise ValueError("Inconsistent joined file-task counters")
            item["outcome"] = values["outcome"]
            result.append(item)
        except (KeyError, ValueError):
            malformed += 1
    return dict(records=result, malformed_records=malformed, scope=SCOPE)


def qualification(rows, requested, expected_hint=None):
    requests = []
    for row in rows:
        observed = row.get("file_read_workers", {})
        events = observed.get("records", [])
        active = [event for event in events if event["file_read_calls"]]
        eligible = bool(active) and not observed.get("malformed_records", 0)
        for event in events:
            eligible = eligible and event["outcome"] == "completed"
            if not event["file_read_calls"]:
                continue
            eligible = eligible and (event["read_pool_requested"] == requested
                and event["file_read_failed_calls"] == 0
                and event["file_read_tasks"] == event["file_read_completed_tasks"]
                and event["file_read_source_bytes"] == event["file_bytes"] > 0)
            if expected_hint is not None:
                eligible = eligible and event[HINT_FIELD] == expected_hint
        # The benchmark profile explicitly disables all RAM/prefetch tiers.
        # Positive completed source bytes prove actual reads, not startup setup.
        n1_calls = sum(event["file_read_calls"] for event in active
            if event["rows"] == 1 and event["file_read_source_bytes"] > 0)
        requests.append(dict(name=row.get("name"), passed=bool(eligible and n1_calls > 0),
            n1_read_calls=n1_calls,
            wide_read_calls=sum(event["file_read_calls"] for event in active if event["rows"] > 1),
            calls=sum(event["file_read_calls"] for event in events),
            tasks=sum(event["file_read_completed_tasks"] for event in events),
            source_read_bytes=sum(event["file_read_source_bytes"] for event in events),
            actual_workers=sorted({event["read_pool_workers"] for event in active}),
            maximum_workers=sorted({event["read_pool_maximum"] for event in active}),
            observed_random_access_hints=sorted({event[HINT_FIELD] for event in active if event[HINT_FIELD] is not None}),
            missing_hint_records=sum(event[HINT_FIELD] is None for event in active),
            malformed_records=observed.get("malformed_records", 0)))
    all_actual = {actual for request in requests for actual in request["actual_workers"]}
    all_maximum = {maximum for request in requests for maximum in request["maximum_workers"]}
    return dict(passed=isinstance(requested, int) and not isinstance(requested, bool)
        and requested in range(17) and expected_hint in (None, 0, 1) and bool(requests)
        and len(all_actual) == len(all_maximum) == 1
        and all(request["passed"] for request in requests), requested_workers=requested,
        requested_random_access_hint=expected_hint,
        requests=requests, scope=SCOPE, physical_io_qualified=False, performance_qualified=False)
