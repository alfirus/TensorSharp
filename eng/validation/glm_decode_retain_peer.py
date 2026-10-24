"""Request-scoped proof of published expert payload retained by partial rebalance.

This is cache activity evidence. It establishes neither new physical capacity,
unique RAM/file-cache pages, physical SSD reads, quality nor performance.
"""
import re

ENV = "TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER"
PARENT_ENV = "TS_HOST_MOE_DECODE_REBALANCE"
TAG = "[HOSTMOE-REBALANCE-RETAIN]"
SCOPE = "published_gpu_payload_retained_not_new_capacity_or_physical_free"
NUMBERS = ("profile", "layer", "slots", "hot", "owned_bytes", "decode_quota",
           "room_bytes", "absent_floor_bytes", "absent_share_bytes")
KEYS = set(NUMBERS) | {"outcome", "scope"}
U64_MAX = (1 << 64) - 1


def enabled(value):
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{ENV} requires canonical 0 or 1")


def _validate(event):
    if set(event) != KEYS:
        raise ValueError("Missing, duplicate or unknown retained-peer fields")
    if any(type(event[key]) is not int or not 0 <= event[key] <= U64_MAX for key in NUMBERS):
        raise ValueError("Retained-peer field is not an unsigned integer")
    if (event["profile"] != 1 or event["outcome"] != "completed" or event["scope"] != SCOPE
            or not 1 <= event["slots"] <= 65536 or not 1 <= event["hot"] <= event["slots"]
            or event["layer"] > 65535 or not 0 < event["decode_quota"] < event["owned_bytes"]
            or not 0 < event["absent_floor_bytes"] <= event["room_bytes"] < event["absent_share_bytes"]):
        raise ValueError("Retained-peer event does not prove a useful eligible peer survived")


def records(log):
    events, malformed = [], 0
    for line in log.splitlines():
        if TAG not in line:
            continue
        try:
            if line.count(TAG) != 1:
                raise ValueError("Ambiguous retained-peer marker")
            fields = line.split(TAG, 1)[1].split()
            pairs = [field.split("=", 1) for field in fields]
            if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
                raise ValueError("Malformed retained-peer field")
            event = dict(pairs)
            for key in NUMBERS:
                if not re.fullmatch(r"0|[1-9][0-9]*", event.get(key, "")):
                    raise ValueError("Noncanonical retained-peer integer")
                event[key] = int(event[key])
            _validate(event)
            events.append(event)
        except (ValueError, TypeError):
            malformed += 1
    return dict(records=events, malformed_records=malformed)


def qualification(rows):
    requests = []
    for row in rows:
        result = dict(name=row.get("name"), valid=False, passed=False, activity="not_observed", retained_hot_slots=0,
                      retained_payload_bytes=0, newly_granted_bytes=0)
        try:
            if (row.get("error") or type(row.get("eval_tokens")) is not int or row["eval_tokens"] <= 0
                    or type(row.get("cache_hits")) is not int or row["cache_hits"] != 0
                    or type(row.get("prompt_tokens")) is not int or row["prompt_tokens"] <= 0):
                raise ValueError("Requires a successful uncached request with actual decode")
            activity = row["decode_retain_peer"]
            if type(activity["malformed_records"]) is not int or activity["malformed_records"] != 0:
                raise ValueError("Malformed retained-peer activity in measured request")
            events = activity["records"]
            if not isinstance(events, list):
                raise ValueError("Retained-peer records are not a list")
            seen = set()
            for event in events:
                _validate(event)
                if event["layer"] in seen:
                    raise ValueError("Duplicate retained peer in measured request")
                seen.add(event["layer"])
            total = sum(event["owned_bytes"] for event in events)
            if total > U64_MAX:
                raise ValueError("Retained payload sum overflow")
            result.update(valid=True, passed=bool(events), activity="observed" if events else "not_observed",
                          retained_hot_slots=sum(event["hot"] for event in events), retained_payload_bytes=total, events=events)
        except (KeyError, ValueError, TypeError, IndexError) as error:
            result["error"] = str(error)
        requests.append(result)
    active = sum(row["passed"] for row in requests)
    return dict(passed=bool(requests) and all(row["valid"] for row in requests) and active > 0, requests=requests,
                active_requests=active, all_requests_active=bool(requests) and active == len(requests),
                scope=SCOPE, physical_capacity_qualified=False,
                limitations="Only requests with actual events qualify retention; empty activity gets no credit. Published ownership is not physical capacity, SSD reads, quality or speed")
