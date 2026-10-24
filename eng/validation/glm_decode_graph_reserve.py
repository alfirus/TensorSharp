"""Strict, request-scoped telemetry checks for the fixed GLM decode profile.

This module measures owned credit. It cannot certify resident expert slots,
physical VRAM freed, output quality, or a speed improvement. Callers must retain
their independent completed-API, exact-output and native-artifact gates.
"""

import re


DECODE_GRAPH_RESERVE_ENV = "TS_GLM_DECODE_GRAPH_RESERVE"
RESERVE_SCOPE = "owned_device_credit_not_physical_free"
RECOVERY_SCOPE = "owned_credit_not_physical_free"
# Retain the initial public name for reserve-event callers.
SCOPE = RESERVE_SCOPE
MIB = 1 << 20
U64_MAX = (1 << 64) - 1
RESERVE_MARKER = "[GLM-DECODE-GRAPH-RESERVE]"
RECOVERY_MARKER = "[GLM-EXPERT-RECOVERY]"
RESERVE_NUMBERS = (
    "rows", "output_rows", "context", "owned_before", "owned_after",
    "live_device", "released", "expert_budget",
)
RECOVERY_NUMBERS = (
    "ceiling_before", "ceiling_candidate", "ceiling_target", "ceiling_after",
    "ceiling_max", "native_accounted", "native_budget", "physical_available",
    "physical_headroom", "ledger_committed", "ledger_reserved",
    "ledger_capacity", "effective_limit", "minimum_growth", "recovery_margin",
)
RECOVERY_REASONS = frozenset((
    "already_max", "physical_or_ledger_slack", "below_threshold", "increased",
    "envelope_refused",
))


def decode_graph_reserve_enabled(value):
    """Accept the experiment's exact external flag, including inherited values."""
    if value not in (None, "0", "1"):
        raise ValueError(DECODE_GRAPH_RESERVE_ENV + " must be unset, 0 or 1")
    return value == "1"


def decode_graph_workspace_bytes(context, host_token_embedding=True):
    """Workspace(context, 1, 1), for the exact 45-layer/34-KDA GLM profile.

    PrepareGraph reports a padded context. This is one whole conservative
    replacement forecast, including recurrent-output state, not measured graph
    payload. Compact raw Host embedding adds 2304 bytes to this forecast.
    """
    if (type(context) is not int or not 256 <= context <= 2048 or context % 256
            or type(host_token_embedding) is not bool):
        raise ValueError("Expected fixed-profile padded context and Boolean Host mode")
    fixed_state = 4 * 34 * (3 * 64 * 128 * (4 - 1) + 64 * 128 * 128)
    masks = 4 * 11 * context
    logits = 4 * 154880
    activation = logits + 4 * (32 * 4096 * 4 + 8 * 2048 * 8)
    return 64 * MIB + masks + activation + fixed_state + (2304 if host_token_embedding else 0)


def _uints(record, fields):
    if any(type(record[key]) is not int or not 0 <= record[key] <= U64_MAX for key in fields):
        raise ValueError("Invalid unsigned integer telemetry")


def _validate_reserve(record):
    if set(record) != set(RESERVE_NUMBERS) | {"scope"} or record["scope"] != RESERVE_SCOPE:
        raise ValueError("Unexpected decode reserve fields/scope")
    _uints(record, RESERVE_NUMBERS)
    # This helper intentionally qualifies the single-primary N1 profile only.
    # ForwardNative reports raw output_rows=0: Target resolves that sentinel
    # to one final logits row. Explicit one-row callers are equally valid.
    if (record["rows"] != 1 or record["output_rows"] not in (0, 1)
            or not 256 <= record["context"] <= 2048 or record["context"] % 256
            or record["live_device"] > record["owned_after"]
            or record["released"] != max(0, record["owned_before"] - record["owned_after"])):
        raise ValueError("Invalid first-N1 reserve shape/ownership")


def _validate_recovery(record):
    fields = set(RECOVERY_NUMBERS) | {"request", "reason", "sample", "unit", "scope"}
    if (set(record) != fields or record["scope"] != RECOVERY_SCOPE or record["unit"] != "bytes"
            or record["sample"] != "before_recovery"
            or record["reason"] not in RECOVERY_REASONS
            or not isinstance(record["request"], str) or not record["request"]
            or any(char.isspace() for char in record["request"])):
        raise ValueError("Unexpected recovery fields/scope")
    _uints(record, RECOVERY_NUMBERS)
    before, candidate, target, after, maximum = (record[key] for key in (
        "ceiling_before", "ceiling_candidate", "ceiling_target", "ceiling_after", "ceiling_max"))
    committed, reserved = record["ledger_committed"], record["ledger_reserved"]
    used = committed + reserved
    if (used > U64_MAX or used > record["ledger_capacity"] or before > used
            or not before <= candidate <= maximum or not before <= target <= candidate
            or after not in (before, target) or record["native_budget"] != after
            or record["native_accounted"] > after
            or not committed <= record["effective_limit"] <= record["ledger_capacity"]
            or record["minimum_growth"] not in (64 * MIB, 512 * MIB)
            or record["recovery_margin"] != 128 * MIB):
        raise ValueError("Invalid recovery bounds/ledger")
    # The physical branch uses committed + max(0, free - headroom), never
    # reserved + free. Reported effective_limit may also be ceiling-bound.
    physical_limit = committed + max(0, record["physical_available"] - record["physical_headroom"])
    if record["effective_limit"] > min(record["ledger_capacity"], physical_limit):
        raise ValueError("Recovery limit spends unobserved physical availability")
    increase = max(0, record["effective_limit"] - used - record["recovery_margin"])
    unrounded = min(maximum, before + increase)
    expected_candidate = max(before, unrounded // MIB * MIB)
    expected_target = expected_candidate if expected_candidate - before >= record["minimum_growth"] else before
    if candidate != expected_candidate or target != expected_target:
        raise ValueError("Recovery target differs from source policy arithmetic")
    expected_reason = (
        "increased" if target > before and after == target else
        "envelope_refused" if target > before else
        "already_max" if before == maximum else
        "physical_or_ledger_slack" if candidate == before else "below_threshold"
    )
    if record["reason"] != expected_reason:
        raise ValueError("Recovery outcome differs from source policy")


def _records(log, marker, numbers, other_fields, validate, scope):
    required = set(numbers) | set(other_fields)
    records, malformed = [], 0
    for line in log.splitlines():
        if marker not in line:
            continue
        try:
            pairs = [token.split("=", 1) for token in line.split(marker, 1)[1].split()]
            if (any(len(pair) != 2 for pair in pairs) or len(pairs) != len(required)
                    or {pair[0] for pair in pairs} != required):
                raise ValueError("Missing, duplicate or unexpected telemetry field")
            values = dict(pairs)
            if any(re.fullmatch(r"[0-9]+", values[key]) is None for key in numbers):
                raise ValueError("Non-unsigned numeric telemetry")
            record = {key: int(values[key]) for key in numbers}
            record.update({key: values[key] for key in other_fields})
            validate(record)
            records.append(record)
        except (KeyError, ValueError, OverflowError, TypeError):
            malformed += 1
    return dict(records=records, malformed_records=malformed, scope=scope)


def decode_graph_reserve_records(request_log):
    return _records(request_log, RESERVE_MARKER, RESERVE_NUMBERS, ("scope",), _validate_reserve, RESERVE_SCOPE)


def expert_recovery_records(request_log):
    return _records(request_log, RECOVERY_MARKER, RECOVERY_NUMBERS,
                    ("request", "reason", "sample", "unit", "scope"), _validate_recovery, RECOVERY_SCOPE)


def decode_graph_reserve_qualification(rows, host_token_embedding=True):
    """Require one actual first-N1 credit/recovery pair per completed request.

    Credit can qualify when physical/ledger limits prevent a budget increase.
    `ceiling_increased` is separate and neither result qualifies new resident
    slots. Logs must already be sliced around each isolated successful API call.
    """
    if type(host_token_embedding) is not bool:
        raise ValueError("Host embedding mode must be explicit Boolean")
    requests, seen_request_ids = [], set()
    for row in rows:
        result = dict(name=row.get("name"), passed=False, ceiling_increased=False)
        try:
            if (row.get("error") or type(row.get("eval_tokens")) is not int or row["eval_tokens"] <= 0
                    or type(row.get("cache_hits")) is not int or row["cache_hits"] != 0
                    or type(row.get("prompt_tokens")) is not int or not 1 <= row["prompt_tokens"] < 2048):
                raise ValueError("Requires a successful uncached request with actual decode tokens")
            reserve, recovery = row["decode_graph_reserve"], row["expert_recovery"]
            if (type(reserve["malformed_records"]) is not int or reserve["malformed_records"] != 0
                    or type(recovery["malformed_records"]) is not int or recovery["malformed_records"] != 0
                    or len(reserve["records"]) != 1 or len(recovery["records"]) != 1):
                raise ValueError("Requires exactly one unambiguous first-N1/recovery pair")
            event, decision = reserve["records"][0], recovery["records"][0]
            _validate_reserve(event)
            _validate_recovery(decision)
            workspace = decode_graph_workspace_bytes(event["context"], host_token_embedding)
            expected_context = ((row["prompt_tokens"] + 1 + 255) // 256) * 256
            live = event["live_device"]
            expected_after = live + workspace
            expected_before = max(2 * workspace, expected_after)
            minimum = 64 * MIB if event["released"] >= 64 * MIB else 512 * MIB
            if (event["context"] != expected_context or expected_after > U64_MAX
                    or event["owned_after"] != expected_after
                    or event["owned_before"] != expected_before or event["released"] <= 0
                    or decision["minimum_growth"] != minimum
                    or event["expert_budget"] != decision["ceiling_after"]
                    or decision["request"] == "-" or decision["request"] in seen_request_ids):
                raise ValueError("First-N1 workspace/release/recovery mismatch")
            seen_request_ids.add(decision["request"])
            result.update(passed=True, workspace_bytes=workspace, released_bytes=event["released"],
                live_device_bytes=live, owned_before=event["owned_before"], owned_after=event["owned_after"],
                expert_budget_before=decision["ceiling_before"], expert_budget_after=decision["ceiling_after"],
                expert_budget_delta=decision["ceiling_after"] - decision["ceiling_before"],
                ceiling_max=decision["ceiling_max"], recovery_reason=decision["reason"],
                ceiling_increased=decision["ceiling_after"] > decision["ceiling_before"])
        except (KeyError, ValueError, TypeError, IndexError) as error:
            result["error"] = str(error)
        requests.append(result)
    return dict(passed=bool(requests) and all(row["passed"] for row in requests), requests=requests,
        ceiling_increased=bool(requests) and all(row["passed"] and row["ceiling_increased"] for row in requests),
        resident_capacity_qualified=False,
        scope="Actual first-N1 owned replacement credit and recovery; resident slots, physical VRAM, quality and speed require separate evidence")
