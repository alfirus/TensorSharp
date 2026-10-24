"""Qualification for the first explicit GLM VMM weight-cache profile.

Requested flags certify neither execution nor physical capacity. Runtime event
qualification is separate from this prelaunch isolation check.
"""
import collections
import re

ENV = "TS_HOST_MOE_VMM_WEIGHTS"
PARENT_ENV = "TS_HOST_MOE_DECODE_REBALANCE"

REQUIRED_ON = (
    "TS_GLM_HOST_TOKEN_EMBEDDING", PARENT_ENV,
    "TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS", "TS_HOST_MOE_EXPERT_CACHE_TIMING",
    "TS_HOST_MOE_DECODE_STAGING_72MB", "TS_HOST_MOE_PREFILL_RING_72MB",
)
REQUIRED_OFF = (
    "TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB", "TS_GLM_DECODE_GRAPH_RESERVE",
    "TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER", "TS_HOST_MOE_PHASE_COLD_GROWTH",
    "TS_HOST_MOE_PREFILL_DOWN_OVERLAP", "TS_HOST_MOE_PREFILL_RING_COALESCE16",
    "TS_HOST_MOE_RAM_CACHE_MB", "TS_HOST_MOE_RAM_CACHE_MAPPED",
    "TS_HOST_MOE_RAM_CACHE_MAPPED_EXCLUSIVE_QUOTA", "TS_HOST_MOE_RAM_CACHE_MAPPED_QUOTA_MB",
    "TS_HOST_MOE_RAM_CACHE_VICTIM", "TS_HOST_MOE_RAM_CACHE_PINNED",
    "TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD", "TS_HOST_MOE_RAM_CACHE_FREQUENCY",
    "TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY", "TS_HOST_MOE_LAYER_PREFETCH",
    "TS_HOST_MOE_EXPERT_CACHE_PREFETCH", "TS_HOST_MOE_DECODE_PREFETCH",
    "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS",
    "TS_HOST_MOE_EXPERT_CACHE_LFU",
    "TS_HOST_MOE_FILE_READ_WORKERS", "TS_HOST_MOE_FILE_RANDOM_ACCESS",
    "TS_GLM_SELECTED_EXPERT_TIMING", "TS_DSV4_PERF", "TS_HOST_MOE_ROUTE_TRACE",
    "TS_GGML_Q8_ORDERED_SMALL_BATCH", "TS_GGML_Q8_ORDERED_GATE_PREFILL",
    "TS_GGML_Q8_GATE_PREFILL_TILE16", "TS_GGML_STRICT_RECURRENT_PROJECTIONS",
    "TS_GLM_OP_OFFLOAD", "TS_GLM_GRAPH_MEMORY_DIAGNOSTICS",
    "TS_GLM_FLASH_MTP", "TS_GLM_FLASH_MTP_SHAPE_PLAN", "TS_GLM_FLASH_MTP_PREFIX_COMMIT",
)
REQUIRED_VALUES = {
    "TS_UNIFIED_MAX_RUNNING_REQUESTS": "1", "MAX_CONTEXT": "2048",
    "TS_SCHED_PREFILL_CHUNK": "512", "TS_HOST_MOE_EXPERT_CACHE_RING_READS": "2",
}


def enabled(value):
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{ENV} must be unset, 0 or 1")


def validate_profile(args, settings):
    if (args.concurrent or not args.unified_memory or args.spec or args.prefix_cache
            or args.repeat != 1 or args.plan != "quick" or args.gpu_telemetry
            or args.server_arg or args.mmproj != "none" or args.require_file_read_workers
            or args.require_file_random_access or args.require_mapped_probation):
        raise ValueError("VMM weight qualification requires one isolated quick-plan N1 text run without other experiment modes")
    for key, value in ([(key, "1") for key in REQUIRED_ON]
                       + [(key, "0") for key in REQUIRED_OFF]
                       + list(REQUIRED_VALUES.items())):
        if settings.get(key) != value:
            raise ValueError(f"VMM weight qualification requires explicit {key}={value}")


TIMING_TAG = "[HOSTMOE-CACHE-TIMING]"
RESHAPE_TAG = "[HOSTMOE-VMM-RESHAPE]"
PRESSURE_TAG = "[HOSTMOE-VMM-PRESSURE]"
SCOPE = "published_gpu_payload_not_new_capacity_or_physical_free"
PRESSURE_SCOPE = "published_gpu_payload_shrink_not_new_capacity_or_physical_free"
LAYERS = frozenset(range(3, 45))
U64_MAX = (1 << 64) - 1
TIMING_NUMBERS = ("layer", "rows", "vmm_profile", "vmm_capacity", "vmm_physical_bytes",
                  "vmm_pending_bytes", "vmm_quarantine_bytes", "vmm_certificate")
RESHAPE_NUMBERS = ("profile", "phase", "refunded_bytes", "grown_bytes", "retained_hot",
                   "moved_slots", "discarded_hot", "budget", "owned_before", "owned_after",
                   "pending_bytes", "quarantine_bytes", "certificate") + tuple(
                       f"g{group}_{kind}_{when}" for group in (3, 4, 11)
                       for kind in ("cap", "bytes") for when in ("before", "after"))
PRESSURE_NUMBERS = ("profile", "mutation", "target", "owned_before", "owned_after",
                    "wide_refunded_bytes", "weight_refunded_bytes", "retained_hot",
                    "moved_slots", "discarded_hot", "active_owner_excluded", "pending_bytes",
                    "quarantine_bytes", "certificate") + tuple(
                        f"g{group}_cap_{when}" for group in (3, 4, 11) for when in ("before", "after"))


def _uint(value):
    if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        value = int(value)
    if type(value) is not int or not 0 <= value <= U64_MAX:
        raise ValueError("Expected a canonical bounded unsigned integer")
    return value


def _pairs(line, tag):
    if line.count(tag) != 1:
        raise ValueError("Duplicate activity marker")
    pairs = [field.split("=", 1) for field in line.split(tag, 1)[1].split()]
    if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
        raise ValueError("Malformed or duplicate activity field")
    return dict(pairs)


def _timing(event):
    if set(event) != set(TIMING_NUMBERS) | {"outcome"}:
        raise ValueError("Unexpected VMM timing fields")
    if any(type(event[key]) is not int or not 0 <= event[key] <= U64_MAX for key in TIMING_NUMBERS):
        raise ValueError("Unbounded VMM timing number")
    if (event["outcome"] != "completed" or event["layer"] not in LAYERS
            or not 1 <= event["rows"] <= 512 or event["vmm_profile"] != 1
            or not 1 <= event["vmm_capacity"] <= 8192 or not event["vmm_physical_bytes"]
            or event["vmm_pending_bytes"] or event["vmm_quarantine_bytes"]
            or event["vmm_certificate"] != 1):
        raise ValueError("Selected call did not complete with a certified usable VMM owner")


def _reshape(event):
    if set(event) != set(RESHAPE_NUMBERS) | {"outcome", "scope"}:
        raise ValueError("Missing or unknown VMM reshape fields")
    if any(type(event[key]) is not int or not 0 <= event[key] <= U64_MAX for key in RESHAPE_NUMBERS):
        raise ValueError("Unbounded VMM reshape number")
    if (event["profile"] != 1 or not event["phase"] or event["outcome"] != "completed"
            or event["scope"] != SCOPE or event["certificate"] != 1
            or event["pending_bytes"] or event["quarantine_bytes"]):
        raise ValueError("VMM reshape lacks a completed clean ownership certificate")
    for group in (3, 4, 11):
        for when in ("before", "after"):
            if not 1 <= event[f"g{group}_cap_{when}"] <= 8192 or not event[f"g{group}_bytes_{when}"]:
                raise ValueError("Invalid VMM group ownership extent")
    deltas = []
    growing = False
    shrink_slots = retained_limit = 0
    for group in (3, 4, 11):
        before, after = (event[f"g{group}_cap_{when}"] for when in ("before", "after"))
        delta = event[f"g{group}_bytes_after"] - event[f"g{group}_bytes_before"]
        if ((after == before and delta != 0) or (after > before and delta < 0)
                or (after < before and delta > 0)):
            raise ValueError("VMM capacity and physical ownership directions disagree")
        deltas.append(delta)
        growing |= after > before
        shrink_slots += max(0, before - after)
        retained_limit += min(before, after)
    refund = sum(-delta for delta in deltas if delta < 0)
    growth = sum(delta for delta in deltas if delta > 0)
    if (growth <= 0 or not growing
            or refund != event["refunded_bytes"] or growth != event["grown_bytes"]
            or event["owned_after"] != event["owned_before"] - refund + growth
            or not 0 < event["owned_after"] <= event["budget"]
            or event["owned_before"] < sum(event[f"g{g}_bytes_before"] for g in (3, 4, 11))
            or event["owned_after"] < sum(event[f"g{g}_bytes_after"] for g in (3, 4, 11))):
        raise ValueError("VMM reshape physical deltas and retained scratch ownership do not reconcile")
    capacity_before = sum(event[f"g{g}_cap_before"] for g in (3, 4, 11))
    capacity_after = sum(event[f"g{g}_cap_after"] for g in (3, 4, 11))
    if (event["retained_hot"] > min(capacity_after, retained_limit)
            or event["moved_slots"] > min(event["retained_hot"], shrink_slots)
            or event["discarded_hot"] > shrink_slots
            or event["retained_hot"] + event["discarded_hot"] > capacity_before):
        raise ValueError("VMM reshape hot-slot counts exceed the real table extents")


def _pressure(event):
    if set(event) != set(PRESSURE_NUMBERS) | {"outcome", "scope"}:
        raise ValueError("Missing or unknown VMM pressure fields")
    if any(type(event[key]) is not int or not 0 <= event[key] <= U64_MAX for key in PRESSURE_NUMBERS):
        raise ValueError("Unbounded VMM pressure number")
    if (event["profile"] != 1 or not event["mutation"] or event["outcome"] != "completed"
            or event["scope"] != PRESSURE_SCOPE or event["certificate"] != 1
            or event["pending_bytes"] or event["quarantine_bytes"]
            or event["active_owner_excluded"] not in (0, 1)):
        raise ValueError("VMM pressure lacks a completed clean ownership certificate")
    before_slots = after_slots = 0
    for group in (3, 4, 11):
        before, after = (event[f"g{group}_cap_{when}"] for when in ("before", "after"))
        if (before, after) != (0, 0) and not 1 <= after <= before <= 8192:
            raise ValueError("Invalid VMM pressure table capacity pair")
        before_slots += before
        after_slots += after
    refund = event["wide_refunded_bytes"] + event["weight_refunded_bytes"]
    shrink_slots = before_slots - after_slots
    if (not after_slots or not 0 < event["owned_after"] <= event["target"] < event["owned_before"]
            or not refund or event["owned_before"] - event["owned_after"] != refund
            or (event["weight_refunded_bytes"] and not shrink_slots)):
        raise ValueError("VMM pressure did not reconcile a real shrink inside the current target")
    if (event["retained_hot"] > after_slots or event["discarded_hot"] > shrink_slots
            or event["retained_hot"] + event["discarded_hot"] > before_slots
            or event["moved_slots"] > min(event["retained_hot"], shrink_slots)):
        raise ValueError("VMM pressure hot-slot counts exceed the real table extents")


def records(log):
    calls, reshapes, pressures, malformed = [], [], [], 0
    for line in log.splitlines():
        if TIMING_TAG not in line and "[HOSTMOE-VMM-" not in line:
            continue
        try:
            if "[HOSTMOE-VMM-" in line:
                tags = [tag for tag in (RESHAPE_TAG, PRESSURE_TAG) if tag in line]
                if len(tags) != 1 or line.count("[HOSTMOE-VMM-") != 1 or TIMING_TAG in line:
                    raise ValueError("Unknown or ambiguous VMM activity marker")
                tag = tags[0]
                event = _pairs(line, tag)
                for key in (RESHAPE_NUMBERS if tag == RESHAPE_TAG else PRESSURE_NUMBERS):
                    event[key] = _uint(event.get(key))
                if tag == RESHAPE_TAG:
                    _reshape(event)
                    reshapes.append(event)
                else:
                    _pressure(event)
                    pressures.append(event)
            else:
                fields = _pairs(line, TIMING_TAG)
                event = {key: _uint(fields.get(key)) for key in TIMING_NUMBERS}
                event["outcome"] = fields.get("outcome")
                if any(key.startswith("vmm_") and key not in TIMING_NUMBERS for key in fields):
                    raise ValueError("Unknown VMM ownership field")
                _timing(event)
                calls.append(event)
        except (ValueError, TypeError):
            malformed += 1
    return dict(calls=calls, reshapes=reshapes, pressures=pressures, malformed_records=malformed)


def qualification(rows, require_pressure=False):
    requests, phases, mutations = [], set(), set()
    previous_phase = previous_mutation = 0
    for row in rows:
        result = dict(name=row.get("name"), passed=False, reshape_events=0, pressure_events=0, newly_granted_bytes=0)
        try:
            if (row.get("error") or type(row.get("eval_tokens")) is not int or row["eval_tokens"] <= 0
                    or type(row.get("cache_hits")) is not int or row["cache_hits"] != 0
                    or type(row.get("prompt_tokens")) is not int or row["prompt_tokens"] <= 1):
                raise ValueError("VMM requires a successful uncached prefill and decode request")
            activity = row["vmm_weights"]
            if type(activity["malformed_records"]) is not int or activity["malformed_records"]:
                raise ValueError("Invalid VMM activity in the measured request")
            pressures = activity.get("pressures", [])
            if (not isinstance(activity["calls"], list) or not isinstance(activity["reshapes"], list)
                    or not isinstance(pressures, list)):
                raise ValueError("VMM activity is not an event list")
            decode, prefill = collections.Counter(), collections.Counter()
            for event in activity["calls"]:
                _timing(event)
                if event["rows"] == 1:
                    decode[event["layer"]] += 1
                else:
                    prefill[event["layer"]] += event["rows"]
            if (set(decode) != LAYERS or set(prefill) != LAYERS
                    or any(count != row["eval_tokens"] for count in decode.values())
                    or any(count != row["prompt_tokens"] for count in prefill.values())):
                raise ValueError("VMM calls do not cover every routed layer and actual prefill/decode token")
            for event in activity["reshapes"]:
                _reshape(event)
                if event["phase"] <= previous_phase:
                    raise ValueError("Reset, duplicate or out-of-order measured VMM reshape epoch")
                phases.add(event["phase"])
                previous_phase = event["phase"]
            for event in pressures:
                _pressure(event)
                if event["mutation"] <= previous_mutation:
                    raise ValueError("Reset, duplicate or out-of-order measured VMM pressure mutation")
                mutations.add(event["mutation"])
                previous_mutation = event["mutation"]
            if require_pressure and not pressures:
                raise ValueError("No actual VMM pressure retirement in this measured request")
            result.update(passed=True, prefill_calls=sum(event["rows"] > 1 for event in activity["calls"]),
                          decode_calls=sum(decode.values()), reshape_events=len(activity["reshapes"]),
                          pressure_events=len(pressures), pressures=pressures,
                          pressure_refunded_bytes=sum(event["wide_refunded_bytes"] + event["weight_refunded_bytes"]
                                                      for event in pressures),
                          refunded_bytes=sum(event["refunded_bytes"] for event in activity["reshapes"]),
                          grown_bytes=sum(event["grown_bytes"] for event in activity["reshapes"]),
                          reshapes=activity["reshapes"])
        except (KeyError, ValueError, TypeError, IndexError) as error:
            result["error"] = str(error)
        requests.append(result)
    return dict(passed=bool(requests) and all(row["passed"] for row in requests) and bool(phases),
                requests=requests, reshape_events=len(phases), pressure_events=len(mutations),
                pressure_required=require_pressure, scope=SCOPE, pressure_scope=PRESSURE_SCOPE, physical_free_qualified=False,
                limitations="Actual selected VMM payload use and checked redistribution only; no total VRAM, RAM, SSD, quality or speed claim")
