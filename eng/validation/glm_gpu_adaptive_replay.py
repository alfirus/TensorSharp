#!/usr/bin/env python3
"""Past-only protected-row ARC-style replacement and diagnostic Belady bound.

The CLI accepts only the saved cold v27 D128 capture, never a checkpoint. No
native policy, payload, admission, physical-I/O, numerical or speed claim.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, deque
import hashlib
import heapq
import json
from pathlib import Path

import glm_ram_miss_cohort_replay as cohort

MAX_CAPACITY = 8192
MAX_ROW = 64
MAX_DEMANDS = 1 << 20
DEPENDENCIES = {
    "glm_ram_miss_cohort_replay.py": "82DD30964A22208CEA812A2DEB227C29F73917C948C4D3CE0129B9DDC4F20492",
    "glm_route_retention_replay.py": "85567D74A3F8821F5DFCC008C9E4ADEBC55E9F7F34AC61D0FB1CB4E2A204D622",
    "glm-route-cache-replay.py": "568A34E815BAF4868ADD8667C2F881A2239AF2D2E9B6DD2C635C5AA0C8F85539",
    "local-model-preflight.py": "845E65A8BE319A4DBB0C94CE170240481AF9809B85E7DD90D2892B9ED684CC56",
}
ARC_SOURCE = "https://www.usenix.org/legacy/event/fast03/tech/full_papers/megiddo/megiddo.pdf"


def bounded(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid bounded " + name)
    return value


def demand_keys(source, layer, ids, capacity):
    """Validate the ENTIRE row before changing any policy state."""
    if type(source) is not str or not 1 <= len(source) <= 128:
        raise ValueError("An immutable source-generation namespace is required")
    bounded(layer, "layer", 0, 65535)
    if not isinstance(ids, (tuple, list)) or not 1 <= len(ids) <= MAX_ROW:
        raise ValueError("A bounded nonempty selected row is required")
    keys = tuple((source, layer, bounded(expert, "expert", 0, 65535)) for expert in ids)
    if len(set(keys)) > capacity:
        raise ValueError("Entire current selected union must fit without bypass")
    return keys


class ProtectedLru:
    """Independent original slot/age implementation, including ascending tie."""
    def __init__(self, capacity):
        self.capacity = bounded(capacity, "capacity", 1, MAX_CAPACITY)
        self.slots = [None] * capacity
        self.ages = [0] * capacity
        self.clock = 0

    @property
    def resident(self):
        return set(self.slots) - {None}

    def snapshot(self):
        return tuple(self.slots), tuple(self.ages), self.clock

    def _touch_order(self, keys):
        return keys

    def visit(self, source, layer, ids):
        keys = demand_keys(source, layer, ids, self.capacity)
        locations = {}
        for slot, key in enumerate(self.slots):
            if key is not None:
                locations.setdefault(key, slot)
        protected = {locations[key] for key in keys if key in locations}
        misses = []
        for key in keys:
            if key in locations:
                continue
            victim = min((i for i in range(self.capacity) if i not in protected),
                         key=lambda i: (self.ages[i], i))
            old = self.slots[victim]
            if old is not None:
                locations.pop(old)
            self.slots[victim] = key
            locations[key] = victim
            protected.add(victim)
            misses.append(key)
        for key in self._touch_order(keys):
            self.clock += 1
            self.ages[locations[key]] = self.clock
        return misses


class ReverseTouchLru(ProtectedLru):
    """Only final age updates reverse; hit protection/admission order is unchanged.

    The saved positions are router SELECTION-score ranks, not unbiased expert
    probabilities: GLM may add routing bias before its descending top-k.
    """
    def _touch_order(self, keys):
        return reversed(keys)


class ProtectedArc:
    """ARC-style integer target, four exact bounded lists, past/current only.

    Relative to scalar ARC: eligible victims exclude the entire current row;
    a fully protected preferred list falls back to the other resident list.
    Integer ghost ratios round DOWN before max(1,...). One unique key causes
    one frequency observation per row; final recency follows original order.
    """
    def __init__(self, capacity):
        self.capacity = bounded(capacity, "capacity", 1, MAX_CAPACITY)
        self.t1, self.t2, self.b1, self.b2 = (OrderedDict() for _ in range(4))
        self.target = 0
        self.stats = dict(b1_hits=0, b2_hits=0, resident_hits=0, misses=0,
                          evictions=0, replacement_list_fallbacks=0,
                          victim_key_checks=0, peak_resident=0, peak_ghosts=0,
                          peak_directory=0, peak_target=0)

    @property
    def resident(self):
        return set(self.t1) | set(self.t2)

    def snapshot(self):
        return self.target, tuple(self.t1), tuple(self.t2), tuple(self.b1), tuple(self.b2)

    def _check(self):
        lists = [set(items) for items in (self.t1, self.t2, self.b1, self.b2)]
        if sum(map(len, lists)) != len(set().union(*lists)):
            raise RuntimeError("ARC resident/ghost identity overlap")
        resident = len(self.t1) + len(self.t2)
        ghosts = len(self.b1) + len(self.b2)
        if not (0 <= self.target <= self.capacity and resident <= self.capacity
                and ghosts <= self.capacity and resident + ghosts <= 2 * self.capacity
                and len(self.t1) + len(self.b1) <= self.capacity):
            raise RuntimeError("ARC exceeded its exact metadata/payload bounds")
        for name, value in (("resident", resident), ("ghosts", ghosts),
                            ("directory", resident + ghosts), ("target", self.target)):
            self.stats["peak_" + name] = max(self.stats["peak_" + name], value)

    def _eligible(self, items, protected):
        for key in items:
            self.stats["victim_key_checks"] += 1
            if key not in protected:
                return key
        return None

    def _replace(self, protected, incoming_b2):
        if len(self.t1) + len(self.t2) < self.capacity:
            return
        recent = bool(self.t1) and (len(self.t1) > self.target
                                    or (incoming_b2 and len(self.t1) == self.target))
        choices = ((self.t1, self.b1), (self.t2, self.b2))
        if not recent:
            choices = tuple(reversed(choices))
        for index, (resident, ghosts) in enumerate(choices):
            victim = self._eligible(resident, protected)
            if victim is not None:
                resident.pop(victim)
                ghosts[victim] = None
                self.stats["evictions"] += 1
                self.stats["replacement_list_fallbacks"] += index
                return
        raise RuntimeError("Validated selected union unexpectedly has no victim")

    def visit(self, source, layer, ids):
        keys = demand_keys(source, layer, ids, self.capacity)
        protected = set(keys)  # All original resident hits BEFORE any miss.
        misses = []
        for key in dict.fromkeys(keys):
            if key in self.t1 or key in self.t2:
                self.t1.pop(key, None)
                self.t2.pop(key, None)
                self.t2[key] = None
                self.stats["resident_hits"] += 1
            elif key in self.b1:
                # Use PRE-removal list lengths. Removing the incoming ghost
                # before replacement keeps even transient ghosts bounded by C.
                self.target = min(self.capacity, self.target + max(1, len(self.b2) // len(self.b1)))
                self.b1.pop(key)
                self._replace(protected, False)
                self.t2[key] = None
                self.stats["b1_hits"] += 1
                misses.append(key)
            elif key in self.b2:
                self.target = max(0, self.target - max(1, len(self.b1) // len(self.b2)))
                self.b2.pop(key)
                self._replace(protected, True)
                self.t2[key] = None
                self.stats["b2_hits"] += 1
                misses.append(key)
            else:
                if len(self.t1) + len(self.b1) == self.capacity:
                    if len(self.t1) < self.capacity:
                        self.b1.popitem(last=False)
                        self._replace(protected, False)
                    else:
                        victim = self._eligible(self.t1, protected)
                        if victim is None:
                            raise RuntimeError("Selected-union preflight failed")
                        self.t1.pop(victim)  # Pure scan: discard without ghost.
                        self.stats["evictions"] += 1
                elif len(self.t1) + len(self.b1) < self.capacity:
                    total = sum(len(x) for x in (self.t1, self.t2, self.b1, self.b2))
                    if total >= self.capacity:
                        if total == 2 * self.capacity:
                            self.b2.popitem(last=False)
                        self._replace(protected, False)
                self.t1[key] = None
                misses.append(key)
            self._check()
        # A duplicate just admitted this row remains T1, rather than fabricating
        # a resident-frequency hit. Original occurrence order still defines MRU.
        for key in keys:
            (self.t1 if key in self.t1 else self.t2).move_to_end(key)
        self.stats["misses"] += len(misses)
        self._check()
        if not protected.issubset(self.resident):
            raise RuntimeError("ARC failed forced admission/current-row protection")
        return misses


class FutureBelady:
    """FUTURE-aware diagnostic bound only, never an online/native candidate."""
    def __init__(self, capacity, rows):
        self.capacity = bounded(capacity, "capacity", 1, MAX_CAPACITY)
        if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= MAX_DEMANDS:
            raise ValueError("Bounded finite diagnostic future required")
        self.rows, self.future = [], {}
        demands = 0
        for index, (source, layer, ids) in enumerate(rows):
            keys = demand_keys(source, layer, ids, capacity)
            self.rows.append(keys)
            for key in dict.fromkeys(keys):
                self.future.setdefault(key, deque()).append(index)
            demands += len(keys)
            if demands > MAX_DEMANDS:
                raise ValueError("Diagnostic future exceeds demand bound")
        self.index, self.entries, self.peak = 0, set(), 0
        self.heap, self.versions = [], {}

    @property
    def resident(self):
        return set(self.entries)

    def _queue(self, key):
        version = self.versions.get(key, 0) + 1
        self.versions[key] = version
        future = self.future[key]
        next_use = future[0] if future else len(self.rows) + 1
        heapq.heappush(self.heap, (-next_use, key, version))

    def _victim(self, protected):
        # The maximum next-use heap avoids scanning capacity for every miss.
        # Current-row keys are temporarily skipped, then restored unchanged.
        skipped = []
        victim = None
        while self.heap:
            item = heapq.heappop(self.heap)
            _, key, version = item
            if key not in self.entries or self.versions.get(key) != version:
                continue
            if key in protected:
                skipped.append(item)
                continue
            victim = key
            break
        for item in skipped:
            heapq.heappush(self.heap, item)
        if victim is None:
            raise RuntimeError("Validated diagnostic row has no eligible victim")
        return victim

    def _bound_heap(self):
        if len(self.heap) > 4 * self.capacity:
            self.heap.clear()
            for key in sorted(self.entries):
                self._queue(key)

    def visit(self, source, layer, ids):
        keys = demand_keys(source, layer, ids, self.capacity)
        if self.index >= len(self.rows) or keys != self.rows[self.index]:
            raise ValueError("Diagnostic future/order differs from actual demands")
        protected, misses = set(keys), []
        for key in dict.fromkeys(keys):
            if self.future[key].popleft() != self.index:
                raise RuntimeError("Broken diagnostic next-use index")
            if key in self.entries:
                self._queue(key)
        for key in dict.fromkeys(keys):
            if key not in self.entries:
                if len(self.entries) == self.capacity:
                    self.entries.remove(self._victim(protected))
                self.entries.add(key)
                self._queue(key)
                misses.append(key)
        self.index += 1
        self._bound_heap()
        self.peak = max(self.peak, len(self.entries))
        if not protected.issubset(self.entries) or len(self.entries) > self.capacity:
            raise RuntimeError("Diagnostic violated forced admission/current union")
        return misses


def analyse(text, metadata, *, capacities, start_position, count, top_k,
            begin_line=1, end_line=None):
    validated = cohort.reconstruct_misses(text, metadata, capacities=capacities,
        start_position=start_position, count=count, top_k=top_k,
        begin_line=begin_line, end_line=end_line)
    lines = text.splitlines()
    last = len(lines) if end_line is None else end_line
    calls, trailing = cohort.retention.associate_forward_calls(
        "\n".join(lines[begin_line - 1:last]), begin_line - 1)
    groups, lookup, source = cohort.retention.validate_metadata(metadata)
    _, routed = cohort.retention.select_window(calls, lookup, start_position, count, top_k)
    if trailing or len(routed) != len(validated["rows"]) or len(routed) * top_k > MAX_DEMANDS:
        raise ValueError("Bounded exact route/layer association required")
    demands = [dict(source=source, layer=route["layer"], ids=route["ids"],
        group=lookup[route["layer"]]["id"], expert_bytes=lookup[route["layer"]]["expert_bytes"],
        p0=validated["rows"][index]["p0"]) for index, (_, route, _) in enumerate(routed)]
    lru = {group: ProtectedLru(size) for group, size in capacities.items()}
    reverse = {group: ReverseTouchLru(size) for group, size in capacities.items()}
    arc = {group: ProtectedArc(size) for group, size in capacities.items()}
    future = {group: FutureBelady(size, [(d["source"], d["layer"], d["ids"])
        for d in demands if d["group"] == group]) for group, size in capacities.items()}
    totals = {name: dict(misses=0, logical_source_bytes=0, per_group={}, per_position={})
              for name in ("lru", "reverse_touch_lru", "arc_past_only", "belady_future_diagnostic")}
    for index, demand in enumerate(demands):
        group, size = demand["group"], demand["expert_bytes"]
        args = demand["source"], demand["layer"], demand["ids"]
        results = (lru[group].visit(*args), reverse[group].visit(*args),
                   arc[group].visit(*args), future[group].visit(*args))
        if [key[2] for key in results[0]] != validated["rows"][index]["gpu_miss_experts"]:
            raise ValueError("Independent LRU disagrees with observed exact miss order")
        for name, missed in zip(totals, results):
            item = totals[name]
            item["misses"] += len(missed)
            item["logical_source_bytes"] += len(missed) * size
            item["per_group"][group] = item["per_group"].get(group, 0) + len(missed) * size
            item["per_position"][demand["p0"]] = item["per_position"].get(demand["p0"], 0) + len(missed) * size
    if (totals["lru"]["logical_source_bytes"] != validated["logical_source_bytes"]
            or totals["lru"]["misses"] != validated["gpu_misses"]):
        raise ValueError("Independent cold LRU does not reproduce every observed byte")
    for group in capacities:
        lower = totals["belady_future_diagnostic"]["per_group"][group]
        if any(lower > totals[name]["per_group"][group]
               for name in ("lru", "reverse_touch_lru", "arc_past_only")):
            raise RuntimeError("Future diagnostic exceeds an eligible online policy")
    baseline = totals["lru"]["logical_source_bytes"]
    return dict(schema_version=1, scope="fixed_cold_GPU_protected_row_policy_logical_counterfactual",
        baseline=validated, policies=totals, arc_stats={group: cache.stats for group, cache in arc.items()},
        arc_final_target={group: cache.target for group, cache in arc.items()},
        arc_net_logical_bytes_saved=baseline - totals["arc_past_only"]["logical_source_bytes"],
        reverse_touch_net_logical_bytes_saved=baseline - totals["reverse_touch_lru"]["logical_source_bytes"],
        future_diagnostic_headroom_bytes=baseline - totals["belady_future_diagnostic"]["logical_source_bytes"],
        independent_lru_matched_every_row=True, every_layer_timer_matched=True,
        arc_uses_future=False, reverse_touch_uses_future=False, belady_uses_future=True,
        native_implementation_qualified=False,
        metadata_physical_admission_qualified=False, physical_io_qualified=False,
        performance_qualified=False, numerical_quality_qualified=False,
        limitations=["Only the saved coldD128 window; no PP pollution, warm seed or resize generalization.",
            "Keys namespace one saved immutable model descriptor, layer and expert; native generation IDs were not captured.",
            "Same raw triple payload capacities; ARC ghost metadata is bounded but uncharged in this offline model.",
            "ARC integer rounding and protected-row fallback are explicit adaptations of scalar ARC.",
            "Reverse-touch changes only final age updates. Captured ranks follow selection score, possibly biased; no probability magnitudes were recorded.",
            "Belady sees the future and is diagnostic only; it cannot select a deployable policy.",
            "No payload, H2D, CUDA kernel, OS cache, physical admission, SSD, time or numerical simulation."],
        primary_algorithm_source=ARC_SOURCE)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name, expected in DEPENDENCIES.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest().upper() != expected:
            raise ValueError("Frozen proof dependency changed: " + name)
    raw = cohort._read_bounded(args.trace, cohort.MAX_TRACE_BYTES, cohort.V27_TRACE_SHA256)
    saved = cohort._read_bounded(args.metadata, cohort.MAX_METADATA_BYTES, cohort.V27_METADATA_SHA256)
    options = {key: value for key, value in cohort.V27_WINDOW.items() if key != "train_tokens"}
    result = analyse(raw.decode("utf-8-sig", errors="strict"),
        json.loads(saved.decode("utf-8-sig", errors="strict")), **options)
    if any(result["baseline"][key] != value for key, value in cohort.V27_BASELINE.items()):
        raise ValueError("Pinned observed cold LRU totals changed")
    result["provenance"] = dict(trace=str(args.trace.resolve()), trace_sha256=cohort.V27_TRACE_SHA256,
        metadata=str(args.metadata.resolve()), metadata_sha256=cohort.V27_METADATA_SHA256,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), dependencies=DEPENDENCIES,
        checkpoint_opened=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("arc_net_logical_bytes_saved", "reverse_touch_net_logical_bytes_saved",
        "future_diagnostic_headroom_bytes", "independent_lru_matched_every_row",
        "performance_qualified")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
