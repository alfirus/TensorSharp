#!/usr/bin/env python3
"""Exact per-layer rolling-frequency GPU replacement, logical counterfactual.

Every current key is protected, every miss admitted. Primary window32 is fixed;
optional16/64 diagnostics never select a policy. CLI opens only pinned captures
and saved descriptors, never a checkpoint. No native/performance qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import glm_gpu_adaptive_replay as adaptive

cohort = adaptive.cohort
MAX_TRACE_BYTES = 64 << 20
MAX_SOURCES = 256
MAX_KEYS = 65536
PRIMARY_WINDOW = 32
WINDOWS = (16, 32, 64)
DEPENDENCIES = dict(adaptive.DEPENDENCIES,
    **{"glm_gpu_adaptive_replay.py": "FC0B2E05EE0E28D6E19B8B5D5B43B2C5B7BEB9A551F6AE0650BEF9937F72358C"})
METADATA_SHA256 = "5cbb184adce69d042c6db180e65a1bff3a71419cdb67e5f77e06c92e38a4a8c7"
PROFILES = {
    "v27": dict(trace_sha256=cohort.V27_TRACE_SHA256, windows=[dict(name="historical_v27",
        role="historical_fixed_policy_check", start_position=33, count=128, top_k=8,
        begin_line=4075, end_line=15225, capacities={3: 299, 4: 750, 11: 87},
        expected_misses=20018, expected_bytes=139102257152)]),
    "v44": dict(trace_sha256="80fd4509f20ea626dd94e260920efab61c951e3d7697f303256f0a5f77040d36",
        windows=[dict(name="decode_a", role="first_topic_no_policy_selection", start_position=33,
            count=384, top_k=8, begin_line=637, end_line=33675,
            capacities={3: 291, 4: 729, 11: 85}, expected_misses=58824,
            expected_bytes=408902172672),
            dict(name="decode_b", role="second_topic_fixed_policy_holdout", start_position=33,
            count=384, top_k=8, begin_line=33786, end_line=66824,
            capacities={3: 283, 4: 708, 11: 83}, expected_misses=67147,
            expected_bytes=466874728448)])}


class RollingFrequency:
    """Bounded exact source/layer rings; once-per-key-per-row observations.

    The window includes CURRENT row: expire oldest, observe all current keys,
    protect all original resident hits, admit misses, touch in original order.
    Other source layers do not age. No epoch halving, saturation or bypass.
    """
    def __init__(self, capacity, source_limits, window=PRIMARY_WINDOW):
        self.capacity = adaptive.bounded(capacity, "capacity", 1, adaptive.MAX_CAPACITY)
        if type(window) is not int or window not in WINDOWS:
            raise ValueError("Only fixed16/32/64 row windows are supported")
        if type(source_limits) is not dict or not 1 <= len(source_limits) <= MAX_SOURCES:
            raise ValueError("A bounded immutable source registry is required")
        self.limits = {}
        for identity, experts in source_limits.items():
            if not isinstance(identity, tuple) or len(identity) != 2:
                raise ValueError("Exact source/layer registry identity required")
            source, layer = identity
            adaptive.demand_keys(source, layer, [0], capacity)
            self.limits[identity] = adaptive.bounded(experts, "source experts", 1, 65536)
        if sum(self.limits.values()) > MAX_KEYS:
            raise ValueError("Source-key registry exceeds metadata bound")
        self.window = window
        self.rings = {key: [None] * window for key in self.limits}
        self.cursors = dict.fromkeys(self.limits, 0)
        self.counts = {key: Counter() for key in self.limits}
        self.observations = dict.fromkeys(self.limits, 0)
        self.slots = [None] * capacity
        self.ages = [0] * capacity
        self.clock = 0
        self.stats = dict(misses=0, evictions=0, resident_hits=0, rows=0,
            distinct_observations=0, expired_rows=0, peak_resident=0,
            peak_active_counter_keys=0, peak_ring_key_references=0, peak_score=0)
        self.active_counter_keys = self.ring_key_references = 0

    @property
    def resident(self):
        return set(self.slots) - {None}

    def score(self, key):
        source, layer, expert = key
        return self.counts[(source, layer)].get(expert, 0)

    def snapshot(self):
        histories = tuple((key, self.cursors[key], self.observations[key],
            tuple(self.rings[key]), tuple(sorted(self.counts[key].items())))
            for key in sorted(self.limits))
        return tuple(self.slots), tuple(self.ages), self.clock, histories

    def _observe(self, identity, unique):
        ring, counts = self.rings[identity], self.counts[identity]
        cursor = self.cursors[identity]
        old = ring[cursor]
        if old is not None:
            self.stats["expired_rows"] += 1
            self.ring_key_references -= len(old)
            for expert in old:
                counts[expert] -= 1
                if counts[expert] == 0:
                    del counts[expert]
                    self.active_counter_keys -= 1
        for expert in unique:
            if expert not in counts:
                self.active_counter_keys += 1
            counts[expert] += 1
            if not 1 <= counts[expert] <= self.window:
                raise RuntimeError("Exact rolling count escaped its row bound")
            self.stats["peak_score"] = max(self.stats["peak_score"], counts[expert])
        ring[cursor] = unique
        self.cursors[identity] = (cursor + 1) % self.window
        self.observations[identity] += 1  # Python integer; not a wrapping timestamp.
        self.ring_key_references += len(unique)
        self.stats["rows"] += 1
        self.stats["distinct_observations"] += len(unique)
        self.stats["peak_active_counter_keys"] = max(self.stats["peak_active_counter_keys"], self.active_counter_keys)
        self.stats["peak_ring_key_references"] = max(self.stats["peak_ring_key_references"], self.ring_key_references)
        if (self.active_counter_keys > sum(self.limits.values())
                or self.ring_key_references > len(self.limits) * self.window * adaptive.MAX_ROW):
            raise RuntimeError("Rolling history exceeded exact metadata bound")

    def visit(self, source, layer, ids):
        keys = adaptive.demand_keys(source, layer, ids, self.capacity)
        identity = source, layer
        if identity not in self.limits or any(key[2] >= self.limits[identity] for key in keys):
            raise ValueError("Unregistered or out-of-source-bounds demand")
        # Full validation precedes ANY history, slot, clock or statistics change.
        self._observe(identity, tuple(dict.fromkeys(key[2] for key in keys)))
        locations = {key: slot for slot, key in enumerate(self.slots) if key is not None}
        protected = {locations[key] for key in keys if key in locations}
        misses = []
        self.stats["resident_hits"] += sum(key in locations for key in dict.fromkeys(keys))
        for key in keys:
            if key in locations:
                continue
            def priority(slot):
                old = self.slots[slot]
                return (-1 if old is None else self.score(old), self.ages[slot], slot)
            victim = min((slot for slot in range(self.capacity) if slot not in protected), key=priority)
            old = self.slots[victim]
            if old is not None:
                locations.pop(old)
                self.stats["evictions"] += 1
            self.slots[victim] = key
            locations[key] = victim
            protected.add(victim)
            misses.append(key)
        for key in keys:
            self.clock += 1
            self.ages[locations[key]] = self.clock
        self.stats["misses"] += len(misses)
        self.stats["peak_resident"] = max(self.stats["peak_resident"], len(self.resident))
        if len(self.resident) > self.capacity or not set(keys).issubset(self.resident):
            raise RuntimeError("Forced admission/current-row protection violated")
        return misses


def analyse(text, metadata, *, capacities, start_position, count, top_k,
            begin_line=1, end_line=None, diagnostic_windows=False):
    if type(diagnostic_windows) is not bool:
        raise ValueError("Diagnostic selection must be an explicit boolean")
    validated = cohort.reconstruct_misses(text, metadata, capacities=capacities,
        start_position=start_position, count=count, top_k=top_k,
        begin_line=begin_line, end_line=end_line)
    lines = text.splitlines()
    last = len(lines) if end_line is None else end_line
    calls, trailing = cohort.retention.associate_forward_calls(
        "\n".join(lines[begin_line - 1:last]), begin_line - 1)
    groups, lookup, source = cohort.retention.validate_metadata(metadata)
    _, routed = cohort.retention.select_window(calls, lookup, start_position, count, top_k)
    if trailing or len(routed) != len(validated["rows"]):
        raise ValueError("Every completed route/layer association must be exact")
    requested_windows = (PRIMARY_WINDOW, 16, 64) if diagnostic_windows else (PRIMARY_WINDOW,)
    lru = {group: adaptive.ProtectedLru(size) for group, size in capacities.items()}
    policies = {}
    for window in requested_windows:
        policies[window] = {}
        for group in groups:
            registry = {(source, layer): group["projections"][0]["shape"][2]
                        for layer in group["layers"]}
            policies[window][group["id"]] = RollingFrequency(capacities[group["id"]], registry, window)
    names = ["lru"] + ["rolling_frequency_" + str(window) for window in requested_windows]
    totals = {name: dict(misses=0, logical_source_bytes=0, per_group={}, per_position={}) for name in names}
    for index, (_, route, _) in enumerate(routed):
        group = lookup[route["layer"]]
        group_id, size = group["id"], group["expert_bytes"]
        args = source, route["layer"], route["ids"]
        observed = lru[group_id].visit(*args)
        proof = validated["rows"][index]
        if [key[2] for key in observed] != proof["gpu_miss_experts"]:
            raise ValueError("Independent original LRU disagrees with every-row proof")
        results = [observed] + [policies[window][group_id].visit(*args) for window in requested_windows]
        for name, missed in zip(names, results):
            item = totals[name]
            item["misses"] += len(missed)
            item["logical_source_bytes"] += len(missed) * size
            item["per_group"][group_id] = item["per_group"].get(group_id, 0) + len(missed) * size
            item["per_position"][proof["p0"]] = item["per_position"].get(proof["p0"], 0) + len(missed) * size
    baseline = totals["lru"]
    if (baseline["misses"] != validated["gpu_misses"]
            or baseline["logical_source_bytes"] != validated["logical_source_bytes"]):
        raise ValueError("Independent original LRU baseline failed exact totals")
    reports = {}
    for window in requested_windows:
        name = "rolling_frequency_" + str(window)
        reports[name] = dict(role="predeclared_primary" if window == PRIMARY_WINDOW else "diagnostic_only_no_policy_selection",
            window_rows_per_layer=window, stats={group: cache.stats for group, cache in policies[window].items()},
            layer_rows={str(layer): cache.observations[(source, layer)]
                for cache in policies[window].values() for _, layer in cache.limits},
            net_logical_bytes_saved=baseline["logical_source_bytes"] - totals[name]["logical_source_bytes"])
    return dict(schema_version=1, scope="fixed_cold_GPU_forced_admission_per_layer_rolling_frequency_logical_counterfactual",
        baseline=validated, policies=totals, rolling=reports, primary_window=PRIMARY_WINDOW,
        primary_policy="rolling_frequency_32", primary_uses_future=False,
        history_includes_current_row=True, duplicates_count_once_per_row=True,
        cross_topic_history_carried=False, windows_selected_from_results=False,
        independent_lru_matched_every_row=True, every_layer_timer_matched=True,
        native_implementation_qualified=False, metadata_physical_admission_qualified=False,
        physical_io_qualified=False, performance_qualified=False, numerical_quality_qualified=False,
        limitations=["Only exact saved cold windows; no warm/PP/capacity-transition generalization.",
            "Fixed32 was declared before either topic;16/64 diagnostic results never select a runtime policy.",
            "Each cold topic resets payload and frequency history; second-topic holdout validates the same policy, not trained resident state.",
            "History observes every distinct current selected key including misses, expires the oldest local row exactly, and never halves a global epoch.",
            "Existing source-qualified decaying-LFU replay ALSO counted misses; this exact per-layer expiry is the distinct policy.",
            "Descriptor namespace/layer/expert keys substitute for native generation IDs absent in the trace.",
            "Fixed payload capacities only; bounded Python metadata is uncharged, no native admission/OS cache/SSD/time/math simulation."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-windows", action="store_true")
    args = parser.parse_args(argv)
    for name, expected in DEPENDENCIES.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest().upper() != expected:
            raise ValueError("Frozen proof dependency changed: " + name)
    profile = PROFILES[args.profile]
    raw = cohort._read_bounded(args.trace, MAX_TRACE_BYTES, profile["trace_sha256"])
    saved = cohort._read_bounded(args.metadata, cohort.MAX_METADATA_BYTES, METADATA_SHA256)
    text, metadata = raw.decode("utf-8-sig", errors="strict"), json.loads(saved.decode("utf-8-sig", errors="strict"))
    results = []
    for spec in profile["windows"]:
        options = {key: value for key, value in spec.items()
                   if key not in ("name", "role", "expected_misses", "expected_bytes")}
        result = analyse(text, metadata, **options, diagnostic_windows=args.diagnostic_windows)
        if (result["baseline"]["gpu_misses"] != spec["expected_misses"]
                or result["baseline"]["logical_source_bytes"] != spec["expected_bytes"]):
            raise ValueError("Pinned completed baseline totals changed")
        result["name"], result["topic_role"] = spec["name"], spec["role"]
        results.append(result)
    output = dict(schema_version=1, profile=args.profile, primary_window=PRIMARY_WINDOW,
        policy_selected_from_results=False, chronological_windows=results,
        provenance=dict(trace=str(args.trace.resolve()), trace_sha256=profile["trace_sha256"],
            metadata=str(args.metadata.resolve()), metadata_sha256=METADATA_SHA256,
            tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), dependencies=DEPENDENCIES,
            checkpoint_opened=False), native_implementation_qualified=False,
        physical_io_qualified=False, performance_qualified=False, numerical_quality_qualified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(profile=args.profile, primary_window=PRIMARY_WINDOW,
        windows=[dict(name=result["name"], baseline=result["baseline"]["logical_source_bytes"],
            net_saved={name: value["net_logical_bytes_saved"] for name, value in result["rolling"].items()})
            for result in results], performance_qualified=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
