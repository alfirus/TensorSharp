#!/usr/bin/env python3
"""Past-only next-own-layer transition replacement, logical cold GPU pools.

One fixed64-transition policy compares against exact original LRU and the
unchanged route-neighbor policy. No native, payload, physical-I/O or speed proof.
"""
from __future__ import annotations

import argparse
from array import array
from collections import deque
import hashlib
import json
import math
from pathlib import Path

import glm_gpu_causal_neighbor_replay as causal

adaptive = causal.adaptive
cohort = causal.cohort
PRIMARY_WINDOW = 64
MAX_LAYERS = 42
MAX_EXPERTS = 288
MAX_ROUTES = 8
POLICY = "conditional_transition64"
DEPENDENCIES = dict(causal.DEPENDENCIES, **{
    "glm_gpu_causal_neighbor_replay.py": "0942124E59707431AC7AED9951AB9841E98E0EBDD6C1DE173C6C85ED9BABBCC3"})
PROFILES = causal.PROFILES
METADATA_SHA256 = causal.METADATA_SHA256


def _identity(value):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("An exact immutable source/layer identity is required")
    adaptive.demand_keys(*value, [0], 1)
    return value


class TransitionCache:
    """Forced admission with exact conditional next-own-layer scores.

    C[a,b] counts completed adjacent rows containing a then b; D[a] counts
    outgoing source ROWS containing a, once per row, not once per destination.
    A query uses only transitions completed BEFORE that query. Its score for b
    is the mean C[a,b]/D[a] over known a in the distinct current route. An LCM
    once per query gives exact rational scores without per-cell Fraction work.
    The latest own-layer map expires at that layer's next absolute call rank.
    All zero/unknown scores tie by the original LRU age and ascending slot.

    The reusable class permits tiny windows1..64 for independent fixtures;
    qualified CLI profiles always use the one primary64, with no sweep.
    """
    def __init__(self, capacity, source_limits, layer_order, window=PRIMARY_WINDOW):
        size = adaptive.bounded(capacity, "capacity", 1, adaptive.MAX_CAPACITY)
        width = adaptive.bounded(window, "transition window", 1, PRIMARY_WINDOW)
        if type(source_limits) is not dict or not 1 <= len(source_limits) <= MAX_LAYERS:
            raise ValueError("A bounded source registry is required")
        limits = {_identity(key): adaptive.bounded(value, "source experts", 1, MAX_EXPERTS)
                  for key, value in source_limits.items()}
        if not isinstance(layer_order, (tuple, list)) or not 1 <= len(layer_order) <= MAX_LAYERS:
            raise ValueError("A bounded complete fixed layer cycle is required")
        order = tuple(_identity(key) for key in layer_order)
        if len(set(order)) != len(order) or not set(limits).issubset(order):
            raise ValueError("Complete layer order must contain each enrolled source exactly once")
        # Validate ALL geometry before creating any ring or dense count table.
        if array("H").itemsize != 2:
            raise ValueError("The bounded payload requires two-byte unsigned counters")
        self.capacity, self.window, self.limits, self.layer_order = size, width, limits, order
        self.rank = {key: index for index, key in enumerate(order)}
        self.pool_order = tuple(key for key in order if key in limits)
        self.matrices = {key: array("H", [0]) * (experts * experts) for key, experts in limits.items()}
        self.denominators = {key: array("H", [0]) * experts for key, experts in limits.items()}
        self.rings = {key: deque() for key in limits}
        self.observations = dict.fromkeys(limits, 0)
        self.predictions = dict.fromkeys(limits)
        self.slots, self.ages = [None] * size, [0] * size
        self.clock = self.events = self.ring_references = 0
        names = ("rows misses evictions resident_hit_occurrences planned_duplicate_occurrences "
                 "prediction_queries fallback_queries known_antecedents prediction_cell_checks "
                 "nonzero_prediction_terms pair_increments pair_decrements denominator_increments "
                 "denominator_decrements committed_transitions expired_transitions victim_checks "
                 "cross_layer_victim_checks zero_score_victim_checks expired_prediction_victim_checks "
                 "peak_cell peak_denominator peak_ring_rows peak_ring_key_references peak_resident")
        self.stats = dict.fromkeys(names.split(), 0)

    @property
    def resident(self):
        return set(self.slots) - {None}

    def snapshot(self):
        histories = []
        for key in sorted(self.limits):
            saved = self.predictions[key]
            prediction = None if saved is None else (saved["due"], saved["scores"], saved["denominator"])
            histories.append((key, self.observations[key], tuple(self.rings[key]),
                              tuple(self.matrices[key]), tuple(self.denominators[key]), prediction))
        return (tuple(self.slots), tuple(self.ages), self.clock, self.events,
                tuple(histories), self.ring_references, tuple(sorted(self.stats.items())))

    def score(self, key, current_global_ordinal):
        if (not isinstance(key, tuple) or len(key) != 3 or key[:2] not in self.limits
                or type(key[2]) is not int or not 0 <= key[2] < self.limits[key[:2]]
                or type(current_global_ordinal) is not int or current_global_ordinal < 0):
            raise ValueError("A registered source key and current nonnegative rank are required")
        saved = self.predictions[key[:2]]
        if saved is None or saved["due"] <= current_global_ordinal:
            return 0, 1
        return saved["scores"][key[2]], saved["denominator"]

    def _predict(self, identity, current, cycle):
        experts, table, totals = self.limits[identity], self.matrices[identity], self.denominators[identity]
        known = tuple(a for a in current if totals[a])
        scores = [0] * experts
        denominator = 1
        self.stats["prediction_queries"] += 1
        self.stats["known_antecedents"] += len(known)
        if known:
            common = math.lcm(*(totals[a] for a in known))
            denominator = common * len(known)
            for a in known:
                multiplier = common // totals[a]
                base = a * experts
                self.stats["prediction_cell_checks"] += experts
                for b in range(experts):
                    value = table[base + b]
                    if value:
                        scores[b] += value * multiplier
                        self.stats["nonzero_prediction_terms"] += 1
            if not all(0 <= value <= denominator for value in scores):
                raise RuntimeError("Conditional membership probability escaped zero..one")
        else:
            self.stats["fallback_queries"] += 1
        self.predictions[identity] = dict(due=len(self.layer_order) * (cycle + 1) + self.rank[identity],
            scores=tuple(scores), denominator=denominator)

    def _victim(self, protected, identity, now):
        best, best_score = None, None
        for slot, old in enumerate(self.slots):
            if slot in protected:
                continue
            if old is None:
                return slot
            self.stats["victim_checks"] += 1
            self.stats["cross_layer_victim_checks"] += old[:2] != identity
            value = self.score(old, now)
            self.stats["zero_score_victim_checks"] += value[0] == 0
            saved = self.predictions[old[:2]]
            self.stats["expired_prediction_victim_checks"] += saved is not None and saved["due"] <= now
            if (best is None or value[0] * best_score[1] < best_score[0] * value[1]
                    or (value[0] * best_score[1] == best_score[0] * value[1]
                        and (self.ages[slot], slot) < (self.ages[best], best))):
                best, best_score = slot, value
        if best is None:
            raise RuntimeError("No unprotected slot despite full-row admission validation")
        return best

    def _adjust(self, identity, previous, current, change):
        experts, table, totals = self.limits[identity], self.matrices[identity], self.denominators[identity]
        for a in previous:
            updated = totals[a] + change
            if not 0 <= updated <= self.window:
                raise RuntimeError("Outgoing-row count underflow/overflow")
            totals[a] = updated
            field = "denominator_increments" if change > 0 else "denominator_decrements"
            self.stats[field] += 1
            self.stats["peak_denominator"] = max(self.stats["peak_denominator"], updated)
            for b in current:
                cell = a * experts + b
                value = table[cell] + change
                if not 0 <= value <= self.window or value > totals[a]:
                    raise RuntimeError("Pair count escaped its outgoing-row denominator")
                table[cell] = value
                field = "pair_increments" if change > 0 else "pair_decrements"
                self.stats[field] += 1
                self.stats["peak_cell"] = max(self.stats["peak_cell"], value)

    def _commit(self, identity, current):
        ring = self.rings[identity]
        if len(ring) == self.window + 1:
            self._adjust(identity, ring[0], ring[1], -1)
            self.ring_references -= len(ring.popleft())
            self.stats["expired_transitions"] += 1
        if ring:
            self._adjust(identity, ring[-1], current, 1)
            self.stats["committed_transitions"] += 1
        ring.append(current)
        self.ring_references += len(current)
        self.observations[identity] += 1
        self.stats["peak_ring_rows"] = max(self.stats["peak_ring_rows"], len(ring))
        self.stats["peak_ring_key_references"] = max(self.stats["peak_ring_key_references"], self.ring_references)

    def visit(self, source, layer, ids):
        if not isinstance(ids, (tuple, list)) or not 1 <= len(ids) <= MAX_ROUTES:
            raise ValueError("One complete bounded top-eight route is required")
        keys = adaptive.demand_keys(source, layer, ids, self.capacity)
        identity = source, layer
        if identity not in self.limits or any(key[2] >= self.limits[identity] for key in keys):
            raise ValueError("Unregistered or out-of-source-bounds demand")
        expected = self.pool_order[self.events % len(self.pool_order)]
        cycle = self.events // len(self.pool_order)
        if identity != expected or self.observations[identity] != cycle:
            raise ValueError("Demand violates the complete fixed own-pool layer-cycle chronology")
        # No table/map/slot/stat mutation occurs before all row/schedule checks.
        now = len(self.layer_order) * cycle + self.rank[identity]
        current = tuple(sorted({key[2] for key in keys}))
        self._predict(identity, current, cycle)
        locations = {}
        for slot, key in enumerate(self.slots):
            if key is not None:
                locations.setdefault(key, slot)
        original = set(locations)
        protected = {locations[key] for key in keys if key in original}
        self.stats["resident_hit_occurrences"] += sum(key in original for key in keys)
        misses = []
        for key in keys:
            if key in locations:
                self.stats["planned_duplicate_occurrences"] += key not in original
                continue
            slot = self._victim(protected, identity, now)
            old = self.slots[slot]
            if old is not None:
                if locations.get(old) == slot:
                    locations.pop(old)
                self.stats["evictions"] += 1
            self.slots[slot], locations[key] = key, slot
            protected.add(slot)
            misses.append(key)
        for key in keys:
            self.clock += 1
            self.ages[locations[key]] = self.clock
        # Only a completed visit supplies prev->current labels to later queries.
        self._commit(identity, current)
        self.events += 1
        self.stats["rows"] += 1
        self.stats["misses"] += len(misses)
        self.stats["peak_resident"] = max(self.stats["peak_resident"], len(self.resident))
        if (not set(keys).issubset(self.resident) or len(self.resident) > self.capacity
                or self.ring_references > len(self.limits) * (self.window + 1) * MAX_ROUTES):
            raise RuntimeError("Forced admission or metadata bound failed")
        return misses


def analyse(text, metadata, *, capacities, start_position, count, top_k, begin_line=1, end_line=None):
    validated = cohort.reconstruct_misses(text, metadata, capacities=capacities,
        start_position=start_position, count=count, top_k=top_k, begin_line=begin_line, end_line=end_line)
    lines = text.splitlines()
    last = len(lines) if end_line is None else end_line
    calls, trailing = cohort.retention.associate_forward_calls("\n".join(lines[begin_line - 1:last]), begin_line - 1)
    groups, lookup, source = cohort.retention.validate_metadata(metadata)
    selected, routed = cohort.retention.select_window(calls, lookup, start_position, count, top_k)
    if trailing or len(routed) != len(validated["rows"]):
        raise ValueError("Every completed route/layer association must be exact")
    order = tuple((source, layer) for layer in sorted(lookup))
    names = ("lru", causal.POLICY, POLICY)
    totals = {name: dict(misses=0, logical_source_bytes=0, per_group={}, per_position={}) for name in names}
    lru = {group: adaptive.ProtectedLru(size) for group, size in capacities.items()}
    # Finish every independent cold-LRU row/byte proof BEFORE policy creation.
    for index, (ordinal, route, timer) in enumerate(routed):
        group = lookup[route["layer"]]
        missed = lru[group["id"]].visit(source, route["layer"], route["ids"])
        proof = dict(p0=selected[ordinal]["timing"]["p0"], layer=route["layer"], group=group["id"],
            gpu_miss_experts=[key[2] for key in missed], gpu_hits=len(route["ids"]) - len(missed),
            expert_bytes=group["expert_bytes"], logical_source_bytes=len(missed) * group["expert_bytes"],
            route_line=route["line"], cache_timer_line=timer["line"])
        if (proof != validated["rows"][index] or proof["logical_source_bytes"] != timer["file_bytes"]
                or ordinal != index // len(order) or (source, route["layer"]) != order[index % len(order)]):
            raise ValueError("Independent original LRU disagrees with an exact saved row field")
        causal._accumulate(totals["lru"], group["id"], proof["p0"], group["expert_bytes"], missed)
    if (totals["lru"]["misses"] != validated["gpu_misses"]
            or totals["lru"]["logical_source_bytes"] != validated["logical_source_bytes"]):
        raise ValueError("Independent original LRU baseline failed exact totals")
    caches = {name: {group["id"]: cls(capacities[group["id"]],
        {(source, layer): group["projections"][0]["shape"][2] for layer in group["layers"]}, order)
        for group in groups} for name, cls in ((causal.POLICY, causal.CausalNeighbor), (POLICY, TransitionCache))}
    for index, (_, route, _) in enumerate(routed):
        group = lookup[route["layer"]]
        for name in names[1:]:
            missed = caches[name][group["id"]].visit(source, route["layer"], route["ids"])
            causal._accumulate(totals[name], group["id"], validated["rows"][index]["p0"], group["expert_bytes"], missed)
    baseline = totals["lru"]["logical_source_bytes"]
    results = {name: dict(net_logical_bytes_saved=baseline - totals[name]["logical_source_bytes"],
        stats={group: cache.stats for group, cache in caches[name].items()},
        layer_rows={str(layer): cache.observations[(source, layer)]
            for cache in caches[name].values() for _, layer in cache.limits}) for name in names[1:]}
    saved = results[POLICY]["net_logical_bytes_saved"]
    return dict(schema_version=1, scope="fixed_cold_shared_GPU_pool_conditional_transition_logical_counterfactual",
        baseline=validated, policies=totals, results=results, primary_policy=POLICY,
        material_screen=dict(minimum_reduction_percent=10, passed=baseline > 0 and saved * 10 >= baseline,
            signed_saved_bytes=saved, baseline_bytes=baseline, all_windows_required=True),
        layer_order=[layer for _, layer in order], primary_uses_future=False,
        current_transition_trained_before_decision=False, denominator_counts_outgoing_source_rows=True,
        source_ids_deduplicated_once_per_transition=True, own_layer_only=True,
        prediction_horizon_own_calls=1, prediction_deadline_full_cycle_ranks=len(order),
        expired_predictions_rebased=False, original_occurrences_touch_lru=True,
        cross_topic_history_carried=False, policy_selected_from_results=False,
        independent_lru_matched_every_row_field=True, every_layer_timer_matched=True,
        raw_uint16_matrix_bytes=sum(2 * sum(cache.limits[key] ** 2 for key in cache.limits)
            for cache in caches[POLICY].values()), native_implementation_qualified=False,
        metadata_physical_admission_qualified=False, physical_io_qualified=False,
        performance_qualified=False, numerical_quality_qualified=False,
        limitations=["Only pinned independent cold windows, not warm/PP/capacity-transition/current-profile evidence.",
            "One fixed64-transition primary, no parameter sweep or topic-dependent selection.",
            "Every cold topic resets payload and all counts/maps; second topic is a fixed-policy holdout.",
            "Only completed past adjacent own-layer transitions; current route is known, its outgoing future is not.",
            "Scores are exact rational conditional means over known current source IDs; zero and expired tie as LRU.",
            "Typed raw matrix bytes exclude Python objects, other metadata, allocator/OS overhead and native admission.",
            "Work counters bound logical operations, not measured native CPU time or latency savings.",
            "Logical bytes are not physical SSD/H2D service/residency or numerical-quality qualification."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name, expected in DEPENDENCIES.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest().upper() != expected:
            raise ValueError("Frozen proof dependency changed: " + name)
    profile = PROFILES[args.profile]
    raw = cohort._read_bounded(args.trace, causal.MAX_TRACE_BYTES, profile["trace_sha256"])
    saved = cohort._read_bounded(args.metadata, cohort.MAX_METADATA_BYTES, METADATA_SHA256)
    text = raw.decode("utf-8-sig", errors="strict")
    metadata = json.loads(saved.decode("utf-8-sig", errors="strict"))
    results = []
    for spec in profile["windows"]:
        options = {key: value for key, value in spec.items()
                   if key not in ("name", "role", "expected_misses", "expected_bytes")}
        result = analyse(text, metadata, **options)
        if (result["baseline"]["gpu_misses"] != spec["expected_misses"]
                or result["baseline"]["logical_source_bytes"] != spec["expected_bytes"]):
            raise ValueError("Pinned completed baseline totals changed")
        result["name"], result["topic_role"] = spec["name"], spec["role"]
        results.append(result)
    output = dict(schema_version=1, profile=args.profile, primary_policy=POLICY,
        primary_parameters=dict(completed_transition_window=PRIMARY_WINDOW, maximum_routes=MAX_ROUTES,
            expert_limit=MAX_EXPERTS, horizon_own_calls=1, material_reduction_percent=10),
        policy_selected_from_results=False, chronological_windows=results,
        all_profile_windows_pass_material_screen=all(x["material_screen"]["passed"] for x in results),
        provenance=dict(trace=str(args.trace.resolve()), trace_sha256=profile["trace_sha256"],
            metadata=str(args.metadata.resolve()), metadata_sha256=METADATA_SHA256,
            tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), dependencies=DEPENDENCIES,
            checkpoint_opened=False), native_implementation_qualified=False,
        physical_io_qualified=False, performance_qualified=False, numerical_quality_qualified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(profile=args.profile, primary_policy=POLICY,
        windows=[dict(name=x["name"], baseline=x["baseline"]["logical_source_bytes"],
            saved=x["results"][POLICY]["net_logical_bytes_saved"],
            material_screen_passed=x["material_screen"]["passed"]) for x in results],
        performance_qualified=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
