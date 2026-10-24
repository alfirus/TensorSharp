#!/usr/bin/env python3
"""Causal route-neighbor replacement in fixed shared GPU pools, logical only.

One predeclared history64/suffix8/intersection2 policy. Each resident key uses
its OWN layer's last prediction on the validated global layer cycle. CLI inputs
are pinned cold captures and descriptors, never checkpoint payloads. No native,
physical-I/O, numerical-quality or speed qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import glm_gpu_adaptive_replay as adaptive

cohort = adaptive.cohort
HISTORY_ROWS = 64
SUFFIX_ROWS = 8
MIN_INTERSECTION = 2
POLICY = "causal_neighbor_64x8"
MAX_SOURCES = 256
MAX_KEYS = 65536
MAX_TRACE_BYTES = 64 << 20
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


def _identity(value):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("Exact immutable source/layer identity required")
    adaptive.demand_keys(*value, [0], 1)
    return value


class CausalNeighbor:
    """Bounded per-layer history, one shared slot pool, forced current union.

    Row t is appended ONLY after the decision. Anchors satisfy i+8<t. A new
    query replaces its own layer's prediction map, clearing it on no match.
    Due ordinals are absolute and never rebased: L*(issue_cycle+offset)+rank.
    Stored expired entries stay bounded until that layer's next refresh; they
    score as unknown immediately at due<=current_global_ordinal.
    """
    def __init__(self, capacity, source_limits, layer_order):
        size = adaptive.bounded(capacity, "capacity", 1, adaptive.MAX_CAPACITY)
        if type(source_limits) is not dict or not 1 <= len(source_limits) <= MAX_SOURCES:
            raise ValueError("A bounded immutable source registry is required")
        limits = {_identity(key): adaptive.bounded(value, "source experts", 1, MAX_KEYS)
                  for key, value in source_limits.items()}
        if sum(limits.values()) > MAX_KEYS:
            raise ValueError("Source-key registry exceeds metadata bound")
        if not isinstance(layer_order, (tuple, list)) or not 1 <= len(layer_order) <= MAX_SOURCES:
            raise ValueError("A bounded complete fixed layer cycle is required")
        order = tuple(_identity(key) for key in layer_order)
        if len(set(order)) != len(order) or not set(limits).issubset(order):
            raise ValueError("The complete layer cycle must contain every source exactly once")
        # Complete geometry/registry validation precedes all ring allocation.
        self.capacity, self.limits, self.layer_order = size, limits, order
        self.rank = {key: index for index, key in enumerate(order)}
        self.pool_order = tuple(key for key in order if key in limits)
        self.rings = {key: [None] * HISTORY_ROWS for key in limits}
        self.observations = dict.fromkeys(limits, 0)
        self.predictions = {key: {} for key in limits}
        self.slots, self.ages = [None] * size, [0] * size
        self.clock = self.events = 0
        self.ring_references = self.stored_predictions = 0
        self.last_match = None
        self.stats = dict(rows=0, misses=0, evictions=0, resident_hit_occurrences=0,
            planned_duplicate_occurrences=0, matched_queries=0, fallback_queries=0,
            candidate_comparisons=0, eligible_suffixes=0, similarity_key_checks=0,
            suffix_key_references=0, victim_checks=0, cross_layer_victim_checks=0,
            finite_prediction_victim_checks=0, unknown_prediction_victim_checks=0,
            expired_prediction_victim_checks=0, expired_history_rows=0,
            minimum_suffix_completion_lag=None, peak_candidates_per_query=0,
            peak_history_rows_per_layer=0, peak_ring_key_references=0,
            peak_prediction_keys_per_layer=0, peak_stored_prediction_keys=0,
            peak_resident=0)

    @property
    def resident(self):
        return set(self.slots) - {None}

    def snapshot(self):
        histories = tuple((key, self.observations[key], tuple(self.rings[key]),
                           tuple(sorted(self.predictions[key].items())))
                          for key in sorted(self.limits))
        return (tuple(self.slots), tuple(self.ages), self.clock, self.events,
                histories, self.ring_references, self.stored_predictions,
                None if self.last_match is None else tuple(sorted(self.last_match.items())),
                tuple(sorted(self.stats.items())))

    def _row(self, identity, ordinal):
        saved = self.rings[identity][ordinal % HISTORY_ROWS]
        if saved is None or saved[0] != ordinal:
            raise RuntimeError("Historical suffix escaped its completed bounded ring")
        return saved[1]

    def predicted_due(self, key, current_global_ordinal):
        """Unknown/expired is None (semantic infinity), never a future lookup."""
        value = self.predictions.get(key[:2], {}).get(key[2])
        return value if value is not None and value > current_global_ordinal else None

    def _predict(self, identity, current, local_row, global_ordinal):
        first = max(0, local_row - HISTORY_ROWS)
        best, best_intersection, best_union, candidates = None, 0, 1, 0
        for anchor in range(first, local_row - SUFFIX_ROWS):
            prior = self._row(identity, anchor)
            intersection = len(current.intersection(prior))
            union = len(current) + len(prior) - intersection
            candidates += 1
            self.stats["similarity_key_checks"] += len(current)
            if intersection < MIN_INTERSECTION:
                continue
            if (best is None or intersection * best_union > best_intersection * union
                    or (intersection * best_union == best_intersection * union and anchor > best)):
                best, best_intersection, best_union = anchor, intersection, union
        self.stats["candidate_comparisons"] += candidates
        self.stats["eligible_suffixes"] += candidates
        self.stats["peak_candidates_per_query"] = max(self.stats["peak_candidates_per_query"], candidates)
        predicted = {}
        if best is None:
            self.last_match = None
            self.stats["fallback_queries"] += 1
        else:
            last = best + SUFFIX_ROWS
            if not last < local_row:
                raise RuntimeError("A prediction used an uncompleted suffix")
            for offset in range(1, SUFFIX_ROWS + 1):
                row = self._row(identity, best + offset)
                self.stats["suffix_key_references"] += len(row)
                due = len(self.layer_order) * (local_row + offset) + self.rank[identity]
                for expert in row:
                    predicted.setdefault(expert, due)
            lag = local_row - last
            old_lag = self.stats["minimum_suffix_completion_lag"]
            self.stats["minimum_suffix_completion_lag"] = lag if old_lag is None else min(old_lag, lag)
            self.stats["matched_queries"] += 1
            self.last_match = dict(query_local_row=local_row, query_global_ordinal=global_ordinal,
                matched_local_row=best, suffix_last_local_row=last,
                intersection=best_intersection, union=best_union,
                suffix_completion_lag=lag, predicted_keys=len(predicted))
        self.stored_predictions += len(predicted) - len(self.predictions[identity])
        self.predictions[identity] = predicted  # No-match discards old confidence.
        self.stats["peak_prediction_keys_per_layer"] = max(self.stats["peak_prediction_keys_per_layer"], len(predicted))
        self.stats["peak_stored_prediction_keys"] = max(self.stats["peak_stored_prediction_keys"], self.stored_predictions)

    def _priority(self, slot, identity, global_ordinal):
        old = self.slots[slot]
        if old is None:
            return 0, 0, 0, self.ages[slot], slot
        self.stats["victim_checks"] += 1
        self.stats["cross_layer_victim_checks"] += old[:2] != identity
        raw = self.predictions[old[:2]].get(old[2])
        due = self.predicted_due(old, global_ordinal)
        if due is None:
            self.stats["unknown_prediction_victim_checks"] += 1
            self.stats["expired_prediction_victim_checks"] += raw is not None and raw <= global_ordinal
            return 1, 0, 0, self.ages[slot], slot  # infinity first
        self.stats["finite_prediction_victim_checks"] += 1
        return 1, 1, -due, self.ages[slot], slot

    def visit(self, source, layer, ids):
        keys = adaptive.demand_keys(source, layer, ids, self.capacity)
        identity = source, layer
        if identity not in self.limits or any(key[2] >= self.limits[identity] for key in keys):
            raise ValueError("Unregistered or out-of-source-bounds demand")
        expected = self.pool_order[self.events % len(self.pool_order)]
        local_row = self.events // len(self.pool_order)
        if identity != expected or self.observations[identity] != local_row:
            raise ValueError("Demand violates the complete fixed layer-cycle chronology")
        # Every invalid-input/schedule check is before any history/map/slot/stat mutation.
        global_ordinal = len(self.layer_order) * local_row + self.rank[identity]
        unique = frozenset(key[2] for key in keys)
        self._predict(identity, unique, local_row, global_ordinal)
        locations = {}
        for slot, key in enumerate(self.slots):
            if key is not None:
                locations.setdefault(key, slot)
        original = set(locations)
        protected = {locations[key] for key in keys if key in locations}
        self.stats["resident_hit_occurrences"] += sum(key in original for key in keys)
        misses = []
        for key in keys:
            if key in locations:
                if key not in original:
                    self.stats["planned_duplicate_occurrences"] += 1
                continue
            victim = min((slot for slot in range(self.capacity) if slot not in protected),
                         key=lambda slot: self._priority(slot, identity, global_ordinal))
            old = self.slots[victim]
            if old is not None:
                locations.pop(old)
                self.stats["evictions"] += 1
            self.slots[victim], locations[key] = key, victim
            protected.add(victim)
            misses.append(key)
        for key in keys:
            self.clock += 1
            self.ages[locations[key]] = self.clock
        ring = self.rings[identity]
        old = ring[local_row % HISTORY_ROWS]
        if old is not None:
            self.ring_references -= len(old[1])
            self.stats["expired_history_rows"] += 1
        ring[local_row % HISTORY_ROWS] = local_row, unique
        self.ring_references += len(unique)
        self.observations[identity] += 1
        self.events += 1
        self.stats["rows"] += 1
        self.stats["misses"] += len(misses)
        self.stats["peak_history_rows_per_layer"] = max(self.stats["peak_history_rows_per_layer"], min(HISTORY_ROWS, local_row + 1))
        self.stats["peak_ring_key_references"] = max(self.stats["peak_ring_key_references"], self.ring_references)
        self.stats["peak_resident"] = max(self.stats["peak_resident"], len(self.resident))
        if (not set(keys).issubset(self.resident) or len(self.resident) > self.capacity
                or self.stored_predictions > len(self.limits) * SUFFIX_ROWS * adaptive.MAX_ROW
                or self.ring_references > len(self.limits) * HISTORY_ROWS * adaptive.MAX_ROW):
            raise RuntimeError("Forced admission or bounded metadata invariant failed")
        return misses


def _accumulate(total, group, position, expert_bytes, missed):
    size = len(missed) * expert_bytes
    total["misses"] += len(missed)
    total["logical_source_bytes"] += size
    total["per_group"][group] = total["per_group"].get(group, 0) + size
    total["per_position"][position] = total["per_position"].get(position, 0) + size


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
    selected, routed = cohort.retention.select_window(calls, lookup, start_position, count, top_k)
    if trailing or len(routed) != len(validated["rows"]):
        raise ValueError("Every completed route/layer association must be exact")
    order = tuple((source, layer) for layer in sorted(lookup))
    totals = {name: dict(misses=0, logical_source_bytes=0, per_group={}, per_position={})
              for name in ("lru", POLICY)}
    lru = {group: adaptive.ProtectedLru(size) for group, size in capacities.items()}
    # Finish EVERY independent cold-LRU field/byte check before creating the policy.
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
        _accumulate(totals["lru"], group["id"], proof["p0"], group["expert_bytes"], missed)
    if (totals["lru"]["misses"] != validated["gpu_misses"]
            or totals["lru"]["logical_source_bytes"] != validated["logical_source_bytes"]):
        raise ValueError("Independent original LRU baseline failed exact totals")
    caches = {group["id"]: CausalNeighbor(capacities[group["id"]],
        {(source, layer): group["projections"][0]["shape"][2] for layer in group["layers"]}, order)
        for group in groups}
    for index, (_, route, _) in enumerate(routed):
        group = lookup[route["layer"]]
        missed = caches[group["id"]].visit(source, route["layer"], route["ids"])
        _accumulate(totals[POLICY], group["id"], validated["rows"][index]["p0"], group["expert_bytes"], missed)
    policy = dict(role="predeclared_primary", history_rows_per_layer=HISTORY_ROWS,
        completed_suffix_rows=SUFFIX_ROWS, minimum_intersection=MIN_INTERSECTION,
        stats={group: cache.stats for group, cache in caches.items()},
        layer_rows={str(layer): cache.observations[(source, layer)]
            for cache in caches.values() for _, layer in cache.limits},
        net_logical_bytes_saved=totals["lru"]["logical_source_bytes"] - totals[POLICY]["logical_source_bytes"])
    return dict(schema_version=1,
        scope="fixed_cold_shared_GPU_pool_causal_route_neighbor_logical_counterfactual",
        baseline=validated, policies=totals, neighbor=policy, primary_policy=POLICY,
        layer_order=[layer for _, layer in order], primary_uses_future=False,
        current_row_in_historical_suffix=False, strict_suffix_completed_before_query=True,
        own_layer_absolute_due_ordinals=True, expired_predictions_rebased=False,
        duplicates_count_once_for_similarity=True, original_occurrences_touch_lru=True,
        cross_topic_history_carried=False, policy_selected_from_results=False,
        independent_lru_matched_every_row_field=True, every_layer_timer_matched=True,
        native_implementation_qualified=False, metadata_physical_admission_qualified=False,
        physical_io_qualified=False, performance_qualified=False, numerical_quality_qualified=False,
        limitations=["Only pinned cold windows; no warm/PP/capacity-transition or current-profile generalization.",
            "One fixed64/8/intersection2 policy; no sweep or topic-dependent selection.",
            "Every cold topic resets payload, per-layer history and predictions; no cross-topic training.",
            "Other-layer keys use their own latest absolute prediction; stale due times are unknown, never rebased.",
            "No-match clears the querying layer map; other-layer still-valid maps remain causal.",
            "Descriptor namespace/layer/expert substitutes for native generation IDs absent in captures.",
            "Python ordinals do not prove native timestamp overflow or metadata admission safety.",
            "History/similarity/victim work adds unmeasured CPU cost; bounded metadata has no native Host charge.",
            "Logical quantized source bytes are not SSD traffic, H2D service time, OS residency, speed or math validation."])


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
    raw = cohort._read_bounded(args.trace, MAX_TRACE_BYTES, profile["trace_sha256"])
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
        primary_parameters=dict(history=HISTORY_ROWS, suffix=SUFFIX_ROWS, minimum_intersection=MIN_INTERSECTION),
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
    print(json.dumps(dict(profile=args.profile, primary_policy=POLICY,
        windows=[dict(name=result["name"], baseline=result["baseline"]["logical_source_bytes"],
            net_saved=result["neighbor"]["net_logical_bytes_saved"]) for result in results],
        performance_qualified=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
