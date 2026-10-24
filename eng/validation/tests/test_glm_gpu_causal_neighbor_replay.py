"""Independent tiny causal-history, pooled-slot and cold-trace references."""
import contextlib
from fractions import Fraction
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_gpu_causal_neighbor_replay as neighbor

SOURCE = "source-generation-a"
OTHER = "source-generation-b"


class PastListReference:
    """Full past lists, exact Fraction match and oldest-first resident keys.

    No circular buffer, slot ages, incremental map accounting or policy calls.
    Only the first argument's already-completed rows are available to visit.
    """
    def __init__(self, capacity, order, identities=None):
        self.capacity, self.order = capacity, tuple(order)
        self.history, self.resident = {}, []
        self.maps = {key: {} for key in (order if identities is None else identities)}
        self.anchor = None

    def visit(self, source, layer, ids):
        identity = source, layer
        past = self.history.setdefault(identity, [])
        t, current = len(past), set(ids)
        choices = []
        for i in range(max(0, t - 64), t - 8):
            overlap = len(current & set(past[i]))
            if overlap >= 2:
                choices.append((Fraction(overlap, len(current | set(past[i]))), i))
        self.anchor = max(choices)[1] if choices else None
        new = {}
        if self.anchor is not None:
            for expert in set().union(*map(set, past[self.anchor + 1:self.anchor + 9])):
                first = next(d for d in range(1, 9) if expert in past[self.anchor + d])
                new[expert] = len(self.order) * (t + first) + self.order.index(identity)
        self.maps[identity] = new
        now = len(self.order) * t + self.order.index(identity)
        keys = [(source, layer, expert) for expert in ids]
        protected, missing = set(keys), []
        for key in keys:
            if key in self.resident:
                continue
            if len(self.resident) == self.capacity:
                candidates = [old for old in self.resident if old not in protected]
                def priority(old):
                    due = self.maps.get(old[:2], {}).get(old[2])
                    return (float("-inf") if due is None or due <= now else -due,
                            self.resident.index(old))
                self.resident.remove(min(candidates, key=priority))
            self.resident.append(key)
            missing.append(key)
        for key in keys:
            self.resident.remove(key)
            self.resident.append(key)
        past.append(tuple(ids))
        return missing


class ListLru:
    """Independent raw trace writer; no neighbor, slot-age or history logic."""
    def __init__(self, capacity):
        self.capacity, self.resident = capacity, []

    def visit(self, layer, ids):
        keys, missing = [(layer, expert) for expert in ids], []
        protected = set(keys)
        for key in keys:
            if key not in self.resident:
                if len(self.resident) == self.capacity:
                    self.resident.remove(next(old for old in self.resident if old not in protected))
                self.resident.append(key)
                missing.append(key)
        for key in keys:
            self.resident.remove(key)
            self.resident.append(key)
        return missing


def make(capacity=3, identities=((SOURCE, 1),), experts=16, order=None):
    return neighbor.CausalNeighbor(capacity, dict.fromkeys(identities, experts),
                                   identities if order is None else order)


def train_match(cache, identity=(SOURCE, 1)):
    """Anchor0 and eight completed suffixes; next query is local row9."""
    for ids in ([0, 1],) + ([2, 3],) * 8:
        cache.visit(*identity, ids)


def metadata():
    groups = []
    for first, layers, hidden, ff in ((1, [1, 2], 2, 2), (3, [3], 2, 4), (4, [4], 4, 4)):
        groups.append(dict(id=first, layers=layers, expert_bytes=hidden * ff * 4 * 3,
            projections=[dict(type=0, shape=shape, expert_bytes=hidden * ff * 4)
                for shape in ([hidden, ff, 8], [hidden, ff, 8], [ff, hidden, 8])]))
    return dict(groups=groups, model_files=[dict(path="never-opened.gguf", bytes=12345, mtime_ns=1)])


def forward(position):
    fields = dict(kind=0, rows=1, p0=position, nout=1, wanth=0, logits=1, reused=0,
        outcome="completed", acquire_ms="0", input_ms="0", compute_ms="1", output_ms="0",
        total_ms="1", scope=neighbor.cohort.retention._FORWARD_SCOPE)
    return neighbor.cohort.retention._FORWARD + " " + " ".join(f"{key}={value}" for key, value in fields.items())


def trace(pairs=None):
    pairs = ((0, 1),) + ((2, 3),) * 8 + ((0, 1), (4, 5), (0, 1)) if pairs is None else pairs
    groups = metadata()["groups"]
    lookup = {layer: group for group in groups for layer in group["layers"]}
    caches = {group["id"]: ListLru(3) for group in groups}
    lines, byte_total = [], 0
    for index, ids in enumerate(pairs):
        for layer in sorted(lookup):
            group = lookup[layer]
            if not index and layer == group["id"]:
                lines.append(f"[HOSTMOE-ROUTE-CREATE] layer={layer} slots=3 experts=8")
            missed = caches[group["id"]].visit(layer, ids)
            size = len(missed) * group["expert_bytes"]
            byte_total += size
            lines.append(f"[HOSTMOE-ROUTE] layer={layer} ids=" + ",".join(map(str, ids)))
            lines.append(f"[HOSTMOE-CACHE-TIMING] layer={layer} rows=1 joint=0 bulk=0 outcome=completed "
                         f"file_bytes={size} scope=nested_host_wall_not_cuda_kernel_or_physical_io")
        lines.append(forward(33 + index))
    return "\n".join(lines), byte_total


def analyse(text=None, data=None, **changes):
    options = dict(capacities={1: 3, 3: 3, 4: 3}, start_position=33, count=12, top_k=2)
    options.update(changes)
    return neighbor.analyse(trace()[0] if text is None else text,
                            metadata() if data is None else data, **options)


class CausalPolicyTests(unittest.TestCase):
    def test_transactional_registry_and_complete_cycle_bounds(self):
        for capacity in (0, -1, True, 1.0, neighbor.adaptive.MAX_CAPACITY + 1):
            with self.assertRaises(ValueError):
                make(capacity)
        for limits in ({}, {("", 1): 8}, {(SOURCE, True): 8}, {(SOURCE, 1): True},
                       {(SOURCE, 1): 0}, {(SOURCE, 1): 65537}, {SOURCE: 8},
                       {(SOURCE, n): 1 for n in range(257)},
                       {(SOURCE, 1): 40000, (SOURCE, 2): 40000}):
            with self.subTest(limits=str(limits)[:80]), self.assertRaises(ValueError):
                neighbor.CausalNeighbor(3, limits, [(SOURCE, 1)])
        for order in ([], [(SOURCE, 1)] * 2, [(OTHER, 1)], [SOURCE], [(SOURCE, True)]):
            with self.subTest(order=order), self.assertRaises(ValueError):
                neighbor.CausalNeighbor(3, {(SOURCE, 1): 8}, order)
        limits, order = {(SOURCE, 1): 8}, [(SOURCE, 1), (SOURCE, 2)]
        cache = neighbor.CausalNeighbor(3, limits, order)
        limits[(SOURCE, 1)] = 100; order.reverse()
        self.assertEqual(cache.limits[(SOURCE, 1)], 8)
        self.assertEqual(cache.layer_order, ((SOURCE, 1), (SOURCE, 2)))

    def test_invalid_entire_row_or_wrong_layer_changes_no_state(self):
        cache = make(3, ((SOURCE, 1), (SOURCE, 2)))
        cache.visit(SOURCE, 1, [0, 1])
        before = cache.snapshot()
        for source, layer, ids in ((SOURCE, 1, [1]), (OTHER, 2, [1]), (SOURCE, 2, []),
                (SOURCE, 2, [0, 16]), (SOURCE, 2, [0, True]), (SOURCE, 2, [0, -1]),
                (SOURCE, 2, [0, 1, 2, 3]), (SOURCE, 2, "0,1"), (SOURCE, 2, [0] * 65)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                cache.visit(source, layer, ids)
            self.assertEqual(cache.snapshot(), before)

    def test_strict_following_eight_completed_boundary_excludes_current_row(self):
        cache = make()
        cache.visit(SOURCE, 1, [0, 1])
        for _ in range(7):
            cache.visit(SOURCE, 1, [2, 3])
        cache.visit(SOURCE, 1, [0, 1])  # t8, anchor0 suffix8 not completed yet.
        self.assertIsNone(cache.last_match)
        self.assertEqual(cache.stats["candidate_comparisons"], 0)
        cache.visit(SOURCE, 1, [0, 1])  # t9, anchor0 suffix1..8 now completed.
        proof = cache.last_match
        self.assertEqual((proof["query_local_row"], proof["matched_local_row"], proof["suffix_last_local_row"]), (9, 0, 8))
        self.assertEqual(proof["suffix_completion_lag"], 1)
        self.assertEqual(cache.predictions[(SOURCE, 1)][2], 10)
        self.assertEqual(cache.predictions[(SOURCE, 1)][0], 17)
        self.assertEqual(cache.stats["peak_candidates_per_query"], 1)

    def test_jaccard_exact_ratio_and_most_recent_equal_similarity_tie(self):
        cache = make(5)
        rows = [[0, 1, 2, 3], [0, 1]] + [[4, 5]] * 8
        for row in rows:
            cache.visit(SOURCE, 1, row)
        cache.visit(SOURCE, 1, [0, 1, 2])
        self.assertEqual(cache.last_match["matched_local_row"], 0)  # 3/4 > 2/3
        newer = make(4)
        for row in [[0, 1], [0, 1, 0]] + [[4, 5]] * 8:
            newer.visit(SOURCE, 1, row)
        newer.visit(SOURCE, 1, [0, 1])
        self.assertEqual(newer.last_match["matched_local_row"], 1)
        self.assertEqual((newer.last_match["intersection"], newer.last_match["union"]), (2, 2))

    def test_intersection_one_never_matches_and_no_match_clears_old_confidence(self):
        cache = make()
        train_match(cache)
        cache.visit(SOURCE, 1, [0, 1])
        self.assertTrue(cache.predictions[(SOURCE, 1)])
        cache.visit(SOURCE, 1, [0, 7])
        self.assertIsNone(cache.last_match)
        self.assertEqual(cache.predictions[(SOURCE, 1)], {})
        self.assertEqual(cache.stored_predictions, 0)

    def test_suffix_predicts_only_earliest_known_offset_and_not_current_keys(self):
        cache = make(4)
        for row in [[0, 1], [2, 3], [4, 5], [2, 4], [3, 5], [4, 5], [4, 5], [4, 5], [4, 5]]:
            cache.visit(SOURCE, 1, row)
        cache.visit(SOURCE, 1, [0, 1])
        self.assertEqual(cache.predictions[(SOURCE, 1)], {2: 10, 3: 10, 4: 11, 5: 11})
        self.assertNotIn(0, cache.predictions[(SOURCE, 1)])

    def test_other_layers_use_own_absolute_cycle_ranks_not_current_layer_history(self):
        identities = ((SOURCE, 1), (SOURCE, 2))
        cache = make(5, identities)
        for cycle in range(9):
            cache.visit(SOURCE, 1, [0, 1] if cycle == 0 else [2, 3])
            cache.visit(SOURCE, 2, [0, 1] if cycle == 0 else [4, 5])
        cache.visit(SOURCE, 1, [0, 1])  # cycle9/rank0, due20 for own2/3.
        own = dict(cache.predictions[(SOURCE, 1)])
        self.assertEqual(own, {2: 20, 3: 20})
        cache.visit(SOURCE, 2, [0, 1])  # cycle9/rank1, due21 for own4/5.
        self.assertEqual(cache.predictions[(SOURCE, 1)], own)
        self.assertEqual(cache.predictions[(SOURCE, 2)], {4: 21, 5: 21})
        self.assertEqual(cache.predicted_due((SOURCE, 1, 2), 19), 20)
        self.assertEqual(cache.predicted_due((SOURCE, 2, 4), 20), 21)
        self.assertIsNone(cache.predicted_due((SOURCE, 1, 2), 20))
        self.assertIsNone(cache.predicted_due((SOURCE, 2, 4), 21))
        cache.visit(SOURCE, 1, [6, 7])  # own no match clears ONLY own prediction.
        self.assertEqual(cache.predictions[(SOURCE, 1)], {})
        self.assertEqual(cache.predictions[(SOURCE, 2)], {4: 21, 5: 21})

    def test_global_rank_includes_other_groups_and_same_layer_other_generation(self):
        identities = ((SOURCE, 1), (OTHER, 1))
        order = ((SOURCE, 1), (SOURCE, 2), (SOURCE, 3), (OTHER, 1))
        cache = make(5, identities, order=order)
        for cycle in range(9):
            cache.visit(SOURCE, 1, [0, 1] if cycle == 0 else [2, 3])
            cache.visit(OTHER, 1, [0, 1] if cycle == 0 else [4, 5])
        cache.visit(SOURCE, 1, [0, 1]); cache.visit(OTHER, 1, [0, 1])
        self.assertEqual(cache.predictions[(SOURCE, 1)], {2: 40, 3: 40})
        self.assertEqual(cache.predictions[(OTHER, 1)], {4: 43, 5: 43})
        self.assertIsNone(cache.predicted_due((OTHER, 1, 2), 39))
        self.assertEqual(cache.last_match["query_global_ordinal"], 39)

    def test_unknown_first_then_farthest_due_then_original_lru_slot_ties(self):
        cache = make(3)
        cache.visit(SOURCE, 1, [0, 1, 2])
        cache.predictions[(SOURCE, 1)] = {0: 10, 1: 12}  # Unit test scoring only.
        cache.stored_predictions = 2
        self.assertLess(cache._priority(2, (SOURCE, 1), 1), cache._priority(1, (SOURCE, 1), 1))
        self.assertLess(cache._priority(1, (SOURCE, 1), 1), cache._priority(0, (SOURCE, 1), 1))
        self.assertLess(cache._priority(1, (SOURCE, 1), 12), cache._priority(0, (SOURCE, 1), 1))
        cache.ages[:] = [7, 7, 7]
        cache.visit(SOURCE, 1, [3])  # No match clears injected map; exact ascending tie.
        self.assertEqual(cache.slots[0], (SOURCE, 1, 3))
        self.assertEqual(cache.slots[1:], [(SOURCE, 1, 1), (SOURCE, 1, 2)])

    def test_all_original_hits_protected_before_any_miss_and_all_misses_admitted(self):
        cache = make(2)
        cache.visit(SOURCE, 1, [0, 1])
        self.assertEqual(cache.visit(SOURCE, 1, [2, 0]), [(SOURCE, 1, 2)])
        self.assertEqual(cache.resident, {(SOURCE, 1, 0), (SOURCE, 1, 2)})
        self.assertEqual(cache.clock, 4)
        self.assertGreater(cache.ages[cache.slots.index((SOURCE, 1, 0))],
                           cache.ages[cache.slots.index((SOURCE, 1, 2))])

    def test_duplicates_admit_and_match_once_but_touch_every_original_occurrence(self):
        cache = make()
        self.assertEqual(cache.visit(SOURCE, 1, [0, 1, 0]), [(SOURCE, 1, 0), (SOURCE, 1, 1)])
        self.assertEqual(cache.rings[(SOURCE, 1)][0], (0, frozenset((0, 1))))
        self.assertEqual((cache.clock, cache.stats["planned_duplicate_occurrences"]), (3, 1))
        self.assertGreater(cache.ages[0], cache.ages[1])
        cache.visit(SOURCE, 1, [0, 0])
        self.assertEqual(cache.stats["resident_hit_occurrences"], 2)
        self.assertEqual(cache.ring_references, 3)

    def test_no_eligible_match_is_exact_original_lru_including_shared_pool(self):
        identities = ((SOURCE, 1), (SOURCE, 2))
        for capacity in (2, 3, 5):
            cache, lru = make(capacity, identities), neighbor.adaptive.ProtectedLru(capacity)
            for cycle in range(30):
                for source, layer in identities:
                    ids = [cycle % 7]  # Always similarity<2, despite repeated history.
                    self.assertEqual(cache.visit(source, layer, ids), lru.visit(source, layer, ids))
                    self.assertEqual((tuple(cache.slots), tuple(cache.ages), cache.clock), lru.snapshot())
            self.assertEqual(cache.stats["matched_queries"], 0)

    def test_ring_wrap_keeps_repeated_rows_and_exact_eligible_suffix_bound(self):
        cache = make()
        for _ in range(3 * 64 + 11):
            cache.visit(SOURCE, 1, [0, 1])
        self.assertEqual(cache.stats["peak_history_rows_per_layer"], 64)
        self.assertEqual(cache.stats["peak_candidates_per_query"], 56)
        self.assertEqual(cache.ring_references, 128)
        self.assertEqual(cache.last_match["matched_local_row"], 202 - 9)
        self.assertEqual(cache.last_match["suffix_completion_lag"], 1)
        self.assertEqual(cache.stats["expired_history_rows"], 203 - 64)
        self.assertEqual(cache.stored_predictions, 2)
        self.assertLessEqual(cache.stats["peak_prediction_keys_per_layer"], 8 * 64)

    def test_independent_full_past_reference_multiple_sources_and_shared_evictions(self):
        identities = ((SOURCE, 1), (SOURCE, 2), (OTHER, 1))
        order = ((SOURCE, 1), (SOURCE, 2), (SOURCE, 3), (OTHER, 1))
        for capacity in (2, 3, 5):
            cache, reference = make(capacity, identities, order=order), PastListReference(capacity, order, identities)
            for cycle in range(145):
                for source, layer in identities:
                    offset = (cycle % 11 + layer) % 7
                    ids = [offset, (offset + 2) % 7]
                    if cycle % 4 == 0:
                        ids.append(ids[0])
                    self.assertEqual(cache.visit(source, layer, ids), reference.visit(source, layer, ids))
                    self.assertEqual(cache.resident, set(reference.resident))
                    aged = [cache.slots[slot] for slot in sorted(range(capacity), key=lambda i: cache.ages[i])
                            if cache.slots[slot] is not None]
                    self.assertEqual(aged, reference.resident)
                    self.assertEqual(cache.predictions, reference.maps)
                    self.assertEqual(None if cache.last_match is None else cache.last_match["matched_local_row"], reference.anchor)
            self.assertGreater(cache.stats["cross_layer_victim_checks"], 0)
            self.assertLessEqual(cache.stats["peak_ring_key_references"], 3 * 64 * 2)
            self.assertLessEqual(cache.stats["peak_stored_prediction_keys"], 3 * 8 * 2)

    def test_independent_exhaustive_tiny_continuations_after_eligible_history(self):
        seed = [[0, 1], [1, 2], [0, 2]] * 3
        choices = ((0, 1), (1, 2), (0, 2))
        for capacity in (2, 3):
            for continuation in itertools.product(choices, repeat=4):
                cache, reference = make(capacity), PastListReference(capacity, ((SOURCE, 1),))
                for ids in seed + list(continuation):
                    self.assertEqual(cache.visit(SOURCE, 1, ids), reference.visit(SOURCE, 1, ids))
                    self.assertEqual(cache.resident, set(reference.resident))
                    self.assertEqual(cache.predictions, reference.maps)

    def test_arbitrary_future_suffix_cannot_change_any_prefix_state(self):
        prefix = [[0, 1], [2, 3], [0, 2]] * 30
        observed = []
        for suffix in ([[6, 7]] * 20, [[0, 1], [4, 5]] * 10):
            cache, snapshots = make(), []
            for ids in prefix + suffix:
                cache.visit(SOURCE, 1, ids)
                snapshots.append(cache.snapshot())
            observed.append(snapshots[:len(prefix)])
        self.assertEqual(observed[0], observed[1])

    def test_same_causal_choice_can_win_or_lose_without_future_selection(self):
        prefix = [[0, 1]] + [[2, 3]] * 7 + [[4, 5], [0, 1]]
        snapshots, differences = [], []
        for actual_next in ([3], [5]):
            cache, lru = make(), neighbor.adaptive.ProtectedLru(3)
            before_policy = before_lru = 0
            for ids in prefix:
                before_policy += len(cache.visit(SOURCE, 1, ids))
                before_lru += len(lru.visit(SOURCE, 1, ids))
            snapshots.append(cache.snapshot())
            self.assertEqual(before_policy, before_lru)
            self.assertIn((SOURCE, 1, 3), cache.resident)
            self.assertIn((SOURCE, 1, 5), lru.resident)
            policy_misses = len(cache.visit(SOURCE, 1, actual_next))
            lru_misses = len(lru.visit(SOURCE, 1, actual_next))
            differences.append(lru_misses - policy_misses)
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(differences, [1, -1])

    def test_absent_match_cannot_keep_an_expired_prediction_alive(self):
        cache = make()
        train_match(cache)
        cache.visit(SOURCE, 1, [0, 1])
        due = cache.predictions[(SOURCE, 1)][2]
        self.assertEqual(cache.predicted_due((SOURCE, 1, 2), due - 1), due)
        self.assertIsNone(cache.predicted_due((SOURCE, 1, 2), due))
        self.assertIsNone(cache.predicted_due((SOURCE, 1, 2), due + 10000))
        cache.visit(SOURCE, 1, [6, 7])
        self.assertEqual(cache.predictions[(SOURCE, 1)], {})


class TraceAndCliTests(unittest.TestCase):
    def test_full_cold_original_lru_every_field_and_byte_before_counterfactual(self):
        text, expected = trace()
        result = analyse(text)
        self.assertEqual(result["baseline"]["layer_timer_checks"], 48)
        self.assertEqual(result["policies"]["lru"]["logical_source_bytes"], expected)
        self.assertEqual(result["primary_policy"], neighbor.POLICY)
        self.assertEqual(result["layer_order"], [1, 2, 3, 4])
        self.assertTrue(result["independent_lru_matched_every_row_field"])
        self.assertTrue(result["strict_suffix_completed_before_query"])
        self.assertTrue(result["own_layer_absolute_due_ordinals"])
        for flag in ("primary_uses_future", "current_row_in_historical_suffix", "expired_predictions_rebased",
                     "cross_topic_history_carried", "policy_selected_from_results", "native_implementation_qualified",
                     "metadata_physical_admission_qualified", "physical_io_qualified", "performance_qualified", "numerical_quality_qualified"):
            self.assertFalse(result[flag])
        policy = result["policies"][neighbor.POLICY]
        self.assertEqual(sum(policy["per_group"].values()), policy["logical_source_bytes"])
        self.assertEqual(sum(policy["per_position"].values()), policy["logical_source_bytes"])
        self.assertEqual(result["neighbor"]["net_logical_bytes_saved"], expected - policy["logical_source_bytes"])
        self.assertEqual((result["neighbor"]["history_rows_per_layer"], result["neighbor"]["completed_suffix_rows"],
                          result["neighbor"]["minimum_intersection"]), (64, 8, 2))

    def test_equal_opposite_byte_errors_reject_before_policy_construction(self):
        text = trace()[0].replace("file_bytes=96", "file_bytes=95", 1)
        text = text.replace("layer=2 rows=1 joint=0 bulk=0 outcome=completed file_bytes=96",
                            "layer=2 rows=1 joint=0 bulk=0 outcome=completed file_bytes=97", 1)
        with mock.patch.object(neighbor, "CausalNeighbor") as policy:
            with self.assertRaisesRegex(ValueError, "miss bytes"):
                analyse(text)
            policy.assert_not_called()

    def test_independent_lru_disagreement_rejects_before_policy_construction(self):
        with mock.patch.object(neighbor.adaptive.ProtectedLru, "visit", return_value=[]) , \
             mock.patch.object(neighbor, "CausalNeighbor") as policy:
            with self.assertRaisesRegex(ValueError, "exact saved row field"):
                analyse()
            policy.assert_not_called()

    def test_cold_topics_reset_histories_and_source_generation_cannot_alias(self):
        before = analyse()
        analyse(trace(((6, 7),) * 12)[0])
        after = analyse()
        self.assertEqual(before, after)
        self.assertTrue(all(n == 12 for n in after["neighbor"]["layer_rows"].values()))
        other = metadata(); other["model_files"][0]["mtime_ns"] += 1
        second = analyse(data=other)
        self.assertNotEqual(before["baseline"]["source_identity"], second["baseline"]["source_identity"])
        self.assertEqual(before["policies"], second["policies"])

    def test_missing_seed_mutations_interleaved_duplicate_failed_rows_reject(self):
        text = trace()[0]
        bads = [text.replace("[HOSTMOE-ROUTE-CREATE] layer=1 slots=3 experts=8\n", "", 1),
                text.replace("slots=3 experts=8", "slots=4 experts=8", 1),
                text.replace("p0=34 ", "p0=33 ", 1), text.replace("outcome=completed", "outcome=failed", 1),
                text.replace("ids=0,1", "ids=0,0", 1), text + "\n[HOSTMOE-ROUTE] layer=1 ids=0,1"]
        marker = "[HOSTMOE-ROUTE] layer=1 ids=2,3"
        for event in ("[HOSTMOE-VMM-PRESSURE] profile=1", "[HOSTMOE-PREFETCH] issued=1",
                      "[HOSTMOE-REBALANCE] layer=1", "[HOSTMOE-ROUTE-CREATE] layer=1 slots=3 experts=8"):
            bads.append(text.replace(marker, event + "\n" + marker, 1))
        for bad in bads:
            with self.subTest(bad=bad[:80]), self.assertRaises(ValueError):
                analyse(bad)

    def test_pinned_windows_have_no_parameter_or_topic_selection(self):
        self.assertEqual((neighbor.HISTORY_ROWS, neighbor.SUFFIX_ROWS, neighbor.MIN_INTERSECTION), (64, 8, 2))
        a, b = neighbor.PROFILES["v44"]["windows"]
        self.assertEqual((a["begin_line"], a["end_line"], b["begin_line"], b["end_line"]), (637, 33675, 33786, 66824))
        self.assertEqual((a["count"], b["count"]), (384, 384))
        self.assertEqual(a["capacities"], {3: 291, 4: 729, 11: 85})
        self.assertEqual(b["capacities"], {3: 283, 4: 708, 11: 83})
        self.assertEqual((a["expected_misses"], b["expected_misses"]), (58824, 67147))
        self.assertEqual((a["expected_bytes"], b["expected_bytes"]), (408902172672, 466874728448))
        self.assertEqual(b["role"], "second_topic_fixed_policy_holdout")
        self.assertEqual(neighbor.MAX_TRACE_BYTES, 64 << 20)

    def test_cli_dependency_and_trace_provenance_refuse_before_outputs(self):
        with mock.patch.dict(neighbor.DEPENDENCIES, {"glm_gpu_adaptive_replay.py": "0" * 64}), \
             mock.patch.object(neighbor.cohort, "_read_bounded") as read:
            with self.assertRaisesRegex(ValueError, "dependency"):
                neighbor.main(["--profile", "v27", "--trace", "never-read.log", "--metadata", "never-read.json", "--output", "never-write.json"])
            read.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "trace.log").write_text(trace()[0], encoding="utf-8")
            (path / "metadata.json").write_text(json.dumps(metadata()), encoding="utf-8")
            with self.assertRaises(ValueError):
                neighbor.main(["--profile", "v44", "--trace", str(path / "trace.log"), "--metadata", str(path / "metadata.json"), "--output", str(path / "out.json")])
            self.assertFalse((path / "out.json").exists())

    def test_cli_fixed_primary_signed_results_and_exact_pinned_window_recipes(self):
        specs = neighbor.PROFILES["v44"]["windows"]
        def fake_analysis(text, data, **options):
            index = 0 if options["begin_line"] == 637 else 1
            return dict(baseline=dict(gpu_misses=specs[index]["expected_misses"],
                        logical_source_bytes=specs[index]["expected_bytes"]),
                primary_policy=neighbor.POLICY, neighbor=dict(net_logical_bytes_saved=-100))
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(neighbor.cohort, "_read_bounded", side_effect=[b"not-parsed", b"{}"]) as read, \
             mock.patch.object(neighbor, "analyse", side_effect=fake_analysis) as analysis, \
             contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "out.json"
            neighbor.main(["--profile", "v44", "--trace", "saved.log", "--metadata", "saved.json", "--output", str(output)])
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(analysis.call_count, 2)
            self.assertEqual(read.call_args_list[0].args[1:], (64 << 20, neighbor.PROFILES["v44"]["trace_sha256"]))
            self.assertEqual(result["primary_policy"], neighbor.POLICY)
            self.assertEqual(result["primary_parameters"], dict(history=64, suffix=8, minimum_intersection=2))
            self.assertFalse(result["policy_selected_from_results"])
            self.assertFalse(result["provenance"]["checkpoint_opened"])
            self.assertEqual([r["name"] for r in result["chronological_windows"]], ["decode_a", "decode_b"])
            self.assertEqual([r["neighbor"]["net_logical_bytes_saved"] for r in result["chronological_windows"]], [-100, -100])
            self.assertEqual(result["chronological_windows"][1]["topic_role"], "second_topic_fixed_policy_holdout")


if __name__ == "__main__":
    unittest.main()
