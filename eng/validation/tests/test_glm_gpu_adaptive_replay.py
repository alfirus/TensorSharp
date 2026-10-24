"""Portable exact protected-row policy references; no checkpoint, payload or GPU."""
from functools import lru_cache
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_gpu_adaptive_replay as replay


SOURCE = "immutable-source-generation-a"


class ListLru:
    """Independent oldest-first key list, no slots, ages or production Cache."""
    def __init__(self, capacity, reverse=False):
        self.capacity, self.reverse, self.keys = capacity, reverse, []

    def visit(self, source, layer, ids):
        keys = [(source, layer, expert) for expert in ids]
        protected, misses = set(keys), []
        for key in keys:
            if key not in self.keys:
                if len(self.keys) == self.capacity:
                    self.keys.remove(next(old for old in self.keys if old not in protected))
                self.keys.append(key)
                misses.append(key)
        for key in reversed(keys) if self.reverse else keys:
            self.keys.remove(key)
            self.keys.append(key)
        return misses


def exact_optimum(capacity, rows):
    """Whole-row DP: protect all hits, admit every current miss, no prefetch.

    Unlike scalar Belady, legal final resident sets contain the ENTIRE current
    union. The only optional survivors are residents of the preceding row.
    """
    if (type(capacity) is not int or not 1 <= capacity <= 4 or not 1 <= len(rows) <= 8
            or len({(source, layer, expert) for source, layer, ids in rows for expert in ids}) > 6):
        raise ValueError("Exponential test oracle is restricted to tiny problems")
    selected = tuple(frozenset((source, layer, expert) for expert in ids)
                     for source, layer, ids in rows)

    @lru_cache(None)
    def solve(index, resident):
        if index == len(selected):
            return 0
        old, current = set(resident), selected[index]
        missing = len(current - old)
        universe = old | current
        keep = min(capacity, len(universe)) - len(current)
        candidates = old - current
        choices = (tuple(sorted(set(extra) | current))
                   for extra in itertools.combinations(sorted(candidates), keep))
        return missing + min(solve(index + 1, choice) for choice in choices)

    return solve(0, ())


def metadata():
    groups = []
    for group, layers, hidden, ff in ((1, [1, 2], 2, 2), (3, [3], 2, 4), (4, [4], 4, 4)):
        groups.append(dict(id=group, layers=layers, expert_bytes=hidden * ff * 4 * 3,
            projections=[dict(type=0, shape=shape, expert_bytes=hidden * ff * 4)
                         for shape in ([hidden, ff, 8], [hidden, ff, 8], [ff, hidden, 8])]))
    return dict(groups=groups,
        model_files=[dict(path="never-opened.gguf", bytes=12345, mtime_ns=1)])


def forward(position):
    fields = dict(kind=0, rows=1, p0=position, nout=1, wanth=0, logits=1, reused=0,
        outcome="completed", acquire_ms="0", input_ms="0", compute_ms="1", output_ms="0",
        total_ms="1", scope=replay.cohort.retention._FORWARD_SCOPE)
    return replay.cohort.retention._FORWARD + " " + " ".join(f"{k}={v}" for k, v in fields.items())


def source_trace(demands=None, capacities=None):
    """Generated route/timer proof comes from the independent list reference."""
    capacities = {1: 3, 3: 3, 4: 3} if capacities is None else capacities
    if demands is None:
        demands = [dict.fromkeys((1, 2, 3, 4), pair) for pair in
                   ((0, 1), (0, 2), (0, 1), (3, 4), (3, 4), (0, 1))]
    data = metadata()
    lookup = {layer: group for group in data["groups"] for layer in group["layers"]}
    caches = {group: ListLru(size) for group, size in capacities.items()}
    lines, expected = [], []
    for index, demand in enumerate(demands):
        for layer in sorted(lookup):
            group = lookup[layer]
            if not index and layer == group["id"]:
                lines.append(f"[HOSTMOE-ROUTE-CREATE] layer={layer} slots={capacities[group['id']]} experts=8")
            ids = demand[layer]
            misses = caches[group["id"]].visit(SOURCE, layer, ids)
            size = len(misses) * group["expert_bytes"]
            expected.append((33 + index, layer, size))
            lines.append(f"[HOSTMOE-ROUTE] layer={layer} ids=" + ",".join(map(str, ids)))
            lines.append(f"[HOSTMOE-CACHE-TIMING] layer={layer} rows=1 joint=0 bulk=0 "
                         f"outcome=completed file_bytes={size} scope=nested_host_wall_not_cuda_kernel_or_physical_io")
        lines.append(forward(33 + index))
    return "\n".join(lines), expected


def analyse(text=None, data=None, **changes):
    options = dict(capacities={1: 3, 3: 3, 4: 3}, start_position=33,
                   count=6, top_k=2, begin_line=1)
    options.update(changes)
    return replay.analyse(source_trace()[0] if text is None else text,
                          metadata() if data is None else data, **options)


class ProtectedPolicyTests(unittest.TestCase):
    def test_capacity_and_whole_row_validation_are_strict_and_nonmutating(self):
        for cls in (replay.ProtectedLru, replay.ReverseTouchLru, replay.ProtectedArc):
            for bad in (0, -1, True, 1.0, replay.MAX_CAPACITY + 1):
                with self.subTest(cls=cls.__name__, capacity=bad), self.assertRaises(ValueError):
                    cls(bad)
            cache = cls(2)
            cache.visit(SOURCE, 1, [0])
            before = cache.snapshot()
            for args in (("", 1, [1]), (SOURCE, True, [1]), (SOURCE, -1, [1]),
                         (SOURCE, 1, []), (SOURCE, 1, [1, True]), (SOURCE, 1, [1, -1]),
                         (SOURCE, 1, [1, 65536]), (SOURCE, 1, [0, 1, 2]),
                         (SOURCE, 1, [0] * (replay.MAX_ROW + 1))):
                with self.subTest(cls=cls.__name__, args=args), self.assertRaises(ValueError):
                    cache.visit(*args)
                self.assertEqual(cache.snapshot(), before)

    def test_original_lru_matches_independent_list_oracle_and_native_cache(self):
        native = replay.cohort.retention._module("glm-route-cache-replay").Cache
        for capacity in (2, 3, 5):
            cache, oracle, prior = replay.ProtectedLru(capacity), ListLru(capacity), native(capacity, False)
            for step in range(100):
                source, layer = (SOURCE if step % 4 else "generation-b"), 1 + step % 3
                ids = [(step * 7) % 9, (step * 7 + 2) % 9]
                if step % 3 == 0:
                    ids.insert(1, ids[0])  # legal helper duplicate, no extra admission
                expected = oracle.visit(source, layer, ids)
                self.assertEqual(cache.visit(source, layer, ids), expected)
                old = prior.visit_with_evictions((source, layer), ids)
                self.assertEqual([((key[0][0], key[0][1], key[1])) for key, _ in old], expected)
                ordered = [cache.slots[i] for i in sorted(range(capacity), key=lambda i: cache.ages[i])
                           if cache.slots[i] is not None]
                self.assertEqual(ordered, oracle.keys)

    def test_all_original_hits_protected_before_early_miss(self):
        for cls in (replay.ProtectedLru, replay.ReverseTouchLru, replay.ProtectedArc):
            cache = cls(2)
            cache.visit(SOURCE, 1, [0, 1])
            self.assertEqual(cache.visit(SOURCE, 1, [2, 0]), [(SOURCE, 1, 2)])
            self.assertEqual(cache.resident, {(SOURCE, 1, 0), (SOURCE, 1, 2)})

    def test_cold_duplicates_admit_once_without_frequency_promotion(self):
        cache = replay.ProtectedArc(3)
        self.assertEqual(cache.visit(SOURCE, 1, [0, 0, 1]), [(SOURCE, 1, 0), (SOURCE, 1, 1)])
        self.assertEqual(tuple(cache.t1), ((SOURCE, 1, 0), (SOURCE, 1, 1)))
        self.assertFalse(cache.t2)
        self.assertEqual(cache.stats["resident_hits"], 0)
        cache.visit(SOURCE, 1, [1, 1, 0])
        self.assertFalse(cache.t1)
        self.assertEqual(tuple(cache.t2), ((SOURCE, 1, 1), (SOURCE, 1, 0)))
        self.assertEqual(cache.stats["resident_hits"], 2)

    def test_full_source_namespace_and_layer_are_distinct(self):
        for cls in (replay.ProtectedLru, replay.ReverseTouchLru, replay.ProtectedArc):
            cache = cls(3)
            for source, layer in ((SOURCE, 1), (SOURCE, 2), ("generation-b", 1)):
                self.assertEqual(cache.visit(source, layer, [0]), [(source, layer, 0)])
            self.assertEqual(len(cache.resident), 3)
            self.assertEqual(cache.visit(SOURCE, 1, [0]), [])

    def test_ghost_feedback_adapts_both_directions_using_past_only(self):
        cache = replay.ProtectedArc(2)
        for ids in ([0], [1], [0], [2]):
            cache.visit(SOURCE, 1, ids)
        self.assertIn((SOURCE, 1, 1), cache.b1)
        cache.visit(SOURCE, 1, [1])
        self.assertEqual(cache.target, 1)
        self.assertIn((SOURCE, 1, 0), cache.b2)
        cache.visit(SOURCE, 1, [0])
        self.assertEqual(cache.target, 0)
        self.assertEqual((cache.stats["b1_hits"], cache.stats["b2_hits"]), (1, 1))

    def test_ghost_ratio_uses_pre_removal_sizes_and_explicit_floor(self):
        cache = replay.ProtectedArc(8)
        key = lambda n: (SOURCE, 1, n)
        # A legal disjoint directory with two B1 and five B2 entries. At a B1
        # hit floor(5/2)==2; using post-removal B1 would incorrectly yield5.
        cache.t1[key(0)] = None
        cache.b1.update((key(n), None) for n in (1, 2))
        cache.b2.update((key(n), None) for n in (3, 4, 5, 6, 7))
        cache._check()
        cache.visit(SOURCE, 1, [1])
        self.assertEqual(cache.target, 2)
        self.assertEqual(cache.stats["b1_hits"], 1)

    def test_protected_preferred_list_falls_back_without_evicting_current_hit(self):
        cache = replay.ProtectedArc(2)
        for ids in ([0], [1], [0], [2]):
            cache.visit(SOURCE, 1, ids)
        self.assertEqual(cache.visit(SOURCE, 1, [3, 2]), [(SOURCE, 1, 3)])
        self.assertEqual(cache.resident, {(SOURCE, 1, 2), (SOURCE, 1, 3)})
        self.assertGreater(cache.stats["replacement_list_fallbacks"], 0)

    def test_bounded_four_lists_and_every_selected_row_always_resident(self):
        for capacity in (2, 3, 7):
            cache = replay.ProtectedArc(capacity)
            for step in range(300):
                args = (SOURCE, 1 + step % 3, [(step * 3) % 13, (step * 5 + 1) % 13])
                cache.visit(*args)
                _, *lists = cache.snapshot()
                flattened = sum((list(items) for items in lists), [])
                self.assertEqual(len(flattened), len(set(flattened)))
                self.assertLessEqual(len(lists[0]) + len(lists[1]), capacity)
                self.assertLessEqual(len(lists[2]) + len(lists[3]), capacity)
                self.assertLessEqual(len(flattened), 2 * capacity)
                self.assertTrue(set(replay.demand_keys(*args, capacity)).issubset(cache.resident))

    def test_arc_prefix_is_unaffected_by_different_future_demands(self):
        prefix = [[0, 1], [0, 2], [3, 0], [2, 3], [0, 1]]
        suffixes = ([[4, 5]] * 5, [[0, 1]] * 5)
        observed = []
        for suffix in suffixes:
            cache, snapshots = replay.ProtectedArc(3), []
            for ids in prefix + suffix:
                cache.visit(SOURCE, 1, ids)
                snapshots.append(cache.snapshot())
            observed.append(snapshots[:len(prefix)])
        self.assertEqual(observed[0], observed[1])

    def test_arc_can_lose_against_lru_without_hidden_policy_selection(self):
        rows = [(SOURCE, 1, [expert]) for expert in (0, 1, 0, 2, 1, 3, 1)]
        lru, arc = replay.ProtectedLru(2), replay.ProtectedArc(2)
        lru_misses = sum(len(lru.visit(*row)) for row in rows)
        arc_misses = sum(len(arc.visit(*row)) for row in rows)
        self.assertEqual((lru_misses, arc_misses), (5, 6))
        self.assertLess(lru_misses - arc_misses, 0)

    def test_reverse_touch_only_changes_final_age_updates(self):
        ordinary, reverse = replay.ProtectedLru(3), replay.ReverseTouchLru(3)
        self.assertEqual(ordinary.visit(SOURCE, 1, [0, 1]), reverse.visit(SOURCE, 1, [0, 1]))
        self.assertEqual(ordinary.slots, reverse.slots)
        self.assertEqual(ordinary.clock, reverse.clock)
        self.assertLess(ordinary.ages[0], ordinary.ages[1])
        self.assertGreater(reverse.ages[0], reverse.ages[1])
        ordinary.visit(SOURCE, 1, [2]); reverse.visit(SOURCE, 1, [2])
        self.assertEqual(ordinary.visit(SOURCE, 1, [3]), reverse.visit(SOURCE, 1, [3]))
        self.assertNotEqual(ordinary.resident, reverse.resident)
        self.assertIn((SOURCE, 1, 0), reverse.resident)
        self.assertNotIn((SOURCE, 1, 0), ordinary.resident)

    def test_reverse_touch_matches_independent_list_with_duplicates(self):
        cache, oracle = replay.ReverseTouchLru(4), ListLru(4, reverse=True)
        for step in range(100):
            args = (SOURCE, 1 + step % 2, [step % 7, (step + 2) % 7, step % 7])
            self.assertEqual(cache.visit(*args), oracle.visit(*args))
            actual = [cache.slots[i] for i in sorted(range(4), key=lambda i: cache.ages[i])
                      if cache.slots[i] is not None]
            self.assertEqual(actual, oracle.keys)


class FutureDiagnosticTests(unittest.TestCase):
    def test_exponential_test_oracle_rejects_large_inputs(self):
        for capacity, rows in ((5, [(SOURCE, 1, [0])]), (2, [(SOURCE, 1, [0])] * 9),
                               (2, [(SOURCE, 1, [n]) for n in range(7)])):
            with self.assertRaises(ValueError):
                exact_optimum(capacity, rows)

    def test_belady_matches_independent_whole_row_optimum_exhaustively(self):
        options = [(value,) for value in range(3)] + list(itertools.combinations(range(3), 2))
        for capacity in (2, 3):
            for horizon in (1, 2, 3, 4):
                for sequence in itertools.product(options, repeat=horizon):
                    rows = [(SOURCE, 1, ids) for ids in sequence]
                    cache = replay.FutureBelady(capacity, rows)
                    actual = sum(len(cache.visit(*row)) for row in rows)
                    self.assertEqual(actual, exact_optimum(capacity, rows), (capacity, sequence))
                    self.assertLessEqual(cache.peak, capacity)

    def test_bound_protects_entire_row_not_scalar_accesses(self):
        rows = [(SOURCE, 1, ids) for ids in ([0, 1], [2, 0], [1, 2], [3, 1], [0, 1])]
        cache = replay.FutureBelady(2, rows)
        count = 0
        for row in rows:
            count += len(cache.visit(*row))
            self.assertEqual(cache.resident, set(replay.demand_keys(*row, 2)))
            self.assertLessEqual(len(cache.heap), 4 * cache.capacity)
        self.assertEqual(count, exact_optimum(2, rows))

    def test_source_keys_duplicates_and_future_bound_vs_online_policies(self):
        rows = [(SOURCE if index % 4 else "generation-b", 1 + index % 3,
                 [index % 5, (index + 1) % 5, index % 5]) for index in range(40)]
        cache = replay.FutureBelady(4, rows)
        best = sum(len(cache.visit(*row)) for row in rows)
        for cls in (replay.ProtectedLru, replay.ReverseTouchLru, replay.ProtectedArc):
            online = cls(4)
            self.assertLessEqual(best, sum(len(online.visit(*row)) for row in rows))

    def test_invalid_or_mismatched_future_order_does_not_advance(self):
        rows = [(SOURCE, 1, [0, 1]), (SOURCE, 1, [1, 2])]
        cache = replay.FutureBelady(2, rows)
        for bad in ((SOURCE, 1, [1, 0]), (SOURCE, 2, [0, 1]), (SOURCE, 1, [0, 1, 2])):
            with self.assertRaises(ValueError):
                cache.visit(*bad)
            self.assertEqual(cache.index, 0)
            self.assertFalse(cache.resident)
            self.assertEqual(cache.future[(SOURCE, 1, 0)][0], 0)
        cache.visit(*rows[0])
        cache.visit(*rows[1])
        with self.assertRaises(ValueError):
            cache.visit(*rows[1])
        for capacity, future in ((0, rows), (True, rows), (2, []), (1, rows)):
            with self.assertRaises(ValueError):
                replay.FutureBelady(capacity, future)


class CapturedTraceProofTests(unittest.TestCase):
    def test_generated_cold_trace_matches_every_heterogeneous_layer_byte(self):
        text, expected = source_trace()
        result = analyse(text)
        self.assertEqual(result["baseline"]["layer_timer_checks"], len(expected))
        self.assertEqual(result["policies"]["lru"]["logical_source_bytes"], sum(x[2] for x in expected))
        self.assertTrue(result["independent_lru_matched_every_row"])
        sizes = {group["id"]: group["expert_bytes"] for group in metadata()["groups"]}
        for group, size in sizes.items():
            self.assertEqual(result["policies"]["lru"]["per_group"][group] % size, 0)
        lower = result["policies"]["belady_future_diagnostic"]["logical_source_bytes"]
        for name in ("lru", "reverse_touch_lru", "arc_past_only"):
            self.assertLessEqual(lower, result["policies"][name]["logical_source_bytes"])
        self.assertEqual(result["arc_net_logical_bytes_saved"],
            result["policies"]["lru"]["logical_source_bytes"] - result["policies"]["arc_past_only"]["logical_source_bytes"])
        self.assertEqual(result["reverse_touch_net_logical_bytes_saved"],
            result["policies"]["lru"]["logical_source_bytes"] - result["policies"]["reverse_touch_lru"]["logical_source_bytes"])
        for flag in ("arc_uses_future", "reverse_touch_uses_future", "native_implementation_qualified",
                     "metadata_physical_admission_qualified", "physical_io_qualified",
                     "performance_qualified", "numerical_quality_qualified"):
            self.assertFalse(result[flag])
        self.assertTrue(result["belady_uses_future"])

    def test_equal_opposite_layer_errors_fail_even_when_total_unchanged(self):
        text = source_trace()[0]
        bad = text.replace("file_bytes=96", "file_bytes=95", 1)
        bad = bad.replace("layer=2 rows=1 joint=0 bulk=0 outcome=completed file_bytes=96",
                          "layer=2 rows=1 joint=0 bulk=0 outcome=completed file_bytes=97", 1)
        with self.assertRaisesRegex(ValueError, "miss bytes"):
            analyse(bad)

    def test_missing_cold_seed_or_later_cache_mutation_is_not_silently_modelled(self):
        text = source_trace()[0]
        variants = [text.replace("[HOSTMOE-ROUTE-CREATE] layer=1 slots=3 experts=8\n", "", 1),
                    text.replace("slots=3 experts=8", "slots=4 experts=8", 1)]
        marker = "[HOSTMOE-ROUTE] layer=1 ids=0,2"
        for event in ("[HOSTMOE-REBALANCE] layer=1", "[HOSTMOE-VMM-PRESSURE] profile=1",
                      "[HOSTMOE-ROUTE-CREATE] layer=1 slots=3 experts=8", "[HOSTMOE-PREFETCH] issued=1"):
            variants.append(text.replace(marker, event + "\n" + marker, 1))
        for bad in variants:
            with self.subTest(bad=bad[:80]), self.assertRaises(ValueError):
                analyse(bad)

    def test_postcall_route_positions_and_entire_layer_sequence_fail_closed(self):
        text = source_trace()[0]
        variants = (text.replace("p0=34 ", "p0=33 ", 1),
                    text.replace("ids=0,1", "ids=0,0", 1),
                    text.replace("outcome=completed", "outcome=failed", 1),
                    text + "\n[HOSTMOE-ROUTE] layer=1 ids=0,1",
                    text.replace("[HOSTMOE-ROUTE] layer=2 ids=0,1\n", "", 1))
        for bad in variants:
            with self.subTest(bad=bad[:80]), self.assertRaises(ValueError):
                analyse(bad)
        with self.assertRaises(ValueError):
            analyse(capacities={1: 3, 3: 3})

    def test_metadata_source_generation_changes_namespace_without_changing_order(self):
        first = analyse()
        other = metadata(); other["model_files"][0]["mtime_ns"] += 1
        second = analyse(data=other)
        self.assertNotEqual(first["baseline"]["source_identity"], second["baseline"]["source_identity"])
        for policy in first["policies"]:
            self.assertEqual(first["policies"][policy], second["policies"][policy])

    def test_cli_refuses_unpinned_input_before_output_and_never_opens_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "trace.log").write_text(source_trace()[0], encoding="utf-8")
            (path / "metadata.json").write_text(json.dumps(metadata()), encoding="utf-8")
            with self.assertRaises(ValueError):
                replay.main(["--trace", str(path / "trace.log"), "--metadata", str(path / "metadata.json"),
                             "--output", str(path / "result.json")])
            self.assertFalse((path / "result.json").exists())

    def test_cli_verifies_dependency_hashes_before_capture_loading(self):
        with mock.patch.dict(replay.DEPENDENCIES, {"glm_ram_miss_cohort_replay.py": "0" * 64}), \
             mock.patch.object(replay.cohort, "_read_bounded") as read:
            with self.assertRaisesRegex(ValueError, "dependency"):
                replay.main(["--trace", "never-read.log", "--metadata", "never-read.json", "--output", "never-write.json"])
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
