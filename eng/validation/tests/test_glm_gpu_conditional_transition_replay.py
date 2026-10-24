"""Generated tiny cases only; no trace, model, device or payload is opened.

The reference rebuilds transition counts from full past row lists and uses
Fraction directly. It does not share the production ring, count adjustment,
LCM arithmetic, score accessor or victim-selection helper.
"""
from collections import Counter
from fractions import Fraction
from itertools import product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_gpu_conditional_transition_replay as replay


A = ("source-a", 3)
B = ("source-b", 28)


class FullPastReference:
    def __init__(self, capacity, limits, order, window=64):
        self.capacity, self.limits, self.order, self.window = capacity, dict(limits), tuple(order), window
        self.pool = tuple(key for key in order if key in limits)
        self.past = {key: [] for key in limits}
        self.maps = dict.fromkeys(limits)
        self.slots, self.ages = [None] * capacity, [0] * capacity
        self.clock = self.events = 0

    def counts(self, identity):
        rows = self.past[identity]
        pairs = list(zip(rows, rows[1:]))[-self.window:]
        matrix, denominator = Counter(), Counter()
        for previous, following in pairs:
            for a in previous:
                denominator[a] += 1
                for b in following:
                    matrix[a, b] += 1
        return matrix, denominator

    def score(self, key, now):
        saved = self.maps[key[:2]]
        if saved is None or saved[0] <= now:
            return Fraction(0)
        return saved[1][key[2]]

    def visit(self, source, layer, ids):
        identity = source, layer
        assert identity == self.pool[self.events % len(self.pool)]
        cycle = self.events // len(self.pool)
        now = len(self.order) * cycle + self.order.index(identity)
        current = tuple(sorted(set(ids)))
        matrix, denominator = self.counts(identity)
        known = [a for a in current if denominator[a]]
        scores = tuple(sum((Fraction(matrix[a, b], denominator[a]) for a in known), Fraction(0))
                       / len(known) if known else Fraction(0)
                       for b in range(self.limits[identity]))
        self.maps[identity] = (len(self.order) * (cycle + 1) + self.order.index(identity), scores)
        keys = [(source, layer, expert) for expert in ids]
        protected = {i for i, key in enumerate(self.slots) if key in keys}
        missing = []
        for key in keys:
            if key in self.slots:
                continue
            empty = [i for i, occupant in enumerate(self.slots) if occupant is None]
            slot = empty[0] if empty else min(
                (i for i in range(self.capacity) if i not in protected),
                key=lambda i: (self.score(self.slots[i], now), self.ages[i], i))
            self.slots[slot] = key
            protected.add(slot)
            missing.append(key)
        for key in keys:
            self.clock += 1
            self.ages[self.slots.index(key)] = self.clock
        self.past[identity].append(current)
        self.events += 1
        return missing


class ListLruReference:
    """Independent oldest-to-newest list, without slot/age implementation."""
    def __init__(self, capacity):
        self.capacity, self.keys = capacity, []

    def visit(self, source, layer, ids):
        selected = [(source, layer, expert) for expert in ids]
        protected = set(selected)
        misses = []
        for key in selected:
            if key in self.keys:
                continue
            if len(self.keys) == self.capacity:
                victim = next(candidate for candidate in self.keys if candidate not in protected)
                self.keys.remove(victim)
            self.keys.append(key)
            misses.append(key)
        for key in selected:
            self.keys.remove(key)
            self.keys.append(key)
        return misses


class ConditionalTransitionTests(unittest.TestCase):
    def make(self, capacity=4, experts=6, window=64):
        return replay.TransitionCache(capacity, {A: experts}, (A,), window=window)

    def visit(self, cache, row, identity=A):
        return cache.visit(identity[0], identity[1], list(row))

    def fraction(self, cache, expert, now, identity=A):
        return Fraction(*cache.score((*identity, expert), now))

    def check_reference(self, cache, reference):
        self.assertEqual(cache.slots, reference.slots)
        self.assertEqual(cache.ages, reference.ages)
        self.assertEqual((cache.clock, cache.events), (reference.clock, reference.events))
        self.assertEqual(cache.resident, set(reference.slots) - {None})
        self.assertEqual(len(cache.resident), len(cache.slots) - cache.slots.count(None))
        for identity, experts in reference.limits.items():
            matrix, denominator = reference.counts(identity)
            self.assertEqual(list(cache.matrices[identity]),
                             [matrix[a, b] for a in range(experts) for b in range(experts)])
            self.assertEqual(list(cache.denominators[identity]), [denominator[a] for a in range(experts)])
            self.assertEqual(list(cache.rings[identity]), reference.past[identity][-(reference.window + 1):])
            self.assertEqual(cache.observations[identity], len(reference.past[identity]))
            saved = reference.maps[identity]
            if saved is None:
                self.assertIsNone(cache.predictions[identity])
            else:
                self.assertEqual(cache.predictions[identity]["due"], saved[0])
                for b in range(experts):
                    self.assertEqual(self.fraction(cache, b, saved[0] - 1, identity), saved[1][b])
                    self.assertEqual(cache.score((*identity, b), saved[0]), (0, 1))

    def test_outgoing_denominator_counts_rows_not_destinations(self):
        cache = self.make(capacity=8, experts=10)
        self.visit(cache, range(8))
        self.visit(cache, range(8))
        self.assertEqual(list(cache.denominators[A]), [1] * 8 + [0, 0])
        for a in range(10):
            for b in range(10):
                self.assertEqual(cache.matrices[A][a * 10 + b], int(a < 8 and b < 8))
        self.assertEqual(cache.stats["denominator_increments"], 8)
        self.assertEqual(cache.stats["pair_increments"], 64)

    def test_duplicate_ids_count_once_but_touch_each_occurrence(self):
        cache = self.make(capacity=4, experts=4)
        self.visit(cache, (0, 0, 1))
        self.visit(cache, (2, 2, 3))
        self.assertEqual(list(cache.rings[A]), [(0, 1), (2, 3)])
        self.assertEqual(list(cache.denominators[A]), [1, 1, 0, 0])
        self.assertEqual(sum(cache.matrices[A]), 4)
        self.assertEqual(cache.clock, 6)
        self.assertEqual(cache.stats["planned_duplicate_occurrences"], 2)

    def test_mean_uses_known_antecedents_and_exact_fraction(self):
        cache = self.make(capacity=6, experts=6)
        for row in ((0,), (2,), (0, 1), (3,), (0,), (2,)):
            self.visit(cache, row)
        before = cache.stats["known_antecedents"]
        self.visit(cache, (0, 1, 5))
        self.assertEqual(cache.stats["known_antecedents"] - before, 2)
        self.assertEqual(self.fraction(cache, 2, 6), Fraction(1, 3))
        self.assertEqual(self.fraction(cache, 3, 6), Fraction(2, 3))
        self.assertEqual(cache.predictions[A]["denominator"], 6)
        self.assertEqual(cache.predictions[A]["scores"][2:4], (2, 4))

    def test_multilabel_membership_scores_need_not_sum_to_one(self):
        cache = self.make(capacity=8, experts=10)
        for _ in range(3):
            self.visit(cache, range(8))
        values = [self.fraction(cache, b, 2) for b in range(10)]
        self.assertEqual(values, [Fraction(1)] * 8 + [Fraction(0)] * 2)
        self.assertEqual(sum(values), 8)

    def test_query_does_not_train_on_current_label(self):
        cache = self.make(experts=4)
        self.visit(cache, (0,))
        self.visit(cache, (1,))
        self.assertEqual(cache.predictions[A]["scores"], (0, 0, 0, 0))
        self.visit(cache, (1,))
        self.assertEqual(self.fraction(cache, 1, 2), 0)
        self.assertEqual(cache.matrices[A][1 * 4 + 1], 1)
        self.visit(cache, (1,))
        self.assertEqual(self.fraction(cache, 1, 3), 1)

    def test_unknown_query_replaces_previous_prediction(self):
        cache = self.make(experts=5)
        for row in ((0,), (1,), (0,)):
            self.visit(cache, row)
        self.assertEqual(self.fraction(cache, 1, 2), 1)
        old_due = cache.predictions[A]["due"]
        self.visit(cache, (4,))
        self.assertGreater(cache.predictions[A]["due"], old_due)
        self.assertEqual(cache.predictions[A]["denominator"], 1)
        self.assertEqual(cache.predictions[A]["scores"], (0, 0, 0, 0, 0))

    def test_deadline_uses_full_42_rank_and_expires_at_equality(self):
        order = tuple(("full", i) for i in range(42))
        first, last = order[3], order[28]
        cache = replay.TransitionCache(6, {first: 4, last: 4}, order)
        for a, b in ((0, 3), (1, 3), (0, 3)):
            self.visit(cache, (a,), first)
            self.visit(cache, (b,), last)
        self.assertEqual(cache.events, 6)
        self.assertEqual(cache.predictions[first]["due"], 129)
        self.assertEqual(self.fraction(cache, 1, 128, first), 1)
        self.assertEqual(cache.score((*first, 1), 129), (0, 1))

    def test_other_layer_query_does_not_rebase_a_saved_deadline(self):
        skipped = ("other-pool", 17)
        cache = replay.TransitionCache(6, {A: 4, B: 4}, (A, skipped, B))
        for row in ((0,), (1,)):
            self.visit(cache, row, A)
            self.visit(cache, (3,), B)
        self.visit(cache, (0,), A)
        saved = dict(cache.predictions[A])
        self.visit(cache, (3,), B)
        self.assertEqual(cache.predictions[A], saved)
        self.assertEqual(saved["due"], 9)
        self.assertEqual(self.fraction(cache, 1, 8), 1)
        self.assertEqual(cache.score((*A, 1), 9), (0, 1))

    def test_source_and_layer_namespaces_keep_independent_labels(self):
        same_layer = ("different-source", A[1])
        same_source = (A[0], A[1] + 1)
        order = (A, same_source, same_layer)
        cache = replay.TransitionCache(6, {key: 4 for key in order}, order)
        for rows in (((0,), (0,), (1,)), ((1,), (2,), (0,)), ((0,), (0,), (1,))):
            for identity, row in zip(order, rows):
                self.visit(cache, row, identity)
        self.assertEqual(self.fraction(cache, 1, 8, A), 1)
        self.assertEqual(self.fraction(cache, 1, 8, same_source), 0)
        self.assertEqual(self.fraction(cache, 2, 8, same_source), 1)
        self.assertEqual(self.fraction(cache, 0, 8, same_layer), 1)
        self.assertEqual(len(cache.resident), len(set(cache.resident)))

    def test_all_unknown_scores_reduce_to_independent_list_lru(self):
        cache = self.make(capacity=5, experts=100)
        lru = ListLruReference(5)
        for i in range(20):
            row = (2 * i, 2 * i + 1)
            self.assertEqual(self.visit(cache, row), lru.visit(*A, row))
            self.assertEqual(cache.resident, set(lru.keys))
            self.assertEqual(cache.predictions[A]["scores"], (0,) * 100)
        self.assertEqual(cache.stats["known_antecedents"], 0)

    def test_protects_all_original_hits_before_first_miss(self):
        cache = self.make(capacity=3, experts=4)
        self.visit(cache, (0, 1, 2))
        self.assertEqual(self.visit(cache, (3, 0)), [(*A, 3)])
        self.assertEqual(cache.resident, {(*A, 0), (*A, 2), (*A, 3)})

    def test_forces_each_unique_miss_and_protects_new_admissions(self):
        cache = self.make(capacity=3, experts=5)
        self.visit(cache, (0, 1, 2))
        self.assertEqual(self.visit(cache, (3, 4, 3)), [(*A, 3), (*A, 4)])
        self.assertEqual(cache.resident, {(*A, 2), (*A, 3), (*A, 4)})
        self.assertEqual(cache.clock, 6)

    def test_original_duplicate_touch_order_preserves_lru_recency(self):
        cache = self.make(capacity=2, experts=4)
        self.visit(cache, (0, 1))
        self.visit(cache, (0, 2, 0))
        self.assertEqual(cache.clock, 5)
        self.assertGreater(cache.ages[cache.slots.index((*A, 0))], cache.ages[cache.slots.index((*A, 2))])
        self.visit(cache, (3,))
        self.assertEqual(cache.resident, {(*A, 0), (*A, 3)})

    def test_equal_nonzero_scores_tie_by_original_lru(self):
        cache = self.make(capacity=3, experts=4)
        for row in ((0,), (1, 2), (0,)):
            self.visit(cache, row)
        self.visit(cache, (0, 3))
        self.assertEqual(self.fraction(cache, 1, 3), 1)
        self.assertEqual(self.fraction(cache, 2, 3), 1)
        self.assertEqual(cache.resident, {(*A, 0), (*A, 2), (*A, 3)})

    def test_empty_and_equal_age_ties_choose_ascending_slot(self):
        cache = self.make(capacity=3, experts=4)
        self.visit(cache, (2, 1, 0))
        self.assertEqual(cache.slots, [(*A, 2), (*A, 1), (*A, 0)])
        # Isolate the final documented slot tie without relying on a real
        # trace naturally generating equal timestamps for two resident keys.
        cache.ages[:] = [3, 3, 3]
        self.visit(cache, (3,))
        self.assertEqual(cache.slots[0], (*A, 3))
        self.assertNotIn((*A, 2), cache.resident)

    def test_window_one_removes_exactly_the_old_transition(self):
        cache = self.make(experts=4, window=1)
        for row in ((0,), (1,), (2,)):
            self.visit(cache, row)
        self.assertEqual(list(cache.rings[A]), [(1,), (2,)])
        self.assertEqual(list(cache.denominators[A]), [0, 1, 0, 0])
        self.assertEqual(sum(cache.matrices[A]), 1)
        self.assertEqual(cache.matrices[A][1 * 4 + 2], 1)
        self.assertEqual(cache.stats["committed_transitions"], 2)
        self.assertEqual(cache.stats["expired_transitions"], 1)
        self.assertEqual(cache.stats["pair_decrements"], 1)
        self.assertEqual(cache.stats["denominator_decrements"], 1)

    def test_full_64_window_has_exact_bounded_counter_storage(self):
        cache = self.make(capacity=8, experts=288)
        for _ in range(65):
            self.visit(cache, range(8))
        self.assertEqual(len(cache.matrices[A]) * cache.matrices[A].itemsize, 288 * 288 * 2)
        self.assertEqual(cache.matrices[A].typecode, "H")
        self.assertEqual(list(cache.denominators[A]), [64] * 8 + [0] * 280)
        self.assertEqual(sum(cache.matrices[A]), 64 * 8 * 8)
        self.assertEqual(max(cache.matrices[A]), 64)
        self.assertEqual(len(cache.rings[A]), 65)
        self.assertEqual(cache.ring_references, 65 * 8)
        self.visit(cache, range(8, 16))
        self.assertEqual(len(cache.rings[A]), 65)
        self.assertEqual(cache.ring_references, 65 * 8)
        self.assertLessEqual(cache.stats["peak_cell"], 64)
        self.assertLessEqual(cache.stats["peak_denominator"], 64)

    def test_repeated_wrap_matches_full_past_without_underflow(self):
        cache = self.make(capacity=4, experts=6, window=3)
        reference = FullPastReference(4, {A: 6}, (A,), 3)
        for i in range(45):
            row = (i % 6, (i * 5 + 1) % 6)
            self.assertEqual(self.visit(cache, row), reference.visit(*A, row))
            self.check_reference(cache, reference)
        self.assertEqual(cache.stats["committed_transitions"], 44)
        self.assertEqual(cache.stats["expired_transitions"], 41)
        self.assertTrue(all(0 <= x <= 3 for x in cache.matrices[A]))
        self.assertTrue(all(0 <= x <= 3 for x in cache.denominators[A]))

    def test_invalid_route_is_atomic_for_all_visible_state(self):
        cache = self.make(capacity=2, experts=4)
        for row in ((0,), (1,), (0,)):
            self.visit(cache, row)
        invalid = (None, (), [], "0", [True], [0.0], [-1], [4], [0, 1, 2], [0] * 9, [0, 4])
        for row in invalid:
            before = cache.snapshot()
            with self.subTest(row=row), self.assertRaises(ValueError):
                cache.visit(*A, row)
            self.assertEqual(cache.snapshot(), before)

    def test_invalid_namespace_or_cycle_is_atomic(self):
        cache = replay.TransitionCache(3, {A: 4, B: 4}, (A, B))
        for identity in (B, ("unknown", A[1]), (A[0], B[1])):
            before = cache.snapshot()
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                self.visit(cache, (0,), identity)
            self.assertEqual(cache.snapshot(), before)
        self.visit(cache, (0,), A)
        before = cache.snapshot()
        with self.assertRaises(ValueError):
            self.visit(cache, (1,), A)
        self.assertEqual(cache.snapshot(), before)
        self.visit(cache, (1,), B)
        self.assertEqual(cache.events, 2)

    def test_constructor_rejects_invalid_geometry_and_allows_spare_slots(self):
        for capacity in (0, -1, True, 1.0, 8193):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                self.make(capacity=capacity)
        for window in (0, 65, True, 1.0):
            with self.subTest(window=window), self.assertRaises(ValueError):
                self.make(window=window)
        for experts in (0, 289, True, 4.0):
            with self.subTest(experts=experts), self.assertRaises(ValueError):
                self.make(experts=experts)
        for limits, order in (({}, (A,)), ({A: 4}, ()), ({A: 4}, (A, A)), ({A: 4}, (B,)),
                              ({("", 3): 4}, (("", 3),)), ({("a", -1): 4}, (("a", -1),))):
            with self.subTest(limits=limits, order=order), self.assertRaises(ValueError):
                replay.TransitionCache(2, limits, order)
        cache = self.make(capacity=10, experts=3)
        self.visit(cache, (0, 1, 2))
        self.assertEqual(cache.slots.count(None), 7)
        self.assertEqual(len(cache.resident), 3)

    def test_score_validation_and_snapshots_do_not_mutate_state(self):
        cache = self.make(experts=4)
        for row in ((0,), (1,), (0,)):
            self.visit(cache, row)
        before = cache.snapshot()
        text = repr(before)
        for key, now in (((*A, -1), 0), ((*A, 4), 0), (("wrong", 3, 0), 0), ((*A, True), 0),
                         ((*A, 1), -1), ((*A, 1), True), ((*A, 1), 1.0)):
            with self.subTest(key=key, now=now), self.assertRaises(ValueError):
                cache.score(key, now)
            self.assertEqual(cache.snapshot(), before)
        self.assertEqual(cache.score((*A, 1), 3), (0, 1))
        self.assertEqual(cache.snapshot(), before)
        self.visit(cache, (2,))
        self.assertEqual(repr(before), text)
        self.assertIsInstance(before, tuple)

    def test_future_suffix_cannot_change_prefix_decisions(self):
        prefix = [(0,), (1,), (0,), (1,), (2,), (0,), (0, 3)]
        left, right = self.make(capacity=3, experts=4), self.make(capacity=3, experts=4)
        reference = FullPastReference(3, {A: 4}, (A,))
        for row in prefix:
            expected = reference.visit(*A, row)
            self.assertEqual(self.visit(left, row), expected)
            self.assertEqual(self.visit(right, row), expected)
            self.assertEqual(left.snapshot(), right.snapshot())
        self.check_reference(left, reference)
        for row in ((2,), (1,), (3,)):
            self.visit(left, row)
        for row in ((3,), (2,), (0,)):
            self.visit(right, row)
        self.assertNotEqual(left.snapshot(), right.snapshot())

    def test_fixed_generated_trace_can_lose_to_lru(self):
        cache = self.make(capacity=3, experts=4)
        lru = ListLruReference(3)
        reference = FullPastReference(3, {A: 4}, (A,))
        policy_misses, lru_misses = [], []
        for row in ((0,), (1,), (0,), (1,), (2,), (0,), (0, 3), (2,)):
            actual = self.visit(cache, row)
            self.assertEqual(actual, reference.visit(*A, row))
            policy_misses.append(len(actual))
            lru_misses.append(len(lru.visit(*A, row)))
        self.assertEqual(policy_misses, [1, 1, 0, 0, 1, 0, 1, 1])
        self.assertEqual(lru_misses, [1, 1, 0, 0, 1, 0, 1, 0])
        self.assertGreater(sum(policy_misses), sum(lru_misses))

    def test_tiny_exhaustive_and_pooled_differential(self):
        alphabet = ((0,), (1,), (2,), (0, 1))
        for length in range(1, 5):
            for trace in product(alphabet, repeat=length):
                for capacity in (2, 3):
                    for window in (1, 2, 64):
                        cache = self.make(capacity=capacity, experts=3, window=window)
                        reference = FullPastReference(capacity, {A: 3}, (A,), window)
                        for row in trace:
                            self.assertEqual(self.visit(cache, row), reference.visit(*A, row),
                                             (trace, capacity, window))
                            self.check_reference(cache, reference)
        order = (A, ("gap", 17), B)
        for window in (1, 3, 64):
            cache = replay.TransitionCache(3, {A: 4, B: 4}, order, window=window)
            reference = FullPastReference(3, {A: 4, B: 4}, order, window)
            for i in range(32):
                identity = A if i % 2 == 0 else B
                row = (i % 4, (i * 3 + 1) % 4)
                self.assertEqual(self.visit(cache, row, identity), reference.visit(*identity, row))
                self.check_reference(cache, reference)


if __name__ == "__main__":
    unittest.main()
