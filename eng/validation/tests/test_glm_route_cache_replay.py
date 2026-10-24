import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('glm_route_cache_replay',
    Path(__file__).parents[1] / 'glm-route-cache-replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class RouteCacheReplayTests(unittest.TestCase):
    def test_route_filter_excludes_startup_and_prefill(self):
        text = ('[HOSTMOE-ROUTE] layer=3 ids=1,2\nREQUEST\n'
                '[HOSTMOE-ROUTE] layer=3 ids=0,1,2,3\n'
                '[HOSTMOE-ROUTE] layer=3 ids=1,2\n')
        rows, skipped = replay.parse_routes(text, top_k=2, after_marker='REQUEST')
        self.assertEqual(rows, [(3, (1, 2))])
        self.assertEqual(skipped, {4: 1})

    def test_duplicate_route_and_absent_marker_fail(self):
        with self.assertRaises(ValueError):
            replay.parse_routes('[HOSTMOE-ROUTE] layer=3 ids=1,1', top_k=2)
        with self.assertRaises(ValueError):
            replay.parse_routes('[HOSTMOE-ROUTE] layer=3 ids=1,2', top_k=2,
                                after_marker='missing')

    def test_source_id_is_part_of_identity(self):
        for lfu in (False, True):
            cache = replay.Cache(2, lfu)
            self.assertEqual(cache.visit(1, [7]), 1)
            self.assertEqual(cache.visit(2, [7]), 1)
            self.assertEqual(cache.visit(1, [7]), 0)
            self.assertEqual(set(cache.slots), {(1, 7), (2, 7)})

    def test_all_hits_protected_before_first_miss(self):
        for lfu in (False, True):
            cache = replay.Cache(2, lfu)
            cache.visit(1, [1, 2])
            self.assertEqual(cache.visit(1, [3, 1]), 1)
            self.assertEqual(set(cache.slots), {(1, 1), (1, 3)})

    def test_frequency_retains_hot_source_but_lru_does_not(self):
        for lfu in (False, True):
            cache = replay.Cache(2, lfu)
            for _ in range(3):
                cache.visit(1, [0])
            cache.visit(2, [0])
            cache.visit(3, [0])
            self.assertEqual((1, 0) in cache.slots, lfu)

    def test_capacity_epoch_halving_and_saturation(self):
        self.assertEqual([replay.capacity_epoch(n) for n in (8, 100, 134, 312, 1024)],
                         [32, 256, 512, 1024, 1024])
        cache = replay.Cache(2, True)
        cache.epoch = 2
        cache.visit(1, [7])
        cache.visit(2, [7])
        self.assertEqual(cache.frequency[(1, 7)], 0)
        self.assertEqual(cache.frequency[(2, 7)], 1)
        cache.frequency[(2, 7)] = 65535
        cache.visit(2, [7])
        self.assertEqual(cache.frequency[(2, 7)], 65535)

    def test_unaffordable_union_fails(self):
        with self.assertRaises(ValueError):
            replay.Cache(1, True).visit(1, [1, 2])

    def test_missed_bytes_use_each_groups_exact_payload(self):
        groups = [dict(id=1, layers=[1], expert_bytes=10,
                       projections=[dict(shape=[2, 2, 4])]),
                  dict(id=2, layers=[2], expert_bytes=30,
                       projections=[dict(shape=[2, 2, 4])])]
        result = replay.replay([(1, [0]), (2, [0]), (1, [0])], groups,
                               {1: 1, 2: 1}, 'lru')
        self.assertEqual(result['misses'], 2)
        self.assertEqual(result['missed_bytes'], 40)

    @staticmethod
    def tiny_groups():
        return [dict(id=1, layers=[1], expert_bytes=10,
                     projections=[dict(shape=[2, 2, 16])])]

    def test_exclusive_promotions_charge_victim_downloads(self):
        rows = [(1, [expert]) for expert in [0, 1, 0, 1]]
        result = replay.replay_host(rows, self.tiny_groups(), {1: 1}, 10, 'exclusive')
        self.assertEqual(result['logical_file_bytes'], 20)
        self.assertEqual(result['host_hit_bytes'], 20)
        self.assertEqual(result['h2d_payload_bytes'], 40)
        self.assertEqual(result['d2h_victim_payload_bytes'], 30)
        self.assertEqual(result['host_retained_bytes'], 10)
        self.assertEqual(result['logical_file_reduction_percent'], 50)

    def test_inclusive_duplicate_gpu_contents_can_defeat_tiny_ram(self):
        rows = [(1, [expert]) for expert in [0, 1, 0, 1]]
        small = replay.replay_host(rows, self.tiny_groups(), {1: 1}, 10, 'inclusive')
        large = replay.replay_host(rows, self.tiny_groups(), {1: 1}, 20, 'inclusive')
        self.assertEqual(small['logical_file_bytes'], 40)
        self.assertEqual(large['logical_file_bytes'], 20)
        self.assertEqual(large['d2h_victim_payload_bytes'], 0)
        self.assertEqual(large['last_half_logical_file_bytes'], 0)
        self.assertEqual(large['last_half_gpu_miss_bytes'], 20)

    def test_zero_ram_matches_original_gpu_replay_without_demotion(self):
        rows = [(1, [expert]) for expert in [0, 1, 0, 1]]
        expected = replay.replay(rows, self.tiny_groups(), {1: 1}, 'lru')['missed_bytes']
        for mode in ['inclusive', 'exclusive']:
            result = replay.replay_host(rows, self.tiny_groups(), {1: 1}, 0, mode)
            self.assertEqual(result['logical_file_bytes'], expected)
            self.assertEqual(result['host_peak_bytes'], 0)
            self.assertEqual(result['d2h_victim_payload_bytes'], 0)

    def test_byte_capacity_and_oversized_entry_bypass(self):
        cache = replay.HostLru(35)
        cache.put((1, 0), 10)
        cache.put((2, 0), 30)
        self.assertEqual(cache.bytes, 30)
        self.assertFalse(cache.get((1, 0)))
        self.assertFalse(cache.put((3, 0), 36))
        self.assertEqual(cache.bytes, 30)
        self.assertTrue(cache.get((2, 0), remove=True))
        self.assertEqual(cache.bytes, 0)
        self.assertEqual(cache.peak_bytes, 30)
        self.assertEqual(cache.evicted_bytes, 10)

    def test_large_scan_displaces_previously_hot_payload(self):
        cache = replay.HostLru(20)
        for _ in range(3):
            cache.put((1, 0), 10)
        cache.put((1, 1), 10)
        cache.put((1, 2), 10)
        self.assertFalse(cache.get((1, 0)))

    def test_eviction_events_preserve_source_and_protect_hits(self):
        cache = replay.Cache(2, False)
        self.assertEqual(cache.visit_with_evictions(1, [1, 2]),
                         [((1, 1), None), ((1, 2), None)])
        self.assertEqual(cache.visit_with_evictions(1, [3, 1]), [((1, 3), (1, 2))])
        self.assertEqual(cache.visit_with_evictions(2, [1]), [((2, 1), (1, 3))])

    def test_host_replay_rejects_invalid_inputs(self):
        for mode, size, rows in [('bad', 10, [(1, [0])]),
                                 ('inclusive', -1, [(1, [0])]),
                                 ('exclusive', 10, [(1, [16])])]:
            with self.assertRaises(ValueError):
                replay.replay_host(rows, self.tiny_groups(), {1: 1}, size, mode)

    def selective(self, ids, policy='residence-hits', threshold=1, window=0, ram=10):
        rows = [(1, [expert]) for expert in ids]
        groups = self.tiny_groups()
        events = replay.gpu_route_events(rows, groups, {1: 1})
        return replay.replay_selective_host(events, groups, ram, policy, threshold, window)

    def test_residence_hits_require_reuse_during_this_gpu_residency(self):
        result = self.selective([0, 0, 1, 0, 1, 1, 0])
        self.assertEqual(result['gpu_miss_bytes'], 50)
        self.assertEqual(result['host_hit_bytes'], 10)
        self.assertEqual(result['logical_file_bytes'], 40)
        self.assertEqual(result['d2h_victim_payload_bytes'], 20)
        self.assertEqual(result['victims_admitted'], 2)
        self.assertEqual(result['victims_rejected'], 2)
        self.assertEqual(result['admission']['peak_resident_keys'], 1)
        higher = self.selective([0, 0, 1, 0, 1, 1, 0], threshold=2)
        self.assertEqual(higher['d2h_victim_payload_bytes'], 0)

    def test_promotion_does_not_inherit_old_residence_hit_count(self):
        result = self.selective([0, 0, 1, 0, 1])
        self.assertEqual(result['host_hit_bytes'], 10)
        self.assertEqual(result['d2h_victim_payload_bytes'], 10)
        self.assertEqual(result['host_retained_bytes'], 0)

    def test_rolling_demand_window_expires_old_reuse_exactly(self):
        short = self.selective([0, 0, 1], 'recent-demands', 2, 2)
        longer = self.selective([0, 0, 1], 'recent-demands', 2, 3)
        self.assertEqual(short['d2h_victim_payload_bytes'], 0)
        self.assertEqual(longer['d2h_victim_payload_bytes'], 10)
        self.assertEqual(short['admission']['peak_history_events'], 2)

    def test_recent_gpu_hits_age_on_misses_without_counting_them(self):
        policy = replay.VictimAdmission('recent-hits', 1, 2)
        policy.observe([(1, 0)], {(1, 0)})
        self.assertEqual(dict(policy.counts), {})
        policy.observe([(1, 0)], set())
        policy.observe([(1, 1)], {(1, 1)})
        self.assertTrue(policy.replace((1, 1), (1, 0)))
        policy.observe([(1, 2)], {(1, 2)})
        self.assertFalse(policy.replace((1, 2), (1, 0)))
        self.assertEqual(list(policy.history), [None, None])
        self.assertEqual(dict(policy.counts), {})

    def test_bounded_history_drops_zero_counters_and_separates_sources(self):
        policy = replay.VictimAdmission('recent-demands', 2, 3)
        policy.observe([(1, 0), (2, 0)], {(1, 0), (2, 0)})
        self.assertFalse(policy.replace((3, 0), (1, 0)))
        for expert in range(100):
            key = (4, expert)
            policy.observe([key], {key})
            self.assertLessEqual(len(policy.history), 3)
            self.assertLessEqual(len(policy.counts), 3)
        self.assertNotIn((1, 0), policy.counts)
        self.assertEqual(policy.peak_history, 3)
        self.assertEqual(policy.peak_keys, 3)

    def test_residence_counter_is_saturating_and_drops_evicted_keys(self):
        policy = replay.VictimAdmission('residence-hits', 2)
        policy.observe([(1, 0)], {(1, 0)})
        policy.replace((1, 0), None)
        for _ in range(100):
            policy.observe([(1, 0)], set())
        self.assertEqual(policy.resident[(1, 0)], 2)
        self.assertTrue(policy.replace((2, 0), (1, 0)))
        self.assertEqual(policy.resident, {(2, 0): 0})

    def test_selective_capacity_refusal_does_not_charge_d2h(self):
        for ram in (0, 9):
            result = self.selective([0, 0, 1, 1, 0], ram=ram)
            self.assertEqual(result['victims_capacity_bypassed'], 2)
            self.assertEqual(result['d2h_victim_payload_bytes'], 0)
            self.assertEqual(result['host_peak_bytes'], 0)
            self.assertEqual(result['host_hit_bytes'], 0)

    def test_admit_all_matches_existing_exclusive_baseline(self):
        groups = [dict(id=1, layers=[1], expert_bytes=10,
                       projections=[dict(shape=[2, 2, 16])]),
                  dict(id=2, layers=[2], expert_bytes=30,
                       projections=[dict(shape=[2, 2, 16])])]
        rows = [(layer, [expert]) for layer, expert in
                [(1, 0), (2, 0), (1, 1), (2, 1), (1, 0), (2, 0)]]
        events = replay.gpu_route_events(rows, groups, {1: 1, 2: 1})
        for ram in (0, 9, 10, 30, 40, 100):
            baseline = replay.replay_host(rows, groups, {1: 1, 2: 1}, ram, 'exclusive')
            result = replay.replay_selective_host(events, groups, ram, 'recent-demands', 1, 100)
            for key in ('host_hit_bytes', 'logical_file_bytes', 'd2h_victim_payload_bytes',
                        'h2d_payload_bytes', 'host_peak_bytes', 'host_retained_bytes',
                        'host_evicted_bytes', 'last_half_logical_file_bytes'):
                self.assertEqual(result[key], baseline[key], (ram, key))
            self.assertLessEqual(result['host_peak_bytes'], ram)
            self.assertEqual(result['gpu_miss_bytes'], result['host_hit_bytes'] + result['logical_file_bytes'])
            self.assertEqual(result['victims_considered'], result['victims_admitted']
                             + result['victims_rejected'] + result['victims_capacity_bypassed'])

    def test_suffix_cannot_change_prefix_events_or_admission(self):
        rows = [(1, [expert]) for expert in [0, 0, 1, 0]]
        groups = self.tiny_groups()
        prefix = replay.gpu_route_events(rows, groups, {1: 1})
        full = replay.gpu_route_events(rows + [(1, [7]), (1, [0])], groups, {1: 1})
        self.assertEqual(prefix, full[:len(prefix)])
        for policy, threshold, window in [('residence-hits', 1, 0), ('recent-demands', 2, 4),
                                          ('recent-hits', 1, 4)]:
            first = replay.replay_selective_host(prefix, groups, 10, policy, threshold, window)
            second = replay.replay_selective_host(full[:len(prefix)], groups, 10, policy, threshold, window)
            self.assertEqual(first, second)
            self.assertEqual(first, replay.replay_selective_host(prefix, groups, 10, policy, threshold, window))

    def test_selective_input_validation(self):
        for policy, threshold, window in [('bad', 1, 0), ('residence-hits', 0, 0),
                                          ('residence-hits', 1, 1), ('recent-demands', 1, 0),
                                          ('recent-hits', 1, -1)]:
            with self.assertRaises(ValueError):
                replay.VictimAdmission(policy, threshold, window)
        for ids in ([16], [-1], [0, 0]):
            with self.assertRaises(ValueError):
                replay.gpu_route_events([(1, ids)], self.tiny_groups(), {1: 1})
        with self.assertRaises(ValueError):
            replay.replay_selective_host([], self.tiny_groups(), -1, 'residence-hits', 1)

    def test_frontier_respects_ram_ceiling_and_two_objectives(self):
        def point(name, capacity, saved, downloaded):
            return dict(name=name, host_capacity_bytes=capacity, host_hit_bytes=saved,
                        d2h_victim_payload_bytes=downloaded)
        runs = [point('a', 4, 10, 20), point('dominated', 4, 9, 20),
                point('more', 4, 12, 30), point('other-capacity', 8, 1, 40),
                point('equal', 4, 10, 20), point('off', 4, 0, 0)]
        self.assertEqual([r['name'] for r in replay.host_pareto_frontier(runs)],
                         ['a', 'more', 'other-capacity', 'equal', 'off'])


if __name__ == '__main__':
    unittest.main()
