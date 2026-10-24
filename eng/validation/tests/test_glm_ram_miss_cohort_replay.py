"""Portable independent cold-trace and train/holdout fixtures; no model/GPU."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_ram_miss_cohort_replay as cohort


def metadata():
    groups = []
    for first, layers, hidden, ff in ((1, [1, 2], 2, 2), (3, [3], 2, 4), (4, [4], 4, 4)):
        shapes = ([hidden, ff, 8], [hidden, ff, 8], [ff, hidden, 8])
        groups.append(dict(id=first, layers=layers, expert_bytes=hidden * ff * 4 * 3,
            projections=[dict(type=0, shape=shape, expert_bytes=hidden * ff * 4) for shape in shapes]))
    return dict(groups=groups, model_files=[dict(path='never-opened.gguf', bytes=12345, mtime_ns=1)])


class LruOracle:
    """Key list with no production Cache implementation or slot simulation."""
    def __init__(self, capacity):
        self.capacity, self.keys = capacity, []  # oldest first

    def demand(self, layer, ids):
        keys = [(layer, expert) for expert in ids]
        protected = set(keys)
        missing = []
        for key in keys:
            if key not in self.keys:
                missing.append(key[1])
                if len(self.keys) == self.capacity:
                    self.keys.remove(next(old for old in self.keys if old not in protected))
                self.keys.append(key)
        for key in keys:
            self.keys.remove(key)
            self.keys.append(key)
        return missing


def forward(position):
    fields = dict(kind=0, rows=1, p0=position, nout=1, wanth=0, logits=1, reused=0,
        outcome='completed', acquire_ms='0', input_ms='0', compute_ms='1', output_ms='0',
        total_ms='1', scope=cohort.retention._FORWARD_SCOPE)
    return cohort.retention._FORWARD + ' ' + ' '.join(f'{key}={value}' for key, value in fields.items())


def source_trace(demands=None, capacities=None, start=33):
    capacities = {1: 3, 3: 3, 4: 3} if capacities is None else capacities
    if demands is None:
        demands = [dict.fromkeys((1, 2, 3, 4), pair) for pair in
            ((0, 1), (0, 2), (0, 1), (3, 4), (3, 4), (0, 1))]
    data = metadata()
    lookup = {layer: group for group in data['groups'] for layer in group['layers']}
    caches = {group: LruOracle(size) for group, size in capacities.items()}
    lines, expected = [], []
    for index, demand in enumerate(demands):
        for layer in sorted(lookup):
            group = lookup[layer]
            if not index and layer == group['id']:
                lines.append(f"[HOSTMOE-ROUTE-CREATE] layer={layer} slots={capacities[group['id']]} experts=8")
            ids = demand[layer]
            misses = caches[group['id']].demand(layer, ids)
            size = len(misses) * group['expert_bytes']
            lines.append(f'[HOSTMOE-ROUTE] layer={layer} ids=' + ','.join(map(str, ids)))
            lines.append(f'[HOSTMOE-CACHE-TIMING] layer={layer} rows=1 joint=0 bulk=0 outcome=completed '
                         f'file_bytes={size} scope=' + cohort.retention._CACHE_SCOPE)
            expected.append(dict(p0=start + index, layer=layer, misses=misses, bytes=size))
        lines.append(forward(start + index))
    return '\n'.join(lines), expected


def run(text=None, data=None, **changes):
    options = dict(capacities={1: 3, 3: 3, 4: 3}, start_position=33, count=6, top_k=2,
        begin_line=1, train_tokens=2, budgets=(96, 192, 384, 768))
    options.update(changes)
    return cohort.analyse(source_trace()[0] if text is None else text,
        metadata() if data is None else data, **options)


class RamMissCohortTests(unittest.TestCase):
    def test_every_cold_gpu_miss_and_timer_matches_independent_oracle(self):
        text, expected = source_trace()
        result = run(text)
        actual = result['baseline']
        self.assertEqual(actual['layer_timer_checks'], 24)
        self.assertEqual(actual['gpu_misses'], sum(len(row['misses']) for row in expected))
        self.assertEqual(actual['logical_source_bytes'], sum(row['bytes'] for row in expected))
        for row, oracle in zip(actual['rows'], expected):
            self.assertEqual((row['p0'], row['layer'], row['gpu_miss_experts'], row['logical_source_bytes']),
                (oracle['p0'], oracle['layer'], oracle['misses'], oracle['bytes']))
            self.assertLess(row['route_line'], row['cache_timer_line'])
        self.assertEqual(result['split'], dict(training_first=33, training_last=34, training_tokens=2,
            holdout_first=35, holdout_last=38, holdout_tokens=4))

    def test_training_gpu_hits_do_not_increment_admission_frequency(self):
        capacities = {1: 4, 3: 2, 4: 2}
        demands = [dict.fromkeys((1, 2, 3, 4), (0, 1)) for _ in range(6)]
        result = run(source_trace(demands, capacities)[0], capacities=capacities, budgets=(10000,))
        selected = result['cohorts'][0]['selected']
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(item['training_gpu_misses'] == 1 for item in selected))
        self.assertEqual(result['training']['gpu_hits_excluded_from_scores'], 8)
        self.assertEqual(result['holdout']['logical_source_bytes'], 0)
        self.assertEqual(result['cohorts'][0]['holdout_potentially_served_logical_bytes'], 0)

    def test_same_expert_in_distinct_sources_is_charged_independently(self):
        result = run(budgets=(10000,))
        selected = {(row['layer'], row['expert']) for row in result['cohorts'][0]['selected']}
        self.assertIn((1, 0), selected)
        self.assertIn((2, 0), selected)
        self.assertEqual(result['cohorts'][0]['selected_payload_bytes'],
            sum(row['payload_bytes'] for row in result['cohorts'][0]['selected']))
        other = metadata(); other['model_files'][0]['mtime_ns'] += 1
        other_result = run(data=other)
        self.assertNotEqual(result['baseline']['source_identity'], other_result['baseline']['source_identity'])
        self.assertNotEqual(result['cohorts'][0]['cohort_sha256'], other_result['cohorts'][0]['cohort_sha256'])
        self.assertEqual(result['cohorts'][0]['source_identity'], result['baseline']['source_identity'])

    def test_holdout_hot_only_keys_never_enter_frozen_selection(self):
        first = [dict.fromkeys((1, 2, 3, 4), pair) for pair in ((0, 1), (0, 2))]
        one = first + [dict.fromkeys((1, 2, 3, 4), (6, 7)) for _ in range(4)]
        two = first + [dict.fromkeys((1, 2, 3, 4), (0, 1)) for _ in range(4)]
        a, b = run(source_trace(one)[0]), run(source_trace(two)[0])
        for left, right in zip(a['cohorts'], b['cohorts']):
            self.assertEqual(left['cohort_sha256'], right['cohort_sha256'])
            self.assertEqual(left['selected'], right['selected'])
            self.assertTrue(all(item['expert'] < 3 for item in left['selected']))
            self.assertEqual(left['cohort_updates_during_holdout'], 0)

    def test_frequency_first_deterministic_ties_and_whole_triple_greedy_fit(self):
        training = [dict(layer=3, group=3, expert_bytes=192, gpu_miss_experts=[0])] * 3
        training += [dict(layer=2, group=1, expert_bytes=48, gpu_miss_experts=[1, 0])] * 2
        training += [dict(layer=1, group=1, expert_bytes=48, gpu_miss_experts=[0])] * 2
        small = cohort.freeze_cohort(training, 96)
        self.assertEqual([(row['layer'], row['expert']) for row in small['selected']], [(1, 0), (2, 0)])
        self.assertEqual(small['selected_payload_bytes'], 96)
        large = cohort.freeze_cohort(training, 240)
        self.assertEqual([(row['layer'], row['expert']) for row in large['selected']], [(3, 0), (1, 0)])
        self.assertEqual(cohort.freeze_cohort(training, 47)['selected_triples'], 0)

    def test_cold_admission_is_separate_from_training_and_gpu_uploads_unchanged(self):
        result = run()
        self.assertEqual(result['training']['logical_source_bytes'] + result['holdout']['logical_source_bytes'],
            result['baseline']['logical_source_bytes'])
        holdout = [row for row in result['baseline']['rows'] if row['p0'] >= 35]
        for item in result['cohorts']:
            chosen = {(row['layer'], row['expert']) for row in item['selected']}
            served = sum(row['expert_bytes'] for row in holdout for expert in row['gpu_miss_experts']
                         if (row['layer'], expert) in chosen)
            self.assertEqual(item['holdout_potentially_served_logical_bytes'], served)
            self.assertLessEqual(item['selected_payload_bytes'], item['budget_bytes'])
            self.assertEqual(item['cold_admission_logical_source_bytes'], item['selected_payload_bytes'])
            self.assertEqual(item['cold_training_plus_admission_logical_source_bytes'],
                result['training']['logical_source_bytes'] + item['selected_payload_bytes'])
            self.assertEqual(item['holdout_potential_bytes_minus_cold_admission'], served - item['selected_payload_bytes'])
            self.assertEqual(item['holdout_gpu_upload_payload_bytes_unchanged'], result['holdout']['logical_source_bytes'])
        for field in ('physical_admission_qualified', 'guaranteed_residency', 'performance_qualified',
                      'physical_io_qualified', 'numerical_quality_qualified', 'selection_uses_holdout'):
            self.assertFalse(result[field])

    def test_missing_or_wrong_fresh_pool_seed_fails_before_scoring(self):
        text = source_trace()[0]
        for bad in (text.replace('[HOSTMOE-ROUTE-CREATE] layer=1 slots=3 experts=8\n', '', 1),
                    text.replace('slots=3 experts=8', 'slots=4 experts=8', 1),
                    text.replace('slots=3 experts=8', 'slots=3 experts=9', 1)):
            with self.subTest(bad=bad[:100]), self.assertRaisesRegex(ValueError, 'seed'):
                run(bad)
        with self.assertRaises(ValueError):
            run(capacities={1: 3, 3: 3})

    def test_later_capacity_source_lifetime_or_policy_mutations_refuse(self):
        text = source_trace()[0]
        marker = '[HOSTMOE-ROUTE] layer=1 ids=0,2'
        for prefix in ('[HOSTMOE-ROUTE-CREATE] layer=1 slots=3 experts=8',
                       '[HOSTMOE-CACHE-FILE] layer=1 calls=2 file_bytes=192',
                       '[HOSTMOE-REBALANCE] layer=1', '[HOSTMOE-VMM-PRESSURE] profile=1',
                       '[HOSTMOE-PREFETCH] issued=1'):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                run(text.replace(marker, prefix + '\n' + marker, 1))
        # Old inherited owner may be retired BEFORE the fresh seed only.
        result = run('[HOSTMOE-REBALANCE] layer=1\n'
            '[HOSTMOE-CACHE-FILE] layer=1 calls=99 file_bytes=999\n' + text)
        self.assertTrue(result['every_layer_timer_matched'])
        with self.assertRaises(ValueError):
            run(text.replace('[HOSTMOE-ROUTE] layer=1 ids=0,1',
                '[HOSTMOE-REBALANCE] layer=1\n[HOSTMOE-ROUTE] layer=1 ids=0,1', 1))

    def test_even_one_layer_byte_mismatch_refuses_instead_of_trusting_total(self):
        text = source_trace()[0]
        self.assertIn('file_bytes=96', text)
        with self.assertRaisesRegex(ValueError, 'miss bytes'):
            run(text.replace('file_bytes=96', 'file_bytes=95', 1))
        # Equal and opposite errors preserve the whole-capture total but fail
        # the independent per-layer proof before training/cohort selection.
        bad = text.replace('file_bytes=96', 'file_bytes=95', 1)
        bad = bad.replace('layer=2 rows=1 joint=0 bulk=0 outcome=completed file_bytes=96',
                          'layer=2 rows=1 joint=0 bulk=0 outcome=completed file_bytes=97', 1)
        with self.assertRaisesRegex(ValueError, 'miss bytes'):
            run(bad)

    def test_missing_interleaved_failed_or_trailing_call_data_refuses(self):
        text = source_trace()[0]
        bads = [text.replace('ids=0,1', 'ids=0,0', 1), text.replace('ids=0,1', 'ids=0,8', 1),
                text.replace('outcome=completed', 'outcome=failed', 1),
                text.replace('p0=34 ', 'p0=33 ', 1), text + '\n[HOSTMOE-ROUTE] layer=1 ids=0,1',
                text + '\n' + forward(39)]
        lines = text.splitlines()
        lines[2], lines[3] = lines[3], lines[2]
        bads.append('\n'.join(lines))
        for bad in bads:
            with self.subTest(bad=bad[:100]), self.assertRaises(ValueError):
                run(bad)
        with self.assertRaises(ValueError):
            run(end_line=len(text.splitlines()) - 1)

    def test_bounded_parameters_and_metadata_fail_closed(self):
        for changes in (dict(train_tokens=0), dict(train_tokens=6), dict(train_tokens=True),
                        dict(budgets=(0,)), dict(budgets=(True,)), dict(budgets=(96, 96)),
                        dict(count=cohort.MAX_CALLS + 1), dict(begin_line=0), dict(end_line=999999),
                        dict(capacities={1: 1, 3: 3, 4: 3})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run(**changes)
        invalid = metadata(); invalid['groups'][0]['projections'][0]['expert_bytes'] += 1
        with self.assertRaises(ValueError):
            run(data=invalid)

    def test_cli_rejects_unpinned_capture_before_any_analysis_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            trace, data, output = [Path(directory) / name for name in ('trace.log', 'metadata.json', 'result.json')]
            trace.write_text(source_trace()[0], encoding='utf-8')
            data.write_text(json.dumps(metadata()), encoding='utf-8')
            arguments = ['--trace', str(trace), '--metadata', str(data), '--output', str(output)]
            with mock.patch.object(cohort, 'analyse') as analysis, self.assertRaisesRegex(ValueError, 'unrecognized'):
                cohort.main(arguments)
            analysis.assert_not_called()
            self.assertFalse(output.exists())
            # Test successful CLI plumbing with tiny independently generated
            # data, explicit local hash pins and the real analyzer, not a model.
            options = dict(capacities={1: 3, 3: 3, 4: 3}, start_position=33, count=6,
                train_tokens=2, top_k=2, begin_line=1)
            baseline = run()['baseline']
            with mock.patch.object(cohort, 'V27_TRACE_SHA256', hashlib.sha256(trace.read_bytes()).hexdigest()), \
                    mock.patch.object(cohort, 'V27_METADATA_SHA256', hashlib.sha256(data.read_bytes()).hexdigest()), \
                    mock.patch.object(cohort, 'V27_WINDOW', options), \
                    mock.patch.object(cohort, 'V27_BASELINE', {key: baseline[key] for key in cohort.V27_BASELINE}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cohort.main(arguments), 0)
                result = json.loads(output.read_text(encoding='utf-8'))
                self.assertFalse(result['provenance']['checkpoint_opened'])
                self.assertEqual(result['baseline']['layer_timer_checks'], 24)
                self.assertEqual(len(result['cohorts']), 4)
                with self.assertRaises(FileExistsError):
                    cohort.main(arguments)

    def test_bounded_input_reader_refuses_oversize_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bounded.bin'; path.write_bytes(b'abcd')
            with self.assertRaisesRegex(ValueError, 'Unbounded'):
                cohort._read_bounded(path, 3, hashlib.sha256(b'abcd').hexdigest())


class WeightedRamOracle:
    """Independent linear list, recomputed byte totals, no production LRU use."""
    def __init__(self, capacity, entries=()):
        self.capacity = capacity
        self.entries = list(entries)
        self.peak = sum(size for _, size in self.entries)

    def demand(self, key, size):
        matches = [index for index, item in enumerate(self.entries) if item[0] == key]
        if matches:
            item = self.entries.pop(matches[0])
            if item[1] != size:
                raise ValueError('oracle immutable size')
            self.entries.append(item)
            return True
        if size <= self.capacity:
            while sum(item[1] for item in self.entries) + size > self.capacity:
                del self.entries[0]
            self.entries.append((key, size))
            self.peak = max(self.peak, sum(item[1] for item in self.entries))
        return False


def weighted_rows(demands):
    return [dict(p0=index, layer=layer, group=layer, expert_bytes=size,
                 gpu_miss_experts=[expert], gpu_hits=0, logical_source_bytes=size)
            for index, (layer, expert, size) in enumerate(demands)]


class SameTotalRamTests(unittest.TestCase):
    identity = 'one-immutable-source-generation'

    @staticmethod
    def streams():
        training = weighted_rows(((1,0,4),(2,0,6),(3,0,3),(1,0,4),(1,1,4),
                                  (2,0,6),(1,0,4),(2,1,6),(1,0,4)))
        holdout = weighted_rows(((3,0,3),(2,0,6),(1,0,4),(3,0,3),(1,1,4),
                                 (1,0,4),(2,1,6),(2,0,6),(1,0,4)))
        return training, holdout

    @classmethod
    def serve_oracle(cls, cache, rows, pinned=None):
        pinned = {} if pinned is None else pinned
        miss_bytes = hits = 0
        for row in rows:
            for expert in row['gpu_miss_experts']:
                key = (cls.identity, row['layer'], expert)
                if key in pinned or cache.demand(key, row['expert_bytes']):
                    hits += 1
                else:
                    miss_bytes += row['expert_bytes']
        return miss_bytes, hits

    def test_weighted_lru_matches_independent_hit_eviction_and_oversize_oracle(self):
        actual, expected = cohort.PayloadLru(10), WeightedRamOracle(10)
        for key,size in (('a',4),('b',6),('a',4),('c',3),('too-large',11),
                         ('b',6),('d',4),('a',4),('d',4)):
            self.assertEqual(actual.demand(key,size),expected.demand(key,size))
            self.assertEqual(list(actual.entries.items()),expected.entries)
            self.assertEqual(actual.owned_bytes,sum(size for _,size in expected.entries))
            self.assertEqual(actual.peak_bytes,expected.peak)
            self.assertLessEqual(actual.owned_bytes,10)
        with self.assertRaises(ValueError):
            actual.demand('d',5)

    def test_zero_dynamic_capacity_never_holds_or_evicts_payload(self):
        cache=cohort.PayloadLru(0)
        self.assertFalse(cache.demand((self.identity,1,0),4))
        self.assertEqual((cache.owned_bytes,cache.peak_bytes,len(cache.entries)),(0,0,0))

    def test_layer_and_generation_are_part_of_exact_ram_key(self):
        cache=cohort.PayloadLru(8)
        for key in (('generationA',1,0),('generationA',2,0)):
            self.assertFalse(cache.demand(key,4))
        self.assertTrue(cache.demand(('generationA',1,0),4))
        self.assertFalse(cache.demand(('generationB',1,0),4))
        self.assertEqual(list(cache.entries),[('generationA',1,0),('generationB',1,0)])
        self.assertEqual(cache.owned_bytes,8)

    def test_newest_fitting_training_entries_are_retained_without_reloads(self):
        cache=cohort.PayloadLru(14)
        for key,size in (('a',2),('b',4),('c',4),('pinned',4)):
            cache.demand(key,size)
        before=list(cache.entries.items())
        result=cache.newest_fit(7,{'pinned'})
        self.assertEqual(list(result.entries.items()),[('a',2),('c',4)])
        self.assertEqual(result.owned_bytes,6)
        self.assertEqual(list(cache.entries.items()),before)

    def test_equal_total_baseline_and_all_candidates_match_independent_oracle(self):
        training,holdout=self.streams()
        cohorts=[cohort.freeze_cohort(training,budget) for budget in (6,10)]
        comparisons=cohort.compare_same_total_ram(training,holdout,cohorts,self.identity,(10,14))
        for report in comparisons:
            total=report['total_payload_budget_bytes']
            seed=WeightedRamOracle(total)
            training_bytes,_=self.serve_oracle(seed,training)
            ordinary=WeightedRamOracle(total,seed.entries)
            baseline_bytes,_=self.serve_oracle(ordinary,holdout)
            self.assertEqual(report['training']['ram_miss_logical_source_bytes'],training_bytes)
            self.assertEqual(report['baseline_holdout']['ram_miss_logical_source_bytes'],baseline_bytes)
            for arm,selected in zip(report['alternatives'],cohorts):
                pinned={(self.identity,item['layer'],item['expert']):item['payload_bytes']
                        for item in selected['selected']}
                pinned_bytes=sum(pinned.values())
                already=sum(size for key,size in seed.entries if key in pinned)
                remaining=total-pinned_bytes
                chosen=[]
                for key,size in reversed(seed.entries):
                    if key not in pinned and size <= remaining:
                        chosen.append((key,size));remaining -= size
                dynamic=WeightedRamOracle(total-pinned_bytes,reversed(chosen))
                missed,_=self.serve_oracle(dynamic,holdout,pinned)
                admission=pinned_bytes-already
                self.assertEqual(arm['holdout']['ram_miss_logical_source_bytes'],missed)
                self.assertEqual(arm['freeze']['incremental_admission_logical_source_bytes'],admission)
                self.assertEqual(arm['whole_window_logical_source_bytes_including_admission'],
                                 training_bytes+admission+missed)
                self.assertEqual(arm['net_logical_bytes_saved_after_incremental_admission'],
                                 baseline_bytes-admission-missed)
                self.assertEqual(arm['net_logical_bytes_saved_after_worst_case_cold_admission'],
                                 baseline_bytes-pinned_bytes-missed)
                self.assertEqual(arm['holdout']['owned_payload_peak_bytes'],pinned_bytes+dynamic.peak)
                self.assertLessEqual(arm['owned_payload_peak_bytes'],total)
                self.assertTrue(arm['total_budget_respected'])
                self.assertEqual(arm['duplicate_payload_bytes'],0)

    def test_incremental_admission_counts_absent_keys_once_and_deduplicates_present(self):
        training,holdout=self.streams()
        selected=cohort.freeze_cohort(training,10)
        arm=cohort.compare_same_total_ram(training,holdout,[selected],self.identity,(10,))[0]['alternatives'][0]
        self.assertEqual([(item['layer'],item['expert']) for item in selected['selected']],[(1,0),(2,0)])
        freeze=arm['freeze']
        self.assertEqual(freeze['training_resident_bytes'],10)
        self.assertEqual(freeze['selected_already_cached_bytes'],4)
        self.assertEqual(freeze['incremental_admission_logical_source_bytes'],6)
        self.assertEqual(freeze['worst_case_full_cold_admission_logical_source_bytes'],10)
        self.assertEqual(freeze['discarded_training_payload_bytes'],6)
        self.assertEqual(freeze['retained_dynamic_bytes'],0)
        self.assertEqual(freeze['owned_payload_after_bytes'],10)

    def test_actual_selected_bytes_leave_unused_cohort_allowance_for_lru(self):
        training,holdout=self.streams()
        selected=cohort.freeze_cohort(training,6)
        arm=cohort.compare_same_total_ram(training,holdout,[selected],self.identity,(10,))[0]['alternatives'][0]
        self.assertEqual(arm['selected_payload_bytes'],4)
        self.assertEqual(arm['holdout']['dynamic_capacity_bytes'],6)
        self.assertEqual(arm['freeze']['incremental_admission_logical_source_bytes'],0)
        self.assertEqual(arm['freeze']['retained_dynamic_bytes'],6)

    def test_empty_cohort_is_exactly_the_same_total_ordinary_lru(self):
        training,holdout=self.streams()
        selected=cohort.freeze_cohort(training,1)
        result=cohort.compare_same_total_ram(training,holdout,[selected],self.identity,(10,))[0]
        arm=result['alternatives'][0]
        self.assertEqual(arm['holdout'],result['baseline_holdout'])
        self.assertEqual(arm['selected_payload_bytes'],0)
        self.assertEqual(arm['freeze']['incremental_admission_logical_source_bytes'],0)
        self.assertEqual(arm['net_logical_bytes_saved_after_incremental_admission'],0)

    def test_bad_static_cohort_can_lose_and_cold_admission_is_not_hidden(self):
        training=weighted_rows(((1,0,4),)*4+((2,0,6),))
        holdout=weighted_rows(((2,0,6),)*3)
        selected=cohort.freeze_cohort(training,4)
        result=cohort.compare_same_total_ram(training,holdout,[selected],self.identity,(6,))[0]
        arm=result['alternatives'][0]
        self.assertEqual(result['baseline_holdout']['ram_miss_logical_source_bytes'],0)
        self.assertEqual(arm['freeze']['incremental_admission_logical_source_bytes'],4)
        self.assertEqual(arm['holdout']['ram_miss_logical_source_bytes'],18)
        self.assertEqual(arm['net_logical_bytes_saved_after_incremental_admission'],-22)

    def test_frozen_seed_and_admission_do_not_consult_holdout_or_other_arms(self):
        training,holdout=self.streams()
        selected=[cohort.freeze_cohort(training,budget) for budget in (6,10)]
        a=cohort.compare_same_total_ram(training,holdout,selected,self.identity,(10,))[0]
        b=cohort.compare_same_total_ram(training,weighted_rows(((3,7,3),)*20),
                                      list(reversed(selected)),self.identity,(10,))[0]
        by_budget={arm['cohort_budget_bytes']:arm for arm in b['alternatives']}
        for arm in a['alternatives']:
            other=by_budget[arm['cohort_budget_bytes']]
            self.assertEqual(arm['selection_sha256'],other['selection_sha256'])
            self.assertEqual(arm['freeze'],other['freeze'])
            self.assertEqual(arm['cohort_updates_during_holdout'],0)
        self.assertEqual(a['training'],b['training'])

    def test_full_analysis_preserves_gpu_byte_proof_and_flags_counterfactual_scope(self):
        result=run(budgets=(96,192),total_ram_budgets=(256,512))
        self.assertEqual(result['schema_version'],2)
        self.assertTrue(result['every_layer_timer_matched'])
        self.assertTrue(result['static_cohorts_are_payload_only_reference'])
        self.assertEqual(len(result['same_total_ram']),2)
        self.assertIn('not the actual OS cache',result['same_total_ram_scope'])
        for comparison in result['same_total_ram']:
            self.assertEqual(comparison['training']['gpu_miss_bytes'],result['training']['logical_source_bytes'])
            self.assertEqual(comparison['baseline_holdout']['h2d_payload_bytes_unchanged'],
                             result['holdout']['logical_source_bytes'])
            for arm in comparison['alternatives']:
                self.assertEqual(arm['holdout']['h2d_payload_bytes_unchanged'],
                                 result['holdout']['logical_source_bytes'])
        for key in ('physical_admission_qualified','guaranteed_residency','physical_io_qualified',
                    'performance_qualified','numerical_quality_qualified','selection_uses_holdout'):
            self.assertFalse(result[key])

    def test_invalid_total_and_inconsistent_or_duplicate_cohort_refuse(self):
        training,holdout=self.streams();selected=cohort.freeze_cohort(training,6)
        for totals in ((),(0,),(True,),(10,10),(2**64,),(5,)):
            with self.subTest(totals=totals),self.assertRaises(ValueError):
                cohort.compare_same_total_ram(training,holdout,[selected],self.identity,totals)
        import copy
        for kind in ('duplicate','bytes'):
            changed=copy.deepcopy(selected)
            if kind=='duplicate':
                changed['selected'] += changed['selected'][:1]
            else:
                changed['selected_payload_bytes'] += 1
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                cohort.compare_same_total_ram(training,holdout,[changed],self.identity,(10,))


if __name__ == '__main__':
    unittest.main()
