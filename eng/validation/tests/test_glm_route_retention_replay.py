"""Small portable CPU fixtures; no model, CUDA or large trace access."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_route_retention_replay as retention


def metadata():
    def group(first, layers, hidden, ff):
        shapes = ([hidden, ff, 16], [hidden, ff, 16], [ff, hidden, 16])
        return dict(id=first, layers=layers, expert_bytes=hidden*ff*4*3,
            projections=[dict(type=0, shape=shape, expert_bytes=hidden*ff*4) for shape in shapes])
    return dict(groups=[group(1, [1, 2], 2, 2), group(3, [3], 2, 4)],
        model_files=[dict(path='immutable-model.gguf', bytes=12345, mtime_ns=1)])


class ReferenceLru:
    """Independent key-list oracle, with all row hits protected before eviction."""
    def __init__(self, capacity):
        self.capacity, self.keys = capacity, []  # least recent first

    def row(self, layer, ids):
        keys = [(layer, expert) for expert in ids]
        protected = set(keys) & set(self.keys)
        missing = set()
        for key in keys:
            if key in self.keys:
                continue
            missing.add(key)
            if len(self.keys) == self.capacity:
                self.keys.remove(next(old for old in self.keys if old not in protected))
            self.keys.append(key)
            protected.add(key)
        for key in keys:
            self.keys.remove(key)
            self.keys.append(key)
        return missing


def forward(position, **changes):
    fields = dict(kind=0, rows=1, p0=position, nout=1, wanth=0, logits=1,
        reused=0, outcome='completed', acquire_ms='1.000000', input_ms='2.000000',
        compute_ms='3.000000', output_ms='4.000000', total_ms='10.000000',
        scope=retention._FORWARD_SCOPE)
    fields.update(changes)
    return retention._FORWARD + ' ' + ' '.join(f'{key}={value}' for key, value in fields.items())


def cache_time(layer, size, **changes):
    fields = dict(layer=layer, rows=1, joint=0, bulk=0, outcome='completed',
        total_ms='1.0', file_bytes=size, scope=retention._CACHE_SCOPE)
    fields.update(changes)
    return retention._CACHE_TIME + ' ' + ' '.join(f'{key}={value}' for key, value in fields.items())


def trace(rows=None, start=100, old_capacity=6, cold_capacity=5):
    """Native-order synthetic capture; actual read bytes come ONLY from oracle."""
    if rows is None:
        rows = [([0, 1], [0, 1]), ([0, 2], [1, 2]), ([1, 2], [0, 1])]
    lines, actual = [], ReferenceLru(old_capacity)
    for index, pair in enumerate(rows):
        if index == 1:
            lines.append('[HOSTMOE-CACHE-FILE] layer=1 calls=2 file_bytes=192')
            actual = ReferenceLru(cold_capacity)
        for layer in (1, 2, 3):
            if layer == 1 and index < 2:
                lines.append(f'[HOSTMOE-ROUTE-CREATE] layer=1 slots={old_capacity if index == 0 else cold_capacity} experts=16')
            ids = pair[layer-1] if layer < 3 else [0, 1]
            lines.append(f'[HOSTMOE-ROUTE] layer={layer} ids=' + ','.join(map(str, ids)))
            size = len(actual.row(layer, ids))*48 if layer < 3 else 192
            lines.append(cache_time(layer, size))
        lines.append(forward(start+index))
    return '\n'.join(lines)


def run(text=None, data=None, **changes):
    options = dict(group_id=1, start_position=100, count=3, initial_capacity=6,
                   cold_capacity=5, retained_capacity=4, top_k=2)
    options.update(changes)
    return retention.replay_retention(trace() if text is None else text,
        metadata() if data is None else data, **options)


class RetentionReplayTests(unittest.TestCase):
    def test_first_routes_belong_to_following_marker_and_are_not_lost(self):
        result = run()
        self.assertEqual(result['positions'], dict(first=100, last=102, count=3))
        first, second, _ = result['calls']
        self.assertEqual(first['cold_misses'], 4)
        self.assertEqual(first['retained_misses'], 4)
        self.assertEqual(first['cold_logical_source_bytes'], 192)
        self.assertLess(first['first_route_line'], first['forward_line'])
        self.assertEqual(second['restored_hits'], 2)
        self.assertEqual(second['net_logical_source_bytes_saved'], 96)
        self.assertEqual(result['preserved_initial_keys'], 4)
        for key in ('candidate_physical_admission_qualified', 'performance_qualified',
                    'physical_io_qualified', 'numerical_quality_qualified'):
            self.assertFalse(result[key])

    def test_undelimited_wide_prefix_refuses_until_explicit_complete_boundary(self):
        prefix = ('[HOSTMOE-ROUTE] layer=3 ids=0,1,0,1\n' +
                  cache_time(3, 192, rows=2, bulk=1) + '\n')
        with self.assertRaisesRegex(ValueError, 'sequence'):
            run(prefix+trace())
        result = run(prefix+trace(), begin_line=3)
        self.assertEqual(result['capture_lines']['first'], 3)
        self.assertEqual(result['calls'][0]['first_route_line'], 4)

    def test_completed_prefill_and_startup_marker_are_unselected(self):
        prefix = ('[HOSTMOE-ROUTE] layer=3 ids=0,1,0,1\n' +
                  cache_time(3, 192, rows=2, bulk=1) + '\n' +
                  forward(0, rows=2) + '\n')
        result = run(prefix+trace())
        self.assertEqual(result['calls'][0]['p0'], 100)
        self.assertEqual(result['calls'][0]['cold_misses'], 4)

    def test_one_token_source_keys_do_not_alias_equal_local_expert_ids(self):
        result = run()
        self.assertEqual(result['preserved_initial_keys'], 4)  # two layers x same IDs0,1
        self.assertEqual(result['calls'][0]['cold_misses'], 4)

    def test_metadata_generation_changes_key_namespace(self):
        original = run()['source_identity']
        other = metadata()
        other['model_files'][0]['mtime_ns'] += 1
        self.assertNotEqual(run(data=other)['source_identity'], original)

    def test_all_route_hits_protected_matches_independent_oracle(self):
        rng = random.Random(704)
        rows = [([0, 1], [0, 1])]
        rows += [(rng.sample(range(8), 2), rng.sample(range(8), 2)) for _ in range(40)]
        result = run(trace(rows), count=len(rows))
        cold, kept = ReferenceLru(6), ReferenceLru(6)
        for index, pair in enumerate(rows):
            if index == 1:
                cold = ReferenceLru(5)
                kept.capacity = 4
            cold_missing, kept_missing = set(), set()
            for layer, ids in enumerate(pair, 1):
                cold_missing |= cold.row(layer, ids)
                kept_missing |= kept.row(layer, ids)
            row = result['calls'][index]
            self.assertEqual(row['cold_misses'], len(cold_missing))
            self.assertEqual(row['retained_misses'], len(kept_missing))
            self.assertEqual(row['restored_hits'], len(cold_missing-kept_missing))
            self.assertEqual(row['regressed_hits'], len(kept_missing-cold_missing))
            self.assertEqual(row['net_logical_source_bytes_saved'],
                             (len(cold_missing)-len(kept_missing))*48)
        self.assertEqual(result['totals']['net_logical_source_bytes_saved'],
            result['totals']['cold_logical_source_bytes']-result['totals']['retained_logical_source_bytes'])

    def test_future_suffix_cannot_change_prior_decisions(self):
        prefix = run(trace([([0, 1], [0, 1]), ([0, 2], [1, 2])]), count=2)
        full = run()
        self.assertEqual(prefix['calls'], full['calls'][:2])

    def test_missing_duplicate_out_of_order_and_interleaved_forward_refuse(self):
        for text in (trace().replace(forward(100), ''),
                     trace().replace(forward(100), forward(100)+'\n'+forward(100)),
                     trace().replace(forward(101), forward(102)),
                     trace().replace(forward(100), forward(100)+'\n'+forward(0, kind=1))):
            with self.subTest(text=text[-150:]), self.assertRaises(ValueError):
                run(text)

    def test_failed_nonfinite_wrong_scope_and_duplicate_timing_field_refuse(self):
        for event in (forward(100, outcome='failed'), forward(100, compute_ms='nan'),
                      forward(100, acquire_ms='-0.1'), forward(100, scope='wrong'),
                      forward(100)+' rows=1', forward(100, p0=1 << 64)):
            with self.subTest(event=event), self.assertRaises(ValueError):
                run(trace().replace(forward(100), event))

    def test_wide_missing_duplicate_reordered_and_invalid_route_refuse(self):
        for replacement in ('', '[HOSTMOE-ROUTE] layer=2 ids=0,1',
                            '[HOSTMOE-ROUTE] layer=1 ids=0,1,2,3',
                            '[HOSTMOE-ROUTE] layer=1 ids=0,0',
                            '[HOSTMOE-ROUTE] layer=1 ids=0,16',
                            '[HOSTMOE-ROUTE] layer=1 ids=0,-1'):
            with self.subTest(route=replacement), self.assertRaises(ValueError):
                run(trace().replace('[HOSTMOE-ROUTE] layer=1 ids=0,1', replacement, 1))

    def test_wrong_target_create_capacity_missing_retire_and_extra_mutation_refuse(self):
        replacements = [('[HOSTMOE-ROUTE-CREATE] layer=1 slots=6 experts=16', ''),
            ('slots=6 experts=16', 'slots=7 experts=16'),
            ('slots=5 experts=16', 'slots=4 experts=16'),
            ('[HOSTMOE-CACHE-FILE] layer=1 calls=2 file_bytes=192', ''),
            ('calls=2 file_bytes=192', 'calls=1 file_bytes=192'),
            ('calls=2 file_bytes=192', 'calls=2 file_bytes=96'),
            (forward(101), forward(101)+'\n[HOSTMOE-CACHE-GROW] layer=1 slots=6'),
            (forward(101), forward(101)+'\n[HOSTMOE-ROUTE-CREATE] layer=1 slots=5 experts=16')]
        for before, after in replacements:
            with self.subTest(after=after), self.assertRaises(ValueError):
                run(trace().replace(before, after))

    def test_measured_per_layer_payload_mismatch_cannot_qualify(self):
        with self.assertRaisesRegex(ValueError, 'file bytes disagree'):
            run(trace().replace(cache_time(1, 96), cache_time(1, 48), 1))

    def test_cache_timer_missing_failed_wide_or_duplicated_refuse(self):
        for event in ('', cache_time(1, 96, rows=2), cache_time(1, 96, outcome='failed'),
                      cache_time(1, 96)+'\n'+cache_time(1, 96)):
            with self.subTest(event=event), self.assertRaises(ValueError):
                run(trace().replace(cache_time(1, 96), event, 1))

    def test_interleaved_route_and_cache_completion_refuse(self):
        route1 = '[HOSTMOE-ROUTE] layer=1 ids=0,1'
        route2 = '[HOSTMOE-ROUTE] layer=2 ids=0,1'
        original = '\n'.join((route1, cache_time(1, 96), route2))
        interleaved = '\n'.join((route1, route2, cache_time(1, 96)))
        with self.assertRaisesRegex(ValueError, 'Next layer route'):
            run(trace().replace(original, interleaved, 1))

    def test_preserved_prefix_must_fit_without_dropping_live_tail(self):
        with self.assertRaisesRegex(ValueError, 'occupied tail'):
            run(retained_capacity=3)

    def test_wrong_quantized_geometry_bytes_groups_and_overflow_refuse(self):
        variants = []
        for kind in ('bytes', 'type', 'shape', 'layers', 'duplicate', 'overflow', 'bool'):
            value = metadata()
            if kind == 'bytes':
                value['groups'][0]['expert_bytes'] = 49
            elif kind == 'type':
                value['groups'][0]['projections'][0]['type'] = 999
            elif kind == 'shape':
                value['groups'][0]['projections'][2]['shape'] = [2, 4, 16]
            elif kind == 'layers':
                value['groups'][1]['layers'] = [2, 3]
            elif kind == 'duplicate':
                value['groups'].append(copy.deepcopy(value['groups'][0]))
            elif kind == 'overflow':
                value['groups'][0]['projections'][0]['shape'][2] = 1 << 64
            else:
                value['groups'][0]['id'] = True
            variants.append(value)
        for value in variants:
            with self.subTest(metadata=value), self.assertRaises(ValueError):
                run(data=value)

    def test_input_bounds_and_bad_slice_cannot_drop_first_token(self):
        for changes in (dict(count=1), dict(top_k=0), dict(count=True),
                        dict(begin_line=2), dict(begin_line=0), dict(end_line=99999),
                        dict(start_position=(1 << 64)-1), dict(retained_capacity=6)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run(**changes)

    def test_trailing_unfinished_records_never_become_completed_call(self):
        result = run(trace()+'\n[HOSTMOE-ROUTE] layer=1 ids=0,1')
        self.assertEqual(result['trailing_unassociated_records'], 1)
        self.assertEqual(result['positions']['count'], 3)

    def test_cli_uses_saved_metadata_only_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, saved, output = folder/'trace.log', folder/'metadata.json', folder/'report.json'
            source.write_text(trace(), encoding='utf-8')
            saved.write_text(json.dumps(metadata()), encoding='utf-8')
            args = ['tool', '--trace', str(source), '--metadata', str(saved), '--group', '1',
                '--start-position', '100', '--calls', '3', '--initial-capacity', '6',
                '--cold-capacity', '5', '--retained-capacity', '4', '--top-k', '2', '--output', str(output)]
            with mock.patch.object(sys, 'argv', args), mock.patch('builtins.print'):
                retention.main()
            report = json.loads(output.read_text())
            self.assertFalse(report['metadata']['checkpoint_opened'])
            self.assertEqual(len(report['trace']['sha256']), 64)
            with mock.patch.object(sys, 'argv', args), self.assertRaises(FileExistsError):
                retention.main()


if __name__ == '__main__':
    unittest.main()
