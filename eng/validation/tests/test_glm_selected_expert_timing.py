"""Portable CPU schema/window fixtures. No model/device or large trace access."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_selected_expert_timing as timing


# Constructed schema fixture, NOT an actual checkpoint execution. The spelling
# matches the native source's frozen fprintf contract independently of _KEYS.
LINE = ('[GLM-SELECTED-EXPERT-TIMING] profile=1 rows=1 p0=1872 outcome=completed '
    'valid=1 boundaries=42 asks=42 ready=42 completed=42 errors=0 '
    'ready_ms=20.000000 ids_ms=2.000000 helper_ms=50.000000 drain_ms=3.000000 '
    'callback_ms=60.000000 tail_ms=10.000000 compute_ms=100.000000 '
    'dsv4_available=1 dsv4_total_calls=100 dsv4_total_views=20 dsv4_total_fused=20 '
    'dsv4_total_nodes=600 dsv4_total_submit_ms=4.000000 '
    'dsv4_helper_calls=42 dsv4_helper_views=0 dsv4_helper_fused=0 '
    'dsv4_helper_nodes=126 dsv4_helper_submit_ms=1.500000 '
    'dsv4_outer_calls=58 dsv4_outer_views=20 dsv4_outer_fused=20 '
    'dsv4_outer_nodes=474 dsv4_outer_submit_ms=2.500000 '
    'scope=host_wall_and_nested_process_global_host_api_submission_not_kernel_or_cpu_exclusive')


def line(position=1872, **changes):
    values = dict(token.split('=') for token in LINE[len(timing.PREFIX)+1:].split(' '))
    values['p0'] = str(position)
    values.update({key: str(value) for key, value in changes.items()})
    return timing.PREFIX+' '+' '.join(f'{key}={value}' for key, value in values.items())


def unavailable(position=1872):
    fields = {key: 0 for key in timing._DS_COUNTS + timing._DS_TIMES}
    return line(position, dsv4_available=0, **fields)


def run(text=LINE, start=1872, count=1):
    return timing.qualify_log(text, ((start, count),))


class SelectedExpertTimingTests(unittest.TestCase):
    def test_full_source_schema_literal_and_nested_fractions(self):
        records = timing.parse_records(LINE)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['completed'], 42)
        result = run()
        self.assertTrue(result['passed'])
        window = result['windows'][0]
        self.assertTrue(window['callback_timing_qualified'])
        self.assertTrue(window['dsv4_submission_observed_for_every_record'])
        self.assertEqual(window['totals']['dsv4_outer_nodes'], 474)
        self.assertEqual(window['callback_host_wall_fractions']['callback_over_compute'], 0.6)
        self.assertEqual(window['nested_process_global_host_submission_fractions']['total_submit_over_compute'], 0.04)
        for key in ('performance_qualified', 'physical_io_qualified', 'numerical_quality_qualified',
                    'kernel_time_qualified', 'cpu_exclusive_time_qualified',
                    'external_process_model_provenance_qualified'):
            self.assertFalse(result[key])

    def test_complete_default_quick_windows_and_startup_exclusion(self):
        text = line(0, outcome='failed', valid=0)
        text += '\n'+'\n'.join(line(position) for start, count in timing.DEFAULT_WINDOWS
                              for position in range(start, start+count))
        result = timing.qualify_log(text)
        self.assertEqual([window['records'] for window in result['windows']], [32, 128])
        self.assertEqual(result['windows'][0]['totals']['boundaries'], 32*42)
        self.assertEqual(result['windows'][1]['totals']['compute_ms'], 12800)

    def test_missing_duplicate_out_of_order_and_interleaved_positions_refuse(self):
        for text in (line(1872), line(1872)+'\n'+line(1872)+'\n'+line(1873),
                     line(1873)+'\n'+line(1872),
                     line(1872)+'\n'+line(0)+'\n'+line(1873)):
            with self.subTest(text=text[:70]), self.assertRaises(ValueError):
                run(text, count=2)

    def test_failed_and_valid_zero_selected_records_refuse(self):
        for changes in (dict(outcome='failed'), dict(valid=0), dict(outcome='other')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run(line(**changes))

    def test_exact_callback_boundary_counts_are_required(self):
        for field in ('boundaries', 'asks', 'ready', 'completed'):
            for value in (0, 41, 43):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    run(line(**{field: value}))
        with self.assertRaises(ValueError):
            run(line(errors=1))

    def test_all_integer_fields_are_uint64_and_booleans_are_exact(self):
        for field in timing._COUNTS + timing._DS_COUNTS:
            for value in ('-1', '1.0', str(1 << 64), '+1', '01'):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    run(line(**{field: value}))
        for field in ('valid', 'dsv4_available'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                run(line(**{field: 2}))

    def test_all_time_fields_are_finite_nonnegative(self):
        for field in timing._TIMES + timing._DS_TIMES:
            for value in ('nan', 'inf', '-inf', '-0.1', '1e309', 'bad'):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    run(line(**{field: value}))

    def test_missing_extra_duplicate_fields_wrong_profile_width_scope_refuse(self):
        for text in (LINE.replace(' profile=1', ''), LINE+' extra=0', LINE+' ready=42',
                     line(profile=2), line(rows=2), line(scope='wrong'),
                     LINE.replace(' profile=1', ' profile=1  '), timing.PREFIX):
            with self.subTest(text=text[:90]), self.assertRaises(ValueError):
                run(text)

    def test_componentwise_dsv4_subset_and_difference_are_exact(self):
        for component in ('calls', 'views', 'fused', 'nodes'):
            for part in ('total', 'helper', 'outer'):
                values = timing.parse_records(LINE)[0]
                field = f'dsv4_{part}_{component}'
                with self.subTest(field=field), self.assertRaises(ValueError):
                    run(line(**{field: values[field]+1}))
        with self.assertRaises(ValueError):
            run(line(dsv4_helper_calls=101, dsv4_outer_calls=0))

    def test_six_decimal_time_decomposition_tolerance_and_limits(self):
        self.assertTrue(run(line(dsv4_outer_submit_ms='2.500002'))['passed'])
        for changes in (dict(dsv4_outer_submit_ms='2.500010'),
                        dict(dsv4_helper_submit_ms='4.000010', dsv4_outer_submit_ms=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run(line(**changes))

    def test_callback_and_compute_nested_walls_are_checked(self):
        # Three rounded parts may exceed the rounded enclosing value by a few ns.
        self.assertTrue(run(line(callback_ms='54.999998', compute_ms='84.999996'))['passed'])
        for changes in (dict(callback_ms='54.999990'), dict(compute_ms='89.999990')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run(line(**changes))

    def test_unavailable_facade_all_zero_still_allows_callback_timing(self):
        result = run(unavailable())
        window = result['windows'][0]
        self.assertTrue(window['callback_timing_qualified'])
        self.assertEqual(window['dsv4_available_records'], 0)
        self.assertFalse(window['dsv4_submission_observed_for_every_record'])
        self.assertIsNone(window['nested_process_global_host_submission_fractions'])
        for field in timing._DS_COUNTS + timing._DS_TIMES:
            with self.subTest(field=field), self.assertRaises(ValueError):
                run(unavailable().replace(field+'=0', field+'=1', 1))

    def test_available_one_with_no_observed_calls_gets_no_facade_credit(self):
        text = unavailable().replace('dsv4_available=0', 'dsv4_available=1')
        window = run(text)['windows'][0]
        self.assertEqual(window['dsv4_available_records'], 1)
        self.assertEqual(window['dsv4_observed_records'], 0)
        self.assertFalse(window['dsv4_submission_observed_for_every_record'])
        self.assertIsNone(window['nested_process_global_host_submission_fractions'])

    def test_partial_availability_does_not_report_complete_facade_fractions(self):
        window = run(LINE+'\n'+unavailable(1873), count=2)['windows'][0]
        self.assertEqual(window['dsv4_available_records'], 1)
        self.assertEqual(window['dsv4_observed_records'], 1)
        self.assertIsNone(window['nested_process_global_host_submission_fractions'])

    def test_total_positive_with_zero_outer_gets_no_observed_profile_credit(self):
        values = dict(dsv4_helper_calls=100, dsv4_outer_calls=0,
            dsv4_helper_views=20, dsv4_outer_views=0,
            dsv4_helper_fused=20, dsv4_outer_fused=0,
            dsv4_helper_nodes=600, dsv4_outer_nodes=0,
            dsv4_helper_submit_ms=4, dsv4_outer_submit_ms=0)
        window = run(line(**values))['windows'][0]
        self.assertTrue(window['callback_timing_qualified'])
        self.assertFalse(window['dsv4_submission_observed_for_every_record'])
        self.assertIsNone(window['nested_process_global_host_submission_fractions'])

    def test_zero_time_denominators_are_null_not_nan(self):
        text = line(**{key: 0 for key in timing._TIMES + timing._DS_COUNTS + timing._DS_TIMES}, dsv4_available=0)
        result = run(text)
        self.assertTrue(all(value is None for value in result['windows'][0]['callback_host_wall_fractions'].values()))
        json.dumps(result, allow_nan=False)

    def test_malformed_tag_outside_window_cannot_be_silently_ignored(self):
        with self.assertRaises(ValueError):
            run(timing.PREFIX+' broken\n'+LINE)
        with self.assertRaises(ValueError):
            run(LINE+'\n'+line(0)+' extra=0')

    def test_invalid_overlap_overflow_empty_windows_refuse(self):
        records = timing.parse_records(LINE)
        for windows in ((), ((1872, 0),), ((-1, 1),), ((True, 1),),
                        (((1 << 64)-1, 2),), ((1872, 1), (1872, 1))):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                timing.qualify_records(records, windows)

    def test_absurd_nonoverflow_call_count_refuses_before_range_expansion(self):
        with self.assertRaisesRegex(ValueError, 'more positions than captured'):
            timing.qualify_records(timing.parse_records(LINE), ((1872, 1 << 63),))

    def test_window_counter_and_time_sum_overflow_refuse(self):
        fields = dict(dsv4_total_nodes=(1 << 64)-1, dsv4_helper_nodes=0,
                      dsv4_outer_nodes=(1 << 64)-1)
        with self.assertRaisesRegex(ValueError, 'sum exceeds'):
            run(line(1872, **fields)+'\n'+line(1873, **fields), count=2)
        fields = {key: 0 for key in timing._DS_COUNTS + timing._DS_TIMES + timing._TIMES}
        fields.update(dsv4_available=0, compute_ms='1e308')
        with self.assertRaisesRegex(ValueError, 'sum overflow'):
            run(line(1872, **fields)+'\n'+line(1873, **fields), count=2)

    def test_cli_custom_window_hashes_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'log.txt', Path(directory)/'report.json'
            source.write_text(LINE, encoding='utf-8')
            args = ['tool', '--log', str(source), '--output', str(output), '--start-position', '1872', '--calls', '1']
            with mock.patch.object(sys, 'argv', args), mock.patch('builtins.print'):
                timing.main()
            result = json.loads(output.read_text())
            self.assertEqual(len(result['log']['sha256']), 64)
            self.assertFalse(result['performance_qualified'])
            with mock.patch.object(sys, 'argv', args), self.assertRaises(FileExistsError):
                timing.main()
            with mock.patch.object(sys, 'argv', args[:-2]), mock.patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit) as caught:
                timing.main()
            self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
