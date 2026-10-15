import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('cupti_summary', Path(__file__).with_name('embeddinggemma2-cupti-summary.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ActivityAccountingTests(unittest.TestCase):
    def fixture(self):
        return dict(completed=True, diagnostic=True, hardware_counters=False, dropped=0, incomplete=0,
                    callback_exceptions=0, truncated=0, errors=[], kernel_count=2, sum_kernel_ms=3,
                    events=[dict(name='vision_kernel<64>', start_ns=10, end_ns=1000010),
                            dict(name='ampere_sgemm', start_ns=20, end_ns=2000020)],
                    kernels=[dict(name='vision_kernel<64>', count=1, sum_ms=1),
                             dict(name='ampere_sgemm', count=1, sum_ms=2)])

    def test_overlapping_kernel_times_remain_diagnostic(self):
        result = module.summarize(self.fixture())
        self.assertEqual(result['sum_kernel_ms'], 3)
        self.assertFalse(result['paired_performance_claim'])
        self.assertEqual(result['categories']['owned_vision_attention']['sum_kernel_ms'], 1)

    def test_incomplete_or_dropped_capture_is_rejected(self):
        for field in ('dropped', 'incomplete', 'callback_exceptions', 'truncated'):
            report = self.fixture()
            report[field] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.summarize(report)
        report = self.fixture()
        report['completed'] = False
        with self.assertRaises(ValueError):
            module.summarize(report)

    def test_isolated_launch_bound_kernel_retains_attention_category(self):
        report = self.fixture()
        report['events'][0]['name'] = 'vision_kernel_minblocks3<64, 128, 16>'
        report['kernels'][0]['name'] = report['events'][0]['name']
        result = module.summarize(report)
        self.assertEqual(result['categories']['owned_vision_attention']['sum_kernel_ms'], 1)

    def test_aggregate_and_timestamp_corruption_are_rejected(self):
        base = self.fixture()
        for mutation in (lambda d: d.update(kernel_count=3),
                         lambda d: d['kernels'][0].update(sum_ms=2),
                         lambda d: d['events'][0].update(end_ns=9),
                         lambda d: d.update(sum_kernel_ms=4)):
            report = copy.deepcopy(base)
            mutation(report)
            with self.assertRaises(ValueError):
                module.summarize(report)


if __name__ == '__main__':
    unittest.main()
