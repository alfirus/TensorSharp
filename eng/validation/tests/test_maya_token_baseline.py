import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock
import threading
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('maya_token_baseline', Path(__file__).parents[1] / 'maya-token-baseline.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class MayaWireTests(unittest.TestCase):
    def test_prefill_default_and_explicit_capacity(self):
        self.assertEqual('auto', mod.prefill_argument({}))
        self.assertEqual('auto', mod.prefill_argument({'prefill': 'auto'}))
        for value in (32, '32', '0032'):
            self.assertEqual('32', mod.prefill_argument({'prefill': value}))

    def test_prefill_rejects_ambiguous_or_unbounded_capacity(self):
        for value in (0, -1, True, 32.0, '32junk', '0', 'AUTO', ' 32', '３２', 2147483648, None):
            with self.assertRaises(ValueError, msg=repr(value)):
                mod.prefill_argument({'prefill': value})

    def reference(self):
        return dict(generated_tokens=[42, 9], prompt_tokens=[1, 2, 3], passed=True, eos=True)

    def row(self):
        return dict(mod.parse_done('DONE 2 3 120.0 20.0 stop 0 0 0 15 20'), tokens=[42, 9])

    def test_complete_exact_eos(self):
        self.assertTrue(mod.qualify(self.row(), self.reference(), [9])['passed'])

    def test_length_never_proves_eos_completion(self):
        row = self.row(); row['finish'] = 'length'
        self.assertFalse(mod.qualify(row, self.reference(), [9])['passed'])

    def test_earlier_eos_is_rejected(self):
        row = self.row(); row['tokens'] = [9, 9]
        ref = self.reference(); ref['generated_tokens'] = [9, 9]
        self.assertFalse(mod.qualify(row, ref, [9])['passed'])

    def test_truncated_or_extra_wire_tokens_fail(self):
        for key, value in [('generated', 3), ('prompt_tokens', 4), ('reused', 1), ('drafts_offered', 1)]:
            row = self.row(); row[key] = value
            self.assertFalse(mod.qualify(row, self.reference(), [9])['passed'], key)

    def test_different_tokens_do_not_infer_semantics(self):
        row = self.row(); row['tokens'] = [43, 9]
        result = mod.qualify(row, self.reference(), [9])
        self.assertFalse(result['passed'])
        self.assertFalse(result['semantic_inferred_from_exact_reference'])
        self.assertTrue(result['checkpoint_eos_observed'])

    def test_failed_reference_not_transferred(self):
        ref = self.reference(); ref['passed'] = False
        self.assertFalse(mod.qualify(self.row(), ref, [9])['passed'])

    def test_separate_answer_check_can_pass_different_tokens(self):
        row = self.row(); row['tokens'] = [43, 9]
        result = mod.qualify(row, self.reference(), [9], dict(passed=True))
        self.assertTrue(result['passed'])
        self.assertFalse(result['exact_reference_tokens'])
        self.assertFalse(result['semantic_inferred_from_exact_reference'])

    def test_parser_failure_cannot_be_overridden_by_exact_tokens(self):
        result = mod.qualify(self.row(), self.reference(), [9], dict(passed=False))
        self.assertFalse(result['passed'])

    def test_parser_pass_cannot_override_missing_eos(self):
        row = self.row(); row['tokens'] = [42, 8]
        self.assertFalse(mod.qualify(row, self.reference(), [9], dict(passed=True))['passed'])

    def test_nonfinite_negative_unknown_done_rejected(self):
        for line in ['DONE 2 3 nan 20 stop 0 0 0 15 20', 'DONE 2 3 20 inf stop 0 0 0 15 20',
                     'DONE -1 3 20 20 stop 0 0 0 15 20', 'DONE 2 3 20 20 unknown 0 0 0 15 20', 'DONE 2 3 20 20 stop']:
            with self.assertRaises(ValueError, msg=line): mod.parse_done(line)

    def test_cleanup_unstarted_threads_does_not_skip_process(self):
        process = Mock(); process.poll.side_effect = [None, 1]
        timer = threading.Timer(1, lambda: None)
        reader = threading.Thread(target=lambda: None)
        sampler = SimpleNamespace(stop=threading.Event(), thread=threading.Thread(target=lambda: None))
        code, errors = mod.cleanup_owned(process, timer, reader, sampler)
        self.assertEqual(1, code)
        self.assertEqual([], errors)
        process.kill.assert_called_once()
        self.assertTrue(sampler.stop.is_set())

    def test_cleanup_join_and_kill_failures_do_not_skip_sampler(self):
        process = Mock(); process.poll.return_value = None; process.kill.side_effect = OSError('kill error')
        process.wait.side_effect = TimeoutError('wait error')
        reader = Mock(ident=1); reader.join.side_effect = RuntimeError('join error'); reader.is_alive.return_value = True
        sampler = SimpleNamespace(stop=threading.Event(), thread=threading.Thread(target=lambda: None))
        code, errors = mod.cleanup_owned(process, None, reader, sampler)
        self.assertIsNone(code)
        self.assertTrue(any('kill error' in x for x in errors))
        self.assertTrue(any('join error' in x for x in errors))
        self.assertTrue(sampler.stop.is_set())


if __name__ == '__main__':
    unittest.main()
