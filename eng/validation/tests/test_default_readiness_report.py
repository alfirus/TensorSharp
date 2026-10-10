import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import copy


SPEC = importlib.util.spec_from_file_location('readiness_report', Path(__file__).parents[1] / 'default-readiness-report.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class SpeculativeReadinessTests(unittest.TestCase):
    def rows(self):
        base = dict(Scenario='spec', Prompt=4, Reused=0, Fresh=4, OutTokens=2,
                    Tokens=[7, 8], Finish='eos', Note='', PrefillTps=10, DecodeTps=10, TtftMs=1,
                    Drafted=0, Accepted=0, VerifySteps=0, PlainSteps=0, Rollbacks=0)
        return [{**base, 'Label': 'plain greedy'},
                {**base, 'Label': 'draft head (auto)', 'Drafted': 4, 'Accepted': 2, 'VerifySteps': 1}]

    def summarize(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.json'
            path.write_text(json.dumps(rows))
            return REPORT.speculative_rows(path)

    def test_complete_engaged_pair_qualifies_without_inventing_missing_timing(self):
        result = self.summarize(self.rows())[0]
        self.assertTrue(result[2])
        self.assertEqual('未测量 → 未测量', result[5])

    def test_identically_truncated_pair_is_not_a_complete_pass(self):
        rows = self.rows()
        for row in rows:
            row['Finish'] = 'length'
        self.assertFalse(self.summarize(rows)[0][2])

    def test_attached_but_unaccepted_draft_is_not_coverage(self):
        rows = self.rows()
        rows[1]['Accepted'] = 0
        self.assertFalse(self.summarize(rows)[0][2])


class MayaReadinessTests(unittest.TestCase):
    def report(self):
        row = dict(tokens=[42, 9], generated=2, prompt_tokens=3, reused=0,
                   drafts_offered=0, drafts_accepted=0, finish='stop', prompt_ms=100, decode_ms=50,
                   quality=dict(passed=True), semantic=dict(passed=True, thinking_closed=True, tool_calls=[], content='42'))
        return dict(completed=True, process_success=True, exit_code=0, identities_unchanged=True,
                    errors=[], timed_out=False, prompt_tokenizer_exact=True,
                    performance_observation_complete=True, telemetry_samples=2, sampling_errors=[],
                    plan=dict(repetitions=2, eos_ids=[9]), rows=[row, copy.deepcopy(row)],
                    reference=dict(prompt_tokens=[1, 2, 3], generated_tokens=[42, 9], passed=True, eos=True, parsed_text='42'))

    def test_semantics_and_exact_tokens_are_independent_and_decode_uses_forward_count(self):
        report = self.report()
        report['rows'][1]['tokens'] = [43, 9]
        result = REPORT.maya_summary(report)
        self.assertTrue(result['quality_passed'])
        self.assertTrue(result['rows'][0]['exact'])
        self.assertFalse(result['rows'][1]['exact'])
        self.assertEqual(20, result['rows'][0]['tg'])
        self.assertEqual(30, result['rows'][0]['pp'])

    def test_incomplete_or_failed_run_cannot_qualify_completed_rows(self):
        for key, value in [('completed', False), ('process_success', False), ('exit_code', 1),
                           ('identities_unchanged', False), ('errors', ['cleanup failed']), ('timed_out', True)]:
            report = self.report(); report[key] = value
            result = REPORT.maya_summary(report)
            self.assertFalse(result['quality_passed'], key)
            self.assertFalse(any(r['passed'] for r in result['rows']), key)
        report = self.report(); report['rows'].pop()
        self.assertFalse(REPORT.maya_summary(report)['quality_passed'])

    def test_claimed_success_cannot_override_non_eos_wire_or_parser_failure(self):
        for changes in [dict(finish='length'), dict(generated=1), dict(reused=1), dict(tokens=[9, 9]),
                        dict(semantic=dict(passed=True, thinking_closed=True, content='41', tool_calls=[]))]:
            report = self.report(); report['rows'][0].update(changes)
            self.assertFalse(REPORT.maya_summary(report)['quality_passed'])

    def test_missing_sampling_does_not_invalidate_semantics_or_invent_performance(self):
        report = self.report(); report.pop('performance_observation_complete')
        result = REPORT.maya_summary(report)
        self.assertTrue(result['quality_passed'])
        self.assertFalse(result['performance_complete'])

    def test_empty_incomplete_run_remains_visible(self):
        report = self.report(); report['rows'] = []; report['completed'] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text(json.dumps(report))
            rows = REPORT.maya_rows(path)
        self.assertEqual(1, len(rows))
        self.assertFalse(rows[0][1])
        self.assertFalse(rows[0][2])
        self.assertEqual('无完整请求', rows[0][4])


if __name__ == '__main__':
    unittest.main()
