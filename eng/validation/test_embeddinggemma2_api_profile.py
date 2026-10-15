"""Negative evidence checks: incorrect coverage/accounting cannot look qualified."""
import hashlib
import importlib.util
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location("api_profile", Path(__file__).with_name("embeddinggemma2-api-profile.py"))
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def fixture():
    http = {"benchmark_inputs": {name: ["example " + name] for name in api.STANDARD}}
    http["engines"] = {"tensorsharp": {"benchmarks": {name: dict(prompt_tokens=3, texts=1) for name in api.STANDARD}}}
    cases = []
    for name in api.STANDARD:
        text = http["benchmark_inputs"][name]
        tokens = [[2, 3, 4]]
        cases.append(dict(name=name, api_direct_bit_identical=True, string_api_samples_ms=[10.0, 10.0],
                          direct_token_api_samples_ms=[9.0, 9.0], separate_tokenization_samples_ms=[.1, .1],
                          sequence_lengths=[3], tokens=tokens, warmup_ms=12.0,
                          text_utf8_sha256=[hashlib.sha256(value.encode()).hexdigest() for value in text],
                          tokens_sha256=hashlib.sha256(struct.pack("<iii", 2, 3, 4)).hexdigest(),
                          compact_compute_calls_per_api=1, prompt_tokens=3))
    candidate = dict(completed=True, cases=cases, measured_string_api_ms=140.0,
                     measured_direct_token_api_ms=126.0, separate_tokenization_ms=1.4, warmup_api_ms=84.0,
                     measured_string_calls=14, measured_direct_token_calls=14, warmup_calls=7,
                     expected_successful_compute_calls=35, expected_compute_rows=105,
                     total_enclosing_embedding_api_ms=350.0)
    profile = dict(successful_compute_calls=35, input_rows=105, compute_elapsed_ms=300.0,
                   input_preparation=dict(elapsed_ms=10.0))
    return candidate, profile, http


class EvidenceTests(unittest.TestCase):
    def quality(self):
        return dict(completed=True, quality_thresholds=dict(min_cosine=.99999, max_absolute_error=.0005),
                    runs=[dict(name=name, exit_code=0, comparison_exit_code=0, executed=count, passed=count, skipped=0,
                               min_cosine=1.0, max_absolute_error=0.0) for name, count in (("eight", 8), ("8192", 1))])

    def test_original_strict_model_gates_required(self):
        api.qualified_graph(self.quality())
        prior = self.quality()
        prior["quality_thresholds"]["max_absolute_error"] = .005
        with self.assertRaises(ValueError):
            api.qualified_graph(prior)

    def test_skipped_model_scenarios_do_not_qualify(self):
        prior = self.quality()
        prior["runs"][1]["skipped"] = 1
        with self.assertRaises(ValueError):
            api.qualified_graph(prior)

    def test_omitted_8192_gate_does_not_qualify(self):
        prior = self.quality()
        prior["runs"].pop()
        with self.assertRaises(ValueError):
            api.qualified_graph(prior)

    def test_complete_consistent_evidence(self):
        result = api.decomposition(*fixture())
        self.assertEqual(result["successful_cases"], 7)
        self.assertEqual(result["api_remainder_ms"], 40.0)

    def reject(self, mutation):
        candidate, profile, http = fixture()
        mutation(candidate, profile, http)
        with self.assertRaises((ValueError, KeyError)):
            api.decomposition(candidate, profile, http)

    def test_incomplete_diagnostic_rejected(self):
        self.reject(lambda c, p, h: c.update(completed=False))

    def test_fallback_or_missing_compute_cannot_pass(self):
        self.reject(lambda c, p, h: p.update(successful_compute_calls=0))

    def test_wrong_actual_compute_rows_rejected(self):
        self.reject(lambda c, p, h: p.update(input_rows=104))

    def test_changed_http_text_rejected(self):
        self.reject(lambda c, p, h: h["benchmark_inputs"].update(single_short=["different text"]))

    def test_wrong_token_hash_rejected(self):
        self.reject(lambda c, p, h: c["cases"][0].update(tokens_sha256="0" * 64))

    def test_wrong_actual_http_token_count_rejected(self):
        self.reject(lambda c, p, h: h["engines"]["tensorsharp"]["benchmarks"]["single_short"].update(prompt_tokens=4))

    def test_bit_identity_failure_rejected(self):
        self.reject(lambda c, p, h: c["cases"][0].update(api_direct_bit_identical=False))

    def test_omitted_abba_measurement_rejected(self):
        self.reject(lambda c, p, h: c["cases"][0].update(direct_token_api_samples_ms=[9.0]))

    def test_incorrect_batch_geometry_rejected(self):
        self.reject(lambda c, p, h: c["cases"][0].update(compact_compute_calls_per_api=2))

    def test_forged_sample_total_rejected(self):
        self.reject(lambda c, p, h: c.update(measured_string_api_ms=100.0, total_enclosing_embedding_api_ms=310.0))

    def test_double_counted_nested_time_rejected(self):
        self.reject(lambda c, p, h: p.update(compute_elapsed_ms=349.0))

    def test_extra_unmeasured_scenario_rejected(self):
        self.reject(lambda c, p, h: h["benchmark_inputs"].update(extra=["not measured"]))


if __name__ == "__main__":
    unittest.main()
