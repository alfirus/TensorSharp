"""Check full-row identity gates and descriptive teacher-forced metrics."""
import copy
import importlib.util
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


SPEC = importlib.util.spec_from_file_location("maya_tensorsharp_logits", Path(__file__).parents[1] / "compare-maya-tensorsharp-logits.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return BENCH.file_sha(path)


class Fixture:
    def __init__(self, root):
        self.root = root
        self.original_path = root / "original.json"
        self.teacher_path = root / "maya.json"
        self.replay_path = root / "replay.json"
        self.candidate_path = root / "tensorsharp.json"
        self.teacher_file = root / "maya-dump.7.f32"
        self.matrix = np.asarray([[0, 3, 1, 0, -2], [0, 0, -2, 1, 3]], dtype="<f4")
        self.matrix.tofile(self.teacher_file)
        self.original = {"schema": 1, "temperature": 0, "thinking": False, "vocab_size": 5,
                         "eos_ids": [4], "tokenizer_model": "fixture",
                         "cases": [{"name": "first", "ids": [0, 2], "max_new": 3, "prompt": "fixture"}]}
        self.original_hash = write_json(self.original_path, self.original)
        self.teacher = {"schema": 1, "status": "completed", "engine": "project-maya",
                        "source_revision": "teacher-revision", "environment": {"DS4_SERVE_DUMP": "/remote/maya-dump"},
                        "command": ["maya-ds4", "--dspark", "none"], "mode": "quality_only",
                        "capture_excluded_from_performance": True, "case_file_sha256": self.original_hash,
                        "results": [{"name": "first", "repeat": 0, "warmup": False, "input_ids": [0, 2],
                                     "output_token_ids": [1, 4], "prompt_tokens": 2, "generated_tokens": 2,
                                     "max_new": 3, "finish_reason": "stop", "draft_offered": 0, "draft_accepted": 0,
                                     "reused_prompt_tokens": 0, "request_index": 7,
                                     "logits_capture": {"path": str(self.teacher_file), "row_count": 2, "vocab": 5,
                                         "request_index": 7, "sha256": BENCH.file_sha(self.teacher_file), "dtype": "float32",
                                         "byte_order": "little", "layout": BENCH.TEACHER_LAYOUT,
                                         "capture_excluded_from_performance": True}}]}
        self.teacher_hash = write_json(self.teacher_path, self.teacher)
        self.replay = self.prepare()
        self.replay_hash = write_json(self.replay_path, self.replay)
        captures = []
        argmax_ids = []
        for index, row in enumerate(self.matrix):
            path = root / f"case-0000-row-{index:04}.f32"
            row.tofile(path)
            argmax = int(np.argmax(row))
            argmax_ids.append(argmax)
            captures.append({"path": str(path), "sha256": BENCH.file_sha(path), "position": 2 + index,
                             "vocab": 5, "argmax": argmax, "forced_id": [1, 4][index]})
        self.candidate = {"schema": 1, "status": "completed", "engine": "TensorSharp",
                          "source_revision": "candidate-revision", "options": {"capture-logits": str(root)},
                          "capture_excluded_from_performance": True, "temperature": 0,
                          "dspark": False, "prefix_reuse": False, "case_file_sha256": self.replay_hash,
                          "results": [{"name": "first", "repeat": 0, "warmup": False, "status": "completed",
                                       "teacher_forced": True, "prompt_tokens": 2, "max_new": 2,
                                       "generated_tokens": 2, "generated_ids": [1, 4], "finish_reason": "stop",
                                       "cache_after_reset": 0, "cache_after_prefill": 2, "cache_after_generation": 3,
                                       "argmax_ids": argmax_ids, "logit_captures": captures}]}

    def prepare(self):
        return BENCH.prepare_replay(self.teacher, self.teacher_hash, self.original, self.original_hash, self.teacher_path)

    def compare(self, **kwargs):
        candidate_hash = write_json(self.candidate_path, self.candidate)
        return BENCH.compare_artifacts(self.replay, self.replay_hash, self.teacher, self.teacher_hash,
                                       self.candidate, candidate_hash, self.replay_path, self.teacher_path,
                                       self.candidate_path, **kwargs)


class NumericalMetricTests(unittest.TestCase):
    def test_known_distributions_give_kl_and_teacher_continuation_perplexity(self):
        teacher = np.log(np.asarray([.7, .2, .1], dtype=np.float64))
        candidate = np.log(np.asarray([.5, .4, .1], dtype=np.float64))
        row = BENCH.compare_vectors(teacher, candidate, 1)
        expected = .7 * math.log(.7 / .5) + .2 * math.log(.2 / .4)
        self.assertAlmostEqual(expected, row["kl_teacher_to_candidate"], places=12)
        self.assertTrue(row["top1_agrees"])
        self.assertFalse(row["teacher_emitted_matches_argmax"])
        summary = BENCH.summarize([row])
        self.assertAlmostEqual(5, summary["teacher_continuation_perplexity"], places=12)
        self.assertAlmostEqual(2.5, summary["candidate_continuation_perplexity"], places=12)

    def test_logit_shift_preserves_probabilities_without_an_allclose_quality_gate(self):
        teacher = np.asarray([1, 2, 3], dtype="<f4")
        row = BENCH.compare_vectors(teacher, teacher + 1000, 2)
        self.assertEqual(0, row["kl_teacher_to_candidate"])
        self.assertTrue(row["top1_agrees"])
        self.assertEqual(1000, row["max_absolute_logit_difference"])
        self.assertAlmostEqual(row["teacher_forced_nll"], row["candidate_forced_nll"])

    def test_full_vocabulary_tail_coordinate_affects_argmax_and_kl(self):
        teacher = np.zeros(129280, dtype="<f4")
        candidate = teacher.copy()
        candidate[-1] = 4
        row = BENCH.compare_vectors(teacher, candidate, 129279)
        self.assertEqual(129279, row["candidate_argmax"])
        self.assertFalse(row["top1_agrees"])
        self.assertGreater(row["kl_teacher_to_candidate"], 0)
        self.assertGreater(row["teacher_forced_nll"], row["candidate_forced_nll"])

    def test_stable_extreme_logits_report_perplexity_overflow_without_nonfinite_json(self):
        row = BENCH.compare_vectors(np.asarray([10000, -10000, 0]), np.asarray([-10000, 10000, 0]), 0)
        self.assertTrue(row["finite"])
        self.assertFalse(row["top1_agrees"])
        self.assertAlmostEqual(20000, row["kl_teacher_to_candidate"])
        summary = BENCH.summarize([row])
        self.assertIsNone(summary["candidate_continuation_perplexity"])
        self.assertTrue(summary["candidate_perplexity_overflows_float64"])
        json.dumps(summary, allow_nan=False)

    def test_nonfinite_coordinates_are_reported_without_slicing_the_vocabulary(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                row = BENCH.compare_vectors(np.asarray([1, 2, 3]), np.asarray([1, invalid, 3]), 2)
                self.assertFalse(row["finite"])
                self.assertEqual([1], row["candidate_nonfinite_coordinates_first32"])
                self.assertNotIn("kl_teacher_to_candidate", row)
                with self.assertRaises(ValueError):
                    BENCH.summarize([row])
                teacher_row = BENCH.compare_vectors(np.asarray([invalid, 2, 3]), np.asarray([1, 2, 3]), 2)
                self.assertFalse(teacher_row["finite"])
                self.assertEqual([0], teacher_row["teacher_nonfinite_coordinates_first32"])

    def test_rows_and_forced_ids_must_match_full_vocabulary(self):
        for teacher, candidate, forced in (([1, 2], [1, 2, 3], 0), ([1], [1], 0),
                                           ([1, 2], [1, 2], 2), ([1, 2], [1, 2], True)):
            with self.subTest(teacher=teacher, candidate=candidate, forced=forced), self.assertRaises(ValueError):
                BENCH.compare_vectors(np.asarray(teacher), np.asarray(candidate), forced)


class ReplayPreparationTests(unittest.TestCase):
    def test_preparation_uses_explicit_request_index_and_actual_eos_continuation_length(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            replay = fixture.prepare()
            case = replay["cases"][0]
            self.assertEqual([0, 2], case["ids"])
            self.assertEqual([1, 4], case["forced_ids"])
            self.assertEqual(2, case["max_new"])
            self.assertEqual(7, case["teacher"]["request_index"])
            self.assertEqual(fixture.teacher_hash, replay["teacher_report_sha256"])
            self.assertTrue(replay["capture_excluded_from_performance"])

    def test_missing_duplicate_or_foreign_teacher_cases_are_rejected(self):
        for change in (lambda f: f.teacher["results"].clear(),
                       lambda f: f.teacher["results"].append(copy.deepcopy(f.teacher["results"][0])),
                       lambda f: f.teacher["results"][0].update(name="foreign")):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                change(fixture)
                with self.assertRaises(ValueError):
                    fixture.prepare()

    def test_same_length_changed_input_ids_or_case_file_hash_are_rejected(self):
        for change in (lambda f: f.teacher["results"][0].update(input_ids=[2, 0]),
                       lambda f: f.teacher.update(case_file_sha256="0" * 64),
                       lambda f: f.teacher["results"][0].update(input_ids=[False, 2])):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                change(fixture)
                with self.assertRaises(ValueError):
                    fixture.prepare()

    def test_truncated_extra_changed_or_wrong_vocab_teacher_payload_is_rejected(self):
        for modification in ("short", "extra", "changed", "vocab"):
            with self.subTest(modification=modification), TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                if modification == "short":
                    fixture.teacher_file.write_bytes(fixture.teacher_file.read_bytes()[:-4])
                elif modification == "extra":
                    fixture.teacher_file.write_bytes(fixture.teacher_file.read_bytes() + b"extra")
                elif modification == "changed":
                    raw = bytearray(fixture.teacher_file.read_bytes())
                    raw[0] ^= 1
                    fixture.teacher_file.write_bytes(raw)
                else:
                    fixture.teacher["results"][0]["logits_capture"]["vocab"] = 4
                with self.assertRaises(ValueError):
                    fixture.prepare()

    def test_nonquality_speculative_reused_or_bad_eos_teacher_is_rejected(self):
        for change in (lambda f: f.teacher.update(capture_excluded_from_performance=False),
                       lambda f: f.teacher.update(mode="performance"),
                       lambda f: f.teacher["results"][0].update(draft_offered=1),
                       lambda f: f.teacher["results"][0].update(reused_prompt_tokens=1),
                       lambda f: f.teacher["results"][0].update(output_token_ids=[4, 1]),
                       lambda f: f.teacher["results"][0].update(output_token_ids=[1, 2])):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                change(fixture)
                with self.assertRaises(ValueError):
                    fixture.prepare()


class ArtifactComparisonTests(unittest.TestCase):
    def test_complete_comparison_includes_all_rows_and_declares_no_quality_threshold(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            result = fixture.compare()
            self.assertEqual("compared", result["status"], result.get("error"))
            self.assertEqual(2, result["summary"]["rows"])
            self.assertEqual(1, result["summary"]["top1_agreement_fraction"])
            self.assertEqual(0, result["summary"]["mean_kl_teacher_to_candidate"])
            self.assertEqual([2, 3], [row["prediction_position"] for row in result["rows"]])
            self.assertFalse(result["quality_qualified"])
            self.assertFalse(result["performance_qualified"])
            self.assertIsNone(result["thresholds"])
            self.assertEqual("teacher-revision", result["declared_sources"]["teacher"]["source_revision"])
            self.assertIsNone(result["declared_sources"]["candidate"]["environment"])

    def test_replay_teacher_or_candidate_hash_identity_mismatch_is_failure(self):
        for change in (lambda f: f.replay.update(teacher_report_sha256="a" * 64),
                       lambda f: f.candidate.update(case_file_sha256="b" * 64),
                       lambda f: f.replay["cases"][0]["teacher"].update(prompt_i32_sha256="c" * 64)):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                change(fixture)
                self.assertEqual("failed", fixture.compare()["status"])

    def test_missing_extra_or_reordered_candidate_rows_are_not_intersected(self):
        for modification in ("short", "extra", "reordered"):
            with self.subTest(modification=modification), TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                captures = fixture.candidate["results"][0]["logit_captures"]
                if modification == "short":
                    captures.pop()
                elif modification == "extra":
                    captures.append(copy.deepcopy(captures[0]))
                else:
                    captures.reverse()
                self.assertEqual("failed", fixture.compare()["status"])

    def test_wrong_cache_position_forced_id_argmax_and_bool_metadata_are_rejected(self):
        changes = (
            lambda row: row.update(cache_after_reset=1),
            lambda row: row.update(cache_after_generation=4),
            lambda row: row.update(generated_ids=[2, 4]),
            lambda row: row["logit_captures"][0].update(position=1),
            lambda row: row["logit_captures"][0].update(forced_id=2),
            lambda row: row["logit_captures"][0].update(vocab=4),
            lambda row: row.update(cache_after_reset=False),
            lambda row: row.update(generated_ids=[True, 4]),
        )
        for change in changes:
            with self.subTest(change=change), TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                change(fixture.candidate["results"][0])
                self.assertEqual("failed", fixture.compare()["status"])
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            row = fixture.candidate["results"][0]
            row["argmax_ids"][0] = row["logit_captures"][0]["argmax"] = 0
            self.assertIn("argmax", fixture.compare()["error"])

    def test_nonfinite_candidate_failure_retains_coordinate_diagnostic(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            capture = fixture.candidate["results"][0]["logit_captures"][0]
            row = fixture.matrix[0].copy()
            row[2] = np.nan
            row.tofile(capture["path"])
            capture["sha256"] = BENCH.file_sha(capture["path"])
            result = fixture.compare()
            self.assertEqual("failed", result["status"])
            self.assertIn("Nonfinite", result["error"])
            self.assertEqual([2], result["rows"][0]["candidate_nonfinite_coordinates_first32"])
            self.assertNotIn("summary", result)

    def test_copied_remote_capture_paths_can_be_explicitly_remapped(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.replay["cases"][0]["teacher"]["capture"]["path"] = "/remote/" + fixture.teacher_file.name
            for capture in fixture.candidate["results"][0]["logit_captures"]:
                capture["path"] = "/remote/" + Path(capture["path"]).name
            result = fixture.compare(teacher_root=fixture.root, candidate_root=fixture.root)
            self.assertEqual("compared", result["status"], result.get("error"))

    def test_bad_numeric_quality_remains_a_descriptive_comparison_not_an_automatic_failure(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            for capture, position in zip(fixture.candidate["results"][0]["logit_captures"], (0, 2)):
                row = np.full(5, -10, dtype="<f4")
                row[position] = 10
                row.tofile(capture["path"])
                capture.update(sha256=BENCH.file_sha(capture["path"]), argmax=position)
            fixture.candidate["results"][0]["argmax_ids"] = [0, 2]
            result = fixture.compare()
            self.assertEqual("compared", result["status"], result.get("error"))
            self.assertEqual(0, result["summary"]["top1_agreement_fraction"])
            self.assertGreater(result["summary"]["mean_kl_teacher_to_candidate"], 10)
            self.assertFalse(result["quality_qualified"])

    def test_cli_prepares_and_compares_fresh_ignored_evidence(self):
        with TemporaryDirectory() as directory, patch.object(BENCH, "ROOT", Path(directory)), patch("builtins.print"):
            root = Path(directory)
            fixture = Fixture(root)
            prepared = root / "artifacts" / "prepared-replay.json"
            self.assertEqual(0, BENCH.main(["prepare", "--cases", str(fixture.original_path),
                "--teacher-report", str(fixture.teacher_path), "--output", str(prepared)]))
            self.assertEqual(fixture.replay, json.loads(prepared.read_text(encoding="utf-8")))
            candidate_hash = write_json(fixture.candidate_path, fixture.candidate)
            output = root / "artifacts" / "comparison.json"
            result = BENCH.main(["compare", "--replay-cases", str(fixture.replay_path),
                                 "--teacher-report", str(fixture.teacher_path), "--candidate-report", str(fixture.candidate_path),
                                 "--output", str(output)])
            self.assertEqual(0, result)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(candidate_hash, report["declared_sources"]["candidate"]["report_sha256"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                BENCH.output_path(output)
            with self.assertRaisesRegex(ValueError, "ignored"):
                BENCH.output_path(root / "tracked.json")


if __name__ == "__main__":
    unittest.main()
