"""Model-free strict acceptance and immutable matrix tests; no native loading."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

source = Path(__file__).resolve().parents[1] / "linux-native-qualification.py"
spec = importlib.util.spec_from_file_location("linux_native_qualification", source)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def evidence(selected=False, flags=None):
    flags = flags or []
    graph = {"ExecutionId": 1, "Rows": 3, "Nodes": 40, "BackendSplits": 1,
             "Slots": [0, 1, 2], "PositionsBefore": [8, 8, 8], "PositionsAfter": [9, 9, 9]}
    pair = {"ExecutionId": 5, "Rows": 2, "Nodes": 30, "BackendSplits": 1,
            "Slots": [1, 2], "PositionsBefore": [1, 1], "PositionsAfter": [2, 2]}
    actual = dict(passed=True, full_logits_bit_exact=True, shared_graph_owner=True, backend="CUDA",
                  logit_elements=2432, logits_sha256="a" * 64, checks=200,
                  selected_experts_tested=selected, dispatch_faults_tested=False,
                  prefill_rows=None, quantized_source=None, swiglu_clamp=0, expert_groups=1,
                  native=[dict(path="/mnt/e/prod/libGgmlOps.so", sha256="b" * 64)],
                  final_budget=[dict(Pool="ram", Committed=0, Reserved=0),
                                dict(Pool="cuda", Committed=0, Reserved=0)],
                  executions=[dict(actual=graph) for _ in range(4)] + [dict(row_departure=pair)])
    if selected:
        actual["executions"].append(dict(minimum_cache_bytes=1000))
    compare = dict(passed=True, differing_elements=0, max_abs=0, rmse=0, logit_elements=2432,
                   logits_sha256="a" * 64, reference_sha256="a" * 64)
    return actual, compare


class LinuxNativeQualificationTests(unittest.TestCase):
    def accept(self, actual, comparison, flags=None):
        tool.validate_glm_report(actual, comparison, Path("/mnt/e/prod/libGgmlOps.so"),
                                 "b" * 64, "CUDA", flags or [])

    def test_exact_case_counts_and_native16_new_regressions(self):
        self.assertEqual(12, len(tool.cpu_cases()))
        self.assertEqual(21, len(tool.cuda_cases()))
        self.assertEqual(32, len(tool.glm_cases("/mnt/e/model.gguf")))
        names = {c[0]: c[2] for c in tool.cuda_cases()}
        self.assertEqual(["--glm-scalar-file-window"], names["scalar-large-file-q8"])
        self.assertEqual(["--pooled-quota", "--quantization", "iq1_m"], names["pooled-largest-quota-iq1"])
        self.assertEqual(["--mixed", "--pipeline-scope", "0"], dict((c[0], c[2]) for c in tool.cpu_cases())["mixed-file-batch-cpu"])

    def test_glm_matrix_keeps_rows_original_iq_clamp_and_fault_modes(self):
        cases = tool.glm_cases("/mnt/e/model with spaces.gguf")
        for mode in ("production", "hooks"):
            for kind in ("q8", "iq"):
                for rows in (8, 9, 10, 16, 128):
                    case = next(c for c in cases if c[0] == f"{mode}-{kind}-rows{rows}")
                    self.assertIn("--prefill-rows=" + str(rows), case[3])
                    self.assertEqual(mode == "hooks", "--dispatch-faults" in case[3])
                    quant = [f for f in case[3] if f.startswith("--quantized-source=")]
                    self.assertEqual(["--quantized-source=/mnt/e/model with spaces.gguf"] if kind == "iq" else [], quant)

    def test_explicit_case_filter_is_subset_coverage_and_rejects_unknown_names(self):
        base = ["--mode", "glm", "--native-root", "/mnt/e/native", "--output", "/mnt/e/new",
                "--managed-source", "/mnt/e/managed", "--managed-manifest", "/mnt/e/pins.json",
                "--probe-sha256", "a" * 64, "--quantized-source", "/mnt/e/model.gguf", "--plan"]
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(0, tool.main(base + ["--only-case", "production-cpu", "--only-case", "hooks-cuda-faults"]))
        plan = json.loads(capture.getvalue())
        self.assertEqual(2, plan["count"])
        self.assertFalse(plan["executed"])
        self.assertEqual(["production-cpu", "hooks-cuda-faults"], [c[0] for c in plan["cases"]])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.main(base + ["--only-case", "nonexistent"])

    def test_exit_zero_ordinary_arena_cannot_certify_windows_only_file_extent_checks(self):
        for name in ("arena-file-q8", "arena-file-iq1"):
            case = dict(exit_code=0, passed=True, skipped_or_unavailable=False)
            tool.annotate_requested_native_coverage(name, case, "PASS ordinary arena precision checks")
            self.assertFalse(case["passed"])
            self.assertTrue(case["skipped_or_unavailable"])
            self.assertFalse(case["registered_file_extents_exercised"])
            portable = dict(exit_code=0, passed=True, skipped_or_unavailable=False)
            tool.annotate_requested_native_coverage(name, portable,
                "PASS: exact file extents, cached reuse, out-of-order staging, partial-read recovery and source release")
            self.assertTrue(portable["passed"])
            self.assertTrue(portable["registered_file_extents_exercised"])
        ordinary = dict(exit_code=0, passed=True)
        tool.annotate_requested_native_coverage("arena-q8", ordinary, "PASS ordinary arena precision checks")
        self.assertTrue(ordinary["passed"])

    def test_accepts_complete_exact_native_and_ledger_evidence(self):
        self.accept(*evidence())
        self.accept(*evidence(True), flags=["--selected-experts"])

    def test_rejects_missing_or_tolerance_based_logit_acceptance(self):
        actual, comparison = evidence()
        for key, value in (("passed", False), ("differing_elements", 1), ("max_abs", 1e-12),
                           ("rmse", 1e-12), ("reference_sha256", "c" * 64), ("logit_elements", 0)):
            with self.subTest(key=key):
                bad = comparison | {key: value}
                with self.assertRaises(RuntimeError):
                    self.accept(actual, bad)
        with self.assertRaises(RuntimeError):
            self.accept(actual, None)

    def test_rejects_unsettled_owners_and_wrong_loaded_native(self):
        actual, comparison = evidence()
        for key in ("Committed", "Reserved"):
            bad = copy.deepcopy(actual)
            bad["final_budget"][0][key] = 1
            with self.assertRaises(RuntimeError):
                self.accept(bad, comparison)
        for key, value in (("path", "/mnt/e/old/libGgmlOps.so"), ("sha256", "c" * 64)):
            bad = copy.deepcopy(actual)
            bad["native"][0][key] = value
            with self.assertRaises(RuntimeError):
                self.accept(bad, comparison)

    def test_rejects_stale_head_even_without_selected_mode(self):
        actual, comparison = evidence()
        bad = copy.deepcopy(actual)
        bad["executions"][-1]["row_departure"]["PositionsAfter"] = [1, 2]
        with self.assertRaises(RuntimeError):
            self.accept(bad, comparison)
    def test_accepts_plain_n3_without_unrequested_selected_departure(self):
        actual, comparison = evidence()
        actual["executions"].pop()
        self.accept(actual, comparison)
        actual["dispatch_faults_tested"] = True
        self.accept(actual, comparison, ["--dispatch-faults"])

    def test_selected_n3_requires_actual_primary_departure_n2(self):
        actual, comparison = evidence(True)
        bad = copy.deepcopy(actual)
        bad["executions"] = [e for e in bad["executions"] if "row_departure" not in e]
        with self.assertRaises(RuntimeError):
            self.accept(bad, comparison, ["--selected-experts"])

    def test_all_modes_require_at_least_four_actual_n3_executions(self):
        for selected in (False, True):
            actual, comparison = evidence(selected)
            actual["executions"].pop(0)
            with self.assertRaises(RuntimeError):
                self.accept(actual, comparison, ["--selected-experts"] if selected else [])

    def test_rejects_silent_coverage_and_minimum_floor_omissions(self):
        actual, comparison = evidence()
        with self.assertRaises(RuntimeError):
            self.accept(actual, comparison, ["--selected-experts"])
        actual["selected_experts_tested"] = True
        with self.assertRaises(RuntimeError):
            self.accept(actual, comparison, ["--selected-experts"])
        actual["executions"].append(dict(minimum_cache_bytes=1))
        with self.assertRaises(RuntimeError):
            self.accept(actual, comparison, ["--selected-experts", "--dispatch-faults"])


if __name__ == "__main__":
    unittest.main()
