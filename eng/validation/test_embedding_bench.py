"""Offline policy regression for explicit reference-only duplicate tolerance."""
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("embedding_bench", ROOT / "benchmarks/EmbeddingBench/embedding_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
http_spec = importlib.util.spec_from_file_location("embeddinggemma2_http", ROOT / "eng/validation/embeddinggemma2-http.py")
http = importlib.util.module_from_spec(http_spec)
http_spec.loader.exec_module(http)


class ReferenceToleranceTests(unittest.TestCase):
    def test_optional_boundary_snapshots_bracket_process_and_exclude_timing(self):
        events = []
        process = SimpleNamespace(pid=123, returncode=None)
        process.poll = lambda: process.returncode
        def wait(**kwargs):
            events.append("exit")
            process.returncode = 0
        process.wait = wait
        def launch(*args, **kwargs):
            events.append("launch")
            return process
        def capture():
            events.append("boundary")
            return {"available": True}
        args = SimpleNamespace(capture_boundary_telemetry=True, model="mock", reference_min_batch_cosine=.9999,
            reference_min_duplicate_cosine=.99999, measure_reference_after_quality_failure=True,
            prewarm_seconds=0, warmup=3, repeats=10, minimum_measure_seconds=1, cases=None)
        with tempfile.TemporaryDirectory() as directory, \
             patch("embeddinggemma2_boundary_telemetry.capture_boundary", side_effect=capture), \
             patch.object(bench.subprocess, "Popen", side_effect=launch), \
             patch.object(bench.os, "killpg", side_effect=lambda *a: events.append("stop")), \
             patch.object(bench, "request", return_value=(200, {})), \
             patch.object(bench, "process_memory", return_value={}), \
             patch.object(bench, "correctness", side_effect=lambda *a: events.append("correctness") or {}), \
             patch.object(bench, "benchmark", side_effect=lambda *a: events.append("timing") or {}):
            report = bench.run_engine("tensorsharp", ["dotnet", "/mock/TensorSharp.Server.Host.dll"],
                                      "mock", args, Path(directory))
        self.assertEqual(events, ["boundary", "launch", "correctness", "timing", "stop", "exit", "boundary"])
        self.assertEqual(report["server_exit_code"], 0)
        self.assertTrue(report["environment_audit"]["resolved_server_environment"])

    def setUp(self):
        vectors = [[1, 0, 0], [.99, .1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [.1, 0, .99],
                   [.7, .7, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1], [-1, -1, -1]]
        self.vectors = [[value / math.sqrt(sum(v*v for v in row)) for value in row] for row in vectors]
        self.observed = .999985

    def embedding(self, url, model, inputs):
        repeated = [bench.TEXTS[0], bench.TEXTS[0], bench.TEXTS[1]]
        if inputs == repeated:
            rows = [self.vectors[0], [self.observed, math.sqrt(1-self.observed**2), 0], self.vectors[1]]
        elif isinstance(inputs, str):
            rows = [self.vectors[bench.TEXTS.index(inputs)]]
        else:
            rows = [self.vectors[bench.TEXTS.index(text)] for text in inputs]
        return rows, dict(prompt_tokens=len(rows), total_tokens=len(rows))

    def test_candidate_and_default_reference_keep_strict_duplicate_gate(self):
        with patch.object(bench, "embedding", self.embedding):
            for candidate, reference_gate in [(True, .999), (False, .99999)]:
                with self.subTest(candidate=candidate):
                    with self.assertRaises(AssertionError) as caught:
                        bench.correctness("unused", "unused", candidate, .999, reference_gate)
                    self.assertEqual(caught.exception.args[0][0], "duplicate inputs")
                    self.assertAlmostEqual(caught.exception.args[0][1], self.observed)
                    self.assertEqual(caught.exception.args[0][2], .99999)

    def test_owned_cpu_library_pinned_even_with_explicit_env_path(self):
        from embeddinggemma2_evidence import command_artifacts
        with tempfile.TemporaryDirectory() as output:
            directory = Path(output)
            host = directory / "TensorSharp.Server.Host.dll"
            host.write_bytes(b"mock host; never executed")
            library = directory / "libTensorSharpCpu.so"
            library.write_bytes(b"mock native math; never loaded")
            explicit = directory / "separate-built-library.so"
            explicit.write_bytes(b"independently built mock library")
            blas = directory / "libopenblas.so"
            blas.write_bytes(b"separately selected mock BLAS provider")
            records = command_artifacts(lambda command: [], ["env", "TENSORSHARP_CPU_NATIVE_LIBRARY=" + str(explicit),
                "TSG_EMBEDDING_BLAS_LIBRARY=" + str(blas), "dotnet", str(host)])
            self.assertEqual({Path(item["path"]) for item in records}, {library.resolve(), explicit.resolve(), blas.resolve()})
            import hashlib
            self.assertEqual(next(item["sha256"] for item in records if Path(item["path"]) == explicit.resolve()),
                             hashlib.sha256(explicit.read_bytes()).hexdigest())
            self.assertEqual(next(item["sha256"] for item in records if Path(item["path"]) == blas.resolve()),
                             hashlib.sha256(blas.read_bytes()).hexdigest())

    def test_explicit_reference_tolerance_records_actual_values_and_gates(self):
        with patch.object(bench, "embedding", self.embedding):
            result = bench.correctness("unused", "unused", False, .999, .9999)
        self.assertAlmostEqual(result["duplicate_cosine"], self.observed)
        self.assertEqual(result["duplicate_consistency_gate"], .9999)
        self.assertAlmostEqual(result["reversed_min_cosine"], 1)
        self.assertEqual(result["reversed_consistency_gate"], .9999)

    def test_diagnostic_reference_collects_strict_failure_and_other_checks(self):
        with patch.object(bench, "embedding", self.embedding):
            result = bench.correctness("unused", "unused", False, measure_reference_after_quality_failure=True)
        self.assertFalse(result["quality_pass"])
        self.assertFalse(result["numerical_checks"]["duplicate"]["passed"])
        self.assertEqual(result["numerical_checks"]["duplicate"]["threshold"], .99999)
        self.assertEqual(set(result["numerical_checks"]), {"batch", "reversed", "duplicate", "retrieval"})
        self.assertTrue(result["numerical_checks"]["retrieval"]["passed"])

    def test_diagnostic_option_cannot_bypass_candidate_failure(self):
        with patch.object(bench, "embedding", self.embedding):
            with self.assertRaises(AssertionError) as caught:
                bench.correctness("unused", "unused", True, measure_reference_after_quality_failure=True)
        self.assertEqual(caught.exception.args[0][0], "duplicate inputs")
        self.assertEqual(caught.exception.args[0][2], .99999)

    def test_all_reference_numerical_failures_record_cosines_and_rankings(self):
        with patch.object(bench, "embedding", self.embedding), patch.object(bench, "cosine", return_value=.5):
            result = bench.correctness("unused", "unused", False, measure_reference_after_quality_failure=True)
        self.assertFalse(result["quality_pass"])
        self.assertTrue(all(not check["passed"] for check in result["numerical_checks"].values()))
        self.assertEqual(result["numerical_checks"]["batch"]["cosines"], [.5] * len(bench.TEXTS))
        self.assertEqual(result["retrieval_cosines"]["add_function"], dict(positive=.5, negative=.5))

    def test_failed_reference_is_timed_but_candidate_is_not(self):
        args = SimpleNamespace(model="unused", reference_min_batch_cosine=.9999, reference_min_duplicate_cosine=.99999,
                               measure_reference_after_quality_failure=True, prewarm_seconds=0, warmup=0, repeats=1,
                               minimum_measure_seconds=0, cases=None)
        process = Mock(pid=123, returncode=0)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as output, patch.object(bench.subprocess, "Popen", return_value=process), \
             patch.object(bench, "request", return_value=(200, {})), patch.object(bench, "embedding", self.embedding), \
             patch.object(bench.os, "killpg"), patch.object(bench, "benchmark", return_value={}) as timing:
            result = bench.run_engine("llama", ["unused"], "unused", args, Path(output))
            self.assertFalse(result["correctness"]["quality_pass"])
            timing.assert_called_once()
            timing.reset_mock()
            with self.assertRaises(AssertionError):
                bench.run_engine("tensorsharp", ["unused"], "unused", args, Path(output))
            timing.assert_not_called()


class ServerMemoryAndCleanupTests(unittest.TestCase):
    def test_linux_memory_retains_byte_units_and_main_process_scope(self):
        status = "Name:\tserver\nVmSize:\t8192 kB\nVmRSS:\t2048 kB\nVmHWM:\t3072 kB\nVmSwap:\t0 kB\n"
        with patch.object(bench.platform, "system", return_value="Linux"), \
             patch.object(bench.Path, "read_text", return_value=status):
            result = bench.process_memory(42)
        self.assertTrue(result["available"])
        self.assertEqual(result["resident_bytes"], 2048 * 1024)
        self.assertEqual(result["high_water_resident_bytes"], 3072 * 1024)
        self.assertEqual(result["virtual_bytes"], 8192 * 1024)
        self.assertEqual(result["swap_bytes"], 0)
        self.assertEqual(result["scope"], "main-server process")

    def test_unavailable_memory_is_not_reported_as_zero(self):
        with patch.object(bench.platform, "system", return_value="Linux"), \
             patch.object(bench.Path, "read_text", side_effect=FileNotFoundError):
            result = bench.process_memory(42)
        self.assertFalse(result["available"])
        self.assertNotIn("resident_bytes", result)
        with patch.object(bench.platform, "system", return_value="Darwin"), \
             patch.object(bench.Path, "read_text") as read:
            self.assertFalse(bench.process_memory(42)["available"])
            read.assert_not_called()

    def test_snapshots_are_outside_measurements_and_cleanup_is_recorded(self):
        args = SimpleNamespace(model="unused", reference_min_batch_cosine=.9999, reference_min_duplicate_cosine=.99999,
                               measure_reference_after_quality_failure=False, prewarm_seconds=0, warmup=0, repeats=1,
                               minimum_measure_seconds=0, cases=None)
        order = []
        process = Mock(pid=123, returncode=0)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as output, patch.object(bench.subprocess, "Popen", return_value=process), \
             patch.object(bench, "request", return_value=(200, {})), patch.object(bench.os, "killpg"), \
             patch.object(bench, "correctness", side_effect=lambda *x: order.append("correctness") or dict(quality_pass=True)), \
             patch.object(bench, "benchmark", side_effect=lambda *x: order.append("timings") or {}), \
             patch.object(bench, "process_memory", side_effect=lambda *x: order.append("memory") or dict(available=False)):
            result = bench.run_engine("candidate", ["unused"], "unused", args, Path(output))
        self.assertEqual(order, ["memory", "correctness", "memory", "timings", "memory"])
        self.assertEqual(result["server_exit_code"], 0)
        self.assertEqual([s["phase"] for s in result["memory_snapshots"]],
                         ["ready", "after-correctness-and-prewarm", "after-all-measurements"])

    def test_cleanup_failure_cannot_be_a_passing_benchmark(self):
        args = SimpleNamespace(model="unused", reference_min_batch_cosine=.9999, reference_min_duplicate_cosine=.99999,
                               measure_reference_after_quality_failure=False, prewarm_seconds=0, warmup=0, repeats=1,
                               minimum_measure_seconds=0, cases=None)
        process = Mock(pid=123, returncode=134)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as output, patch.object(bench.subprocess, "Popen", return_value=process), \
             patch.object(bench, "request", return_value=(200, {})), patch.object(bench.os, "killpg"), \
             patch.object(bench, "correctness", return_value=dict(quality_pass=True)), \
             patch.object(bench, "benchmark", return_value={}), patch.object(bench, "process_memory", return_value=dict(available=False)):
            with self.assertRaisesRegex(RuntimeError, "failed graceful shutdown: 134"):
                bench.run_engine("candidate", ["unused"], "unused", args, Path(output))


class ReferenceReportingTests(unittest.TestCase):
    def test_changed_runtime_binary_invalidates_otherwise_passing_report(self):
        def engine(name, command, url, args, output):
            return dict(correctness=dict(vectors=[[1, 0]], usage=dict(prompt_tokens=2), quality_pass=True),
                        benchmarks=dict(single_short=dict(median_ms=1)))

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"mock model")
            argv = ["bench", "--tensorsharp-command", '["candidate"]', "--llama-command", '["reference"]',
                    "--model", "test", "--model-file", str(model), "--output", directory]
            initial = [{"path": "candidate", "sha256": "before"}]
            changed = [{"path": "candidate", "sha256": "after"}]
            with patch.object(sys, "argv", argv), patch.object(bench, "EXTRA_SCENARIOS", {}), \
                 patch.object(bench, "run_engine", side_effect=engine), \
                 patch.object(bench, "command_artifacts", side_effect=[initial, [], changed, []]):
                with self.assertRaisesRegex(RuntimeError, "Runtime binaries changed"):
                    bench.main()
            import json
            result = json.loads((Path(directory) / "results.json").read_text())
            self.assertTrue(result["comparison"]["quality_pass"])
            self.assertFalse(result["binary_artifacts_unchanged"])
            self.assertFalse(result["strict_pass"])

    def test_reference_failure_still_fails_when_cross_engine_and_timings_pass(self):
        def engine(name, command, url, args, output):
            return dict(correctness=dict(vectors=[[1, 0]], usage=dict(prompt_tokens=2), quality_pass=name != "llama"),
                        benchmarks=dict(single_short=dict(median_ms=1)))

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"mock model")
            argv = ["bench", "--tensorsharp-command", '["candidate"]', "--llama-command", '["reference"]',
                    "--model", "test", "--model-file", str(model), "--output", directory,
                    "--measure-reference-after-quality-failure", "--require-performance"]
            with patch.object(sys, "argv", argv), patch.object(bench, "EXTRA_SCENARIOS", {}), \
                 patch.object(bench, "run_engine", side_effect=engine), patch.object(bench, "command_artifacts", return_value=[]):
                with self.assertRaisesRegex(AssertionError, "Strict engine numerical checks failed"):
                    bench.main()
            import json
            result = json.loads((Path(directory) / "results.json").read_text())
            self.assertTrue(result["comparison"]["cross_engine_quality_pass"])
            self.assertTrue(result["comparison"]["performance_pass"])
            self.assertFalse(result["comparison"]["quality_pass"])
            self.assertFalse(result["strict_pass"])

    def test_all_16_raw_metrics_survive_early_8192_failure_and_strict_run_fails(self):
        cases = {"tokens_8192": [[1, 2]], **{f"raw_{i}": [[i, 2]] for i in range(15)}}

        def engine(name, command, url, args, output):
            rows = [dict(prompt_tokens=2, vectors=[[0, 1] if name == "llama" and key == "tokens_8192" else [1, 0]], median_ms=1)
                    for key in cases]
            return dict(correctness=dict(vectors=[[1, 0]], usage=dict(prompt_tokens=2), quality_pass=name != "llama"),
                        benchmarks=dict(zip(cases, rows)))

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"mock model")
            argv = ["bench", "--tensorsharp-command", '["candidate"]', "--llama-command", '["reference"]',
                    "--model", "test", "--model-file", str(model), "--output", directory,
                    "--measure-reference-after-quality-failure"]
            with patch.object(sys, "argv", argv), patch.object(bench, "EXTRA_SCENARIOS", cases), \
                 patch.object(bench, "run_engine", side_effect=engine), patch.object(bench, "command_artifacts", return_value=[]):
                with self.assertRaises(AssertionError):
                    bench.main()
            import json
            result = json.loads((Path(directory) / "results.json").read_text())
            checks = result["comparison"]["extra_scenarios"]
            self.assertEqual(len(checks), 16)
            self.assertFalse(checks["tokens_8192"]["passed"])
            self.assertEqual(checks["tokens_8192"]["threshold"], .999)
            self.assertEqual(checks["tokens_8192"]["cosines"], [0])
            self.assertTrue(checks["raw_14"]["passed"])
            self.assertFalse(result["strict_pass"])
            self.assertFalse(result["comparison"]["quality_pass"])
            self.assertEqual(set(result["engines"]), {"llama", "tensorsharp"})

    def test_model_raw_reference_failure_records_all_cases_candidate_stops(self):
        cases = [dict(text=f"text {i}", tokens=[i, 2]) for i in range(16)]
        unit = [1.0] + [0.0] * 767
        other = [0.0, 1.0] + [0.0] * 766

        def embedding(url, model, inputs):
            return [other if inputs == [0, 2] else unit], dict(prompt_tokens=2)

        def base():
            return dict(dimensions=768, numerical_checks={}, quality_pass=True)

        with patch.object(bench, "embedding", embedding):
            result = http.add_model_correctness(bench, base(), "unused", "unused", False, cases, True)
            self.assertEqual(len(result["raw_token_parity"]), 16)
            self.assertFalse(result["raw_token_parity"][0]["passed"])
            self.assertEqual(result["raw_token_parity"][0]["threshold"], .99999)
            self.assertTrue(result["raw_token_parity"][-1]["passed"])
            self.assertFalse(result["quality_pass"])
            with self.assertRaises(AssertionError):
                http.add_model_correctness(bench, base(), "unused", "unused", True, cases, True)


if __name__ == "__main__":
    unittest.main()
