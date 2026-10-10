import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest


SPEC = importlib.util.spec_from_file_location("bounded_probe", Path(__file__).parents[1] / "run-bounded-probe.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class BoundedProbeTests(unittest.TestCase):
    def test_telemetry_finish_error_keeps_durable_child_outcome(self):
        class BrokenFinish:
            report = {'passed': False}
            error = None
            def prepare(self, output): pass
            def start(self, child): pass
            def finish(self): raise RuntimeError('injected finish failure')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            result = PROBE.run([sys.executable, '-c', 'print("finished")'], output, 10, BrokenFinish())
            self.assertEqual(0, result['exit_code'])
            self.assertTrue(result['complete'])
            self.assertFalse(result['telemetry']['passed'])
            self.assertIn('injected finish failure', result['cleanup_errors'][0])
            self.assertEqual(result, json.loads((output/'execution.json').read_text()))

    def test_telemetry_start_failure_reaps_child_before_reraising(self):
        class BrokenStart:
            report = {'passed': False}
            error = None
            def prepare(self, output): pass
            def start(self, child):
                self.child = child
                raise RuntimeError('injected start failure')
            def finish(self): return self.report
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            telemetry = BrokenStart()
            with self.assertRaisesRegex(RuntimeError, 'start failure'):
                PROBE.run([sys.executable, '-c', 'import time; time.sleep(30)'], output, 10, telemetry)
            self.assertIsNotNone(telemetry.child.poll())
            result = json.loads((output/'execution.json').read_text())
            self.assertFalse(result['complete'])
            self.assertIn('start failure', result['error'])
            self.assertIsNotNone(result['exit_code'])

    def test_success_retains_stdout_and_does_not_claim_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = PROBE.run([sys.executable, "-c", "print('complete evidence')"], output, 10)
            self.assertTrue(result["complete"])
            self.assertFalse(result["timed_out"])
            self.assertEqual(0, result["exit_code"])
            self.assertIsNone(result["quality_passed"])
            self.assertIn("complete evidence", (output / "process.log").read_text())
            self.assertEqual(result, json.loads((output / "execution.json").read_text()))

    def test_nonzero_exit_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            result = PROBE.run([sys.executable, "-c", "raise SystemExit(17)"], Path(directory) / "run", 10)
            self.assertTrue(result["complete"])
            self.assertEqual(17, result["exit_code"])
            self.assertFalse(result["timed_out"])

    def test_timeout_waits_for_owned_child_exit_and_keeps_incomplete_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = PROBE.run([sys.executable, "-c", "import time; time.sleep(30)"], output, 0.2)
            self.assertTrue(result["timed_out"])
            self.assertFalse(result["complete"])
            self.assertIsNotNone(result["exit_code"])
            self.assertNotEqual(0, result["exit_code"])
            self.assertEqual(result, json.loads((output / "execution.json").read_text()))

    def test_live_native_pin_rejects_wrong_library_and_multiple_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / 'GgmlOps.dll'
            native.write_bytes(b'fixture-native')
            paths = [str(native)]
            fake = SimpleNamespace(native_module_paths=lambda pid: paths)
            wrong = PROBE.ProbeTelemetry(root, '0' * 64, fake)
            wrong.output = root
            with self.assertRaisesRegex(ValueError, 'required pin'):
                wrong.observe_native(123)
            self.assertIsNone(wrong.native)
            right = PROBE.ProbeTelemetry(root, PROBE.digest(native), fake)
            right.output = root
            right.observe_native(123)
            self.assertEqual(str(native.resolve()), right.native['path'])
            paths.append(str(root / 'another.dll'))
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                right.observe_native(123)

    def test_missing_samples_and_modified_managed_file_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'TensorSharp.Models.dll').write_bytes(b'before')
            native = root / 'GgmlOps.dll'
            native.write_bytes(b'native')
            telemetry = PROBE.ProbeTelemetry(root, PROBE.digest(native), SimpleNamespace(native_module_paths=lambda pid: [str(native)]))
            telemetry.prepare(root)
            telemetry.observe_native(123)
            (root / 'TensorSharp.Models.dll').write_bytes(b'after')
            result = telemetry.finish()
            self.assertTrue(result['native_verified'])
            self.assertFalse(result['managed_files_unchanged'])
            self.assertIsNone(result['memory']['peak_rss_bytes'])
            self.assertIsNone(result['memory']['peak_board_mib'])
            self.assertFalse(result['passed'])

    def test_telemetry_rejection_stops_owned_child_without_claiming_completion(self):
        class Rejection:
            report = {'passed': False}
            error = None
            def prepare(self, output): pass
            def start(self, child): self.error = 'injected native pin mismatch'
            def finish(self): return self.report
        with tempfile.TemporaryDirectory() as directory:
            started = time.monotonic()
            result = PROBE.run([sys.executable, '-c', 'import time; time.sleep(30)'],
                Path(directory) / 'run', 10, Rejection())
            self.assertLess(time.monotonic() - started, 5)
            self.assertFalse(result['complete'])
            self.assertFalse(result['timed_out'])
            self.assertNotEqual(0, result['exit_code'])
            self.assertIsNone(result['quality_passed'])
            self.assertIn('mismatch', result['error'])


if __name__ == "__main__":
    unittest.main()
