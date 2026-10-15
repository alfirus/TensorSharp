import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / "embeddinggemma2_boundary_telemetry.py"
SPEC = importlib.util.spec_from_file_location("boundary_telemetry", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BoundaryTelemetryTests(unittest.TestCase):
    def test_real_v1_units_and_fractional_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpu = root / "cpu"
            cpu.mkdir()
            for name, value in {"cpu.stat": "nr_periods 21\nnr_throttled 8\nthrottled_time 1500\n",
                                "cpu.cfs_quota_us": "765000", "cpu.cfs_period_us": "100000"}.items():
                (cpu / name).write_text(value)
            report = MODULE.cpu_snapshot(root)
            self.assertTrue(report["available"])
            self.assertEqual(report["quota_cores"], 7.65)
            self.assertEqual(report["counters"]["throttled_time"], 1500)
            self.assertEqual(report["version"], "v1")

    def test_v2_unlimited_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cpu.stat").write_text("usage_usec 123\nthrottled_usec 9\n")
            (root / "cpu.max").write_text("max 100000\n")
            report = MODULE.cpu_snapshot(root)
            self.assertTrue(report["available"])
            self.assertIsNone(report["quota_cores"])
            self.assertEqual(report["counters"]["throttled_usec"], 9)

    def test_missing_or_invalid_cpu_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(MODULE.cpu_snapshot(root)["available"])
            (root / "cpu.stat").write_text("bad_counter not_integer")
            (root / "cpu.max").write_text("100000 100000")
            self.assertFalse(MODULE.cpu_snapshot(root)["available"])

    def test_cpu_discovery_and_affinity_oserrors_are_unavailable(self):
        with mock.patch.object(MODULE.Path, "is_file", side_effect=OSError("permission denied")):
            self.assertFalse(MODULE.cpu_snapshot()["available"])
        with mock.patch.object(MODULE.os, "sched_getaffinity", create=True, side_effect=OSError("unavailable")):
            self.assertFalse(MODULE.affinity_snapshot()["available"])

    def test_gpu_read_only_command_and_missing_failure(self):
        self.assertFalse(MODULE.gpu_snapshot(which=lambda _: None)["available"])
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "nvidia-smi"
            executable.write_bytes(b"pinned fake executable")
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                return subprocess.CompletedProcess(command, 0,
                    stdout='A40, 580, P0, 1740, 1740, 7000, 40, 100, 0\n', stderr='')
            report = MODULE.gpu_snapshot(which=lambda _: str(executable), run=run)
            self.assertTrue(report["available"])
            self.assertEqual(report["gpu0"]["clocks.current.sm"], "1740")
            self.assertEqual(report["units"]["power.draw"], "W")
            self.assertEqual(commands[0][1], "--id=0")
            self.assertTrue(all(arg.startswith(("--query", "--format", "--id")) for arg in commands[0][1:]))
            failed = MODULE.gpu_snapshot(which=lambda _: str(executable), run=lambda *a, **k:
                subprocess.CompletedProcess(a[0], 1, stdout='', stderr='driver unavailable'))
            self.assertFalse(failed["available"])

    def test_gpu_missing_numeric_values_are_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "nvidia-smi"
            executable.write_bytes(b"pinned fake executable")
            report = MODULE.gpu_snapshot(which=lambda _: str(executable), run=lambda *a, **k:
                subprocess.CompletedProcess(a[0], 0,
                    stdout='A40, 580, P0, N/A, 1740, 7000, 40, [Not Supported], 0\n', stderr=''))
            self.assertTrue(report["available"])
            self.assertEqual(report["gpu0"]["clocks.current.graphics"], "N/A")
            self.assertEqual(report["unavailable_fields"], ["clocks.current.graphics", "power.draw"])


if __name__ == "__main__":
    unittest.main()
