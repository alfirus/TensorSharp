import importlib.util
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PATH = Path(__file__).resolve().parents[1] / "glm_gpu_clock_telemetry.py"
SPEC = importlib.util.spec_from_file_location("glm_gpu_clock_telemetry", PATH)
GPU = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GPU)
DEVICE = "GPU-12345678-1234-1234-1234-123456789abc"
STAMP = "2026/10/23 12:00:01.000"


def csv_line(**changes):
    values = dict(zip(GPU.FIELDS, (STAMP, DEVICE, "P0", "1980", "9001", "88.25", "120", "72", "40",
        "65", "13200", "4", "4", "8", "8", "0x0000000000000000", "Not Active", "Not Active",
        "Not Active", "Not Active", "Not Active")))
    values.update(changes)
    return ", ".join(values[key] for key in GPU.FIELDS) + "\n"


class SampleTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            path.write_text(text)
            return GPU.parse_samples(path, DEVICE, 0)

    def test_raw_missing_sensors_are_unavailable_not_zero(self):
        report = self.parse(csv_line(**{"power.draw": "N/A", "clocks.current.memory": "[Not Supported]"}))
        self.assertEqual(report["malformed_rows"], 0)
        self.assertEqual(report["samples"][0]["raw"]["power.draw"], "N/A")
        self.assertNotIn("power.draw", report["samples"][0]["numeric"])
        self.assertEqual(report["unavailable_fields"]["clocks.current.memory"], 1)

    def test_zero_is_a_real_observation_and_event_reasons_stay_raw(self):
        report = self.parse(csv_line(**{"utilization.gpu": "0", "clocks_event_reasons.sw_power_cap": "Active"}))
        self.assertEqual(report["samples"][0]["numeric"]["utilization.gpu"], 0)
        self.assertEqual(report["samples"][0]["raw"]["clocks_event_reasons.sw_power_cap"], "Active")

    def test_wrong_device_malformed_nonfinite_and_reversed_rows_are_not_certified(self):
        for text in (csv_line(uuid=DEVICE[:-1] + "d"), "not,csv\n", csv_line(**{"power.draw": "nan"}),
                     csv_line(**{"utilization.gpu": "101"}), csv_line(pstate="broken"),
                     csv_line(**{"clocks_event_reasons.active": "not-a-mask"}),
                     csv_line(**{"clocks_event_reasons.hw_slowdown": "maybe"})):
            with self.subTest(text=text):
                self.assertEqual(self.parse(text)["malformed_rows"], 1)
        report = self.parse(csv_line() + csv_line(timestamp="2026/10/23 12:00:00.000"))
        self.assertEqual(len(report["samples"]), 1)
        self.assertEqual(report["malformed_rows"], 1)

    def test_row_and_byte_limits_bound_untrusted_input(self):
        with mock.patch.object(GPU, "MAX_ROWS", 2):
            report = self.parse(csv_line() * 3)
        self.assertEqual(len(report["samples"]), 2)
        self.assertIn("row allowance", report["errors"][0])
        with mock.patch.object(GPU, "MAX_BYTES", 1):
            report = self.parse(csv_line())
        self.assertEqual(report["samples"], [])

    def test_request_windows_exclude_external_samples_and_do_not_infer_phases(self):
        parsed = self.parse(csv_line(timestamp="2026/10/23 12:00:00.000") + csv_line()
                            + csv_line(timestamp="2026/10/23 12:00:02.000"))
        stamp = int(datetime(2026, 10, 23, 12, 0, 1).replace(tzinfo=GPU.timezone.utc).timestamp() * 1e9)
        window = GPU.request_window(stamp - 500_000_000, 100, stamp + 500_000_000, 1_000_000_100)
        result = GPU.summarize_window(parsed["samples"], window)
        self.assertTrue(result["valid"] and result["core_observed"])
        self.assertEqual(result["sample_count"], 1)
        self.assertFalse(result["exact_prefill_decode_alignment"])
        self.assertEqual(result["numeric"]["power.draw"]["median"], 88.25)
        self.assertEqual(result["pstate_histogram"], {"P0": 1})

    def test_reversed_missing_and_clock_jumped_client_windows_are_invalid(self):
        for window in (None, {}, GPU.request_window(10, 10, 5, 20),
                       GPU.request_window(0, 0, 10_000_000_000, 1_000_000_000)):
            self.assertFalse(GPU.summarize_window([], window)["valid"])


class FakeChild:
    assigned_before_resume = True
    def __init__(self, command, stdout, stderr):
        self.command, self.returncode, self.stop_calls = command, None, 0
        stdout.write(csv_line().encode())
    def poll(self):
        return self.returncode
    def stop(self):
        self.stop_calls += 1
        requested = self.returncode is None
        if requested:
            self.returncode = 1
        return dict(completed=True, requested=requested, pid=1234, exit_code=self.returncode,
                    method="fake_owned_child", errors=[])


class OwnerTests(unittest.TestCase):
    def make(self, root, child=FakeChild, identity=None):
        executable = Path(root) / "nvidia-smi.exe"
        executable.write_bytes(b"fake executable; never launched")
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, identity or DEVICE + "\n", ""))
        owner = GPU.GpuClockTelemetry(root, which=lambda _: str(executable), run=run,
                                      child_factory=child, platform="win32")
        return owner, run

    def test_command_is_one_persistent_read_only_exact_uuid_child_and_cleanup_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            owner, run = self.make(root)
            self.assertTrue(owner.start())
            self.assertEqual(run.call_count, 1)
            command = owner.child.command
            self.assertEqual(command[1], "--id=" + DEVICE)
            self.assertEqual(command[-2:], ["-lms", "200"])
            self.assertTrue(all(part.startswith(("--id=", "--query-gpu=", "--format=")) for part in command[1:-2]))
            self.assertNotIn("--power-limit", " ".join(command))
            self.assertNotIn("--lock", " ".join(command))
            self.assertTrue(owner.finish()["cleanup"]["completed"])
            self.assertIs(owner.finish(), owner.report)
            self.assertEqual(owner.child.stop_calls, 1)
            self.assertFalse(owner.report["timing_eligible"])
            self.assertFalse(owner.report["completed"])  # No measured request windows.

    def test_platform_or_binary_unavailable_does_not_launch(self):
        with tempfile.TemporaryDirectory() as root:
            run = mock.Mock()
            owner = GPU.GpuClockTelemetry(root, platform="linux", run=run)
            self.assertFalse(owner.start())
            run.assert_not_called()
            self.assertEqual(owner.finish()["status"], "unavailable")
            owner = GPU.GpuClockTelemetry(root, platform="win32", which=lambda _: None, run=run)
            self.assertFalse(owner.start())
            run.assert_not_called()

    def test_multiple_gpu_identity_and_unverified_job_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            owner, _ = self.make(root, identity=DEVICE + "\n" + DEVICE[:-1] + "d\n")
            self.assertFalse(owner.start())
            self.assertIsNone(owner.child)
        class Unverified(FakeChild):
            assigned_before_resume = False
        with tempfile.TemporaryDirectory() as root:
            owner, _ = self.make(root, Unverified)
            self.assertFalse(owner.start())
            self.assertEqual(owner.child.stop_calls, 1)

    def test_deadline_and_early_exit_cannot_be_reported_completed(self):
        with tempfile.TemporaryDirectory() as root:
            owner, _ = self.make(root)
            owner.maximum = 1
            self.assertTrue(owner.start())
            owner.report["start_monotonic_ns"] -= 2_000_000_000
            owner.thread.join(timeout=2)
            self.assertEqual(owner.finish()["stop_reason"], "lifetime_bound")
            self.assertFalse(owner.report["completed"])
        with tempfile.TemporaryDirectory() as root:
            owner, _ = self.make(root)
            self.assertTrue(owner.start())
            owner.child.returncode = 7
            self.assertEqual(owner.finish()["stop_reason"], "early_child_exit")

    def test_actual_request_clock_observations_and_changed_tool_identity(self):
        zone = datetime.now().astimezone().tzinfo
        base = int(datetime(2026, 10, 23, 12).replace(tzinfo=zone).timestamp() * 1e9)
        for changed, outside in ((False, False), (True, False), (False, True)):
            with self.subTest(changed=changed, outside=outside), tempfile.TemporaryDirectory() as root:
                owner, _ = self.make(root)
                with mock.patch.object(GPU.time, "time_ns", return_value=base) as wall, mock.patch.object(
                        GPU.time, "monotonic_ns", return_value=100) as mono:
                    self.assertTrue(owner.start())
                    window = GPU.request_window(base + 500_000_000, 500_000_100,
                                                base + 1_500_000_000, 1_500_000_100)
                    if outside:
                        window = GPU.request_window(base - 500_000_000, 500_000_100,
                                                    base + 1_500_000_000, 2_500_000_100)
                    wall.return_value, mono.return_value = base + 2_000_000_000, 2_000_000_100
                    if changed:
                        Path(owner.report["tool_path"]).write_bytes(b"changed executable")
                    result = owner.finish([dict(name="decode", round=0, request_window=window)])
                self.assertEqual(result["clock_power_observed"], not outside)
                self.assertEqual(result["completed"], not changed and not outside)

    def test_cleanup_failure_and_outside_owner_window_are_not_passes(self):
        class FailedCleanup(FakeChild):
            def stop(self):
                result = super().stop()
                result["completed"] = False
                return result
        with tempfile.TemporaryDirectory() as root:
            owner, _ = self.make(root, FailedCleanup)
            self.assertTrue(owner.start())
            self.assertFalse(owner.finish()["cleanup"]["completed"])
            self.assertFalse(owner.report["completed"])
            self.assertEqual(owner.child.stop_calls, GPU.MAX_CLEANUP_ATTEMPTS)
            owner.finish()
            self.assertEqual(owner.child.stop_calls, GPU.MAX_CLEANUP_ATTEMPTS)
        with self.assertRaises(ValueError):
            GPU.GpuClockTelemetry("unused", max_seconds=901)

    def test_cleanup_retry_keeps_same_child_and_records_initial_failure_without_pass(self):
        class RetryCleanup(FakeChild):
            def stop(self):
                result = super().stop()
                result["completed"] = self.stop_calls > 1
                return result
        with tempfile.TemporaryDirectory() as root:
            owner, run = self.make(root, RetryCleanup)
            self.assertTrue(owner.start())
            exact_child = owner.child
            result = owner.finish()
            self.assertIs(owner.child, exact_child)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(exact_child.stop_calls, 2)
            self.assertTrue(result["cleanup"]["completed"])
            self.assertTrue(result["cleanup_had_failure"])
            self.assertEqual(len(result["cleanup_history"]), 2)
            self.assertFalse(result["completed"])


@unittest.skipUnless(sys.platform == "win32", "Windows Job lifecycle unavailable; no GPU required")
class WindowsJobCpuTests(unittest.TestCase):
    def test_real_suspended_cpu_child_is_assigned_before_run_and_reaped(self):
        with tempfile.TemporaryDirectory() as root:
            with (Path(root) / "out").open("wb") as stdout, (Path(root) / "err").open("wb") as stderr:
                child = GPU._WindowsJobChild([sys.executable, "-c", "import time; time.sleep(60)"], stdout, stderr)
                try:
                    self.assertTrue(child.assigned_before_resume)
                    self.assertIsNone(child.poll())
                finally:
                    stopped = child.stop()
                self.assertTrue(stopped["completed"], stopped)
                self.assertIsNotNone(child.returncode)

    def test_parent_exit_closes_job_and_does_not_leave_cpu_child(self):
        import ctypes as c
        from ctypes import wintypes as w
        code = ("import importlib.util,os,sys; from pathlib import Path; "
                "s=importlib.util.spec_from_file_location('gpu',sys.argv[1]); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "f=open(os.devnull,'wb'); "
                "p=m._WindowsJobChild([sys.executable,'-c','import time; time.sleep(60)'],f,f); "
                "print(p.pid,flush=True); os._exit(0)")
        proc = subprocess.Popen([sys.executable, "-c", code, str(PATH)], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, creationflags=0x08000000)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except BaseException:
            proc.kill(); proc.wait(timeout=5)
            raise
        self.assertEqual(proc.returncode, 0, stderr)
        pid = int(stdout.strip())
        kernel = c.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = (w.DWORD, w.BOOL, w.DWORD), w.HANDLE
        kernel.WaitForSingleObject.argtypes, kernel.WaitForSingleObject.restype = (w.HANDLE, w.DWORD), w.DWORD
        kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = (w.HANDLE,), w.BOOL
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            self.assertEqual(c.get_last_error(), 87)  # Already gone, not access denied.
        else:
            try:
                self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
            finally:
                self.assertTrue(kernel.CloseHandle(handle))


if __name__ == "__main__":
    unittest.main()
