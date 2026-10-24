"""Bounded Windows GPU observations; sampled runs are never performance controls.

Only the opt-in owner starts a child. No import-time hardware calls. Field names
match the saved installed --help-query-gpu, not an assumed driver capability.
Windows Job ownership follows run-memory-job.py, without memory/CPU/WS limits.
"""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time


FIELDS = (
    "timestamp", "uuid", "pstate", "clocks.current.sm", "clocks.current.memory",
    "power.draw", "power.limit", "utilization.gpu", "utilization.memory",
    "temperature.gpu", "memory.used", "pcie.link.gen.gpucurrent",
    "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max",
    "clocks_event_reasons.active", "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_slowdown", "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown", "clocks_event_reasons.hw_power_brake_slowdown",
)
NUMERIC = FIELDS[3:15]
OBSERVED_CORE = ("clocks.current.sm", "clocks.current.memory", "power.draw", "utilization.gpu")
UNITS = {key: ("MHz" if key.startswith("clocks.current.") else
               "W" if key.startswith("power.") else
               "percent" if key.startswith("utilization.") else
               "degrees C" if key == "temperature.gpu" else
               "MiB" if key == "memory.used" else "integer") for key in NUMERIC}
UNAVAILABLE = {"N/A", "[N/A]", "Not Supported", "[Not Supported]", "Unknown", "[Unknown]"}
MAX_SECONDS, MAX_BYTES, MAX_ROWS = 900, 8 << 20, 5000
MAX_CLEANUP_ATTEMPTS = 3
QUERY_HELP_SHA256 = "3514C125B2A129C9B000A669FE6A294AC381D5DF6520372B7D473D2AB5E018F9"
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")
SCOPE = "whole_board_sampled_observations_not_kernel_cpu_exclusive_bandwidth_or_performance"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_window(start_unix_ns, start_monotonic_ns, end_unix_ns, end_monotonic_ns):
    """Client HTTP bounds only; contains no inferred prefill/decode boundary."""
    return dict(start_unix_ns=start_unix_ns, start_monotonic_ns=start_monotonic_ns,
                end_unix_ns=end_unix_ns, end_monotonic_ns=end_monotonic_ns)


def parse_samples(path, expected_uuid, utc_offset_seconds):
    path = Path(path)
    result = dict(samples=[], malformed_rows=0, unavailable_fields={key: 0 for key in FIELDS},
                  errors=[], bytes=path.stat().st_size)
    if result["bytes"] > MAX_BYTES:
        result["errors"].append("CSV exceeds bounded byte allowance")
        return result
    previous = None
    zone = timezone(timedelta(seconds=utc_offset_seconds))
    with path.open(encoding="utf-8-sig", errors="strict", newline="") as stream:
        for index, columns in enumerate(csv.reader(stream)):
            if index >= MAX_ROWS:
                result["errors"].append("CSV exceeds bounded row allowance")
                break
            try:
                if len(columns) != len(FIELDS):
                    raise ValueError("Unexpected CSV column count")
                raw = dict(zip(FIELDS, (value.strip() for value in columns)))
                if raw["uuid"] != expected_uuid:
                    raise ValueError("Sample UUID differs from exact selected physical GPU")
                stamp = datetime.strptime(raw["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=zone).timestamp()
                if previous is not None and stamp < previous:
                    raise ValueError("Sample timestamps reversed")
                numeric = {}
                for key in FIELDS:
                    if raw[key] in UNAVAILABLE:
                        result["unavailable_fields"][key] += 1
                    elif key in NUMERIC:
                        value = float(raw[key])
                        if not math.isfinite(value) or value < 0 or (key.startswith("utilization.") and value > 100):
                            raise ValueError("Invalid numeric observation")
                        numeric[key] = value
                    elif key == "pstate" and not re.fullmatch(r"P(?:[0-9]|1[0-5])", raw[key]):
                        raise ValueError("Invalid pstate observation")
                    elif key == "clocks_event_reasons.active" and not re.fullmatch(r"(?:0x[0-9a-fA-F]+|[0-9]+)", raw[key]):
                        raise ValueError("Invalid raw clock-event mask")
                    elif key.startswith("clocks_event_reasons.") and key != "clocks_event_reasons.active" and raw[key] not in ("Active", "Not Active"):
                        raise ValueError("Invalid raw clock-event state")
                previous = stamp
                result["samples"].append(dict(unix=stamp, raw=raw, numeric=numeric))
            except (ValueError, OverflowError) as error:
                result["malformed_rows"] += 1
                if len(result["errors"]) < 8:
                    result["errors"].append(str(error))
    return result


def summarize_window(samples, window):
    result = dict(valid=False, sample_count=0, core_observed=False,
                  exact_prefill_decode_alignment=False, error=None)
    keys = ("start_unix_ns", "start_monotonic_ns", "end_unix_ns", "end_monotonic_ns")
    if not isinstance(window, dict) or any(type(window.get(key)) is not int or window[key] < 0 for key in keys):
        result["error"] = "Missing actual client request time anchors"
        return result
    wall = (window["end_unix_ns"] - window["start_unix_ns"]) / 1e9
    monotonic = (window["end_monotonic_ns"] - window["start_monotonic_ns"]) / 1e9
    if wall <= 0 or monotonic <= 0 or abs(wall - monotonic) > max(.5, monotonic * .001):
        result["error"] = "Reversed request interval or wall-clock discontinuity"
        return result
    start, end = window["start_unix_ns"] / 1e9, window["end_unix_ns"] / 1e9
    rows = [row for row in samples if start <= row["unix"] <= end]
    result.update(valid=True, sample_count=len(rows), request_wall_seconds=wall,
                  first_sample_after_start_seconds=rows[0]["unix"] - start if rows else None,
                  last_sample_before_end_seconds=end - rows[-1]["unix"] if rows else None,
                  maximum_sample_gap_seconds=max((b["unix"] - a["unix"] for a, b in zip(rows, rows[1:])), default=None),
                  numeric={}, pstate_histogram={}, clock_event_histograms={},
                  unavailable_fields={key: 0 for key in FIELDS})
    for key in NUMERIC:
        values = [row["numeric"][key] for row in rows if key in row["numeric"]]
        result["numeric"][key] = dict(sample_count=len(values), minimum=min(values) if values else None,
                                      median=statistics.median(values) if values else None,
                                      maximum=max(values) if values else None, unit=UNITS[key])
    for row in rows:
        for key, value in row["raw"].items():
            if value in UNAVAILABLE:
                result["unavailable_fields"][key] += 1
            elif key == "pstate":
                counts = result["pstate_histogram"]
                counts[value] = counts.get(value, 0) + 1
            elif key.startswith("clocks_event_reasons."):
                counts = result["clock_event_histograms"].setdefault(key, {})
                counts[value] = counts.get(value, 0) + 1
    result["core_observed"] = bool(result["pstate_histogram"]) and all(
        result["numeric"][key]["sample_count"] > 0 for key in OBSERVED_CORE)
    return result


class _WindowsJobChild:
    """One suspended child, assigned to a verified kill-on-close Job before run."""

    def __init__(self, command, stdout, stderr):
        if sys.platform != "win32":
            raise RuntimeError("Crash-safe Windows sampler ownership unavailable on this platform")
        import ctypes as c
        from ctypes import wintypes as w
        import msvcrt
        self._c, self._w = c, w
        self._job = self._process = self._thread = None
        self.pid, self.returncode, self.assigned_before_resume = None, None, False
        kernel = c.WinDLL("kernel32", use_last_error=True)

        class Basic(c.Structure):
            _fields_ = [("per_process_time", c.c_int64), ("per_job_time", c.c_int64),
                        ("flags", w.DWORD), ("minimum_ws", c.c_size_t), ("maximum_ws", c.c_size_t),
                        ("active_processes", w.DWORD), ("affinity", c.c_size_t),
                        ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class Io(c.Structure):
            _fields_ = [(name, c.c_uint64) for name in ("read", "write", "other", "read_bytes", "write_bytes", "other_bytes")]
        class Extended(c.Structure):
            _fields_ = [("basic", Basic), ("io", Io), ("process_limit", c.c_size_t),
                        ("job_limit", c.c_size_t), ("process_peak", c.c_size_t), ("job_peak", c.c_size_t)]
        class Startup(c.Structure):
            _fields_ = [("cb", w.DWORD), ("reserved", w.LPWSTR), ("desktop", w.LPWSTR), ("title", w.LPWSTR),
                        ("x", w.DWORD), ("y", w.DWORD), ("width", w.DWORD), ("height", w.DWORD),
                        ("xchars", w.DWORD), ("ychars", w.DWORD), ("fill", w.DWORD), ("flags", w.DWORD),
                        ("show", w.WORD), ("reserved_size", w.WORD), ("reserved_data", c.c_void_p),
                        ("stdin", w.HANDLE), ("stdout", w.HANDLE), ("stderr", w.HANDLE)]
        class Process(c.Structure):
            _fields_ = [("process", w.HANDLE), ("thread", w.HANDLE), ("pid", w.DWORD), ("tid", w.DWORD)]
        def api(name, result, *args):
            fn = getattr(kernel, name)
            fn.restype, fn.argtypes = result, args
            return fn
        self._close = api("CloseHandle", w.BOOL, w.HANDLE)
        self._wait = api("WaitForSingleObject", w.DWORD, w.HANDLE, w.DWORD)
        self._code = api("GetExitCodeProcess", w.BOOL, w.HANDLE, c.POINTER(w.DWORD))
        self._terminate = api("TerminateProcess", w.BOOL, w.HANDLE, w.UINT)
        create_job = api("CreateJobObjectW", w.HANDLE, c.c_void_p, w.LPCWSTR)
        set_job = api("SetInformationJobObject", w.BOOL, w.HANDLE, c.c_int, c.c_void_p, w.DWORD)
        query_job = api("QueryInformationJobObject", w.BOOL, w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p)
        assign = api("AssignProcessToJobObject", w.BOOL, w.HANDLE, w.HANDLE)
        member = api("IsProcessInJob", w.BOOL, w.HANDLE, w.HANDLE, c.POINTER(w.BOOL))
        create = api("CreateProcessW", w.BOOL, w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p,
                     w.BOOL, w.DWORD, c.c_void_p, w.LPCWSTR, c.POINTER(Startup), c.POINTER(Process))
        resume = api("ResumeThread", w.DWORD, w.HANDLE)
        def check(ok):
            if not ok:
                raise c.WinError(c.get_last_error())
        try:
            self._job = create_job(None, None)  # Unnamed, non-inherited, sampler only.
            check(self._job)
            limits = Extended()
            limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE only. No resource limits.
            check(set_job(self._job, 9, c.byref(limits), c.sizeof(limits)))
            actual = Extended()
            check(query_job(self._job, 9, c.byref(actual), c.sizeof(actual), None))
            if actual.basic.flags != 0x2000 or actual.process_limit or actual.job_limit:
                raise RuntimeError("Sampler-only Job flags read-back differs")
            with open(os.devnull, "rb") as null:
                handles = [msvcrt.get_osfhandle(f.fileno()) for f in (null, stdout, stderr)]
                inherited = [os.get_handle_inheritable(handle) for handle in handles]
                try:
                    for handle in handles:
                        os.set_handle_inheritable(handle, True)
                    startup = Startup(cb=c.sizeof(Startup), flags=0x100,
                                      stdin=handles[0], stdout=handles[1], stderr=handles[2])
                    proc = Process()
                    line = c.create_unicode_buffer(subprocess.list2cmdline(command))
                    check(create(command[0], line, None, None, True, 0x4 | 0x08000000,
                                 None, None, c.byref(startup), c.byref(proc)))
                    self._process, self._thread, self.pid = proc.process, proc.thread, proc.pid
                    check(assign(self._job, self._process))
                    belongs = w.BOOL()
                    check(member(self._process, self._job, c.byref(belongs)))
                    if not belongs.value:
                        raise RuntimeError("Suspended sampler did not join its owned Job")
                    self.assigned_before_resume = True
                    if resume(self._thread) == 0xffffffff:
                        raise c.WinError(c.get_last_error())
                finally:
                    for handle, value in zip(handles, inherited):
                        os.set_handle_inheritable(handle, value)
            if self._thread:
                check(self._close(self._thread)); self._thread = None
        except BaseException as error:
            cleanup = self.stop()
            if not cleanup["completed"]:
                # Preserve the exact handles for the owner's bounded retry;
                # failed startup must not falsely certify successful cleanup.
                error.owned_gpu_child = self
            raise

    def poll(self):
        if self.returncode is not None or not self._process:
            return self.returncode
        state = self._wait(self._process, 0)
        if state == 258:
            return None
        if state != 0:
            raise self._c.WinError(self._c.get_last_error())
        value = self._w.DWORD()
        if not self._code(self._process, self._c.byref(value)):
            raise self._c.WinError(self._c.get_last_error())
        self.returncode = value.value
        return self.returncode

    def stop(self):
        errors = []
        requested = False
        if self._process:
            try:
                if self.poll() is None:
                    requested = True
                    if not self._terminate(self._process, 1) and self.poll() is None:
                        errors.append("TerminateProcess failed")
            except Exception as error:
                errors.append(str(error))
        # Closing the non-inherited Job also kills any still-owned descendants.
        if self._job:
            if self._close(self._job):
                self._job = None
            else:
                errors.append("Close sampler Job failed")
        if self._process:
            try:
                if self._wait(self._process, 5000) != 0:
                    errors.append("Owned sampler did not exit within cleanup bound")
                else:
                    self.poll()
            except Exception as error:
                errors.append(str(error))
            if self.returncode is not None:
                if self._close(self._process):
                    self._process = None
                else:
                    errors.append("Close sampler process handle failed")
        if self._thread:
            if self._close(self._thread):
                self._thread = None
            else:
                errors.append("Close sampler initial thread failed")
        return dict(completed=not errors and not self._process and not self._job and not self._thread,
                    pid=self.pid, requested=requested, exit_code=self.returncode,
                    method="owned_TerminateProcess_and_kill_on_job_close", errors=errors)


class GpuClockTelemetry:
    def __init__(self, directory, max_seconds=MAX_SECONDS, *, which=shutil.which,
                 run=subprocess.run, child_factory=_WindowsJobChild, platform=None):
        if type(max_seconds) is not int or not 1 <= max_seconds <= MAX_SECONDS:
            raise ValueError("GPU telemetry lifetime must be 1 through 900 seconds")
        self.directory, self.maximum = Path(directory), max_seconds
        self.which, self.run, self.factory = which, run, child_factory
        self.platform = sys.platform if platform is None else platform
        self.child = self.stdout = self.stderr = self.thread = None
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.finished = False
        self.report = dict(requested=True, available=False, completed=False, status="not_started",
            timing_eligible=False, interval_ms=200, max_seconds=max_seconds, fields=list(FIELDS), units=UNITS,
            scope=SCOPE, uuid=None, command=None, tool_path=None, tool_sha256=None,
            module_sha256=sha256(__file__), csv_path=str(self.directory / "gpu-telemetry.csv"),
            field_reference_help_sha256=QUERY_HELP_SHA256,
            stderr_path=str(self.directory / "gpu-telemetry-stderr.log"), sample_count=0,
            malformed_rows=0, unavailable_fields={}, requests=[], errors=[], cleanup=None,
            cleanup_attempts=0, cleanup_history=[], cleanup_had_failure=False,
            start_unix_ns=None, start_monotonic_ns=None, end_unix_ns=None, end_monotonic_ns=None,
            utc_offset_seconds=None, stop_reason=None, crash_safe_job=False,
            observed_lifetime_seconds=None, clock_power_observed=False, exact_prefill_decode_alignment=False)

    def start(self):
        if self.report["status"] != "not_started":
            raise RuntimeError("GPU telemetry owner may start only once")
        try:
            if self.platform != "win32":
                raise RuntimeError("Crash-safe Windows GPU telemetry unavailable on this platform")
            executable = self.which("nvidia-smi")
            if not executable:
                raise RuntimeError("nvidia-smi unavailable")
            executable = str(Path(executable).resolve(strict=True))
            self.report.update(tool_path=executable, tool_sha256=sha256(executable))
            identity_command = [executable, "--query-gpu=uuid", "--format=csv,noheader,nounits"]
            identity = self.run(identity_command, capture_output=True, text=True, timeout=5,
                                creationflags=0x08000000, check=False)
            ids = [line.strip() for line in identity.stdout.splitlines() if line.strip()]
            if identity.returncode != 0 or len(ids) != 1 or not UUID.fullmatch(ids[0]):
                raise RuntimeError("Expected exactly one actual physical NVIDIA GPU UUID")
            self.report["uuid"] = ids[0]
            command = [executable, "--id=" + ids[0], "--query-gpu=" + ",".join(FIELDS),
                       "--format=csv,noheader,nounits", "-lms", "200"]
            self.report["command"] = command
            self.stdout = Path(self.report["csv_path"]).open("xb", buffering=0)
            self.stderr = Path(self.report["stderr_path"]).open("xb", buffering=0)
            self.child = self.factory(command, self.stdout, self.stderr)
            if not self.child.assigned_before_resume:
                raise RuntimeError("Sampler was not assigned to verified kill-on-close Job before run")
            self.report.update(available=True, status="running", crash_safe_job=True,
                start_unix_ns=time.time_ns(), start_monotonic_ns=time.monotonic_ns(),
                utc_offset_seconds=int(datetime.now().astimezone().utcoffset().total_seconds()))
            self.thread = threading.Thread(target=self._watch, name="owned-gpu-telemetry-bound", daemon=True)
            self.thread.start()
            return True
        except Exception as error:
            self.child = getattr(error, "owned_gpu_child", self.child)
            self.report["status"] = "unavailable"
            self.report["errors"].append(str(error))
            self._stop_child("startup_failed")
            return False

    def _stop_child(self, reason):
        with self.lock:
            if ((self.report["cleanup"] is not None and self.report["cleanup"]["completed"])
                    or self.report["cleanup_attempts"] >= MAX_CLEANUP_ATTEMPTS):
                return
            reason = self.report["stop_reason"] or reason
            self.report["cleanup_attempts"] += 1
            try:
                self.report["cleanup"] = self.child.stop() if self.child else dict(completed=True, started=False)
                if reason == "owner_finished" and self.report["cleanup"].get("requested") is False and self.report["cleanup"].get("exit_code") is not None:
                    reason = "early_child_exit"
            except Exception as error:
                self.report["cleanup"] = dict(completed=False, errors=[str(error)])
            finally:
                self.report["stop_reason"] = reason
                for stream in (self.stdout, self.stderr):
                    if stream:
                        try:
                            stream.close()
                        except Exception as error:
                            self.report["cleanup"]["completed"] = False
                            self.report["errors"].append(str(error))
                self.report.update(end_unix_ns=time.time_ns(), end_monotonic_ns=time.monotonic_ns())
                self.report["cleanup_history"].append(self.report["cleanup"])
                if not self.report["cleanup"]["completed"]:
                    self.report["cleanup_had_failure"] = True

    def _watch(self):
        try:
            while not self.stopped.wait(.1):
                if (time.monotonic_ns() - self.report["start_monotonic_ns"]) / 1e9 >= self.maximum:
                    self._stop_child("lifetime_bound")
                    return
                if any(Path(self.report[key]).stat().st_size > MAX_BYTES for key in ("csv_path", "stderr_path")):
                    self._stop_child("file_bound")
                    return
                if self.child.poll() is not None:
                    self._stop_child("early_child_exit")
                    return
        except Exception as error:
            self.report["errors"].append(str(error))
            self._stop_child("watchdog_error")

    def finish(self, records=()):
        if self.finished:
            return self.report
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=6)
        self._stop_child("owner_finished")
        while (not self.report["cleanup"]["completed"]
               and self.report["cleanup_attempts"] < MAX_CLEANUP_ATTEMPTS):
            self._stop_child("cleanup_retry")
        self.finished = True
        if not self.report["available"]:
            return self.report
        try:
            current_offset = int(datetime.now().astimezone().utcoffset().total_seconds())
            if current_offset != self.report["utc_offset_seconds"]:
                raise ValueError("Local timezone offset changed during GPU observations")
            owner_wall = (self.report["end_unix_ns"] - self.report["start_unix_ns"]) / 1e9
            owner_monotonic = (self.report["end_monotonic_ns"] - self.report["start_monotonic_ns"]) / 1e9
            self.report["observed_lifetime_seconds"] = owner_monotonic
            if owner_monotonic > MAX_SECONDS:
                raise ValueError("Sampler exceeded its 900-second diagnostic lifetime allowance")
            if owner_wall < 0 or owner_monotonic < 0 or abs(owner_wall - owner_monotonic) > max(.5, owner_monotonic * .001):
                raise ValueError("Sampler wall-clock discontinuity prevents request alignment")
            parsed = parse_samples(self.report["csv_path"], self.report["uuid"], current_offset)
            self.report.update(sample_count=len(parsed["samples"]), malformed_rows=parsed["malformed_rows"],
                               unavailable_fields=parsed["unavailable_fields"])
            self.report["errors"].extend(parsed["errors"])
            self.report["requests"] = [dict(name=row.get("name"), round=row.get("round"),
                observations=summarize_window(parsed["samples"], row.get("request_window"))) for row in records]
            windows = [row["observations"] for row in self.report["requests"]]
            for row, window in zip(records, windows):
                if window["valid"] and not (self.report["start_unix_ns"] <= row["request_window"]["start_unix_ns"]
                        <= row["request_window"]["end_unix_ns"] <= self.report["end_unix_ns"]):
                    window.update(valid=False, core_observed=False, error="Request window exceeds owned sampler lifetime")
            self.report["clock_power_observed"] = bool(windows) and all(row["core_observed"] for row in windows)
            self.report["completed"] = bool(parsed["samples"]) and not self.report["errors"] and bool(windows) and all(
                row["valid"] and row["sample_count"] > 0 for row in windows) and self.report["cleanup"]["completed"] and not self.report["cleanup_had_failure"] and (
                self.report["stop_reason"] == "owner_finished")
            self.report["status"] = "completed" if self.report["completed"] else "partial"
            if sha256(self.report["tool_path"]) != self.report["tool_sha256"]:
                self.report.update(completed=False, status="partial")
                self.report["errors"].append("Sampler executable identity changed during observation")
        except Exception as error:
            self.report.update(completed=False, status="partial")
            self.report["errors"].append(str(error))
        return self.report
