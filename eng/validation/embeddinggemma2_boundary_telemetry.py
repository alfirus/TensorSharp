"""Read-only snapshots outside benchmark intervals; never infer timed clocks."""
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time


GPU_FIELDS = (
    "name", "driver_version", "pstate", "clocks.current.graphics", "clocks.current.sm",
    "clocks.current.memory", "temperature.gpu", "power.draw", "utilization.gpu",
)
GPU_UNITS = {"clocks.current.graphics": "MHz", "clocks.current.sm": "MHz",
             "clocks.current.memory": "MHz", "temperature.gpu": "degrees C",
             "power.draw": "W", "utilization.gpu": "percent"}


def cpu_snapshot(root=Path("/sys/fs/cgroup")):
    root = Path(root)
    layouts = (
        ("v1", root / "cpu", ("cpu.stat", "cpu.cfs_quota_us", "cpu.cfs_period_us")),
        ("v2", root, ("cpu.stat", "cpu.max")),
    )
    for version, directory, names in layouts:
        try:
            if not all((directory / name).is_file() for name in names):
                continue
            values = {name: (directory / name).read_text().strip() for name in names}
            counters = {}
            for line in values["cpu.stat"].splitlines():
                key, value = line.split()
                counters[key] = int(value)
            quota, period = ((values["cpu.cfs_quota_us"], values["cpu.cfs_period_us"])
                             if version == "v1" else values["cpu.max"].split())
            quota = None if quota in ("-1", "max") else int(quota)
            period = int(period)
            if period <= 0 or (quota is not None and quota <= 0):
                raise ValueError("Invalid CPU quota/period")
            return dict(available=True, version=version, directory=str(directory), raw=values,
                        counters=counters, quota_us=quota, period_us=period,
                        quota_cores=None if quota is None else quota / period,
                        counter_units="v1 throttled_time=nanoseconds; v2 throttled_usec=microseconds",
                        scope="Mounted CPU cgroup directory sampled directly, without /proc/self/cgroup membership resolution; counters include all processes in this directory.")
        except (OSError, ValueError) as error:
            return dict(available=False, version=version, error=str(error))
    return dict(available=False, error="No supported cgroup v1/v2 CPU files")


def gpu_snapshot(which=shutil.which, run=subprocess.run):
    executable = which("nvidia-smi")
    if not executable:
        return dict(available=False, error="nvidia-smi unavailable")
    command = [executable, "--id=0", "--query-gpu=" + ",".join(GPU_FIELDS),
               "--format=csv,noheader,nounits"]
    try:
        result = run(command, capture_output=True, text=True, timeout=5, check=False)
        if result.returncode != 0:
            return dict(available=False, command=command, exit_code=result.returncode,
                        error=result.stderr[:1000])
        rows = list(csv.reader(result.stdout.splitlines()))
        if len(rows) != 1 or len(rows[0]) != len(GPU_FIELDS):
            raise ValueError("Expected exactly GPU0 and all requested fields")
        fields = {key: value.strip() for key, value in zip(GPU_FIELDS, rows[0])}
        return dict(available=True, command=command,
                    executable_sha256=hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                    gpu0=fields, units=GPU_UNITS,
                    unavailable_fields=[key for key, value in fields.items()
                                        if value in ("N/A", "[N/A]", "[Not Supported]")],
                    value_format="Raw nvidia-smi CSV strings; N/A or unsupported values remain unavailable, never zero.")
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return dict(available=False, command=command, error=str(error))


def affinity_snapshot():
    try:
        if not hasattr(os, "sched_getaffinity"):
            return dict(available=False, error="CPU affinity API unavailable")
        return dict(available=True, allowed_cpu_count=len(os.sched_getaffinity(0)))
    except OSError as error:
        return dict(available=False, error=str(error))


def capture_boundary():
    started = time.monotonic_ns()
    affinity = affinity_snapshot()
    return dict(utc_unix_ns=time.time_ns(), monotonic_start_ns=started,
                cpu=cpu_snapshot(), gpu=gpu_snapshot(),
                affinity=affinity, allowed_cpu_count=affinity.get("allowed_cpu_count"),
                capture_elapsed_ms=(time.monotonic_ns() - started) / 1e6,
                scope="Read-only before-launch or after-shutdown snapshot, outside all timing; not continuous telemetry. CPU counters cover the entire engine process interval including startup/prewarm, not individual measured requests.")
