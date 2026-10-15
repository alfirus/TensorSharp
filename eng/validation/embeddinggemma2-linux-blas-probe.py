#!/usr/bin/env python3
"""Run the owned standalone probe and record actual binary/library provenance.

Compile separately into an ignored directory, then invoke only in a coordinated
quiet CPU window. This runner does not install libraries, change BLAS thread
settings/environment, build ggml, or modify production defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def hardware() -> dict:
    cpu_model = platform.processor()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        for line in cpu_info.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    quota = Path("/sys/fs/cgroup/cpu.max")
    cgroup_v1 = {}
    for directory in (Path("/sys/fs/cgroup/cpu"), Path("/sys/fs/cgroup/cpu,cpuacct")):
        for name in ("cpu.cfs_quota_us", "cpu.cfs_period_us"):
            path = directory / name
            if path.is_file():
                cgroup_v1[str(path)] = path.read_text().strip()
    return {"platform": platform.platform(), "machine": platform.machine(),
            "cpu_model": cpu_model, "cpu_count": os.cpu_count(),
            "allowed_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "cgroup_cpu_max": quota.read_text().strip() if quota.is_file() else None,
            "cgroup_v1_cpu_quota": cgroup_v1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    command = [str(binary), "--workers", str(args.workers)]
    if args.library:
        command += ["--library", str(args.library.resolve(strict=True))]
    if args.benchmark:
        command += ["--benchmark"]
    sources = (Path(__file__).resolve(), Path(__file__).with_suffix(".cpp").resolve(),
               Path(__file__).resolve().parents[2] / "TensorSharp.GGML.Native/embedding_linux_blas.h")
    before = {str(path): sha256(path) for path in (binary, *sources)}
    if args.library:
        before[str(args.library.resolve(strict=True))] = sha256(args.library.resolve(strict=True))
    # Existing environment is observed, never rewritten. Runtime metadata,
    # rather than the mere presence of OPENBLAS_NUM_THREADS, qualifies worker1.
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        report = {"completed": False, "reason": "probe-did-not-return-complete-json"}
    report.update({
        "process_exit_code": completed.returncode,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "binary": {"path": str(binary), "sha256": sha256(binary)},
        "hardware": hardware(),
        "observed_environment": {key: os.environ.get(key) for key in
            ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_CORETYPE")},
        "ggml_scope": "Standalone helper/probe does not link or execute ggml. Callback/model integration requires separate actual gates against unchanged upstream.",
        "source": {str(path): sha256(path) for path in sources},
        "artifacts_before": before,
        "artifacts_after": {path: sha256(Path(path)) for path in before},
    })
    provider = report.get("library")
    if provider and Path(provider).is_file():
        path = Path(provider).resolve(strict=True)
        report["loaded_library"] = {"path": str(path), "sha256": sha256(path),
                                    "reported_build_config": report.get("config")}
    success = completed.returncode == 0 and report.get("completed") is True
    report["artifacts_unchanged"] = report["artifacts_before"] == report["artifacts_after"]
    success &= report["artifacts_unchanged"]
    if success:
        counts = report.get("counts", {})
        success = counts.get("gemm_calls", 0) > 0 and counts.get("output_elements", 0) > 0
        success &= report.get("actual_blas_workers") == 1 and report.get("existing_worker_budget") == args.workers
        success &= report.get("cancellation_passed") is True and report.get("guard_lifetime_passed") is True
        success &= report.get("in_flight_close_observed") is True and counts.get("active_calls") == 0
    report["passing"] = bool(success)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passing": bool(success), "exit_code": completed.returncode,
                      "library": provider, "output": str(args.output)}))
    return 0 if success else completed.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
