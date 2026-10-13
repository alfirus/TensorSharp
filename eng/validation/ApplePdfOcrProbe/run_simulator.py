#!/usr/bin/env python3
"""Build and run actual TensorSharp PDF OCR on an isolated iOS simulator.

Creates and deletes its own simulator; never opens or clears TensorAgent data.
Requires macOS, Xcode, an installed iOS runtime, and the .NET iOS workload.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import shutil
import statistics
import subprocess
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default="com.apple.CoreSimulator.SimRuntime.iOS-27-0")
    parser.add_argument("--device-type", default="com.apple.CoreSimulator.SimDeviceType.iPhone-17-Pro")
    parser.add_argument("--dotnet", default=str(pathlib.Path.home() / ".dotnet/dotnet"))
    parser.add_argument("--output", default="artifacts/knowledge/ios-pdf-ocr")
    parser.add_argument("--samples", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[3]
    output = pathlib.Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(command, log, timeout=120, check=True, env=None):
        commands.append(command)
        (output / "commands.json").write_text(json.dumps(commands, indent=2))
        result = subprocess.run(command, cwd=repo, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout)
        (output / log).write_text(result.stdout)
        if check and result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}); see {output / log}")
        return result.stdout.strip()

    project = repo / "eng/validation/ApplePdfOcrProbe/ApplePdfOcrProbe.csproj"
    if not args.no_build:
        env = dict(os.environ, DOTNET_ROOT=str(pathlib.Path(args.dotnet).resolve().parent))
        run([args.dotnet, "build", str(project), "-c", "Debug", "-p:TensorSharpAppleTargets=true",
             "-p:TensorSharpSkipGgmlNative=true", "-p:TensorSharpSkipMlxNative=true", "-m:1", "-nologo"],
            "build.log", timeout=300, env=env)
    app = project.parent / "bin/Debug/net10.0-ios/iossimulator-arm64/ApplePdfOcrProbe.app"
    if not app.is_dir():
        raise RuntimeError(f"Probe app missing: {app}")
    hashes = {name: hashlib.sha256((app / name).read_bytes()).hexdigest()
              for name in ["ApplePdfOcrProbe.dll", "TensorSharp.Models.dll", "TensorSharp.Knowledge.dll"]}
    fixture_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted((app / "Fixtures").glob("*.pdf"))}
    (output / "environment.json").write_text(json.dumps({"host": platform.platform(), "runtime": args.runtime,
        "deviceType": args.device_type, "samplesPerPage": args.samples, "appAssembliesSha256": hashes,
        "fixturesSha256": fixture_hashes}, indent=2))
    bundle = "ai.tensorsharp.validation.pdfocr"
    device = None
    try:
        device = run(["xcrun", "simctl", "create", "TensorSharp PDF OCR " + uuid.uuid4().hex[:8],
                      args.device_type, args.runtime], "create.log")
        print(f"Created isolated simulator for {args.runtime}", flush=True)
        run(["xcrun", "simctl", "boot", device], "boot.log")
        run(["xcrun", "simctl", "bootstatus", device, "-b"], "bootstatus.log", timeout=180)
        run(["xcrun", "simctl", "install", device, str(app)], "install.log")
        env = dict(os.environ, SIMCTL_CHILD_PDF_PROBE_SAMPLES=str(args.samples))
        launched = run(["xcrun", "simctl", "launch", "--terminate-running-process",
             "--stdout=" + str(output / "stdout.log"), "--stderr=" + str(output / "stderr.log"),
             device, bundle], "launch.log", env=env)
        match = re.search(r":\s*(\d+)\s*$", launched)
        process_id = int(match.group(1)) if match else None
        container = pathlib.Path(run(["xcrun", "simctl", "get_app_container", device, bundle, "data"], "container.log"))
        report = container / "Documents/probe-result.json"
        deadline = time.monotonic() + args.timeout
        peak_rss_kib = 0
        print("Probe running: actual .NET iOS -> CoreGraphics/Vision -> knowledge store", flush=True)
        while time.monotonic() < deadline:
            fatal = container / "Documents/probe-fatal.txt"
            if fatal.is_file():
                shutil.copyfile(fatal, output / "fatal.txt")
                raise RuntimeError(f"Probe failed before report completion; see {output / 'fatal.txt'}")
            if report.is_file():
                result = json.loads(report.read_text())
                if result.get("status") == "complete":
                    shutil.copyfile(report, output / "result.json")
                    samples = sorted(sample for page in result.get("benchmarks", []) for sample in page["samplesMs"])
                    if samples:
                        (output / "benchmark-summary.json").write_text(json.dumps({"sampleCount": len(samples),
                            "pageCount": len(result["benchmarks"]), "p50Ms": statistics.median(samples),
                            "p95Ms": samples[math.ceil(len(samples) * .95) - 1], "p95Method": "nearest-rank",
                            "minMs": samples[0], "maxMs": samples[-1],
                            "notes": "Aggregate across six distinct fixture pages; warm host simulator timings, no model inference."}, indent=2))
                    (output / "runner-metrics.json").write_text(json.dumps({"sampledPeakAppRssKiB": peak_rss_kib,
                        "samplingIntervalSeconds": 1, "notes": "Host simulator app RSS; excludes separate OS services. Not physical device memory."}, indent=2))
                    print(json.dumps({"passed": result["passed"], "failed": result["failed"],
                                      "cases": len(result["cases"]), "elapsedMs": result["elapsedMs"],
                                      "report": str(output / "result.json")}), flush=True)
                    return 0 if result["passed"] else 1
            if process_id:
                try:
                    os.kill(process_id, 0)
                except ProcessLookupError:
                    reports = list((pathlib.Path.home() / "Library/Logs/DiagnosticReports").glob("ApplePdfOcrProbe-*.ips"))
                    if reports:
                        shutil.copyfile(max(reports, key=lambda p: p.stat().st_mtime), output / "crash.ips")
                    raise RuntimeError("Probe exited before producing a report; inspect stdout/stderr and crash logs.")
                rss = subprocess.run(["ps", "-o", "rss=", "-p", str(process_id)], text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5).stdout.strip()
                if rss.isdigit():
                    peak_rss_kib = max(peak_rss_kib, int(rss))
            time.sleep(1)
        raise TimeoutError(f"Probe did not finish within {args.timeout}s; inspect stdout/stderr logs.")
    finally:
        if device:
            run(["xcrun", "simctl", "terminate", device, bundle], "terminate.log", check=False)
            run(["xcrun", "simctl", "shutdown", device], "shutdown.log", check=False)
            run(["xcrun", "simctl", "delete", device], "delete.log", check=False)


if __name__ == "__main__":
    raise SystemExit(main())
