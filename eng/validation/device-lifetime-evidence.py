#!/usr/bin/env python3
"""Collect actual lifetime-test counts, memory samples, and runtime file hashes."""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trx", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--expected-tests", type=int, required=True)
    parser.add_argument("--dependency-revision", required=True)
    parser.add_argument("--dependency-status", type=Path, required=True)
    parser.add_argument("--backend", choices=["cuda", "mlx"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_tests <= 0:
        parser.error("An empty test run cannot pass validation.")
    tree = ET.parse(args.trx)
    namespace = {"x": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
    rows = [{"name": result.get("testName"), "outcome": result.get("outcome"), "duration": result.get("duration")}
        for result in tree.findall(".//x:UnitTestResult", namespace)]
    output = "\n".join(item.text or "" for item in tree.findall(".//x:StdOut", namespace))
    samples = [{"iteration": int(index), "free_bytes": int(free), "baseline_minus_free_bytes": int(delta)}
        for index, free, delta in re.findall(r"Context reload (\d+)/8: free=(\d+) bytes, baseline-minus-free=(-?\d+) bytes", output)]
    baseline = re.search(r"free-memory baseline after warmup: (\d+) bytes", output)
    files = [path for path in args.binary_dir.iterdir() if path.is_file() and
        (path.name.startswith("TensorSharp.") and path.suffix == ".dll" or path.name in
        ["InferenceWeb.Tests.dll", "libGgmlOps.so", "libGgmlOps.dylib", "libmlxc.dylib", "libmlx.dylib"])]
    files += list((args.binary_dir / "cuda_kernels").glob("*.ptx")) if args.backend == "cuda" else []
    passed = len(rows) == args.expected_tests and all(row["outcome"] == "Passed" for row in rows)
    if args.backend == "cuda":
        passed = passed and len(samples) == 8 and [row["iteration"] for row in samples] == list(range(1, 9))
    report = {"backend": args.backend, "passed": passed, "executed": len(rows),
        "passed_tests": sum(row["outcome"] == "Passed" for row in rows),
        "skipped_tests": sum(row["outcome"] in ["NotExecuted", "Skipped"] for row in rows),
        "not_passed_tests": sum(row["outcome"] != "Passed" for row in rows), "tests": rows,
        "ggml_revision": args.dependency_revision, "ggml_worktree_clean": not args.dependency_status.read_text().strip(),
        "driver_free_baseline_bytes": int(baseline.group(1)) if baseline else None, "reload_samples": samples,
        "sha256": {str(path.relative_to(args.binary_dir)): digest(path) for path in sorted(files)},
        "limitations": ["Small synthetic weights exercise ownership, eviction, pointer identity and reload behavior.",
            "No full-model memory/performance claim; direct H3 release covers tracked linear cache entries, not raw audio/3D encoder tensors."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": passed, "executed": len(rows), "reload_samples": samples}))
    return 0 if passed and report["ggml_worktree_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
