#!/usr/bin/env python3
"""Run named LTX generation commands serially in one exclusive device lane.

Input is a JSON array of {name, argv, timeout_seconds?}. Each command may name
its own inner output report. Reports and subprocess logs go under --output;
failed attempts are retained and are never counted as passing scenarios.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.cases_json.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected a nonempty case array.")
    names = set()
    for case in cases:
        name = case["name"]
        if name in names or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
            raise ValueError("Case names must be unique lower-case filename components.")
        names.add(name)
        if not case.get("argv") or not all(isinstance(item, str) for item in case["argv"]):
            raise ValueError("Each case needs a nonempty argv array of strings.")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"completed": False, "passed": False, "started_unix": time.time(),
              "scope": "Serial device lane; process timings include setup/codecs. Use inner model timings for comparisons.",
              "executions": []}
    def save():
        (args.output / "suite.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    for case in cases:
        command = args.output / (case["name"] + "-command.json")
        command.write_text(json.dumps(case["argv"], ensure_ascii=False, indent=2) + "\n")
        output = args.output / (case["name"] + "-process.json")
        started = time.monotonic()
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("ltx25-generation-runner.py")),
                                 "--command-json", str(command), "--output", str(output),
                                 "--timeout", str(case.get("timeout_seconds", 1800))])
        execution = {"name": case["name"], "exit_code": result.returncode, "passed": result.returncode == 0,
                     "wall_seconds": time.monotonic() - started, "process_report": str(output)}
        report["executions"].append(execution)
        save()
        print(json.dumps(execution), flush=True)
        if result.returncode and not args.continue_on_failure:
            break
    report["completed"] = len(report["executions"]) == len(cases)
    report["passed"] = report["completed"] and all(item["passed"] for item in report["executions"])
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
