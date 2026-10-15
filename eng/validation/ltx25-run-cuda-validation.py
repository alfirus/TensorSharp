#!/usr/bin/env python3
"""Build, pin CUDA deployment bytes, test, and run a serial LTX comparison.

Configuration is explicit JSON: module_source, modules (filename:sha256:target),
probe_output, test_filter, environment, cases_json. Generated receipts/logs live
under --output. Builds precede deployment; tests always use --no-build.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(config["environment"])
    skip = ["-p:TensorSharpSkipGgmlNative=true", "-p:TensorSharpSkipMlxNative=true"]
    tests = "InferenceWeb.Tests/InferenceWeb.Tests.csproj"
    probe = "eng/validation/Ltx25EndToEndProbe/Ltx25EndToEndProbe.csproj"
    steps = [
        ("build-tests", ["dotnet", "build", tests, "-c", "Release", "--no-incremental", *skip]),
        ("build-probe", ["dotnet", "build", probe, "-c", "Release", "--no-incremental", *skip]),
        ("publish-probe", ["dotnet", "publish", probe, "-c", "Release", "--no-build", *skip, "-o", config["probe_output"]]),
    ]
    deploy = [sys.executable, str(Path(__file__).with_name("ltx25-deploy-cuda-modules.py")),
              "--source", config["module_source"], "--destination", "InferenceWeb.Tests/bin/Release/net10.0/cuda_kernels",
              "--destination", str(Path(config["probe_output"]) / "cuda_kernels")]
    for module in config["modules"]:
        deploy += ["--module", module]
    steps += [
        ("deploy-modules", [*deploy, "--output", str(args.output / "deployed-modules.json")]),
        ("strict-tests", ["dotnet", "test", tests, "-c", "Release", "--no-build", "--filter", config["test_filter"],
                           "--logger", "console;verbosity=detailed", "--logger", "trx;LogFileName=tests.trx",
                           "--results-directory", str(args.output / "tests")]),
        ("verify-before-generation", [*deploy, "--verify-only", "--output", str(args.output / "generation-modules.json")]),
        ("serial-generation", [sys.executable, str(Path(__file__).with_name("ltx25-generation-suite.py")),
                               "--cases-json", config["cases_json"], "--output", str(args.output / "suite")]),
    ]
    report = {"completed": False, "passed": False, "config": config, "steps": []}
    receipt = args.output / "runner.json"
    for name, argv in steps:
        log = args.output / (name + ".log")
        with log.open("xb") as output:
            result = subprocess.run(argv, env=env, stdout=output, stderr=subprocess.STDOUT)
        report["steps"].append({"name": name, "argv": argv, "exit_code": result.returncode, "log": str(log)})
        receipt.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["steps"][-1]), flush=True)
        if result.returncode:
            return result.returncode
    report.update(completed=True, passed=True)
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
