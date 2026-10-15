#!/usr/bin/env python3
"""Run actual owned-CPU attention/optional VNNI quality against independent NumPy.

Run only in a coordinated quiet window. Builds, HTTP timings and long sweeps
are separate stages. Unsupported/fallback math must not count as native passes.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def native_evidence(log, library, require_q16=False, require_fused=False):
    records = [json.loads(line[len("EMBEDDING_CPU_MATH "):]) for line in log.read_text().splitlines()
               if line.startswith("EMBEDDING_CPU_MATH ")]
    if len(records) != 1:
        raise ValueError("Require exactly one actual CPU math disposal record")
    evidence = records[0]
    if evidence["backend"] != "CPU" or evidence["graph"] != "managed" or not evidence["native_math"] or \
       evidence["successful_attention_tiles"] <= 0 or evidence["successful_attention_rows"] <= 0 or evidence["managed_attention_tiles"] != 0 or \
       evidence["library_sha256"] != sha(library):
        raise ValueError("Fallback/unavailable math or a different library cannot establish a native model pass")
    if require_q16 and (evidence.get("projection_math") != "owned-native-q8-q16-vnni" or
                        evidence.get("projection_library_sha256") != sha(library) or
                        evidence.get("projection_fallback_reason") is not None or
                        any(evidence.get(name, 0) <= 0 for name in ("successful_projection_calls", "successful_projection_pack_rows",
                                                                 "successful_projection_groups", "successful_projection_rows")) or
                        evidence.get("managed_q16_projection_calls") != 0 or evidence.get("q16_float32_fallback_rows") != 0):
        raise ValueError("Missing native projection work, library mismatch or Q16 fallback cannot establish a combined native pass")
    if require_fused and (not evidence.get("fused_gate_gelu_requested") or evidence.get("successful_fused_gate_gelu_calls", 0) <= 0 or
                          evidence.get("successful_fused_gate_gelu_rows", 0) <= 0):
        raise ValueError("Requested FFN fusion did not execute successful calls/rows")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--eight-cases", type=Path, required=True)
    parser.add_argument("--eight-reference", type=Path, required=True)
    parser.add_argument("--long-cases", type=Path, required=True)
    parser.add_argument("--long-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--native-q16", action="store_true", help="Also require actual VNNI projections with zero eligible-projection fallback")
    parser.add_argument("--projection-bridge", type=Path, help="Fresh linked production matrix/ABI arithmetic probe, required with --native-q16")
    parser.add_argument("--q16-probe", type=Path, help="Fresh independent scalar/double Q16 arithmetic probe, required with --native-q16")
    parser.add_argument("--fused-gelu", action="store_true", help="Require actual opt-in joined FFN GELU/up fusion")
    parser.add_argument("--gated-gelu-probe", type=Path, help="Fresh linked fusion numerical probe, required with --fused-gelu")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    if args.native_q16 and (args.projection_bridge is None or args.q16_probe is None):
        parser.error("Native Q16 requires both fresh arithmetic probes")
    if args.fused_gelu and not args.native_q16:
        parser.error("This qualified fusion gate requires the composite native Q16 route")
    if args.fused_gelu and args.gated_gelu_probe is None:
        parser.error("FFN fusion requires its fresh linked numerical probe")
    args.output.mkdir(parents=True, exist_ok=True)
    checkout = args.root / "ExternalProjects/ggml"
    dependency = dict(revision=git(checkout, "rev-parse", "HEAD"), tracked_changes=git(checkout, "status", "--porcelain", "--untracked-files=no"))
    if dependency["tracked_changes"]:
        raise ValueError("Require unchanged upstream ggml")
    flags = dict(TENSORSHARP_EMBEDDING_NATIVE_ATTENTION="1", TENSORSHARP_CPU_NATIVE_LIBRARY=str(args.library.resolve()),
                 TENSORSHARP_EMBEDDING_Q16_AVX512="1" if args.native_q16 else "0",
                 TENSORSHARP_EMBEDDING_NATIVE_Q16="1" if args.native_q16 else "0", TENSORSHARP_EMBEDDING_Q8_AVX="0",
                 TENSORSHARP_EMBEDDING_FUSED_GATE_GELU="1" if args.fused_gelu else "0",
                 TENSORSHARP_EMBEDDING_CPU_PROFILE="0", DOTNET_PROCESSOR_COUNT=str(args.threads),
                 TS_CPU_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads))
    env = dict(os.environ, **flags)
    env.pop("TS_CPU_DISABLE_AVX512", None)
    sources = ["TensorSharp.Models/Embeddings/ManagedEmbeddingGemma2Encoder.cs", "TensorSharp.Models/Embeddings/OwnedEmbeddingCpuAttention.cs",
               "TensorSharp.Models/Embeddings/ManagedEmbeddingEncoder.cs", "TensorSharp.Models/Embeddings/ManagedEmbeddingQ8Q16Avx512Matrix.cs",
               "TensorSharp.Models/Embeddings/OwnedEmbeddingCpuProjection.cs", "TensorSharp.Cpu.Native/embedding_q16_vnni.cpp",
               "TensorSharp.Models/Embeddings/ManagedEmbeddingGatedGelu.cs",
               "eng/validation/EmbeddingGemma2GatedGeluProbe/Program.cs",
               "eng/validation/EmbeddingGemma2GatedGeluProbe/EmbeddingGemma2GatedGeluProbe.csproj",
               "TensorSharp.Cpu.Native/embedding_attention_avx512.cpp", "TensorSharp.Cpu.Native/Packaging.targets",
               "eng/validation/EmbeddingGemma2Probe/Program.cs", "eng/validation/embeddinggemma2-owned-attention-quality.py"]
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), flags=flags,
                  source_artifacts=[dict(path=name, sha256=sha(args.root / name)) for name in sources],
                  binary_artifacts=[dict(path=str(path.resolve()), sha256=sha(path)) for path in args.probe.parent.iterdir()
                                    if path.is_file() and path.suffix in (".dll", ".so", ".dylib")],
                  library=dict(path=str(args.library.resolve()), sha256=sha(args.library)), dependency_before=dependency,
                  quality_thresholds=dict(min_cosine=.99999, max_absolute_error=.0005), arithmetic=[], runs=[], completed=False,
                  limitations="Actual managed CPU graph with stateless owned native attention and optional Q8/Q16 VNNI projection and exact-order joined FFN fusion; softmax/masking and sensitive PLE/InputGate/output F32 projections remain managed. Exact recorded text inputs only; no HTTP/performance or retrieval leaderboard claim.")
    try:
        if args.native_q16:
            arithmetic = [("projection-bridge", args.projection_bridge, [str(args.library)], [], 14),
                          ("q16-arithmetic", args.q16_probe, [], [], 10)]
            if args.fused_gelu:
                arithmetic.append(("gated-gelu", args.gated_gelu_probe, [], ["--numerical-only"], 12))
            for name, probe, extra, suffix, expected_count in arithmetic:
                candidate = args.output / (name + ".json")
                log = args.output / (name + ".log")
                argv = ["dotnet", str(probe), *extra, str(candidate), *suffix]
                run = dict(name=name, command=argv, probe_sha256=sha(probe))
                report["arithmetic"].append(run)
                with log.open("w") as stream:
                    run["exit_code"] = subprocess.run(argv, env=env, cwd=args.root, stdout=stream, stderr=subprocess.STDOUT).returncode
                if run["exit_code"]:
                    return 1
                result = json.loads(candidate.read_text())
                run.update(executed=len(result["cases"]), passed=sum(case.get("passed", case.get("scalar_bit_exact", case.get("legacy_bit_exact", False))) for case in result["cases"]),
                           skipped=0, report_sha256=sha(candidate), log_sha256=sha(log))
                if not result["completed"] or run["executed"] != expected_count or run["passed"] != expected_count:
                    return 1
                if name == "gated-gelu" and (result["timing_executed"] or not result["worker_lifetime_passed"] or
                                             not result["in_flight_cancellation_executed_and_pool_reused"] or
                                             any(not case["independent_double_bound_passed"] or not case["up_unchanged"] or
                                                 not case["suffix_canaries_passed"] or case["invalid_guards_rejected"] != 7 or
                                                 not case["pre_cancel_rejected_and_pool_reusable"] for case in result["cases"])):
                    return 1
                if name == "projection-bridge" and (not result["native_math_executed"] or result["native_calls"] != expected_count or
                                                    result["library_sha256"] != sha(args.library)):
                    raise ValueError("Production native bridge arithmetic was unavailable or used a different library")
        for name, cases, reference, expected_count in (("eight", args.eight_cases, args.eight_reference, 8), ("8192", args.long_cases, args.long_reference, 1)):
            candidate = args.output / (name + ".json")
            log = args.output / (name + ".log")
            comparison = args.output / (name + "-comparison.json")
            argv = ["dotnet", str(args.probe), "--model", str(args.model), "--cases", str(cases), "--output", str(candidate),
                    "--backend", "CPU", "--threads", str(args.threads), "--warmup", "1", "--repeats", "1"]
            run = dict(name=name, command=argv)
            report["runs"].append(run)
            with log.open("w") as stream:
                run["exit_code"] = subprocess.run(argv, env=env, cwd=args.root, stdout=stream, stderr=subprocess.STDOUT).returncode
            if run["exit_code"]:
                return 1
            run["actual_math"] = native_evidence(log, args.library, args.native_q16, args.fused_gelu)
            compare_argv = [sys.executable, str(args.root / "eng/validation/embeddinggemma2-compare.py"), "--reference", str(reference),
                            "--candidate", str(candidate), "--case-manifest", str(cases), "--output", str(comparison),
                            "--min-cosine", ".99999", "--max-absolute-error", ".0005"]
            with (args.output / (name + "-comparison.log")).open("w") as stream:
                run["comparison_exit_code"] = subprocess.run(compare_argv, cwd=args.root, stdout=stream, stderr=subprocess.STDOUT).returncode
            result = json.loads(comparison.read_text())
            run.update(executed=len(result["cases"]), passed=sum(case["passed"] for case in result["cases"]), skipped=0,
                       min_cosine=result["min_cosine"], max_absolute_error=result["max_absolute_error"],
                       candidate_sha256=sha(candidate), comparison_sha256=sha(comparison), log_sha256=sha(log))
            if run["comparison_exit_code"] or run["executed"] != expected_count:
                return 1
        report["completed"] = True
        return 0
    finally:
        report["dependency_after"] = dict(revision=git(checkout, "rev-parse", "HEAD"), tracked_changes=git(checkout, "status", "--porcelain", "--untracked-files=no"))
        if report["dependency_after"] != dependency:
            report["completed"] = False
        (args.output / "quality-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(completed=report["completed"], runs=report["runs"]), indent=2))
        if report["dependency_after"] != dependency:
            raise RuntimeError("Dependency changed during validation")


if __name__ == "__main__":
    raise SystemExit(main())
