#!/usr/bin/env python3
"""Build/run the eng-only F32 materialized attention screen with immutable guards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


REVISION = "3612c0af93f2cfc8f048295300a61187723b12ae"
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "embeddinggemma2-vision-cuda-materialized-probe.cu"
README = HERE / "embeddinggemma2-vision-cuda-materialized-probe.md"


def digest(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def read_bound(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Bound JSON changed: " + str(path))
    return json.loads(data)


def snapshot(paths):
    result = {}
    for path in sorted({Path(p).absolute() for p in paths}, key=str):
        # Retain both spelling and target to detect changed aliases.
        result[str(path)] = dict(sha256=digest(path), resolved=str(path.resolve(strict=True)))
    return result


def require_same(before):
    current = snapshot(before)
    if current != before:
        raise ValueError("Source/toolchain/runtime guard changed")


def extend_snapshot(before, paths):
    require_same(before)
    after = dict(before)
    missing = [Path(p).absolute() for p in paths if str(Path(p).absolute()) not in before]
    after.update(snapshot(missing))
    require_same(after)
    return after


def command(argv, cwd, log, allow_failure=False):
    with Path(log).open("xb") as stream:
        run = subprocess.run([str(x) for x in argv], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if run.returncode and not allow_failure:
        raise RuntimeError(f"Command failed ({run.returncode}); preserved {log}")
    return run.returncode


def dependencies(path, cwd):
    text = Path(path).read_text().replace("\\\n", "")
    target, separator, body = text.partition(":")
    if not separator or target.strip() != "probe":
        raise ValueError("Unexpected preprocessor dependency target")
    values = shlex.split(body)
    if not values:
        raise ValueError("No preprocessor dependencies")
    return [Path(v) if Path(v).is_absolute() else Path(cwd) / v for v in values]


def ldd_files(text):
    out = []
    for line in text.splitlines():
        if "not found" in line:
            raise ValueError("Unresolved runtime library: " + line.strip())
        match = re.search(r"(?:=>\s*)?(/\S+)\s+\(", line)
        if match:
            out.append(Path(match.group(1)))
    return out


def instruction_review(path, expected, artifacts):
    review = read_bound(path, expected)
    if review.get("reviewed") is not True or review.get("materialized_f32_layout_and_pedantic_calls") is not True or not review.get("reviewer"):
        raise ValueError("Explicit independent F32 attention layout/compiled-kernel/API review is required")
    for key in ("source", "executable", "ptx", "sass"):
        if review.get(key + "_sha256") != digest(artifacts[key]):
            raise ValueError("Instruction review binding differs: " + key)
    return review


def copied_source_ignore_policy(path, expected, repository, output):
    if (path is None) != (expected is None):
        raise ValueError("Copied-source ignore policy and its expected hash must be supplied together")
    if path is None:
        return None
    if (repository / ".git").exists() or (repository / ".git").is_symlink():
        raise ValueError("Use the normal Git route for a source root with Git metadata")
    proof = read_bound(path, expected)
    if type(proof.get("schema")) is not int or proof["schema"] != 1 or proof.get("checked_in_git_repository") is not True:
        raise ValueError("Require schema1 actual local Git ignore proof")
    prefixes = proof.get("ignored_prefixes")
    if not isinstance(prefixes, dict) or set(prefixes) != {"docs/validation", "artifacts"} or any(value is not True for value in prefixes.values()):
        raise ValueError("Copied-source policy must prove exactly docs/validation and artifacts")
    gitignore = repository / ".gitignore"
    if digest(gitignore) != proof.get("gitignore_sha256"):
        raise ValueError("Copied root .gitignore differs from locally proved bytes")
    # No descendant ignore rules may reinterpret this explicitly proved policy.
    for base, parent in ((repository, output.parent), (repository.resolve(), output.parent.resolve())):
        current = base
        for part in parent.relative_to(base).parts:
            current /= part
            nested = current / ".gitignore"
            if nested.exists() or nested.is_symlink():
                raise ValueError("Nested .gitignore overrides copied-source policy: " + str(nested))
    return dict(path=str(Path(path).absolute()), sha256=expected, root_gitignore_path=str(gitignore),
                root_gitignore_sha256=proof["gitignore_sha256"], proof=proof)


def fresh_ignored_output(path, repository, *, copied_policy=None, expected_policy_sha256=None):
    path = Path(path).absolute()
    if path.exists():
        raise ValueError("Choose a fresh output directory")
    relative = path.relative_to(repository)
    if ".." in relative.parts:
        raise ValueError("Generated output must be traversal-free")
    if not (relative.parts[:2] == ("docs", "validation") or relative.parts[:1] == ("artifacts",)):
        raise ValueError("Generated evidence must be under docs/validation or artifacts")
    if not path.parent.resolve().is_relative_to(repository.resolve()):
        raise ValueError("Output parent alias escapes the repository")
    physical = (path.parent.resolve() / path.name).relative_to(repository.resolve())
    if not (physical.parts[:2] == ("docs", "validation") or physical.parts[:1] == ("artifacts",)):
        raise ValueError("Output alias escapes the allowed evidence prefixes")
    policy = copied_source_ignore_policy(copied_policy, expected_policy_sha256, repository, path)
    if policy is None:
        checked = subprocess.run(["git", "check-ignore", "--quiet", "--", str(relative / "probe")], cwd=repository, check=False)
        if checked.returncode:
            raise ValueError("Generated output must be ignored by this repository")
    return path


BASELINE_SHA256 = "1a610f3f4b264c8fc7e53b1c01f0eba04899f8b7e5457a19f0d4b89d16454b71"
BASELINE_BLOCK_SHA256 = "f1f81e1b2c5e0498fabf71867b5186c07759dadba68612271a5fe6debe4fd76c"
FIXTURES = {
    "single": (1,1,1,1,True), "batched-padded": (37,37,3,2,True),
    "strided-key-tail": (65,129,2,1,True), "query-key-tail": (129,521,4,1,True),
    "long-key-tail": (3,8193,2,1,True), "large-values": (65,257,2,1,True),
    "exact-80000": (129,17,2,2,True), "source-bits": (129,2,2,2,True),
    "actual-1089": (1089,1089,12,1,False), "actual-2304": (2304,2304,12,1,False),
    "actual-2340": (2340,2340,12,1,False), "canonical-key-query-tails": (131,133,12,1,False),
    "canonical-padded-batches": (129,257,12,2,False), "canonical-interleaved-batches": (65,129,12,2,False),
    "unnormalized-range-refusal": (2,17,1,1,False),
}


def baseline_copy(path, expected, source):
    data = Path(path).read_bytes()
    if expected != BASELINE_SHA256 or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Require exact unchanged owned online baseline source")
    text = data.decode("utf-8")
    block = text[text.index("struct view {"):text.index("// A block owns one query/head")]
    owned = Path(source).read_text()
    begin = "// BEGIN EXACT OWNED BASELINE\n"
    copied = owned[owned.index(begin)+len(begin):owned.index("// END EXACT OWNED BASELINE")]
    if copied != block or hashlib.sha256(block.encode()).hexdigest() != BASELINE_BLOCK_SHA256:
        raise ValueError("Copied baseline block differs byte-for-byte")
    return dict(path=str(Path(path).absolute()), source_sha256=expected, block_sha256=BASELINE_BLOCK_SHA256,
                copied_block_byte_identical=True)


def required_coverage(result, mode):
    if (result.get("mode") != mode or result.get("completed") is not True or result.get("passed") is not True or
            result.get("eligible_candidate_quality_passed") is not True or result.get("candidate_shapes_executed") != 14 or
            result.get("unavailable_candidate_shapes") != 1 or result.get("baseline_cases_failed") != 0 or
            result.get("failed_candidate_cases_or_timing_phases") != 0 or
            result.get("all_original_fixtures_candidate_passed") is not True or result.get("cleanup_succeeded") is not True or
            result.get("private_cublas_pedantic_math_mode_verified") is not True or
            result.get("host_controls_passed") != 3711 or result.get("host_refusal_controls_passed") != 195):
        return False
    cases = result.get("quality_cases", [])
    if len(cases) != len(FIXTURES) or len({row.get("case") for row in cases}) != len(FIXTURES):
        return False
    for row in cases:
        if row.get("case") not in FIXTURES:
            return False
        q,k,h,b,original = FIXTURES[row["case"]]
        if tuple(row.get(key) for key in ("queries","keys","heads","batches","original_fixture")) != (q,k,h,b,original):
            return False
        if row["case"] == "unnormalized-range-refusal":
            if (row.get("eligible") is not False or row.get("candidate_executed") is not False or
                    row.get("range_control_executed") is not True or row.get("refusal_no_publication_and_private_output_unchanged") is not True):
                return False
        elif (row.get("eligible") is not True or row.get("candidate_executed") is not True or
              row.get("baseline_quality", {}).get("passed") is not True or
              row.get("combined_initial_warm_final_quality", {}).get("passed") is not True):
            return False
    timings = result.get("timings", [])
    if mode == "quality":
        return timings == []
    return (len(timings) == 12 and {(row.get("case"),row.get("phase")) for row in timings} ==
            {(name,phase) for name in ("actual-1089","actual-2304","actual-2340") for phase in range(4)} and
            all(row.get("quality", {}).get("passed") is True for row in timings))


def complete_vector_artifacts(output, mode):
    result = {}
    prefixes = [mode + ".json"]
    if mode == "benchmark":
        prefixes.append("benchmark.json-preflight")
    for prefix in prefixes:
        for name,(queries,_,heads,batches,_) in FIXTURES.items():
            if name == "unnormalized-range-refusal":
                continue
            for route in ("baseline","candidate"):
                path = Path(output) / (prefix + "-" + name + "-" + route + ".f32")
                if not path.is_file() or path.stat().st_size != queries * heads * batches * 64 * 4:
                    raise ValueError("Incomplete saved output vector: " + str(path))
                result[str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("quality", "benchmark"), default="quality")
    p.add_argument("--nvcc", type=Path)
    p.add_argument("--host-compiler", type=Path)
    p.add_argument("--cuda-root", type=Path)
    p.add_argument("--native", type=Path)
    p.add_argument("--expected-native-sha256")
    p.add_argument("--ggml", type=Path)
    p.add_argument("--baseline-source", type=Path)
    p.add_argument("--expected-baseline-source-sha256")
    p.add_argument("--qualified-build", type=Path)
    p.add_argument("--instruction-review", type=Path)
    p.add_argument("--expected-instruction-review-sha256")
    p.add_argument("--additional-guard", type=Path, action="append", default=[])
    p.add_argument("--copied-source-ignore-policy", type=Path)
    p.add_argument("--expected-copied-source-ignore-policy-sha256")
    a = p.parse_args(argv)
    if (a.copied_source_ignore_policy is None) != (a.expected_copied_source_ignore_policy_sha256 is None):
        p.error("--copied-source-ignore-policy and its expected SHA256 must be supplied together")
    repository = HERE.parents[1]
    out = fresh_ignored_output(a.output, repository, copied_policy=a.copied_source_ignore_policy,
                               expected_policy_sha256=a.expected_copied_source_ignore_policy_sha256)
    out.mkdir(parents=True)
    policy = copied_source_ignore_policy(a.copied_source_ignore_policy, a.expected_copied_source_ignore_policy_sha256, repository, out)
    status = dict(completed=False, passed=False, production_edited=False, model_qualified=False,
                  performance_qualified=False, mode=a.mode, source_sha256=digest(SOURCE),
                  copied_source_ignore_policy=policy)
    before = None
    generated = None
    try:
        if sys.platform != "linux":
            raise RuntimeError("CUDA build/device screen requires Linux; Mac host controls are separate/unqualified")
        selectors = ("NVIDIA_TF32_OVERRIDE", "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "CUDA_LAUNCH_BLOCKING",
                     "CUDA_DEVICE_MAX_CONNECTIONS", "CUBLAS_WORKSPACE_CONFIG")
        status["environment"] = {key: os.environ.get(key) for key in selectors}
        for key in ("LD_PRELOAD", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "COMPILER_PATH", "GCC_EXEC_PREFIX", "LIBRARY_PATH",
                    "TSG_CUPTI_ACTIVITY_OUTPUT", "TSG_CUPTI_TRANSFER_ACTIVITY_OUTPUT"):
            if os.environ.get(key):
                raise ValueError("Unbound instrumentation/compiler override must be absent: " + key)
        if a.mode == "quality":
            if a.qualified_build or a.instruction_review or a.expected_instruction_review_sha256:
                raise ValueError("Quality mode creates a fresh build; no prior review accepted")
            if any(x is None for x in (a.nvcc, a.host_compiler, a.cuda_root, a.native, a.expected_native_sha256, a.ggml, a.baseline_source, a.expected_baseline_source_sha256)):
                raise ValueError("Quality requires CUDA/compiler/native/upstream and exact owned baseline source/hash")
            baseline_source = a.baseline_source.absolute()
            status["baseline_copy"] = baseline_copy(baseline_source, a.expected_baseline_source_sha256, SOURCE)
            native = a.native.absolute()
            if digest(native) != a.expected_native_sha256:
                raise ValueError("Frozen native hash differs")
            revision = subprocess.check_output(["git", "-C", str(a.ggml), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(a.ggml), "status", "--porcelain", "--untracked-files=no"], text=True)
            if revision != REVISION or dirty:
                raise ValueError("Require exact unchanged upstream ggml")
            tracked = subprocess.check_output(["git", "-C", str(a.ggml), "ls-files", "-z"])
            paths = [a.ggml / os.fsdecode(v) for v in tracked.split(b"\0") if v]
            paths += [SOURCE, README, Path(__file__).absolute(), baseline_source, native, a.nvcc, a.host_compiler, *a.additional_guard]
            if policy is not None:
                paths += [Path(policy["path"]), Path(policy["root_gitignore_path"])]
            cuda = a.cuda_root.absolute()
            if not a.nvcc.resolve().is_relative_to(cuda.resolve()):
                raise ValueError("nvcc must belong to explicit CUDA root")
            tools = [cuda / "bin" / n for n in ("ptxas", "fatbinary", "nvlink", "cudafe++", "cuobjdump")]
            tools += [cuda / "nvvm/bin/cicc", cuda / "nvvm/libdevice/libdevice.10.bc"]
            ldd_tool = Path(shutil.which("ldd") or "/unavailable/ldd").absolute()
            paths += tools + [ldd_tool]
            # Explicit CUDA link inputs plus all matching actual shared-library
            # spellings/targets. No claim about arbitrary implicit dlopen files.
            lib = cuda / "lib64"
            for pattern in ("libcublas.so*", "libcublasLt.so*", "libcudart.so*", "libcudadevrt.a"):
                files = list(lib.glob(pattern))
                if not files:
                    raise ValueError("Missing explicit CUDA library family " + pattern)
                paths += files
            for name in ("cc1plus", "collect2", "as", "ld"):
                value = subprocess.check_output([str(a.host_compiler), "-print-prog-name=" + name], text=True).strip()
                if not Path(value).is_absolute():
                    value = subprocess.check_output(["which", value], text=True).strip()
                paths.append(Path(value))
            common = [a.nvcc, "--compiler-bindir", a.host_compiler, "-std=c++17", "-O3", "-gencode=arch=compute_86,code=[sm_86,compute_86]",
                      "--ftz=false", "--fmad=false", "--cudart=shared", "-Xcompiler=-Wall,-Wextra,-Werror", "-I", a.ggml / "include"]
            # Initial guard spans dependency discovery, then preprocessor closure
            # is added before compiling the executable.
            initial = snapshot(paths)
            if (initial[str(SOURCE.absolute())]["sha256"] != status["source_sha256"] or
                    initial[str(native)]["sha256"] != a.expected_native_sha256 or
                    initial[str(baseline_source)]["sha256"] != a.expected_baseline_source_sha256):
                raise ValueError("Validated source or frozen native changed before initial guard")
            if policy is not None and (initial[policy["path"]]["sha256"] != policy["sha256"] or
                                       initial[policy["root_gitignore_path"]]["sha256"] != policy["root_gitignore_sha256"]):
                raise ValueError("Copied-source ignore policy changed before initial guard")
            before = initial
            (out / "initial-before.json").write_text(json.dumps(initial, indent=2) + "\n")
            command(common + ["-M", "-MT", "probe", "-MF", out / "preflight.d", SOURCE], out, out / "preflight.log")
            require_same(initial)
            paths += dependencies(out / "preflight.d", out)
            before = extend_snapshot(initial, paths)
            (out / "before.json").write_text(json.dumps(before, indent=2) + "\n")
            keep = out / "keep"
            keep.mkdir()
            compile_argv = common + ["-Xptxas=-v", "--keep", "--keep-dir", keep, SOURCE,
                                     "-L", lib, "-lcublas", "-Xlinker", "-rpath", "-Xlinker", lib, "-o", out / "probe"]
            status["compile_command"] = [str(v) for v in compile_argv]
            copied_source_ignore_policy(a.copied_source_ignore_policy, a.expected_copied_source_ignore_policy_sha256, repository, out)
            command(compile_argv, out, out / "compile.log")
            require_same(before)
            # The new executable is a read input to every following tool. Bind
            # it before disassembly/ldd, retaining all previously proved inputs.
            before = extend_snapshot(before, [out / "probe", out / "compile.log", out / "preflight.d",
                                              *[path for path in keep.rglob("*") if path.is_file()]])
            require_same(before)
            command([cuda / "bin/cuobjdump", "--dump-ptx", out / "probe"], out, out / "ptx.txt")
            require_same(before)
            before = extend_snapshot(before, [out / "ptx.txt"])
            require_same(before)
            command([cuda / "bin/cuobjdump", "--dump-sass", out / "probe"], out, out / "sass.txt")
            require_same(before)
            before = extend_snapshot(before, [out / "sass.txt"])
            require_same(before)
            linked = subprocess.check_output([str(ldd_tool), str(out / "probe")], text=True)
            require_same(before)
            (out / "ldd.txt").write_text(linked)
            paths += ldd_files(linked)
            paths += [out / "probe", out / "ptx.txt", out / "sass.txt", out / "compile.log", out / "preflight.d", out / "ldd.txt"]
            before = extend_snapshot(before, paths)
            (out / "execution-before.json").write_text(json.dumps(before, indent=2) + "\n")
            artifacts = dict(source=SOURCE, executable=out / "probe", ptx=out / "ptx.txt", sass=out / "sass.txt")
            status["artifacts"] = {k: str(v) for k, v in artifacts.items()}
            status["artifact_sha256"] = {k: digest(v) for k, v in artifacts.items()}
            status["ggml_revision"] = revision
            ptx = (out / "ptx.txt").read_text()
            status["instruction_presence_only"] = {name: name in ptx for name in ("vision_kernel", "exponential_kernel", "divide_kernel")}
            status["instruction_presence_only"]["precision_qualified"] = False

        else:
            if not a.qualified_build or not a.instruction_review or not a.expected_instruction_review_sha256:
                raise ValueError("Benchmark requires qualified-build and hash-bound independent instruction review")
            if any(x is not None for x in (a.nvcc, a.host_compiler, a.cuda_root, a.native, a.expected_native_sha256, a.ggml, a.baseline_source, a.expected_baseline_source_sha256)) or a.additional_guard:
                raise ValueError("Benchmark reuses exact reviewed quality build; new compile inputs are prohibited")
            prior_bytes = (a.qualified_build / "status.json").read_bytes()
            prior = json.loads(prior_bytes)
            if prior.get("completed") is not True or prior.get("passed") is not True or prior.get("mode") != "quality":
                raise ValueError("Prior quality build was not qualified")
            prior_policy = prior.get("copied_source_ignore_policy")
            if bool(prior_policy) != bool(policy) or (policy is not None and
                    (prior_policy.get("sha256") != policy["sha256"] or prior_policy.get("root_gitignore_sha256") != policy["root_gitignore_sha256"])):
                raise ValueError("Benchmark must retain the same reviewed ignore policy route and bytes")
            before_bytes = (a.qualified_build / "execution-before.json").read_bytes()
            before = json.loads(before_bytes)
            if hashlib.sha256(before_bytes).hexdigest() != prior.get("execution_before_sha256"):
                raise ValueError("Prior input manifest changed")
            require_same(before)
            artifacts = {k: Path(v) for k, v in prior["artifacts"].items()}
            if artifacts["source"].resolve() != SOURCE.resolve() or prior["source_sha256"] != digest(SOURCE):
                raise ValueError("Current source differs from reviewed quality source")
            status["baseline_copy"] = prior["baseline_copy"]
            status["instruction_review"] = instruction_review(a.instruction_review, a.expected_instruction_review_sha256, artifacts)
            quality = read_bound(a.qualified_build / "quality.json", prior.get("quality_report_sha256"))
            if not required_coverage(quality, "quality"):
                raise ValueError("Prior hash-bound quality report lacks required coverage")
            paths = list(before) + [Path(path) for path in prior.get("produced_vectors", {})] + [a.qualified_build / "status.json", a.qualified_build / "execution-before.json",
                                    a.instruction_review, a.qualified_build / "quality.json"]
            if policy is not None:
                paths += [Path(policy["path"]), Path(policy["root_gitignore_path"])]
            before = extend_snapshot(before, paths)
            bound_metadata = {a.qualified_build / "status.json": hashlib.sha256(prior_bytes).hexdigest(),
                              a.qualified_build / "execution-before.json": hashlib.sha256(before_bytes).hexdigest(),
                              a.qualified_build / "quality.json": prior["quality_report_sha256"],
                              a.instruction_review: a.expected_instruction_review_sha256}
            if any(before[str(path.absolute())]["sha256"] != sha for path, sha in bound_metadata.items()):
                raise ValueError("Parsed proof metadata changed before benchmark input guard")
            if any(before[path]["sha256"] != value["sha256"] for path,value in prior.get("produced_vectors", {}).items()):
                raise ValueError("Prior complete output vector changed")
            if complete_vector_artifacts(a.qualified_build, "quality") != prior.get("produced_vectors"):
                raise ValueError("Prior saved complete vector metadata differs")
            status["artifacts"] = prior["artifacts"]
        require_same(before)
        copied_source_ignore_policy(a.copied_source_ignore_policy, a.expected_copied_source_ignore_policy_sha256, repository, out)
        status["probe_exit_code"] = command([artifacts["executable"], "--mode", a.mode, "--output", out / (a.mode + ".json")],
                                             out, out / "run.log", allow_failure=True)
        report_bytes = (out / (a.mode + ".json")).read_bytes()
        status["probe_report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
        if a.mode == "quality":
            status["quality_report_sha256"] = status["probe_report_sha256"]
            status["execution_before_sha256"] = digest(out / "execution-before.json")
        result = json.loads(report_bytes)
        status["probe_result_summary"] = {key: result.get(key) for key in ("completed", "passed", "eligible_candidate_quality_passed",
                                          "candidate_shapes_executed", "unavailable_candidate_shapes", "failed_candidate_cases_or_timing_phases",
                                          "baseline_cases_failed", "error")}
        status["produced_vectors"] = {str(path): dict(sha256=digest(path), bytes=path.stat().st_size) for path in sorted(out.glob("*.f32"))}
        generated = snapshot([*status["produced_vectors"], out / (a.mode + ".json")])
        if generated[str((out / (a.mode + ".json")).absolute())]["sha256"] != status["probe_report_sha256"]:
            raise ValueError("Parsed numerical report changed before output guard")
        if status["probe_exit_code"] != 0 or not required_coverage(result, a.mode):
            raise ValueError("Strict eligible quality/coverage failed; see preserved probe report")
        status["produced_vectors"] = complete_vector_artifacts(out, a.mode)
        generated = extend_snapshot(generated, status["produced_vectors"])
        (out / "generated-vector-guards.json").write_text(json.dumps(generated, indent=2) + "\n")
        status["completed"] = True
        status["passed"] = True
        status["coverage"] = dict(candidate_shapes_executed=14, unavailable=1, all_original_fixtures_candidate_passed=True)
    except Exception as error:
        status["error"] = str(error)
    finally:
        if generated is not None:
            try:
                require_same(generated)
                status["generated_outputs_unchanged"] = True
            except Exception as error:
                status["completed"] = status["passed"] = False
                status["generated_outputs_unchanged"] = False
                status["generated_guard_error"] = str(error)
        if before is not None:
            try:
                require_same(before)
                copied_source_ignore_policy(a.copied_source_ignore_policy, a.expected_copied_source_ignore_policy_sha256, repository, out)
                status["input_guards_unchanged"] = True
            except Exception as error:
                status["completed"] = status["passed"] = False
                status["input_guards_unchanged"] = False
                status["guard_error"] = str(error)
            try:
                after = snapshot(before)
            except Exception as error:
                after = dict(available=False, error=str(error))
            (out / "execution-after.json").write_text(json.dumps(after, indent=2) + "\n")
        status["guard_scope"] = "Source/README/runner, exact unchanged upstream tracked files, frozen native, preprocessor closure, selected CUDA/compiler tools, explicit CUDA libraries and ldd-resolved ELF libraries. No claim to bind other unenumerated implicit or dynamically loaded inputs."
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({k: status.get(k) for k in ("completed", "passed", "error", "input_guards_unchanged")}))
    return 0 if status["completed"] and status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
