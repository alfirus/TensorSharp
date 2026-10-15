#!/usr/bin/env python3
"""Build one isolated SM86 large-image CUDA launch-bound candidate, without a normal build."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HELPER = Path(__file__).with_name("embeddinggemma2-cpu-alignment-candidate.py")
# Re-reviewed optional source-transform/scope arguments preserve the original
# default compilation/link isolation and all input/output guards.
HELPER_SHA256 = "5618ee77b53f01015478b1dc8b68373add948bc7fcb67a203be3a6c5116fe64b"
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA256:
    raise ValueError("Reviewed isolation helper changed; independent re-review required")
spec = importlib.util.spec_from_file_location("isolation_controls", HELPER)
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)
require, sha, argv, resolved = controls.require, controls.sha, controls.argv, controls.resolved
option, executable = controls.option, controls.executable

KERNEL_START = "template <int WIDTH>\n__global__ __launch_bounds__(128) void vision_kernel("
KERNEL_END = "// A block owns one query/head and one key partition."
DISPATCH = """        if (q.ne[0] == 64)
            vision_kernel<64><<<blocks, 128, 0, stream>>>(q, k, v, desc.f0, static_cast<float *>(dst->data));
        else
"""
GUARDED_DISPATCH = """        if (q.ne[0] == 64) {
            const bool qualified_large_image =
                ggml_cuda_info().devices[context->device].cc == 860 &&
                (q.ne[1] == 2304 || q.ne[1] == 2340) &&
                k.ne[0] == 64 && v.ne[0] == 64 && k.ne[1] == q.ne[1] && v.ne[1] == q.ne[1] &&
                q.ne[2] == 12 && k.ne[2] == 12 && v.ne[2] == 12 &&
                q.ne[3] == 1 && k.ne[3] == 1 && v.ne[3] == 1 &&
                q.type == GGML_TYPE_F32 && k.type == GGML_TYPE_F32 && v.type == GGML_TYPE_F32 && desc.f0 == 1.0f;
            if (qualified_large_image)
                vision_kernel_minblocks3<64><<<blocks, 128, 0, stream>>>(q, k, v, desc.f0, static_cast<float *>(dst->data));
            else
                vision_kernel<64><<<blocks, 128, 0, stream>>>(q, k, v, desc.f0, static_cast<float *>(dst->data));
        } else
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def scoped_source(source):
    require(source.count(KERNEL_START) == 1 and source.count(KERNEL_END) == 1,
            "Require the exact original owned vision kernel boundaries")
    require(source.count(DISPATCH) == 1 and "vision_kernel_minblocks3" not in source,
            "Require the exact original head64 dispatch")
    first, last = source.index(KERNEL_START), source.index(KERNEL_END)
    require(first < last, "Kernel boundary order differs")
    kernel = source[first:last]
    require(kernel.count("vision_kernel(") == 1 and kernel.count("__launch_bounds__(128)") == 1,
            "Ambiguous original kernel declaration")
    candidate = kernel.replace("__launch_bounds__(128)", "__launch_bounds__(128, 3)").replace(
        "void vision_kernel(", "void vision_kernel_minblocks3(")
    require(candidate.replace("__launch_bounds__(128, 3)", "__launch_bounds__(128)").replace(
        "void vision_kernel_minblocks3(", "void vision_kernel(") == kernel, "Kernel arithmetic changed")
    return (source[:last] + candidate + source[last:]).replace(DISPATCH, GUARDED_DISPATCH)


def expand_includes(command, build):
    """Only CMake's exact include-only CUDA options file is supported; no recursive responses."""
    command = list(command)
    index = option(command, "--options-file")
    path = resolved(command[index], build)
    require(path == (build / "CMakeFiles/GgmlOps.dir/includes_CUDA.rsp").resolve(),
            "Only the captured GgmlOps CUDA include response is supported")
    data = path.read_bytes()
    entries = argv(data.decode())
    i = 0
    while i < len(entries):
        value = entries[i]
        if value in {"-I", "-isystem"}:
            i += 1
            require(i < len(entries) and not entries[i].startswith("-"), "Missing include directory")
            require(resolved(entries[i], build).is_dir(), "Missing include directory")
        else:
            prefix = next((p for p in ("-isystem=", "-I") if value.startswith(p) and len(value) > len(p)), None)
            require(prefix is not None and resolved(value[len(prefix):], build).is_dir(),
                    "CUDA response may contain only existing include directories")
        i += 1
    command[index - 1:index + 1] = entries
    require(not any(x.startswith("@") or x.startswith("--options-file") or x in {"-optf"} for x in command),
            "Other or recursive compiler options files are unsupported")
    return command, path, digest(data)


def compile_recipe(build, source):
    database = build / "compile_commands.json"
    data = database.read_bytes()
    entries = json.loads(data)
    matches = []
    for entry in entries:
        if resolved(entry["file"], Path(entry["directory"])) != source:
            continue
        require(Path(entry["directory"]).resolve() == build, "Unexpected CUDA compile cwd")
        command = entry.get("arguments") or argv(entry["command"])
        if resolved(command[option(command, "-o")], build) == build / "CMakeFiles/GgmlOps.dir/ggml_ops_attention_precision.cu.o":
            matches.append(command)
    require(len(matches) == 1, "Require one exact GgmlOps owned CUDA object entry")
    command, response, response_hash = expand_includes(matches[0], build)
    return command, {database: digest(data), response: response_hash}, {
        resolved(entry["file"], Path(entry["directory"])) for entry in entries}


def rewrite_compile(command, build, source, staged_source, staged_object, staged_dep):
    command = list(command)
    require(Path(command[0]).name == "nvcc", "Captured CUDA compiler must be nvcc")
    require(resolved(command[option(command, "-c")], build) == source, "Unexpected CUDA compile source")
    original_object = resolved(command[option(command, "-o")], build)
    require(original_object == build / "CMakeFiles/GgmlOps.dir/ggml_ops_attention_precision.cu.o",
            "Unexpected owned CUDA object")
    require([x for x in command if x.startswith("--generate-code")] == ["--generate-code=arch=compute_86,code=[sm_86]"],
            "Only the reviewed native SM86 CUDA recipe is supported")
    require(all(command.count(x) == 1 for x in ("--ftz=false", "--prec-div=true", "--prec-sqrt=true")),
            "Original precise CUDA math flags required")
    forbidden = ("-save-temps", "--save-temps", "--keep", "-keep", "-fdump", "-fprofile", "-flto", "-fplugin",
                 "--use_fast_math", "-use_fast_math", "--maxrregcount", "-maxrregcount", "--options-file=", "-optf=",
                 "--output-directory", "-odir", "--compiler-bindir", "-ccbin", "--dependency-output", "-Wp,",
                 "--ptx", "--cubin", "--fatbin", "--device-link", "--dryrun", "--threads")
    require(not any(x.startswith(forbidden) or x in {"-ptx", "-cubin", "-fatbin", "-dlink", "-M", "-MM", "-MQ", "-Xlinker", "-Xcompiler", "-Xptxas"}
                    or x.startswith(("-Xcompiler=", "-Xptxas=")) and x not in {"-Xcompiler=-fPIC"}
                    for x in command), "Unsupported CUDA side outputs, host overrides, or arithmetic flags")
    simple = {"-forward-unknown-to-host-compiler", "-O3", "-std=c++17", "-Xcompiler=-fPIC", "--ftz=false",
              "--prec-div=true", "--prec-sqrt=true", "-MD", "--generate-code=arch=compute_86,code=[sm_86]"}
    i = 1
    while i < len(command):
        value = command[i]
        if value in {"-c", "-o", "-MF", "-MT", "-I", "-isystem", "-x"}:
            i += 1
            require(i < len(command), "Missing captured CUDA option operand")
            require(value != "-x" or command[i] == "cu", "Require CUDA source language")
        else:
            require(value in simple or value.startswith(("-D", "-I", "-isystem=")),
                    "Unreviewed CUDA compiler option: " + value)
        i += 1
    command[option(command, "-c")] = str(staged_source)
    command[option(command, "-o")] = str(staged_object)
    for flag, destination in (("-MF", staged_dep), ("-MT", staged_object)):
        index = option(command, flag, False)
        if index is None:
            command += [flag, str(destination)]
        else:
            command[index] = str(destination)
    require(not any(x.startswith("@") for x in command), "CUDA response files must be fully expanded")
    if "-MD" not in command:
        command.append("-MD")
    command += ["-I", str(source.parent)]
    return command, original_object


def dependencies(data, cwd):
    value = data.decode().replace("\\\n", " ")
    require(":" in value, "Malformed original CUDA dependencies")
    return {resolved(x, cwd) for x in controls.shlex.split(value.split(":", 1)[1])}


def dependency_aliases(data, cwd):
    value = data.decode().replace("\\\n", " ")
    require(":" in value, "Malformed original CUDA dependencies")
    return {(Path(x) if Path(x).is_absolute() else cwd / x).absolute()
            for x in controls.shlex.split(value.split(":", 1)[1])}


def explicit_library_aliases(command, cwd):
    result = set()
    search = [(Path(x[2:]) if Path(x[2:]).is_absolute() else cwd / x[2:]).absolute()
              for x in command if x.startswith("-L") and len(x) > 2]
    for value in command[1:]:
        if not value.startswith("-") and re.search(r"\.(?:o|a|so(?:\.\d+)*)$", value):
            result.add((Path(value) if Path(value).is_absolute() else cwd / value).absolute())
        elif value.startswith("-l") and len(value) > 2:
            found = next((folder / ("lib" + value[2:] + suffix) for folder in search
                          for suffix in (".so", ".a") if (folder / ("lib" + value[2:] + suffix)).is_file()), None)
            if found:
                result.add(found)
    return result


def cuda_toolchain(nvcc):
    root = nvcc.parent.parent
    files = {nvcc}
    for relative in ("bin/ptxas", "bin/fatbinary", "bin/nvlink", "bin/cudafe++", "bin/bin2c", "bin/nvcc.profile", "nvvm/bin/cicc"):
        path = root / relative
        require(path.is_file(), "Missing captured CUDA toolchain input: " + str(path))
        files.add(path)
    bitcode = list((root / "nvvm/libdevice").glob("*.bc"))
    require(bitcode, "CUDA libdevice bitcode must be bound")
    files.update(bitcode)
    files.update((root / "nvvm/lib64").glob("libnvvm.so*"))
    return files


def build_candidate(args):
    build, source, output = args.build_dir.resolve(), args.source.resolve(), args.output_dir.resolve()
    native, ggml = args.native_library.resolve(), args.ggml_dir.resolve()
    require(controls.ignored_output(output) and not output.is_relative_to(build) and not output.is_relative_to(source.parent),
            "Require a fresh ignored evidence directory outside native sources/build")
    require(not output.exists(), "Preserve prior output snapshots")
    for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "GCC_EXEC_PREFIX", "COMPILER_PATH", "LD_PRELOAD"):
        require(os.environ.get(name) is None, "Uncaptured toolchain injection environment: " + name)
    source_data = source.read_bytes()
    require(digest(source_data) == args.expected_source_sha256 and sha(native) == args.expected_native_sha256,
            "Expected production source/native identity differs")
    candidate = scoped_source(source_data.decode())
    revision = subprocess.check_output(["git", "-C", str(ggml), "rev-parse", "HEAD"], text=True).strip()
    require(revision == args.expected_ggml_revision, "Upstream dependency revision mismatch")
    status_command = ["git", "-C", str(ggml), "status", "--porcelain", "--untracked-files=no"]
    require(not subprocess.check_output(status_command, text=True).strip(), "Upstream tracked checkout is modified")
    original_compile, metadata, sources = compile_recipe(build, source)
    link_file = build / "CMakeFiles/GgmlOps.dir/link.txt"
    link_data = link_file.read_bytes(); metadata[link_file] = digest(link_data)
    original_link = argv(link_data.decode())
    require(native.name == "libGgmlOps.so" and "-shared" in original_link, "Only captured Linux shared GgmlOps links supported")
    compiler, linker, host = executable(original_compile[0]), executable(original_link[0]), executable("g++")
    require(host == linker, "Default nvcc g++ host and captured C++ linker must resolve identically")
    staged_source, staged_object = output / source.name, output / "ggml_ops_attention_precision.cu.o"
    staged_dep, staged_library = output / "ggml_ops_attention_precision.cu.o.d", output / native.name
    compile_command, original_object = rewrite_compile(original_compile, build, source, staged_source, staged_object, staged_dep)
    link_command = controls.rewrite_link(original_link, build, original_object, native, staged_object, staged_library)
    original_dep = Path(str(original_object) + ".d")
    dep_data = original_dep.read_bytes(); metadata[original_dep] = digest(dep_data)
    guard_paths = {source, native, original_object, compiler, linker, host, HELPER, Path(__file__).resolve(), *metadata, *sources}
    guard_paths.update(p for p in source.parent.iterdir() if p.is_file() and p.suffix in {".h", ".hpp", ".c", ".cpp", ".cu", ".m", ".mm"})
    guard_paths.update(dependencies(dep_data, build))
    guard_paths.update(dependency_aliases(dep_data, build))
    guard_paths.update(controls.library_files(original_link, build, linker))
    guard_paths.update(explicit_library_aliases(original_link, build))
    guard_paths.update(cuda_toolchain(compiler))
    for name in ("cc1plus", "collect2", "ld"):
        found = controls.compiler_file(host, name, True)
        if found:
            guard_paths.add(found)
    for name in (original_compile[0], original_link[0], "g++"):
        guard_paths.add(Path(shutil.which(name)).absolute())
    require(all(p.is_file() for p in guard_paths), "A captured compile/link/toolchain dependency is missing")
    before = {str(p): sha(p) for p in sorted(guard_paths)}
    expected = {**metadata, source: args.expected_source_sha256, native: args.expected_native_sha256, HELPER: HELPER_SHA256}
    require(all(before[str(p)] == value for p, value in expected.items()), "Parsed metadata/source changed during capture")
    output.mkdir(parents=True); staged_source.write_text(candidate)
    report = {"completed": False, "built": False, "production_changed": False, "model_qualified": False,
              "scope": "Staged-only duplicated unchanged head64 Q128K16 kernel with minBlocks3 and exact SM86/N2304-or2340/12head/batch1/F32/scale1 self-attention dispatch.",
              "ggml_revision": revision, "source_sha256": args.expected_source_sha256, "staged_source_sha256": sha(staged_source),
              "before_hashes": before, "original_compile_command": original_compile, "compile_command": compile_command,
              "original_link_command": original_link, "link_command": link_command, "runs": [], "quality_gates_executed": False,
              "guard_scope": "Original dependency closure, all owned sources and compile-database sources, explicit/resolved link libraries plus standard C++/C/OpenMP libraries, CUDA tools/profile/libdevice, host tools and metadata. Arbitrary CUDA response/side-output flags are rejected."}
    report["toolchain_environment"] = {name: os.environ.get(name) for name in
        ("PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH",
         "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "GCC_EXEC_PREFIX", "COMPILER_PATH", "LD_PRELOAD")}
    try:
        if args.execute:
            for name, command in (("compile", compile_command), ("link", link_command)):
                require(all(sha(path) == value for path, value in before.items()), "Guarded input changed before " + name)
                require(sha(staged_source) == report["staged_source_sha256"], "Staged source changed")
                with (output / (name + ".log")).open("w") as log:
                    result = subprocess.run(command, cwd=build, stdout=log, stderr=subprocess.STDOUT)
                report["runs"].append({"name": name, "exit_code": result.returncode})
                require(result.returncode == 0, name + " failed; see isolated log")
                if name == "compile":
                    fresh = dependencies(staged_dep.read_bytes(), build) - {staged_source}
                    require(fresh <= guard_paths, "Candidate discovered an unbound include dependency")
            require(staged_object.is_file() and staged_library.is_file(), "Candidate outputs missing")
            report["built"] = True; report["candidate_library_sha256"] = sha(staged_library); report["candidate_object_sha256"] = sha(staged_object)
        report["completed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        report["after_hashes"] = {str(p): sha(p) if p.is_file() else None for p in sorted(guard_paths)}
        report["guarded_inputs_unchanged"] = report["after_hashes"] == before
        report["ordinary_native_unchanged"] = sha(native) == args.expected_native_sha256 if native.is_file() else False
        report["ggml_revision_after"] = subprocess.check_output(["git", "-C", str(ggml), "rev-parse", "HEAD"], text=True).strip()
        report["ggml_tracked_status_after"] = subprocess.check_output(status_command, text=True).strip()
        report["upstream_unchanged"] = report["ggml_revision_after"] == revision and not report["ggml_tracked_status_after"]
        if not report["guarded_inputs_unchanged"] or not report["upstream_unchanged"]:
            report["completed"] = False
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    require(report["completed"], "Candidate inputs changed; build is unqualified")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "source", "native-library", "ggml-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-source-sha256", "expected-native-sha256", "expected-ggml-revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true")
    report = build_candidate(parser.parse_args())
    print(json.dumps({k: report[k] for k in ("completed", "built", "ordinary_native_unchanged", "guarded_inputs_unchanged", "model_qualified")}))


if __name__ == "__main__":
    main()
