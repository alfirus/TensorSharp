#!/usr/bin/env python3
"""Build a private macOS ARM RMS dispatch experiment from captured native inputs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
HELPER = HERE / "embeddinggemma2-cpu-alignment-candidate.py"
HELPER_SHA = "5618ee77b53f01015478b1dc8b68373add948bc7fcb67a203be3a6c5116fe64b"
REVISION = "3612c0af93f2cfc8f048295300a61187723b12ae"
OLD = """#if defined(__linux__) && defined(__x86_64__)
    // Qualified Linux CPU Q/K/V shapes retain their exact F32 arithmetic while
    // assigning head*token rows, rather than only heads, across the worker team.
    // The helper retains upstream RMS below its measured token crossover.
    // Default only the measured seven/eight-worker configurations; other worker
    // counts retain the ordinary path until their own timing qualification.
    e->cpu_flat_qkv_rms = validated_cpu_geometry && ((threads > 0 ? threads : 4) == 7 || (threads > 0 ? threads : 4) == 8);
#endif
"""
NEW = OLD.replace("#if defined(__linux__) && defined(__x86_64__)",
                  "#if (defined(__linux__) && defined(__x86_64__)) || (defined(__APPLE__) && defined(__aarch64__))").replace(
    "// Qualified Linux CPU Q/K/V shapes", "// Qualified Linux and macOS ARM CPU Q/K/V shapes")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
        return h.hexdigest()


def helper():
    require(sha(HELPER) == HELPER_SHA, "Captured compile helper changed; independent review required")
    spec = importlib.util.spec_from_file_location("rms_arm_captured_compile", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scoped_source(source):
    require(source.count(OLD) == 1 and NEW not in source, "Require exactly the known Linux RMS dispatch block")
    require("cpu_qkv_rms_minimum_tokens" not in source and '#include "embedding_cpu_rms.h"' in source,
            "Use the unchanged owned RMS header; do not insert a different crossover")
    require(source.index("bool validated_cpu_geometry =") < source.index(OLD), "Require preceding qualified geometry")
    return source.replace(OLD, NEW)


def snapshot(paths):
    return {str(p): {"sha256": sha(p), "resolved": str(p.resolve(strict=True))}
            for p in sorted({Path(x).absolute() for x in paths}, key=str)}


def unchanged(before):
    require(snapshot(before) == before, "A source/compiler/link input changed")


def extend_snapshot(before, paths):
    unchanged(before)
    result = dict(before)
    result.update(snapshot(path for path in paths if str(Path(path).absolute()) not in before))
    unchanged(result)
    return result


def safe_output(path, repository, build, source):
    path, repository = Path(path).absolute(), Path(repository).resolve()
    require(not path.exists() and not path.is_symlink(), "Preserve existing outputs; choose a new directory")
    relative = path.relative_to(repository)
    require(".." not in relative.parts and len(relative.parts) > 1, "Traversal-free child output required")
    physical = path.parent.resolve() / path.name
    rel = physical.relative_to(repository)
    allowed = lambda parts: parts[:2] == ("docs", "validation") or parts[:1] == ("artifacts",)
    require(allowed(relative.parts) and allowed(rel.parts), "Generated outputs must remain under ignored evidence prefixes")
    require(not physical.is_relative_to(build.resolve()) and not physical.is_relative_to(source.parent.resolve()),
            "Output must not modify the ordinary native source/build")
    run = subprocess.run(["git", "check-ignore", "--quiet", "--", str(relative / "probe")], cwd=repository)
    require(run.returncode == 0, "Output must be ignored by the actual repository")
    return path


def mac_argv(text):
    command = shlex.split(text)
    require(command and not any(arg in (";", "&&", "||", "|", ">", "<") or "$(" in arg or "`" in arg for arg in command), "Shell syntax is unsupported")
    for i, arg in enumerate(command):
        require(not arg.startswith("@") or i > 0 and command[i - 1] == "-install_name" and arg == "@rpath/libGgmlOps.dylib", "Uncaptured link response file")
    return command


def rewrite_link(command, build, original_object, native, staged_object, staged_native, base):
    command = list(command)
    require("-dynamiclib" in command and "-shared" not in command, "Require captured Mach-O dynamic-library recipe")
    require(base.resolved(command[base.option(command, "-o")], build) == native, "Unexpected ordinary native link output")
    matches = [i for i, arg in enumerate(command) if not arg.startswith("-") and base.resolved(arg, build) == original_object]
    require(len(matches) == 1, "Encoder object must occur exactly once")
    command[matches[0]] = str(staged_object)
    command[base.option(command, "-o")] = str(staged_native)
    require(command[base.option(command, "-install_name")] == "@rpath/libGgmlOps.dylib", "Keep original install name")
    i = 1
    while i < len(command):
        arg = command[i]
        if arg in ("-o", "-install_name", "-arch", "-framework"):
            require(i + 1 < len(command), "Missing captured link operand"); i += 2; continue
        require(arg in ("-O3", "-DNDEBUG", "-dynamiclib", "-Wl,-headerpad_max_install_names", "-lm") or
                not arg.startswith("-") and re.search(r"\.(?:o|a|dylib)$", arg) and not arg.startswith("@"),
                "Unsupported linker option/input/side output: " + arg)
        i += 1
    return command


def validate_compile(command):
    i = 1
    while i < len(command):
        arg = command[i]
        if arg in ("-arch", "-MF", "-MT", "-o", "-c"):
            require(i + 1 < len(command), "Missing captured compiler operand"); i += 2; continue
        require(arg in ("-O3", "-DNDEBUG", "-std=gnu++17", "-std=c++17", "-fPIC", "-MD", "-Wall", "-Wextra", "-Werror") or
                arg.startswith(("-D", "-I", "-F")), "Unsupported compiler option/side output: " + arg)
        i += 1


def dependency_spellings(path, cwd):
    text = path.read_text().replace("\\\n", " ")
    require(":" in text, "Malformed original dependency closure")
    return [(Path(value) if Path(value).is_absolute() else cwd / value).absolute() for value in shlex.split(text.split(":", 1)[1])]


def link_inputs(command, build, sdk, base):
    result = set()
    frameworks = []
    for i, arg in enumerate(command[1:], 1):
        if command[i - 1] in ("-o", "-install_name"):
            continue
        if arg == "-framework":
            require(i + 1 < len(command), "Missing framework operand")
            name = command[i + 1]
            require(re.fullmatch(r"[A-Za-z0-9_]+", name), "Unknown framework spelling")
            path = sdk / "System/Library/Frameworks" / (name + ".framework") / (name + ".tbd")
            require(path.is_file(), "Cannot bind framework stub: " + str(path))
            result.add(path); frameworks.append(name)
        elif not arg.startswith("-") and re.search(r"\.(?:o|a|dylib)$", arg):
            path = Path(arg) if Path(arg).is_absolute() else build / arg
            require(path.is_file(), "Missing captured link input: " + str(path)); result.add(path.absolute())
        elif arg.startswith("-l"):
            require(arg == "-lm", "Unqualified implicit -l library: " + arg)
    # Bind SDK link metadata, including reexports and C++ auto-link stubs. These
    # are link-time interfaces, not the dyld shared-cache implementation bytes.
    result.update(sdk.rglob("*.tbd"))
    for name in ("libSystem.tbd", "libc++.tbd", "libm.tbd"):
        require((sdk / "usr/lib" / name).is_file(), "Missing system SDK link stub: " + name)
        result.add(sdk / "usr/lib" / name)
    return result, frameworks


def run(command, cwd, log):
    with log.open("xb") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def build_candidate(args, *, source_transform=None, scope=None, additional_guard_paths=()):
    require(platform.system() == "Darwin" and platform.machine() == "arm64", "Actual staging requires macOS ARM64")
    source, native, build, ggml = args.source.resolve(), args.native_library.resolve(), args.build_dir.resolve(), args.ggml_dir.resolve()
    require(sha(source) == args.expected_source_sha256 and sha(native) == args.expected_native_sha256, "Expected ordinary source/native identity differs")
    require(native.name == "libGgmlOps.dylib", "Require the ordinary macOS native library")
    repository = HERE.parents[1]
    out = safe_output(args.output_dir, repository, build, source)
    for key in ("DEVELOPER_DIR", "SDKROOT", "TOOLCHAINS", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH", "CCC_OVERRIDE_OPTIONS", "MACOSX_DEPLOYMENT_TARGET", *[key for key in os.environ if key.startswith("DYLD_")]):
        require(not os.environ.get(key), "Uncaptured compiler/runtime override: " + key)
    raw_source = source.read_bytes()
    require(hashlib.sha256(raw_source).hexdigest() == args.expected_source_sha256, "Source bytes changed before parsing")
    original = raw_source.decode("utf-8"); candidate = (scoped_source if source_transform is None else source_transform)(original)
    base = helper()
    capture_paths = [source, native, HELPER, build / "CMakeFiles/GgmlOps.dir/link.txt"]
    capture_paths += [p for p in (build / "compile_commands.json", build / "CMakeFiles/GgmlOps.dir/flags.make", build / "CMakeFiles/GgmlOps.dir/build.make") if p.is_file()]
    captured = snapshot(capture_paths)
    compile_argv, metadata, sources = base.compile_recipe(build, source)
    link_file = build / "CMakeFiles/GgmlOps.dir/link.txt"
    link_argv = mac_argv(link_file.read_text()); unchanged(captured)
    require(compile_argv[0] == link_argv[0], "Require the same captured C++ driver spelling for compilation and linking")
    driver = Path(shutil.which(compile_argv[0])).absolute()
    require(driver.is_file() and driver.name in ("c++", "clang++"), "Require the actual captured C++ driver; resolving clang++ to clang loses C++ autolink")
    require("-arch" in compile_argv and compile_argv[base.option(compile_argv, "-arch")] == "arm64", "Require captured ARM64 compilation")
    require("-arch" in link_argv and link_argv[base.option(link_argv, "-arch")] == "arm64", "Require captured ARM64 linking")
    validate_compile(compile_argv)
    require(not any(arg in ("-Ofast", "-ffast-math", "-fassociative-math", "-ffinite-math-only", "-funsafe-math-optimizations", "-ffp-contract=fast") for arg in compile_argv), "Unqualified floating-point compilation")
    sdk = Path(subprocess.check_output(["/usr/bin/xcrun", "--show-sdk-path"], text=True).strip()).absolute()
    actual_cpp = Path(subprocess.check_output(["/usr/bin/xcrun", "--find", "clang++"], text=True).strip()).absolute()
    actual_ld = Path(subprocess.check_output(["/usr/bin/xcrun", "--find", "ld"], text=True).strip()).absolute()
    revision = subprocess.check_output(["git", "-C", str(ggml), "rev-parse", "HEAD"], text=True).strip()
    require(revision == REVISION and args.expected_ggml_revision == REVISION, "Require the exact unchanged ggml revision")
    require(not subprocess.check_output(["git", "-C", str(ggml), "status", "--porcelain"], text=True).strip(), "Upstream ggml is modified")
    tracked = subprocess.check_output(["git", "-C", str(ggml), "ls-files", "-z"])
    paths = {ggml / os.fsdecode(x) for x in tracked.split(b"\0") if x}
    paths.update(capture_paths)
    paths.update(Path(path).absolute() for path in additional_guard_paths)
    paths.update([source, native, HELPER, Path(__file__).absolute(), HERE / "embeddinggemma2-cpu-rms-arm-candidate.md", build / "CMakeCache.txt", link_file, driver, actual_cpp, actual_ld, Path("/usr/bin/xcrun"), Path("/usr/bin/xcode-select"), *metadata, *sources])
    paths.update(p for p in source.parent.iterdir() if p.is_file() and p.suffix in (".h", ".hpp", ".cpp", ".c", ".cu", ".mm", ".m"))
    commands = {}; original_object = None
    for name in ("baseline", "candidate"):
        private = out / name
        command, obj = base.rewrite_compile(compile_argv, build, source, private / source.name, private / "ggml_ops_embeddings.cpp.o", private / "encoder.d")
        require(command[0] == compile_argv[0], "Compiler driver spelling was changed")
        if original_object is not None:
            require(original_object == obj, "Mismatched encoder object")
        original_object = obj
        commands[name] = {"compile": command, "link": rewrite_link(link_argv, build, obj, native, private / "ggml_ops_embeddings.cpp.o", private / native.name, base)}
    original_dep = Path(str(original_object) + ".d")
    require(original_dep.is_file(), "Captured encoder include dependency closure is required")
    paths.update([original_object, original_dep]); paths.update(dependency_spellings(original_dep, build)); paths.update(base.dependencies(original_dep, build))
    libraries, frameworks = link_inputs(link_argv, build, sdk, base); paths.update(libraries)
    unchanged(captured)
    before = snapshot(paths)
    require(all(before[path] == value for path, value in captured.items()), "Captured recipe/source binding was rebased")
    require(before[str(source)]["sha256"] == args.expected_source_sha256 and before[str(native)]["sha256"] == args.expected_native_sha256 and before[str(HELPER)]["sha256"] == HELPER_SHA,
            "Validated source/native/helper changed before initial guard")
    out.mkdir(parents=True)
    for name, text in (("baseline", original), ("candidate", candidate)):
        (out / name).mkdir(); (out / name / source.name).write_text(text)
    (out / "reference").mkdir(); shutil.copy2(native, out / "reference" / native.name)
    derived = snapshot([out / name / source.name for name in commands] + [out / "reference" / native.name])
    report = {"completed": False, "built": False, "model_qualified": False, "performance_qualified": False,
              "scope": scope if scope is not None else "Isolated macOS ARM64 validated Gemma2 Q/K/V RMS dispatch at seven/eight workers and the unchanged minimum8-token helper; no arithmetic/header/ordinary-build changes.",
              "ggml_revision": revision, "ordinary_native_sha256": args.expected_native_sha256,
              "ordinary_encoder_object_sha256": sha(original_object), "source_sha256": args.expected_source_sha256,
              "staged_sources": {name: {"path": str(out / name / source.name), "sha256": sha(out / name / source.name)} for name in commands},
              "original_compile_command": compile_argv, "original_link_command": link_argv, "commands": commands,
              "driver_spelling_preserved": True, "actual_cpp": str(actual_cpp), "actual_ld": str(actual_ld), "sdk": str(sdk), "frameworks": frameworks,
              "compiler_version": subprocess.check_output([str(driver), "--version"], text=True),
              "before": before, "runs": [], "reference_native_sha256": sha(out / "reference" / native.name),
              "limitations": ["Build/source proof only; root must independently qualify full-model/HTTP/raw/multimodal/timing behavior before any promotion.",
                  "Recompiled baseline and candidate share all captured nonencoder objects, archives, flags and link metadata. Baseline must be compared against frozen539 if its rebuilt byte hash differs.",
                  "SDK .tbd files bind link interfaces, not Apple dyld shared-cache implementation bytes. System shared-cache libraries are not file-hashed."]}
    try:
        for name in commands:
            require(sha(out / name / source.name) == report["staged_sources"][name]["sha256"], "Staged source changed")
            if args.execute:
                for phase, command in commands[name].items():
                    unchanged(before); unchanged(derived)
                    code = run(command, build, out / name / (phase + ".log"))
                    report["runs"].append({"name": name, "phase": phase, "command": command, "exit_code": code})
                    require(code == 0, name + " " + phase + " failed; see preserved isolated log")
                    unchanged(before); unchanged(derived)
                    if phase == "compile":
                        headers = {p for p in base.dependencies(out / name / "encoder.d", build) if p != (out / name / source.name).resolve()}
                        require(headers <= {p.resolve() for p in paths}, "New compile escaped the captured original header closure")
                        derived = extend_snapshot(derived, [out / name / "ggml_ops_embeddings.cpp.o", out / name / "encoder.d"])
                    else:
                        derived = extend_snapshot(derived, [out / name / native.name])
                report[name + "_native_sha256"] = sha(out / name / native.name)
                report[name + "_object_sha256"] = sha(out / name / "ggml_ops_embeddings.cpp.o")
        if args.execute:
            report["built"] = True
            report["rebuilt_baseline_matches_ordinary_native"] = report["baseline_native_sha256"] == args.expected_native_sha256
            report["baseline_vs_frozen_qualification_required"] = not report["rebuilt_baseline_matches_ordinary_native"]
        report["completed"] = True
    except Exception as error:
        report["error"] = str(error)
    finally:
        try:
            report["after"] = snapshot(paths); report["guarded_inputs_unchanged"] = report["after"] == before
            require(report["guarded_inputs_unchanged"], "Original source/build/compile/link inputs changed")
            report["derived_inputs_before"] = derived
            report["derived_inputs_after"] = snapshot(derived)
            require(report["derived_inputs_after"] == derived, "Generated source/object/dependency/native output changed after binding")
            require(all(sha(out / name / source.name) == report["staged_sources"][name]["sha256"] for name in commands), "Staged source changed")
            require(sha(out / "reference" / native.name) == args.expected_native_sha256, "Reference copy changed")
            report["ggml_revision_after"] = subprocess.check_output(["git", "-C", str(ggml), "rev-parse", "HEAD"], text=True).strip()
            report["ggml_status_after"] = subprocess.check_output(["git", "-C", str(ggml), "status", "--porcelain"], text=True).strip()
            require(report["ggml_revision_after"] == REVISION and not report["ggml_status_after"], "Upstream changed")
        except Exception as error:
            report["completed"] = False; report["guard_error"] = str(error)
        (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "source", "native-library", "ggml-dir", "output-dir"):
        p.add_argument("--" + name, required=True, type=Path)
    for name in ("expected-source-sha256", "expected-native-sha256", "expected-ggml-revision"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--execute", action="store_true")
    report = build_candidate(p.parse_args())
    print(json.dumps({key: report.get(key) for key in ("completed", "built", "guarded_inputs_unchanged", "rebuilt_baseline_matches_ordinary_native", "model_qualified", "error", "guard_error")}))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
