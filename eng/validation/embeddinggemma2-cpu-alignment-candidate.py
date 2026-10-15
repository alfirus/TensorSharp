#!/usr/bin/env python3
"""Stage one scoped padding experiment; never invoke or modify the normal build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

ANCHOR = "    e->weights_buffer = ggml_backend_alloc_ctx_tensors(e->weights_ctx, e->backend);\n"
INSERTION = "    if (validated_cpu_geometry && e->cpu_float_linears) e->cpu_alignment = 1;\n"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scoped_source(source):
    require(source.count(ANCHOR) == 1, "Require exactly one known weights-buffer anchor")
    require(INSERTION not in source, "Candidate insertion already present")
    prefix = source.split(ANCHOR)[0]
    require(prefix.rstrip().endswith("if (all_quantized_optimized) e->cpu_alignment = 4;\n    }"),
            "Anchor must follow the complete optimized-weight loop")
    require("bool validated_cpu_geometry =" in prefix and "e->cpu_float_linears = std::all_of" in prefix,
            "Validated geometry and all-F32 predicates are required")
    return source.replace(ANCHOR, INSERTION + ANCHOR)


def argv(text):
    result = shlex.split(text)
    require(result and not any(x.startswith("@") for x in result), "Response files are unsupported")
    require(not any(x in {";", "&&", "||", "|", ">", "<"} or "$(" in x or "`" in x for x in result),
            "Shell expansion/operators are unsupported")
    return result


def option(command, name, required=True):
    found = [i for i, value in enumerate(command) if value == name]
    require(len(found) <= 1 and (bool(found) or not required), "Require unique " + name)
    if not found:
        return None
    index = found[0]
    require(index + 1 < len(command), "Missing operand for " + name)
    return index + 1


def resolved(value, cwd):
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).resolve()


def compile_recipe(build, source):
    database = build / "compile_commands.json"
    metadata = []
    if database.is_file():
        entries = json.loads(database.read_text())
        matches = [e for e in entries if resolved(e["file"], Path(e["directory"])) == source]
        if matches:
            require(len(matches) == 1, "Ambiguous encoder compile database entry")
            entry = matches[0]
            require(Path(entry["directory"]).resolve() == build, "Compile directory differs from build root")
            command = entry.get("arguments") or argv(entry["command"])
            require(not any(str(x).startswith("@") for x in command), "Response files are unsupported")
            metadata.append(database)
            return list(command), metadata, [resolved(e["file"], Path(e["directory"])) for e in entries]
    flags_file = build / "CMakeFiles/GgmlOps.dir/flags.make"
    make_file = build / "CMakeFiles/GgmlOps.dir/build.make"
    flags = {}
    for line in flags_file.read_text().splitlines():
        match = re.fullmatch(r"(CXX_DEFINES|CXX_INCLUDES|CXX_FLAGS) = (.*)", line)
        if match:
            require(match[1] not in flags, "Duplicate CXX variable")
            flags[match[1]] = match[2]
    require(len(flags) == 3, "Require exactly the three captured CXX variables")
    recipes = [line.strip() for line in make_file.read_text().splitlines()
               if line.startswith("\t") and " -c " in line and str(source) in line]
    require(len(recipes) == 1, "Require exactly one known encoder compile recipe")
    recipe = recipes[0]
    for name, value in flags.items():
        recipe = recipe.replace("$(" + name + ")", value)
    metadata += [flags_file, make_file]
    return argv(recipe), metadata, [source]


def rewrite_compile(command, build, source, staged_source, staged_object, staged_dep):
    command = list(command)
    require(resolved(command[option(command, "-c")], build) == source, "Unexpected -c source")
    original_object = resolved(command[option(command, "-o")], build)
    require(original_object.name == "ggml_ops_embeddings.cpp.o", "Unexpected encoder object")
    command[option(command, "-c")] = str(staged_source)
    command[option(command, "-o")] = str(staged_object)
    for flag, destination in (("-MF", staged_dep), ("-MT", staged_object)):
        index = option(command, flag, False)
        if index is None:
            command += [flag, str(destination)]
        else:
            command[index] = str(destination)
    side_prefixes = ("-save-temps", "--save-temps", "-MJ", "--serialize-diagnostics", "-fdump", "-fprofile",
                     "-ftest-coverage", "--coverage", "-fstack-usage", "-ftime-trace", "-gsplit-dwarf",
                     "-fmodule", "-fplugin", "-flto", "-Wp,")
    require(not any(x.startswith(side_prefixes) or x in {"-M", "-MM", "-MP", "-MG", "-MQ", "-Xclang"}
                    for x in command),
            "Uncaptured compiler side outputs are unsupported")
    if "-MD" not in command:
        command.append("-MD")
    command += ["-iquote", str(source.parent)]
    return command, original_object


def rewrite_link(command, build, original_object, original_library, staged_object, staged_library):
    command = list(command)
    require(resolved(command[option(command, "-o")], build) == original_library, "Unexpected native link output")
    matches = [i for i, x in enumerate(command) if not x.startswith("-") and resolved(x, build) == original_object]
    require(len(matches) == 1, "Link must contain encoder object exactly once")
    command[matches[0]] = str(staged_object)
    command[option(command, "-o")] = str(staged_library)
    require(not any(x.startswith("@") for x in command), "Link response files are unsupported")
    require(not any(x == "-Xlinker" or (x.startswith("-Wl,") and
                    any(y in x for y in ("-Map", "--out-implib", "--output-def", "--version-script",
                                         "--dependency-file", "--plugin", ",-T", ",@")))
                    for x in command), "Uncaptured linker side outputs/scripts are unsupported")
    return command


def dependencies(path, cwd):
    text = path.read_text().replace("\\\n", " ")
    require(":" in text, "Malformed dependency file")
    return [resolved(x, cwd) for x in shlex.split(text.split(":", 1)[1])]


def executable(name):
    path = shutil.which(name)
    require(path is not None, "Missing compiler/linker executable: " + name)
    return Path(path).resolve()


def compiler_file(compiler, name, program=False):
    argument = ("-print-prog-name=" if program else "-print-file-name=") + name
    value = subprocess.check_output([str(compiler), argument], text=True).strip()
    path = Path(value)
    if program and not path.is_absolute():
        found = shutil.which(value)
        path = Path(found) if found else path
    return path.resolve() if path.is_file() else None


def library_files(command, build, compiler):
    result = set()
    directories = [resolved(x[2:], build) for x in command if x.startswith("-L") and len(x) > 2]
    for x in command[1:]:
        if not x.startswith("-") and re.search(r"\.(?:o|a|so(?:\.\d+)*)$", x):
            path = resolved(x, build)
            require(path.is_file(), "Missing explicit link input: " + str(path))
            result.add(path)
        elif x.startswith("-l") and len(x) > 2:
            name = x[2:]
            candidates = [d / ("lib" + name + suffix) for d in directories for suffix in (".so", ".a")]
            found = next((p.resolve() for p in candidates if p.is_file()), None)
            if found is None:
                found = compiler_file(compiler, "lib" + name + ".so") or compiler_file(compiler, "lib" + name + ".a")
            require(found is not None, "Cannot bind -l input: " + name)
            result.add(found)
    for name in ("libstdc++.so", "libgcc_s.so", "libgcc.a", "libc.so", "libc.so.6", "libm.so",
                 "libgomp.so", "libpthread.so", "libdl.so", "librt.so",
                 "crtbeginS.o", "crtendS.o", "crti.o", "crtn.o"):
        found = compiler_file(compiler, name)
        if found:
            result.add(found)
    # GNU .so files can be linker scripts naming concrete system libraries.
    pending = list(result)
    while pending:
        path = pending.pop()
        if path.stat().st_size > 65536:
            continue
        content = path.read_bytes()
        if not (b"GROUP" in content or b"INPUT" in content):
            continue
        for name in re.findall(rb"/[^\s()]+|[A-Za-z0-9_+.-]+\.(?:so(?:\.\d+)*|a)", content):
            child = Path(os.fsdecode(name))
            if not child.is_absolute():
                child = path.parent / child
                if not child.is_file():
                    child = compiler_file(compiler, os.fsdecode(name))
            if child is None:
                continue
            if child.is_file() and child.resolve() not in result:
                result.add(child.resolve())
                pending.append(child.resolve())
    return result


def ignored_output(path):
    parts = path.parts
    return "artifacts" in parts or any(parts[i:i + 2] == ("docs", "validation") for i in range(len(parts) - 1))


def build_candidate(args, *, source_transform=None, scope=None):
    build, source, output = args.build_dir.resolve(), args.source.resolve(), args.output_dir.resolve()
    native, ggml = args.native_library.resolve(), args.ggml_dir.resolve()
    require(ignored_output(output) and not output.is_relative_to(build) and not output.is_relative_to(source.parent),
            "Output must be a new ignored docs/validation or artifacts directory outside native source/build")
    require(not output.exists(), "Output directory already exists; preserve prior snapshots")
    require(sha(source) == args.expected_source_sha256, "Production source hash mismatch")
    require(sha(native) == args.expected_native_sha256, "Ordinary native library hash mismatch")
    candidate = (source_transform or scoped_source)(source.read_text())
    revision = subprocess.check_output(["git", "-C", str(ggml), "rev-parse", "HEAD"], text=True).strip()
    require(revision == args.expected_ggml_revision, "Upstream ggml revision mismatch")
    require(not subprocess.check_output(["git", "-C", str(ggml), "status", "--porcelain", "--untracked-files=no"], text=True).strip(),
            "Upstream tracked checkout is modified")
    original_compile, metadata, sources = compile_recipe(build, source)
    link_file = build / "CMakeFiles/GgmlOps.dir/link.txt"
    original_link = argv(link_file.read_text())
    require(native.name == "libGgmlOps.so" and "-shared" in original_link,
            "Only captured Linux ELF GgmlOps shared-library links are supported")
    compiler, linker = executable(original_compile[0]), executable(original_link[0])
    require(compiler == linker, "Require one captured C++ driver for compile and link")
    staged_source, staged_object = output / source.name, output / "ggml_ops_embeddings.cpp.o"
    staged_dep, staged_library = output / "ggml_ops_embeddings.cpp.o.d", output / native.name
    compile_command, original_object = rewrite_compile(original_compile, build, source, staged_source, staged_object, staged_dep)
    link_command = rewrite_link(original_link, build, original_object, native, staged_object, staged_library)
    original_dep = Path(str(original_object) + ".d")
    require(original_dep.is_file(), "Original encoder dependency file required to bind actual included headers")
    guard_paths = {source, native, original_object, original_dep, link_file, compiler, *metadata, *sources}
    guard_paths.update(p for p in source.parent.iterdir()
                       if p.is_file() and p.suffix in {".h", ".hpp", ".c", ".cpp", ".cu", ".m", ".mm"})
    guard_paths.add(Path(shutil.which(original_compile[0])).absolute())
    guard_paths.add(Path(shutil.which(original_link[0])).absolute())
    guard_paths.update(dependencies(original_dep, build))
    guard_paths.update(library_files(original_link, build, linker))
    for name in ("cc1plus", "collect2", "ld"):
        found = compiler_file(compiler, name, True)
        if found:
            guard_paths.add(found)
    require(all(p.is_file() for p in guard_paths), "A guarded build input is missing")
    before = {str(p): sha(p) for p in sorted(guard_paths)}
    output.mkdir(parents=True)
    staged_source.write_text(candidate)
    report = {"completed": False, "built": False, "production_changed": False, "model_qualified": False,
              "scope": scope or "One scoped validated all-F32 CPU alignment1 insertion in an isolated owned source copy.",
              "ggml_revision": revision, "source_sha256": sha(source), "staged_source_sha256": sha(staged_source),
              "original_compile_command": original_compile, "compile_command": compile_command,
              "original_link_command": original_link, "link_command": link_command, "before_hashes": before,
              "link_guard_scope": "Explicit objects/libraries, resolved -l names, recursive GNU linker scripts, standard C++/C/OpenMP toolchain libraries and compiler/linker tools. Arbitrary injected plugins/side-output flags are unsupported.",
              "runs": [], "quality_gates_executed": False}
    try:
        if args.execute:
            for name, command in (("compile", compile_command), ("link", link_command)):
                require(all(sha(p) == digest for p, digest in before.items()), "Guarded input changed before " + name)
                with (output / (name + ".log")).open("w") as log:
                    result = subprocess.run(command, cwd=build, stdout=log, stderr=subprocess.STDOUT)
                report["runs"].append({"name": name, "exit_code": result.returncode})
                require(result.returncode == 0, name + " failed; see isolated log")
            require(staged_library.is_file() and staged_object.is_file(), "Candidate outputs missing")
            # New include discovery may not escape the original captured header closure.
            fresh_headers = {p for p in dependencies(staged_dep, build) if p != staged_source}
            require(fresh_headers <= guard_paths, "Candidate discovered unguarded header dependencies")
            report["built"] = True
            report["candidate_library_sha256"] = sha(staged_library)
            report["candidate_object_sha256"] = sha(staged_object)
        report["completed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        report["after_hashes"] = {str(p): sha(p) if p.is_file() else None for p in sorted(guard_paths)}
        report["guarded_inputs_unchanged"] = report["after_hashes"] == before
        report["ordinary_native_unchanged"] = native.is_file() and sha(native) == args.expected_native_sha256
        report["ggml_revision_after"] = subprocess.check_output(["git", "-C", str(ggml), "rev-parse", "HEAD"], text=True).strip()
        report["ggml_tracked_status_after"] = subprocess.check_output(
            ["git", "-C", str(ggml), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        report["upstream_unchanged"] = report["ggml_revision_after"] == revision and not report["ggml_tracked_status_after"]
        if not report["guarded_inputs_unchanged"] or not report["upstream_unchanged"]:
            report["completed"] = False
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    require(report["guarded_inputs_unchanged"], "Guarded inputs changed; candidate is unqualified")
    require(report["upstream_unchanged"], "Upstream checkout changed; candidate is unqualified")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "source", "native-library", "ggml-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-source-sha256", "expected-native-sha256", "expected-ggml-revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true", help="Otherwise prepare/hash/record only; no compile/link")
    args = parser.parse_args()
    report = build_candidate(args)
    print(json.dumps({k: report[k] for k in ("completed", "built", "guarded_inputs_unchanged", "ordinary_native_unchanged", "model_qualified")}))


if __name__ == "__main__":
    main()
