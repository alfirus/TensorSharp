#!/usr/bin/env python3
"""Compile the owned LTX denoiser probe against an unchanged SD.cpp build."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--kind", choices=("denoiser", "generation"), default="denoiser")
    args = parser.parse_args()
    build, output = args.build.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = json.loads((build / "compile_commands.json").read_text())
    original = next(item for item in commands if item["file"].endswith("/examples/cli/main.cpp"))
    root = Path(original["file"]).parents[2]
    files = json.loads(args.source_manifest.read_text())
    for item in files:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("Source differs from clean export manifest: " + item["path"])
    pins = json.loads(args.build_manifest.read_text())
    expected_binary = pins["remote_cuda_build" if "-DGGML_USE_CUDA" in original["command"] else "local_metal_build"]["sha256"]
    if sha(build / "bin/sd-cli") != expected_binary:
        raise ValueError("Baseline CLI differs from verified build manifest")
    source = Path(__file__).with_name("ltx25-sdcpp-" + args.kind + ".cpp").resolve()
    obj, binary = output / "probe.o", output / ("ltx25-sdcpp-" + args.kind)
    command = shlex.split(original["command"])
    command[command.index("-o") + 1] = str(obj)
    command = [str(source) if item == original["file"] else item for item in command]
    subprocess.run(command, cwd=original["directory"], check=True)
    directory = build / "examples/cli"
    link = shlex.split((directory / "CMakeFiles/sd-cli.dir/link.txt").read_text())
    link = [str(obj) if item.endswith("CMakeFiles/sd-cli.dir/main.cpp.o") else item for item in link]
    link[link.index("-o") + 1] = str(binary)
    subprocess.run(link, cwd=directory, check=True)
    record = {"source": str(source), "source_sha256": sha(source), "binary": str(binary), "binary_sha256": sha(binary),
              "build_compile_commands_sha256": sha(build / "compile_commands.json"), "compile_argv": command, "link_argv": link,
              "source_revisions": {"sdcpp": pins["sdc_revision"], "ggml": pins["ggml_revision"]},
              "clean_source_manifest_sha256": sha(args.source_manifest), "verified_source_files": len(files),
              "linked_libraries_sha256": {str((directory / item).resolve()): sha((directory / item).resolve()) for item in link if item.endswith((".a", ".so", ".dylib"))},
              "source_modifications": "None; owned probe includes unchanged upstream headers and links existing unchanged libraries."}
    (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"binary": str(binary), "sha256": record["binary_sha256"]}))


if __name__ == "__main__":
    main()
