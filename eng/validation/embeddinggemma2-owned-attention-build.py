#!/usr/bin/env python3
"""Build isolated owned CPU math snapshots; never overwrite inference bundles.

Run only in a coordinated build window. The existing unchanged-GGML native
library is packaged without rebuilding it. CUDA package content is unused by
the subsequent CPU model gate. No live process command lines are collected.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    root, base, art = args.root.resolve(), args.output.resolve(), args.artifacts.resolve()
    if base.relative_to(root).parts[:2] != ("docs", "validation") or art.relative_to(root).parts[0] != "artifacts":
        parser.error("Use ignored validation/artifact directories for isolated outputs")
    base.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)

    def output(argv):
        return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT).strip()

    quota = dict(available=False)
    if Path("/sys/fs/cgroup/cpu.max").exists():
        quota = dict(available=True, kind="v2", raw=Path("/sys/fs/cgroup/cpu.max").read_text().strip())
    elif Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").exists():
        quota = dict(available=True, kind="v1", quota_us=Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip(),
                     period_us=Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), commands=[],
                  compiler=output(["c++", "--version"]), dotnet=output(["dotnet", "--version"]), cpu_quota=quota,
                  ggml_revision=output(["git", "-C", str(root / "ExternalProjects/ggml"), "rev-parse", "HEAD"]),
                  ggml_changes=output(["git", "-C", str(root / "ExternalProjects/ggml"), "status", "--porcelain", "--untracked-files=no"]),
                  binary_artifacts=[], completed=False,
                  limitations="Build evidence only. Isolated owned CPU library/probe/Host outputs; existing GGML library and CUDA content are unused by the pure CPU text graph. No live process argv collected.")
    if report["ggml_changes"]:
        raise ValueError("Require unchanged upstream ggml")
    names = ["TensorSharp.Models/Embeddings/OwnedEmbeddingCpuAttention.cs", "TensorSharp.Models/Embeddings/ManagedEmbeddingGemma2Encoder.cs",
             "TensorSharp.Models/Embeddings/OwnedEmbeddingCpuProjection.cs", "TensorSharp.Models/Embeddings/ManagedEmbeddingQ8Q16Avx512Matrix.cs",
             "TensorSharp.Models/Embeddings/ManagedEmbeddingEncoder.cs",
             "TensorSharp.Models/Embeddings/ManagedEmbeddingGatedGelu.cs", "eng/validation/EmbeddingGemma2GatedGeluProbe/Program.cs",
             "eng/validation/EmbeddingGemma2GatedGeluProbe/EmbeddingGemma2GatedGeluProbe.csproj",
             "TensorSharp.Models/TensorSharp.Models.csproj", "TensorSharp.Cpu.Native/Packaging.targets", "TensorSharp.Cpu.Native/CMakeLists.txt",
             "TensorSharp.Cpu.Native/embedding_attention_avx512.cpp", "TensorSharp.Cpu.Native/embedding_q16_vnni.cpp",
             "eng/validation/EmbeddingGemma2Probe/Program.cs", "eng/validation/EmbeddingGemma2OwnedAttentionBridgeProbe/Program.cs",
             "eng/validation/EmbeddingGemma2OwnedProjectionBridgeProbe/Program.cs",
             "eng/validation/EmbeddingGemma2Q16Probe/Program.cs", "eng/validation/EmbeddingGemma2Q16Probe/EmbeddingGemma2Q16Probe.csproj",
             "eng/validation/embeddinggemma2-owned-attention-build.py", "eng/validation/embeddinggemma2-owned-attention-quality.py"]
    report["source_artifacts"] = [dict(path=name, sha256=sha(root / name)) for name in names]
    library = art / "native/libTensorSharpCpu.so"
    common = ["-c", "Release", "-p:TensorSharpSkipGgmlNative=true", "-p:TensorSharpSkipMlxNative=true",
              "-p:AppendTargetFrameworkToOutputPath=false", "-p:TensorSharpCpuNativeLibrary=" + str(library),
              "-p:TensorSharpCpuNativeRuntimeIdentifier=linux-x64"]
    commands = [("redact-historical", ["python3", str(root / "eng/validation/sanitize-historical-cpu-evidence.py"), "--workspace", str(root),
                 *[argument for relative in ("docs/validation/embeddinggemma2/q16-experiment", "artifacts/embeddinggemma2/q16-experiment",
                                              "artifacts/embeddinggemma2/cpu-q16-model-probe", "artifacts/embeddinggemma2/vnni-experiment")
                   for argument in ("--evidence-root", str(root / relative))]]),
                ("native-configure", ["cmake", "-S", str(root / "TensorSharp.Cpu.Native"), "-B", str(art / "native"), "-DCMAKE_BUILD_TYPE=Release"]),
                ("native-build", ["cmake", "--build", str(art / "native"), "-j", "2"]),
                ("bridge-build", ["dotnet", "build", str(root / "eng/validation/EmbeddingGemma2OwnedAttentionBridgeProbe/EmbeddingGemma2OwnedAttentionBridgeProbe.csproj"),
                                  "-c", "Release", "-p:OutputPath=" + str(art / "bridge") + "/", "-p:AppendTargetFrameworkToOutputPath=false"]),
                ("projection-bridge-build", ["dotnet", "build", str(root / "eng/validation/EmbeddingGemma2OwnedProjectionBridgeProbe/EmbeddingGemma2OwnedProjectionBridgeProbe.csproj"),
                                  "-c", "Release", "-p:OutputPath=" + str(art / "projection-bridge") + "/", "-p:AppendTargetFrameworkToOutputPath=false"]),
                ("q16-arithmetic-build", ["dotnet", "build", str(root / "eng/validation/EmbeddingGemma2Q16Probe/EmbeddingGemma2Q16Probe.csproj"),
                                  "-c", "Release", "-p:OutputPath=" + str(art / "q16-probe") + "/", "-p:AppendTargetFrameworkToOutputPath=false"]),
                ("gated-gelu-build", ["dotnet", "build", str(root / "eng/validation/EmbeddingGemma2GatedGeluProbe/EmbeddingGemma2GatedGeluProbe.csproj"),
                                  "-c", "Release", "-p:OutputPath=" + str(art / "gated-gelu-probe") + "/", "-p:AppendTargetFrameworkToOutputPath=false"]),
                ("probe-build", ["dotnet", "build", str(root / "eng/validation/EmbeddingGemma2Probe/EmbeddingGemma2Probe.csproj"), *common, "-p:OutputPath=" + str(art / "probe") + "/"]),
                ("host-build", ["dotnet", "build", str(root / "TensorSharp.Server.Host/TensorSharp.Server.Host.csproj"), *common, "-p:OutputPath=" + str(art / "host") + "/"])]
    try:
        for name, argv in commands:
            print("starting", name, flush=True)
            with (base / (name + ".log")).open("w") as log:
                code = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT).returncode
            report["commands"].append(dict(name=name, argv=argv, exit_code=code))
            print("finished", name, "exit", code, flush=True)
            if code:
                return 1
        report["native_dependencies"] = output(["ldd", str(library)])
        report["binary_artifacts"] = [dict(path=str(path), sha256=sha(path)) for directory in (art / "native", art / "bridge", art / "projection-bridge", art / "q16-probe", art / "gated-gelu-probe", art / "probe", art / "host")
                                      for path in directory.iterdir() if path.is_file() and (path.suffix == ".dll" or path.name in ("libTensorSharpCpu.so", "libGgmlOps.so"))]
        report["completed"] = True
        return 0
    finally:
        (base / "build-provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
