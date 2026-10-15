#!/usr/bin/env python3
"""Run actual H3 oracle cases, rejecting missing fixtures, skips and empty runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

STAGES = {
    "text": ("MiniMaxH3VaeOracleTests.TextEncoderMatchesTheUpstreamReferenceForTheFirstLayers",
             ["h3_te_embeddings.bin", "h3_te_layer1_out.bin", "h3_te_token_ids.txt"]),
    "dit": ("MiniMaxH3VaeOracleTests.DitForwardMatchesTheUpstreamReferenceForTheFirstBlocks",
            ["h3_dit_video_latent.bin", "h3_dit_audio_latent.bin", "h3_dit_text_hidden.bin", "h3_dit_video_out.bin", "h3_dit_audio_out.bin"]),
    "video_decode": ("MiniMaxH3VaeOracleTests.VideoVaeDecodeMatchesTheUpstreamReference",
                     ["h3_video_latent_in.bin", "h3_video_pixels_out.bin"]),
    "video_encode": ("MiniMaxH3VaeOracleTests.VideoVaeEncodeMatchesTheUpstreamReference",
                     ["h3_video_encode_image.bin", "h3_video_encode_latent.bin"]),
    "audio_decode": ("MiniMaxH3VaeOracleTests.AudioVaeDecodeMatchesTheUpstreamReference",
                     ["h3_audio_latent_in.bin", "h3_audio_wave_out.bin"]),
    "audio_encode": ("MiniMaxH3VaeOracleTests.AudioVaeEncodeMatchesTheUpstreamReference",
                     ["h3_audio_enc_wave_in.bin", "h3_audio_enc_mean.bin"]),
    "vision": ("MiniMaxH3VisionOracleTests.VisionTowerMatchesTheUpstreamReference",
               ["h3_vision_pixels.bin", "h3_vision_merged.bin", "h3_vision_ds0.bin", "h3_vision_ds1.bin", "h3_vision_ds2.bin"]),
    "video_encode_3d": ("MiniMaxH3VaeOracleTests.VideoVaeMultiFrameEncodeMatchesTheUpstreamReference",
                        ["h3_video_enc3d_in_t5.bin", "h3_video_enc3d_mean_t5.bin", "h3_video_enc3d_in_t17.bin", "h3_video_enc3d_mean_t17.bin"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "cpu", "mlx", "ggml_cuda", "ggml_cpu", "ggml_metal"])
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", action="append", choices=STAGES)
    parser.add_argument("--primitives", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Enable per-node profiling; timings are diagnostic only")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--project", type=Path, default=Path("InferenceWeb.Tests/InferenceWeb.Tests.csproj"))
    parser.add_argument("--assembly",type=Path,help="Use a frozen built test assembly instead of the shared project output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = args.stage or list(STAGES)
    if args.primitives:
        if args.backend not in ("cuda", "mlx"):
            raise SystemExit("Primitive device suite requires CUDA or MLX")
        prefix = "Cuda" if args.backend == "cuda" else "Mlx"
        filters = ["FullyQualifiedName~MiniMaxH3TensorOpsTests." + prefix,
                   "FullyQualifiedName~MiniMaxH3VaeDeviceTests." + prefix]
        if args.backend == "cuda":
            filters.append("FullyQualifiedName~GgmlClampTests.CudaClamp")
        expected_count = 16 if args.backend == "cuda" else 10
    else:
        filters = ["FullyQualifiedName~" + STAGES[name][0] for name in selected]
        required = set(file for name in selected for file in STAGES[name][1])
        missing = [file for file in sorted(required) if not (args.models / "fixtures" / file).is_file()]
        if missing:
            raise SystemExit("Required oracle fixtures missing: " + ", ".join(missing))
        components = {
            "text": "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "vision": "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
            "dit": "minimax_h3_fl2va_pruned-Q4_K.gguf", "video_encode": "minimax_h3_video_vae_fp16.safetensors",
            "video_decode": "minimax_h3_video_vae_fp16.safetensors", "video_encode_3d": "minimax_h3_video_vae_fp16.safetensors",
            "audio_encode": "minimax_h3_audio_vae_fp32.safetensors", "audio_decode": "minimax_h3_audio_vae_fp32.safetensors"}
        missing = [components[name] for name in selected if not (args.models / components[name]).is_file()]
        if missing: raise SystemExit("Required model components missing: " + ", ".join(missing))
        expected_count = len(selected) + int("video_encode_3d" in selected)
    env = os.environ.copy()
    env.update(TS_MINIMAX_H3_DIR=str(args.models.resolve()), TS_TEST_H3_BACKEND=args.backend,
               TS_TEST_GGML_BACKEND={"ggml_cpu": "cpu", "cpu": "cpu", "mlx": "cpu", "ggml_metal": "metal"}.get(args.backend, "cuda"),
               TS_TEST_CUDA="1", TS_TEST_MLX="1")
    if args.backend in ("cpu","ggml_cpu"):
        env.update(TS_CPU_THREADS="8", DOTNET_PROCESSOR_COUNT="8", OMP_NUM_THREADS="8", GGML_N_THREADS="8")
    if args.profile:
        if not args.backend.startswith("ggml_"): raise SystemExit("Native per-node profiling requires a GGML backend")
        env.update(TS_GGML_NODE_PROFILE="1", TS_GGML_NODE_PROFILE_TAG="minimax-h3/", TS_GGML_NODE_PROFILE_EVERY="1")
    command = ["dotnet", "test", str(args.project), "-c", "Release", "--no-build", "--no-restore", "--filter", "|".join(filters),
               "--logger", "trx;LogFileName=results.trx", "--results-directory", str(args.output.resolve()),
               "--logger", "console;verbosity=detailed"]
    if args.assembly:
        if not args.assembly.is_file(): raise SystemExit("Frozen test assembly is missing")
        command=["dotnet","vstest",str(args.assembly.resolve()),"--TestCaseFilter:"+"|".join(filters),
                 "--Logger:trx;LogFileName=results.trx","--ResultsDirectory:"+str(args.output.resolve()),
                 "--Logger:console;verbosity=detailed"]
    started = time.time()
    timeout = False
    with (args.output / "run.log").open("w") as log:
        try:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                exit_code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
                exit_code, timeout = 124, True
        except OSError as error:
            log.write(str(error)); exit_code = 127
    records = []
    trx = args.output / "results.trx"
    if trx.exists():
        root = ET.parse(trx).getroot()
        for result in root.iter("{http://microsoft.com/schemas/VisualStudio/TeamTest/2010}UnitTestResult"):
            records.append({"test": result.attrib.get("testName"), "outcome": result.attrib.get("outcome"), "duration": result.attrib.get("duration")})
    passed = exit_code == 0 and len(records) == expected_count and all(row["outcome"] == "Passed" for row in records)
    report = {"backend": args.backend, "primitives": args.primitives, "profile": args.profile,
              "timing_scope": "diagnostic per-node synchronization" if args.profile else "oracle verification process", "stages": [] if args.primitives else selected,
              "argv": command, "started_at": started, "wall_seconds": time.time()-started, "timeout": timeout,
              "exit_code": exit_code, "expected_count": expected_count, "actual_count": len(records), "passed": passed, "results": records}
    if not args.primitives:
        report["fixtures"] = {file: hashlib.sha256((args.models/"fixtures"/file).read_bytes()).hexdigest() for file in sorted(required)}
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ["backend", "wall_seconds", "exit_code", "actual_count", "passed"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
