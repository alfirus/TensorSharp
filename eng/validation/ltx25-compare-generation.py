#!/usr/bin/env python3
"""Compare completed, matched native/unchanged-SD.cpp AV generation reports.

Pixel/audio differences are descriptive, not a perceptual-quality pass criterion.
The two implementations have different BF16 reductions and RoPE arithmetic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(data):
    return {"shape": list(data.shape), "finite": bool(np.isfinite(data).all()),
            "mean": float(data.mean()), "std": float(data.std()),
            "min": float(data.min()), "max": float(data.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    native, baseline = [json.loads(path.read_text()) for path in (args.native, args.baseline)]
    if not native.get("completed") or not baseline.get("completed"):
        raise ValueError("Both generation reports must be completed.")
    if native["prompt"] != baseline["prompt"]:
        raise ValueError("Prompts differ.")
    native_refs, baseline_refs = native.get("reference_inputs", []), baseline.get("reference_inputs", [])
    def reference_identity(rows):
        identities = []
        for row in rows:
            path = Path(row["path"])
            digest = sha(path)
            if row.get("sha256", digest) != digest:
                raise ValueError("Reference image changed since generation: " + str(path))
            identities.append((digest, row["frame_index"], float(row["strength"])))
        return sorted(identities)
    if reference_identity(native_refs) != reference_identity(baseline_refs):
        raise ValueError("Reference images, target positions or strengths differ.")
    shape = native["settings"]
    for key in ("width", "height", "frames", "fps", "seed"):
        if shape[key.title()] != baseline["settings"][key]:
            raise ValueError("Generation setting differs: " + key)
    if native["sigmas"] != baseline["settings"]["sigmas"]:
        # .NET/System.Text.Json and C++ serialize F32 to different decimal lengths.
        if not np.array_equal(np.asarray(native["sigmas"], dtype=np.float32),
                              np.asarray(baseline["settings"]["sigmas"], dtype=np.float32)):
            raise ValueError("Sigma schedules differ.")
    frames, height, width = [shape[key] for key in ("Frames", "Height", "Width")]
    report = {"scope": "Actual full-pipeline matched AV generation. Timings exclude codecs/file writes; factory timing separate.",
              "limitations": "One prompt/seed/geometry is not a broad perceptual-quality or performance claim. First runs include loading and JIT caches; repeated prompts use each implementation's conditioning cache. CPU/file caches may already be warm. Pixel differences are descriptive, not a quality ranking.",
              "numpy_version": np.__version__, "prompt": native["prompt"], "settings": shape,
              "reference_inputs": native_refs,
              "reports": {str(path): sha(path) for path in (args.native, args.baseline)},
              "media": [], "timing": {}}
    for engine, data, path in (("native", native, args.native), ("baseline", baseline, args.baseline)):
        times = [item["generation_ms"] for item in data["executions"] if item.get("kind", "benchmark") == "benchmark"]
        report["timing"][engine] = {"factory_ms": data["model_factory_ms"], "first_ms": times[0],
                                     "factory_plus_first_ms": data["model_factory_ms"] + times[0],
                                     "repeat_ms": times[1:], "repeat_median_ms": statistics.median(times[1:]) if len(times) > 1 else None}
        for execution in data["executions"]:
            repeat = execution["repeat"]
            stem = str(path) + "." + str(repeat)
            rgb_path = Path(stem + (".rgb.f32" if engine == "native" else ".rgb.u8"))
            rgb = np.fromfile(rgb_path, dtype=np.float32 if engine == "native" else np.uint8).reshape(frames, height, width, 3)
            if engine == "baseline":
                rgb = rgb.astype(np.float32) / 255
            if engine == "native":
                audio_paths = [Path(stem + f".audio-{channel}.f32") for channel in range(2)]
                audio = np.stack([np.fromfile(file, dtype=np.float32) for file in audio_paths], axis=1)
            else:
                audio_paths = [Path(stem + ".audio.f32")]
                audio = np.fromfile(audio_paths[0], dtype=np.float32).reshape(-1, 2)
            metrics = {"engine": engine, "repeat": repeat, "rgb": measure(rgb), "audio": measure(audio),
                       "audio_seconds": len(audio) / 48000, "mean_adjacent_frame_change": float(np.abs(np.diff(rgb, axis=0)).mean()),
                       "hashes": {str(file): sha(file) for file in [rgb_path, *audio_paths]}}
            if not metrics["rgb"]["finite"] or not metrics["audio"]["finite"] or metrics["rgb"]["std"] < 1e-4:
                raise ValueError("Nonfinite or degenerate output.")
            report["media"].append(metrics)
            if repeat == 0:
                if engine == "native":
                    native_rgb, native_audio = rgb, audio
                else:
                    rgb_delta = native_rgb - rgb
                    report["difference"] = {"rgb_mae": float(np.abs(rgb_delta).mean()),
                        "rgb_rmse": float(np.sqrt(np.mean(rgb_delta ** 2))),
                        "audio_same_shape": native_audio.shape == audio.shape}
                    if native_audio.shape == audio.shape:
                        report["difference"]["audio_relative_l2"] = float(np.linalg.norm(native_audio - audio) / max(np.linalg.norm(audio), 1e-30))
    for key in ("first_ms", "factory_plus_first_ms", "repeat_median_ms"):
        a, b = report["timing"]["native"][key], report["timing"]["baseline"][key]
        if a is not None and b is not None:
            report["timing"][key + "_native_speedup"] = b / a
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["timing"], indent=2))


if __name__ == "__main__":
    main()
