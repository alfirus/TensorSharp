#!/usr/bin/env python3
"""Compare TensorSharp's x2 spatial latent upsampler with the pinned original.

Runs the unchanged ``LatentUpsampler`` and ``upsample_video`` (un-normalize with
the video VAE's per-channel statistics, upsample, normalize) in Float32 on CPU,
on the same channel-major [128,T,H,W] latents the TensorSharp probe used, and
reports error metrics against the probe's output.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Unchanged Lightricks/LTX-2 checkout")
    parser.add_argument("--upsampler", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True, help="Float32 [128,T,H,W]; shape in <latents>.json")
    parser.add_argument("--probe", type=Path, required=True, help="TensorSharp probe output stem (<stem>.f32)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit("Unexpected official LTX revision: " + revision)
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "packages"], text=True).strip():
        raise SystemExit("Official LTX source must remain unchanged")
    if args.output.exists():
        raise SystemExit("Refusing to replace an existing report")
    sys.path.insert(0, str(source / "packages/ltx-core/src"))
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from ltx_core.model.upsampler import LatentUpsampler

    shape = json.loads(Path(str(args.latents) + ".json").read_text())["shape"]
    latent = torch.from_numpy(np.fromfile(args.latents, dtype=np.float32).reshape(shape)).unsqueeze(0)
    with safe_open(str(args.upsampler), "pt") as handle:
        config = json.loads(handle.metadata()["config"])
    config.pop("_class_name")
    model = LatentUpsampler(**config)
    model.load_state_dict({k: v.float() for k, v in load_file(str(args.upsampler)).items()})
    model.eval()
    with safe_open(str(args.vae), "pt") as handle:
        mean = handle.get_tensor("per_channel_statistics.mean-of-means").float().view(1, -1, 1, 1, 1)
        std = handle.get_tensor("per_channel_statistics.std-of-means").float().view(1, -1, 1, 1, 1)
    with torch.inference_mode():
        # upsample_video, with the encoder's statistics passed directly.
        reference = model(latent * std + mean)
        reference = ((reference - mean) / std)[0].numpy()
    observed = np.fromfile(str(args.probe) + ".f32", dtype=np.float32).reshape(reference.shape)
    diff = observed.astype(np.float64) - reference.astype(np.float64)
    report = {
        "revision": revision, "upsampler": str(args.upsampler.resolve()), "latents": str(args.latents.resolve()),
        "input_shape": shape, "output_shape": list(reference.shape),
        "max_abs": float(np.abs(diff).max()),
        "relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(reference.astype(np.float64))),
        "cosine": float((observed.astype(np.float64) * reference).sum()
                        / np.linalg.norm(observed.astype(np.float64)) / np.linalg.norm(reference.astype(np.float64))),
        "reference_rms": float(np.sqrt((reference.astype(np.float64) ** 2).mean())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
