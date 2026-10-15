#!/usr/bin/env python3
"""Decode TensorSharp LTX 2.5 video latents with the pinned publisher decoders.

Feeds channel-major [128,T,H,W] Float32 latents (the normalized denoiser output
TensorSharp hands to its video VAE) to the unchanged original ``VideoDecoder``
pipeline block. ``--vae`` selects the checkpoint: the default diffusion VAE
(``ltx-2.5-video-vae-bf16.safetensors``, the publisher's DistilledPipeline
decoder) or the convolutional one. Writes uint8 RGB frames [F,H,W,3] and an
MP4 so a decoder can be compared on identical latents. Without NATTEN or
Triton the original code selects its eager tiled-SDPA neighborhood attention.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Unchanged Lightricks/LTX-2 checkout")
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True, help="Float32 [128,T,H,W]; shape in <latents>.json")
    parser.add_argument("--output", type=Path, required=True, help="Output stem")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--tiled", action="store_true", help="Resolve the pipeline's AUTO_TILING instead of one tile")
    parser.add_argument("--ffmpeg", type=Path, help="Optional ffmpeg for an H.264 preview of the frames")
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit("Unexpected official LTX revision: " + revision)
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "packages"], text=True).strip():
        raise SystemExit("Official LTX source must remain unchanged")
    sys.path.insert(0, str(source / "packages/ltx-core/src"))
    sys.path.insert(0, str(source / "packages/ltx-pipelines/src"))
    import numpy as np
    import torch
    from ltx_core.model.video_vae import AUTO_TILING
    from ltx_core.types import VideoPixelShape
    from ltx_pipelines.utils.blocks import VideoDecoder
    from ltx_pipelines.utils.helpers import ensure_tiling_config, tiling_scale_factors_for_vae

    shape = json.loads(Path(str(args.latents) + ".json").read_text())["shape"]
    latents = np.fromfile(args.latents, dtype=np.float32).reshape(shape)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    latent = torch.from_numpy(latents).unsqueeze(0).to(device)
    decoder = VideoDecoder(str(args.vae.resolve()), dtype, device)
    tiling = None
    channels, frames, height, width = shape
    pixel_frames = (frames - 1) * 8 + 1
    if args.tiled:
        tiling = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=tiling_scale_factors_for_vae(str(args.vae)),
            vae_checkpoint_path=str(args.vae),
            video_shape=VideoPixelShape(batch=1, frames=pixel_frames, height=height * 32, width=width * 32, fps=args.fps),
            diffvae_optimization=decoder.diffvae_optimization,
            device=device,
            keyframes=False,
        )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    started = time.perf_counter()
    with torch.inference_mode():
        chunks = [chunk.float().cpu() for chunk in decoder(latent, tiling, generator)]
    seconds = time.perf_counter() - started
    video = torch.cat(chunks, dim=0)
    rgb = (video.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.output) + ".npy", rgb)
    if args.ffmpeg:
        subprocess.run([str(args.ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{rgb.shape[2]}x{rgb.shape[1]}", "-r", str(args.fps), "-i", "-",
                        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(args.output) + ".mp4"],
                       input=rgb.tobytes(), check=True)
    report = {
        "revision": revision, "vae": str(args.vae.resolve()), "latents": str(args.latents.resolve()), "shape": shape,
        "device": args.device, "dtype": args.dtype, "seed": args.seed, "tiling": repr(tiling),
        "frames": int(rgb.shape[0]), "height": int(rgb.shape[1]), "width": int(rgb.shape[2]), "decode_seconds": seconds,
    }
    Path(str(args.output) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
