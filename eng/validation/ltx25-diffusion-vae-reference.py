#!/usr/bin/env python3
"""Decode LTX 2.5 latents with the pinned original diffusion video decoder.

Builds the decoder exactly as the publisher's ``VideoDecoder`` pipeline block
does (checkpoint loader, SD ops, default DiffVAE mode), then runs its untiled
``_decode_pixels`` in the requested dtype on Float32 [128,T,H,W] latents. The
pixel noise that decode draws from ``--seed`` is re-drawn with an identically
seeded generator and saved (``<output>.noise.f32`` + ``.json``), so
``Ltx25DiffusionVaeProbe --noise`` can decode the same canvas. With ``--probe``
the probe's raw output is compared with the original's.
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
    parser.add_argument("--vae", type=Path, required=True, help="ltx-2.5-video-vae-bf16.safetensors")
    parser.add_argument("--latents", type=Path, required=True, help="Float32 [128,T,H,W]; shape in <latents>.json")
    parser.add_argument("--output", type=Path, required=True, help="Output stem")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe", type=Path, help="Ltx25DiffusionVaeProbe output stem decoded with this noise")
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
    from ltx_core.model.video_vae import diffusion_tiling
    from ltx_pipelines.utils.blocks import VideoDecoder

    shape = json.loads(Path(str(args.latents) + ".json").read_text())["shape"]
    latent = torch.from_numpy(np.fromfile(args.latents, dtype=np.float32).reshape(shape)).unsqueeze(0)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    block = VideoDecoder(str(args.vae.resolve()), dtype, device)
    decoder = block._prepared_builder().build(device=device, dtype=dtype).eval()
    latent = latent.to(device=device, dtype=dtype)
    with torch.inference_mode():
        # The noise an untiled single-step x0 decode draws for its one tile.
        padded, _ = diffusion_tiling.ensure_min_latent_shape(latent, decoder.stage_min_tile_sizes)
        strides = [tuple(u.stride) for u in decoder.upsamples]
        s4 = diffusion_tiling.stage4_thw_from_latent(strides, *padded.shape[2:], drop_leading_frame=True)
        canvas = diffusion_tiling.stage5_pixel_shape_from_stage4(
            *s4, upsample_stride=strides[3], patch_size=decoder.patch_size,
            stage5_kernel_t=decoder.stage5_kernel[0], drop_leading_frame=True, pad_trailing=True)
        noise = torch.randn((1, 3, *canvas), dtype=dtype, generator=torch.Generator("cpu").manual_seed(args.seed))
        started = time.perf_counter()
        output = next(decoder._decode_pixels(latent, None, torch.Generator("cpu").manual_seed(args.seed)))
        seconds = time.perf_counter() - started
    raw = output[0].float().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw.astype("<f4").tofile(str(args.output) + ".f32")
    noise[0].float().numpy().astype("<f4").tofile(str(args.output) + ".noise.f32")
    Path(str(args.output) + ".noise.f32.json").write_text(json.dumps({"shape": list(noise.shape[1:])}) + "\n")
    report = {
        "revision": revision, "vae": str(args.vae.resolve()), "latents": str(args.latents.resolve()), "shape": shape,
        "device": args.device, "dtype": args.dtype, "seed": args.seed, "canvas": list(canvas),
        "output_shape": list(raw.shape), "decode_seconds": seconds,
    }
    if args.probe:
        observed = np.fromfile(str(args.probe) + ".f32", dtype=np.float32).reshape(raw.shape).astype(np.float64)
        expected = raw.astype(np.float64)
        diff = observed - expected
        report["comparison"] = {
            "relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(expected)),
            "max_abs": float(np.abs(diff).max()),
            "rgb_max_abs_after_clamp": float(np.abs(np.clip((observed + 1) / 2, 0, 1) - np.clip((expected + 1) / 2, 0, 1)).max()),
        }
    Path(str(args.output) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
