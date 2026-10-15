#!/usr/bin/env python3
"""Generate the tiny LTX 2.5 diffusion video decoder fixture.

Builds the unchanged original ``DiffusionVideoDecoder`` through the original
configurator from a tiny ``CausalDiffusionVAE`` config: 8 latent channels,
head_dim 16, stage channels 64/32/16/16/16, the production upsample ladder and
patch size (the original crops with the production 8x32x32 scale factors), one
block per deterministic stage with 3x1x3 neighborhoods, and two diffusion blocks
with a 3x5x5 neighborhood. Weights are seeded random values stored as BF16 and
loaded back exactly. The 2x1x2 latent is below the decoder's 3x1x3 floor, so the
original edge padding and content crop run. Pixel noise is drawn exactly as the
untiled ``_decode_pixels`` draws it and saved, so a port can consume it.

Without NATTEN the original installs its eager tiled-SDPA neighborhood attention.
The script asserts that its step-by-step trace reproduces the original
``_decode_pixels`` output bit for bit before writing the fixture.
Writes ``diffusion-vae-tiny.safetensors``, ``diffusion-vae.json`` and
``diffusion-vae.bin`` (little-endian Float32 arrays at the recorded offsets).
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"

CONFIG = {
    "_class_name": "CausalDiffusionVAE",
    "model_output_type": "x0",
    "decoder": {
        "_class_name": "NADiffusionDecoder",
        "in_channels": 8,
        "out_channels": 3,
        "patch_size": 4,
        "head_dim": 16,
        "stage_channels": [64, 32, 16, 16, 16],
        "stage_depths": [1, 1, 1, 1, 2],
        "stage_kernels": [[3, 1, 3], [3, 1, 3], [3, 1, 3], [3, 1, 3], [3, 5, 5]],
        "upsamples": [[[1, 2, 2], 2], [[2, 1, 1], 2], [[2, 2, 2], 1], [[2, 2, 2], 1]],
        "stage5_kernel": [3, 5, 5],
        "timestep_scale_multiplier": 1000.0,
        "default_num_inference_steps": 1,
    },
}


def checkpoint_name(name):
    """Module parameter name -> published checkpoint key (fused QKV, mlp-indexed t_embedder)."""
    name = name.replace("t_embedder.timestep_embedder.linear_1.", "t_embedder.mlp.0.")
    name = name.replace("t_embedder.timestep_embedder.linear_2.", "t_embedder.mlp.2.")
    return name if name.startswith("per_channel_statistics.") else "decoder." + name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Unchanged Lightricks/LTX-2 checkout")
    parser.add_argument("--output", type=Path, required=True, help="Fixture directory")
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit("Unexpected official LTX revision: " + revision)
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "packages"], text=True).strip():
        raise SystemExit("Official LTX source must remain unchanged")
    sys.path.insert(0, str(source / "packages/ltx-core/src"))
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from ltx_core.model.video_vae import diffusion_tiling
    from ltx_core.model.video_vae.model_configurator import _build_diffusion_video_decoder
    from ltx_core.model.video_vae.transformer import apply_diffvae_mode, DiffVAEMode

    torch.manual_seed(20261015)
    decoder = _build_diffusion_video_decoder(CONFIG)
    state = {}
    for name, value in decoder.state_dict().items():
        if name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight") \
                or name.endswith("norm_out.weight"):
            value = 1 + 0.2 * torch.randn_like(value)
        elif name == "per_channel_statistics.std-of-means":
            value = 0.5 + torch.rand_like(value)
        elif name.endswith(".bias") or name.endswith("scale_shift_table") or name == "type_emb" \
                or name == "per_channel_statistics.mean-of-means":
            value = 0.1 * torch.randn_like(value)
        else:
            fan_in = value.shape[-1] if value.dim() > 1 else 1
            value = torch.randn_like(value) / fan_in ** 0.5
        state[name] = value.to(torch.bfloat16)
    decoder.load_state_dict({name: value.float() for name, value in state.items()})
    decoder = apply_diffvae_mode(decoder, DiffVAEMode.CHUNKED_EAGER).eval()

    channels = CONFIG["decoder"]["in_channels"]
    latent = torch.randn(1, channels, 2, 1, 2)
    seed = 4242
    with torch.inference_mode():
        reference = next(decoder._decode_pixels(latent, None, torch.Generator("cpu").manual_seed(seed)))
        # Trace the same untiled decode: floor padding, ghost pad, stages 1-3,
        # stage 4 (+ crop), then the one-step x0 prediction from saved noise.
        padded, (t_pad, h_pad, w_pad) = diffusion_tiling.ensure_min_latent_shape(latent, decoder.stage_min_tile_sizes)
        ghost = diffusion_tiling.pad_trailing_latent_for_natten_border(padded, decoder._natten_trailing_pad_latent_frames)
        feat = decoder.forward_stages_1_to_3(ghost, drop_leading_frame=True)
        # Deferred (chunked) mode: stage 4 stops before upsamples[3], cropped to content.
        context = decoder.forward_stage_4(feat, drop_leading_frame=True, pad_trailing=True)
        strides = [tuple(u.stride) for u in decoder.upsamples]
        s4_t, s4_h, s4_w = diffusion_tiling.stage4_thw_from_latent(
            strides, padded.shape[2], padded.shape[3], padded.shape[4], drop_leading_frame=True)
        frames, height, width = diffusion_tiling.stage5_pixel_shape_from_stage4(
            s4_t, s4_h, s4_w, upsample_stride=strides[3], patch_size=decoder.patch_size,
            stage5_kernel_t=decoder.stage5_kernel[0], drop_leading_frame=True, pad_trailing=True)
        noise = torch.randn((1, 3, frames, height, width), dtype=torch.float32,
                            generator=torch.Generator("cpu").manual_seed(seed))
        timestep = decoder.default_inference_timesteps.unsqueeze(0)
        prediction = decoder._decode_one_tile(feat, noise, is_origin=True, timestep=timestep, pad_trailing=True)
        content_frames = (latent.shape[2] - 1) * 8 + 1
        traced = diffusion_tiling.crop_pixels_to_content(
            prediction, content_frames, latent.shape[3] * 32, latent.shape[4] * 32,
            h_pad=h_pad, w_pad=w_pad, spatial_scale=(32, 32))
    if not torch.equal(traced, reference):
        raise SystemExit("Step-by-step trace differs from the original _decode_pixels output")

    args.output.mkdir(parents=True, exist_ok=True)
    weights = args.output / "diffusion-vae-tiny.safetensors"
    tensors = {}
    for name, value in state.items():
        key = checkpoint_name(name)
        if ".attn.qkv.to_" in name:
            fused = key.replace(".to_q.", ".").replace(".to_k.", ".").replace(".to_v.", ".")
            tensors.setdefault(fused, {})[name.split(".attn.qkv.")[1].split(".")[0]] = value
        else:
            tensors[key] = value.contiguous()
    for key in [k for k, v in tensors.items() if isinstance(v, dict)]:
        parts = tensors.pop(key)
        tensors[key] = torch.cat([parts["to_q"], parts["to_k"], parts["to_v"]], 0).contiguous()
    save_file(tensors, str(weights), metadata={"config": json.dumps({"vae": CONFIG}), "model_version": "2.5.0"})
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()

    arrays = {
        "latent": latent[0], "noise": noise[0], "feat_stage4_input": feat[0], "context": context[0],
        "prediction": prediction[0], "output": reference[0],
    }
    blob, entries, offset = bytearray(), {}, 0
    for name, value in arrays.items():
        data = value.contiguous().numpy().astype("<f4").tobytes()
        entries[name] = {"offset": offset, "shape": list(value.shape)}
        blob += data
        offset += len(data)
    (args.output / "diffusion-vae.bin").write_bytes(bytes(blob))
    fixture = {
        "source_revision": revision, "checkpoint_sha256": digest, "seed": seed,
        "bin_sha256": hashlib.sha256(bytes(blob)).hexdigest(), "arrays": entries,
        "layouts": {
            "latent": "[C,T,H,W] normalized", "noise": "[3,F,H,W] stage-5 canvas pixels",
            "feat_stage4_input": "[T,H,W,C] channels-last, with ghost frames",
            "context": "[T,H,W,C] stage-4 output before upsamples[3], after the ghost crop",
            "prediction": "[3,F,H,W] x0 before the content crop", "output": "[3,F,H,W] in [-1,1]",
        },
    }
    (args.output / "diffusion-vae.json").write_text(json.dumps(fixture, indent=1) + "\n")
    print(weights, digest, {k: list(v.shape) for k, v in arrays.items()})


if __name__ == "__main__":
    main()
