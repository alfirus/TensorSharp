#!/usr/bin/env python3
"""Generate the tiny LTX 2.5 x2 spatial latent upsampler fixture.

Builds the unchanged original ``LatentUpsampler`` (3D, spatial x2, no rational
resampler) with 16 latent and 64 mid channels and one block per stage, so each
GroupNorm(32) group pools two channels. Weights are seeded random values stored
as BF16; the oracle runs in Float32 on the BF16-rounded weights, as TensorSharp
widens them. A three-frame input exercises Conv3d zero padding at both temporal
edges. ``upsample_video`` is reproduced with explicit per-channel statistics.
Writes ``latent-upsampler-tiny.safetensors`` and ``latent-upsampler.json``.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"


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
    import torch
    from safetensors.torch import save_file
    from ltx_core.model.upsampler import LatentUpsampler

    torch.manual_seed(20261015)
    config = dict(in_channels=16, mid_channels=64, num_blocks_per_stage=1, dims=3, spatial_upsample=True,
                  temporal_upsample=False, spatial_scale=2.0, rational_resampler=False)
    model = LatentUpsampler(**config).eval()
    state = {}
    for name, value in model.state_dict().items():
        if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name == "initial_norm.weight":
            value = 1 + 0.2 * torch.randn_like(value)
        elif name.endswith(".bias"):
            value = 0.1 * torch.randn_like(value)
        else:
            fan_in = value[0].numel()
            value = torch.randn_like(value) / fan_in ** 0.5
        state[name] = value.to(torch.bfloat16)
    model.load_state_dict({name: value.float() for name, value in state.items()})
    channels = config["in_channels"]
    mean = (0.3 * torch.randn(channels)).to(torch.bfloat16).float()
    std = (0.5 + torch.rand(channels)).to(torch.bfloat16).float()
    latent = torch.randn(1, channels, 3, 3, 4)
    with torch.inference_mode():
        output = model(latent * std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1))
        output = (output - mean.view(1, -1, 1, 1, 1)) / std.view(1, -1, 1, 1, 1)
    args.output.mkdir(parents=True, exist_ok=True)
    weights = args.output / "latent-upsampler-tiny.safetensors"
    save_file(state, str(weights), metadata={"config": json.dumps({"_class_name": "LatentUpsampler", **config})})
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    fixture = {
        "source_revision": revision, "checkpoint_sha256": digest, "config": config,
        "input_shape": list(latent.shape[1:]), "input": latent.flatten().tolist(),
        "mean": mean.tolist(), "std": std.tolist(),
        "output_shape": list(output.shape[1:]), "output": output.flatten().tolist(),
    }
    (args.output / "latent-upsampler.json").write_text(json.dumps(fixture) + "\n")
    print(weights, digest, list(output.shape))


if __name__ == "__main__":
    main()
