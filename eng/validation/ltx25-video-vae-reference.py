#!/usr/bin/env python3
"""Execute unchanged original LTX convolutional decode with tiny or actual weights.

The tiny fixture uses synthetic BF16-stored weights widened to F32. Actual
checkpoint runs accept channel-major F32 latents and save complete intermediate
outputs. This is a decoder numerical gate, not a rendered/perceptual quality or
whole-pipeline performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--latents", type=Path)
    parser.add_argument("--shape", nargs=3, type=int, default=[2, 1, 2], metavar=("T", "H", "W"))
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise SystemExit("Reference requires the clean pinned original source")
    if min(args.shape) <= 0 or not 1 <= args.threads <= 32:
        raise SystemExit("Invalid input dimensions or thread count")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Refusing to replace an existing reference output directory")
    sys.path.insert(0, str(source / "packages/ltx-core/src"))
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from ltx_core.model.video_vae.model_configurator import _build_conv_video_decoder

    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        with safe_open(args.checkpoint, framework="pt") as handle:
            metadata = handle.metadata()
            config = json.loads(metadata["config"])["vae"]
            if config.get("_class_name") != "CausalVideoAutoencoder":
                raise SystemExit("The selected checkpoint is not the convolutional video VAE")
            with torch.device("meta"):
                model = _build_conv_video_decoder(config)
            state = {}
            for name in model.state_dict():
                key = name if name.startswith("per_channel_statistics.") else "decoder." + name
                state[name] = handle.get_tensor(key).to(getattr(torch, args.dtype))
            model.load_state_dict(state, strict=True, assign=True)
        checkpoint = args.checkpoint
    else:
        config = dict(_class_name="CausalVideoAutoencoder", dims=3, in_channels=3, out_channels=3,
                      latent_channels=2, decoder_base_channels=2, norm_layer="pixel_norm",
                      spatial_padding_mode="zeros", causal_decoder=False, timestep_conditioning=False,
                      use_quant_conv=False, patch_size=2,
                      decoder_blocks=[["res_x", {"num_layers": 1}], ["compress_space", {"multiplier": 2}],
                                      ["res_x", {"num_layers": 1}], ["compress_time", {"multiplier": 2}],
                                      ["res_x", {"num_layers": 1}], ["compress_all", {"multiplier": 1}],
                                      ["res_x", {"num_layers": 1}], ["compress_all", {"multiplier": 2}],
                                      ["res_x", {"num_layers": 1}]])
        model = _build_conv_video_decoder(config)
        state = {}
        for number, (name, value) in enumerate(model.state_dict().items()):
            flat = torch.arange(value.numel(), dtype=torch.float32)
            if name.endswith("std-of-means"):
                values = 0.75 + flat * .125
            elif name.endswith("mean-of-means"):
                values = flat * .125 - .0625
            elif name.endswith("bias"):
                values = torch.sin(flat * .37 + number) * .015
            else:
                values = torch.sin(flat * .173 + number * .31) * (.15 / (value.shape[1] * 27) ** .5)
            stored = values.reshape(value.shape).to(torch.bfloat16).contiguous()
            key = name if name.startswith("per_channel_statistics.") else "decoder." + name
            state[key] = stored
        checkpoint = args.output / "video-vae-tiny.safetensors"
        save_file(state, checkpoint, metadata={"config": json.dumps({"vae": config})})
        model.load_state_dict({name.removeprefix("decoder."): value.to(torch.float32) for name, value in state.items()}, strict=True)
        if args.dtype != "float32":
            model.to(getattr(torch, args.dtype))
    model.eval()
    shape = [1, config["latent_channels"], *args.shape]
    if args.latents:
        inputs = np.fromfile(args.latents, dtype="<f4")
        if inputs.size != int(np.prod(shape)) or not np.isfinite(inputs).all():
            raise SystemExit("Input latents must be finite and match the declared shape")
        inputs = torch.from_numpy(inputs.copy()).reshape(shape)
    else:
        flat = torch.arange(int(np.prod(shape)), dtype=torch.float32)
        inputs = (torch.sin(flat * .29) * .6 + torch.cos(flat * .13) * .15).reshape(shape)
    inputs.numpy().astype("<f4").tofile(args.output / "input.f32")
    stages = {}

    def save_stage(name, output):
        values = output.detach().float().cpu().numpy()[0]
        path = args.output / (name.replace(".", "_") + ".f32")
        values.astype("<f4").tofile(path)
        stages[name] = dict(shape=list(values.shape), file=path.name, sha256=sha256(path))
        if not args.checkpoint:
            stages[name]["values"] = values.reshape(-1).tolist()

    model.conv_in.register_forward_hook(lambda _m, _i, output: save_stage("conv_in", output))
    model.conv_out.register_forward_hook(lambda _m, _i, output: save_stage("conv_out", output))
    for index, block in enumerate(model.up_blocks):
        block.register_forward_hook(lambda _m, _i, output, n=index: save_stage(f"up_blocks.{n}", output))
        for layer, residual in enumerate(getattr(block, "res_blocks", [])):
            residual.register_forward_hook(lambda _m, _i, output, n=index, r=layer: save_stage(f"up_blocks.{n}.res_blocks.{r}", output))
    started = time.perf_counter()
    with torch.inference_mode():
        save_stage("denormalized", model.per_channel_statistics.un_normalize(inputs.to(getattr(torch, args.dtype))))
        output = model(inputs.to(getattr(torch, args.dtype)))
        save_stage("decoded", output)
    report = dict(completed=True, source_revision=revision, torch_version=torch.__version__, dtype=args.dtype,
                  dependency_versions={name: version(name) for name in ["numpy", "safetensors", "einops", "av", "colour-science"]},
                  checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  input_shape=shape[1:], input_sha256=sha256(args.output / "input.f32"),
                  elapsed_seconds=time.perf_counter() - started, stages=stages,
                  scope="Original convolutional decoder component only; actual weights" if args.checkpoint else
                        "Original convolutional decoder synthetic BF16-stored weights widened to Float32; no actual trained-weight coverage")
    if not args.checkpoint:
        report["input"] = inputs.reshape(-1).tolist()
    report["source_sha256"] = {name: sha256(source / "packages/ltx-core/src/ltx_core/model" / name)
        for name in ["video_vae/model_configurator.py", "video_vae/conv_video_decoder.py", "video_vae/convolution.py", "video_vae/sampling.py",
                     "video_vae/resnet.py", "video_vae/ops.py", "common/normalization.py"]}
    (args.output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["completed", "dtype", "input_shape", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
