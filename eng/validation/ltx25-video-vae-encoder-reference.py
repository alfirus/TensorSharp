#!/usr/bin/env python3
"""Compare LTX video encoding against an unchanged pinned official checkout.

Synthetic BF16 fixtures exercise every encoder stage. Actual checkpoint runs
save stage tensors and timings, without claiming end-to-end/perceptual coverage.
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
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pixels", type=Path)
    parser.add_argument("--shape", nargs=3, type=int, default=[1, 16, 32], metavar=("T", "H", "W"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--static-frame", action="store_true", help="Repeat one RGB frame before executing the real temporal encoder; compact synthetic output retains terminal values")
    parser.add_argument("--terminal-only", action="store_true", help="Save only normalized output, avoiding large intermediate snapshots; does not reduce the official encoder's working memory")
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise SystemExit("Reference requires the clean pinned original LTX checkout")
    if min(args.shape) <= 0 or not 1 <= args.threads <= 32:
        raise SystemExit("Invalid shape or thread count")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Refusing to replace existing reference output")
    sys.path.insert(0, str(source / "packages/ltx-core/src"))
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from ltx_core.model.video_vae.model_configurator import VideoEncoderConfigurator

    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        with safe_open(args.checkpoint, framework="pt") as handle:
            config = json.loads(handle.metadata()["config"])
            with torch.device("meta"):
                model = VideoEncoderConfigurator.from_metadata({"config": config})
            state = {}
            for name in model.state_dict():
                key = name if name.startswith("per_channel_statistics.") else "encoder." + name
                state[name] = handle.get_tensor(key).float()
            model.load_state_dict(state, strict=True, assign=True)
        checkpoint = args.checkpoint
    else:
        config = {"vae": dict(_class_name="CausalVideoAutoencoder", dims=3, in_channels=3, out_channels=3,
            latent_channels=2, norm_layer="pixel_norm", spatial_padding_mode="zeros", latent_log_var="uniform", patch_size=2,
            encoder_blocks=[["res_x", {"num_layers": 1}], ["compress_space_res", {"multiplier": 2}],
                ["res_x", {"num_layers": 1}], ["compress_time_res", {"multiplier": 2}],
                ["res_x", {"num_layers": 1}], ["compress_all_res", {"multiplier": 2}],
                ["res_x", {"num_layers": 1}], ["compress_all_res", {"multiplier": 1}], ["res_x", {"num_layers": 1}]])}
        model = VideoEncoderConfigurator.from_metadata({"config": config})
        state = {}
        for number, (name, value) in enumerate(model.state_dict().items()):
            flat = torch.arange(value.numel(), dtype=torch.float32)
            if name.endswith("std-of-means"):
                values = .75 + flat * .125
            elif name.endswith("mean-of-means"):
                values = flat * .125 - .0625
            elif name.endswith("bias"):
                values = torch.sin(flat * .37 + number) * .015
            else:
                values = torch.sin(flat * .173 + number * .31) * (.15 / (value.shape[1] * 27) ** .5)
            stored = values.reshape(value.shape).bfloat16().contiguous()
            key = name if name.startswith("per_channel_statistics.") else "encoder." + name
            state[key] = stored
        checkpoint = args.output / "video-vae-encoder-tiny.safetensors"
        save_file(state, checkpoint, metadata={"config": json.dumps(config)})
        model.load_state_dict({name.removeprefix("encoder."): value.float() for name, value in state.items()}, strict=True)
    model.eval()
    t, h, w = args.shape
    if args.pixels:
        supplied = np.fromfile(args.pixels, dtype="<f4")
        input_frames = 1 if args.static_frame and supplied.size == h * w * 3 else t
        rgb = supplied.reshape(input_frames, h, w, 3)
    else:
        flat = torch.arange(t * h * w * 3, dtype=torch.float32)
        rgb = (.5 + torch.sin(flat * .029) * .3 + torch.cos(flat * .013) * .15).numpy().reshape(t, h, w, 3)
    if args.static_frame:
        rgb = np.broadcast_to(rgb[:1], (t, h, w, 3)).copy()
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise SystemExit("Pixels must be finite RGB in [0,1]")
    rgb.astype("<f4").tofile(args.output / "input.f32")
    inputs = torch.from_numpy(rgb.copy()).permute(3, 0, 1, 2).unsqueeze(0) * 2 - 1
    stages = {}

    def save_stage(name, output):
        if args.terminal_only and name != "encoded":
            return
        values = output.detach().float().cpu().numpy()[0]
        path = args.output / (name.replace(".", "_") + ".f32")
        values.astype("<f4").tofile(path)
        stages[name] = dict(shape=list(values.shape), file=path.name, sha256=sha256(path))
        if not args.checkpoint and (not args.static_frame or name == "encoded"):
            stages[name]["values"] = values.reshape(-1).tolist()

    model.conv_in.register_forward_pre_hook(lambda _m, inputs: save_stage("patchified", inputs[0]))
    model.conv_in.register_forward_hook(lambda _m, _i, output: save_stage("conv_in", output))
    model.conv_out.register_forward_hook(lambda _m, _i, output: save_stage("conv_out", output))
    for index, block in enumerate(model.down_blocks):
        block.register_forward_hook(lambda _m, _i, output, n=index: save_stage(f"down_blocks.{n}", output))
        for layer, residual in enumerate(getattr(block, "res_blocks", [])):
            residual.register_forward_hook(lambda _m, _i, output, n=index, r=layer: save_stage(f"down_blocks.{n}.res_blocks.{r}", output))
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(inputs)
        save_stage("encoded", output)
    report = dict(completed=True, source_revision=revision, torch_version=torch.__version__, dtype="float32",
        dependency_versions={name: version(name) for name in ["numpy", "safetensors", "einops", "av", "colour-science"]},
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), input_shape=[t, h, w, 3],
        input_sha256=sha256(args.output / "input.f32"), elapsed_seconds=time.perf_counter() - started, stages=stages,
        scope="Official causal encoder component only; no perceptual or end-to-end coverage")
    if not args.checkpoint:
        report["static_frame_rgb" if args.static_frame else "input"] = (rgb[0] if args.static_frame else rgb).reshape(-1).tolist()
    report["static_rgb_video"] = args.static_frame
    report["terminal_only"] = args.terminal_only
    report["source_sha256"] = {name: sha256(source / "packages/ltx-core/src/ltx_core/model" / name) for name in
        ["video_vae/video_vae.py", "video_vae/model_configurator.py", "video_vae/convolution.py", "video_vae/sampling.py",
         "video_vae/resnet.py", "video_vae/ops.py", "common/normalization.py"]}
    (args.output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["completed", "input_shape", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
