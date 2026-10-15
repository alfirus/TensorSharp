#!/usr/bin/env python3
"""Generate image-guide and masked-Euler fixtures with an unchanged official LTX checkout."""
import argparse
import ast
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip() == REVISION
    assert not subprocess.check_output(["git", "-C", str(args.reference), "status", "--porcelain"], text=True).strip()
    sys.path.insert(0, str(args.reference.resolve() / "packages/ltx-core/src"))
    import torch
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
    from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape
    from ltx_core.utils import to_denoised
    # Load the two pure helper functions from their original AST, avoiding
    # unrelated pipeline/media optional dependencies. No source is modified.
    path = args.reference / "packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                node.name in ("post_process_latent", "timesteps_from_mask")]
    assert len(selected) == 2
    helpers = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), helpers)
    torch.set_num_threads(1)
    tools = VideoLatentTools(VideoLatentPatchifier(1), VideoLatentShape(1, 128, 3, 1, 1), 24)
    state = tools.create_initial_state("cpu", torch.float32)
    images = []
    for index, strength in ((0, 1.0), (7, .35), (16, .8)):
        latent = torch.linspace(-.7, .9, 128).reshape(1, 128, 1, 1, 1) + index * .03
        images.append({"frame_index": index, "strength": strength, "latent": latent.flatten().tolist()})
        conditioning = (VideoConditionByLatentIndex(latent, strength, 0) if index == 0 else
                        VideoConditionByKeyframeIndex(latent, index, strength))
        state = conditioning.apply_to(state, tools)
    noise = torch.linspace(-1.3, 1.7, state.latent.numel()).reshape_as(state.latent)

    class FixedNoiser(GaussianNoiser):
        def _sample_noise(self, latent_state):
            return noise

    state = FixedNoiser(torch.Generator())(state)
    flat = lambda tensor: tensor.detach().flatten().tolist()
    report = {"source_revision": REVISION, "images": images, "noise": flat(noise),
              "clean": flat(state.clean_latent), "mask": flat(state.denoise_mask),
              "keyframe_mask": flat(state.keyframes_mask),
              "positions": flat(state.positions.transpose(1, 2)), "initial": flat(state.latent), "steps": []}
    sigmas = torch.tensor([1, .4, 0], dtype=torch.float32)
    report["sigmas"] = flat(sigmas)
    for step in range(len(sigmas) - 1):
        timesteps = helpers["timesteps_from_mask"](state.denoise_mask, sigmas[step])
        velocity = torch.sin(state.latent * .1 + step * .25) * .2
        denoised = to_denoised(state.latent, velocity, timesteps)
        denoised = helpers["post_process_latent"](denoised, state.denoise_mask, state.clean_latent)
        state = replace(state, latent=EulerDiffusionStep().step(state.latent, denoised, sigmas, step))
        report["steps"].append({"timesteps": flat(timesteps), "velocity": flat(velocity), "output": flat(state.latent)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
