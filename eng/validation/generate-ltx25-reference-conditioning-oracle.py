#!/usr/bin/env python3
"""Generate IC-LoRA reference-video and masked-Euler fixtures with an unchanged official LTX checkout."""
import argparse
import ast
import hashlib
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
    from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
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
    tools = VideoLatentTools(VideoLatentPatchifier(1), VideoLatentShape(1, 128, 16, 1, 2), 24)
    state = tools.create_initial_state("cpu", torch.float32)
    reference = torch.linspace(-.7, .9, 128 * 16 * 1 * 2).reshape(1, 128, 16, 1, 2)
    state = VideoConditionByReferenceLatent(reference, downscale_factor=1, strength=1.0).apply_to(state, tools)
    noise = torch.linspace(-1.3, 1.7, state.latent.numel()).reshape_as(state.latent)

    class FixedNoiser(GaussianNoiser):
        def _sample_noise(self, latent_state):
            return noise

    state = FixedNoiser(torch.Generator())(state)
    flat = lambda tensor: tensor.detach().flatten().tolist()
    report = {"source_revision": REVISION, "reference_shape": [128, 16, 1, 2], "reference_latent": flat(reference), "noise": flat(noise),
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
    report["source_sha256"] = {name: hashlib.sha256((args.reference / "packages/ltx-core/src/ltx_core" / name).read_bytes()).hexdigest()
        for name in ["conditioning/types/reference_video_cond.py", "components/patchifiers.py", "tools.py", "components/noisers.py"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
