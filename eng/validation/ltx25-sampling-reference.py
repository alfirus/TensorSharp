#!/usr/bin/env python3
"""Generate a tiny Float32 AV Euler fixture with the pinned original LTX code.

This executes the original X0 conversion and Euler step, with deterministic
synthetic velocities. It does not execute weights, a text encoder or a VAE.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != "be6d6d0a69d11a0a399c462ed92606692e5f8697":
        raise SystemExit("Unexpected official LTX revision: " + revision)
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise SystemExit("Official LTX source must remain unchanged")
    sys.path.insert(0, str(source / "packages/ltx-core/src"))
    import torch
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.utils import to_denoised

    video = torch.tensor([0.5, -1.25, 2.0, 0.125, -0.75, 1.5], dtype=torch.float32)
    audio = torch.tensor([-0.25, 0.75, -1.5, 3.25], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.725, 0.421875, 0.0], dtype=torch.float32)
    fixture = dict(
        source_revision=revision,
        torch_version=torch.__version__,
        dtype="float32",
        source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in [
            "packages/ltx-core/src/ltx_core/utils.py",
            "packages/ltx-core/src/ltx_core/components/diffusion_steps.py"]},
        initial_video=video.tolist(), initial_audio=audio.tolist(), sigmas=sigmas.tolist(), steps=[],
        limitations="Synthetic Float32 velocity/X0/Euler arithmetic only. No model weights, noise RNG, conditioning, text encoding, VAE, or video generation executed.",
    )
    stepper = EulerDiffusionStep()
    for step in range(len(sigmas) - 1):
        # Opposite-stream means make sequentially updating one stream before
        # predicting the other observable in the multi-step fixture.
        video_velocity = video * 0.25 + audio.mean() * 0.125 + (step + 1) * 0.0625
        audio_velocity = audio * -0.375 + video.mean() * 0.1875 - (step + 1) * 0.03125
        denoised_video = to_denoised(video, video_velocity, sigmas[step])
        denoised_audio = to_denoised(audio, audio_velocity, sigmas[step])
        video = stepper.step(video, denoised_video, sigmas, step)
        audio = stepper.step(audio, denoised_audio, sigmas, step)
        fixture["steps"].append(dict(video_velocity=video_velocity.tolist(), audio_velocity=audio_velocity.tolist(),
                                     video=video.tolist(), audio=audio.tolist()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
