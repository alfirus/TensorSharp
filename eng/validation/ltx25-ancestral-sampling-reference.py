#!/usr/bin/env python3
"""Generate a tiny Float32 AV ancestral-Euler fixture with the pinned original LTX code.

The published DistilledPipeline samples LTX 2.5+ checkpoints with
euler_ancestral_denoising_loop (eta=1, s_noise=1). This runs that unchanged loop
over the eight distilled sigmas with a deterministic synthetic denoiser and a
recorded noise sequence (new_noise_fn override), so the X0 conversion, both
post_process_latent points, the renoise and the terminal step are covered. Video
tokens use clean (mask 0), partial-strength (mask 0.2) and generated (mask 1)
conditioning. model_dtype is Float32, which is what TensorSharp keeps between
steps; the publisher's pipeline stores BF16. No weights, text encoder or VAE run.
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit("Unexpected official LTX revision: " + revision)
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise SystemExit("Official LTX source must remain unchanged")
    sys.path.insert(0, str(source / "packages/ltx-core/src"))
    sys.path.insert(0, str(source / "packages/ltx-pipelines/src"))
    import torch
    from ltx_core.types import LatentState
    from ltx_core.utils import to_denoised
    from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
    from ltx_pipelines.utils.helpers import timesteps_from_mask
    from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
    from ltx_pipelines.utils.types import DenoisedLatentResult, VideoAudio

    sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32)
    # Three video tokens x two channels: a strength-1 first frame, a strength-0.8 guide, a generated token.
    video_mask = torch.tensor([[[0.0], [0.2], [1.0]]], dtype=torch.float32)
    clean_video = torch.tensor([[[0.75, -0.5], [1.25, 0.25], [0.0, 0.0]]], dtype=torch.float32)
    noise_generator = torch.Generator().manual_seed(20261014)
    initial_noise = torch.randn((1, 3, 2), generator=noise_generator)
    initial_video = initial_noise * video_mask + clean_video * (1 - video_mask)
    initial_audio = torch.randn((1, 2, 2), generator=noise_generator)
    audio_mask = torch.ones((1, 2, 1), dtype=torch.float32)
    clean_audio = torch.zeros((1, 2, 2), dtype=torch.float32)

    video_state = LatentState(latent=initial_video.clone(), denoise_mask=video_mask, positions=torch.zeros(1, 3, 3, 2),
                              clean_latent=clean_video)
    audio_state = LatentState(latent=initial_audio.clone(), denoise_mask=audio_mask, positions=torch.zeros(1, 1, 2, 2),
                              clean_latent=clean_audio)

    steps = []

    def denoiser(_transformer, video, audio, schedule, step_index):
        sigma = schedule[step_index]
        # Opposite-stream means make the joint update observable, as in the Euler fixture.
        video_velocity = video.latent * 0.25 + audio.latent.mean() * 0.125 + (step_index + 1) * 0.0625
        audio_velocity = audio.latent * -0.375 + video.latent.mean() * 0.1875 - (step_index + 1) * 0.03125
        steps.append(dict(sigma=float(sigma), video=video.latent.flatten().tolist(), audio=audio.latent.flatten().tolist(),
                          video_velocity=video_velocity.flatten().tolist(), audio_velocity=audio_velocity.flatten().tolist(),
                          video_noise=None, audio_noise=None))
        # X0Model: to_denoised(latent, velocity, timesteps) with per-token timesteps = mask * sigma.
        denoised_video = to_denoised(video.latent, video_velocity, timesteps_from_mask(video.denoise_mask, sigma))
        denoised_audio = to_denoised(audio.latent, audio_velocity, timesteps_from_mask(audio.denoise_mask, sigma))
        return VideoAudio(DenoisedLatentResult(denoised=denoised_video), DenoisedLatentResult(denoised=denoised_audio))

    def recorded_noise(latent, generator):
        noise = torch.randn(latent.shape, generator=generator, dtype=latent.dtype)
        stream = "video_noise" if steps[-1]["video_noise"] is None else "audio_noise"
        steps[-1][stream] = noise.flatten().tolist()
        return noise

    final = euler_ancestral_denoising_loop(sigmas, video_state, audio_state, transformer=None, denoiser=denoiser,
                                           noise_seed=10042, new_noise_fn=recorded_noise, model_dtype=torch.float32,
                                           eta=1.0, s_noise=1.0)
    fixture = dict(
        source_revision=revision,
        torch_version=torch.__version__,
        dtype="float32",
        eta=1.0,
        s_noise=1.0,
        source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in [
            "packages/ltx-core/src/ltx_core/utils.py",
            "packages/ltx-core/src/ltx_core/components/diffusion_steps.py",
            "packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py",
            "packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py"]},
        sigmas=sigmas.tolist(), video_tokens=3, audio_tokens=2,
        video_mask=video_mask.flatten().tolist(), clean_video=clean_video.flatten().tolist(),
        initial_video=initial_video.flatten().tolist(), initial_audio=initial_audio.flatten().tolist(),
        steps=steps, final_video=final.video.latent.flatten().tolist(), final_audio=final.audio.latent.flatten().tolist(),
        limitations="Synthetic Float32 velocity/X0/ancestral-Euler arithmetic with recorded noise. No model weights, "
                    "noise RNG parity, text encoding, VAE, or video generation executed.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), steps=len(steps), revision=revision)))


if __name__ == "__main__":
    main()
