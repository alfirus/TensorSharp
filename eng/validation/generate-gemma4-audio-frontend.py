#!/usr/bin/env python3
"""Generate an independent NumPy oracle for the Gemma 4 / EmbeddingGemma 2 audio frontend."""
import argparse
import json
from pathlib import Path

import numpy as np

SOURCE_REVISION = "976e7a41c6615559f7a2ccb79fd5212c378b2c95"
SOURCE_PATH = "src/transformers/models/gemma4/feature_extraction_gemma4.py"


def spectrogram(samples, log_policy="huggingface_add"):
    frame_length, hop, fft_length, bins, rate = 320, 160, 512, 128, 16000
    waveform = np.pad(np.asarray(samples, dtype=np.float32), (frame_length // 2, 0))
    frames = max(0, (waveform.size - (frame_length + 1)) // hop + 1)
    if not frames:
        return np.empty((0, bins), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(waveform, frame_length + 1)[::hop, :-1]
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_length) / frame_length)).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(windows * hann, n=fft_length, axis=-1))
    mel_max = 2595 * np.log10(1 + 8000 / 700)
    hz = 700 * (10 ** (np.linspace(0, mel_max, bins + 2) / 2595) - 1)
    frequencies = np.linspace(0, rate // 2, fft_length // 2 + 1)
    slopes = hz[None, :] - frequencies[:, None]
    lower = -slopes[:, :-2] / np.diff(hz)[:-1]
    upper = slopes[:, 2:] / np.diff(hz)[1:]
    filters = np.maximum(0, np.minimum(lower, upper))
    values = magnitude @ filters
    return np.log(np.maximum(values, 0.001) if log_policy == "llama_clamp" else values + 0.001).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("InferenceWeb.Tests/Fixtures/EmbeddingGemma2/audio-frontend.json"))
    parser.add_argument("--log-policy", choices=["huggingface_add", "llama_clamp"], default="huggingface_add")
    args = parser.parse_args()
    cases = []
    inputs = [(f"silence-{length}", np.zeros(length, dtype=np.float32)) for length in [0, 160, 161, 320, 321, 769, 800, 1280]]
    impulse = np.zeros(1024, dtype=np.float32)
    impulse[0], impulse[160], impulse[319], impulse[800] = .25, -.125, .1, -.2
    inputs.append(("impulses", impulse))
    indices = np.arange(1024, dtype=np.float64)
    inputs.append(("two-tone", (.2 * np.sin(2 * np.pi * 440 * indices / 16000)
                                 + .05 * np.cos(2 * np.pi * 1230 * indices / 16000)).astype(np.float32)))
    for name, samples in inputs:
        mel = spectrogram(samples, args.log_policy)
        padded = np.pad(samples, (0, (-samples.size) % 128))
        padded_frames = spectrogram(padded, args.log_policy).shape[0]
        cases.append({"name": name, "samples": samples.tolist(), "expected_frames": mel.shape[0],
                      "expected_mel": mel.reshape(-1).tolist(), "expected_tokens_padded": (padded_frames + 3) // 4,
                      "expected_tokens_valid": (mel.shape[0] + 3) // 4})
    report = {"source_revision": SOURCE_REVISION, "source_path": SOURCE_PATH,
              "source_url": f"https://github.com/huggingface/transformers/blob/{SOURCE_REVISION}/{SOURCE_PATH}",
              "algorithm": "semicausal160+periodicHann320+FFT512+magnitude+HTK128+" + ("log(max(mel,0.001))" if args.log_policy == "llama_clamp" else "log(mel+0.001)"),
              "sample_rate": 16000, "mel_bins": 128, "absolute_tolerance": 2e-6,
              "numpy_version": np.__version__, "cases": cases}
    if args.log_policy == "llama_clamp":
        report["comparison_source_revision"] = "4413a192410fad1f054e9216114087a3e01527a0"
        report["comparison_source_url"] = "https://github.com/ggml-org/llama.cpp/blob/4413a192410fad1f054e9216114087a3e01527a0/tools/mtmd/mtmd-audio.cpp"
        report["limitations"] = "Explicit compatibility policy only; the Hugging Face default fixture and policy remain separate."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
