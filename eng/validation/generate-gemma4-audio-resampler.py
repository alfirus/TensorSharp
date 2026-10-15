#!/usr/bin/env python3
"""Independent torchaudio sinc fixtures and PCM files for Gemma resampler tests."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import wave
import numpy as np
import torch
import torchaudio


def resample(samples, rate):
    # vLLM and SGLang instantiate transforms.Resample, whose default kernel
    # construction uses float64 before storing float32 convolution weights.
    return torchaudio.transforms.Resample(rate, 16000,
        lowpass_filter_width=64, rolloff=.99, resampling_method="sinc_interp_hann")(torch.from_numpy(samples)).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--media-output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = []
    rng = np.random.default_rng(19012026)
    for rate in (8000, 11025, 16001, 22050, 24000, 44100, 48000, 96000):
        for name, samples in (("random", rng.uniform(-.5, .5, 257).astype(np.float32)),
                              ("impulse", np.eye(1, 257, 128, dtype=np.float32).ravel())):
            cases.append(dict(name=f"{name}-{rate}", source_rate=rate, samples=samples.tolist(), expected=resample(samples, rate).tolist()))
    for rate, samples in ((48000, np.array([1000 / 32768, -2000 / 32768], dtype=np.float32)),
                          (24000, np.array([.25], dtype=np.float32)),
                          (8000, np.array([.25], dtype=np.float32))):
        cases.append(dict(name=f"short-{rate}", source_rate=rate, samples=samples.tolist(), expected=resample(samples, rate).tolist()))
    for name, frequency in (("passband", 1000), ("alias", 11000)):
        samples = (.25 * np.sin(2 * np.pi * frequency * np.arange(4800) / 48000)).astype(np.float32)
        cases.append(dict(name=name, source_rate=48000, samples=samples.tolist(), expected=resample(samples, 48000).tolist()))
    source = Path(torchaudio.transforms.Resample.__init__.__code__.co_filename)
    kernel_source = Path(torchaudio.functional.resample.__code__.co_filename)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(dict(method="Executed primary torchaudio.transforms.Resample", torch=torch.__version__,
        torchaudio=torchaudio.__version__, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        kernel_source_sha256=hashlib.sha256(kernel_source.read_bytes()).hexdigest(),
        filter_width=64, rolloff=.99, window="sinc_interp_hann", target_rate=16000,
        absolute_tolerance=2e-6, cases=cases), separators=(",", ":")) + "\n")
    args.media_output.mkdir(parents=True, exist_ok=True)
    files = []
    for rate in (44100, 48000):
        time = np.arange(rate) / rate
        envelope = np.minimum(1, np.minimum(time / .01, (1 - time) / .01))
        signal = envelope * (.2 * np.sin(2 * np.pi * 440 * time) + .08 * np.sin(2 * np.pi * 880 * time) + .04 * np.sin(2 * np.pi * 11000 * time))
        pcm = np.rint(signal * 32767).astype("<i2")
        source_path = args.media_output / f"tone-{rate}.wav"
        with wave.open(str(source_path), "wb") as writer:
            writer.setparams((1, 2, rate, 0, "NONE", "not compressed")); writer.writeframes(pcm.tobytes())
        expected = resample(pcm.astype(np.float32) / 32768, rate)
        # IEEE float WAV preserves the independently computed reference PCM.
        expected_path = args.media_output / f"tone-{rate}-reference-16000.wav"
        data = expected.astype("<f4").tobytes()
        expected_path.write_bytes(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 1, 16000, 64000, 4, 32) + b"data" + struct.pack("<I", len(data)) + data)
        files.append(dict(source=str(source_path.resolve()), reference=str(expected_path.resolve()),
            source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), reference_sha256=hashlib.sha256(expected_path.read_bytes()).hexdigest()))
    (args.media_output / "resampling-media.json").write_text(json.dumps(dict(files=files), indent=2) + "\n")


if __name__ == "__main__":
    main()
