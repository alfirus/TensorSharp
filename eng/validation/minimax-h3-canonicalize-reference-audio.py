#!/usr/bin/env python3
"""Trim 32 kHz stereo PCM16 reference audio for an explicit matched benchmark.

This prepares a fresh WAV and adjacent .wav.json manifest. It does not change
either engine's reference-audio preprocessing or generation behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import wave


SAMPLE_RATE = 32000
CHANNELS = 2
SAMPLE_BYTES = 2
HOP = 800
SCOPE = "explicit benchmark canonicalization, not pipeline change"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize(source, output, frames, fps):
    if frames <= 0 or fps <= 0:
        raise ValueError("--frames and --fps must be positive")
    source = source.resolve()
    output = output.resolve()
    manifest_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or manifest_path.exists():
        raise ValueError("Refusing existing output WAV or manifest")

    with wave.open(str(source), "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getcomptype()) != (
            SAMPLE_RATE, CHANNELS, SAMPLE_BYTES, "NONE"
        ):
            raise ValueError("Input must be uncompressed 32 kHz stereo PCM16 WAV")
        source_frames = reader.getnframes()
        duration_bound_frames = frames * SAMPLE_RATE // fps
        output_frames = min(source_frames, duration_bound_frames) // HOP * HOP
        if output_frames == 0:
            raise ValueError("Reference audio is empty after trimming to a whole 800-sample hop")
        payload = reader.readframes(output_frames)
        if len(payload) != output_frames * CHANNELS * SAMPLE_BYTES:
            raise ValueError("Input WAV has a truncated PCM payload")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        with wave.open(stream, "wb") as writer:
            writer.setnchannels(CHANNELS)
            writer.setsampwidth(SAMPLE_BYTES)
            writer.setframerate(SAMPLE_RATE)
            writer.writeframes(payload)
    manifest = {
        "scope": SCOPE,
        "rule": "floor(min(input sample frames, frames/fps*32000)/800)*800",
        "request": {"frames": frames, "fps": fps, "duration_bound_sample_frames": duration_bound_frames},
        "format": {"sample_rate": SAMPLE_RATE, "channels": CHANNELS, "pcm_bits": SAMPLE_BYTES * 8},
        "source": {"path": str(source), "sha256": sha256(source), "sample_frames": source_frames},
        "output": {"path": str(output), "sha256": sha256(output), "sample_frames": output_frames,
                   "seconds": output_frames / SAMPLE_RATE, "pcm_sha256": hashlib.sha256(payload).hexdigest()},
        "processed_input_equivalence_established": False,
        "limitations": [
            "Identical reference files do not prove identical engine preprocessing.",
            "TensorSharp pads a reference shorter than its selected output duration; inspect effective lengths when the source is short.",
        ],
    }
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--fps", required=True, type=int)
    args = parser.parse_args()
    try:
        path, manifest = canonicalize(args.input, args.output, args.frames, args.fps)
    except (ValueError, OSError, EOFError, wave.Error) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps({"manifest": str(path), "output": manifest["output"], "scope": SCOPE}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
