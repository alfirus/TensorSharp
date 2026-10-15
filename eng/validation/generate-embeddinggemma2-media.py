#!/usr/bin/env python3
"""Generate deterministic EmbeddingGemma 2 media inputs, without model weights."""
import argparse
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import wave
import zlib


def write_png(path, width, height, pixel):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    rows = b"".join(b"\0" + bytes(v for x in range(width) for v in pixel(x, y)) for y in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/validation/embeddinggemma2-reference/media"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_png(output / "red.png", 192, 192, lambda x, y: (255, 0, 0))
    write_png(output / "pattern.png", 192, 128,
              lambda x, y: ((x * 255 // 191), (y * 255 // 127), (x + 2 * y) % 256))
    # Already aligned to the HF 280-token canvas. These distinguish encoder
    # numerical parity from llama.cpp's different resize and letterbox policy.
    write_png(output / "red-matched.png", 768, 768, lambda x, y: (255, 0, 0))
    write_png(output / "pattern-matched.png", 960, 624,
              lambda x, y: ((x * 255 // 959), (y * 255 // 623), (x + 2 * y) % 256))
    rate, seconds = 16000, 1
    pcm = bytearray()
    for i in range(rate * seconds):
        envelope = min(1, i / 160, (rate * seconds - 1 - i) / 160)
        value = .2 * envelope * (math.sin(2 * math.pi * 440 * i / rate) + .4 * math.sin(2 * math.pi * 880 * i / rate))
        pcm.extend(struct.pack("<h", round(32767 * value)))
    with wave.open(str(output / "tone.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(pcm)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, color in enumerate(colors):
        write_png(output / f"frame-{i:02d}.png", 192, 192, lambda x, y, c=color: c)
        write_png(output / f"matched-frame-{i:02d}.png", 528, 528, lambda x, y, c=color: c)
    ffmpeg = shutil.which(args.ffmpeg)
    video = None
    if ffmpeg:
        video = output / "colors.mp4"
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-framerate", "1", "-i", str(output / "frame-%02d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "1", str(video)], check=True)
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-framerate", "1", "-i", str(output / "matched-frame-%02d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "1", str(output / "colors-matched.mp4")], check=True)
    manifest = {"red": str(output / "red.png"), "pattern": str(output / "pattern.png"),
                "audio": str(output / "tone.wav"), "video": str(video) if video else None,
                "video_available": video is not None, "video_frames": 4, "video_fps": 1, "audio_seconds": seconds}
    manifest.update({"red_matched": str(output / "red-matched.png"), "pattern_matched": str(output / "pattern-matched.png"),
                     "video_matched": str(output / "colors-matched.mp4") if video else None})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
