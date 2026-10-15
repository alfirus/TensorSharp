#!/usr/bin/env python3
"""Render decoded LTX reference-conditioned samples for human visual review.

This preview is not a perceptual parity score. The reference row is a Pillow
center-crop preview, not an oracle for the model's exact resize arithmetic.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    native = json.loads(args.native.read_text())
    if not native.get("completed"):
        raise ValueError("Native generation did not complete.")
    settings = native["settings"]
    frames, width, height = [settings[key] for key in ("Frames", "Width", "Height")]
    if args.tile <= 0 or args.columns < 2:
        raise ValueError("Positive tile size and at least two columns are required.")
    indices = np.linspace(0, frames - 1, args.columns).round().astype(int).tolist()
    engines = [("TensorSharp", args.native, np.float32)]
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        if not baseline.get("completed"):
            raise ValueError("Baseline generation did not complete.")
        if (baseline["settings"]["frames"], baseline["settings"]["width"], baseline["settings"]["height"]) != (frames, width, height):
            raise ValueError("Video geometry differs.")
        engines.append(("Unchanged SD.cpp", args.baseline, np.uint8))
    tile_width, tile_height = args.tile, max(1, round(args.tile * height / width))
    sheet = Image.new("RGB", (tile_width * args.columns, tile_height * (1 + len(engines)) + 32), (22, 22, 22))
    draw = ImageDraw.Draw(sheet)
    source = ImageOps.fit(Image.open(args.image).convert("RGB"), (tile_width, tile_height))
    for column in range(args.columns):
        sheet.paste(source, (column * tile_width, 32))
    for row, (label, report, dtype) in enumerate(engines, 1):
        suffix = ".rgb.f32" if dtype == np.float32 else ".rgb.u8"
        samples = np.fromfile(str(report) + "." + str(args.repeat) + suffix, dtype=dtype).reshape(frames, height, width, 3)
        if dtype == np.float32:
            samples = np.clip(samples * 255, 0, 255).astype(np.uint8)
        for column, index in enumerate(indices):
            frame = Image.fromarray(samples[index]).resize((tile_width, tile_height))
            position = (column * tile_width, 32 + row * tile_height)
            sheet.paste(frame, position)
            draw.text((position[0] + 4, position[1] + 4), f"{label}: frame {index}", fill="yellow")
    draw.text((6, 8), "Uploaded reference preview; decoded generation samples. Visual review only.", fill="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(json.dumps({"output": str(args.output), "frames": indices, "numpy": np.__version__}))


if __name__ == "__main__":
    main()
