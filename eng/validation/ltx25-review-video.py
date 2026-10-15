#!/usr/bin/env python3
"""CPU-only contact sheets and descriptive source-preservation/motion statistics.

These measurements do not establish semantic prompt adherence, perceptual
quality, or a quality winner. Review the exported frames for those conclusions.
Reference RGB should be the exact preprocessed bytes supplied to the encoder.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image(values):
    return Image.fromarray(np.clip(values * 255 + .5, 0, 255).astype(np.uint8))


def compare(reference, observed):
    error = reference.astype(np.float64) - observed.astype(np.float64)
    mse = float(np.mean(error * error))
    return dict(mae=float(np.abs(error).mean()), rmse=math.sqrt(mse),
                psnr_db=-10 * math.log10(mse) if mse else None,
                exact_equal=bool(np.array_equal(reference, observed)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-rgb", type=Path, required=True)
    parser.add_argument("--video", action="append", nargs=4, metavar=("LABEL", "PATH", "DTYPE", "FRAMES"), required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.width, args.height, args.fps, args.samples, args.tile_size) <= 0:
        raise SystemExit("Geometry, FPS, sample count and tile size must be positive")
    if args.output.exists():
        raise SystemExit("Refusing to replace existing review output")
    args.output.mkdir(parents=True)
    reference = np.fromfile(args.reference_rgb, dtype="<f4").reshape(args.height, args.width, 3)
    if not np.isfinite(reference).all() or reference.min() < 0 or reference.max() > 1:
        raise SystemExit("Reference must contain finite Float32 RGB01 pixels")
    crop = None
    if args.crop:
        x, y, width, height = args.crop
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > args.width or y + height > args.height:
            raise SystemExit("Crop exceeds the RGB geometry")
        crop = (x, y, x + width, y + height)
    samples = max(2, args.samples)
    row_height = args.tile_size + 28
    sheet = Image.new("RGB", (samples * args.tile_size, (len(args.video) + 1) * row_height), "#151515")
    crop_sheet = Image.new("RGB", sheet.size, "#151515") if crop else None
    draw = ImageDraw.Draw(sheet); crop_draw = ImageDraw.Draw(crop_sheet) if crop else None
    reference_image = image(reference)
    reference_image.save(args.output / "reference.png")
    for col in range(samples):
        sheet.paste(reference_image.resize((args.tile_size, args.tile_size), Image.Resampling.LANCZOS), (col * args.tile_size, 28))
        if crop:
            crop_sheet.paste(reference_image.crop(crop).resize((args.tile_size, args.tile_size), Image.Resampling.LANCZOS), (col * args.tile_size, 28))
    draw.text((5, 7), "Exact preprocessed uploaded reference (repeated for comparison)", fill="white")
    if crop: crop_draw.text((5, 7), "Reference detail crop (visual assessment only)", fill="white")
    report = dict(scope=__doc__, reference_rgb=str(args.reference_rgb.resolve()), reference_sha256=sha256(args.reference_rgb),
                  geometry=[args.height, args.width, 3], fps=args.fps, crop_xywh=args.crop, videos=[],
                  limitations="PSNR/MAE measure pixel similarity to the source, not desired motion or prompt adherence. More or less pixel motion is not a quality ranking. Semantic gesture success requires visual review of the full clip.")
    for row, (label, filename, kind, frames) in enumerate(args.video, 1):
        path = Path(filename); frames = int(frames)
        dtype = {"f32": "<f4", "u8": "u1"}.get(kind)
        if dtype is None or frames <= 0:
            raise SystemExit("Videos require f32 or u8 data and positive frame count")
        values = np.fromfile(path, dtype=dtype).reshape(frames, args.height, args.width, 3)
        if kind == "u8": values = values.astype(np.float32) / 255
        if not np.isfinite(values).all(): raise SystemExit("Nonfinite video: " + filename)
        indices = np.round(np.linspace(0, frames - 1, samples)).astype(int).tolist()
        adjacent = np.abs(np.diff(values, axis=0))
        metrics = dict(label=label, path=str(path.resolve()), sha256=sha256(path), frames=frames,
                       duration_seconds=frames / args.fps, sampled_frames=indices,
                       reference_to_first_frame=compare(reference, values[0]),
                       reference_to_last_frame=compare(reference, values[-1]),
                       first_to_last_frame=compare(values[0], values[-1]),
                       adjacent_frame_mae_mean=float(adjacent.mean()) if frames > 1 else 0,
                       adjacent_frame_mae_by_transition=adjacent.mean(axis=(1, 2, 3)).tolist())
        if crop:
            x, y, width, height = args.crop
            region = values[:, y:y + height, x:x + width]
            metrics["detail_crop"] = dict(reference_to_first_frame=compare(reference[y:y + height, x:x + width], region[0]),
                first_to_last_frame=compare(region[0], region[-1]),
                adjacent_frame_mae_mean=float(np.abs(np.diff(region, axis=0)).mean()) if frames > 1 else 0)
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        for col, index in enumerate(indices):
            frame_image = image(values[index])
            frame_image.save(args.output / f"{safe_label}-frame-{index:03}.png")
            top = row * row_height
            sheet.paste(frame_image.resize((args.tile_size, args.tile_size), Image.Resampling.LANCZOS), (col * args.tile_size, top + 28))
            draw.text((col * args.tile_size + 5, top + 7), f"{label}: {index} ({index / args.fps:.2f}s)", fill="white")
            if crop:
                crop_sheet.paste(frame_image.crop(crop).resize((args.tile_size, args.tile_size), Image.Resampling.LANCZOS), (col * args.tile_size, top + 28))
                crop_draw.text((col * args.tile_size + 5, top + 7), f"{label}: {index} ({index / args.fps:.2f}s)", fill="white")
        report["videos"].append(metrics)
    sheet.save(args.output / "contact-sheet.png")
    if crop: crop_sheet.save(args.output / "detail-contact-sheet.png")
    (args.output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), videos=[dict(label=x["label"], first_frame_psnr_db=x["reference_to_first_frame"]["psnr_db"],
                      adjacent_frame_mae_mean=x["adjacent_frame_mae_mean"]) for x in report["videos"]])))


if __name__ == "__main__":
    main()
