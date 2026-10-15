#!/usr/bin/env python3
"""Download the pinned official LTX2.5 companion set with scoped HF CLI auth.

Never collect tokens or live process argv. CLI output is deliberately not echoed
because transport failures may include temporary signed URLs. Requires a local
HF CLI environment already authenticated by the user; no model is executed.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--access-manifest", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=10)
    parser.add_argument("--video-vae", choices=("diffusion", "conv"), default="diffusion",
                        help="Official diffusion decoder, or conv decoder required by stable-diffusion.cpp")
    args = parser.parse_args()
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < 0:
        parser.error("--reserve-gib must be finite and nonnegative")
    manifest = json.loads(args.access_manifest.read_text())
    if not manifest["access_confirmed"] or not manifest["credential_available"]:
        raise ValueError("Official access must first be authenticated and confirmed")
    video_vae = "vae/ltx-2.5-video-vae" + ("-conv" if args.video_vae == "conv" else "") + "-bf16.safetensors"
    names = ["text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
             video_vae, "vae/ltx-2.5-audio-vae-bf16.safetensors"]
    files = []
    for name in names:
        matches = [item for item in manifest["files"] if item["path"] == name]
        if len(matches) != 1:
            raise ValueError("Require exactly one pinned metadata entry for selected companion: " + name)
        files.append(matches[0])
    if any(not item["access_confirmed"] or item["commit_hash"] != manifest["revision"] or
           item["etag"] != item["lfs_sha256"] for item in files):
        raise ValueError("Require pinned official metadata for every companion")
    args.models.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), repository=manifest["repository"],
                  revision=manifest["revision"], models=str(args.models.resolve()), access_manifest_sha256=sha(args.access_manifest),
                  downloader_sha256=sha(Path(__file__)), hf_cli=str(args.hf.resolve()), video_vae=args.video_vae,
                  files=[], completed=False,
                  limitations="Authenticated official file download/hash validation only; no text encoder, VAE, denoiser or whole generation was executed. No credentials or temporary signed transport URLs are logged.")
    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        valid_existing = {}
        resumable = {}
        for item in files:
            path = args.models / item["path"]
            valid_existing[item["path"]] = path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["lfs_sha256"]
            # Only this pinned component's own resume file can reduce future
            # disk growth. Unrelated cache/model files never count as credit.
            cache = args.models / ".cache/huggingface/download" / Path(item["path"]).parent
            sizes = [p.stat().st_size for p in cache.glob("*." + item["lfs_sha256"] + ".incomplete") if p.is_file()]
            resumable[item["path"]] = min(item["bytes"], max(sizes, default=0)) if not path.exists() else 0
        missing = sum(item["bytes"] for item in files if not valid_existing[item["path"]])
        remaining = sum(item["bytes"] - resumable[item["path"]] for item in files if not valid_existing[item["path"]])
        report["free_bytes_before"] = shutil.disk_usage(args.models).free
        report["required_bytes"] = sum(item["bytes"] for item in files)
        report["missing_bytes"] = missing
        report["resumable_bytes"] = resumable
        report["remaining_disk_growth_bytes"] = remaining
        report["reserve_bytes"] = int(args.reserve_gib * 1024 ** 3)
        if report["free_bytes_before"] < remaining + report["reserve_bytes"]:
            raise ValueError("Insufficient disk space for required companions plus reserve")
        for item in files:
            path = args.models / item["path"]
            record = dict(path=item["path"], expected_bytes=item["bytes"], expected_sha256=item["lfs_sha256"], completed=False)
            report["files"].append(record)
            save()
            start = time.monotonic()
            if not valid_existing[item["path"]]:
                argv = [str(args.hf), "download", manifest["repository"], item["path"], "--revision", manifest["revision"],
                        "--local-dir", str(args.models), "--quiet"]
                if path.exists():
                    argv.append("--force-download")  # Repair a known corrupt completed target.
                record["command"] = argv
                print("downloading", item["path"], "bytes", item["bytes"], flush=True)
                # Disable Xet's additional chunk cache. local-dir writes directly
                # to the requested component set, retaining resumable metadata.
                code = subprocess.run(argv, env=dict(os.environ, HF_HUB_DISABLE_XET="1"),
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode
                record["exit_code"] = code
                if code:
                    raise RuntimeError("HF CLI download failed; only sanitized exit code recorded")
            record["actual_bytes"] = path.stat().st_size
            record["actual_sha256"] = sha(path)
            record["elapsed_seconds_including_hash"] = time.monotonic() - start
            record["completed"] = record["actual_bytes"] == item["bytes"] and record["actual_sha256"] == item["lfs_sha256"]
            save()
            if not record["completed"]:
                raise ValueError("Official companion size/hash mismatch")
            print("verified", item["path"], "sha256", record["actual_sha256"], flush=True)
        report["completed"] = True
        return 0
    except Exception as error:
        report["error_type"] = type(error).__name__
        print("download incomplete; sanitized error type", type(error).__name__, flush=True)
        return 1
    finally:
        report["free_bytes_after"] = shutil.disk_usage(args.models).free
        save()


if __name__ == "__main__":
    raise SystemExit(main())
