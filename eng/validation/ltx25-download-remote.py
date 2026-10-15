#!/usr/bin/env python3
"""Provision pinned LTX weights without putting HF credentials on a test VM.

Local mode resolves authenticated, temporary CDN locations using cached HF auth.
Locations travel only through SSH stdin and never enter argv, files or reports.
The remote receiver retains resumable owned partials and verifies every hash.
Run local mode in a Python environment containing huggingface_hub.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receive(args):
    request = json.load(sys.stdin)
    models = args.models.resolve()
    models.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"started_unix": time.time(), "status": "downloading", "files": [],
              "limitations": "Pinned size/hash download validation only; no inference executed.",
              "authentication": "Temporary signed CDN locations received via SSH stdin; no HF token received or persisted."}
    records = []
    for item in request["files"]:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe model path")
        records.append({key: value for key, value in item.items() if key != "location"})
    report["files"] = records
    needed = sum(item["bytes"] for item in records
                 if not (models / item["path"]).is_file())
    if shutil.disk_usage(models).free < needed + 5 * 1024 ** 3:
        raise ValueError("Insufficient disk space plus 5 GiB reserve")

    def download(pair):
        item, record = pair
        target = models / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".ltx25-download-partial")
        started = time.monotonic()
        try:
            if target.exists():
                if target.stat().st_size != item["bytes"] or sha(target) != item["sha256"]:
                    raise ValueError("Existing completed target differs; preserved")
            else:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > item["bytes"]:
                    raise ValueError("Owned partial exceeds expected size; preserved")
                if offset < item["bytes"]:
                    headers = {"Range": "bytes=" + str(offset) + "-"} if offset else {}
                    with urllib.request.urlopen(urllib.request.Request(item["location"], headers=headers), timeout=180) as source:
                        if offset and (source.status != 206 or not source.headers.get("Content-Range", "").startswith("bytes " + str(offset) + "-")):
                            raise ValueError("Server did not honor partial range; preserved")
                        with partial.open("ab" if partial.exists() else "xb") as output:
                            shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
                if partial.stat().st_size != item["bytes"] or sha(partial) != item["sha256"]:
                    raise ValueError("Downloaded model size/hash mismatch; partial preserved")
                partial.replace(target)
            record.update(completed=True, actual_bytes=target.stat().st_size, actual_sha256=item["sha256"])
        except Exception as error:
            record.update(completed=False, error_type=type(error).__name__)
        record["elapsed_seconds_including_hash"] = time.monotonic() - started
        print(json.dumps(record), flush=True)
        return record["completed"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(download, zip(request["files"], records)))
    report.update(status="verified" if all(outcomes) else "incomplete", wall_seconds=time.time() - report["started_unix"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(outcomes) else 1


def provision(args):
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    files = []
    public = json.loads(args.public_manifest.read_text())
    for item in public["shards"]:
        files.append({"path": item["url"].rsplit("/", 1)[1], "bytes": item["bytes"],
                      "sha256": item["sha256"], "repository": public["repository"],
                      "revision": public["revision"], "location": item["url"]})
    companions = json.loads(args.companion_manifest.read_text())
    names = ["text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
             "vae/ltx-2.5-video-vae-conv-bf16.safetensors", "vae/ltx-2.5-audio-vae-bf16.safetensors"]
    if not companions["access_confirmed"] or not companions["credential_available"]:
        raise ValueError("Authenticated official access must be confirmed first")
    for name in names:
        item = next(row for row in companions["files"] if row["path"] == name)
        if not item["access_confirmed"] or item["commit_hash"] != companions["revision"] or item["etag"] != item["lfs_sha256"]:
            raise ValueError("Official metadata is not pinned")
        metadata = get_hf_file_metadata(hf_hub_url(companions["repository"], name, revision=companions["revision"]), token=True)
        if metadata.commit_hash != companions["revision"] or metadata.etag != item["lfs_sha256"] or metadata.size != item["bytes"]:
            raise ValueError("Authenticated metadata differs from manifest")
        files.append({"path": name, "bytes": item["bytes"], "sha256": item["lfs_sha256"],
                      "repository": companions["repository"], "revision": companions["revision"], "location": metadata.location})
    command = ["ssh", "-o", "BatchMode=yes", "-p", str(args.port), "-i", str(args.identity), args.host,
               shlex.join(["python3", str(args.remote_script), "--receive", "--models", str(args.models), "--output", str(args.output)])]
    # Capture transport errors: neither signed locations nor credential-bearing
    # exception representations are allowed into terminal output or evidence.
    result = subprocess.run(command, input=json.dumps({"files": files}), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stdout, end="", flush=True)
    print("remote_downloader_exit_code", result.returncode, flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receive", action="store_true")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path)
    parser.add_argument("--companion-manifest", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--remote-script", type=Path)
    args = parser.parse_args()
    try:
        return receive(args) if args.receive else provision(args)
    except Exception as error:
        print("download_incomplete_error_type", type(error).__name__, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
