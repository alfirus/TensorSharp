#!/usr/bin/env python3
"""Verify an actual multimodal embedding host, requests, and graceful disposal.

This is a lifecycle/API regression check, not a reference quality or timing gate.
Use generate-embeddinggemma2-media.py to create the required media directory.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-dll", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--expected-backend", required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18490)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ["dotnet", str(args.host_dll.resolve()), "--model", str(args.model.resolve()),
               "--mmproj", str(args.mmproj.resolve()), "--embeddings", "--backend", args.backend,
               "--embedding-threads", str(args.threads), "--host", "127.0.0.1", "--port", str(args.port), "--no-webui"]
    report = dict(command=command, expected_backend=args.expected_backend,
                  model_sha256=file_hash(args.model), projector_sha256=file_hash(args.mmproj),
                  cases=[], completed=False, graceful_shutdown=False,
                  limitations="API, finite normalized outputs, and lifecycle only; no reference quality or performance claim.")
    binaries = [args.host_dll, *args.host_dll.parent.glob("libGgmlOps.*"), *args.host_dll.parent.glob("cuda_kernels/*.ptx")]
    report["binary_artifacts"] = [dict(path=str(path.resolve()), sha256=file_hash(path)) for path in binaries]
    url = f"http://127.0.0.1:{args.port}"

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error

    def media(kind, filename, mime):
        path = args.media_dir / filename
        encoded = base64.b64encode(path.read_bytes()).decode()
        report.setdefault("media_files", {})[filename] = file_hash(path)
        if kind == "image":
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
        if kind == "audio":
            return {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}
        return {"type": "input_video", "input_video": {"url": f"data:{mime};base64,{encoded}"}}

    text = {"type": "text", "text": "task: search result | query: Describe this input."}
    image = media("image", "red.png", "image/png")
    audio = media("audio", "tone.wav", "audio/wav")
    video = media("video", "colors.mp4", "video/mp4")
    cases = {"text": [text], "image": [text, image], "audio": [text, audio],
             "video": [text, video], "ordered_mixed": [text, image, audio, video]}
    failure = None
    process = None
    with args.output.with_suffix(".server.log").open("w") as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Host exited during startup: {process.returncode}")
                try:
                    information = request("/api/models")
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Host was not ready within 180 seconds")
                    time.sleep(.5)
            report["actual_backend"] = information["loadedBackend"]
            if report["actual_backend"] != args.expected_backend:
                raise ValueError(f"Backend mismatch: {report['actual_backend']}")
            model_id = request("/v1/models")["data"][0]["id"]
            for name, content in cases.items():
                result = request("/v1/embeddings", {"model": model_id, "input": [{"content": content}], "encoding_format": "float"})
                vector = result["data"][0]["embedding"]
                norm = math.sqrt(sum(value * value for value in vector))
                if len(vector) != 768 or any(not math.isfinite(value) for value in vector) or abs(norm - 1) > 1e-5:
                    raise ValueError(f"{name} returned invalid normalized 768-dimensional output")
                tokens = result["usage"]["prompt_tokens"]
                if not 0 < tokens <= 8192:
                    raise ValueError(f"{name} returned invalid prompt token count {tokens}")
                report["cases"].append(dict(name=name, prompt_tokens=tokens, norm=norm, embedding=vector))
                print(f"{args.expected_backend} {name}: {tokens} tokens, finite normalized output", flush=True)
        except Exception as error:
            failure = error
            report["error"] = str(error)
        finally:
            if process is not None:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                        report["forced_shutdown"] = True
                report["exit_code"] = process.returncode
                report["graceful_shutdown"] = process.returncode == 0 and not report.get("forced_shutdown", False)
            report["completed"] = failure is None and report["graceful_shutdown"]
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    if failure is not None:
        raise failure
    if not report["completed"]:
        raise RuntimeError(f"Host disposal failed: exit {report.get('exit_code')}")
    print("Graceful shutdown: exit 0", flush=True)


if __name__ == "__main__":
    main()
