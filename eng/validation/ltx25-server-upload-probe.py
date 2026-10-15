#!/usr/bin/env python3
"""Replay an image-conditioned request against an already running real HTTP host."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5001")
    parser.add_argument("--image", required=True, type=Path, action="append")
    parser.add_argument("--mode", choices=("i2v", "ref"), default="i2v")
    parser.add_argument("--prompt", default="请让图中的女人来个飞吻")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text()
    if args.frames is None:
        args.frames = 121 if args.mode == "ref" else 17
    if args.mode == "i2v" and len(args.image) != 1:
        parser.error("i2v requires one image; use ref for an identity sheet")
    width, height = (768, 448) if args.mode == "ref" else (512, 512)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"completed": False, "url": args.url, "image": str(args.image[0].resolve()),
              "images": [str(p.resolve()) for p in args.image], "mode": args.mode,
              "prompt": args.prompt, "frames": args.frames, "events": [], "uploads": []}
    save = lambda: args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    save()

    def get(path):
        with urllib.request.urlopen(args.url + path, timeout=1800) as response:
            return response.read()

    try:
        models = json.loads(get("/api/models"))
        report["models"] = models
        caps = models["video"]
        assert models["architecture"] == "ltx-2.5" and caps["supportsImageConditioning"]
        assert caps["supportsEndImageConditioning"] and caps["supportsKeyframeConditioning"]
        if args.mode == "ref":
            assert caps["supportsReferenceConditioning"]
        for image in args.image:
            raw = image.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            boundary = "TensorSharpProbe" + uuid.uuid4().hex
            multipart = ((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="reference.png"\r\n'
                          'Content-Type: image/png\r\n\r\n').encode() + raw + f"\r\n--{boundary}--\r\n".encode())
            request = urllib.request.Request(args.url + "/api/upload", data=multipart,
                        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
            with urllib.request.urlopen(request, timeout=60) as response:
                upload = json.load(response)
            assert upload["mediaType"] == "image"
            assert hashlib.sha256(get(upload["url"])).hexdigest() == digest
            report["uploads"].append({"image": str(image.resolve()), "sha256": digest, "upload": upload})
        report["upload"] = report["uploads"][0]["upload"]
        report["image_sha256"] = report["uploads"][0]["sha256"]
        body = {"prompt": args.prompt, "frames": args.frames,
                "width": width, "height": height, "fps": 24, "steps": 8, "cfg": 1,
                "sampler": "euler", "seed": args.seed, "generateAudio": True, "videoMode": args.mode}
        if args.mode == "ref":
            body["referenceImages"] = [item["upload"]["file"] for item in report["uploads"]]
        else:
            body["imagePath"] = report["upload"]["file"]
        report["request"] = body
        save()
        request = urllib.request.Request(args.url + "/api/video-generate/stream", data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        started = time.perf_counter()
        terminal = None
        with urllib.request.urlopen(request, timeout=1800) as response:
            assert response.headers.get_content_type() == "text/event-stream"
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                report["events"].append(event)
                save()
                if event.get("error"):
                    raise RuntimeError(event["error"])
                if event.get("done"):
                    terminal = event
        report["http_generation_ms"] = (time.perf_counter() - started) * 1000
        assert terminal and terminal["frames"] == args.frames and terminal["seed"] == args.seed
        assert terminal["width"] == width and terminal["height"] == height
        assert terminal["audioMuxed"] and terminal["codec"] == "h264"
        report["terminal"] = terminal
        for key, suffix in (("url", ".mp4"), ("audioUrl", ".wav")):
            content = get(terminal[key])
            output = Path(str(args.output) + suffix)
            if output.exists():
                raise FileExistsError(output)
            output.write_bytes(content)
            report[key] = {"path": str(output.resolve()), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        report["completed"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        save()
    print(json.dumps({"completed": True, "http_generation_ms": report["http_generation_ms"], "report": str(args.output)}))


if __name__ == "__main__":
    main()
