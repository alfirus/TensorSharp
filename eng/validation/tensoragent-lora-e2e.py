#!/usr/bin/env python3
"""Exercise installed Turbo LoRAs through a real, isolated TensorAgentHost.

Launch eng/validation/TensorAgentHost with an isolated --root containing the
installed official Turbo catalog model and the two adapters, then run:

  python3 eng/validation/tensoragent-lora-e2e.py --connection <root>/connection.json \
      --output docs/validation/tensoragent-lora-e2e

Uses the page's catalog, choice, upload, session and chat HTTP routes. Each case
checks the applied adapter, ordered references, eight steps, a retrievable PNG,
and its saved conversation. It leaves the isolated host's final choice selected.
This is functional coverage and a serial timing sample, not a quality benchmark;
review the saved images. No model downloads or native UI interaction occur.
"""

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
import statistics
import struct
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid


CASES = [
    ("qwen-image-2.1-casual-phone", "lbxstyle, a casual phone photo of a small orange cat sitting beside a blue ceramic mug on a sunlit kitchen table, natural daylight, candid everyday scene."),
    ("qwen-image-2.1-voxel-style", "Transform the image into a voxel style."),
]


def check(condition, reason):
    if not condition:
        raise AssertionError(reason)


class Host:
    def __init__(self, connection, timeout):
        self.base = connection["baseUrl"].rstrip("/")
        check(urlsplit(self.base).hostname in ("127.0.0.1", "localhost", "::1"), "An isolated loopback host is required.")
        self.cookie = connection["cookie"]
        self.timeout = timeout
        self.timings = []

    def open(self, route, data=None, content_type="application/json"):
        headers = {"Cookie": self.cookie}
        if data is not None:
            headers["Content-Type"] = content_type
        return urlopen(Request(self.base + route, data=data, headers=headers), timeout=self.timeout)

    def json(self, route, body=None):
        before = time.monotonic()
        with self.open(route, None if body is None else json.dumps(body).encode()) as response:
            result = json.load(response)
        self.timings.append({"route": route, "seconds": time.monotonic() - before})
        return result

    def upload(self, path):
        boundary = "tensoragent-validation-" + uuid.uuid4().hex
        name = path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                f'Content-Type: {mimetypes.guess_type(name)[0] or "application/octet-stream"}\r\n\r\n').encode()
        body += path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        with self.open("/api/upload", body, f"multipart/form-data; boundary={boundary}") as response:
            result = json.load(response)
        check(result.get("ok") and result.get("mediaType") == "image", f"Image upload failed: {result}")
        with self.open(result["url"]) as response:
            check(response.read() == path.read_bytes(), "Uploaded reference bytes changed.")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Sequential samples per case; first includes adapter setup, later samples are warm.")
    parser.add_argument("--baseline", action="store_true",
                        help="Also time the same text-to-image prompt with no adapter.")
    args = parser.parse_args()
    check(args.repetitions > 0, "--repetitions must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    host = Host(json.loads(args.connection.read_text()), args.timeout)
    report = {"status": "failed", "cases": [], "limitations": [
        "Real host HTTP routes; native/WebView controls are validated separately.",
        "Sequential timing samples on one model/backend; adapter setup and prompt-cache warming affect the first sample.",
        "Baseline shares the Casual Phone prompt; Voxel is a different image-editing workload and is not a speed comparison.",
        "Image quality requires visual inspection; successful generation alone is not a quality score.",
    ]}
    try:
        catalog = host.json("/api/agent/loras")
        (args.output / "catalog.json").write_text(json.dumps(catalog, indent=2))
        check(catalog.get("loadedModelIsImage"), "Load a catalog image model first.")
        check("turbo" in catalog.get("loadedModel", ""), "Load the official Turbo catalog entry.")
        report["loadedModel"] = catalog["loadedModel"]
        report["connection"] = {key: value for key, value in json.loads(args.connection.read_text()).items()
                                if key in ("backend", "model", "loadSeconds", "catalogModel", "engine")}
        entries = {item["id"]: item for item in catalog["loras"]}
        for identifier, _ in CASES:
            check(identifier in entries and entries[identifier]["state"] == "Installed", f"Missing installed compatible LoRA: {identifier}")
        output_image = None
        cases = ([("baseline", CASES[0][1])] if args.baseline else []) + CASES
        for identifier, prompt, run in [(identifier, prompt, run) for identifier, prompt in cases
                                        for run in range(1, args.repetitions + 1)]:
            entry = entries.get(identifier)
            chosen = [{"id": identifier, "strength": entry["defaultStrength"]}] if entry else []
            label = f"{identifier}-run{run}"
            state = host.json("/api/agent/loras/choice", {"loras": chosen})
            check(state["chosen"] == chosen, "Choice route did not save the requested adapter/strength.")
            check(host.json("/api/agent/settings")["imageLoras"] == chosen, "Settings did not persist the selection.")
            refs = [output_image] if identifier == CASES[1][0] else []
            uploads = [host.upload(path) for path in refs]
            session = host.json("/api/sessions?conversation=new", {})
            message = {"role": "user", "content": prompt,
                       "stillImagePaths": [upload["file"] for upload in uploads],
                       "attachments": uploads}
            request = {"sessionId": session["sessionId"], "messages": [message]}
            (args.output / f"{label}-request.json").write_text(json.dumps(request, indent=2))
            started = time.monotonic()
            frames = []
            step_times = []
            with host.open("/api/chat", json.dumps(request).encode()) as response, (args.output / f"{label}.jsonl").open("w") as log:
                for line in response:
                    if line.startswith(b"data: "):
                        frame = json.loads(line[6:])
                        frames.append(frame)
                        if "image_step" in frame:
                            step_times.append(time.monotonic() - started)
                        # Keep metadata; previews are large and the completed image is saved separately.
                        log.write(json.dumps({key: value for key, value in frame.items() if key != "preview"}) + "\n")
                        log.flush()
            elapsed = time.monotonic() - started
            check(not [frame for frame in frames if frame.get("error")], "Chat returned an error; see frame log.")
            check(any(frame.get("done") and not frame.get("aborted") for frame in frames), "No completed image turn.")
            steps = [frame for frame in frames if "image_step" in frame]
            check(steps and steps[-1]["image_step"] == 8 and all(frame["image_steps"] == 8 for frame in steps), "Turbo must run its eight-step schedule.")
            check(all(frame.get("image_loras", []) == ([entry["name"]] if entry else []) for frame in steps),
                  "Generation did not report exactly the selected adapters.")
            final = next(frame for frame in frames if frame.get("imageUrl"))
            check(final["imageSources"] == [upload["file"] for upload in uploads], "Reference image order/count changed.")
            with host.open(final["imageUrl"]) as response:
                image = response.read()
            check(image[:8] == b"\x89PNG\r\n\x1a\n", "Final upload is not a PNG.")
            width, height = struct.unpack(">II", image[16:24])
            check((width, height) == (final["width"], final["height"]) and min(width, height) >= 256, "Image dimensions do not match the final frame.")
            image_path = args.output / f"{label}.png"
            image_path.write_bytes(image)
            # Every repeated edit uses the same source, not the previous edited result.
            if identifier == CASES[0][0]:
                output_image = image_path
            saved = host.json("/api/sessions?conversation=" + session["conversationId"], {})
            check(any(message.get("imageUrl") == final["imageUrl"] for message in saved["messages"]), "Saved conversation lost the image.")
            case = {"id": identifier, "run": run, "status": "passed", "seconds": elapsed,
                    "firstStepSeconds": step_times[0],
                    "subsequentStepMeanSeconds": (step_times[-1] - step_times[0]) / max(1, len(step_times) - 1),
                    "finalDecodeAndSaveSeconds": elapsed - step_times[-1],
                    "image": str(image_path.resolve()), "sha256": hashlib.sha256(image).hexdigest(),
                    "width": width, "height": height, "references": [upload["file"] for upload in uploads],
                    "steps": steps[-1]["image_step"], "conversationId": session["conversationId"]}
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
        report["timingSummary"] = {
            identifier: {"samples": len(rows), "firstSeconds": rows[0]["seconds"],
                         "medianSeconds": statistics.median(row["seconds"] for row in rows),
                         "warmMedianSeconds": statistics.median(row["seconds"] for row in rows[1:]) if len(rows) > 1 else None}
            for identifier, _ in cases
            for rows in [[row for row in report["cases"] if row["id"] == identifier]]
        }
        if args.baseline:
            baseline = next(row for row in report["cases"] if row["id"] == "baseline")
            casual = next(row for row in report["cases"] if row["id"] == CASES[0][0])
            check(baseline["sha256"] != casual["sha256"], "Enabling the style adapter did not change the fixed-seed image.")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["httpTimings"] = host.timings
        (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
