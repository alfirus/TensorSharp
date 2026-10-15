#!/usr/bin/env python3
"""Run a deterministic three-turn real-model interview through the OpenAI API."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path


def fetch(base, path, payload=None):
    request = urllib.request.Request(base.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18592")
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=Path("InferenceWeb.Tests/Fixtures/TopSkills/interview.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-skill-rounds", type=int, default=24, help="Record the prestarted server's actual round cap")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(args.fixture.read_text())
    if len(prompts) != 3:
        parser.error("The interview fixture must contain exactly three user turns; no-op validation cannot pass.")
    model = fetch(args.base, "/v1/models")["data"][0]["id"]
    messages, rounds = [], []
    for index, prompt in enumerate(prompts, start=1):
        messages.append({"role": "user", "content": prompt})
        start = time.perf_counter()
        response = fetch(args.base, "/v1/chat/completions", {"model": model,
            "messages": messages, "skills": ["grill-me"], "skills_discovery": False,
            "temperature": 0, "seed": 42, "think": False, "max_tokens": 1536, "stream": False})
        elapsed = time.perf_counter() - start
        (args.output / f"round-{index}.json").write_text(json.dumps(response, indent=2))
        answer = response["choices"][0]["message"].get("content") or ""
        messages.append({"role": "assistant", "content": answer})
        checks = {"nonempty": bool(answer.strip()), "finished": response["choices"][0]["finish_reason"] != "length",
            "no_raw_or_unhandled_tools": not response["choices"][0]["message"].get("tool_calls") and not re.search(r"(?:^|\n)\s*call:[\w-]+\{|<\|tool_call\|>", answer)}
        if index < 3:
            checks.update(asks_questions="?" in answer,
                recommends=bool(re.search(r"recommend|suggest|➡", answer, re.I)))
        if index == 2:
            checks.update(asks_recovery="recover" in answer.lower(), asks_pairing=bool(re.search(r"pair|identity|device", answer, re.I)),
                settled_topics_not_reasked=not re.search(r"\*\*(?:Q\d+[^\n]*[-:]\s*)?(?:Offline Changes|Conflict Resolution|Encryption Level)\*\*", answer, re.I))
        if index == 3:
            checks.update(stops_questions="?" not in answer, gives_concrete_plan=len(answer) >= 300,
                preserves_no_cloud="cloud" in answer.lower(), preserves_manual_conflicts="manual" in answer.lower(),
                preserves_recovery="backup" in answer.lower(), preserves_pairing="fingerprint" in answer.lower())
            log = args.server_log.read_text(errors="replace")
            marker = "/v1/chat/completions request: id=" + response["id"]
            section = log.split(marker, 1)[-1].split("/v1/chat/completions request: id=", 1)[0] if marker in log else ""
            checks["no_implementation_tools"] = bool(section) and not re.search(r"skills\.tool[^\n]*tool=(?:shell|write_file|apply_patch)\b", section)
        rounds.append({"round": index, "elapsed_seconds": elapsed, "usage": response.get("usage"), "checks": checks, "all_checks_pass": all(checks.values())})
        print(json.dumps(rounds[-1]), flush=True)
    report = {"model": model, "model_sha256": digest(args.model_file), "binary_sha256": digest(args.binary),
        "backend": "ggml_cpu", "context_tokens": 16384, "max_tokens_per_round": 1536, "max_skill_rounds": args.max_skill_rounds,
        "temperature": 0, "seed": 42, "rounds": rounds,
        "limitations": ["One fixed three-turn conversation; no broad model-quality or baseline-superiority claim.", "Checks are bounded behavioral oracles; responses are retained for review.", "No physical phone/device execution is measured."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(row["all_checks_pass"] for row in rounds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
