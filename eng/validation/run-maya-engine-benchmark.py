#!/usr/bin/env python3
"""Measure an unchanged maya-ds4 executable with shared, pretokenized cases.

Case file: {"cases": [{"name": "short", "ids": [0, 42], "max_new": 64}]}
The same IDs can be used by a TensorSharp token replay probe. Native engine
timings and client wall timings are reported separately. An incomplete case or
protocol error makes the command fail; no skipped case counts as a pass.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_done(line: str, observed_tokens: list[int]) -> dict:
    fields = line.split()
    if len(fields) != 11 or fields[0] != "DONE":
        raise ValueError(f"Unexpected Maya DONE record: {line!r}")
    generated, prompt = int(fields[1]), int(fields[2])
    prompt_ms, decode_ms = float(fields[3]), float(fields[4])
    if generated != len(observed_tokens):
        raise ValueError(f"Maya generated {generated} but emitted {len(observed_tokens)} tokens")
    if generated < 1 or prompt < 1 or min(prompt_ms, decode_ms) < 0:
        raise ValueError("Maya reported an empty request or a negative timing")
    if not math.isfinite(prompt_ms) or not math.isfinite(decode_ms):
        raise ValueError("Maya reported nonfinite timing")
    if fields[5] not in {"stop", "length"}:
        raise ValueError(f"Maya request did not complete: {fields[5]}")
    reused = int(fields[8])
    if reused < 0 or reused > prompt:
        raise ValueError("Maya reported an invalid reused prompt count")
    return {
        "generated_tokens": generated,
        "prompt_tokens": prompt,
        "prefill_ms": prompt_ms,
        "decode_ms": decode_ms,
        "finish_reason": fields[5],
        "draft_accepted": int(fields[6]),
        "draft_offered": int(fields[7]),
        "reused_prompt_tokens": reused,
        "expert_vram_hits": int(fields[9]),
        "expert_lookups": int(fields[10]),
        "native_decode_tokens_per_second": generated * 1000 / decode_ms if decode_ms else None,
        "native_fresh_prefill_tokens_per_second": (prompt - reused) * 1000 / prompt_ms if prompt_ms else None,
    }


class Engine:
    def __init__(self, command: list[str], log_path: Path, timeout: float,
                 capture_prefix: Path | None = None, vocab: int | None = None):
        self.request_index = 0
        self.capture_prefix = capture_prefix
        self.vocab = vocab
        self.log = log_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, text=True, bufsize=1)
        except Exception:
            self.log.close()
            raise
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.timeout = timeout
        def read():
            try:
                for line in self.process.stdout:
                    self.lines.put(line.rstrip("\r\n"))
            finally:
                self.lines.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def line(self, deadline: float) -> str:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("Maya exceeded the request deadline")
        try:
            line = self.lines.get(timeout=remaining)
        except queue.Empty as error:
            raise TimeoutError("Maya exceeded the request deadline") from error
        if line is None:
            raise RuntimeError(f"Maya stdout closed (process exit {self.process.poll()})")
        if line.startswith("ERR"):
            raise RuntimeError(line)
        return line

    def ready(self) -> float:
        start = time.perf_counter()
        deadline = start + self.timeout
        while not self.line(deadline).startswith("READY "):
            pass
        return time.perf_counter() - start

    def generate(self, case: dict) -> dict:
        if "forced_ids" in case:
            raise ValueError("The stock Maya GEN protocol does not support teacher-forced output IDs")
        ids = case["ids"]
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError(f"{case['name']}: ids must be nonempty, nonnegative integer token IDs")
        max_new = case.get("max_new", 64)
        if type(max_new) is not int or max_new < 1:
            raise ValueError(f"{case['name']}: max_new must be positive")
        self.request_index += 1
        start = time.perf_counter()
        self.process.stdin.write(f"GEN {max_new} temperature=0 spec=0 " + ",".join(map(str, ids)) + "\n")
        self.process.stdin.flush()
        first = None
        last = None
        tokens: list[int] = []
        while True:
            line = self.line(start + self.timeout)
            if line.startswith("T "):
                last = time.perf_counter()
                if first is None:
                    first = last
                tokens.append(int(line.split()[1]))
            elif line.startswith("DONE "):
                result = parse_done(line, tokens)
                if result["prompt_tokens"] != len(ids):
                    raise ValueError(f"{case['name']}: prompt token count differs from input")
                if result["generated_tokens"] > max_new:
                    raise ValueError(f"{case['name']}: generated more than max_new")
                if result["finish_reason"] == "length" and result["generated_tokens"] != max_new:
                    raise ValueError(f"{case['name']}: truncated generation without a stop token")
                if result["draft_offered"] or result["draft_accepted"]:
                    raise ValueError("Maya drafted despite the exact non-speculative baseline request")
                wall = time.perf_counter() - start
                report = {"name": case["name"], "max_new": max_new, "wall_seconds": wall,
                        "request_index": self.request_index,
                        "input_ids": ids,
                        "time_to_first_token_ms": (first - start) * 1000 if first else None,
                        "wall_tokens_per_second": len(tokens) / wall,
                        "stream_decode_tokens_per_second": ((len(tokens) - 1) / (last - first)
                                                             if len(tokens) > 1 and last > first else None),
                        "output_token_ids": tokens, "done_record": line, **result}
                if self.capture_prefix is not None:
                    path = Path(str(self.capture_prefix) + f".{self.request_index}.f32")
                    expected = len(tokens) * self.vocab * 4
                    if not path.is_file() or path.stat().st_size != expected:
                        raise RuntimeError(f"Maya logit capture has missing/incomplete rows: {path}")
                    report["logits_capture"] = {"path": str(path), "row_count": len(tokens), "vocab": self.vocab,
                                                "request_index": self.request_index, "sha256": sha256_file(path),
                                                "capture_excluded_from_performance": True,
                                                "dtype": "float32", "byte_order": "little",
                                                "layout": "row i predicts output_token_ids[i] after prompt + output[:i]"}
                return report

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait()
        self.reader.join(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            try:
                stream.close()
            except OSError:
                pass
        self.log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", action="store_true", help="Run every case once before measured suites")
    parser.add_argument("--warmups", type=int, default=0, help="Number of complete unmeasured warmup suites")
    parser.add_argument("--capture-logits", type=Path, help="Quality-only: DS4_SERVE_DUMP filename prefix")
    parser.add_argument("--seed-profile", type=Path,
                        help="Copy Maya's shipped expert-routing profile into a fresh pack, as its installer does")
    parser.add_argument("--reset-between-cases", action="store_true",
                        help="Run an unmeasured different-prefix token before each case to force fresh KV")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--engine-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0 or args.warmups < 0:
        parser.error("--repeats/--timeout must be positive and --warmups nonnegative")
    warmups = max(args.warmups, int(args.warmup))
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if not cases:
        parser.error("The case file contains no cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile = args.pack / "expert-profile.bin"
    seeded = False
    if args.seed_profile and not profile.exists():
        geometry = dict(line.split(" ", 1) for line in (args.pack / "ds4.txt").read_text().splitlines() if " " in line)
        expected = int(geometry["num_hidden_layers"]) * int(geometry["n_routed_experts"]) * 4
        if args.seed_profile.stat().st_size != expected:
            parser.error("The shipped expert profile does not match this pack's geometry")
        shutil.copyfile(args.seed_profile, profile)
        seeded = True
    for setting in ("DISK_SKIP_W", "RAM_SKIP_W", "DISK_SKIP_AUTO", "PROMPT_SKIP"):
        # Explicit exact baseline controls. These are recorded in the report;
        # an inherited tuning environment cannot silently skip model work.
        os.environ["STRATA_DS4_" + setting] = "0"
        os.environ["MAYA_DS4_" + setting] = "0"
    capture_prefix = None
    vocab = None
    if args.capture_logits:
        capture_prefix = args.capture_logits.resolve()
        capture_prefix.parent.mkdir(parents=True, exist_ok=True)
        os.environ["DS4_SERVE_DUMP"] = str(capture_prefix)
        geometry = dict(line.split(" ", 1) for line in (args.pack / "ds4.txt").read_text().splitlines() if " " in line)
        vocab = int(geometry["vocab_size"])
        if vocab < 1:
            parser.error("The pack has an invalid vocabulary size")
    elif os.environ.get("DS4_SERVE_DUMP"):
        parser.error("Use --capture-logits to label a logit-capturing run as quality-only")
    if args.reset_between_cases:
        # Maya's unmodified protocol has no reset command. Disable saved slots
        # so a different-prefix guard followed by the real case forces reset.
        os.environ["STRATA_DS4_SLOT_KEEP"] = "0"
        os.environ["MAYA_DS4_SLOT_KEEP"] = "0"
    command = [str(args.exe.resolve()), "--pack", str(args.pack.resolve()), "--serve",
               "--dspark", "none", "--vision", "off", *args.engine_args]
    report = {"schema": 1, "engine": "project-maya", "source_revision": args.source_revision,
              "mode": "quality_only" if capture_prefix else "performance",
              "capture_excluded_from_performance": capture_prefix is not None,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "command": command,
              "environment": {k: v for k, v in sorted(os.environ.items())
                              if k.startswith(("DS4_", "MAYA_DS4_", "STRATA_DS4_"))},
              "case_file": str(args.cases.resolve()),
              "case_file_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
              "expert_profile": {"path": str(profile.resolve()), "seeded": seeded,
                                 "initial_sha256": sha256_file(profile) if profile.exists() else None},
              "results": [], "warmup": [], "status": "running"}
    engine = None
    try:
        engine = Engine(command, args.output.with_suffix(".engine.log"), args.timeout, capture_prefix, vocab)
        report["load_seconds"] = engine.ready()
        for repeat in range(-warmups, args.repeats):
            for case in cases:
                if args.reset_between_cases:
                    first_id = case["ids"][0]
                    engine.generate({"name": "reset_guard", "ids": [1 if first_id == 0 else 0], "max_new": 1})
                result = {"repeat": repeat, "warmup": repeat < 0, **engine.generate(case)}
                if args.reset_between_cases and result["reused_prompt_tokens"] != 0:
                    raise RuntimeError("Maya reused prompt tokens despite the reset guard")
                report["warmup" if repeat < 0 else "results"].append(result)
                print(json.dumps(result), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["status"] = "completed"
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], flush=True)
        return 1
    finally:
        if engine:
            engine.close()
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
