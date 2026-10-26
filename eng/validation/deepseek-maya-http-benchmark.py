#!/usr/bin/env python3
"""Retain comparable OpenAI streaming measurements from TensorSharp and Maya.

No third-party Python packages or model tokenizer are needed. Token throughput
requires the server's actual usage counters; SSE frame counts are never tokens.
Outputs must stay in this checkout's ignored artifacts/ or docs/validation/.
The smoke checks screen a few outputs; they do not qualify model quality.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
from pathlib import Path
import socket
import statistics
import sys
import threading
import time
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
LIMITATIONS = [
    "Client decode throughput uses SSE arrival times, which include network effects and can be distorted by buffered or batched tokens.",
    "SSE frame counts are delivery observations, not token counts; actual token counts come only from server usage.",
    "Prompt tokens / TTFT is a request-level latency ratio, not a pure prefill kernel measurement.",
    "Repeated identical requests can reuse prompt state; cached-token counters are preserved when reported, but missing counters do not prove no reuse.",
    "The bounded smoke checks do not establish broad model quality, numerical equivalence, or release qualification.",
    "This client does not prove which devices or nodes executed the request; preserve server logs and resource telemetry separately.",
]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(text):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Invalid JSON constant: " + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def default_cases(suite, max_tokens):
    smoke = [
        {"name": "arithmetic", "messages": [{"role": "user", "content":
            "What is 123 + 456? Reply with only the integer."}], "expected_text": "579"},
        {"name": "json_instruction", "messages": [{"role": "user", "content":
            'Return exactly this valid JSON object with no markdown or explanation: {"ready":true,"count":3}.'}],
            "expected_json": {"ready": True, "count": 3}},
        {"name": "extraction", "messages": [{"role": "user", "content":
            "Extract the access code from this record. Reply with only the code. Record: city=Oslo; access_code=R7M42; count=16."}],
            "expected_text": "R7M42"},
    ]
    throughput = [
        {"name": "short_generation", "messages": [{"role": "user", "content":
            "Write a detailed tutorial on implementing a hash table with collision handling, resizing, and deletion. Include examples and discuss correctness. Keep writing until the tutorial is complete."}]},
        {"name": "long_prompt_generation", "messages": [{"role": "user", "content":
            ("A queue preserves FIFO order. A stack preserves LIFO order. A hash table maps keys to values and resolves collisions. " * 32)
            + "\nUsing the notes above, write a detailed comparison of these three data structures, including worked examples, complexity, and applications. Keep writing until the comparison is complete."}]},
    ]
    cases = smoke if suite == "smoke" else throughput if suite == "throughput" else smoke + throughput
    for case in cases:
        case["max_tokens"] = min(max_tokens, 96) if case in smoke else max_tokens
    return cases


def validate_cases(cases):
    require(isinstance(cases, list) and cases, "Cases must be a nonempty JSON array")
    names = set()
    for case in cases:
        require(isinstance(case, dict), "Every case must be an object")
        name = case.get("name")
        require(isinstance(name, str) and name and name not in names, "Case names must be unique nonempty strings")
        names.add(name)
        require(isinstance(case.get("messages"), list) and case["messages"], "Case needs messages: " + name)
        require(not ("expected_text" in case and "expected_json" in case), "Choose one correctness expectation: " + name)
        if "expected_text" in case:
            require(isinstance(case["expected_text"], str), "expected_text must be a string")
        if "max_tokens" in case:
            require(type(case["max_tokens"]) is int and case["max_tokens"] > 0, "max_tokens must be positive")
        if "extra_body" in case:
            require(isinstance(case["extra_body"], dict), "extra_body must be an object")
    return cases


def request_body(case, model, max_tokens, extra_body):
    extra = {**extra_body, **case.get("extra_body", {})}
    require(not ({"model", "messages", "stream", "stream_options", "n"} & extra.keys()),
            "extra_body cannot override model, messages, stream, stream_options or n")
    return {"model": model, "messages": copy.deepcopy(case["messages"]),
            "max_tokens": case.get("max_tokens", max_tokens), "temperature": 0, "top_p": 1,
            **extra, "stream": True, "stream_options": {"include_usage": True}}


def body_identity(body):
    # Model aliases differ between servers. Every other submitted field matters.
    return digest(canonical({key: value for key, value in body.items() if key != "model"}))


def actual_count(usage, key):
    value = usage.get(key)
    return value if type(value) is int and value >= 0 else None


class StreamState:
    def __init__(self):
        self.events = []
        self.content = []
        self.reasoning = []
        self.tool_fragments = {}
        self.first_ms = None
        self.last_ms = None
        self.frames = 0
        self.finish_reason = None
        self.usage = {}
        self.server_timings = []
        self.done = False

    def add(self, event, elapsed_ms):
        self.events.append({"elapsed_ms": elapsed_ms, "data": event})
        require(not self.done, "Received an event after [DONE]")
        if event == "[DONE]":
            self.done = True
            return
        require(isinstance(event, dict), "SSE event must be a JSON object")
        require(not event.get("error"), "Server stream error: " + str(event.get("error")))
        if event.get("usage") is not None:
            require(isinstance(event["usage"], dict), "usage must be an object")
            self.usage.update(event["usage"])
        if "timings" in event:
            self.server_timings.append({"elapsed_ms": elapsed_ms, "timings": event["timings"]})
        choices = event.get("choices", [])
        require(isinstance(choices, list) and len(choices) <= 1, "Expected at most one completion choice")
        if not choices:
            return
        choice = choices[0]
        require(isinstance(choice, dict), "Completion choice must be an object")
        require(choice.get("index", 0) == 0, "Expected completion choice index 0")
        finish = choice.get("finish_reason")
        previous_finish = self.finish_reason
        if finish is not None:
            require(isinstance(finish, str) and finish, "Invalid finish reason")
            require(self.finish_reason in (None, finish), "Conflicting finish reasons")
            self.finish_reason = finish
        delta = choice.get("delta", {})
        require(isinstance(delta, dict), "Completion delta must be an object")
        generated = False
        for field, target in (("content", self.content), ("reasoning_content", self.reasoning)):
            piece = delta.get(field)
            if piece is not None:
                require(isinstance(piece, str), "Invalid text delta: " + field)
                if piece:
                    target.append(piece)
                    generated = True
        tools = delta.get("tool_calls") or []
        require(isinstance(tools, list), "Invalid tool_calls delta")
        for tool in tools:
            require(isinstance(tool, dict) and type(tool.get("index")) is int and tool["index"] >= 0,
                    "Tool delta needs a nonnegative integer index")
            entry = self.tool_fragments.setdefault(tool["index"], {"id": "", "type": "", "function": {"name": "", "arguments": ""}})
            for field in ("id", "type"):
                if tool.get(field):
                    require(isinstance(tool[field], str), "Invalid tool " + field)
                    # OpenAI sends these complete on the first fragment.
                    require(not entry[field] or entry[field] == tool[field], "Conflicting tool " + field)
                    entry[field] = tool[field]
            function = tool.get("function", {})
            require(isinstance(function, dict), "Invalid tool function delta")
            for field in ("name", "arguments"):
                if function.get(field):
                    require(isinstance(function[field], str), "Invalid tool function " + field)
                    entry["function"][field] += function[field]
                    generated = True
        if generated:
            require(previous_finish is None, "Generated content after finish reason")
            self.frames += 1
            if self.first_ms is None:
                self.first_ms = elapsed_ms
            self.last_ms = elapsed_ms

    def result(self):
        prompt = actual_count(self.usage, "prompt_tokens")
        completion = actual_count(self.usage, "completion_tokens")
        window_ms = None if self.first_ms is None else self.last_ms - self.first_ms
        decode = ((completion - 1) * 1000 / window_ms
                  if completion is not None and completion > 1 and window_ms is not None and window_ms > 0 else None)
        details = self.usage.get("prompt_tokens_details")
        cached = actual_count(details, "cached_tokens") if isinstance(details, dict) else None
        return {"events": self.events, "output_text": "".join(self.content), "reasoning_text": "".join(self.reasoning),
                "tool_calls": [self.tool_fragments[index] for index in sorted(self.tool_fragments)],
                "finish_reason": self.finish_reason, "done_received": self.done,
                "raw_usage": self.usage, "prompt_tokens": prompt, "completion_tokens": completion,
                "cached_prompt_tokens": cached, "usage_present": prompt is not None and completion is not None,
                "ttft_ms": self.first_ms, "last_output_frame_ms": self.last_ms,
                "output_frames": self.frames, "decode_window_ms": window_ms, "client_decode_tokens_per_second": decode,
                "prompt_tokens_per_ttft_second": prompt * 1000 / self.first_ms
                    if prompt is not None and self.first_ms is not None and self.first_ms > 0 else None,
                "server_timings": self.server_timings}

    def validate(self):
        result = self.result()
        require(self.done, "Stream ended without [DONE]")
        require(self.finish_reason is not None, "Stream ended without a finish reason")
        require(result["usage_present"], "Stream did not report actual integer prompt and completion token usage")
        require(result["completion_tokens"] == 0 or self.first_ms is not None,
                "Server reported generated tokens but delivered no output frames")
        require(result["completion_tokens"] != 0 or self.first_ms is None,
                "Server delivered output but reported zero generated tokens")
        require(self.frames <= result["completion_tokens"], "More generated output frames than reported completion tokens")


def sse_events(response, deadline, raw_log=None, started=None):
    """Yield full SSE data events, supporting CRLF and multiline data fields."""
    data = []
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Request exceeded its total deadline")
        raw = response.readline()
        if not raw:
            require(not data, "Stream ended in an unterminated SSE event")
            return
        if raw_log is not None:
            raw_log.append({"elapsed_ms": (time.perf_counter() - started) * 1000,
                            "line": raw.decode("utf-8", errors="replace").rstrip("\r\n")})
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                yield "[DONE]" if payload == "[DONE]" else strict_json(payload)
                data = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            data.append(value if separator else "")


class DeadlineConnection:
    """Enforce a whole-request deadline even if a server trickles SSE forever.

    HTTPResponse's socket file keeps the descriptor available while a
    close-delimited response is open. shutdown interrupts a blocked read.
    """
    def __init__(self, address, timeout, deadline=None):
        cls = http.client.HTTPSConnection if address.scheme == "https" else http.client.HTTPConnection
        self.connection = cls(address.hostname, address.port, timeout=timeout)
        self.timeout = timeout
        self.deadline = deadline if deadline is not None else time.monotonic() + timeout
        self.expired = threading.Event()
        self.guard = None
        self.timer = None

    def __enter__(self):
        try:
            self.connection.connect()
        except BaseException:
            self.connection.close()
            raise
        self.guard = self.connection.sock

        def expire():
            self.expired.set()
            try:
                self.guard.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self.connection.close()
            raise TimeoutError("Connection setup exceeded request deadline")
        self.timer = threading.Timer(remaining, expire)
        self.timer.daemon = True
        self.timer.start()
        return self

    def __exit__(self, *args):
        if self.timer:
            self.timer.cancel()
        self.connection.close()


def run_request(url, body, timeout, headers=None, evidence=None):
    result = evidence if evidence is not None else {}
    state = StreamState()
    started = time.perf_counter()
    deadline = time.monotonic() + timeout
    address = urlsplit(url)
    result.update(status="failed", request_body=body, request_body_sha256=digest(canonical(body)),
                  workload_sha256=body_identity(body), started_utc=datetime.now(timezone.utc).isoformat(), sse_lines=[])
    owned = None
    try:
        with DeadlineConnection(address, timeout, deadline) as owned:
            conn = owned.connection
            conn.request("POST", address.path.rstrip("/") + "/v1/chat/completions", canonical(body),
                         {"Content-Type": "application/json", "Accept": "text/event-stream", **(headers or {})})
            response = conn.getresponse()
            result.update(http_status=response.status, headers_ms=(time.perf_counter() - started) * 1000,
                          response_headers=dict(response.getheaders()))
            if response.status != 200:
                result["error_body"] = response.read(65536).decode("utf-8", errors="replace")
                raise ValueError("Chat returned HTTP " + str(response.status))
            require("text/event-stream" in response.getheader("Content-Type", "").lower(), "Chat did not return SSE")
            for event in sse_events(response, deadline, result["sse_lines"], started):
                if owned.expired.is_set():
                    raise TimeoutError("Request exceeded its total deadline")
                state.add(event, (time.perf_counter() - started) * 1000)
                if state.done:
                    break
            if owned.expired.is_set():
                raise TimeoutError("Request exceeded its total deadline")
            state.validate()
            result["status"] = "complete"
    except Exception as error:
        if owned is not None and owned.expired.is_set():
            error = TimeoutError("Request exceeded its total deadline")
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        result.update(state.result())
        result["total_wall_ms"] = (time.perf_counter() - started) * 1000
        count = result["completion_tokens"]
        result["end_to_end_tokens_per_second"] = count * 1000 / result["total_wall_ms"] if count is not None else None
    return result


def check_correctness(case, result):
    if "expected_text" not in case and "expected_json" not in case:
        return {"status": "not_evaluated", "detail": "Throughput prompt has no automated semantic oracle"}
    if result.get("status") != "complete":
        return {"status": "failed", "detail": "Request did not complete the streaming protocol"}
    if result.get("finish_reason") != "stop":
        return {"status": "failed", "detail": "Correctness check requires EOS/stop, not a capped continuation"}
    try:
        text = result["output_text"].strip()
        if "expected_text" in case:
            require(text == case["expected_text"], "Answer did not exactly match expected text")
        else:
            require(canonical(strict_json(text)) == canonical(case["expected_json"]), "Answer did not match expected JSON")
        return {"status": "passed", "detail": "Bounded exact-answer check passed"}
    except Exception as error:
        return {"status": "failed", "detail": str(error)}


def summarize(rows):
    measured = [row for row in rows if row["phase"] == "measured"]
    complete = [row for row in measured if row["status"] == "complete"]
    result = {"attempted": len(measured), "protocol_complete": len(complete), "failed": len(measured) - len(complete),
              "correctness": {state: sum(row["correctness"]["status"] == state for row in measured)
                              for state in ("passed", "failed", "not_evaluated")}}
    for key in ("ttft_ms", "total_wall_ms", "client_decode_tokens_per_second", "end_to_end_tokens_per_second"):
        values = [row[key] for row in complete if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
        result[key] = {"samples": len(values), "median": statistics.median(values) if values else None,
                       "min": min(values) if values else None, "max": max(values) if values else None}
    timed = [row for row in complete if row.get("client_decode_tokens_per_second") is not None]
    result["pooled_client_decode_tokens_per_second"] = (
        sum(row["completion_tokens"] - 1 for row in timed) * 1000 / sum(row["decode_window_ms"] for row in timed)
        if timed else None)
    result["prompt_token_counts"] = sorted(set(row["prompt_tokens"] for row in complete))
    result["completion_token_counts"] = sorted(set(row["completion_tokens"] for row in complete))
    result["finish_reasons"] = sorted(set(row["finish_reason"] for row in complete))
    return result


def compare_reports(current, baseline):
    problems = []
    for key in ("cases_sha256", "workload_identities", "concurrency", "warmups", "repeats", "case_order"):
        if current.get(key) != baseline.get(key):
            problems.append("Different " + key)
    cases = {}
    for name, now in current.get("summary_by_case", {}).items():
        old = baseline.get("summary_by_case", {}).get(name)
        if old is None:
            continue
        comparable = not problems
        reasons = []
        for summary, label in ((now, "current"), (old, "baseline")):
            if summary["failed"] or summary["correctness"]["failed"]:
                reasons.append(label + " has request/correctness failures")
        if now["prompt_token_counts"] != old["prompt_token_counts"]:
            reasons.append("different actual prompt-token counts (check prompt encoders)")
        if now["completion_token_counts"] != old["completion_token_counts"]:
            reasons.append("different actual completion-token counts")
        if now["finish_reasons"] != old["finish_reasons"]:
            reasons.append("different finish reasons")
        ratios = {}
        for key in ("ttft_ms", "total_wall_ms", "client_decode_tokens_per_second", "end_to_end_tokens_per_second"):
            current_median, baseline_median = now[key]["median"], old[key]["median"]
            ratios[key + "_current_over_baseline"] = (
                current_median / baseline_median if current_median is not None and baseline_median else None)
        cases[name] = {"comparable_workload": comparable and not reasons, "limitations": reasons, **ratios}
    return {"global_mismatches": problems, "cases": cases,
            "performance_qualified": False,
            "qualification_note": "Ratios are observations. Review model hashes, hardware, cache, correctness and independent device/node engagement evidence before drawing conclusions."}


def output_path(value):
    path = Path(value).resolve()
    require(any(path.is_relative_to((ROOT / folder).resolve()) for folder in ("artifacts", "docs/validation")),
            "Output must remain in this checkout's ignored artifacts/ or docs/validation/")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "Output already exists; choose a new evidence file")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Server base URL, without /v1")
    parser.add_argument("--model", required=True, help="Exact served model alias")
    parser.add_argument("--engine", required=True, help="Engine and configuration label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "throughput", "all"), default="all")
    parser.add_argument("--cases", type=Path, help="Optional case JSON array, replacing the built-in suite")
    parser.add_argument("--extra-body", default="{}", help="JSON object of explicit sampling/thinking/cache options")
    parser.add_argument("--extra-body-file", type=Path)
    parser.add_argument("--provenance", type=Path, help="JSON containing model hashes, revisions, launch, hardware and topology")
    parser.add_argument("--compare-with", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1, help="Untimed-in-summary requests per case and worker")
    parser.add_argument("--repeats", type=int, default=3, help="Measured batches per case")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800, help="Whole-request deadline, seconds")
    parser.add_argument("--api-key-env", help="Read a bearer token from this environment variable; never saved")
    args = parser.parse_args(argv)
    require(args.max_tokens > 0 and args.repeats > 0 and args.warmups >= 0 and args.concurrency > 0
            and math.isfinite(args.timeout) and args.timeout > 0,
            "Token/repeat/concurrency/timeout limits must be positive; warmups may be zero")
    address = urlsplit(args.url)
    require(address.scheme in ("http", "https") and address.hostname and not address.username
            and not address.password and not address.query and not address.fragment, "Invalid server base URL")
    require(not address.path.rstrip("/").endswith("/v1"), "--url is the server base URL, without /v1")
    extra = strict_json(args.extra_body_file.read_text(encoding="utf-8") if args.extra_body_file else args.extra_body)
    require(isinstance(extra, dict), "extra-body must be a JSON object")
    cases = validate_cases(strict_json(args.cases.read_text(encoding="utf-8")) if args.cases
                           else default_cases(args.suite, args.max_tokens))
    bodies = {case["name"]: request_body(case, args.model, args.max_tokens, extra) for case in cases}
    output = output_path(args.output)
    headers = {}
    if args.api_key_env:
        import os
        key = os.environ.get(args.api_key_env)
        require(bool(key), "API key environment variable is empty")
        headers["Authorization"] = "Bearer " + key
    provenance = strict_json(args.provenance.read_text(encoding="utf-8")) if args.provenance else None
    report = {"schema_version": 1, "engine": args.engine, "url": args.url, "model": args.model,
              "started_utc": datetime.now(timezone.utc).isoformat(), "runner_sha256": digest(Path(__file__).read_bytes()),
              "python_version": sys.version, "provenance": provenance,
              "provenance_sha256": digest(args.provenance.read_bytes()) if args.provenance else None,
              "cases": cases, "cases_sha256": digest(canonical(cases)),
              "workload_identities": {name: body_identity(body) for name, body in bodies.items()},
              "warmups": args.warmups, "repeats": args.repeats, "concurrency": args.concurrency,
              "timeout_seconds": args.timeout, "case_order": [case["name"] for case in cases],
              "cache_policy": "Repeated identical request bodies; no per-request nonce or automatic cache reset",
              "limitations": LIMITATIONS, "performance_qualified": False, "quality_qualified": False,
              "status": "running", "requests": []}

    def save():
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)

    save()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for case in cases:
                for phase, count in (("warmup", args.warmups), ("measured", args.repeats)):
                    for repeat in range(count):
                        barrier = threading.Barrier(args.concurrency)

                        def run(worker):
                            barrier.wait(timeout=30)
                            result = run_request(args.url, bodies[case["name"]], args.timeout, headers)
                            result.update(case=case["name"], phase=phase, repeat=repeat, worker=worker)
                            result["correctness"] = check_correctness(case, result)
                            return result

                        futures = [pool.submit(run, worker) for worker in range(args.concurrency)]
                        batch = [future.result() for future in futures]
                        report["requests"].extend(batch)
                        save()
                        states = ",".join(row["status"] for row in batch)
                        print(f"{args.engine} {case['name']} {phase} {repeat + 1}/{count}: {states}", flush=True)
        report["summary_by_case"] = {case["name"]: summarize([row for row in report["requests"] if row["case"] == case["name"]])
                                     for case in cases}
        all_ok = all(row["status"] == "complete" and row["correctness"]["status"] != "failed" for row in report["requests"])
        report["status"] = "completed_bounded_checks_passed" if all_ok else "completed_with_failures"
        if args.compare_with:
            baseline = strict_json(args.compare_with.read_text(encoding="utf-8"))
            report["comparison"] = compare_reports(report, baseline)
            report["comparison_baseline_sha256"] = digest(args.compare_with.read_bytes())
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(str(output), flush=True)
        return 0 if all_ok else 1
    except BaseException as error:
        report.update(status="interrupted_or_failed", error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
