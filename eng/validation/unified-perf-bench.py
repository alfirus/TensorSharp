#!/usr/bin/env python3
"""Sequential prefill/decode throughput bench for one owned TensorSharp server.

Starts TensorSharp.Server.Host (optionally with --unified-memory), sends a fixed
plan of requests to /api/chat/ollama one at a time, and records the server's own
attributed timings (prompt_eval_* / eval_*), cache hits, wall time, process
memory and board VRAM. Requests never overlap, so GPU work is never concurrent.

Prefill tok/s is only reported for requests with zero prompt-cache hits; decode
tok/s is eval_count / eval_duration as the server reports it (it includes the
final EOS forward). Results go to <output>/report.json; keep outputs under the
ignored artifacts/ directory.

Example (unified Qwen3.8 Flash Next, no prefix cache, fast startup):
  python eng/validation/unified-perf-bench.py --unified-memory \
      --model C:/Works/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf \
      --output artifacts/perf/qwen-baseline-v1
"""
import argparse
import hashlib
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # memory sampling is optional
    psutil = None

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER = ROOT / "TensorSharp.Server.Host" / "bin" / "TensorSharp.Server.Host.exe"

# Distinct natural/technical text windows so each long prompt routes to a
# realistic, different expert mix (repeated synthetic text inflates hit rates).
SOURCES = {
    "corpus": ROOT / "benchmarks" / "engine_comparison" / "assets" / "prefill_corpus.txt",
    "paged": ROOT / "docs" / "PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md",
    "spec": ROOT / "docs" / "speculative_decoding.md",
}

PLANS = {
    # name, source, start char, length chars, instruction, max new tokens
    "default": [
        ("prefill-a", "corpus", 0, 9000, "Summarize the document above in five sentences.", 64),
        ("prefill-b", "paged", 2000, 9000, "Summarize the document above in five sentences.", 64),
        ("decode-a", None, 0, 0, "Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall.", 256),
        ("prefill-c", "spec", 4000, 9000, "Summarize the document above in five sentences.", 64),
        ("decode-b", None, 0, 0, "Explain in detail how a modern CPU executes instructions, covering pipelining, caches and branch prediction.", 256),
    ],
    "quick": [
        ("prefill-a", "corpus", 0, 9000, "Summarize the document above in five sentences.", 32),
        ("decode-a", None, 0, 0, "Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall.", 128),
    ],
    "prefill-c": [
        ("prefill-c", "spec", 4000, 9000, "Summarize the document above in five sentences.", 8),
    ],
    "decode": [
        ("decode-a", None, 0, 0, "Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall.", 256),
    ],
    "long": [
        ("prefill-4k", "spec", 0, 18000, "Summarize the document above in five sentences.", 64),
        ("decode-a", None, 0, 0, "Write a long, detailed essay about the history of the Roman Empire, from its founding to its fall.", 256),
    ],
}


def prompt_text(source, start, length, instruction):
    if source is None:
        return instruction
    text = SOURCES[source].read_text(encoding="utf-8", errors="replace")[start:start + length]
    return f"Document:\n\n{text}\n\n{instruction}"


def http_json(port, method, path, body, timeout):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {raw[:400]}")
        return json.loads(raw)
    finally:
        connection.close()


def board_vram_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:
        return None


class Sampler:
    """Samples process working set/private bytes every 0.5 s (peaks can be missed)."""

    def __init__(self):
        self.process, self.peak_rss, self.peak_private = None, 0, 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.wait(0.5):
            try:
                info = self.process.memory_info()
                self.peak_rss = max(self.peak_rss, info.rss)
                self.peak_private = max(self.peak_private, getattr(info, "private", 0) or 0)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", default="none")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=52190)
    parser.add_argument("--plan", choices=sorted(PLANS), default="default")
    parser.add_argument("--unified-memory", action="store_true")
    parser.add_argument("--prefix-cache", action="store_true", help="keep the prefix cache (and its startup warmup) on")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="run the whole plan N times in the same server")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--server-arg", action="append", default=[], help="extra server argument (repeatable)")
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE for the server process (repeatable)")
    args = parser.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit("Port is occupied; refusing to touch another server")

    env = dict(os.environ)
    settings = {}
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
        settings[key] = value
    command = [str(args.server), "--model", str(args.model), "--mmproj", args.mmproj, "--backend", "ggml_cuda",
               "--host", "127.0.0.1", "--port", str(args.port), "--no-spec", "--no-multi-agent",
               "--temperature", "0", "--top-k", "0", "--top-p", "1", "--repeat-penalty", "1", "--seed", "17"]
    if not args.prefix_cache:
        command.append("--no-prefix-cache")
    if args.unified_memory:
        command.append("--unified-memory")
    command += args.server_arg

    report = dict(command=command, environment=settings, plan=args.plan, records=[], started_unix=time.time())

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sampler = Sampler()
    server = None
    try:
        report["vram_before_mib"] = board_vram_mib()
        with (out / "server.log").open("w", encoding="utf-8", errors="replace") as log:
            started = time.monotonic()
            server = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            if psutil:
                sampler.process = psutil.Process(server.pid)
                sampler.thread.start()
            deadline = started + args.timeout
            model_id = None
            while model_id is None:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited during startup (code {server.returncode})")
                try:
                    models = http_json(args.port, "GET", "/v1/models", None, 5).get("data", [])
                    model_id = models[0]["id"] if models else None
                except (OSError, ValueError, RuntimeError):
                    pass
                if model_id is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError("startup deadline exceeded")
                    time.sleep(1)
            report["startup_s"] = round(time.monotonic() - started, 3)
            report["vram_after_load_mib"] = board_vram_mib()
            print(f"ready in {report['startup_s']} s", flush=True)
            save()
            for round_index in range(args.repeat):
                for name, source, start, length, instruction, max_new in PLANS[args.plan]:
                    body = dict(model=model_id, stream=False, think=args.think, skills=[], skills_discovery=False, multi_agent=False,
                                messages=[dict(role="user", content=prompt_text(source, start, length, instruction))],
                                options=dict(temperature=0, top_k=0, top_p=1, min_p=0, repeat_penalty=1, seed=17,
                                             num_predict=max_new, stop=[]))
                    row = dict(name=name, round=round_index)
                    t0 = time.monotonic()
                    try:
                        response = http_json(args.port, "POST", "/api/chat/ollama", body, args.timeout)
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                        report["records"].append(row)
                        save()
                        raise
                    row["wall_s"] = round(time.monotonic() - t0, 3)
                    prompt_tokens = response.get("prompt_eval_count") or 0
                    hits = response.get("prompt_cache_hit_tokens") or 0
                    prompt_ns = response.get("prompt_eval_duration") or 0
                    eval_tokens = response.get("eval_count") or 0
                    eval_ns = response.get("eval_duration") or 0
                    row.update(prompt_tokens=prompt_tokens, cache_hits=hits, prompt_s=round(prompt_ns / 1e9, 3),
                               eval_tokens=eval_tokens, eval_s=round(eval_ns / 1e9, 3),
                               done_reason=response.get("done_reason"),
                               content_head=(response.get("message") or {}).get("content", "")[:240],
                               content_sha1=hashlib.sha1(((response.get("message") or {}).get("content", ""))
                                                         .encode("utf-8")).hexdigest())
                    if prompt_ns and hits == 0:
                        row["prefill_tps"] = round(prompt_tokens / (prompt_ns / 1e9), 2)
                    if eval_ns and eval_tokens:
                        row["decode_tps"] = round(eval_tokens / (eval_ns / 1e9), 2)
                    row["vram_mib"] = board_vram_mib()
                    report["records"].append(row)
                    save()
                    print(json.dumps({k: row.get(k) for k in ("name", "round", "prompt_tokens", "cache_hits", "prompt_s", "prefill_tps",
                                                              "eval_tokens", "eval_s", "decode_tps", "wall_s", "done_reason")}), flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], flush=True)
    finally:
        sampler.stop.set()
        if server is not None:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
            report["server_exit"] = server.returncode
        report["peak_rss_gib"] = round(sampler.peak_rss / 2**30, 3)
        report["peak_private_gib"] = round(sampler.peak_private / 2**30, 3)
        report["finished_unix"] = time.time()
        save()
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    sys.exit(main())
