#!/usr/bin/env python3
"""EmbeddingGemma 2 real CLI/projector and OpenAI/Ollama media regression.

Uses the standard library and a probe-format manifest. Runs one owned server;
records exact media/model/projector/binary hashes, response vectors and timings.
This is protocol and lifecycle validation, not a quality leaderboard or timing
comparison. Generated reports belong in ignored docs/validation/ or artifacts/.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddinggemma2_evidence import command_artifacts
import embeddinggemma2_http_environment as child_environment


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", type=json.loads, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18380")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-backend", required=True)
    parser.add_argument("--environment-note", action="append", default=[])
    parser.add_argument("--require-embedding-default", action="store_true",
        help="Require a direct frozen host launch with known optimization override variables unset.")
    args = parser.parse_args()
    temporary = args.output.resolve() / "requests-temp"
    inherited = dict(os.environ, TMPDIR=str(temporary) + os.sep, TMP=str(temporary), TEMP=str(temporary))
    # Resolve/guard before file reads, server logs or process creation. Popen
    # receives these exact audited arguments/environment, with env prefixes
    # flattened rather than launching another environment-mutating wrapper.
    launch = child_environment.audit_child_environment(args.command, inherited, args.require_embedding_default)
    spec = importlib.util.spec_from_file_location("embedding_bench", ROOT / "benchmarks/EmbeddingBench/embedding_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    bench.KEEP_ALIVE = True
    args.output.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=True)
    cases = json.loads(args.cases.read_text())["cases"]
    if not cases:
        raise ValueError("The manifest must contain cases")
    payloads, media = [], {}
    for case in cases:
        content = []
        for part in case["parts"]:
            kind, value = part["kind"], part["value"]
            if kind == "text":
                content.append(dict(type="text", text=value))
                continue
            path = Path(value)
            if not path.is_absolute():
                path = args.cases.parent / path
            path = path.resolve()
            media[str(path)] = fingerprint(path)
            encoded = base64.b64encode(path.read_bytes()).decode()
            if kind == "image":
                mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[path.suffix.lower()]
                content.append(dict(type="image_url", image_url=dict(url=f"data:{mime};base64,{encoded}")))
            elif kind == "audio":
                content.append(dict(type="input_audio", input_audio=dict(format=path.suffix[1:], data=encoded)))
            elif kind == "video":
                mime = {".mp4": "video/mp4", ".webm": "video/webm"}[path.suffix.lower()]
                content.append(dict(type="input_video", input_video=dict(url=f"data:{mime};base64,{encoded}")))
            else:
                raise ValueError("Unsupported manifest part " + kind)
        payloads.append(dict(content=content))
    result = dict(command=launch.reported_command, expected_backend=args.expected_backend, model=fingerprint(args.model_file),
        projector=fingerprint(args.mmproj), manifest=fingerprint(args.cases), media=list(media.values()),
        binary_artifacts=command_artifacts(bench.command_artifacts, launch.argv) if launch.resolved_server_environment else [], cases=[], errors=[],
        environment_notes=args.environment_note, completed=False, passed=False,
        environment_audit=launch.metadata(), require_embedding_default=args.require_embedding_default,
        audio_log_policy=launch.audio_log_policy(),
        validation_tools=[fingerprint(Path(__file__).resolve()), fingerprint(Path(child_environment.__file__).resolve())],
        limitations=["Synthetic fixture protocol and preprocessing checks; no independent model quality or performance claim.",
                     "Each modality/backend is tested only when explicitly present in this report."])
    report = args.output / "http-media.json"

    def save():
        report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    def no_request_files():
        leftovers = sorted(str(path) for path in temporary.glob("tensorsharp-embedding-*"))
        assert not leftovers, ("Request temporary media was not cleaned up", leftovers)

    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(launch.argv, stdout=log, stderr=subprocess.STDOUT, env=launch.environment, start_new_session=True)
        try:
            started = time.monotonic()
            while time.monotonic() - started < 180:
                if process.poll() is not None:
                    raise RuntimeError("Server exited before startup; inspect server.log")
                try:
                    if bench.request(args.url, "/v1/models")[0] == 200:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError("Server startup")
            status, discovery = bench.request(args.url, "/api/models")
            assert status == 200, discovery
            result["discovery"] = discovery
            assert discovery["loadedBackend"] == args.expected_backend, discovery
            for case, payload in zip(cases, payloads):
                started = time.perf_counter()
                vectors, usage = bench.embedding(args.url, args.model, payload)
                elapsed = (time.perf_counter() - started) * 1000
                bench.assert_vectors(vectors, 768)
                assert len(vectors) == 1 and usage["prompt_tokens"] > 0, usage
                status, ollama = bench.request(args.url, "/api/embed", dict(model=args.model, input=payload))
                assert status == 200, (status, ollama)
                bench.assert_vectors(ollama["embeddings"], 768)
                assert len(ollama["embeddings"]) == 1, ollama
                similarity = bench.cosine(vectors[0], ollama["embeddings"][0])
                assert similarity > .99999, (case["name"], similarity)
                assert ollama["prompt_eval_count"] == usage["prompt_tokens"], (usage, ollama)
                result["cases"].append(dict(name=case["name"], parts=case["parts"], embedding=vectors[0],
                    ollama_embedding=ollama["embeddings"][0], tokens=usage["prompt_tokens"], protocol_cosine=similarity,
                    first_request_ms=elapsed))
                no_request_files()
                save()
            # One image verifies media + base64 output + MRL dimensions together.
            index = next(i for i, case in enumerate(cases) if any(part["kind"] == "image" for part in case["parts"]))
            status, encoded = bench.request(args.url, "/v1/embeddings", dict(model=args.model, input=payloads[index], dimensions=256, encoding_format="base64"))
            assert status == 200, (status, encoded)
            raw = base64.b64decode(encoded["data"][0]["embedding"], validate=True)
            assert len(raw) == 256 * 4, len(raw)
            vector = list(struct.unpack("<256f", raw))
            bench.assert_vectors([vector], 256)
            similarity = bench.cosine(vector, result["cases"][index]["embedding"][:256])
            assert similarity > .99999, similarity
            result["base64_mrl"] = dict(dimensions=256, cosine=similarity)
            invalid = [dict(content=[]), dict(content=[{}]), dict(content=[dict(type="invalid")]),
                dict(content=[dict(type="text", text=123)]),
                dict(content=[dict(type="image_url", image_url=dict(url="https://example.com/image.png"))]),
                dict(content=[dict(type="image_url", image_url=dict(url="data:image/png;base64,!!!"))]),
                dict(content=[dict(type="image_url", image_url=dict(url="data:image/png;base64,"))]),
                dict(content=[dict(type="image_url", image_url=dict(url="data:text/plain;base64,AQID"))]),
                dict(content=[dict(type="input_audio", input_audio=dict(format="invalid", data="AQID"))]),
                dict(content=[dict(type="input_audio", input_audio=dict(format="wav", data="AQID"))]),
                dict(content=[dict(type="input_video", input_video=dict(url="data:video/mp4;base64,AQID"))])]
            for endpoint in ("/v1/embeddings", "/api/embed"):
                for payload in invalid:
                    status, response = bench.request(args.url, endpoint, dict(model=args.model, input=payload))
                    assert status == 400, (endpoint, payload, status, response)
                    assert "error" in response, response
                    result["errors"].append(dict(endpoint=endpoint, input=payload, status=status, response=response))
                    no_request_files()
            # Errors must leave the same loaded model usable.
            vectors, _ = bench.embedding(args.url, args.model, payloads[0])
            assert bench.cosine(vectors[0], result["cases"][0]["embedding"]) > .99999
            no_request_files()
            assert process.poll() is None, process.returncode
            result["completed"] = result["passed"] = True
        finally:
            for connection in getattr(bench.CONNECTIONS, "connections", {}).values():
                connection.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            result["server_exit_code"] = process.returncode
            # ASP.NET handles SIGTERM and disposes its service provider. A signal
            # exit alone cannot establish that model/tower resources were freed.
            if process.returncode != 0:
                result["passed"] = False
            save()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
