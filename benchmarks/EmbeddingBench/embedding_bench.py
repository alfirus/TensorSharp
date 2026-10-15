#!/usr/bin/env python3
"""Real HTTP embedding correctness and latency comparison, with sequential engines.

Commands are JSON arrays (no shell), must bind the corresponding URL, and are
recorded verbatim. Uses only the Python standard library. See README.md.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import urllib.parse


KEEP_ALIVE = False
CONNECTIONS = threading.local()
EXTRA_SCENARIOS = {}


TEXTS = [
    "query: find the function that adds two numbers",
    "def add(a, b):\n    return a + b",
    "def read_file(path):\n    with open(path) as f:\n        return f.read()",
    "query: read a text file from disk",
    "The cat sits on the mat.",
    "A kitten rests on a rug.",
    "The stock exchange closed lower today.",
    "你好，世界！查找读取文件的函数。",
    "Café naïve résumé — Straße ﬁ ﬃ ＡＢＣ e\u0301 😀",
    "\t  white\nspace\r\n normalization  ",
    "query: what is snowflake?",
    "The Data Cloud!",
    "Mexico City of Course!",
]


def request(url, path, body=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    if KEEP_ALIVE:
        connections = getattr(CONNECTIONS,"connections",None)
        if connections is None:
            connections = CONNECTIONS.connections = {}
        if url not in connections:
            target = urllib.parse.urlparse(url)
            connection_type = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
            connections[url] = connection_type(target.hostname,target.port,timeout=180)
        connection = connections[url]
        try:
            connection.request("GET" if data is None else "POST",urllib.parse.urlparse(url).path.rstrip("/")+path,
                               body=data,headers={"Content-Type":"application/json"})
            response = connection.getresponse()
            payload = response.read()
            return response.status,json.loads(payload)
        except Exception:
            connection.close()
            connections.pop(url,None)
            raise
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def embedding(url, model, inputs, **kwargs):
    status, result = request(url, "/v1/embeddings", dict(model=model, input=inputs, **kwargs))
    if status != 200:
        raise AssertionError((status, result))
    assert result["object"] == "list"
    rows = result["data"]
    assert [r["index"] for r in rows] == list(range(len(rows)))
    assert all(r["object"] == "embedding" for r in rows)
    vectors = [r["embedding"] for r in rows]
    return vectors, result["usage"]


def norm(a):
    return math.sqrt(sum(x*x for x in a))


def cosine(a, b):
    assert len(a) == len(b)
    return sum(x*y for x, y in zip(a, b)) / (norm(a) * norm(b))


def assert_vectors(vectors, dimensions):
    for row in vectors:
        assert len(row) == dimensions
        assert all(math.isfinite(x) for x in row)
        assert abs(norm(row) - 1) < 1e-4, norm(row)


def scenarios():
    sentence = "The embedding service indexes source code and retrieves relevant functions. "
    return {
        "single_short": [sentence],
        "single_medium": [sentence * 10],
        "single_long": [sentence * 28],
        "batch8_short": [sentence + str(i) for i in range(8)],
        "batch8_medium": [sentence * 10 + str(i) for i in range(8)],
        "batch32_short": [sentence + str(i) for i in range(32)],
        "batch_mixed": [sentence * n for n in (1, 3, 8, 16, 2, 5, 10, 20)],
        **EXTRA_SCENARIOS,
    }


def command_artifacts(command):
    """Bind measurements to the binaries, hashing outside the measured work."""
    paths = set()
    for argument in command:
        path = Path(argument)
        if path.is_file() and path.suffix == ".dll":
            paths.add(path)
            paths.update(path.parent.glob("TensorSharp*.dll"))
            paths.update(path.parent.glob("*GgmlOps*"))
        elif path.is_file() and path.name.startswith("llama-server"):
            paths.add(path)
            paths.update(path.parent.glob("libllama*"))
            paths.update(path.parent.glob("libggml*"))
    artifacts = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as binary:
            for block in iter(lambda: binary.read(1024 * 1024), b""):
                digest.update(block)
        artifacts.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest()))
    return artifacts


def benchmark(url, model, warmup, repeats, minimum_seconds=0, cases=None):
    results = {}
    for name, inputs in scenarios().items():
        if cases is not None and name not in cases:
            continue
        for _ in range(warmup):
            embedding(url, model, inputs)
        samples = []
        measure_start = time.perf_counter()
        while len(samples) < repeats or time.perf_counter() - measure_start < minimum_seconds:
            start = time.perf_counter()
            vectors, usage = embedding(url, model, inputs)
            samples.append((time.perf_counter() - start) * 1000)
        median = statistics.median(samples)
        results[name] = dict(samples_ms=samples, sample_count=len(samples), median_ms=median,
                             p95_ms=sorted(samples)[min(len(samples)-1, math.ceil(len(samples)*.95)-1)],
                             texts=len(inputs), prompt_tokens=usage["prompt_tokens"],
                             texts_per_second=len(inputs)*1000/median,
                             tokens_per_second=usage["prompt_tokens"]*1000/median)
        if name in EXTRA_SCENARIOS:
            assert_vectors(vectors, len(vectors[0]))
            results[name]["vectors"] = vectors
        print(name, round(median, 3), "ms", flush=True)
    return results


def numerical_gate(checks, name, passed, metrics, failure, is_tensorsharp,
                   measure_reference_after_quality_failure=False):
    """Record strict numerical failures; only an explicit reference run continues."""
    checks[name] = dict(passed=bool(passed), **metrics)
    assert passed or (measure_reference_after_quality_failure and not is_tensorsharp), failure


def correctness(url, model, is_tensorsharp, reference_min_batch_cosine=.9999, reference_min_duplicate_cosine=.99999,
                measure_reference_after_quality_failure=False):
    checks = {}
    vectors, usage = embedding(url, model, TEXTS)
    assert len(vectors) == len(TEXTS)
    dim = len(vectors[0])
    assert_vectors(vectors, dim)
    singles = [embedding(url, model, s)[0][0] for s in TEXTS]
    batch_cosines = [cosine(a,b) for a,b in zip(vectors,singles)]
    consistency_gate = .9999 if is_tensorsharp else reference_min_batch_cosine
    numerical_gate(checks, "batch", min(batch_cosines) > consistency_gate,
                   dict(cosines=batch_cosines, threshold=consistency_gate), batch_cosines,
                   is_tensorsharp, measure_reference_after_quality_failure)
    reversed_rows, _ = embedding(url, model, list(reversed(TEXTS)))
    reverse_cosines = [cosine(a,b) for a,b in zip(vectors,reversed(reversed_rows))]
    numerical_gate(checks, "reversed", min(reverse_cosines) > .9999,
                   dict(cosines=reverse_cosines, threshold=.9999), ("reversed input order", reverse_cosines, .9999),
                   is_tensorsharp, measure_reference_after_quality_failure)
    repeated, _ = embedding(url, model, [TEXTS[0], TEXTS[0], TEXTS[1]])
    duplicate_cosine = cosine(repeated[0], repeated[1])
    duplicate_gate = .99999 if is_tensorsharp else reference_min_duplicate_cosine
    numerical_gate(checks, "duplicate", duplicate_cosine > duplicate_gate,
                   dict(cosine=duplicate_cosine, threshold=duplicate_gate), ("duplicate inputs", duplicate_cosine, duplicate_gate),
                   is_tensorsharp, measure_reference_after_quality_failure)
    retrieval_cosines = {name: dict(positive=cosine(vectors[q],vectors[p]), negative=cosine(vectors[q],vectors[n]))
                         for name, q, p, n in (("add_function",0,1,2), ("read_file",3,2,1),
                                                ("cat_paraphrase",4,5,6), ("snowflake",10,11,12))}
    rankings = {name: values["positive"] > values["negative"] for name, values in retrieval_cosines.items()}
    numerical_gate(checks, "retrieval", all(rankings.values()), dict(rankings=rankings, cosines=retrieval_cosines), rankings,
                   is_tensorsharp, measure_reference_after_quality_failure)
    extra = {}
    if is_tensorsharp:
        status, models = request(url, "/v1/models")
        assert status == 200 and models["data"]
        status, tags = request(url, "/api/tags")
        assert status == 200 and tags["models"]
        status, ollama = request(url, "/api/embed", dict(model=model,input=TEXTS))
        assert status == 200, ollama
        assert min(cosine(a,b) for a,b in zip(vectors,ollama["embeddings"])) > .99999
        assert ollama["prompt_eval_count"] == usage["prompt_tokens"]
        assert ollama["total_duration"] > 0
        encoded, _ = embedding(url,model,TEXTS[0],encoding_format="base64")
        decoded = list(struct.unpack("<"+"f"*dim,base64.b64decode(encoded[0])))
        assert max(abs(a-b) for a,b in zip(decoded,singles[0])) < 1e-7
        small, _ = embedding(url,model,TEXTS[0],dimensions=min(256,dim))
        assert_vectors(small,min(256,dim))
        assert cosine(small[0],singles[0][:len(small[0])]) > .99999
        invalid = [
            {}, dict(model=model,input=[]), dict(model=model,input=""),
            dict(model=model,input=["a",3]), dict(model=model,input=[-1]),
            dict(model=model,input=[2147483647]), dict(model=model,input=[[1],[]]),
            dict(model=model,input="a",dimensions=0),
            dict(model=model,input="a",dimensions=dim+1),
            dict(model=model,input="a",encoding_format="other"),
            dict(model="nonexistent-model",input="a"),
        ]
        for body in invalid:
            status, error = request(url,"/v1/embeddings",body)
            assert 400 <= status < 500, (body,status,error)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            concurrent_rows = list(pool.map(lambda text: embedding(url,model,text)[0][0],TEXTS[:8]))
        assert min(cosine(a,b) for a,b in zip(singles,concurrent_rows)) > .9999
        extra = dict(invalid_requests=len(invalid),concurrent_requests=8,
                     protocols=["OpenAI float","OpenAI base64","Ollama embed"],
                     dimensions_tested=min(256,dim))
    return dict(dimensions=dim,vectors=vectors,single_vectors=singles,
                usage=usage,batch_min_cosine=min(batch_cosines),batch_consistency_gate=consistency_gate,
                reversed_min_cosine=min(reverse_cosines),reversed_consistency_gate=.9999,
                duplicate_cosine=duplicate_cosine,duplicate_consistency_gate=duplicate_gate,
                rankings=rankings,retrieval_cosines=retrieval_cosines,numerical_checks=checks,
                quality_pass=all(check["passed"] for check in checks.values()),**extra)


def process_memory(pid):
    """Read Linux main-server memory outside timing; child/GPU memory is excluded."""
    if platform.system() != "Linux":
        return dict(available=False, scope="main-server process", reason="Linux /proc is unavailable")
    try:
        status = (Path("/proc") / str(pid) / "status").read_text()
        fields = {}
        for line in status.splitlines():
            key, separator, value = line.partition(":")
            if separator and key in ("VmRSS", "VmHWM", "VmSize", "VmSwap"):
                pieces = value.split()
                if len(pieces) == 2 and pieces[1] == "kB":
                    fields[key] = int(pieces[0]) * 1024
        if "VmRSS" not in fields or "VmHWM" not in fields:
            return dict(available=False, scope="main-server process", reason="RSS/high-water fields are unavailable")
        return dict(available=True, scope="main-server process", pid=pid,
                    resident_bytes=fields["VmRSS"], high_water_resident_bytes=fields["VmHWM"],
                    virtual_bytes=fields.get("VmSize"), swap_bytes=fields.get("VmSwap"),
                    source="Linux /proc/<pid>/status; kB fields converted with 1024 bytes",
                    limitations="Excludes child-process RSS and GPU device memory; high-water includes startup.")
    except (OSError, ValueError) as error:
        return dict(available=False, scope="main-server process", reason=type(error).__name__)


def run_engine(name, command, url, args, output):
    record = dict(command=command, url=url, memory_snapshots=[])
    capture = None
    if getattr(args, "capture_boundary_telemetry", False):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eng/validation"))
        from embeddinggemma2_boundary_telemetry import capture_boundary
        from embeddinggemma2_http_environment import audit_child_environment
        capture = capture_boundary
        record["environment_audit"] = audit_child_environment(command, os.environ).metadata()
        record["boundary_telemetry_before"] = capture()
    log = open(output / (name + ".log"),"w")
    process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    start = time.monotonic()
    try:
        while time.monotonic()-start < 180:
            if process.poll() is not None:
                raise RuntimeError(name+" exited: "+str(process.returncode)+"; see "+str(output/(name+".log")))
            try:
                status, _ = request(url,"/v1/models")
                if status == 200:
                    break
            except (OSError,ValueError):
                pass
            time.sleep(.25)
        else:
            raise TimeoutError(name+" startup")
        record["memory_snapshots"].append(dict(phase="ready", **process_memory(process.pid)))
        print(name,"ready",flush=True)
        checks = correctness(url,args.model,name=="tensorsharp",args.reference_min_batch_cosine,args.reference_min_duplicate_cosine,
                             args.measure_reference_after_quality_failure)
        # Small encoders can finish the correctness corpus before a managed
        # runtime promotes hot methods. Give both engines the same timed
        # prewarm, in addition to the per-shape warmups below.
        prewarm_start = time.perf_counter()
        prewarm_calls = 0
        while time.perf_counter() - prewarm_start < args.prewarm_seconds:
            embedding(url,args.model,scenarios()["single_short"])
            prewarm_calls += 1
        prewarm = dict(calls=prewarm_calls,elapsed_seconds=time.perf_counter()-prewarm_start)
        record["memory_snapshots"].append(dict(phase="after-correctness-and-prewarm", **process_memory(process.pid)))
        timings = benchmark(url,args.model,args.warmup,args.repeats,args.minimum_measure_seconds,args.cases)
        record["memory_snapshots"].append(dict(phase="after-all-measurements", **process_memory(process.pid)))
        record.update(correctness=checks, prewarm=prewarm, benchmarks=timings)
        return record
    finally:
        for connection in getattr(CONNECTIONS,"connections",{}).values():
            connection.close()
        CONNECTIONS.connections = {}
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        log.close()
        record["server_exit_code"] = process.returncode
        if capture:
            record["boundary_telemetry_after"] = capture()
        if process.returncode != 0:
            raise RuntimeError(name+" failed graceful shutdown: "+str(process.returncode))


def main():
    global KEEP_ALIVE, EXTRA_SCENARIOS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorsharp-command",required=True,type=json.loads)
    parser.add_argument("--llama-command",required=True,type=json.loads)
    parser.add_argument("--tensorsharp-url",default="http://127.0.0.1:18380")
    parser.add_argument("--llama-url",default="http://127.0.0.1:18381")
    parser.add_argument("--model",required=True)
    parser.add_argument("--model-file",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--warmup",type=int,default=3)
    parser.add_argument("--repeats",type=int,default=10)
    parser.add_argument("--cases",nargs="+",
                        help="Measure selected latency cases; correctness checks still run in full")
    parser.add_argument("--scenario-file",type=Path,
                        help="JSON object mapping extra case names to nonempty arrays of input strings")
    parser.add_argument("--prewarm-seconds",type=float,default=10,
                        help="Symmetric runtime warmup before steady-state timing (default: 10 seconds)")
    parser.add_argument("--minimum-measure-seconds",type=float,default=0,
                        help="Measure each case for at least this duration as well as the requested repeats")
    parser.add_argument("--min-cosine",type=float,default=.999)
    parser.add_argument("--reference-min-batch-cosine",type=float,default=.9999,
                        help="Explicit llama batch/individual tolerance; TensorSharp always requires .9999")
    parser.add_argument("--reference-min-duplicate-cosine",type=float,default=.99999,
                        help="Explicit llama duplicate-input tolerance; TensorSharp and the default reference always require .99999")
    parser.add_argument("--measure-reference-after-quality-failure",action="store_true",
                        help="Collect llama numerical failures and timings without relaxing gates; candidate failures still stop, and the overall run fails")
    parser.add_argument("--max-slowdown",type=float,default=1.05)
    parser.add_argument("--require-performance",action="store_true")
    parser.add_argument("--tensorsharp-first",action="store_true",help="Reverse engine order for a repeat run")
    parser.add_argument("--keep-alive",action="store_true",
                        help="Reuse one HTTP connection per client thread, as embedding SDKs normally do")
    parser.add_argument("--capture-boundary-telemetry",action="store_true",
                        help="Read-only GPU/cgroup snapshots before launch and after shutdown, outside timing")
    args=parser.parse_args()
    if args.scenario_file:
        EXTRA_SCENARIOS = json.loads(args.scenario_file.read_text())
        if not isinstance(EXTRA_SCENARIOS, dict) or not EXTRA_SCENARIOS or any(
                not isinstance(name, str) or not isinstance(inputs, list) or not inputs or
                any(not isinstance(value, str) or not value for value in inputs)
                for name, inputs in EXTRA_SCENARIOS.items()):
            parser.error("scenario-file must map case names to nonempty arrays of nonempty strings")
        # Keep the standard short case available for symmetric runtime prewarm.
        standard_names = {"single_short", "single_medium", "single_long", "batch8_short",
                          "batch8_medium", "batch32_short", "batch_mixed"}
        if standard_names.intersection(EXTRA_SCENARIOS):
            parser.error("extra scenarios must have distinct names from the standard cases")
    if args.cases and set(args.cases).difference(scenarios()):
        parser.error("unknown cases: " + ", ".join(sorted(set(args.cases).difference(scenarios()))))
    KEEP_ALIVE = args.keep_alive
    if args.repeats < 1 or args.warmup < 0 or not math.isfinite(args.prewarm_seconds) or args.prewarm_seconds < 0:
        parser.error("repeats must be positive; warmup and finite prewarm-seconds must be nonnegative")
    if not math.isfinite(args.minimum_measure_seconds) or args.minimum_measure_seconds < 0:
        parser.error("minimum-measure-seconds must be finite and nonnegative")
    if any(not math.isfinite(value) or not -1 <= value <= 1 for value in
           (args.reference_min_batch_cosine, args.reference_min_duplicate_cosine)):
        parser.error("reference cosine gates must be finite and between -1 and 1")
    args.output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256()
    with args.model_file.open("rb") as model_file:
        for block in iter(lambda:model_file.read(1024*1024),b""):
            digest.update(block)
    result=dict(platform=platform.platform(),machine=platform.machine(),strict_pass=False,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                model_file=str(args.model_file),model_sha256=digest.hexdigest(),
                runner_arguments=list(os.sys.argv[1:]),
                thresholds=dict(cross_engine_min_cosine=args.min_cosine,maximum_latency_ratio=args.max_slowdown,
                                reference_batch_min_cosine=args.reference_min_batch_cosine,
                                reference_duplicate_min_cosine=args.reference_min_duplicate_cosine,
                                candidate_batch_min_cosine=.9999,candidate_duplicate_min_cosine=.99999,
                                reversed_order_min_cosine=.9999,performance_required=args.require_performance),
                warmup=args.warmup,repeats=args.repeats,prewarm_seconds=args.prewarm_seconds,
                measure_reference_after_quality_failure=args.measure_reference_after_quality_failure,
                minimum_measure_seconds=args.minimum_measure_seconds,
                connection_mode="persistent-per-thread" if KEEP_ALIVE else "new-per-request",inputs=TEXTS,
                benchmark_inputs={key:value for key,value in scenarios().items() if args.cases is None or key in args.cases},engines={},
                binary_artifacts={"tensorsharp":command_artifacts(args.tensorsharp_command),
                                  "llama":command_artifacts(args.llama_command)})
    destination=args.output/"results.json"
    try:
        engines=[("llama",args.llama_command,args.llama_url),
                 ("tensorsharp",args.tensorsharp_command,args.tensorsharp_url)]
        if args.tensorsharp_first:
            engines.reverse()
        for name,command,url in engines:
            result["engines"][name]=run_engine(name,command,url,args,args.output)
            destination.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
        ts=result["engines"]["tensorsharp"]
        llama=result["engines"]["llama"]
        cosines=[cosine(a,b) for a,b in zip(ts["correctness"]["vectors"],llama["correctness"]["vectors"])]
        error=max(abs(x-y) for a,b in zip(ts["correctness"]["vectors"],llama["correctness"]["vectors"]) for x,y in zip(a,b))
        ratios={name:ts["benchmarks"][name]["median_ms"]/v["median_ms"] for name,v in llama["benchmarks"].items()}
        result["comparison"]=dict(cosines=cosines,min_cosine=min(cosines),max_absolute_error=error,
                                  cross_engine_quality_pass=min(cosines)>=args.min_cosine,
                                  latency_ratio_tensorsharp_over_llama=ratios,
                                  performance_pass=all(v <= args.max_slowdown for v in ratios.values()))
        extra_checks = {}
        for name in EXTRA_SCENARIOS:
            if name not in ts["benchmarks"]:
                continue
            a, b = ts["benchmarks"][name], llama["benchmarks"][name]
            assert a["prompt_tokens"] == b["prompt_tokens"], (name, "token accounting differs")
            assert len(a["vectors"]) == len(b["vectors"]), (name, "row count differs")
            similarities = [cosine(x,y) for x,y in zip(a["vectors"],b["vectors"])]
            extra_checks[name] = dict(cosines=similarities, min_cosine=min(similarities), prompt_tokens=a["prompt_tokens"],
                                     threshold=args.min_cosine, passed=min(similarities)>=args.min_cosine)
        if extra_checks:
            result["comparison"]["extra_scenarios"] = extra_checks
        assert ts["correctness"]["usage"] == llama["correctness"]["usage"], "token accounting differs"
        result["comparison"]["quality_pass"] = (ts["correctness"]["quality_pass"] and llama["correctness"]["quality_pass"] and
                                                 min(cosines)>=args.min_cosine and all(item["passed"] for item in extra_checks.values()))
        # All raw-input comparison metrics are now attached before any numeric
        # failure aborts; a failing 8192-token case must not hide later cases.
        for name, check in extra_checks.items():
            assert check["passed"], (name, check)
        assert min(cosines)>=args.min_cosine, result["comparison"]
        assert ts["correctness"]["quality_pass"] and llama["correctness"]["quality_pass"], "Strict engine numerical checks failed; timings are diagnostic only"
        if args.require_performance:
            assert result["comparison"]["performance_pass"], ratios
        result["strict_pass"] = True
        print(json.dumps(result["comparison"],indent=2),flush=True)
    finally:
        result["binary_artifacts_after"] = {
            "tensorsharp": command_artifacts(args.tensorsharp_command),
            "llama": command_artifacts(args.llama_command),
        }
        result["binary_artifacts_unchanged"] = result["binary_artifacts"] == result["binary_artifacts_after"]
        if not result["binary_artifacts_unchanged"]:
            result["strict_pass"] = False
        destination.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
        if not result["binary_artifacts_unchanged"]:
            raise RuntimeError("Runtime binaries changed during validation; repeat using a frozen build directory")


if __name__=="__main__":
    main()
