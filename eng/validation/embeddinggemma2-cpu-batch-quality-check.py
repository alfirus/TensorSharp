#!/usr/bin/env python3
"""Independently check saved frozen CPU batch reports; never run a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import embeddinggemma2_http_environment as http_environment

COSINE = .99999
ERROR = .0005
WIDTHS = (128, 256, 512)
LENGTHS = {
    "bucket16": [9, 15, 16, 9], "bucket32": [17, 31, 32, 17],
    "bucket512": [497, 511, 512, 497],
    "bucket528-window-cutoff": [513, 527, 528, 513],
    "equal32-control": [23] * 4, "multiple-buckets-control": [15, 17, 33, 15],
    "observed-batch32-short-geometry": [15] * 10 + [16] * 22,
    "bucket8192-boundary": [8191, 8192],
}
CANONICAL_LENGTHS = {"canonical-batch32-short-distinct": [15] * 10 + [16] * 22,
                     "canonical-batch32-short-duplicate-control": [15, 16, 15]}
ROUTES = ("native-whole-token-batch", "native-whole-token-reversed",
          "native-whole-external-batch", "native-whole-external-reversed",
          "public-sorted-microbatch")
THREAD_KEYS = {"TS_GGML_CPU_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS",
               "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME", "DOTNET_PROCESSOR_COUNT", "GGML_CPU_DISABLE_FUSION"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(data):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(data, object_pairs_hook=unique_pairs, parse_constant=invalid,
                      parse_int=lambda value: -0.0 if value == "-0" else int(value))


def token_hash(tokens):
    require(isinstance(tokens, list) and tokens and
            all(type(v) is int and 0 <= v <= 2147483647 for v in tokens), "Invalid token IDs")
    return digest(struct.pack("<" + "i" * len(tokens), *tokens))


def vector(values, width, expected_sha=None):
    require(isinstance(values, list) and len(values) == width and
            all(type(v) in (int, float) and math.isfinite(v) for v in values), "Invalid complete finite vector")
    raw = struct.pack("<" + "f" * width, *values)
    result = struct.unpack("<" + "f" * width, raw)
    require(all(math.isfinite(v) for v in result), "F32 conversion overflow")
    if expected_sha is not None:
        require(valid_sha(expected_sha) and digest(raw) == expected_sha, "Saved vector byte hash differs")
    norm = math.sqrt(math.fsum(v * v for v in result))
    require(abs(norm - 1) <= 1e-5, "Vector is not unit normalized")
    return result


def compare(a, b):
    require(len(a) == len(b) and a, "Comparison shape differs")
    cosine = math.fsum(x * y for x, y in zip(a, b)) / math.sqrt(
        math.fsum(x * x for x in a) * math.fsum(y * y for y in b))
    error = max(abs(x - y) for x, y in zip(a, b))
    return {"passed": cosine >= COSINE and error <= ERROR,
            "cosine": cosine, "max_absolute_error": error}


def prefix(values, width):
    norm = math.sqrt(math.fsum(x * x for x in values[:width]))
    require(norm > 0, "Zero MRL prefix")
    return vector([x / norm for x in values[:width]], width)


def aggregate(gates):
    require(gates, "No comparisons executed")
    return {"comparisons": len(gates), "passed": all(g["passed"] for g in gates),
            "minimum_cosine": min(g["cosine"] for g in gates),
            "maximum_absolute_error": max(g["max_absolute_error"] for g in gates)}


def case_names(scope):
    require(scope in ("all", "core", "long", "canonical"), "Unknown quality scope")
    if scope == "canonical":
        return list(CANONICAL_LENGTHS)
    names = list(LENGTHS)
    return names[:-1] if scope == "core" else names[-1:] if scope == "long" else names


def audit_report(report, expected, native_key, names):
    for flag in ("completed", "available", "quality_qualified", "model_disposed",
                 "input_binary_hashes_unchanged", "planned_quality_route_set"):
        require(report.get(flag) is True, "Unqualified report flag: " + flag)
    require(report.get("diagnostic") is False and report.get("route_set") == "all", "Diagnostic route cannot qualify")
    require(report.get("workers") in (7, 8) and report.get("policy") in ("default", "optional"), "Unqualified workers/policy")
    require(report.get("batch_single_cosine_minimum") == COSINE and
            report.get("comparison_max_absolute_error") == ERROR, "Original thresholds differ")
    before, after = report["before_hashes"], report["after_hashes"]
    require(before and before == after and all(valid_sha(v) for v in before.values()), "Recorded input guard differs")
    require(report["loaded_native"]["sha256"] == expected[native_key], "Loaded native identity differs")
    identities = {"model": "embeddinggemma-2-Q8_0.gguf", "models": "TensorSharp.Models.dll",
                  "runtime": "TensorSharp.Runtime.dll", "server": "TensorSharp.Server.dll", native_key: "libGgmlOps.so"}
    for key, basename in identities.items():
        hashes = {value for path, value in before.items() if Path(path).name == basename}
        require(hashes == {expected[key]}, "Recorded identity differs: " + key)
    flags = report["optimization_flags"]
    require(set(flags) == set(http_environment.OPTIMIZATION_FLAGS), "Incomplete selector audit")
    require(set(report["thread_environment"]) == THREAD_KEYS, "Incomplete thread selector audit")
    optional = {"TSG_EMBEDDING_LINUX_BLAS", "TSG_EMBEDDING_CPU_ATTENTION", "TSG_EMBEDDING_BLAS_LIBRARY"}
    require(all(value is None for key, value in flags.items()
                if report["policy"] != "optional" or key not in optional), "Unqualified override")
    if report["policy"] == "optional":
        require(flags.get("TSG_EMBEDDING_LINUX_BLAS") == "1" and flags.get("TSG_EMBEDDING_CPU_ATTENTION") == "1",
                "Optional policy selectors differ")
        provider = flags.get("TSG_EMBEDDING_BLAS_LIBRARY")
        require(provider in before and report["thread_environment"].get("OPENBLAS_NUM_THREADS") == "1", "Unbound optional provider")
    require(report["thread_environment"].get("TS_GGML_CPU_THREADS") in (None, str(report["workers"])) and
            report["thread_environment"].get("GGML_CPU_DISABLE_FUSION") is None, "Conflicting worker/fusion policy")
    cases = report["cases"]
    require(len(cases) == len(names) and {c["name"] for c in cases} == set(names), "Case coverage differs")
    require(report.get("include_long") is ("bucket8192-boundary" in names), "Long coverage flag differs")
    if "bucket8192-boundary" in names:
        require(report.get("long_branch_qualified") is True and valid_sha(report["long_fixture"]["sha256"]), "Unbound long fixture")
        require(before.get(report["long_fixture"]["path"]) == report["long_fixture"]["sha256"], "Original long fixture is not in the input guard")
    if set(LENGTHS).difference({"bucket8192-boundary"}).issubset(names):
        require(report.get("full_core_qualified") is True, "Core coverage flag differs")
    manifest = report["fixture_manifest"]
    require(len(manifest) == len(names) and {c["plan"]["name"] for c in manifest} == set(names), "Fixture manifest coverage differs")
    bound_manifest = {c["plan"]["name"]: c for c in manifest}
    output, all_gates, details = {}, [], []
    expected_calls = dict(public_token_api=0, public_external_singleton_api=0,
                          native_token_wrappers=0, native_external_wrappers=0, token_row_api=0)
    for case in cases:
        name, tokens = case["name"], case["tokens"]
        is_long = name == "bucket8192-boundary"
        require(case.get("completed") is True and case.get("quality_route_set_executed") is True, "Incomplete case")
        lengths = (CANONICAL_LENGTHS if name in CANONICAL_LENGTHS else LENGTHS)[name]
        require([len(t) for t in tokens] == lengths, "Fixture lengths differ")
        hashes = [token_hash(t) for t in tokens]
        if name == "canonical-batch32-short-distinct":
            require(len(set(hashes)) == 32, "Canonical32 token rows are not distinct")
        elif name == "observed-batch32-short-geometry":
            require(len(set(hashes[:10])) == 1 and len(set(hashes[10:])) == 1 and len(set(hashes)) == 2,
                    "Observed batch32 repeated token groups differ")
        elif not is_long:
            require(hashes[0] == hashes[-1], "Mandatory duplicate input differs")
        require(bound_manifest[name]["tokens"] == tokens and bound_manifest[name]["token_sha256_le"] == hashes,
                "Case and manifest tokens differ")
        if is_long:
            require(tokens[0] == tokens[1][:-2] + tokens[1][-1:] and
                    hashes[1] == report["long_fixture"]["original8192_token_sha256_le"], "Long original/shorter identity differs")
        singles = case["singleton_evidence"]
        require(len(singles) == len(set(hashes)) and {s["token_sha256_le"] for s in singles} == set(hashes), "Singleton coverage differs")
        by_hash, evidence, gates = {}, {}, []
        for single in singles:
            key = token_hash(single["tokens"])
            require(key == single["token_sha256_le"] and valid_sha(single["token_rows_sha256_le"]), "Singleton identity differs")
            full = vector(single["token_embedding"], 768, single["token_embedding_sha256_le"])
            by_hash[key] = full
            evidence["singleton:" + key] = full
            if is_long:
                require(single["external_embedding"] is None and single["external_embedding_sha256_le"] is None, "Unexpected long external singleton")
            else:
                external = vector(single["external_embedding"], 768, single["external_embedding_sha256_le"])
                evidence["external-singleton:" + key] = external
                gates.append(compare(full, external))
        routes = case["routes"]
        wanted = ROUTES[:3] if is_long else ROUTES
        require(len(routes) == len(wanted) and {r["route"] for r in routes} == set(wanted), "Route coverage differs")
        for route in routes:
            key = route["route"]
            require(route.get("completed") is True and route.get("batch_single_comparison_executed") is True and
                    route.get("returned_order_reversed_for_comparison") is key.endswith("reversed"), "Incomplete or misordered route")
            values, byte_hashes = route["vectors_in_original_order"], route["vector_sha256_le"]
            mrl = route["singleton_and_mrl_gates"]
            require(len(values) == len(tokens) == len(byte_hashes) == len(mrl), "Batch vector count differs")
            vectors = [vector(v, 768, h) for v, h in zip(values, byte_hashes)]
            for i, full in enumerate(vectors):
                gates.append(compare(by_hash[hashes[i]], full)); evidence[key + ":" + str(i)] = full
                require(mrl[i]["index"] == i, "MRL index differs")
                prefixes = mrl[i]["mrl"]
                require(len(prefixes) == 3 and {p["dimensions"] for p in prefixes} == set(WIDTHS), "MRL coverage differs")
                for p in prefixes:
                    width = p["dimensions"]; actual = vector(p["vector"], width, p["sha256_le"])
                    require(max(abs(x-y) for x,y in zip(actual, prefix(full, width))) <= 1e-6, "MRL helper prefix differs")
                    gates.append(compare(prefix(by_hash[hashes[i]], width), actual))
                    evidence[key + ":" + str(i) + ":" + str(width)] = actual
            first = {}
            for h, v in zip(hashes, vectors):
                if h in first:
                    gates.append(compare(first[h], v))
                first[h] = v
        # Compare every route to the original-order native result, including reversed/API routes.
        for route in wanted[1:]:
            for i in range(len(tokens)):
                gates.append(compare(evidence[wanted[0] + ":" + str(i)], evidence[route + ":" + str(i)]))
        output[name] = {"tokens": tokens, "row_hashes": {s["token_sha256_le"]: s["token_rows_sha256_le"] for s in singles}, "vectors": evidence}
        details.append({"name": name, **aggregate(gates)}); all_gates.extend(gates)
        unique = len(set(hashes)); expected_calls["public_token_api"] += unique + (0 if is_long else 1)
        expected_calls["public_external_singleton_api"] += 0 if is_long else unique
        expected_calls["token_row_api"] += unique; expected_calls["native_token_wrappers"] += 2
        expected_calls["native_external_wrappers"] += 1 if is_long else 2
    require(all(report["observed_probe_call_attempts"].get(k) == v for k,v in expected_calls.items()), "Observed call budget differs")
    expected_learned = sum(expected_calls[k] for k in ("public_token_api", "public_external_singleton_api", "native_token_wrappers", "native_external_wrappers"))
    expected_rows = 0
    for case in cases:
        unique = {token_hash(t): t for t in case["tokens"]}
        is_long = case["name"] == "bucket8192-boundary"
        expected_rows += sum(len(t) for t in unique.values()) * (1 if is_long else 2)
        expected_rows += sum(len(t) for t in case["tokens"]) * (3 if is_long else 5)
    require(report.get("planned_learned_forwards") == expected_learned and
            report.get("planned_token_row_graphs") == expected_calls["token_row_api"] and
            report.get("planned_actual_learned_token_rows") == expected_rows, "Recorded forward/row budget differs")
    return output, {"producer_passed": report.get("passed") is True, "cases": details, **aggregate(all_gates)}


def canonical_source(report, benchmark, expected):
    require(report.get("canonical_benchmark_cases_qualified") is True and
            report.get("full_core_qualified") is False and report.get("long_branch_qualified") is False,
            "Canonical coverage must remain separate")
    source = report["canonical_benchmark_input"]
    require(source["sha256"] == benchmark["sha256"] and
            report["before_hashes"].get(source["path"]) == source["sha256"], "Unbound canonical benchmark report")
    original = benchmark["data"]
    require(original["model_sha256"] == expected["model"] and original["binary_artifacts_unchanged"] is True and
            original["binary_artifacts"] == original["binary_artifacts_after"], "Original benchmark identities differ")
    artifacts = original["binary_artifacts"]["tensorsharp"]
    for key, basename in (("models", "TensorSharp.Models.dll"), ("runtime", "TensorSharp.Runtime.dll"),
                          ("server", "TensorSharp.Server.dll"), ("baseline_native", "libGgmlOps.so")):
        matching = [row for row in artifacts if Path(row["path"]).name == basename]
        require(len(matching) == 1 and matching[0]["sha256"] == expected[key], "Original canonical runtime differs: " + key)
    texts = original["benchmark_inputs"]["batch32_short"]
    require(len(texts) == 32 and all(isinstance(t, str) and t.strip() for t in texts) and len(set(texts)) == 32,
            "Original canonical texts must be32 distinct strings")
    require(source["texts"] == texts and source["source_case"] == "batch32_short" and
            type(original["strict_pass"]) is bool and source["original_strict_pass"] is original["strict_pass"] and
            source["source_native_sha256"] == expected["baseline_native"], "Canonical source binding differs")
    cases = {case["name"]: case for case in report["cases"]}
    tokens = cases["canonical-batch32-short-distinct"]["tokens"]
    require(cases["canonical-batch32-short-duplicate-control"]["tokens"] == [tokens[0], tokens[10], tokens[0]],
            "Canonical duplicate control must use original rows0/10/0")


def check(baseline, candidate, expected, scope="all", benchmark=None):
    require(set(expected) == {"model", "models", "runtime", "server", "baseline_native", "candidate_native"} and
            all(valid_sha(v) for v in expected.values()), "Expected identity manifest differs")
    require((baseline["workers"], baseline["policy"]) == (candidate["workers"], candidate["policy"]), "Worker/policy mismatch")
    require(baseline["thread_environment"] == candidate["thread_environment"], "Recorded thread policy differs")
    for key in http_environment.OPTIMIZATION_FLAGS:
        if key != "TSG_EMBEDDING_BLAS_LIBRARY":
            require(baseline["optimization_flags"].get(key) == candidate["optimization_flags"].get(key), "Recorded compute policy differs")
    if baseline["policy"] == "optional":
        providers = [report["before_hashes"].get(report["optimization_flags"].get("TSG_EMBEDDING_BLAS_LIBRARY")) for report in (baseline, candidate)]
        require(valid_sha(providers[0]) and providers[0] == providers[1], "Optional provider identities differ")
    names = case_names(scope)
    a, ar = audit_report(baseline, expected, "baseline_native", names)
    b, br = audit_report(candidate, expected, "candidate_native", names)
    require((scope == "canonical") == (benchmark is not None), "Canonical scope requires its bound original benchmark")
    if benchmark is not None:
        require(valid_sha(benchmark["sha256"]), "Invalid canonical benchmark digest")
        for report in (baseline, candidate):
            canonical_source(report, benchmark, expected)
    comparisons = []
    if "bucket8192-boundary" in names:
        require(baseline["long_fixture"]["sha256"] == candidate["long_fixture"]["sha256"], "Original long fixture file differs")
    for name in names:
        require(a[name]["tokens"] == b[name]["tokens"] and a[name]["row_hashes"] == b[name]["row_hashes"], "Baseline/candidate complete input identity differs")
        require(a[name]["vectors"].keys() == b[name]["vectors"].keys(), "Baseline/candidate route coverage differs")
        gates = [compare(a[name]["vectors"][key], b[name]["vectors"][key]) for key in a[name]["vectors"]]
        comparisons.append({"name": name, **aggregate(gates)})
    return {"completed": True, "passed": ar["producer_passed"] and br["producer_passed"] and ar["passed"] and br["passed"] and all(c["passed"] for c in comparisons),
            "scope": scope, "workers": baseline["workers"], "policy": baseline["policy"],
            "thresholds": {"min_cosine": COSINE, "max_absolute_error": ERROR},
            "baseline_recomputed": ar, "candidate_recomputed": br, "baseline_candidate": comparisons,
            "fresh_model_forwards": 0, "performance_qualified": False, "native_admission_qualified": False,
            "limitations": ["Saved-vector numerical/recorded-input-guard audit only; no fresh model math, independent NumPy model oracle or actual VM file rehash.",
                            "Native dispatch, optional operator/provider execution, process/runtime closure and latency require separate bound evidence."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "expected-identities", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline", "candidate", "expected-identities"):
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--scope", choices=("all", "core", "long", "canonical"), default="all")
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--benchmark-sha256")
    args = parser.parse_args()
    if (args.scope == "canonical") != (args.benchmark is not None and args.benchmark_sha256 is not None) or (args.scope != "canonical" and (args.benchmark is not None or args.benchmark_sha256 is not None)):
        parser.error("Only canonical scope requires both --benchmark and --benchmark-sha256")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        result = {"completed": False, "passed": False, "fresh_model_forwards": 0}
        before = {}
        try:
            inputs = []
            for name in ("baseline", "candidate", "expected_identities"):
                path = getattr(args, name); data = path.read_bytes(); sha = digest(data)
                require(sha == getattr(args, name + "_sha256"), "Expected input hash differs: " + name)
                before[str(path.resolve())] = sha; inputs.append(read_json(data))
            before[str(Path(__file__).resolve())] = digest(Path(__file__).read_bytes())
            dependency = Path(http_environment.__file__).resolve()
            before[str(dependency)] = digest(dependency.read_bytes())
            benchmark = None
            if args.benchmark is not None:
                data = args.benchmark.read_bytes(); sha = digest(data)
                require(sha == args.benchmark_sha256, "Expected canonical benchmark hash differs")
                before[str(args.benchmark.resolve())] = sha
                benchmark = {"sha256": sha, "data": read_json(data)}
            result.update(check(*inputs, scope=args.scope, benchmark=benchmark))
        except Exception as error:
            result.update(completed=False, passed=False, error=str(error))
        finally:
            after = {}
            for path in before:
                try:
                    after[path] = digest(Path(path).read_bytes()) if Path(path).is_file() else None
                except OSError as error:
                    after[path] = None
                    result["guard_error"] = str(error)
            unchanged = bool(before) and before == after
            result.update(before_hashes=before, after_hashes=after, local_report_source_hashes_unchanged=unchanged)
            if not unchanged:
                result.update(completed=False, passed=False)
            json.dump(result, destination, indent=2, allow_nan=False); destination.write("\n")
    print(json.dumps({k: result[k] for k in ("completed", "passed", "fresh_model_forwards")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
