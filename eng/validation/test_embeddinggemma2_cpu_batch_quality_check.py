import copy
import importlib.util
import json
import math
from pathlib import Path
import struct
import unittest

SPEC = importlib.util.spec_from_file_location("batch_quality_check", Path(__file__).with_name("embeddinggemma2-cpu-batch-quality-check.py"))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def hashed(values):
    return CHECK.digest(struct.pack("<" + "f" * len(values), *values))


IDENTITIES = {name: str(i) * 64 for i, name in enumerate(
    ("model", "models", "runtime", "server", "baseline_native", "candidate_native"), 1)}


def fixture(scope="core", native="baseline_native", policy="default"):
    full = list(CHECK.vector([1 / math.sqrt(768)] * 768, 768))
    before = {"/model/embeddinggemma-2-Q8_0.gguf": IDENTITIES["model"],
              "/snapshot/TensorSharp.Models.dll": IDENTITIES["models"],
              "/snapshot/TensorSharp.Runtime.dll": IDENTITIES["runtime"],
              "/snapshot/TensorSharp.Server.dll": IDENTITIES["server"],
              "/snapshot/libGgmlOps.so": IDENTITIES[native]}
    flags = dict.fromkeys(CHECK.http_environment.OPTIMIZATION_FLAGS)
    threads = dict.fromkeys(CHECK.THREAD_KEYS)
    threads.update(OPENBLAS_NUM_THREADS="1", TS_GGML_CPU_THREADS="8")
    if policy == "optional":
        flags.update(TSG_EMBEDDING_LINUX_BLAS="1", TSG_EMBEDDING_CPU_ATTENTION="1", TSG_EMBEDDING_BLAS_LIBRARY="/provider/libblas.so")
        before["/provider/libblas.so"] = "a" * 64
    result = dict(completed=True, passed=True, available=True, quality_qualified=True, model_disposed=True,
                  input_binary_hashes_unchanged=True, planned_quality_route_set=True, diagnostic=False, route_set="all",
                  workers=8, policy=policy, batch_single_cosine_minimum=CHECK.COSINE, comparison_max_absolute_error=CHECK.ERROR,
                  before_hashes=before, after_hashes=dict(before), loaded_native={"sha256": IDENTITIES[native]},
                  optimization_flags=flags, thread_environment=threads, include_long=scope in ("long", "all"),
                  long_branch_qualified=scope in ("long", "all"), full_core_qualified=scope in ("core", "all"),
                  fixture_manifest=[], cases=[], observed_probe_call_attempts=dict(public_token_api=0,
                  public_external_singleton_api=0, native_token_wrappers=0, native_external_wrappers=0, token_row_api=0))
    for name in CHECK.case_names(scope):
        lengths = (CHECK.CANONICAL_LENGTHS if scope == "canonical" else CHECK.LENGTHS)[name]; long = name == "bucket8192-boundary"
        tokens = [[2] + [100 + i] * (n-2) + [1] for i,n in enumerate(lengths)]
        if long:
            tokens[0] = tokens[1][:-2] + tokens[1][-1:]
            result["long_fixture"] = {"path": "/fixtures/long.json", "sha256": "b" * 64, "original8192_token_sha256_le": CHECK.token_hash(tokens[1])}
            result["before_hashes"]["/fixtures/long.json"] = "b" * 64
            result["after_hashes"]["/fixtures/long.json"] = "b" * 64
        elif name == "observed-batch32-short-geometry":
            tokens = [list(tokens[0])] * 10 + [list(tokens[10])] * 22
        elif name != "canonical-batch32-short-distinct":
            tokens[-1] = list(tokens[0])
        hashes = [CHECK.token_hash(t) for t in tokens]
        result["fixture_manifest"].append({"plan": {"name": name}, "tokens": tokens, "token_sha256_le": hashes})
        singles = []
        for key, row in dict(zip(hashes, tokens)).items():
            singles.append(dict(token_sha256_le=key, tokens=row, token_rows_sha256_le="c" * 64,
                                token_embedding=full, token_embedding_sha256_le=hashed(full),
                                external_embedding=None if long else full, external_embedding_sha256_le=None if long else hashed(full)))
        routes = []
        for route in CHECK.ROUTES[:3] if long else CHECK.ROUTES:
            mrl = [{"index": i, "mrl": [{"dimensions": w, "vector": list(CHECK.prefix(full, w)),
                    "sha256_le": hashed(CHECK.prefix(full, w))} for w in CHECK.WIDTHS]} for i in range(len(tokens))]
            routes.append(dict(route=route, completed=True, batch_single_comparison_executed=True,
                               returned_order_reversed_for_comparison=route.endswith("reversed"),
                               vectors_in_original_order=[full] * len(tokens), vector_sha256_le=[hashed(full)] * len(tokens),
                               singleton_and_mrl_gates=mrl))
        result["cases"].append(dict(name=name, tokens=tokens, completed=True, quality_route_set_executed=True,
                                    singleton_evidence=singles, routes=routes))
        calls = result["observed_probe_call_attempts"]; unique = len(set(hashes))
        calls["public_token_api"] += unique + (0 if long else 1)
        calls["public_external_singleton_api"] += 0 if long else unique
        calls["native_token_wrappers"] += 2; calls["native_external_wrappers"] += 1 if long else 2; calls["token_row_api"] += unique
    if scope == "canonical":
        result["full_core_qualified"] = result["long_branch_qualified"] = False
        result["canonical_benchmark_cases_qualified"] = True
        originals = result["cases"][0]["tokens"]
        # The second case uses exactly original rows0/10/0. Rebuild its evidence
        # to preserve true token/singleton identities in this synthetic fixture.
        replacement = copy.deepcopy(result["cases"][1])
        replacement["tokens"] = [originals[0], originals[10], originals[0]]
        for old, new in zip(replacement["singleton_evidence"], (originals[0], originals[10])):
            old["tokens"] = new; old["token_sha256_le"] = CHECK.token_hash(new)
        result["cases"][1] = replacement
        result["fixture_manifest"][1].update(tokens=replacement["tokens"], token_sha256_le=[CHECK.token_hash(t) for t in replacement["tokens"]])
        result["canonical_benchmark_input"] = dict(path="/original/results.json", sha256="e"*64,
            source_case="batch32_short", original_strict_pass=False, texts=["text " + str(i) for i in range(32)],
            source_native_sha256=IDENTITIES["baseline_native"])
        result["before_hashes"]["/original/results.json"] = "e"*64
        result["after_hashes"]["/original/results.json"] = "e"*64
    calls = result["observed_probe_call_attempts"]
    result["planned_learned_forwards"] = sum(calls.values()) - calls["token_row_api"]
    result["planned_token_row_graphs"] = calls["token_row_api"]
    result["planned_actual_learned_token_rows"] = sum(
        sum(len(t) for t in {CHECK.token_hash(t):t for t in c["tokens"]}.values()) * (1 if c["name"]=="bucket8192-boundary" else 2) +
        sum(len(t) for t in c["tokens"]) * (3 if c["name"]=="bucket8192-boundary" else 5) for c in result["cases"])
    return result


def benchmark_fixture():
    artifacts = [{"path":"/original/"+file, "sha256":IDENTITIES[key], "bytes":100} for key,file in
                 (("models","TensorSharp.Models.dll"),("runtime","TensorSharp.Runtime.dll"),
                  ("server","TensorSharp.Server.dll"),("baseline_native","libGgmlOps.so"))]
    data = dict(model_sha256=IDENTITIES["model"], binary_artifacts_unchanged=True,
                binary_artifacts={"tensorsharp":artifacts},binary_artifacts_after={"tensorsharp":copy.deepcopy(artifacts)},
                benchmark_inputs={"batch32_short":["text "+str(i) for i in range(32)]},strict_pass=False)
    return {"data":data,"sha256":"e"*64}


class SavedBatchChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = fixture()
        cls.candidate = fixture(native="candidate_native")

    def test_full_core_saved_math(self):
        result = CHECK.check(self.baseline, self.candidate, IDENTITIES, "core")
        self.assertTrue(result["passed"])
        self.assertEqual(7, len(result["baseline_candidate"]))
        self.assertEqual(0, result["fresh_model_forwards"])
        self.assertFalse(result["native_admission_qualified"])

    def test_long_routes_and_original_identity(self):
        self.assertTrue(CHECK.check(fixture("long"), fixture("long", "candidate_native"), IDENTITIES, "long")["passed"])

    def test_optional_provider_is_bound(self):
        self.assertTrue(CHECK.check(fixture(policy="optional"), fixture(native="candidate_native", policy="optional"), IDENTITIES, "core")["passed"])

    def reject(self, mutate, message):
        candidate = copy.deepcopy(self.candidate); mutate(candidate)
        with self.assertRaisesRegex((ValueError, KeyError), message):
            CHECK.check(self.baseline, candidate, IDENTITIES, "core")

    def test_missing_case(self):
        self.reject(lambda r: r["cases"].pop(), "coverage")

    def test_missing_route(self):
        self.reject(lambda r: r["cases"][0]["routes"].pop(), "coverage")

    def test_false_reverse_order(self):
        self.reject(lambda r: r["cases"][0]["routes"][1].update(returned_order_reversed_for_comparison=False), "misordered")

    def test_nonfinite_last_component(self):
        def change(r):
            r["cases"][0]["routes"][0]["vectors_in_original_order"] = copy.deepcopy(r["cases"][0]["routes"][0]["vectors_in_original_order"])
            r["cases"][0]["routes"][0]["vectors_in_original_order"][-1][-1] = float("nan")
        self.reject(change, "finite")

    def test_incorrect_vector_byte_hash(self):
        self.reject(lambda r: r["cases"][0]["routes"][0]["vector_sha256_le"].__setitem__(-1, "d"*64), "byte hash")

    def test_unit_norm(self):
        with self.assertRaisesRegex(ValueError, "unit normalized"):
            CHECK.vector([0.] * 768, 768)

    def test_numerical_failure_is_not_overridden_by_producer_pass(self):
        candidate = copy.deepcopy(self.candidate)
        original = list(CHECK.vector([1/math.sqrt(768)] * 768, 768)); original[0] += .002
        norm = math.sqrt(math.fsum(x*x for x in original)); altered = list(CHECK.vector([x/norm for x in original], 768))
        route = candidate["cases"][0]["routes"][0]
        route["vectors_in_original_order"][0] = altered; route["vector_sha256_le"][0] = hashed(altered)
        route["singleton_and_mrl_gates"][0]["mrl"] = [{"dimensions": w,"vector": list(CHECK.prefix(altered,w)),"sha256_le": hashed(CHECK.prefix(altered,w))} for w in CHECK.WIDTHS]
        result = CHECK.check(self.baseline, candidate, IDENTITIES, "core")
        self.assertTrue(candidate["passed"]); self.assertFalse(result["passed"])
        self.assertGreater(result["baseline_candidate"][0]["maximum_absolute_error"], .0005)

    def test_actual_mrl_helper_value(self):
        def change(r):
            p = r["cases"][0]["routes"][0]["singleton_and_mrl_gates"][0]["mrl"][0]
            v = p["vector"].copy(); v.reverse(); v[0] += .00002; n=math.sqrt(sum(x*x for x in v)); v=[x/n for x in v]
            p.update(vector=v, sha256_le=hashed(v))
        self.reject(change, "MRL helper")

    def test_missing_selector_audit(self):
        self.reject(lambda r: r["optimization_flags"].pop("TSG_CUPTI_ACTIVITY_OUTPUT"), "selector audit")

    def test_instrumentation_cannot_qualify(self):
        a,b=copy.deepcopy(self.baseline),copy.deepcopy(self.candidate)
        for report in (a,b): report["optimization_flags"]["TSG_EMBEDDING_CPU_PROFILE"]="1"
        with self.assertRaisesRegex(ValueError,"override"):
            CHECK.check(a,b,IDENTITIES,"core")

    def test_diagnostic_cannot_qualify(self):
        self.reject(lambda r: r.update(diagnostic=True), "Diagnostic")

    def test_native_identity(self):
        self.reject(lambda r: r["loaded_native"].update(sha256="0"*64), "native identity")

    def test_recorded_guard(self):
        self.reject(lambda r: r["after_hashes"].update({"/snapshot/libGgmlOps.so":"d"*64}), "guard")

    def test_worker_policy_mismatch(self):
        self.reject(lambda r: r.update(workers=7), "mismatch")

    def test_call_budget(self):
        self.reject(lambda r: r["observed_probe_call_attempts"].update(native_token_wrappers=1), "budget")

    def test_original_thresholds(self):
        self.reject(lambda r: r.update(comparison_max_absolute_error=.01), "thresholds")

    def test_complete_token_row_identity(self):
        self.reject(lambda r: r["cases"][0]["singleton_evidence"][0].update(token_rows_sha256_le="d"*64), "input identity")

    def test_input_token_hash(self):
        self.reject(lambda r: r["fixture_manifest"][0]["token_sha256_le"].__setitem__(0,"d"*64), "manifest tokens")

    def test_missing_duplicate_fixture(self):
        def change(r):
            r["cases"][0]["tokens"][-1] = [2] + [999] * 7 + [1]
        self.reject(change, "Mandatory duplicate")

    def test_observed_batch32_must_exercise_repeated_groups(self):
        def change(r):
            r["cases"][-1]["tokens"][1] = [2] + [999] * 13 + [1]
        self.reject(change, "repeated token groups")

    def test_long_fixture_must_be_guarded(self):
        a,b = fixture("long"),fixture("long", "candidate_native")
        b["before_hashes"].pop("/fixtures/long.json");b["after_hashes"].pop("/fixtures/long.json")
        with self.assertRaisesRegex(ValueError, "long fixture is not in"):
            CHECK.check(a,b,IDENTITIES,"long")

    def test_optional_provider_bytes_must_match(self):
        a,b = fixture(policy="optional"),fixture(native="candidate_native",policy="optional")
        b["before_hashes"]["/provider/libblas.so"] = "d" * 64
        b["after_hashes"]["/provider/libblas.so"] = "d" * 64
        with self.assertRaisesRegex(ValueError,"provider identities"):
            CHECK.check(a,b,IDENTITIES,"core")

    def test_thread_policy_must_match(self):
        self.reject(lambda r: r["thread_environment"].update(OMP_WAIT_POLICY="passive"), "thread policy")

    def test_thread_selectors_must_be_complete(self):
        a,b=copy.deepcopy(self.baseline),copy.deepcopy(self.candidate)
        a["thread_environment"].pop("KMP_BLOCKTIME");b["thread_environment"].pop("KMP_BLOCKTIME")
        with self.assertRaisesRegex(ValueError,"thread selector audit"):
            CHECK.check(a,b,IDENTITIES,"core")

    def test_negative_zero_byte_identity(self):
        values=CHECK.read_json('[1,' + ','.join(['0']*766) + ',-0]')
        expected=CHECK.digest(struct.pack('<768f',1.,*([0.]*766),-0.))
        result=CHECK.vector(values,768,expected)
        self.assertEqual(-1.,math.copysign(1.,result[-1]))

    def test_duplicate_json_key(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            CHECK.read_json('{"passed":false,"passed":true}')

    def test_json_nan(self):
        with self.assertRaisesRegex(ValueError,"Nonfinite"):
            CHECK.read_json('{"value":NaN}')

    def test_canonical_distinct_scope_and_budget(self):
        a,b=fixture("canonical"),fixture("canonical","candidate_native")
        result=CHECK.check(a,b,IDENTITIES,"canonical",benchmark_fixture())
        self.assertTrue(result["passed"]);self.assertEqual(78,a["planned_learned_forwards"])
        self.assertEqual(34,a["planned_token_row_graphs"]);self.assertEqual(3806,a["planned_actual_learned_token_rows"])
        self.assertFalse(a["full_core_qualified"])

    def test_canonical_requires_original_report(self):
        with self.assertRaisesRegex(ValueError,"requires.*benchmark"):
            CHECK.check(fixture("canonical"),fixture("canonical","candidate_native"),IDENTITIES,"canonical")

    def test_canonical_original_guard(self):
        b=fixture("canonical","candidate_native");b["before_hashes"].pop("/original/results.json");b["after_hashes"].pop("/original/results.json")
        with self.assertRaisesRegex(ValueError,"Unbound canonical"):
            CHECK.check(fixture("canonical"),b,IDENTITIES,"canonical",benchmark_fixture())

    def test_canonical_text_binding(self):
        original=benchmark_fixture();original["data"]["benchmark_inputs"]["batch32_short"][0]="different"
        with self.assertRaisesRegex(ValueError,"source binding"):
            CHECK.check(fixture("canonical"),fixture("canonical","candidate_native"),IDENTITIES,"canonical",original)

    def test_canonical_original_runtime(self):
        original=benchmark_fixture();original["data"]["binary_artifacts"]["tensorsharp"][0]["sha256"]="f"*64
        original["data"]["binary_artifacts_after"]=copy.deepcopy(original["data"]["binary_artifacts"])
        with self.assertRaisesRegex(ValueError,"runtime differs"):
            CHECK.check(fixture("canonical"),fixture("canonical","candidate_native"),IDENTITIES,"canonical",original)

    def test_canonical_cannot_claim_core_scope(self):
        b=fixture("canonical","candidate_native");b["full_core_qualified"]=True
        with self.assertRaisesRegex(ValueError,"separate"):
            CHECK.check(fixture("canonical"),b,IDENTITIES,"canonical",benchmark_fixture())

    def test_canonical_all32_token_sequences_must_be_distinct(self):
        b=fixture("canonical","candidate_native")
        b["cases"][0]["tokens"][20]=list(b["cases"][0]["tokens"][21])
        with self.assertRaisesRegex(ValueError,"not distinct"):
            CHECK.check(fixture("canonical"),b,IDENTITIES,"canonical",benchmark_fixture())

    def test_canonical_duplicate_uses_exact_original_row10(self):
        b=fixture("canonical","candidate_native");case=b["cases"][1]
        replacement=list(b["cases"][0]["tokens"][11]);case["tokens"][1]=replacement
        single=case["singleton_evidence"][1];single.update(tokens=replacement,token_sha256_le=CHECK.token_hash(replacement))
        b["fixture_manifest"][1].update(tokens=case["tokens"],token_sha256_le=[CHECK.token_hash(t) for t in case["tokens"]])
        with self.assertRaisesRegex(ValueError,"rows0/10/0"):
            CHECK.check(fixture("canonical"),b,IDENTITIES,"canonical",benchmark_fixture())

    def test_original_benchmark_failure_is_retained(self):
        original=benchmark_fixture();original["data"]["strict_pass"]=True
        with self.assertRaisesRegex(ValueError,"source binding"):
            CHECK.check(fixture("canonical"),fixture("canonical","candidate_native"),IDENTITIES,"canonical",original)

    def test_forward_budget_is_recomputed(self):
        self.reject(lambda r:r.update(planned_learned_forwards=1),"budget")

    def test_unknown_scope_rejects(self):
        with self.assertRaisesRegex(ValueError,"Unknown quality scope"):
            CHECK.check(self.baseline,self.candidate,IDENTITIES,"unknown")


if __name__ == "__main__":
    unittest.main()
