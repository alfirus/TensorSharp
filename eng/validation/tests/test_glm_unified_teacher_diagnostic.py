"""Independent small completed-record fixtures for the numerical diagnostic gate."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location("teacher_diagnostic", Path(__file__).resolve().parents[1] / "glm_unified_teacher_diagnostic.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def inputs_fixture():
    catalog = []
    for layer in range(3, 45):
        triple = (16, 16, 18) if layer < 14 else (19, 19, 18) if layer < 42 else (16, 16, 23)
        catalog.append(dict(layer=layer, gate_type=triple[0], up_type=triple[1], down_type=triple[2],
                            gate_bytes=100, up_bytes=100, down_bytes=200))
    cases = []
    for ordinal, name in enumerate(("math", "extraction", "squares", "code")):
        instruction, teacher = gate.TASKS[name]
        prompt, targets = [65 + ordinal] * 1872, list(range(70, 71 + ordinal))
        rendered = "Background context\n" + instruction + "\nassistant\n"
        cases.append(dict(name=name, instruction=instruction, teacher_text=teacher, rendered_prompt=rendered,
                          prompt_ids=prompt, teacher_ids=targets, full_teacher_tokens=len(targets), teacher_truncated=False,
                          rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(), prompt_sha256=gate.token_hash(prompt),
                          teacher_sha256=gate.token_hash(targets), full_history_sha256=gate.token_hash(prompt + targets)))
    return dict(schema=1, model_path="C:/immutable/model.gguf", model_length=1000000, model_modified_utc_ticks=123456789,
                architecture="glm5next", tokenizer_type="actual.tokenizer", template_sha256="a" * 64,
                source_catalog_sha256=hashlib.sha256(json.dumps(catalog, separators=(",", ":")).encode()).hexdigest(),
                source_catalog=catalog, case_provenance="existing authored cases; BSD-3-Clause", padding_provenance="existing sentence",
                selection="fixed-answers-before-first-forward; teacher suffix is full rendered-plus-teacher encoding minus exact prompt prefix; first32 suffix tokens",
                cases=cases)


def report_fixture(inputs, profile, delta=0.0):
    cases = []
    for ordinal, source in enumerate(inputs["cases"]):
        rows = []
        for index, target in enumerate(source["teacher_ids"]):
            nll = 1.5 + ordinal * .25 + delta
            rows.append(dict(index=index, position=1872 + index, target=target,
                history_sha256=gate.token_hash(source["prompt_ids"] + source["teacher_ids"][:index]),
                forwarded_history_sha256=gate.token_hash(source["prompt_ids"] + source["teacher_ids"][:index + 1]),
                score=dict(target=target, argmax=80, target_rank=2, target_log_probability=-nll, nll=nll,
                           log_normalizer=5.0, top_margin=.25, logits_sha256="b" * 64),
                sampled_token=target, tokens_forwarded=1, head_after=1873 + index))
        total = sum(row["score"]["nll"] for row in rows)
        cases.append(dict(name=source["name"], prefill_submissions=[1872], rows=rows, total_nll=total, mean_nll=total / len(rows)))
    environment = {**dict.fromkeys(gate.ENV_ON, "1"), **dict.fromkeys(gate.ENV_CLEARED),
                   **dict.fromkeys(gate.ENV_OFF, "0"), **gate.ENV_FIXED,
                   "TS_SCHED_PREFILL_CHUNK": "512" if profile == "pp512" else None}
    count, total = sum(len(x["rows"]) for x in cases), sum(x["total_nll"] for x in cases)
    return dict(schema=1, profile=profile, scope=gate.SCOPE, status="numerical_diagnostic_unqualified_until_log_validation",
                inputs_sha256="c" * 64, model_path=inputs["model_path"], mapped_native_path="C:/private/GgmlOps.dll",
                models_mvid="11111111-1111-1111-1111-111111111111", tests_mvid="22222222-2222-2222-2222-222222222222",
                requested_context=2048, requested_prefill=512, requested_adaptive_max=512 if profile == "pp512" else 1024,
                scheduler_chunk_env="512" if profile == "pp512" else None,
                resolved_context=2048, resolved_prefill=512 if profile == "pp512" else 898, resolved_forward_tokens=2048,
                max_running=1, host_embedding=0, environment=environment, cases=cases,
                teacher_tokens=count, total_nll=total, mean_nll=total / count,
                checked_model_free_and_ordinary_scope_drained=True, checked_cache_release_attachment=False,
                native_handle_zero=True, active_allocations=0, ledger_reserved=0, ledger_committed=0,
                general_quality_qualified=False, strict_canonical_api_qualified=False, performance_qualified=False,
                source_file_offsets_independently_captured=False, binary_identity_requires_external_launcher_pins=True)


def record(tag, values):
    return tag + " " + " ".join(f"{key}={value}" for key, value in values.items())


def cache(layer, rows):
    # Explicit actual-schema fields, independent of the parser's field tuple.
    fields = "file_sort_requested file_sort_calls file_sort_changed_calls file_sort_misses file_sort_bytes file_sort_inversions_before file_sort_inversions_after file_view_requested file_view_calls file_view_bytes file_view_fallbacks file_prefetch_requested file_prefetch_calls file_prefetch_ranges file_prefetch_bytes file_prefetch_failures vmm_profile vmm_capacity vmm_physical_bytes vmm_pending_bytes vmm_quarantine_bytes vmm_certificate".split()
    values = dict.fromkeys(fields, 0)
    values.update(layer=layer, rows=rows, joint=0, bulk=int(rows > 1), outcome="completed", file_bytes=100,
                  file_windows=1, graph_submissions=1, file_read_calls=1, file_read_tasks=3,
                  file_read_completed_tasks=3, file_read_failed_calls=0, file_read_source_bytes=100,
                  read_pool_requested=0, file_read_random_access=0, file_view_buffered_bytes=100,
                  scope="nested_host_wall_not_cuda_kernel_or_physical_io")
    return record(gate.CACHE, values)


def forward(rows, p0, logits):
    return record(gate.FORWARD, dict(kind=0, rows=rows, p0=p0, nout=1, wanth=0, logits=logits,
        outcome="completed", scope="host_wall_including_existing_sync_and_expert_callbacks"))


def log_fixture(inputs, report, chunks):
    lines = ["unrelated startup", cache(3, 13), forward(13, 0, 1)]
    for source, result in zip(inputs["cases"], report["cases"]):
        name = source["name"]
        lines.append(record("[GLM-TEACHER-CASE-BEGIN]", dict(case=name, profile=report["profile"],
            prompt_sha=source["prompt_sha256"], teacher_sha=source["teacher_sha256"], inputs_sha="c" * 64)))
        lines.append(record("[GLM-TEACHER-PP-BEGIN]", dict(case=name, rows=1872)))
        for layer in range(45):
            p0 = 0
            for ordinal, width in enumerate(chunks):
                if layer >= 3:
                    lines.append(cache(layer, width))
                lines.append(forward(width, p0, int(layer == 44 and ordinal == len(chunks) - 1)))
                p0 += width
        lines.append(record("[GLM-TEACHER-PP-END]", dict(case=name, head=1872)))
        for row in result["rows"]:
            lines.append(record("[GLM-TEACHER-N1-BEGIN]", dict(case=name, index=row["index"], p0=row["position"], target=row["target"],
                history_sha=row["history_sha256"], logits_sha=row["score"]["logits_sha256"])))
            lines.extend(cache(layer, 1) for layer in range(3, 45))
            lines.append(forward(1, row["position"], 1))
            lines.append(record("[GLM-TEACHER-N1-END]", dict(case=name, index=row["index"], p0=row["position"], target=row["target"], head=row["head_after"])))
        lines.append(record("[GLM-TEACHER-CASE-END]", dict(case=name, profile=report["profile"], tokens=len(result["rows"]), outcome="completed")))
    lines.append(record("[GLM-TEACHER-DONE]", dict(profile=report["profile"], cases=4, checked_model_free=1, scope=gate.SCOPE)))
    return "\n".join(lines)


class TeacherDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.inputs = inputs_fixture()
        self.fixed = report_fixture(self.inputs, "pp512")
        self.adaptive = report_fixture(self.inputs, "adaptive1024", .125)
        self.fixed_log = log_fixture(self.inputs, self.fixed, [512, 512, 512, 336])
        self.adaptive_log = log_fixture(self.inputs, self.adaptive, [898, 898, 76])

    def compare(self, **changes):
        values = dict(inputs=self.inputs, inputs_hash="c" * 64, fixed=self.fixed, fixed_log=self.fixed_log,
                      adaptive=self.adaptive, adaptive_log=self.adaptive_log)
        values.update(changes)
        return gate.compare(**values)

    def test_complete_pair_preserves_negative_observation_and_no_quality_credit(self):
        result = self.compare()
        self.assertTrue(result["diagnostic_complete"])
        self.assertAlmostEqual(.125, result["delta_mean_nll"])
        self.assertEqual(10, result["teacher_tokens"])
        for key in ("general_quality_qualified", "strict_canonical_api_qualified", "performance_qualified",
                    "binary_immutability_externally_qualified", "test_runner_zero_skips_externally_qualified"):
            self.assertIs(False, result[key])
        self.assertEqual(168, result["actual_chunk_and_source_closure"]["pp512"][0]["selected_source_frames"])
        self.assertEqual(126, result["actual_chunk_and_source_closure"]["adaptive1024"][0]["selected_source_frames"])

    def test_actual_adaptive_width_is_not_hardwired_to_898(self):
        for chunks in ([1024, 848], [900, 900, 72], [600, 600, 600, 72]):
            with self.subTest(chunks=chunks):
                result = self.compare(adaptive_log=log_fixture(self.inputs, self.adaptive, chunks))
                self.assertEqual(chunks, result["actual_chunk_and_source_closure"]["adaptive1024"][0]["actual_pp_chunks"])

    def test_requested_adaptive_without_actual_wide_call_refuses(self):
        with self.assertRaises(ValueError):
            self.compare(adaptive_log=log_fixture(self.inputs, self.adaptive, [512, 512, 512, 336]))

    def test_actual_wide_over_1024_or_wrong_sum_refuses(self):
        for chunks in ([1100, 772], [898, 898, 75]):
            with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                self.compare(adaptive_log=log_fixture(self.inputs, self.adaptive, chunks))

    def test_all42_sources_required_no_count_only_duplicate_acceptance(self):
        for bad in (self.fixed_log.replace(cache(4, 512), cache(3, 512), 1),
                    self.fixed_log.replace(cache(44, 336) + "\n", "", 1)):
            with self.subTest(), self.assertRaises(ValueError):
                self.compare(fixed_log=bad)

    def test_identical_perlayer_chunk_sequence_is_required(self):
        bad = self.adaptive_log.replace(cache(4, 898), cache(4, 897), 1)
        with self.assertRaises(ValueError):
            self.compare(adaptive_log=bad)

    def test_missing_or_extra_dense_layer_forward_refuses(self):
        for bad in (self.fixed_log.replace(forward(512, 0, 0) + "\n", "", 1),
                    self.fixed_log.replace(forward(512, 0, 0), forward(512, 0, 0) + "\n" + forward(512, 0, 0), 1)):
            with self.subTest(), self.assertRaises(ValueError):
                self.compare(fixed_log=bad)

    def test_selected_source_requires_following_matching_timer(self):
        bad = self.fixed_log.replace(cache(3, 512) + "\n" + forward(512, 0, 0),
                                     forward(512, 0, 0) + "\n" + cache(3, 512), 1)
        with self.assertRaises(ValueError):
            self.compare(fixed_log=bad)

    def test_failed_forward_or_cache_has_no_diagnostic_credit(self):
        for tag in (gate.CACHE, gate.FORWARD):
            lines = self.fixed_log.splitlines()
            index = next(i for i, line in enumerate(lines) if i > 5 and tag in line)
            lines[index] = lines[index].replace("outcome=completed", "outcome=failed")
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                self.compare(fixed_log="\n".join(lines))

    def test_joined_integer_read_counters_and_bytes_are_exact(self):
        for original, replacement in (("file_read_completed_tasks=3", "file_read_completed_tasks=2"),
                ("file_read_calls=1", "file_read_calls=4"),
                ("file_read_source_bytes=100", "file_read_source_bytes=99"),
                ("file_view_buffered_bytes=100", "file_view_buffered_bytes=101"),
                ("file_read_tasks=3", "file_read_tasks=18446744073709551616")):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.compare(fixed_log=self.fixed_log.replace(original, replacement))

    def test_zero_byte_frame_cannot_claim_read_activity(self):
        bad = self.fixed_log.replace("file_bytes=100", "file_bytes=0").replace("file_read_source_bytes=100", "file_read_source_bytes=0").replace("file_view_buffered_bytes=100", "file_view_buffered_bytes=0")
        with self.assertRaises(ValueError):
            self.compare(fixed_log=bad)

    def test_n1_layer_order_position_and_timer_are_not_set_only(self):
        for bad in (self.fixed_log.replace(cache(4, 1), cache(3, 1), 1),
                    self.fixed_log.replace(forward(1, 1872, 1), forward(1, 1873, 1), 1),
                    self.fixed_log.replace(forward(1, 1872, 1) + "\n", "", 1)):
            with self.subTest(), self.assertRaises(ValueError):
                self.compare(fixed_log=bad)

    def test_missing_done_or_case_end_never_qualifies(self):
        for suffix in ("[GLM-TEACHER-DONE]", "[GLM-TEACHER-CASE-END]"):
            bad = "\n".join(line for line in self.fixed_log.splitlines() if suffix not in line)
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                self.compare(fixed_log=bad)

    def test_native_work_outside_step_or_mixed_case_markers_refuses(self):
        bad = self.fixed_log.replace("[GLM-TEACHER-PP-END] case=math head=1872",
            "[GLM-TEACHER-PP-END] case=math head=1872\n" + cache(3, 1))
        with self.assertRaises(ValueError):
            self.compare(fixed_log=bad)
        with self.assertRaises(ValueError):
            self.compare(fixed_log=self.fixed_log.replace("[GLM-TEACHER-N1-END] case=math", "[GLM-TEACHER-N1-END] case=code", 1))

    def test_wrong_forced_target_head_or_input_history_refuses(self):
        for key, value in (("sampled_token", 99), ("head_after", 1872), ("tokens_forwarded", 0), ("history_sha256", "d" * 64)):
            bad = copy.deepcopy(self.fixed); bad["cases"][0]["rows"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare(fixed=bad)

    def test_score_is_from_full_finite_distribution_for_exact_target(self):
        for key, value in (("target", 99), ("argmax", 154880), ("target_rank", 0), ("nll", float("nan")),
                           ("target_log_probability", 1), ("top_margin", -1), ("logits_sha256", "not-sha")):
            bad = copy.deepcopy(self.fixed); bad["cases"][0]["rows"][0]["score"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare(fixed=bad)

    def test_token_weighted_aggregation_not_equal_weighted_case_average(self):
        bad = copy.deepcopy(self.fixed)
        bad["mean_nll"] = sum(x["mean_nll"] for x in bad["cases"]) / 4
        self.assertNotEqual(self.fixed["mean_nll"], bad["mean_nll"])
        with self.assertRaises(ValueError):
            self.compare(fixed=bad)

    def test_boolean_integer_and_nonfinite_scores_rejected(self):
        bad = copy.deepcopy(self.fixed); bad["cases"][0]["rows"][0]["head_after"] = True
        with self.assertRaises(ValueError):
            self.compare(fixed=bad)
        with self.assertRaises(ValueError):
            gate.integer(True)
        with self.assertRaises(ValueError):
            gate.number(float("inf"))

    def test_binary_labels_or_other_experiments_different_refuse(self):
        bad = copy.deepcopy(self.adaptive); bad["models_mvid"] = "different"
        with self.assertRaises(ValueError):
            self.compare(adaptive=bad)
        for key in ("TS_HOST_MOE_CAUSAL_ROUTE_NEIGHBOR", "TS_HOST_MOE_FILE_SOURCE_VIEW", "TS_GLM_HOST_TOKEN_EMBEDDING"):
            bad = copy.deepcopy(self.adaptive); bad["environment"][key] = "1"
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare(adaptive=bad)

    def test_missing_environment_and_requested_resolved_confusion_refuse(self):
        bad = copy.deepcopy(self.fixed); del bad["environment"]["TS_HOST_MOE_FILE_READ_SORT"]
        with self.assertRaises(ValueError):
            self.compare(fixed=bad)
        bad = copy.deepcopy(self.adaptive); bad["scheduler_chunk_env"] = "1024"
        with self.assertRaises(ValueError):
            self.compare(adaptive=bad)

    def test_incomplete_cleanup_or_broader_quality_credit_refuse(self):
        for key, value in (("checked_model_free_and_ordinary_scope_drained", False), ("checked_cache_release_attachment", True),
                           ("ledger_committed", 1), ("active_allocations", 1),
                           ("general_quality_qualified", True), ("strict_canonical_api_qualified", True)):
            bad = copy.deepcopy(self.fixed); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare(fixed=bad)

    def test_inputs_token_hash_catalog_and_fixed_teacher_are_immutable(self):
        mutations = [lambda x: x["cases"][0]["prompt_ids"].__setitem__(0, 66),
                     lambda x: x["cases"][0].__setitem__("teacher_text", "chosen after result"),
                     lambda x: x["source_catalog"][0].__setitem__("gate_bytes", 101),
                     lambda x: x["source_catalog"].__setitem__(1, copy.deepcopy(x["source_catalog"][0]))]
        for mutate in mutations:
            bad = copy.deepcopy(self.inputs); mutate(bad)
            with self.subTest(), self.assertRaises(ValueError):
                self.compare(inputs=bad)

    def test_duplicate_record_field_or_duplicate_case_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compare(fixed_log=self.fixed_log.replace("layer=3 rows=512", "layer=3 layer=3 rows=512", 1))
        bad = copy.deepcopy(self.fixed); bad["cases"][1]["name"] = "math"
        with self.assertRaises(ValueError):
            self.compare(fixed=bad)

    def test_json_duplicate_and_oversize_provenance_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"; path.write_text('{"schema":1,"schema":1}')
            with self.assertRaises(ValueError):
                gate.read_json(path)
        with self.assertRaises(ValueError):
            gate.token_hash([0] * 2049)

    def test_log_bound_and_record_unsigned_syntax_are_strict(self):
        with self.assertRaises(ValueError):
            gate.validate_log("x" * (gate.MAX_BYTES + 1), self.fixed, self.inputs, "c" * 64)
        for value in ("-1", "1.0", "True", "01", "18446744073709551616"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gate.uint({"value": value}, "value")


if __name__ == "__main__":
    unittest.main()
