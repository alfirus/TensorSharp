import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location("unified_perf", Path(__file__).parents[1] / "unified-perf-bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class ResponseComparisonTests(unittest.TestCase):
    def setUp(self):
        self.request = {"messages": [{"role": "user", "content": "6 * 7?"}], "think": False}
        self.response = {"message": {"role": "assistant", "content": "42", "thinking": "Multiply."},
                         "done": True, "done_reason": "stop", "prompt_eval_count": 12,
                         "eval_count": 4, "eval_duration": 1234}
        self.report = {"records": [{"round": 0, "name": "math",
            "response_identity": bench.response_identity(self.request, self.response)}]}

    def test_ignores_timings_but_retains_reasoning_tools_and_completion(self):
        response = copy.deepcopy(self.response)
        response["eval_duration"] = 9999
        self.assertEqual(bench.response_identity(self.request, response), self.report["records"][0]["response_identity"])
        for field, value in (("thinking", "Different"), ("tool_calls", [{"function": {"name": "x"}}])):
            changed = copy.deepcopy(self.report)
            changed["records"][0]["response_identity"]["message"][field] = value
            self.assertFalse(bench.compare_responses(changed, self.report)["passed"])
        for field in ("done", "done_reason", "eval_tokens", "prompt_tokens", "request_sha256"):
            changed = copy.deepcopy(self.report)
            changed["records"][0]["response_identity"][field] = "different"
            if field == "done_reason":
                self.assertFalse(bench.compare_responses(changed, self.report)["passed"])
            else:
                with self.assertRaises(ValueError):
                    bench.compare_responses(changed, self.report)

    def test_refuses_empty_duplicate_incomplete_or_mismatched_requests(self):
        for rows in ([], [{"round": 0, "name": "math"}],
                     [{"round": 0, "name": "math", "response_identity": {}}], self.report["records"] * 2,
                     [dict(self.report["records"][0], name="different")]):
            with self.assertRaises(ValueError):
                bench.compare_responses({"records": rows}, self.report)

    def test_exact_response_passes_and_request_hash_ignores_key_order(self):
        self.assertTrue(bench.compare_responses(self.report, self.report)["passed"])
        reversed_request = dict(reversed(list(self.request.items())))
        self.assertEqual(bench.response_identity(reversed_request, self.response),
                         bench.response_identity(self.request, self.response))


class DiskObservationTests(unittest.TestCase):
    def test_reports_each_devices_delta_without_inventing_model_attribution(self):
        before = {"disk0": {"read_bytes": 100, "read_count": 10}, "disk1": {"read_bytes": 9, "read_count": 1}}
        after = {"disk0": {"read_bytes": 140, "read_count": 12}, "disk1": {"read_bytes": 9, "read_count": 1}}
        self.assertEqual(bench.disk_read_delta(before, after),
                         {"disk0": {"read_bytes": 40, "read_count": 2}, "disk1": {"read_bytes": 0, "read_count": 0}})

    def test_unavailable_reset_or_changed_devices_never_count_as_zero_io(self):
        observation = {"disk0": {"read_bytes": 100, "read_count": 10}}
        for before, after in ((None, observation), ({}, {}), (observation, {}),
                              (observation, {"disk0": {"read_bytes": 99, "read_count": 11}}),
                              (observation, {"disk1": {"read_bytes": 100, "read_count": 10}})):
            self.assertIsNone(bench.disk_read_delta(before, after))
        with patch.object(bench, "psutil", None):
            self.assertIsNone(bench.disk_read_snapshot())


class SemanticQualificationTests(unittest.TestCase):
    def test_speculation_requires_actual_logged_steps_not_the_command_flag(self):
        self.assertEqual(bench.speculation_records("--spec enabled; plain fallback"), [])
        line = "Speculative decoding stats for chat-test: drafted=8 accepted=5 acceptance=62 % verifySteps=4 plainSteps=3 rollbacks=2 | phaseMs draft=9"
        self.assertEqual(bench.speculation_records(line), [{"request": "chat-test", "drafted": 8,
            "accepted": 5, "verify_steps": 4, "plain_steps": 3, "rollbacks": 2}])

    def test_speculation_costs_are_scoped_to_each_complete_phase_record(self):
        counts = "Speculative decoding stats for chat-test: drafted=8 accepted=5 acceptance=62 % verifySteps=4 plainSteps=3 rollbacks=2"
        full = counts + " | phaseMs draft=9 verify=100 snapshot=2 rollback=60 catchUp=4 plain=30 | governor plain=10.0ms/tok"
        records = bench.speculation_records(full + "\n" + counts + " | phaseMs draft=999")
        self.assertEqual(records[0]["phase_ms"], {"draft": 9, "verify": 100, "snapshot": 2,
                                                "rollback": 60, "catchUp": 4, "plain": 30})
        self.assertNotIn("phase_ms", records[1])

    def test_requires_known_answer_complete_generation_and_matching_engine_eos(self):
        response = {"message": {"content": "42"}, "done": True, "done_reason": "stop",
                    "prompt_eval_count": 12, "eval_count": 3}
        log = "generation.timings promptTokens=12 generatedTokens=3 finishReason=eos engineCompletionAvailable=True"
        prompt = bench.FIXED_TASKS.PROMPTS["math"]
        self.assertTrue(bench.semantic_result("math", prompt, response, log)["passed"])
        for invalid in ("", log + "\n" + log, log.replace("eos", "max_tokens"),
                        log.replace("Tokens=3", "Tokens=4"), log.replace("True", "False")):
            self.assertFalse(bench.semantic_result("math", prompt, response, invalid)["passed"])
        self.assertFalse(bench.semantic_result("math", "Different prompt", response, log)["passed"])
        for change in ({"done": False}, {"message": {"content": "43"}},
                       {"message": {"content": "42", "tool_calls": [{}]}}, {"done_reason": "length"}):
            self.assertFalse(bench.semantic_result("math", prompt, response | change, log)["passed"])


class CacheTimingTests(unittest.TestCase):
    def test_keeps_width_layer_and_failure_distinct_and_does_not_sum_overlapping_stages(self):
        line = "[HOSTMOE-CACHE-TIMING] layer=3 rows=1 outcome=completed total_ms=9 plan_ms=1 allocation_ms=0 wide_replace_ms=0 file_pipeline_ms=6 file_bytes=64 file_windows=1 upload_enqueue_ms=2 upload_wait_ms=1 input_bridge_ms=0 compute_ms=4 graph_submissions=1 output_bridge_ms=0"
        result = bench.cache_timing_summary("\n".join((line, line, line.replace("rows=1", "rows=3"),
            line.replace("completed", "exception"), line.replace("file_bytes=64", "file_bytes=oops"))))
        self.assertEqual(result["malformed_records"], 1)
        groups = {(g["rows"], g["outcome"]): g for g in result["groups"]}
        self.assertEqual(groups[(1, "completed")]["calls"], 2)
        self.assertEqual(groups[(1, "completed")]["total_ms"], 18)
        self.assertEqual(groups[(1, "completed")]["file_bytes"], 128)
        self.assertEqual(groups[(1, "completed")]["file_windows"], 2)
        self.assertEqual(groups[(3, "completed")]["calls"], 1)
        self.assertEqual(groups[(1, "exception")]["calls"], 1)
        self.assertEqual(bench.cache_timing_summary("")["groups"], [])
        self.assertEqual(bench.cache_timing_summary(line.replace("file_windows=1", ""))["malformed_records"], 1)


class ForwardTimingTests(unittest.TestCase):
    def test_coalescing_requires_fewer_completed_reads_with_exact_payload_and_actual_ring(self):
        line = ("[HOSTMOE-CACHE-TIMING] rows=512 bulk=1 joint=0 outcome=completed file_bytes=200000000 "
                "ring_calls=1 ring_expanded_calls=1 ring_arena_peak_bytes=75497472 ring_slot_peak_bytes=37748736 "
                "ring_coalesced_calls=1 ring_read_tasks_before=100 ring_read_tasks_after=30 "
                "ring_coalesced_payload_bytes=200000000 ring_coalesced_bytes=190000000 "
                "ring_coalesced_read_cap_bytes=16777216")
        result = bench.coalesced_ring_summary("\n".join((line, line, line.replace("completed", "exception"))))
        self.assertEqual(result["completed_coalesced_calls"], 2)
        self.assertEqual(result["read_tasks_before"], 200)
        self.assertEqual(result["read_tasks_after"], 60)
        self.assertEqual(result["logical_payload_bytes"], 400000000)
        self.assertEqual(result["logical_merged_bytes"], 380000000)
        self.assertEqual(result["malformed_records"], 0)
        invalid = (line.replace("ring_read_tasks_after=30", "ring_read_tasks_after=100"),
                   line.replace("file_bytes=200000000", "file_bytes=200000001"),
                   line.replace("ring_coalesced_bytes=190000000", "ring_coalesced_bytes=200000001"),
                   line.replace("16777216", "2097152"), line.replace("rows=512", "rows=1"),
                   line.replace("37748736", "524288"))
        self.assertEqual(bench.coalesced_ring_summary("\n".join(invalid))["malformed_records"], 6)
        self.assertEqual(bench.coalesced_ring_summary("[HOSTMOE-CACHE-TIMING] rows=512")["completed_coalesced_calls"], 0)

    def test_prefill_ring_requires_actual_completed_wide_plans_not_retained_storage(self):
        line = ("[HOSTMOE-CACHE-TIMING] rows=512 bulk=1 joint=0 outcome=completed "
                "file_bytes=200000000 file_windows=6 ring_calls=1 ring_expanded_calls=1 "
                "ring_arena_peak_bytes=75497472 ring_slot_peak_bytes=37748736")
        tiny = line.replace("ring_expanded_calls=1", "ring_expanded_calls=0").replace(
            "75497472", "1048576").replace("37748736", "524288")
        excluded = (tiny, line.replace("completed", "exception"), line.replace("rows=512", "rows=1"),
                    line.replace("bulk=1", "bulk=0"), line.replace("joint=0", "joint=1"))
        self.assertEqual(bench.prefill_ring_summary("\n".join(excluded))["expanded_ring_calls"], 0)
        result = bench.prefill_ring_summary("\n".join((line, line, *excluded)))
        self.assertEqual(result["expanded_ring_calls"], 2)
        self.assertEqual(result["completed_ring_calls"], 3)
        self.assertEqual(result["logical_file_bytes_in_expanded_calls"], 400000000)
        self.assertEqual(result["windows_in_expanded_calls"], 12)
        self.assertEqual(result["actual_slot_peak_bytes"], 36 * 2**20)
        self.assertEqual(result["malformed_records"], 0)
        invalid = (line.replace("ring_calls=1", "ring_calls=0"), line.replace("37748736", "1"),
                   tiny.replace("ring_expanded_calls=0", "ring_expanded_calls=1"),
                   line.replace("ring_slot_peak_bytes=37748736", ""))
        self.assertEqual(bench.prefill_ring_summary("\n".join(invalid))["malformed_records"], 4)
        self.assertEqual(bench.prefill_ring_summary("[HOSTMOE-CACHE-TIMING] rows=512")["expanded_ring_calls"], 0)

    def test_cold_growth_preserves_actual_capacity_and_refusal_observations(self):
        line = "[HOSTMOE-PHASE-COLD-GROW] layer=4 slots=353->417 bytes=2305709312->2721038336 lost_hot_slots=300 reload=immutable_source fallback=0 reserved=7433841152 budget=7435452416"
        fallback = line.replace("353->417", "353->353").replace("2305709312->2721038336", "2305709312->2305709312").replace("fallback=0", "fallback=1")
        result = bench.phase_cold_growth_records(line + "\n" + fallback)
        self.assertEqual(result["malformed_records"], 0)
        self.assertEqual(result["records"][0]["new_slots"], 417)
        self.assertEqual(result["records"][1]["fallback"], 1)
        invalid = (line.replace("lost_hot_slots=300", "lost_hot_slots=400"),
                   line.replace("budget=7435452416", "budget=1"), line.replace("353->417", "353->1"),
                   line.replace("fallback=0", "fallback=oops"))
        self.assertEqual(bench.phase_cold_growth_records("\n".join(invalid))["malformed_records"], 4)
        self.assertEqual(bench.phase_cold_growth_records("")["records"], [])

    def test_scalar_staging_requires_an_actual_completed_large_single_window(self):
        line = "[HOSTMOE-CACHE-TIMING] rows=1 outcome=completed file_windows=1 file_bytes=40000000"
        invalid = (line.replace("rows=1", "rows=2"), line.replace("completed", "exception"),
                   line.replace("file_windows=1", "file_windows=2"), line.replace("40000000", "33554432"),
                   line.replace("40000000", "80000000"), line.replace("40000000", "oops"))
        self.assertEqual(bench.scalar_staging_summary("\n".join(invalid))["expanded_scalar_calls"], 0)
        result = bench.scalar_staging_summary("\n".join((line, line, *invalid)))
        self.assertEqual(result["expanded_scalar_calls"], 2)
        self.assertEqual(result["logical_file_bytes"], 80000000)

    def test_retains_failed_calls_and_positions_without_double_counting_nested_work(self):
        line = "[GLM-MTP-FORWARD-TIMING] kind=0 rows=2 p0=33 nout=2 wanth=1 logits=1 reused=1 outcome=completed acquire_ms=1 input_ms=2 compute_ms=90 output_ms=7 total_ms=100"
        result = bench.forward_timing_records("\n".join((line, line.replace("completed", "failed"),
            line.replace("total_ms=100", "total_ms=101"), line.replace("compute_ms=90", "compute_ms=nan"),
            line.replace("p0=33", "p0=-1"), "[HOSTMOE-CACHE-TIMING] total_ms=60")))
        self.assertEqual(result["malformed_records"], 3)
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(result["records"][0]["p0"], 33)
        self.assertEqual(result["records"][0]["compute_ms"], 90)
        self.assertEqual(result["records"][1]["outcome"], "failed")
        self.assertEqual(bench.forward_timing_records("")["records"], [])


class VictimCacheQualificationTests(unittest.TestCase):
    line = ("[HOSTMOE-RAM-VICTIM] candidates=10 eligible=8 admitted=7 refusals=1 "
            "d2h_bytes=100 completed_bytes=100 promoted_bytes=50 failures=0 "
            "metadata_bytes=20 payload_bytes=200 configured_bytes=300 history_events=99 history_keys=8")

    def test_request_delta_excludes_startup_and_gauges_can_shrink(self):
        after = self.line.replace("=100", "=160").replace("promoted_bytes=50", "promoted_bytes=90").replace("payload_bytes=200", "payload_bytes=150")
        result = bench.victim_cache_summary(self.line, self.line + "\n" + after)
        self.assertEqual(result["delta"]["completed_bytes"], 60)
        self.assertEqual(result["delta"]["promoted_bytes"], 40)
        self.assertEqual(result["delta"]["failures"], 0)
        self.assertEqual(result["after"]["payload_bytes"], 150)

    def test_missing_malformed_and_reset_counters_never_become_zero_usage(self):
        for before, after in (("", self.line), (self.line, ""),
                              (self.line, self.line.replace("completed_bytes=100", "completed_bytes=99")),
                              (self.line, self.line.replace("promoted_bytes=50", "promoted_bytes=oops")),
                              (self.line, self.line.replace("failures=0", "failures=-1"))):
            self.assertIsNone(bench.victim_cache_summary(before, after)["delta"])

    def test_optional_allocator_telemetry_requires_complete_monotonic_snapshots(self):
        storage = (" allocation_attempts=4 allocation_bytes=400 allocation_ns=200 pool_reuses=2 "
                   "resident_reuses=3 release_calls=1 release_ns=50 recycled_bytes=100 "
                   "free_bytes=20 free_entries=1 resident_bytes=180 resident_entries=9")
        before = self.line + storage
        after = before.replace("allocation_ns=200", "allocation_ns=260").replace("pool_reuses=2", "pool_reuses=5").replace("free_bytes=20", "free_bytes=10")
        result = bench.victim_cache_summary(before, after)
        self.assertEqual(result["storage_delta"]["allocation_ns"], 60)
        self.assertEqual(result["storage_delta"]["pool_reuses"], 3)
        self.assertEqual(result["after"]["free_bytes"], 10)
        self.assertIsNone(bench.victim_cache_summary(self.line, self.line)["storage_delta"])
        for invalid in (after.replace(" release_ns=50", ""),
                        after.replace("release_ns=50", "release_ns=oops"),
                        after.replace("release_ns=50", "release_ns=49")):
            self.assertIsNone(bench.victim_cache_summary(before, invalid)["storage_delta"])


class MappedRamObservationTests(unittest.TestCase):
    def setUp(self):
        self.values = dict(capacity_charge_bytes=1 << 20, logical_resident_bytes=8192,
            owned_charge_bytes=8192 + 4096 + 65536, locked_page_bytes=8192,
            metadata_charge_bytes=4096 + 65536, resident_entries=1, active_leases=0,
            quarantine_charge_bytes=0, hits=10, misses=3, hit_copy_bytes=4096,
            admitted_logical_bytes=8192, bypass_logical_bytes=1024, evictions=1,
            pending_admissions=0, quota_restore_pending=1, quota_conflict=0, cleanup_failures=0)

    def line(self, **changes):
        values = self.values | changes
        return "[HOSTMOE-RAM-MAPPED] " + " ".join(f"{k}={v}" for k, v in values.items())

    def test_actual_monotonic_copies_exclude_prior_warmup(self):
        result = bench.mapped_ram_summary(self.line(), self.line(hits=12, hit_copy_bytes=6144))
        self.assertEqual(result["delta"]["hits"], 2)
        self.assertEqual(result["delta"]["hit_copy_bytes"], 2048)
        self.assertEqual(result["delta"]["admitted_logical_bytes"], 0)
        self.assertEqual(result["after"]["locked_page_bytes"], 8192)

    def test_missing_snapshots_and_counter_resets_are_unavailable(self):
        for before, after in (("", self.line()), (self.line(), ""),
                              (self.line(), self.line(hit_copy_bytes=1))):
            self.assertIsNone(bench.mapped_ram_summary(before, after)["delta"])

    def test_incomplete_invalid_and_malformed_prefix_cannot_be_hidden(self):
        invalid = [self.line().replace(" locked_page_bytes=8192", ""), self.line(locked_page_bytes=-1),
                   self.line(owned_charge_bytes=8192), self.line(quarantine_charge_bytes=1 << 20),
                   self.line(quota_conflict=2), self.line(active_leases="oops")]
        for line in invalid:
            result = bench.mapped_ram_summary(self.line(), line + "\n" + self.line())
            self.assertIsNone(result["delta"])
            self.assertEqual(result["malformed_records"], 1)

    def test_cleanup_retained_ownership_is_observable_not_zeroed(self):
        result = bench.mapped_ram_summary(self.line(), self.line(capacity_charge_bytes=0,
            logical_resident_bytes=0, resident_entries=0, quarantine_charge_bytes=12288))
        self.assertEqual(result["after"]["owned_charge_bytes"], 77824)
        self.assertEqual(result["after"]["quarantine_charge_bytes"], 12288)


class MappedRamCostFixture:
    def setUp(self):
        self.values = dict.fromkeys(bench.MAPPED_RAM_COST_COUNTERS + bench.MAPPED_RAM_COST_GAUGES, 10)
        self.values.update(admission_calls=10, admission_successes=8,
            guarded_copy_calls=10, guarded_copy_failures=2, guarded_copy_bytes=9000,
            probation_observations=100, probation_firsts=70, probation_repeats=30,
            peak_claim_bytes=65536, quota_min_high_water_bytes=131072,
            probation_keys=100, probation_metadata_bytes=1 << 20)

    def line(self, omit=(), **changes):
        values = self.values | changes
        return "[HOSTMOE-RAM-MAPPED-COST] " + " ".join(
            f"{key}={value}" for key, value in values.items() if key not in omit)


class MappedRamCostTests(MappedRamCostFixture, unittest.TestCase):
    def test_required_cost_fields_match_native_producer_contract(self):
        root = Path(__file__).resolve().parents[3]
        native = (root / "TensorSharp.GGML.Native/ggml_ops_host_moe_mapped_ram.h").read_text()
        macro = native.split("#define TSG_MAPPED_RAM_METRICS(X)", 1)[1].split("// Optional immutable", 1)[0]
        native_fields = re.findall(r"X\((\w+)\)", macro)
        self.assertEqual(len(native_fields), len(set(native_fields)))
        self.assertEqual(set(native_fields) | {"probation_observations", "probation_firsts", "probation_repeats",
                                              "probation_keys", "probation_metadata_bytes"}, set(self.values))

    def test_request_delta_excludes_warmup_and_keeps_gauges_out_of_counters(self):
        middle = self.line(plan_calls=12, guarded_copy_bytes=9500,
                           probation_observations=103, probation_firsts=72, probation_repeats=31)
        after = self.line(plan_calls=15, guarded_copy_bytes=11000, probation_keys=3,
                          probation_observations=107, probation_firsts=73, probation_repeats=34,
                          peak_claim_bytes=98304)
        result = bench.mapped_ram_cost_summary(self.line(), middle + "\n" + after)
        self.assertEqual(result["delta"]["plan_calls"], 5)
        self.assertEqual(result["delta"]["guarded_copy_bytes"], 2000)
        self.assertEqual(result["delta"]["probation_firsts"], 3)
        self.assertEqual(result["delta"]["probation_repeats"], 4)
        self.assertEqual(result["after"]["probation_keys"], 3)
        self.assertEqual(result["after"]["peak_claim_bytes"], 98304)
        self.assertFalse(set(bench.MAPPED_RAM_COST_GAUGES) & result["delta"].keys())
        self.assertEqual(result["reset_records"], 0)

    def test_missing_snapshots_and_each_required_field_fail_closed(self):
        for before, after in (("", self.line()), (self.line(), ""), ("", "")):
            with self.subTest(before=bool(before), after=bool(after)):
                self.assertIsNone(bench.mapped_ram_cost_summary(before, after)["delta"])
        for field in self.values:
            with self.subTest(field=field):
                result = bench.mapped_ram_cost_summary(self.line(), self.line(omit=(field,)) + "\n" + self.line())
                self.assertIsNone(result["delta"])
                self.assertEqual(result["malformed_records"], 1)

    def test_duplicate_and_non_uint64_fields_cannot_be_hidden_by_valid_tail(self):
        invalid = [self.line() + " plan_calls=10"]
        invalid += [self.line(plan_calls=value) for value in (-1, "+1", "1_0", "1.0", "nan", "oops", 2**64)]
        for line in invalid:
            with self.subTest(line=line):
                result = bench.mapped_ram_cost_summary(self.line(), line + "\n" + self.line())
                self.assertIsNone(result["delta"])
                self.assertEqual(result["malformed_records"], 1)

    def test_invalid_relations_and_unowned_history_are_malformed(self):
        invalid = [dict(admission_successes=11), dict(guarded_copy_failures=11),
                   dict(probation_observations=101), dict(probation_keys=4097),
                   dict(probation_metadata_bytes=1), dict(probation_metadata_bytes=0)]
        for changes in invalid:
            with self.subTest(changes=changes):
                result = bench.mapped_ram_cost_summary(self.line(), self.line(**changes))
                self.assertIsNone(result["delta"])
                self.assertEqual(result["malformed_records"], 1)

    def test_every_counter_reset_is_unavailable(self):
        for counter in bench.MAPPED_RAM_COST_COUNTERS:
            with self.subTest(counter=counter):
                self.assertIsNone(bench.mapped_ram_cost_summary(self.line(), self.line(**{counter: 0}))["delta"])

    def test_intermediate_reset_and_recovery_cannot_hide_in_either_log(self):
        for before, request in (
                (self.line(), self.line(plan_calls=0) + "\n" + self.line(plan_calls=20)),
                (self.line() + "\n" + self.line(plan_calls=0) + "\n" + self.line(plan_calls=20),
                 self.line(plan_calls=30))):
            result = bench.mapped_ram_cost_summary(before, request)
            self.assertIsNone(result["delta"])
            self.assertEqual(result["reset_records"], 1)
            self.assertEqual(result["malformed_records"], 0)

    def test_high_water_decrease_rejects_reset_but_cleanup_gauges_may_shrink(self):
        for field in bench.MAPPED_RAM_COST_HIGH_WATERS:
            with self.subTest(field=field):
                result = bench.mapped_ram_cost_summary(self.line(), self.line(**{field: 0}) + "\n" + self.line())
                self.assertIsNone(result["delta"])
                self.assertEqual(result["reset_records"], 1)
        result = bench.mapped_ram_cost_summary(self.line(), self.line(probation_keys=0, probation_metadata_bytes=0))
        self.assertIsNotNone(result["delta"])
        self.assertEqual(result["after"]["probation_keys"], 0)
        self.assertEqual(result["reset_records"], 0)

    def test_unknown_extra_telemetry_does_not_invalidate_complete_contract(self):
        result = bench.mapped_ram_cost_summary(self.line() + " future_counter=4",
                                               self.line() + " future_counter=5 scope=nested_wall")
        self.assertIsNotNone(result["delta"])


class MappedProbationQualificationTests(MappedRamCostFixture, unittest.TestCase):
    def records(self, before, *requests):
        records = []
        for request in requests:
            records.append(dict(mapped_ram_cost=bench.mapped_ram_cost_summary(before, request)))
            before += "\n" + request
        return records

    def test_first_and_repeat_activity_aggregate_across_complete_measured_requests(self):
        first = self.line(probation_observations=103, probation_firsts=73)
        repeated = self.line(probation_observations=107, probation_firsts=73, probation_repeats=34)
        q = bench.mapped_probation_qualification(self.records(self.line(), first, repeated, repeated))
        self.assertTrue(q["passed"])
        self.assertTrue(q["attributed"])
        self.assertEqual((q["observations"], q["firsts"], q["repeats"]), (7, 3, 4))

    def test_enable_flag_warmup_only_or_one_kind_of_activity_cannot_qualify(self):
        for records in ([], [{}], self.records(self.line(), self.line()),
                        self.records(self.line(), self.line(probation_observations=103, probation_firsts=73)),
                        self.records(self.line(), self.line(probation_observations=104, probation_repeats=34))):
            with self.subTest(records=records):
                self.assertFalse(bench.mapped_probation_qualification(records)["passed"])

    def test_missing_malformed_or_recovered_reset_request_blocks_whole_profile(self):
        active = self.line(probation_observations=107, probation_firsts=73, probation_repeats=34)
        invalid = ["", self.line(plan_calls="oops") + "\n" + active,
                   self.line(plan_calls=0) + "\n" + active]
        for request in invalid:
            with self.subTest(request=request):
                q = bench.mapped_probation_qualification(self.records(self.line(), request))
                self.assertFalse(q["passed"])
                self.assertFalse(q["attributed"])
        records = self.records(self.line(), active) + [{}]
        self.assertFalse(bench.mapped_probation_qualification(records)["passed"])

    def test_activity_without_retained_charged_metadata_does_not_qualify(self):
        after = self.line(probation_observations=107, probation_firsts=73, probation_repeats=34,
                          probation_keys=0, probation_metadata_bytes=0)
        q = bench.mapped_probation_qualification(self.records(self.line(), after))
        self.assertTrue(q["attributed"])
        self.assertFalse(q["passed"])


class MappedProbationCliTests(unittest.TestCase):
    class ReachedProvenance(Exception):
        pass

    def invoke(self, arguments, expected_exception):
        argv = ["unified-perf-bench.py", "--model", "not-opened.gguf", "--output", "not-created"] + arguments
        with patch.object(bench.sys, "argv", argv), patch.object(bench.Path, "mkdir"), \
                patch.dict(bench.os.environ, {bench.STRICT_GRAPH_CAP_ENV: "0"}), \
                patch.object(bench.socket, "socket") as socket_mock, \
                patch.object(bench.subprocess, "Popen") as launch, \
                patch.object(bench, "benchmark_provenance", side_effect=self.ReachedProvenance) as provenance, \
                contextlib.redirect_stderr(io.StringIO()):
            socket_mock.return_value.__enter__.return_value.connect_ex.return_value = 1
            with self.assertRaises(expected_exception) as caught:
                bench.main()
            launch.assert_not_called()
            self.preflight_port = socket_mock.return_value.__enter__.return_value.connect_ex.call_args.args[0][1]
            return caught.exception, provenance.call_count

    def test_default_port_avoids_default_windows_client_range_and_preserves_override(self):
        _, calls = self.invoke([], self.ReachedProvenance)
        self.assertEqual(calls, 1)
        self.assertEqual(self.preflight_port, 42190)
        self.invoke(["--port", "43123"], self.ReachedProvenance)
        self.assertEqual(self.preflight_port, 43123)

    def test_flag_is_defined_and_valid_requirements_reach_prelaunch_boundary(self):
        _, calls = self.invoke(["--require-mapped-probation", "--unified-memory", "--env",
            "TS_HOST_MOE_RAM_CACHE_MAPPED=1", "--env", "TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS=1"], self.ReachedProvenance)
        self.assertEqual(calls, 1)

    def test_missing_mode_diagnostics_unified_or_isolation_refuse_before_server(self):
        for arguments in (["--require-mapped-probation"],
                          ["--require-mapped-probation", "--env", "TS_HOST_MOE_RAM_CACHE_MAPPED=0"],
                          ["--require-mapped-probation", "--env", "TS_HOST_MOE_RAM_CACHE_MAPPED=1"],
                          ["--require-mapped-probation", "--unified-memory", "--env", "TS_HOST_MOE_RAM_CACHE_MAPPED=1"],
                          ["--require-mapped-probation", "--unified-memory", "--concurrent", "--env",
                           "TS_HOST_MOE_RAM_CACHE_MAPPED=1", "--env", "TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS=1"]):
            with self.subTest(arguments=arguments):
                error, calls = self.invoke(arguments, SystemExit)
                self.assertEqual(error.code, 2)
                self.assertEqual(calls, 0)


class StrictGraphCapRecordsTests(unittest.TestCase):
    cap = 512 * 2**20

    @classmethod
    def line(cls, **changes):
        values = dict(profile=1, cap_bytes=cls.cap, submitted=1872, chunk=512, context=2048,
                      outcome="completed", graph_builds=42, graph_reuses=3,
                      peak_graph_device_bytes=400 * 2**20, cap_refusals=0,
                      scope=bench.STRICT_GRAPH_CAP_SCOPE)
        values.update(changes)
        return "[GLM-STRICT-GRAPH-CAP] " + " ".join(f"{key}={value}" for key, value in values.items())

    def row(self, log, cap=None):
        return dict(strict_layer_major_graph_cap=bench.strict_layer_major_graph_cap_records(log, cap or self.cap))

    def test_completed_calls_are_local_and_aggregate_with_blank_ordinary_decode(self):
        rows = [self.row(self.line()), self.row("ordinary short prefill/decode"),
                self.row(self.line(graph_builds=0, graph_reuses=45, peak_graph_device_bytes=300 * 2**20))]
        q = bench.strict_layer_major_graph_cap_qualification(rows, self.cap)
        self.assertTrue(q["passed"])
        self.assertEqual(q["completed_calls"], 2)
        self.assertEqual((q["graph_builds"], q["graph_reuses"]), (42, 48))
        self.assertEqual(q["peak_graph_device_bytes"], 400 * 2**20)
        self.assertNotIn("delta", rows[0]["strict_layer_major_graph_cap"])
        self.assertIn("not total VRAM", q["scope"])
        other = 768 * 2**20
        self.assertTrue(bench.strict_layer_major_graph_cap_qualification(
            [self.row(self.line(cap_bytes=other), other)], other)["passed"])

    def test_flag_startup_only_or_no_actual_completed_event_cannot_qualify(self):
        for rows in ([], [{}], [self.row("")], [self.row(self.line(outcome="failed", graph_builds=0,
                              graph_reuses=0, peak_graph_device_bytes=0, cap_refusals=1))]):
            with self.subTest(rows=rows):
                self.assertFalse(bench.strict_layer_major_graph_cap_qualification(rows, self.cap)["passed"])

    def test_failed_call_is_retained_and_blocks_valid_following_call(self):
        failed = self.line(outcome="failed", peak_graph_device_bytes=0, graph_builds=0, graph_reuses=0, cap_refusals=1)
        row = self.row(failed + "\n" + self.line())
        self.assertEqual(row["strict_layer_major_graph_cap"]["malformed_records"], 0)
        self.assertEqual(len(row["strict_layer_major_graph_cap"]["records"]), 2)
        q = bench.strict_layer_major_graph_cap_qualification([row], self.cap)
        self.assertEqual(q["failed_calls"], 1)
        self.assertFalse(q["passed"])

    def test_each_missing_duplicate_non_uint_or_overflow_field_is_malformed(self):
        invalid = []
        fields = bench.STRICT_GRAPH_CAP_NUMBERS + ("outcome", "scope")
        for field in fields:
            invalid.append(re.sub(r" " + field + r"=[^\s]+", "", self.line()))
            invalid.append(self.line() + f" {field}=1")
        for field in bench.STRICT_GRAPH_CAP_NUMBERS:
            for value in (-1, "+1", "1.0", "1_0", "oops", 2**64):
                invalid.append(self.line(**{field: value}))
        invalid += [self.line() + " partial", self.line() + " extra=1",
                    self.line(outcome="exception"), self.line(scope="total_vram"),
                    self.line(graph_builds=2**64 - 1, graph_reuses=1)]
        for line in invalid:
            with self.subTest(line=line):
                row = self.row(line + "\n" + self.line())
                self.assertEqual(row["strict_layer_major_graph_cap"]["malformed_records"], 1)
                self.assertFalse(bench.strict_layer_major_graph_cap_qualification([row], self.cap)["passed"])

    def test_completed_geometry_cap_payload_and_activity_are_checked(self):
        invalid = [dict(profile=0), dict(cap_bytes=768 * 2**20), dict(chunk=256), dict(context=4096),
                   dict(submitted=512), dict(submitted=513), dict(submitted=520), dict(submitted=1537),
                   dict(submitted=2057), dict(graph_builds=0, graph_reuses=0),
                   dict(peak_graph_device_bytes=0), dict(peak_graph_device_bytes=self.cap + 1), dict(cap_refusals=1)]
        for changes in invalid:
            with self.subTest(changes=changes):
                row = self.row(self.line(**changes))
                self.assertEqual(row["strict_layer_major_graph_cap"]["malformed_records"], 1)
                self.assertFalse(bench.strict_layer_major_graph_cap_qualification([row], self.cap)["passed"])
        for submitted in (521, 1024, 1033, 1536, 1872, 2048):
            with self.subTest(submitted=submitted):
                self.assertTrue(bench.strict_layer_major_graph_cap_qualification(
                    [self.row(self.line(submitted=submitted))], self.cap)["passed"])
        self.assertFalse(bench.strict_layer_major_graph_cap_qualification([self.row(self.line())], 768 * 2**20)["passed"])


class StrictGraphCapCliTests(unittest.TestCase):
    class ReachedProvenance(Exception):
        pass

    @staticmethod
    def valid_args(cap="512"):
        return ["--unified-memory", "--env", f"{bench.STRICT_GRAPH_CAP_ENV}={cap}", "--env",
                "TS_UNIFIED_MAX_RUNNING_REQUESTS=1", "--env", "TS_SCHED_PREFILL_CHUNK=512",
                "--env", "MAX_CONTEXT=2048"]

    def invoke(self, arguments, expected_exception, inherited=None):
        argv = ["unified-perf-bench.py", "--model", "not-opened.gguf", "--output", "not-created"] + arguments
        with patch.object(bench.sys, "argv", argv), patch.dict(bench.os.environ, inherited or {}, clear=True), \
                patch.object(bench.Path, "mkdir"), patch.object(bench.socket, "socket") as socket_mock, \
                patch.object(bench.subprocess, "Popen") as launch, \
                patch.object(bench, "benchmark_provenance", side_effect=self.ReachedProvenance) as provenance, \
                contextlib.redirect_stderr(io.StringIO()):
            socket_mock.return_value.__enter__.return_value.connect_ex.return_value = 1
            with self.assertRaises(expected_exception) as caught:
                bench.main()
            launch.assert_not_called()
            return caught.exception, provenance.call_count

    def test_enabled_512_768_and_disabled_zero_are_explicitly_supported(self):
        for cap in ("512", "768"):
            _, calls = self.invoke(self.valid_args(cap), self.ReachedProvenance)
            self.assertEqual(calls, 1)
        self.assertEqual(self.invoke(["--env", f"{bench.STRICT_GRAPH_CAP_ENV}=0"], self.ReachedProvenance)[1], 1)

    def test_invalid_value_mode_or_missing_explicit_geometry_refuses_before_provenance(self):
        invalid = [["--env", f"{bench.STRICT_GRAPH_CAP_ENV}={value}"]
                   for value in ("", "1", "513", "0512", "+512", "512.0", " 512")]
        invalid += [self.valid_args() + extra for extra in
                    (["--concurrent"], ["--spec"], ["--prefix-cache"], ["--repeat", "0"],
                     ["--server-arg=--spec"], ["--server-arg=--draft-model=fixture.gguf"])]
        invalid += [[arg for arg in self.valid_args() if arg != "--unified-memory"]]
        for key, value in (("TS_UNIFIED_MAX_RUNNING_REQUESTS", "1"), ("TS_SCHED_PREFILL_CHUNK", "512"),
                           ("MAX_CONTEXT", "2048")):
            args = self.valid_args()
            index = args.index(f"{key}={value}")
            invalid.append(args[:index - 1] + args[index + 1:])
            invalid.append(args + ["--env", f"{key}=2"])
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                error, calls = self.invoke(arguments, SystemExit)
                self.assertEqual(error.code, 2)
                self.assertEqual(calls, 0)

    def test_inherited_enabled_cap_cannot_bypass_qualification(self):
        error, calls = self.invoke([], SystemExit, {bench.STRICT_GRAPH_CAP_ENV: "512"})
        self.assertEqual((error.code, calls), (2, 0))

    def run_mocked_plan(self, startup_log, request_logs):
        # Exercise real main() log slicing, row insertion and final failure, with
        # all network/process/GPU/provenance observation replaced by small mocks.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            argv = ["unified-perf-bench.py", "--model", "not-opened.gguf", "--output", str(output),
                    "--plan", "quick"] + self.valid_args()
            process = Mock(returncode=0)
            process.poll.return_value = None
            sampler = SimpleNamespace(stop=Mock(), peak_rss=0, peak_private=0)
            logs = iter(request_logs)
            def launch(*args, **kwargs):
                kwargs["stdout"].write(startup_log + "\n")
                kwargs["stdout"].flush()
                return process
            def http(port, method, path, body, timeout):
                if method == "GET":
                    return {"data": [{"id": "synthetic-fixture"}]}
                with (output / "server.log").open("a", encoding="utf-8") as stream:
                    stream.write(next(logs) + "\n")
                return dict(message=dict(role="assistant", content="fixture"), done=True, done_reason="stop",
                            prompt_eval_count=1872, eval_count=2, prompt_eval_duration=100, eval_duration=100)
            with patch.object(bench.sys, "argv", argv), patch.dict(bench.os.environ, {}, clear=True), \
                    patch.object(bench.socket, "socket") as socket_mock, \
                    patch.object(bench.subprocess, "Popen", side_effect=launch), \
                    patch.object(bench, "benchmark_provenance", return_value={}), \
                    patch.object(bench, "board_vram_mib", return_value=None), patch.object(bench, "psutil", None), \
                    patch.object(bench, "Sampler", return_value=sampler), \
                    patch.object(bench, "prompt_text", return_value="fixture"), \
                    patch.object(bench, "http_json", side_effect=http), contextlib.redirect_stdout(io.StringIO()):
                socket_mock.return_value.__enter__.return_value.connect_ex.return_value = 1
                code = bench.main()
            return code, json.loads((output / "report.json").read_text(encoding="utf-8"))

    def test_startup_only_event_does_not_qualify_actual_main_request_slices(self):
        code, report = self.run_mocked_plan(StrictGraphCapRecordsTests.line(), ["", ""])
        self.assertEqual(code, 1)
        self.assertFalse(report["strict_layer_major_graph_cap_qualification"]["passed"])
        self.assertTrue(all(row["strict_layer_major_graph_cap"]["records"] == [] for row in report["records"]))

    def test_actual_main_qualifies_one_long_call_and_allows_blank_decode_row(self):
        code, report = self.run_mocked_plan(StrictGraphCapRecordsTests.line(), [StrictGraphCapRecordsTests.line(), ""])
        self.assertEqual(code, 0)
        q = report["strict_layer_major_graph_cap_qualification"]
        self.assertEqual((q["passed"], q["completed_calls"], q["failed_calls"], q["malformed_records"]), (True, 1, 0, 0))
        self.assertEqual(report["records"][1]["strict_layer_major_graph_cap"]["records"], [])


class HostTokenEmbeddingRecordsTests(unittest.TestCase):
    @staticmethod
    def line(**changes):
        values = dict(version=1, mode="raw_q4k_cuda_get_rows", source_bytes=356843520, row_bytes=2304,
                      rows=512, packed_bytes=1179648, host_pack_bytes=1179648,
                      device_pack_max_bytes=1179648, scope=bench.HOST_TOKEN_EMBEDDING_SCOPE)
        values.update(changes)
        return "[GLM-HOST-TOKEN-EMBEDDING] " + " ".join(f"{key}={value}" for key, value in values.items())

    @staticmethod
    def row(log, prompt=1872, hits=0):
        return dict(name="fixture", prompt_tokens=prompt, cache_hits=hits,
                    host_token_embedding=bench.host_token_embedding_records(log))

    def test_prompt_coverage_requires_actual_uploads_for_every_uncached_request(self):
        full = "\n".join([self.line()] * 4)
        self.assertTrue(bench.host_token_embedding_qualification([self.row(full)])["passed"])
        for rows in ([], [self.row("")], [self.row(self.line())], [self.row(full, hits=1)],
                     [self.row(full), self.row("")], [dict(prompt_tokens=1, cache_hits=0)],
                     [self.row(full, prompt=0)], [self.row(full, prompt=True)]):
            with self.subTest(rows=rows):
                self.assertFalse(bench.host_token_embedding_qualification(rows)["passed"])

    def test_tail_scalar_duplicate_calls_and_upload_totals_preserve_scope(self):
        log = "\n".join([self.line()] * 3 + [self.line(rows=336, packed_bytes=336 * 2304),
                                               self.line(rows=1, packed_bytes=2304)])
        q = bench.host_token_embedding_qualification([self.row(log)])
        self.assertTrue(q["passed"])
        self.assertEqual(q["requests"][0]["packed_rows"], 1873)
        self.assertEqual(q["requests"][0]["packed_bytes"], 1873 * 2304)
        self.assertIn("separately", q["scope"])

    def test_missing_duplicate_unknown_and_noncanonical_numbers_refuse(self):
        valid = self.line()
        invalid = [valid.replace(" version=1", ""), valid + " rows=512", valid + " extra=1", valid + " partial"]
        for field in ("version", "source_bytes", "row_bytes", "rows", "packed_bytes", "host_pack_bytes", "device_pack_max_bytes"):
            invalid += [self.line(**{field: value}) for value in (-1, "+1", "1.0", "oops", 2**64)]
        for line in invalid:
            with self.subTest(line=line):
                row = self.row(line + "\n" + "\n".join([valid] * 4))
                self.assertEqual(row["host_token_embedding"]["malformed_records"], 1)
                self.assertFalse(bench.host_token_embedding_qualification([row])["passed"])

    def test_wrong_profile_footprints_math_and_scope_refuse(self):
        changes = [dict(version=0), dict(mode="cpu_float_get_rows"), dict(source_bytes=1), dict(row_bytes=2305),
                   dict(rows=0), dict(rows=513), dict(packed_bytes=1), dict(host_pack_bytes=1179649),
                   dict(device_pack_max_bytes=1179649), dict(scope="physical_ssd_avoided")]
        for change in changes:
            with self.subTest(change=change):
                observation = bench.host_token_embedding_records(self.line(**change))
                self.assertEqual((observation["records"], observation["malformed_records"]), ([], 1))


class HostTokenEmbeddingCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke
    run_mocked_plan = StrictGraphCapCliTests.run_mocked_plan

    @staticmethod
    def valid_args(value="1"):
        return ["--unified-memory", "--env", f"{bench.HOST_TOKEN_EMBEDDING_ENV}={value}", "--env",
                "TS_UNIFIED_MAX_RUNNING_REQUESTS=1", "--env", "TS_SCHED_PREFILL_CHUNK=512",
                "--env", "MAX_CONTEXT=2048"]

    def test_explicit_on_and_off_reach_provenance(self):
        for value in ("0", "1"):
            self.assertEqual(self.invoke(self.valid_args(value), self.ReachedProvenance)[1], 1)

    def test_invalid_or_inherited_mode_cannot_bypass_qualification(self):
        invalid = [["--env", f"{bench.HOST_TOKEN_EMBEDDING_ENV}={value}"]
                   for value in ("", "2", "01", "+1", "1.0", " 1")]
        invalid += [self.valid_args() + extra for extra in (["--spec"], ["--concurrent"], ["--prefix-cache"],
                    ["--repeat", "0"], ["--server-arg=--draft-model=fixture.gguf"])]
        for key, value in (("TS_UNIFIED_MAX_RUNNING_REQUESTS", "1"), ("TS_SCHED_PREFILL_CHUNK", "512"), ("MAX_CONTEXT", "2048")):
            args = self.valid_args()
            index = args.index(f"{key}={value}")
            invalid.append(args[:index - 1] + args[index + 1:])
        for args in invalid:
            with self.subTest(args=args):
                error, calls = self.invoke(args, SystemExit)
                self.assertEqual((error.code, calls), (2, 0))
        error, calls = self.invoke([], SystemExit, {bench.HOST_TOKEN_EMBEDDING_ENV: "1"})
        self.assertEqual((error.code, calls), (2, 0))

    def test_startup_uploads_do_not_qualify_measured_requests(self):
        full = "\n".join([HostTokenEmbeddingRecordsTests.line()] * 4)
        code, report = self.run_mocked_plan(full, ["", ""])
        self.assertEqual(code, 1)
        self.assertFalse(report["host_token_embedding_qualification"]["passed"])
        self.assertTrue(all(row["host_token_embedding"]["records"] == [] for row in report["records"]))

    def test_actual_main_qualifies_both_request_slices(self):
        full = "\n".join([HostTokenEmbeddingRecordsTests.line()] * 4)
        code, report = self.run_mocked_plan(full, [full, full])
        self.assertEqual(code, 0)
        q = report["host_token_embedding_qualification"]
        self.assertTrue(q["passed"])
        self.assertEqual([row["packed_rows"] for row in q["requests"]], [2048, 2048])


class PrefillDownOverlapRecordsTests(unittest.TestCase):
    @staticmethod
    def line(**changes):
        values = dict(profile=1, layer=3, rows=512, outcome="completed", selected_misses=288,
            prefix_submissions=1, suffix_submissions=1, gate_up_bytes=100663296, down_bytes=50331648,
            gate_up_windows=3, down_windows=2, arena_bytes=75497472, slot_bytes=37748736,
            distinct_upload_backend=1, gate_up_file_ms=10.2, prefix_submit_ms=0.1,
            down_file_ms=5.4, join_ms=0.2, suffix_compute_ms=1.4, scope=bench.PREFILL_DOWN_OVERLAP_SCOPE)
        values.update(changes)
        return "[HOSTMOE-PREFILL-DOWN-OVERLAP] " + " ".join(f"{key}={value}" for key, value in values.items())

    def test_completed_execution_retains_exact_bytes_and_host_intervals(self):
        result = bench.prefill_down_overlap_records(self.line())
        self.assertEqual(result["malformed_records"], 0)
        self.assertEqual(result["records"][0]["down_bytes"], 50331648)
        self.assertEqual(result["records"][0]["prefix_submit_ms"], 0.1)
        self.assertIn("no measured CUDA overlap", result["scope"])

    def test_failed_partial_execution_never_qualifies_as_completed(self):
        observed = bench.prefill_down_overlap_records(self.line(outcome="failed", suffix_submissions=0, down_bytes=0, down_windows=0))
        self.assertEqual(observed["malformed_records"], 0)
        q = bench.prefill_down_overlap_qualification([dict(name="fixture", cache_hits=0, prefill_down_overlap=observed)])
        self.assertFalse(q["passed"])
        self.assertEqual(q["requests"][0]["failed_calls"], 1)

    def test_malformed_duplicate_overflow_nonfinite_and_unqualified_shapes_refuse(self):
        changes = [dict(profile=2), dict(rows=8), dict(rows=513), dict(selected_misses=0),
            dict(selected_misses=289), dict(selected_misses=4097), dict(prefix_submissions=0), dict(suffix_submissions=2),
            dict(distinct_upload_backend=0), dict(gate_up_bytes=0), dict(down_windows=0),
            dict(gate_up_windows=2), dict(down_windows=3),
            dict(gate_up_bytes=2**64 - 1, gate_up_windows=(2**64 - 1 + 37748735) // 37748736),
            dict(arena_bytes=75497474), dict(slot_bytes=1), dict(down_bytes=2**64),
            dict(prefix_submit_ms="nan"), dict(join_ms="inf"), dict(down_file_ms=-1),
            dict(layer=-1), dict(scope="kernel_overlap_proven"), dict(outcome="pending")]
        lines = [self.line(**change) for change in changes]
        lines += [self.line() + " rows=512", self.line().replace(" down_bytes=50331648", "")]
        for line in lines:
            with self.subTest(line=line):
                self.assertEqual(bench.prefill_down_overlap_records(line)["malformed_records"], 1)

    def test_each_measured_request_needs_activity_and_mixed_failure_refuses(self):
        good = bench.prefill_down_overlap_records(self.line())
        rows = [dict(name="a", cache_hits=0, prefill_down_overlap=good), dict(name="b", cache_hits=0)]
        self.assertFalse(bench.prefill_down_overlap_qualification(rows)["passed"])
        rows[1]["prefill_down_overlap"] = good
        self.assertTrue(bench.prefill_down_overlap_qualification(rows)["passed"])
        rows[1]["prefill_down_overlap"] = bench.prefill_down_overlap_records(self.line() + "\n" + self.line(outcome="failed"))
        self.assertFalse(bench.prefill_down_overlap_qualification(rows)["passed"])
        self.assertFalse(bench.prefill_down_overlap_qualification([])["passed"])


class PrefillDownOverlapCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke

    def run_mocked_plan(self, startup_log, request_logs):
        packed = "\n".join([HostTokenEmbeddingRecordsTests.line()] * 4)
        return StrictGraphCapCliTests.run_mocked_plan(self, startup_log, [packed + "\n" + log for log in request_logs])

    @staticmethod
    def valid_args(value="1"):
        return ["--unified-memory", "--env", f"{bench.PREFILL_DOWN_OVERLAP_ENV}={value}", "--env",
                f"{bench.HOST_TOKEN_EMBEDDING_ENV}=1", "--env",
                "TS_UNIFIED_MAX_RUNNING_REQUESTS=1", "--env", "TS_SCHED_PREFILL_CHUNK=512", "--env", "MAX_CONTEXT=2048",
                "--env", "TS_HOST_MOE_PREFILL_RING_COALESCE16=0", "--env", "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS=0",
                "--env", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS=0"]

    def test_exact_flag_and_supported_profile_reach_provenance(self):
        for value in (None, "0", "1"):
            self.assertEqual(bench.prefill_down_overlap_enabled(value), value == "1")
        for value in ("0", "1"):
            self.assertEqual(self.invoke(self.valid_args(value), self.ReachedProvenance)[1], 1)

    def test_invalid_inherited_and_conflicting_profiles_refuse_before_launch(self):
        invalid = [["--env", f"{bench.PREFILL_DOWN_OVERLAP_ENV}={value}"]
                   for value in ("", "2", "01", "+1", "1.0", " 1")]
        invalid += [self.valid_args() + extra for extra in (["--spec"], ["--concurrent"], ["--prefix-cache"],
            ["--repeat", "0"], ["--server-arg=--draft-model=fixture.gguf"],
            ["--env", f"{bench.HOST_TOKEN_EMBEDDING_ENV}=0"],
            ["--env", "TS_HOST_MOE_PREFILL_RING_COALESCE16=1"], ["--env", "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS=1"],
            ["--env", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS=1"])]
        for args in invalid:
            with self.subTest(args=args):
                error, calls = self.invoke(args, SystemExit)
                self.assertEqual((error.code, calls), (2, 0))
        error, calls = self.invoke([], SystemExit, {bench.PREFILL_DOWN_OVERLAP_ENV: "1"})
        self.assertEqual((error.code, calls), (2, 0))

    def test_startup_only_split_does_not_qualify_measured_requests(self):
        code, report = self.run_mocked_plan(PrefillDownOverlapRecordsTests.line(), ["", ""])
        self.assertEqual(code, 1)
        self.assertFalse(report["prefill_down_overlap_qualification"]["passed"])
        self.assertTrue(all(row["prefill_down_overlap"]["records"] == [] for row in report["records"]))

    def test_actual_main_qualifies_both_request_slices(self):
        line = PrefillDownOverlapRecordsTests.line()
        code, report = self.run_mocked_plan(line, [line, line])
        self.assertEqual(code, 0)
        self.assertTrue(report["prefill_down_overlap_qualification"]["passed"])
        self.assertEqual([row["completed_calls"] for row in report["prefill_down_overlap_qualification"]["requests"]], [1, 1])


class DecodeGraphReserveCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke

    @staticmethod
    def valid_args(value="1"):
        return HostTokenEmbeddingCliTests.valid_args() + ["--env",
            f"{bench.DECODE_GRAPH_RESERVE_ENV}={value}", "--env", "TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS=1"]

    def run_mocked_plan(self, startup_log, request_logs):
        packed = "\n".join([HostTokenEmbeddingRecordsTests.line()] * 4)
        return StrictGraphCapCliTests.run_mocked_plan(self, startup_log,
            [packed + "\n" + log for log in request_logs])

    def test_exact_explicit_and_inherited_flags_require_supported_profile(self):
        for value in ("0", "1"):
            self.assertEqual(self.invoke(self.valid_args(value), self.ReachedProvenance)[1], 1)
        invalid = [["--env", f"{bench.DECODE_GRAPH_RESERVE_ENV}={value}"]
            for value in ("", "2", "01", "+1", "1.0", " 1")]
        invalid += [self.valid_args() + extra for extra in (["--spec"], ["--concurrent"],
            ["--prefix-cache"], ["--repeat", "0"], ["--server-arg=--draft-model=fixture.gguf"],
            ["--env", f"{bench.HOST_TOKEN_EMBEDDING_ENV}=0"],
            ["--env", f"{bench.STRICT_GRAPH_CAP_ENV}=512"],
            ["--env", f"{bench.PREFILL_DOWN_OVERLAP_ENV}=1"],
            ["--env", "TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS=0"])]
        for args in invalid:
            with self.subTest(args=args):
                error, calls = self.invoke(args, SystemExit)
                self.assertEqual((error.code, calls), (2, 0))
        error, calls = self.invoke([], SystemExit, {bench.DECODE_GRAPH_RESERVE_ENV: "1"})
        self.assertEqual((error.code, calls), (2, 0))

    def test_startup_only_credit_cannot_qualify_measured_requests(self):
        from test_glm_decode_graph_reserve import reserve_line, recovery_line
        log = reserve_line() + "\n" + recovery_line()
        code, report = self.run_mocked_plan(log, ["", ""])
        self.assertEqual(code, 1)
        self.assertFalse(report["decode_graph_reserve_qualification"]["passed"])
        self.assertTrue(all(not row["decode_graph_reserve"]["records"] for row in report["records"]))

    def test_actual_main_requires_credit_and_recovery_in_both_request_slices(self):
        from test_glm_decode_graph_reserve import reserve_line, recovery_line
        logs = [reserve_line() + "\n" + recovery_line(request=f"request-{i}") for i in range(2)]
        code, report = self.run_mocked_plan("", logs)
        self.assertEqual(code, 0)
        self.assertTrue(report["decode_graph_reserve_qualification"]["passed"])
        self.assertFalse(report["decode_graph_reserve_qualification"]["resident_capacity_qualified"])
        code, report = self.run_mocked_plan(logs[1], [logs[0], ""])
        self.assertEqual(code, 1)
        self.assertFalse(report["decode_graph_reserve_qualification"]["passed"])


class FileReadWorkersCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke
    run_mocked_plan = StrictGraphCapCliTests.run_mocked_plan

    @staticmethod
    def valid_args(value="8"):
        settings = dict(TS_GLM_HOST_TOKEN_EMBEDDING="1", TS_UNIFIED_MAX_RUNNING_REQUESTS="1",
            TS_SCHED_PREFILL_CHUNK="512", MAX_CONTEXT="2048", TS_HOST_MOE_EXPERT_CACHE_RING_READS="2",
            TS_HOST_MOE_EXPERT_CACHE_TIMING="1", TS_HOST_MOE_FILE_READ_WORKERS=value)
        for name in ("TS_HOST_MOE_RAM_CACHE_MB", "TS_HOST_MOE_RAM_CACHE_MAPPED", "TS_HOST_MOE_RAM_CACHE_VICTIM",
            "TS_HOST_MOE_LAYER_PREFETCH", "TS_HOST_MOE_EXPERT_CACHE_PREFETCH", "TS_HOST_MOE_DECODE_PREFETCH",
            "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS",
            "TS_HOST_MOE_PREFILL_RING_COALESCE16", "TS_HOST_MOE_PHASE_COLD_GROWTH",
            "TS_HOST_MOE_DECODE_REBALANCE", "TS_HOST_MOE_FILE_RANDOM_ACCESS", "TS_GLM_SELECTED_EXPERT_TIMING",
            "TS_DSV4_PERF", "TS_HOST_MOE_ROUTE_TRACE", "TS_GGML_Q8_ORDERED_SMALL_BATCH",
            "TS_GGML_Q8_ORDERED_GATE_PREFILL", "TS_GGML_Q8_GATE_PREFILL_TILE16",
            "TS_GGML_STRICT_RECURRENT_PROJECTIONS", "TS_GLM_OP_OFFLOAD",
            "TS_GLM_GRAPH_MEMORY_DIAGNOSTICS"):
            settings[name] = "0"
        result = ["--unified-memory", "--require-file-read-workers"]
        for name, value in settings.items():
            result.extend(("--env", name+"="+value))
        return result

    @staticmethod
    def complete_log():
        from test_glm_file_read_workers import line
        return "\n".join([HostTokenEmbeddingRecordsTests.line()] * 4 + [line()])

    def test_explicit_auto_and_override_reach_provenance(self):
        for value in ("0", "1", "8", "16"):
            self.assertEqual(self.invoke(self.valid_args(value), self.ReachedProvenance)[1], 1)

    def test_unqualified_or_inherited_override_refuses(self):
        arguments = self.valid_args()
        arguments.remove("--require-file-read-workers")
        self.assertEqual(self.invoke(arguments, SystemExit)[1], 0)
        self.assertEqual(self.invoke([], SystemExit, {"TS_HOST_MOE_FILE_READ_WORKERS": "8"})[1], 0)

    def test_incompatible_profile_refuses_before_provenance(self):
        invalid = [self.valid_args(value) for value in ("", "08", "17", "-1")]
        invalid += [self.valid_args()+extra for extra in (["--spec"], ["--concurrent"], ["--prefix-cache"],
            ["--env", "TS_HOST_MOE_RAM_CACHE_MB=4096"], ["--env", "TS_HOST_MOE_FILE_RANDOM_ACCESS=1"],
            ["--env", "TS_HOST_MOE_EXPERT_CACHE_TIMING=0"], ["--env", "TS_HOST_MOE_ROUTE_TRACE=1"],
            ["--server-arg=--prefill-chunk-size=256"], ["--env", "TS_GGML_Q8_ORDERED_SMALL_BATCH=1"])]
        invalid += [[arg for arg in self.valid_args() if arg != "--unified-memory"]]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.assertEqual(self.invoke(arguments, SystemExit)[1], 0)

    def test_actual_main_qualifies_both_request_slices(self):
        code, report = self.run_mocked_plan("", [self.complete_log()] * 2)
        self.assertEqual(code, 0)
        self.assertTrue(report["file_read_workers_qualification"]["passed"])

    def test_startup_only_activity_never_qualifies(self):
        host_log = "\n".join([HostTokenEmbeddingRecordsTests.line()] * 4)
        code, report = self.run_mocked_plan(self.complete_log(), [host_log, host_log])
        self.assertEqual(code, 1)
        self.assertFalse(report["file_read_workers_qualification"]["passed"])


class FileRandomAccessCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke
    run_mocked_plan = StrictGraphCapCliTests.run_mocked_plan

    @staticmethod
    def valid_args(value="1"):
        return FileReadWorkersCliTests.valid_args("0") + ["--require-file-random-access",
            "--env", "TS_HOST_MOE_FILE_RANDOM_ACCESS="+value]

    @staticmethod
    def complete_log(hint=1):
        from test_glm_file_read_workers import line
        return "\n".join([HostTokenEmbeddingRecordsTests.line()] * 4 + [line(
            read_pool_requested=0, read_pool_workers=15, read_pool_slots=16, file_read_random_access=hint)])

    def test_control_and_candidate_reach_provenance(self):
        for value in ("0", "1"):
            self.assertEqual(self.invoke(self.valid_args(value), self.ReachedProvenance)[1], 1)

    def test_missing_joined_qualification_or_invalid_hint_refuses(self):
        invalid = [self.valid_args(value) for value in ("", "2", "01")]
        without_workers = self.valid_args()
        without_workers.remove("--require-file-read-workers")
        invalid.append(without_workers)
        without_hint_proof = self.valid_args()
        without_hint_proof.remove("--require-file-random-access")
        invalid.append(without_hint_proof)
        for arguments in invalid:
            self.assertEqual(self.invoke(arguments, SystemExit)[1], 0)

    def test_actual_main_qualifies_applied_hint_for_both_requests(self):
        code, report = self.run_mocked_plan("", [self.complete_log()] * 2)
        self.assertEqual(code, 0)
        self.assertTrue(report["file_read_workers_qualification"]["passed"])
        self.assertEqual(report["file_read_workers_qualification"]["requested_random_access_hint"], 1)

    def test_wrong_applied_hint_never_qualifies(self):
        code, report = self.run_mocked_plan("", [self.complete_log(0)] * 2)
        self.assertEqual(code, 1)
        self.assertFalse(report["file_read_workers_qualification"]["passed"])

    def test_startup_hint_never_supplies_measured_activity(self):
        host = "\n".join([HostTokenEmbeddingRecordsTests.line()] * 4)
        code, report = self.run_mocked_plan(self.complete_log(), [host, host])
        self.assertEqual(code, 1)
        self.assertFalse(report["file_read_workers_qualification"]["passed"])


class DecodeRetainPeerCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke
    run_mocked_plan = StrictGraphCapCliTests.run_mocked_plan

    @staticmethod
    def valid_args(value="1"):
        arguments = FileReadWorkersCliTests.valid_args("0")
        arguments.remove("--require-file-read-workers")
        return arguments + ["--env", "TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS=1",
            "--env", "TS_HOST_MOE_DECODE_REBALANCE=1",
            "--env", "TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER="+value]

    @staticmethod
    def complete_log():
        from test_glm_decode_retain_peer import RetainedPeerTests
        return "\n".join([HostTokenEmbeddingRecordsTests.line()] * 4 + [RetainedPeerTests().log()])

    def test_candidate_and_disabled_control_reach_provenance(self):
        for value in ("0", "1"):
            self.assertEqual(self.invoke(self.valid_args(value), self.ReachedProvenance)[1], 1)

    def test_bad_or_inherited_profile_refuses_before_launch(self):
        invalid = [self.valid_args(value) for value in ("", "01", "2")]
        invalid += [self.valid_args()+extra for extra in (["--spec"], ["--concurrent"], ["--prefix-cache"],
            ["--gpu-telemetry"], ["--env", "TS_HOST_MOE_DECODE_REBALANCE=0"],
            ["--env", "TS_HOST_MOE_PHASE_COLD_GROWTH=1"], ["--env", "TS_HOST_MOE_FILE_RANDOM_ACCESS=1"],
            ["--env", "TS_GGML_Q8_ORDERED_SMALL_BATCH=1"], ["--server-arg=--prefill-chunk-size=256"])]
        for arguments in invalid:
            self.assertEqual(self.invoke(arguments, SystemExit)[1], 0)
        self.assertEqual(self.invoke([], SystemExit, {"TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER": "1"})[1], 0)

    def test_actual_main_qualifies_both_request_slices(self):
        code, report = self.run_mocked_plan("", [self.complete_log()] * 2)
        self.assertEqual(code, 0)
        self.assertTrue(report["decode_retain_peer_qualification"]["passed"])

    def test_startup_cannot_supply_measured_retention(self):
        host = "\n".join([HostTokenEmbeddingRecordsTests.line()] * 4)
        code, report = self.run_mocked_plan(self.complete_log(), [host, host])
        self.assertEqual(code, 1)
        self.assertFalse(report["decode_retain_peer_qualification"]["passed"])
        self.assertTrue(all(not row["decode_retain_peer"]["records"] for row in report["records"]))

    def test_conditional_activity_leaves_inactive_request_without_credit(self):
        host = "\n".join([HostTokenEmbeddingRecordsTests.line()] * 4)
        code, report = self.run_mocked_plan("", [self.complete_log(), host])
        self.assertEqual(code, 0)
        self.assertTrue(report["decode_retain_peer_qualification"]["requests"][0]["passed"])
        self.assertFalse(report["decode_retain_peer_qualification"]["requests"][1]["passed"])
        self.assertEqual(1, report["decode_retain_peer_qualification"]["active_requests"])
        self.assertFalse(report["decode_retain_peer_qualification"]["all_requests_active"])


class VmmWeightsCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke
    run_mocked_plan = StrictGraphCapCliTests.run_mocked_plan

    @staticmethod
    def valid_args(value="1"):
        from test_glm_vmm_weights import VmmProfileTests
        settings = VmmProfileTests().settings() | {bench._vmm_weights_helper.ENV: value}
        return ["--unified-memory", "--plan", "quick"] + [part for key, item in settings.items()
            for part in ("--env", f"{key}={item}")]

    @staticmethod
    def complete_log(phase=1, reshape=True, pressure=None):
        from test_glm_vmm_weights import VmmActivityTests
        activity = []
        for line in VmmActivityTests.log(phase, reshape, prompt=1872).splitlines():
            if bench._vmm_weights_helper.TIMING_TAG in line:
                wide = " rows=1 " not in line
                line += (f" bulk={int(wide)} joint=0 file_windows=1 file_bytes=40000000"
                         f" ring_calls={int(wide)} ring_expanded_calls={int(wide)}"
                         f" ring_arena_peak_bytes={72 * 2**20 if wide else 0}"
                         f" ring_slot_peak_bytes={36 * 2**20 if wide else 0}")
            activity.append(line)
        if pressure is not None:
            activity.append(VmmActivityTests.pressure(pressure))
        return "\n".join([HostTokenEmbeddingRecordsTests.line()] * 4 + activity)

    def run_pressure_plan(self, startup_log, request_logs):
        arguments = self.valid_args() + ["--require-vmm-pressure"]
        with patch.object(self, "valid_args", return_value=arguments):
            return self.run_mocked_plan(startup_log, request_logs)

    def test_explicit_candidate_and_control_reach_provenance(self):
        for value in ("0", "1"):
            self.assertEqual(self.invoke(self.valid_args(value), self.ReachedProvenance)[1], 1)

    def test_inherited_or_confounded_candidate_refuses_before_launch(self):
        invalid = [self.valid_args(value) for value in ("", "01", "2")]
        invalid += [self.valid_args() + extra for extra in (["--spec"], ["--concurrent"],
            ["--prefix-cache"], ["--gpu-telemetry"], ["--repeat", "2"],
            ["--env", "TS_HOST_MOE_DECODE_REBALANCE=0"],
            ["--env", "TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER=1"],
            ["--env", "TS_HOST_MOE_RAM_CACHE_MB=1024"], ["--server-arg=--context-size=1024"]) ]
        for arguments in invalid:
            self.assertEqual(self.invoke(arguments, SystemExit)[1], 0)
        self.assertEqual(self.invoke([], SystemExit, {bench._vmm_weights_helper.ENV: "1"})[1], 0)

    def test_actual_main_requires_all_layers_of_both_phases_and_checked_reshape(self):
        code, report = self.run_mocked_plan("", [self.complete_log(), self.complete_log(2)])
        self.assertEqual(code, 0)
        self.assertTrue(report["vmm_weights_qualification"]["passed"])
        self.assertEqual(report["vmm_weights_qualification"]["reshape_events"], 2)

    def test_startup_cannot_supply_request_activity(self):
        measured = self.complete_log(reshape=False)
        code, report = self.run_mocked_plan(self.complete_log(), [measured, measured])
        self.assertEqual(code, 1)
        self.assertFalse(report["vmm_weights_qualification"]["passed"])
        self.assertTrue(all(not row["vmm_weights"]["reshapes"] for row in report["records"]))

    def test_no_reshape_or_partial_phase_cannot_qualify(self):
        code, report = self.run_mocked_plan("", [self.complete_log(reshape=False)] * 2)
        self.assertEqual(code, 1)
        self.assertFalse(report["vmm_weights_qualification"]["passed"])
        bad = self.complete_log(2).replace("vmm_certificate=1", "vmm_certificate=0", 1)
        code, report = self.run_mocked_plan("", [self.complete_log(), bad])
        self.assertEqual(code, 1)
        self.assertFalse(report["vmm_weights_qualification"]["passed"])

    def test_pressure_flag_requires_the_explicit_vmm_profile_before_launch(self):
        for arguments in (["--require-vmm-pressure"], self.valid_args("0") + ["--require-vmm-pressure"]):
            error, calls = self.invoke(arguments, SystemExit)
            self.assertEqual((error.code, calls), (2, 0))
        self.assertEqual(self.invoke(self.valid_args() + ["--require-vmm-pressure"], self.ReachedProvenance)[1], 1)

    def test_actual_main_requires_pressure_in_each_measured_request(self):
        code, report = self.run_pressure_plan("", [self.complete_log(1, pressure=2), self.complete_log(3, pressure=5)])
        self.assertEqual(code, 0)
        q = report["vmm_weights_qualification"]
        self.assertTrue(q["passed"])
        self.assertTrue(q["pressure_required"])
        self.assertEqual(q["pressure_events"], 2)
        self.assertTrue(all(row["pressure_events"] == 1 for row in q["requests"]))
        code, report = self.run_pressure_plan("", [self.complete_log(1, pressure=2), self.complete_log(3)])
        self.assertEqual(code, 1)
        self.assertFalse(report["vmm_weights_qualification"]["passed"])

    def test_startup_pressure_cannot_satisfy_request_scoped_requirement(self):
        from test_glm_vmm_weights import VmmActivityTests
        code, report = self.run_pressure_plan(VmmActivityTests.pressure(),
            [self.complete_log(2), self.complete_log(4)])
        self.assertEqual(code, 1)
        self.assertTrue(all(not row["vmm_weights"]["pressures"] for row in report["records"]))
        self.assertFalse(report["vmm_weights_qualification"]["passed"])

    def test_pressure_order_and_unknown_native_tags_fail_even_without_requirement(self):
        for bad in (self.complete_log(3, pressure=2), self.complete_log(3) + "\n[HOSTMOE-VMM-UNKNOWN] profile=1"):
            code, report = self.run_mocked_plan("", [self.complete_log(1, pressure=2), bad])
            self.assertEqual(code, 1)
            self.assertFalse(report["vmm_weights_qualification"]["passed"])


class FileReadSortCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke

    @staticmethod
    def valid_args(value="1"):
        from test_glm_file_read_sort import FileReadSortProfileTests
        settings = FileReadSortProfileTests.settings() | {bench._file_sort_helper.ENV:value}
        return ["--unified-memory", "--plan", "quick", "--require-file-read-sort"] + [part
            for key,item in settings.items() for part in ("--env",f"{key}={item}")]

    @staticmethod
    def complete_log(requested=1, wide_changed=True, decode_changed=True):
        from test_glm_file_read_sort import complete_log
        activity=[]
        for line in complete_log(1872,2,requested,wide_changed,decode_changed).splitlines():
            if bench._file_sort_helper.MARKER in line:
                wide=" rows=1 " not in line
                line=line.replace("file_bytes=4096","file_bytes=40000000")
                line=line.replace("file_sort_bytes=4096","file_sort_bytes=40000000")
                line += (f" joint=0 file_windows=1 graph_submissions=1 total_ms=1 plan_ms=0"
                    f" allocation_ms=0 wide_replace_ms=0 file_pipeline_ms=0.5 upload_enqueue_ms=0.1"
                    f" upload_wait_ms=0.1 input_bridge_ms=0 compute_ms=0.2 output_bridge_ms=0"
                    f" ring_calls={int(wide)} ring_expanded_calls={int(wide)}"
                    f" ring_arena_peak_bytes={72*2**20 if wide else 0}"
                    f" ring_slot_peak_bytes={36*2**20 if wide else 0}")
            else:
                line += (" nout=1 wanth=0 logits=1 reused=0 acquire_ms=0 input_ms=0 compute_ms=1"
                         " output_ms=0 total_ms=1 scope=host_wall_including_existing_sync_and_expert_callbacks")
            activity.append(line)
        return "\n".join(activity)

    def run_mocked_plan(self,startup_log,request_logs,requested=1,auto=False):
        packed="\n".join([HostTokenEmbeddingRecordsTests.line()]*4)
        arguments=self.valid_args(str(requested))
        if auto:
            arguments.remove("--require-file-read-sort")
        with patch.object(self,"valid_args",return_value=arguments):
            return StrictGraphCapCliTests.run_mocked_plan(self,startup_log,
                [packed+"\n"+text for text in request_logs])

    def test_explicit_control_candidate_and_automatic_on_reach_provenance(self):
        for value in ("0","1"):
            self.assertEqual(self.invoke(self.valid_args(value),self.ReachedProvenance)[1],1)
        automatic=self.valid_args();automatic.remove("--require-file-read-sort")
        self.assertEqual(self.invoke(automatic,self.ReachedProvenance)[1],1)

    def test_malformed_inherited_or_confounded_mode_refuses_before_launch(self):
        invalid=[self.valid_args(value) for value in ("","01","2","+1")]
        invalid += [self.valid_args()+extra for extra in (["--spec"],["--concurrent"],
            ["--prefix-cache"],["--gpu-telemetry"],["--repeat","2"],["--think"],
            ["--env","TS_HOST_MOE_DECODE_REBALANCE=0"],
            ["--env","TS_HOST_MOE_VMM_WEIGHTS=1"],
            ["--env","TS_HOST_MOE_FILE_READ_WORKERS=8"],
            ["--env","TS_HOST_MOE_FILE_RANDOM_ACCESS=1"],
            ["--env","TS_HOST_MOE_RAM_CACHE_MB=1024"],
            ["--env","TS_HOST_MOE_PREFILL_DOWN_OVERLAP=1"],
            ["--server-arg=--context-size=1024"])]
        invalid.append(["--require-file-read-sort"])
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.assertEqual(self.invoke(arguments,SystemExit)[1],0)
        self.assertEqual(self.invoke([],SystemExit,{bench._file_sort_helper.ENV:"1"})[1],0)

    def test_actual_request_slices_cover_all_layers_and_both_phases(self):
        for requested in (0,1):
            text=self.complete_log(requested)
            code,report=self.run_mocked_plan("",[text,text],requested)
            self.assertEqual(code,0)
            q=report["file_read_sort_qualification"]
            self.assertTrue(q["passed"])
            self.assertEqual(q["requested_sort"],requested)
            self.assertTrue(all(qr["phases"]["prefill"]["calls"]==168 and
                qr["phases"]["decode"]["calls"]==84 for qr in q["requests"]))

    def test_startup_changed_order_cannot_supply_measured_request_activity(self):
        unchanged=self.complete_log(wide_changed=False,decode_changed=False)
        code,report=self.run_mocked_plan(self.complete_log(),[unchanged,unchanged])
        self.assertEqual(code,1)
        self.assertFalse(report["file_read_sort_qualification"]["passed"])
        self.assertTrue(all(not qr["phases"]["prefill"]["changed_activity_observed"]
            and not qr["phases"]["decode"]["changed_activity_observed"]
            for qr in report["file_read_sort_qualification"]["requests"]))

    def test_on_requires_real_changes_in_both_phases_of_each_request(self):
        for text in (self.complete_log(wide_changed=False),self.complete_log(decode_changed=False)):
            code,report=self.run_mocked_plan("",[self.complete_log(),text])
            self.assertEqual(code,1)
            self.assertFalse(report["file_read_sort_qualification"]["passed"])

    def test_failed_missing_or_wrong_position_activity_is_not_qualified(self):
        valid=self.complete_log()
        for bad in (valid.replace("outcome=completed","outcome=exception",1),
                    valid.replace("layer=3 rows=512","layer=4 rows=512",1),
                    valid.replace("p0=1872","p0=1873",1),
                    valid.replace("file_sort_requested=1","file_sort_requested=0",1)):
            code,report=self.run_mocked_plan("",[valid,bad])
            self.assertEqual(code,1)
            self.assertFalse(report["file_read_sort_qualification"]["passed"])


class FileSourceViewCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke

    @staticmethod
    def valid_args(value="1"):
        helper = bench._file_view_helper
        settings = {**{key:"1" for key in helper.REQUIRED_ON},
            **{key:"0" for key in helper.REQUIRED_OFF}, **helper.REQUIRED_VALUES,
            helper.ENV:value}
        return ["--unified-memory","--plan","quick","--require-file-source-view"] + [part
            for key,item in settings.items() for part in ("--env",f"{key}={item}")]

    @staticmethod
    def complete_log(requested=1):
        activity=[]
        for line in FileReadSortCliTests.complete_log(requested=0).splitlines():
            if bench._file_view_helper.MARKER in line:
                line += (f" file_read_completed_tasks=2 file_read_failed_calls=0 file_read_source_bytes=40000000"
                    f" file_view_requested={requested} file_view_calls={2 if requested else 0}"
                    f" file_view_bytes={40000000 if requested else 0} file_view_fallbacks=0"
                    f" file_view_buffered_bytes={0 if requested else 40000000}"
                    f" file_prefetch_requested=0 file_prefetch_calls=0 file_prefetch_ranges=0"
                    f" file_prefetch_bytes=0 file_prefetch_failures=0")
            activity.append(line)
        return "\n".join(activity)

    def run_mocked_plan(self,startup_log,request_logs,requested=1,auto=False):
        packed="\n".join([HostTokenEmbeddingRecordsTests.line()]*4)
        arguments=self.valid_args(str(requested))
        if auto: arguments.remove("--require-file-source-view")
        with patch.object(self,"valid_args",return_value=arguments):
            return StrictGraphCapCliTests.run_mocked_plan(self,startup_log,
                [packed+"\n"+text for text in request_logs])

    def test_explicit_control_candidate_and_automatic_on_reach_provenance(self):
        for value in ("0","1"):
            self.assertEqual(self.invoke(self.valid_args(value),self.ReachedProvenance)[1],1)
        automatic=self.valid_args();automatic.remove("--require-file-source-view")
        self.assertEqual(self.invoke(automatic,self.ReachedProvenance)[1],1)

    def test_invalid_inherited_or_confounded_modes_refuse_before_launch(self):
        invalid=[self.valid_args(value) for value in ("","01","2","+1")]
        invalid += [self.valid_args()+extra for extra in (["--spec"],["--concurrent"],
            ["--prefix-cache"],["--gpu-telemetry"],["--repeat","2"],["--think"],
            ["--env","TS_HOST_MOE_FILE_READ_SORT=1"],
            ["--env","TS_HOST_MOE_VMM_WEIGHTS=1"],
            ["--env","TS_HOST_MOE_RAM_CACHE_MB=1024"],
            ["--env","TS_HOST_MOE_FILE_READ_WORKERS=8"],
            ["--server-arg=--context-size=1024"])]
        invalid.append(["--require-file-source-view"])
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.assertEqual(self.invoke(arguments,SystemExit)[1],0)
        self.assertEqual(self.invoke([],SystemExit,{bench._file_view_helper.ENV:"1"})[1],0)

    def test_actual_request_slices_cover_all_layers_and_both_phases(self):
        for requested in (0,1):
            text=self.complete_log(requested)
            code,report=self.run_mocked_plan("",[text,text],requested)
            self.assertEqual(code,0)
            q=report["file_source_view_qualification"]
            self.assertTrue(q["passed"])
            self.assertEqual(q["requested_view"],requested)
            self.assertTrue(all(qr["phases"]["prefill"]["calls"]==168 and
                qr["phases"]["decode"]["calls"]==84 for qr in q["requests"]))

    def test_startup_and_buffered_fallback_cannot_supply_measured_activity(self):
        buffered=self.complete_log().replace('file_view_calls=2','file_view_calls=0').replace(
            'file_view_bytes=40000000','file_view_bytes=0').replace(
            'file_view_buffered_bytes=0','file_view_buffered_bytes=40000000')
        code,report=self.run_mocked_plan(self.complete_log(),[buffered,buffered])
        self.assertEqual(code,1)
        self.assertFalse(report["file_source_view_qualification"]["passed"])

    def test_missing_bad_position_failed_read_or_fallback_is_not_qualified(self):
        valid=self.complete_log()
        for bad in (valid.replace('p0=1872','p0=1873',1),
            valid.replace('layer=3 rows=512','layer=4 rows=512',1),
            valid.replace('file_read_failed_calls=0','file_read_failed_calls=1',1),
            valid.replace('file_view_fallbacks=0','file_view_fallbacks=1',1),
            valid.replace('outcome=completed','outcome=exception',1)):
            code,report=self.run_mocked_plan("",[valid,bad])
            self.assertEqual(code,1)
            self.assertFalse(report["file_source_view_qualification"]["passed"])


class FileSourceViewPrefetchCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke

    @staticmethod
    def valid_args(value="1"):
        helper = bench._file_view_helper
        settings = {**{key:"1" for key in helper.REQUIRED_ON},
            **{key:"0" for key in helper.REQUIRED_OFF}, **helper.REQUIRED_VALUES,
            helper.ENV:value,helper.PREFETCH_ENV:value}
        return ["--unified-memory","--plan","quick","--require-file-source-view-prefetch"] + [part
            for key,item in settings.items() for part in ("--env",f"{key}={item}")]

    @staticmethod
    def complete_log(requested=1):
        text = FileSourceViewCliTests.complete_log(requested)
        if requested:
            text = text.replace("file_prefetch_requested=0 file_prefetch_calls=0 file_prefetch_ranges=0"
                " file_prefetch_bytes=0 file_prefetch_failures=0",
                "file_prefetch_requested=1 file_prefetch_calls=1 file_prefetch_ranges=2"
                " file_prefetch_bytes=40000000 file_prefetch_failures=0")
        return text

    def run_mocked_plan(self,startup_log,request_logs,requested=1,auto=False):
        packed="\n".join([HostTokenEmbeddingRecordsTests.line()]*4)
        arguments=self.valid_args(str(requested))
        if auto: arguments.remove("--require-file-source-view-prefetch")
        with patch.object(self,"valid_args",return_value=arguments):
            return StrictGraphCapCliTests.run_mocked_plan(self,startup_log,
                [packed+"\n"+text for text in request_logs])

    def test_control_candidate_and_automatic_on_reach_provenance(self):
        for value in ("0","1"):
            self.assertEqual(self.invoke(self.valid_args(value),self.ReachedProvenance)[1],1)
        arguments=self.valid_args();arguments.remove("--require-file-source-view-prefetch")
        self.assertEqual(self.invoke(arguments,self.ReachedProvenance)[1],1)

    def test_invalid_inherited_pair_and_confounds_refuse_before_launch(self):
        invalid=[self.valid_args(value) for value in ("","01","2","+1")]
        invalid += [self.valid_args()+extra for extra in (["--spec"],["--concurrent"],
            ["--prefix-cache"],["--gpu-telemetry"],["--repeat","2"],["--think"],
            ["--require-file-source-view"],["--env","TS_HOST_MOE_FILE_SOURCE_VIEW=0"],
            ["--env","TS_HOST_MOE_FILE_READ_SORT=1"],["--env","TS_HOST_MOE_VMM_WEIGHTS=1"],
            ["--env","TS_HOST_MOE_RAM_CACHE_MB=1024"],["--env","TS_HOST_MOE_FILE_READ_WORKERS=8"],
            ["--server-arg=--context-size=1024"])]
        invalid.append(["--require-file-source-view-prefetch"])
        invalid.append(self.valid_args("0")+["--env","TS_HOST_MOE_FILE_SOURCE_VIEW=1"])
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.assertEqual(self.invoke(arguments,SystemExit)[1],0)
        self.assertEqual(self.invoke([],SystemExit,
            {bench._file_view_helper.PREFETCH_ENV:"1"})[1],0)

    def test_actual_request_slices_require_all_layers_and_both_phases(self):
        for requested in (0,1):
            text=self.complete_log(requested)
            code,report=self.run_mocked_plan("",[text,text],requested)
            self.assertEqual(code,0)
            q=report["file_source_view_prefetch_qualification"]
            self.assertTrue(q["passed"])
            self.assertEqual((q["requested_view"],q["requested_prefetch"]),(requested,requested))
            self.assertTrue(all(qr["phases"]["prefill"]["calls"]==168 and
                qr["phases"]["decode"]["calls"]==84 for qr in q["requests"]))

    def test_startup_activity_cannot_fill_missing_measured_prefetch(self):
        empty=self.complete_log().replace("file_prefetch_calls=1","file_prefetch_calls=0").replace(
            "file_prefetch_ranges=2","file_prefetch_ranges=0").replace(
            "file_prefetch_bytes=40000000","file_prefetch_bytes=0")
        code,report=self.run_mocked_plan(self.complete_log(),[empty,empty])
        self.assertEqual(code,1)
        self.assertFalse(report["file_source_view_prefetch_qualification"]["passed"])

    def test_advisory_failure_partial_coverage_and_missing_schema_refuse(self):
        valid=self.complete_log()
        for bad in (valid.replace("file_prefetch_failures=0","file_prefetch_failures=1",1),
            valid.replace("file_prefetch_bytes=40000000","file_prefetch_bytes=20000000",1),
            valid.replace("file_prefetch_ranges=2","file_prefetch_ranges=1",1),
            valid.replace(" file_prefetch_calls=1","",1)):
            code,report=self.run_mocked_plan("",[valid,bad])
            self.assertEqual(code,1)
            self.assertTrue(report["file_source_view_qualification"]["passed"])
            self.assertFalse(report["file_source_view_prefetch_qualification"]["passed"])
        # Exact N1 association is an earlier view gate. Its refusal deliberately
        # stops before a prefetch qualification can be produced.
        code,report=self.run_mocked_plan("",[valid,valid.replace("p0=1872","p0=1873",1)])
        self.assertEqual(code,1)
        self.assertFalse(report["file_source_view_qualification"]["passed"])
        self.assertNotIn("file_source_view_prefetch_qualification",report)

    def test_automatic_on_is_qualified_and_off_cannot_claim_activity(self):
        valid=self.complete_log()
        code,report=self.run_mocked_plan("",[valid,valid],auto=True)
        self.assertEqual(code,0)
        self.assertTrue(report["file_source_view_prefetch_qualification"]["passed"])
        code,report=self.run_mocked_plan("",[valid,valid],requested=0)
        self.assertEqual(code,1)
        self.assertFalse(report["file_source_view_qualification"]["passed"])

    def test_pure_source_view_requires_explicit_prefetch_off(self):
        arguments=FileSourceViewCliTests.valid_args()
        flag=bench._file_view_helper.PREFETCH_ENV+"=0"
        index=arguments.index(flag)
        del arguments[index-1:index+1]
        self.assertEqual(self.invoke(arguments,SystemExit)[1],0)
        self.assertEqual(self.invoke(FileSourceViewCliTests.valid_args()+
            ["--env",bench._file_view_helper.PREFETCH_ENV+"=1"],SystemExit)[1],0)


class LayerMajorDeviceBatonCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke

    @staticmethod
    def valid_args(value="1"):
        helper = bench._device_baton_helper
        settings = {**{key: "1" for key in helper.REQUIRED_ON},
                    **{key: "0" for key in helper.REQUIRED_OFF},
                    **helper.REQUIRED_VALUES, helper.ENV: value}
        return ["--unified-memory", "--plan", "quick", "--require-layer-major-device-baton"] + [part
            for key, item in settings.items() for part in ("--env", f"{key}={item}")]

    @staticmethod
    def complete_log(prompt, count, requested):
        from test_glm_layer_major_device_baton import request_log
        activity = []
        for line in request_log(prompt, count, requested).splitlines():
            if bench._file_view_helper.MARKER in line:
                wide = " rows=1 " not in line
                line = line.replace("=4096", "=40000000")
                line += (f" joint=0 file_windows=1 graph_submissions=1 total_ms=1 plan_ms=0"
                    f" allocation_ms=0 wide_replace_ms=0 file_pipeline_ms=0.5 upload_enqueue_ms=0.1"
                    f" upload_wait_ms=0.1 input_bridge_ms=0 compute_ms=0.2 output_bridge_ms=0"
                    f" ring_calls={int(wide)} ring_expanded_calls={int(wide)}"
                    f" ring_arena_peak_bytes={72*2**20 if wide else 0}"
                    f" ring_slot_peak_bytes={36*2**20 if wide else 0}")
            elif bench._file_view_helper.FORWARD_MARKER in line:
                line += (" nout=1 wanth=0 logits=1 reused=0 acquire_ms=0 input_ms=0 compute_ms=1"
                         " output_ms=0 total_ms=1 scope=host_wall_including_existing_sync_and_expert_callbacks")
            activity.append(line)
        widths = [min(512, prompt - start) for start in range(0, prompt, 512)] + [1] * count
        packed = [HostTokenEmbeddingRecordsTests.line(rows=width, packed_bytes=2304*width)
                  for width in widths]
        return "\n".join(packed + activity)

    def run_plan(self, startup, logs, requested=1, auto=False):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            arguments = self.valid_args(str(requested))
            if auto:
                arguments.remove("--require-layer-major-device-baton")
            argv = ["unified-perf-bench.py", "--model", "not-opened.gguf", "--output", str(output)] + arguments
            process = Mock(returncode=0)
            process.poll.return_value = None
            sampler = SimpleNamespace(stop=Mock(), peak_rss=0, peak_private=0)
            pending = iter(zip(bench._device_baton_helper.QUICK_REQUESTS, logs))
            def launch(*args, **kwargs):
                kwargs["stdout"].write(startup + "\n")
                kwargs["stdout"].flush()
                return process
            def http(port, method, path, body, timeout):
                if method == "GET":
                    return {"data": [{"id": "synthetic-fixture"}]}
                (_, prompt, count), text = next(pending)
                with (output / "server.log").open("a", encoding="utf-8") as stream:
                    stream.write(text + "\n")
                return dict(message=dict(role="assistant", content="fixture"), done=True, done_reason="length",
                    prompt_eval_count=prompt, eval_count=count, prompt_eval_duration=100, eval_duration=100)
            with patch.object(bench.sys, "argv", argv), patch.dict(bench.os.environ, {}, clear=True), \
                    patch.object(bench.socket, "socket") as socket_mock, \
                    patch.object(bench.subprocess, "Popen", side_effect=launch), \
                    patch.object(bench, "benchmark_provenance", return_value={}), \
                    patch.object(bench, "board_vram_mib", return_value=None), patch.object(bench, "psutil", None), \
                    patch.object(bench, "Sampler", return_value=sampler), \
                    patch.object(bench, "prompt_text", return_value="fixture"), \
                    patch.object(bench, "http_json", side_effect=http), contextlib.redirect_stdout(io.StringIO()):
                socket_mock.return_value.__enter__.return_value.connect_ex.return_value = 1
                code = bench.main()
            return code, json.loads((output / "report.json").read_text(encoding="utf-8"))

    def test_control_candidate_and_automatic_on_reach_provenance(self):
        for value in ("0", "1"):
            self.assertEqual(self.invoke(self.valid_args(value), self.ReachedProvenance)[1], 1)
        automatic = self.valid_args()
        automatic.remove("--require-layer-major-device-baton")
        self.assertEqual(self.invoke(automatic, self.ReachedProvenance)[1], 1)

    def test_invalid_inherited_or_confounded_profile_refuses_before_launch(self):
        invalid = [self.valid_args(value) for value in ("", "01", "+1", "2")]
        invalid += [self.valid_args() + extra for extra in (["--spec"], ["--concurrent"],
            ["--prefix-cache"], ["--gpu-telemetry"], ["--repeat", "2"], ["--think"],
            ["--require-file-source-view"], ["--require-file-source-view-prefetch"],
            ["--env", "TS_GLM_HOST_TOKEN_EMBEDDING=0"],
            ["--env", "TS_HOST_MOE_DECODE_REBALANCE=0"],
            ["--env", "TS_HOST_MOE_DECODE_STAGING_72MB=0"],
            ["--env", "TS_HOST_MOE_PREFILL_RING_72MB=0"],
            ["--env", "TS_HOST_MOE_VMM_WEIGHTS=1"],
            ["--env", "TS_HOST_MOE_FILE_SOURCE_VIEW=1"],
            ["--env", "TS_HOST_MOE_FILE_READ_WORKERS=8"],
            ["--env", "TS_GLM_TRACE=0"], ["--server-arg=--context-size=1024"]) ]
        invalid.append(["--require-layer-major-device-baton"])
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.assertEqual(self.invoke(arguments, SystemExit)[1], 0)
        self.assertEqual(self.invoke([], SystemExit, {bench._device_baton_helper.ENV: "1"})[1], 0)

    def test_actual_long_bank_short_absence_and_all42_source_calls(self):
        for requested in (0, 1):
            logs = [self.complete_log(prompt, count, requested)
                    for _, prompt, count in bench._device_baton_helper.QUICK_REQUESTS]
            code, report = self.run_plan("", logs, requested)
            self.assertEqual(code, 0)
            q = report["layer_major_device_baton_qualification"]
            self.assertTrue(q["passed"])
            self.assertEqual(q["completed_calls"], requested)
            self.assertEqual([row["completed_calls"] for row in q["requests"]], [requested, 0])
            self.assertEqual([row["source_phases"]["decode"]["calls"] for row in q["requests"]], [1344, 5376])

    def test_startup_event_cannot_supply_missing_measured_bank(self):
        from test_glm_layer_major_device_baton import bank_line
        logs = [self.complete_log(prompt, count, 0)
                for _, prompt, count in bench._device_baton_helper.QUICK_REQUESTS]
        code, report = self.run_plan(bank_line(), logs)
        self.assertEqual(code, 1)
        self.assertFalse(report["layer_major_device_baton_qualification"]["passed"])
        self.assertTrue(all(not row["layer_major_device_baton"]["records"] for row in report["records"]))

    def test_failed_partial_duplicate_or_unclosed_bank_refuses(self):
        logs = [self.complete_log(prompt, count, 1)
                for _, prompt, count in bench._device_baton_helper.QUICK_REQUESTS]
        from test_glm_layer_major_device_baton import bank_line
        for bad in (logs[0].replace("outcome=completed input_calls=176", "outcome=failed input_calls=176", 1),
                    logs[0].replace("input_calls=176", "input_calls=175", 1),
                    logs[0].replace("held_ticket_bytes=0", "held_ticket_bytes=122683392", 1),
                    logs[0] + "\n" + bank_line()):
            code, report = self.run_plan("", [bad, logs[1]])
            self.assertEqual(code, 1)
            self.assertFalse(report["layer_major_device_baton_qualification"]["passed"])

    def test_short_event_or_missing_source_layer_or_wrong_position_refuses(self):
        logs = [self.complete_log(prompt, count, 1)
                for _, prompt, count in bench._device_baton_helper.QUICK_REQUESTS]
        from test_glm_layer_major_device_baton import bank_line
        for long_log, short_log in ((logs[0], logs[1] + "\n" + bank_line()),
            (logs[0].replace("layer=3 rows=512", "layer=4 rows=512", 1), logs[1]),
            (logs[0], logs[1].replace("p0=33", "p0=34", 1))):
            code, report = self.run_plan("", [long_log, short_log])
            self.assertEqual(code, 1)
            self.assertFalse(report["layer_major_device_baton_qualification"]["passed"])

    def test_automatic_on_is_qualified_and_off_never_claims_bank(self):
        logs = [self.complete_log(prompt, count, 1)
                for _, prompt, count in bench._device_baton_helper.QUICK_REQUESTS]
        code, report = self.run_plan("", logs, auto=True)
        self.assertEqual(code, 0)
        self.assertTrue(report["layer_major_device_baton_qualification"]["passed"])
        code, report = self.run_plan("", logs, requested=0)
        self.assertEqual(code, 1)
        self.assertFalse(report["layer_major_device_baton_qualification"]["passed"])


class CausalRouteNeighborCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke

    @staticmethod
    def valid_args(value="1"):
        helper = bench._causal_neighbor_helper
        settings = {**{key: "1" for key in helper.REQUIRED_ON},
                    **{key: "0" for key in helper.REQUIRED_OFF},
                    **helper.REQUIRED_VALUES, helper.ENV: value}
        return ["--unified-memory", "--plan", "quick", "--require-causal-route-neighbor"] + [part
            for key, item in settings.items() for part in ("--env", f"{key}={item}")]

    @staticmethod
    def complete_log(prompt, count, requested):
        from test_glm_causal_route_neighbor import request_log
        activity = []
        for line in request_log(prompt, count, requested).splitlines():
            if bench._file_view_helper.MARKER in line:
                wide = " rows=1 " not in line
                line = line.replace("=4096", "=40000000")
                line += (f" joint=0 file_windows=1 graph_submissions=1 total_ms=1 plan_ms=0"
                    f" allocation_ms=0 wide_replace_ms=0 file_pipeline_ms=0.5 upload_enqueue_ms=0.1"
                    f" upload_wait_ms=0.1 input_bridge_ms=0 compute_ms=0.2 output_bridge_ms=0"
                    f" ring_calls={int(wide)} ring_expanded_calls={int(wide)}"
                    f" ring_arena_peak_bytes={72*2**20 if wide else 0}"
                    f" ring_slot_peak_bytes={36*2**20 if wide else 0}")
            elif bench._file_view_helper.FORWARD_MARKER in line:
                line += (" nout=1 wanth=0 logits=1 reused=0 acquire_ms=0 input_ms=0 compute_ms=1"
                         " output_ms=0 total_ms=1 scope=host_wall_including_existing_sync_and_expert_callbacks")
            activity.append(line)
        widths = [min(512, prompt - start) for start in range(0, prompt, 512)] + [1] * count
        packed = [HostTokenEmbeddingRecordsTests.line(rows=width, packed_bytes=2304*width)
                  for width in widths]
        return "\n".join(packed + activity)

    def run_plan(self, startup, logs, requested=1, auto=False):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            arguments = self.valid_args(str(requested))
            if auto:
                arguments.remove("--require-causal-route-neighbor")
            argv = ["unified-perf-bench.py", "--model", "not-opened.gguf", "--output", str(output)] + arguments
            process = Mock(returncode=0)
            process.poll.return_value = None
            sampler = SimpleNamespace(stop=Mock(), peak_rss=0, peak_private=0)
            pending = iter(zip(bench._causal_neighbor_helper.QUICK_REQUESTS, logs))
            def launch(*args, **kwargs):
                kwargs["stdout"].write(startup + "\n")
                kwargs["stdout"].flush()
                return process
            def http(port, method, path, body, timeout):
                if method == "GET":
                    return {"data": [{"id": "synthetic-fixture"}]}
                (_, prompt, count), text = next(pending)
                with (output / "server.log").open("a", encoding="utf-8") as stream:
                    stream.write(text + "\n")
                return dict(message=dict(role="assistant", content="fixture"), done=True, done_reason="length",
                    prompt_eval_count=prompt, eval_count=count, prompt_eval_duration=100, eval_duration=100)
            with patch.object(bench.sys, "argv", argv), patch.dict(bench.os.environ, {}, clear=True), \
                    patch.object(bench.socket, "socket") as socket_mock, \
                    patch.object(bench.subprocess, "Popen", side_effect=launch), \
                    patch.object(bench, "benchmark_provenance", return_value={}), \
                    patch.object(bench, "board_vram_mib", return_value=None), patch.object(bench, "psutil", None), \
                    patch.object(bench, "Sampler", return_value=sampler), \
                    patch.object(bench, "prompt_text", return_value="fixture"), \
                    patch.object(bench, "http_json", side_effect=http), contextlib.redirect_stdout(io.StringIO()):
                socket_mock.return_value.__enter__.return_value.connect_ex.return_value = 1
                code = bench.main()
            return code, json.loads((output / "report.json").read_text(encoding="utf-8"))

    def test_control_candidate_and_auto_on_reach_provenance(self):
        for value in ("0","1"):
            self.assertEqual(self.invoke(self.valid_args(value),self.ReachedProvenance)[1],1)
        automatic=self.valid_args();automatic.remove("--require-causal-route-neighbor")
        self.assertEqual(self.invoke(automatic,self.ReachedProvenance)[1],1)

    def test_invalid_inherited_and_confounded_profiles_refuse_before_launch(self):
        invalid=[self.valid_args(value) for value in ("","01","+1","2")]
        invalid += [self.valid_args()+extra for extra in (["--spec"],["--concurrent"],
            ["--prefix-cache"],["--gpu-telemetry"],["--repeat","2"],["--think"],
            ["--require-layer-major-device-baton"],["--require-file-source-view"],
            ["--env","TS_HOST_MOE_EXPERT_CACHE_LFU=1"],
            ["--env","TS_GLM_LAYER_MAJOR_DEVICE_BATON=1"],
            ["--env","TS_HOST_MOE_FILE_SOURCE_VIEW=1"],
            ["--env","TS_HOST_MOE_DECODE_REBALANCE=0"],
            ["--env","TS_GLM_HOST_TOKEN_EMBEDDING=0"],
            ["--env","TS_HOST_MOE_FILE_READ_WORKERS=8"],["--env","TS_GLM_TRACE=0"],
            ["--server-arg=--context-size=1024"])]
        invalid.append(["--require-causal-route-neighbor"])
        for args in invalid:
            with self.subTest(args=args):
                self.assertEqual(self.invoke(args,SystemExit)[1],0)
        self.assertEqual(self.invoke([],SystemExit,{bench._causal_neighbor_helper.ENV:"1"})[1],0)

    def test_full_measured_outer_events_each_decode_matches_and_all42_sources(self):
        for requested in (0,1):
            logs=[self.complete_log(prompt,count,requested)
                  for _,prompt,count in bench._causal_neighbor_helper.QUICK_REQUESTS]
            code,report=self.run_plan("",logs,requested)
            self.assertEqual(code,0)
            q=report["causal_route_neighbor_qualification"]
            self.assertTrue(q["passed"])
            self.assertEqual([row["completed_calls"] for row in q["requests"]],[33*requested,129*requested])
            self.assertEqual([row["committed"] for row in q["requests"]],[1344*requested,5376*requested])
            self.assertEqual([row["source_phases"]["decode"]["calls"] for row in q["requests"]],[1344,5376])

    def test_startup_cannot_supply_any_missing_measured_event(self):
        from test_glm_causal_route_neighbor import event_line
        logs=[self.complete_log(prompt,count,0) for _,prompt,count in bench._causal_neighbor_helper.QUICK_REQUESTS]
        code,report=self.run_plan(event_line(),logs)
        self.assertEqual(code,1)
        self.assertFalse(report["causal_route_neighbor_qualification"]["passed"])
        self.assertTrue(all(not row["causal_route_neighbor"]["records"] for row in report["records"]))

    def test_failed_partial_duplicate_or_learning_prefill_refuses(self):
        from test_glm_causal_route_neighbor import event_line
        logs=[self.complete_log(prompt,count,1) for _,prompt,count in bench._causal_neighbor_helper.QUICK_REQUESTS]
        for bad in (logs[0].replace("profile=1 outcome=completed epoch=1","profile=1 outcome=failed epoch=1",1),
            logs[0].replace("queries=42 committed=42","queries=42 committed=41",1),
            logs[0].replace("queries=0 committed=0","queries=1 committed=0",1),
            logs[0]+"\n"+event_line()):
            code,report=self.run_plan("",[bad,logs[1]])
            self.assertEqual(code,1)
            self.assertFalse(report["causal_route_neighbor_qualification"]["passed"])

    def test_missing_layer_wrong_p0_or_one_request_without_match_refuses(self):
        logs=[self.complete_log(prompt,count,1) for _,prompt,count in bench._causal_neighbor_helper.QUICK_REQUESTS]
        no_match=logs[1].replace("matches=1 no_matches=41 compared_anchors=1 min_completed_lag=1",
                                "matches=0 no_matches=42 compared_anchors=0 min_completed_lag=0")
        for left,right in ((logs[0].replace("layer=3 rows=512","layer=4 rows=512",1),logs[1]),
                           (logs[0],logs[1].replace("p0=33","p0=34",1)),(logs[0],no_match)):
            code,report=self.run_plan("",[left,right])
            self.assertEqual(code,1)
            self.assertFalse(report["causal_route_neighbor_qualification"]["passed"])

    def test_automatic_on_is_qualified_and_control_cannot_claim_events(self):
        logs=[self.complete_log(prompt,count,1) for _,prompt,count in bench._causal_neighbor_helper.QUICK_REQUESTS]
        code,report=self.run_plan("",logs,auto=True)
        self.assertEqual(code,0)
        self.assertTrue(report["causal_route_neighbor_qualification"]["passed"])
        code,report=self.run_plan("",logs,requested=0)
        self.assertEqual(code,1)
        self.assertFalse(report["causal_route_neighbor_qualification"]["passed"])


class GpuClockTelemetryCliTests(unittest.TestCase):
    ReachedProvenance = StrictGraphCapCliTests.ReachedProvenance
    invoke = StrictGraphCapCliTests.invoke

    def test_concurrent_diagnostics_refuse_before_provenance_or_child_launch(self):
        error, calls = self.invoke(["--gpu-telemetry", "--concurrent"], SystemExit)
        self.assertEqual((error.code, calls), (2, 0))

    def test_optional_diagnostic_option_reaches_provenance(self):
        self.assertEqual(self.invoke(["--gpu-telemetry"], self.ReachedProvenance)[1], 1)

    def mocked_main(self, enabled, observed=True, http_failure=False):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            argv = ["unified-perf-bench.py", "--model", "not-opened.gguf", "--output", str(output), "--plan", "quick"]
            if enabled:
                argv.append("--gpu-telemetry")
            process = Mock(returncode=0)
            process.poll.return_value = None
            owner = Mock()
            owner.report = dict(requested=True, timing_eligible=False)
            owner.start.return_value = True
            owner.finish.return_value = dict(completed=observed, clock_power_observed=observed, timing_eligible=False)
            telemetry = SimpleNamespace(GpuClockTelemetry=Mock(return_value=owner), request_window=lambda *values:
                dict(zip(("start_unix_ns", "start_monotonic_ns", "end_unix_ns", "end_monotonic_ns"), values)))
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=Mock()))
            def http(port, method, path, body, timeout):
                if method == "GET":
                    return {"data": [{"id": "synthetic-fixture"}]}
                if http_failure:
                    raise RuntimeError("injected HTTP failure")
                return dict(message=dict(role="assistant", content="fixture"), done=True, done_reason="stop",
                            prompt_eval_count=1872, eval_count=2, prompt_eval_duration=100, eval_duration=100)
            sampler = SimpleNamespace(stop=Mock(), peak_rss=0, peak_private=0)
            with patch.object(bench.sys, "argv", argv), patch.dict(bench.os.environ, {}, clear=True), \
                    patch.object(bench.socket, "socket") as socket_mock, \
                    patch.object(bench.subprocess, "Popen", return_value=process), \
                    patch.object(bench, "benchmark_provenance", return_value={}), \
                    patch.object(bench, "board_vram_mib", return_value=None), patch.object(bench, "psutil", None), \
                    patch.object(bench, "Sampler", return_value=sampler), \
                    patch.object(bench, "prompt_text", return_value="fixture"), \
                    patch.object(bench, "http_json", side_effect=http), \
                    patch.object(bench.importlib.util, "spec_from_file_location", return_value=spec), \
                    patch.object(bench.importlib.util, "module_from_spec", return_value=telemetry), \
                    contextlib.redirect_stdout(io.StringIO()):
                socket_mock.return_value.__enter__.return_value.connect_ex.return_value = 1
                code = bench.main()
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            return code, report, owner, telemetry, process

    def test_default_does_not_import_start_or_report_hardware_sampler(self):
        code, report, owner, telemetry, _ = self.mocked_main(False)
        self.assertEqual(code, 0)
        telemetry.GpuClockTelemetry.assert_not_called()
        owner.start.assert_not_called()
        self.assertNotIn("gpu_telemetry", report)
        self.assertTrue(all("request_window" not in row for row in report["records"]))

    def test_actual_client_bounds_and_observations_are_diagnostic_only(self):
        code, report, owner, _, _ = self.mocked_main(True)
        self.assertEqual(code, 0)
        self.assertFalse(report["timing_eligible"])
        owner.start.assert_called_once()
        owner.finish.assert_called_once()
        self.assertEqual(len(report["records"]), 2)
        for row in report["records"]:
            window = row["request_window"]
            self.assertGreaterEqual(window["end_unix_ns"], window["start_unix_ns"])
            self.assertGreaterEqual(window["end_monotonic_ns"], window["start_monotonic_ns"])
            self.assertEqual(set(window), {"start_unix_ns", "start_monotonic_ns", "end_unix_ns", "end_monotonic_ns"})
            self.assertIn("response_identity", row)

    def test_missing_observations_and_http_failure_cannot_be_passes_and_both_cleanup(self):
        code, report, owner, _, process = self.mocked_main(True, observed=False)
        self.assertEqual(code, 1)
        self.assertIn("telemetry did not complete", report["error"])
        owner.finish.assert_called_once()
        process.terminate.assert_called_once()
        code, report, owner, _, process = self.mocked_main(True, http_failure=True)
        self.assertEqual(code, 1)
        self.assertIn("injected HTTP failure", report["error"])
        self.assertIn("request_window", report["records"][0])
        owner.finish.assert_called_once()
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
