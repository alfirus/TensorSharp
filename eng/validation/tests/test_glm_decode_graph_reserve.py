"""Portable CPU telemetry tests; no CUDA/model/native calls."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_decode_graph_reserve as reserve


def recovery_fields(**changes):
    # A 212MiB physically bounded increase from the saved Host B1 D32 ceiling.
    # Effective-used-margin =222298112; before/after are exact MiB multiples.
    values = dict(request="chat-fixed-a", reason="increased",
        ceiling_before=7951351808, ceiling_candidate=8173649920,
        ceiling_target=8173649920, ceiling_after=8173649920, ceiling_max=8698761992,
        native_accounted=7935697920, native_budget=8173649920,
        physical_available=2503999488, physical_headroom=1073741824,
        ledger_committed=11811160064, ledger_reserved=1073741824,
        ledger_capacity=16106127360, effective_limit=13241417728,
        minimum_growth=67108864, recovery_margin=134217728,
        sample="before_recovery", unit="bytes", scope=reserve.RECOVERY_SCOPE)
    values.update(changes)
    return values


def reserve_fields(context=2048, live=0, **changes):
    # Literal profile forecasts are independent expected values.
    workspace = {256: 222996736, 2048: 223075584}.get(context, 223075584)
    values = dict(rows=1, output_rows=0, context=context, owned_before=2 * workspace,
        owned_after=workspace + live, live_device=live, released=workspace - live,
        expert_budget=8173649920, scope=reserve.SCOPE)
    values.update(changes)
    return values


def line(marker, values):
    return marker + " " + " ".join(f"{key}={value}" for key, value in values.items())


def recovery_line(**changes):
    return line(reserve.RECOVERY_MARKER, recovery_fields(**changes))


def reserve_line(context=2048, live=0, **changes):
    return line(reserve.RESERVE_MARKER, reserve_fields(context, live, **changes))


def row(prompt=1872, request="chat-fixed-a", **changes):
    context = 2048 if prompt == 1872 else 256
    values = dict(name="prefill-a" if prompt == 1872 else "decode-a", prompt_tokens=prompt,
        eval_tokens=32 if prompt == 1872 else 128, cache_hits=0,
        decode_graph_reserve=reserve.decode_graph_reserve_records(reserve_line(context)),
        expert_recovery=reserve.expert_recovery_records(recovery_line(request=request)))
    values.update(changes)
    return values


class FlagAndGeometryTests(unittest.TestCase):
    def test_flag_is_exact_and_inherited_values_are_not_silently_ignored(self):
        for value, expected in ((None, False), ("0", False), ("1", True)):
            self.assertEqual(reserve.decode_graph_reserve_enabled(value), expected)
        for value in ("", "01", " 1", "1 ", "+1", "true", "2", 1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reserve.decode_graph_reserve_enabled(value)

    def test_exact_padded_context_workspace_and_raw_input_tax(self):
        self.assertEqual(reserve.decode_graph_workspace_bytes(2048), 223075584)
        self.assertEqual(reserve.decode_graph_workspace_bytes(256), 222996736)
        for context in range(256, 2049, 256):
            self.assertEqual(reserve.decode_graph_workspace_bytes(context), 222985472 + 44 * context)
            self.assertEqual(reserve.decode_graph_workspace_bytes(context, False), 222983168 + 44 * context)
        for context in (-1, 0, 1, 255, 257, 2049, 4096, True, 256.0):
            with self.subTest(context=context), self.assertRaises(ValueError):
                reserve.decode_graph_workspace_bytes(context)


class ReserveParserTests(unittest.TestCase):
    def test_real_schema_retains_owned_credit_not_physical_claim(self):
        observed = reserve.decode_graph_reserve_records("logger prefix " + reserve_line())
        self.assertEqual(observed["malformed_records"], 0)
        self.assertEqual(observed["records"], [reserve_fields()])
        self.assertEqual(observed["scope"], "owned_device_credit_not_physical_free")

    def test_every_missing_duplicate_extra_non_uint_and_overflow_refuses(self):
        good = reserve_line()
        invalid = [good + " extra=0", good + " partial"]
        for key in reserve.RESERVE_NUMBERS + ("scope",):
            invalid.append(line(reserve.RESERVE_MARKER, {k: v for k, v in reserve_fields().items() if k != key}))
            invalid.append(good + f" {key}=1")
        for key in reserve.RESERVE_NUMBERS:
            for value in (-1, "+1", "1.0", "1e3", "nan", "inf", reserve.U64_MAX + 1):
                invalid.append(reserve_line(**{key: value}))
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                observed = reserve.decode_graph_reserve_records(candidate + "\n" + good)
                self.assertEqual((len(observed["records"]), observed["malformed_records"]), (1, 1))

    def test_n1_output_context_release_and_live_bounds(self):
        invalid = [dict(rows=2), dict(output_rows=2), dict(context=1),
            dict(context=2304), dict(context=513), dict(released=1), dict(live_device=446151169),
            dict(scope="physical_vram_freed"), dict(scope=reserve.RECOVERY_SCOPE)]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.assertEqual(reserve.decode_graph_reserve_records(
                    reserve_line(**changes))["malformed_records"], 1)
        zero = reserve_line(live=223075584)
        self.assertEqual(reserve.decode_graph_reserve_records(zero)["malformed_records"], 0)
        for raw_output_rows in (0, 1):
            self.assertEqual(reserve.decode_graph_reserve_records(reserve_line(
                output_rows=raw_output_rows))["malformed_records"], 0)


class RecoveryParserTests(unittest.TestCase):
    def test_retained_native_recovery_line_matches_source_schema(self):
        # Existing v17 log, first detailed recovery, not a new device execution.
        # It validates the unchanged recovery schema, not the new Host profile.
        native = ("[GLM-EXPERT-RECOVERY] request=chat-3276bec21dc34fba99cdf78a087dda35 reason=increased "
            "ceiling_before=6351672832 ceiling_candidate=6645874688 ceiling_target=6645874688 "
            "ceiling_after=6645874688 ceiling_max=8161075848 native_accounted=4964327424 "
            "native_budget=6645874688 physical_available=3491639296 physical_headroom=1073709056 "
            "ledger_committed=11816320896 ledger_reserved=1988840960 ledger_capacity=14920220672 "
            "effective_limit=14234251136 minimum_growth=67108864 recovery_margin=134217728 "
            "sample=before_recovery unit=bytes scope=owned_credit_not_physical_free")
        observed = reserve.expert_recovery_records(native)
        self.assertEqual(observed["malformed_records"], 0)
        self.assertEqual(observed["records"][0]["ceiling_after"], 6645874688)

    def test_actual_source_schema_and_physical_target_arithmetic(self):
        observed = reserve.expert_recovery_records(recovery_line())
        self.assertEqual(observed["malformed_records"], 0)
        self.assertEqual(observed["records"], [recovery_fields()])
        self.assertEqual(13241417728 - 11811160064 - 1073741824 - 134217728, 212 * reserve.MIB)

    def test_all_source_outcomes_and_non_mib_maximum(self):
        before = 7951351808
        cases = [
            recovery_fields(),
            recovery_fields(reason="envelope_refused", ceiling_after=before, native_budget=before),
            recovery_fields(reason="below_threshold", minimum_growth=512 * reserve.MIB,
                ceiling_target=before, ceiling_after=before, native_budget=before),
            recovery_fields(reason="physical_or_ledger_slack", physical_available=1,
                effective_limit=11811160064, ceiling_candidate=before, ceiling_target=before,
                ceiling_after=before, native_budget=before),
            recovery_fields(reason="already_max", ceiling_max=before, ceiling_candidate=before,
                ceiling_target=before, ceiling_after=before, native_budget=before),
        ]
        for values in cases:
            with self.subTest(reason=values["reason"]):
                self.assertEqual(reserve.expert_recovery_records(
                    line(reserve.RECOVERY_MARKER, values))["malformed_records"], 0)
        # Maximum has a fractional MiB; target never rounds above it.
        capped = recovery_fields(ceiling_max=8062425352, ceiling_candidate=8061452288,
            ceiling_target=8061452288, ceiling_after=8061452288, native_budget=8061452288)
        self.assertEqual(reserve.expert_recovery_records(
            line(reserve.RECOVERY_MARKER, capped))["malformed_records"], 0)

    def test_every_missing_duplicate_non_uint_overflow_and_extra_refuses(self):
        values = recovery_fields()
        good = recovery_line()
        invalid = [good + " partial", good + " extra=0"]
        for key in values:
            invalid.append(line(reserve.RECOVERY_MARKER, {k: v for k, v in values.items() if k != key}))
            invalid.append(good + f" {key}=1")
        for key in reserve.RECOVERY_NUMBERS:
            for value in (-1, "+1", "1.5", "nan", "inf", reserve.U64_MAX + 1):
                invalid.append(recovery_line(**{key: value}))
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                observed = reserve.expert_recovery_records(candidate + "\n" + good)
                self.assertEqual((len(observed["records"]), observed["malformed_records"]), (1, 1))

    def test_cannot_spend_reservations_as_physical_free_or_hide_wrong_policy(self):
        invalid = [dict(physical_available=2503999487), dict(effective_limit=13241417729),
            dict(ledger_committed=reserve.U64_MAX, ledger_reserved=1),
            dict(ledger_capacity=12884901887), dict(ceiling_max=7951351807),
            dict(ceiling_candidate=8173649921), dict(ceiling_target=7951351808),
            dict(ceiling_after=8173649919), dict(native_budget=7951351808),
            dict(native_accounted=8173649921), dict(reason="already_max"),
            dict(reason="physical_or_ledger_slack"), dict(reason="failure"),
            dict(minimum_growth=1), dict(recovery_margin=0), dict(unit="MiB"),
            dict(sample="after_recovery"), dict(scope="physical_vram_freed"), dict(scope=reserve.RESERVE_SCOPE)]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.assertEqual(reserve.expert_recovery_records(
                    recovery_line(**changes))["malformed_records"], 1)


class QualificationTests(unittest.TestCase):
    def test_verbatim_v26_raw_zero_profile_pair(self):
        # Actual startup records retained in aborted candidate1, lines152-153.
        # Parser/shape coverage only: startup is excluded by benchmark slicing,
        # and aborted candidate1 is not model/performance qualification.
        recovery = ("[GLM-EXPERT-RECOVERY] request=chat-05c38c2fdcc04aaabe74ad92d36d89ac "
            "reason=physical_or_ledger_slack ceiling_before=8062425352 ceiling_candidate=8062425352 "
            "ceiling_target=8062425352 ceiling_after=8062425352 ceiling_max=8698761992 "
            "native_accounted=8050831360 native_budget=8062425352 physical_available=1346691072 "
            "physical_headroom=1073709056 ledger_committed=14387751296 ledger_reserved=234590728 "
            "ledger_capacity=14934900736 effective_limit=14660733312 minimum_growth=67108864 "
            "recovery_margin=134217728 sample=before_recovery unit=bytes scope=owned_credit_not_physical_free")
        event = ("[GLM-DECODE-GRAPH-RESERVE] rows=1 output_rows=0 context=256 owned_before=445993472 "
            "owned_after=222996736 live_device=0 released=222996736 expert_budget=8062425352 "
            "scope=owned_device_credit_not_physical_free")
        result = reserve.decode_graph_reserve_qualification([dict(name="literal-schema-only", prompt_tokens=13,
            eval_tokens=1, cache_hits=0, decode_graph_reserve=reserve.decode_graph_reserve_records(event),
            expert_recovery=reserve.expert_recovery_records(recovery))])
        self.assertTrue(result["passed"])
        self.assertFalse(result["ceiling_increased"])
        self.assertEqual(result["requests"][0]["released_bytes"], 222996736)

    def test_two_first_n1_requests_keep_credit_growth_and_residency_separate(self):
        result = reserve.decode_graph_reserve_qualification([row(), row(33, "chat-fixed-b")])
        self.assertTrue(result["passed"])
        self.assertTrue(result["ceiling_increased"])
        self.assertFalse(result["resident_capacity_qualified"])
        self.assertEqual([r["workspace_bytes"] for r in result["requests"]], [223075584, 222996736])
        self.assertEqual([r["expert_budget_delta"] for r in result["requests"]], [212 * reserve.MIB] * 2)

    def test_positive_release_with_physical_decline_is_not_capacity_growth(self):
        candidate = row()
        before = 7951351808
        candidate["expert_recovery"] = reserve.expert_recovery_records(recovery_line(
            reason="physical_or_ledger_slack", physical_available=1, effective_limit=11811160064,
            ceiling_candidate=before, ceiling_target=before, ceiling_after=before, native_budget=before))
        candidate["decode_graph_reserve"] = reserve.decode_graph_reserve_records(reserve_line(expert_budget=before))
        result = reserve.decode_graph_reserve_qualification([candidate])
        self.assertTrue(result["passed"])
        self.assertFalse(result["ceiling_increased"])
        self.assertEqual(result["requests"][0]["expert_budget_delta"], 0)

    def test_live_graph_is_retained_and_never_counted_as_released(self):
        candidate = row()
        candidate["decode_graph_reserve"] = reserve.decode_graph_reserve_records(reserve_line(live=32 * reserve.MIB))
        result = reserve.decode_graph_reserve_qualification([candidate])
        self.assertTrue(result["passed"])
        self.assertEqual(result["requests"][0]["released_bytes"], 223075584 - 32 * reserve.MIB)
        self.assertEqual(result["requests"][0]["owned_after"], 223075584 + 32 * reserve.MIB)
        for raw_output_rows in (0, 1):
            candidate["decode_graph_reserve"] = reserve.decode_graph_reserve_records(
                reserve_line(live=32 * reserve.MIB, output_rows=raw_output_rows))
            self.assertTrue(reserve.decode_graph_reserve_qualification([candidate])["passed"])

    def test_no_decode_cached_failed_or_incomplete_requests_never_qualify(self):
        invalid = [dict(eval_tokens=0), dict(eval_tokens=True), dict(eval_tokens=-1), dict(cache_hits=1),
            dict(cache_hits=False), dict(prompt_tokens=0), dict(prompt_tokens=2048),
            dict(error="HTTP failed"), dict(decode_graph_reserve={}), dict(expert_recovery={})]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.assertFalse(reserve.decode_graph_reserve_qualification([row(**changes)])["passed"])
        self.assertFalse(reserve.decode_graph_reserve_qualification([])["passed"])

    def test_startup_only_missing_duplicate_and_malformed_activity_fail(self):
        for key, parser, good in (
                ("decode_graph_reserve", reserve.decode_graph_reserve_records, reserve_line()),
                ("expert_recovery", reserve.expert_recovery_records, recovery_line())):
            for log in ("", good + "\n" + good, good + "\n" + good + " duplicate=1"):
                candidate = row()
                candidate[key] = parser(log)
                with self.subTest(key=key, log=log):
                    self.assertFalse(reserve.decode_graph_reserve_qualification([candidate])["passed"])

    def test_context_budget_workspace_and_first_threshold_must_match(self):
        invalid = [reserve_line(context=256), reserve_line(expert_budget=7951351808),
            reserve_line(owned_before=446151169, released=223075585),
            reserve_line(owned_after=223075585, released=223075583),
            reserve_line(live=223075584)]
        for log in invalid:
            candidate = row()
            candidate["decode_graph_reserve"] = reserve.decode_graph_reserve_records(log)
            with self.subTest(log=log):
                self.assertFalse(reserve.decode_graph_reserve_qualification([candidate])["passed"])
        candidate = row()
        candidate["expert_recovery"] = reserve.expert_recovery_records(recovery_line(
            reason="below_threshold", minimum_growth=512 * reserve.MIB,
            ceiling_target=7951351808, ceiling_after=7951351808, native_budget=7951351808))
        candidate["decode_graph_reserve"] = reserve.decode_graph_reserve_records(reserve_line(expert_budget=7951351808))
        self.assertFalse(reserve.decode_graph_reserve_qualification([candidate])["passed"])

    def test_duplicate_or_unbound_native_request_id_cannot_reuse_a_decision(self):
        self.assertFalse(reserve.decode_graph_reserve_qualification([row(), row(33)])["passed"])
        candidate = row()
        candidate["expert_recovery"] = reserve.expert_recovery_records(recovery_line(request="-"))
        self.assertFalse(reserve.decode_graph_reserve_qualification([candidate])["passed"])

    def test_revalidates_parsed_records_and_host_mode_is_explicit(self):
        candidate = copy.deepcopy(row())
        candidate["expert_recovery"]["records"][0]["native_budget"] = 1
        self.assertFalse(reserve.decode_graph_reserve_qualification([candidate])["passed"])
        self.assertFalse(reserve.decode_graph_reserve_qualification([row()], False)["passed"])
        with self.assertRaises(ValueError):
            reserve.decode_graph_reserve_qualification([row()], host_token_embedding="1")


if __name__ == "__main__":
    unittest.main()
