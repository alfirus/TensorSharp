import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("glm_layer_major_device_baton",
    Path(__file__).resolve().parents[1] / "glm_layer_major_device_baton.py")
baton = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baton)


def bank_line(**changes):
    values = dict(profile=1, submitted=1872, chunk=512, bank_bytes=122683392,
        outcome="completed", input_calls=176, output_calls=176,
        input_bytes=5398069248, output_bytes=5398069248, host_hidden_bytes=0,
        live_bank_bytes=0, held_ticket_bytes=0, scope=baton.SCOPE)
    values.update(changes)
    return baton.MARKER + " " + " ".join(f"{key}={value}" for key, value in values.items())


def source_line(layer, rows, **changes):
    values = dict(layer=layer, rows=rows, bulk=int(rows > 1), file_bytes=4096,
        file_read_completed_tasks=1, file_read_failed_calls=0, file_read_source_bytes=4096,
        file_view_requested=0, file_view_calls=0, file_view_bytes=0,
        file_view_fallbacks=0, file_view_buffered_bytes=4096,
        file_prefetch_requested=0, file_prefetch_calls=0, file_prefetch_ranges=0,
        file_prefetch_bytes=0, file_prefetch_failures=0, outcome="completed", scope=baton._source.SCOPE)
    values.update(changes)
    return baton._source.MARKER + " " + " ".join(f"{key}={value}" for key, value in values.items())


def request_log(prompt, count, requested):
    lines = [source_line(layer, min(512, prompt - start))
             for layer in range(3, 45) for start in range(0, prompt, 512)]
    if requested and prompt > 512:
        lines.append(bank_line())
    for position in range(prompt, prompt + count):
        lines += [source_line(layer, 1) for layer in range(3, 45)]
        lines.append(baton._source.FORWARD_MARKER
                     + f" kind=0 rows=1 p0={position} outcome=completed")
    return "\n".join(lines)


def request(name, prompt, count, text):
    return dict(name=name, round=0, prompt_tokens=prompt, eval_tokens=count,
                cache_hits=0, layer_major_device_baton=baton.records(text))


class DeviceBatonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = {requested: [request_log(prompt, count, requested)
            for _, prompt, count in baton.QUICK_REQUESTS] for requested in (0, 1)}
        cls.decoded = {requested: [request(name, prompt, count, text)
            for (name, prompt, count), text in zip(baton.QUICK_REQUESTS, cls.logs[requested])]
            for requested in (0, 1)}

    def rows(self, requested=1):
        return copy.deepcopy(self.decoded[requested])

    def test_exact_explicit_setting_and_invalid_values(self):
        for value, expected in ((None, None), ("0", 0), ("1", 1)):
            self.assertEqual(baton.setting(value), expected)
        for value in ("", "01", "+1", "true", " 1", "2", True, 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                baton.setting(value)

    def test_native_exact_completed_geometry(self):
        observed = baton.records(bank_line())
        self.assertEqual(observed["malformed_records"], 0)
        event = observed["records"][0]
        self.assertEqual((event["bank_bytes"], event["input_calls"], event["output_calls"]),
                         (122683392, 176, 176))
        self.assertEqual((event["input_bytes"], event["output_bytes"]), (5398069248, 5398069248))

    def test_unknown_missing_duplicate_and_bad_scope_are_malformed(self):
        valid = bank_line()
        for text in (valid + " extra=1", valid + " input_calls=176",
                     valid.replace(" input_calls=176", ""), valid + " trailing",
                     valid + " " + baton.MARKER, valid.replace(baton.SCOPE, "other")):
            with self.subTest(text=text):
                self.assertGreater(baton.records(text)["malformed_records"], 0)

    def test_bounded_canonical_integers(self):
        for value in ("-1", "01", "+1", "1.0", "nan", "inf", str(2**64), "True"):
            self.assertEqual(baton.records(bank_line(input_calls=value))["malformed_records"], 1)

    def test_unsupported_geometry_and_host_bounce_rejected(self):
        for changes in (dict(profile=2), dict(chunk=256), dict(submitted=512),
                        dict(submitted=2049), dict(bank_bytes=122683391), dict(host_hidden_bytes=1)):
            self.assertEqual(baton.records(bank_line(**changes))["malformed_records"], 1)

    def test_each_partial_transfer_or_live_owner_refuses(self):
        for key in ("input_calls", "output_calls", "input_bytes", "output_bytes"):
            correct = 176 if key.endswith("calls") else 5398069248
            for value in (correct - 1, correct + 1):
                self.assertEqual(baton.records(bank_line(**{key: value}))["malformed_records"], 1)
        for key in ("live_bank_bytes", "held_ticket_bytes"):
            self.assertEqual(baton.records(bank_line(**{key: 1}))["malformed_records"], 1)

    def test_full_fixed_control_and_candidate_without_scope_overclaims(self):
        for requested in (0, 1):
            q = baton.qualification(self.rows(requested), requested)
            self.assertTrue(q["passed"])
            self.assertEqual(q["completed_calls"], requested)
            self.assertEqual([row["completed_calls"] for row in q["requests"]], [requested, 0])
            self.assertFalse(q["all_kind5_owners_closed_qualified"])
            for key in ("residency_qualified", "physical_io_qualified", "performance_qualified",
                        "numerical_quality_qualified"):
                self.assertFalse(q[key])

    def test_only_long_prefill_requires_activity(self):
        rows = self.rows()
        rows[0]["layer_major_device_baton"]["records"] = []
        self.assertFalse(baton.qualification(rows, 1)["passed"])
        self.assertTrue(baton.qualification(self.rows(), 1)["requests"][1]["passed"])

    def test_off_and_short_request_cannot_claim_banks(self):
        rows = self.rows(0)
        rows[0]["layer_major_device_baton"]["records"] = self.rows()[0]["layer_major_device_baton"]["records"]
        self.assertFalse(baton.qualification(rows, 0)["passed"])
        rows = self.rows()
        rows[1]["layer_major_device_baton"]["records"] = rows[0]["layer_major_device_baton"]["records"]
        self.assertFalse(baton.qualification(rows, 1)["passed"])

    def test_failed_and_duplicate_events_never_qualify(self):
        for failed in (False, True):
            rows = self.rows()
            event = rows[0]["layer_major_device_baton"]["records"][0]
            if failed:
                event["outcome"] = "failed"
            else:
                rows[0]["layer_major_device_baton"]["records"].append(dict(event))
            self.assertFalse(baton.qualification(rows, 1)["passed"])

    def test_event_must_follow_all_prefill_and_precede_any_decode(self):
        for key, value in (("prefill_calls_before", 167), ("prefill_calls_before", 169),
                           ("decode_calls_before", 1)):
            rows = self.rows()
            rows[0]["layer_major_device_baton"]["records"][0][key] = value
            self.assertFalse(baton.qualification(rows, 1)["passed"])

    def test_postcall_association_and_every_selected_layer_required(self):
        for bad in (self.logs[1][0].replace("p0=1872", "p0=1873", 1),
                    self.logs[1][0].replace("layer=3 rows=512", "layer=4 rows=512", 1),
                    self.logs[1][0].replace("outcome=completed", "outcome=exception", 1)):
            rows = self.rows()
            rows[0]["layer_major_device_baton"] = baton.records(bad)
            self.assertFalse(baton.qualification(rows, 1)["passed"])

    def test_source_byte_conservation_no_fallback_or_prefetch(self):
        for before, after in (("file_read_source_bytes=4096", "file_read_source_bytes=4095"),
                              ("file_read_failed_calls=0", "file_read_failed_calls=1"),
                              ("file_view_fallbacks=0", "file_view_fallbacks=1"),
                              ("file_prefetch_requested=0", "file_prefetch_requested=1")):
            rows = self.rows()
            rows[0]["layer_major_device_baton"] = baton.records(self.logs[1][0].replace(before, after, 1))
            self.assertFalse(baton.qualification(rows, 1)["passed"])

    def test_prefill_chronology_not_just_width_counter(self):
        rows = self.rows()
        source = rows[0]["layer_major_device_baton"]["source_activity"]["records"]
        source[0], source[4] = source[4], source[0]
        self.assertFalse(baton.qualification(rows, 1)["passed"])

    def test_error_cache_hits_bool_and_wrong_api_counts_refuse(self):
        for key, value in (("error", "native failed"), ("cache_hits", 1), ("cache_hits", False),
                           ("prompt_tokens", 1871), ("eval_tokens", 31), ("round", False),
                           ("name", "startup")):
            rows = self.rows()
            rows[0][key] = value
            self.assertFalse(baton.qualification(rows, 1)["passed"])

    def test_missing_extra_reordered_requests_refuse(self):
        rows = self.rows()
        for bad in ([], rows[:1], rows + [rows[1]], list(reversed(rows)), None, [None]):
            self.assertFalse(baton.qualification(bad, 1)["passed"])
        for requested in (None, True, "1", 2):
            self.assertFalse(baton.qualification(rows, requested)["passed"])

    def test_forged_decoded_types_and_malformed_count_refuse(self):
        for field, value in (("input_calls", True), ("live_bank_bytes", "0"),
                             ("prefill_calls_before", True)):
            rows = self.rows()
            rows[0]["layer_major_device_baton"]["records"][0][field] = value
            self.assertFalse(baton.qualification(rows, 1)["passed"])
        for value in (True, -1, 1, "0"):
            rows = self.rows()
            rows[0]["layer_major_device_baton"]["malformed_records"] = value
            self.assertFalse(baton.qualification(rows, 1)["passed"])
        for value in (None, [], True):
            rows = self.rows()
            rows[0]["layer_major_device_baton"] = value
            self.assertFalse(baton.qualification(rows, 1)["passed"])
        rows = self.rows()
        rows[0]["layer_major_device_baton"]["records"][0]["unknown"] = 1
        self.assertFalse(baton.qualification(rows, 1)["passed"])

    def test_event_can_parse_other_native_length_but_not_pass_fixed_quick_plan(self):
        text = bank_line(submitted=2048, bank_bytes=134217728,
                         input_bytes=5905580032, output_bytes=5905580032)
        self.assertEqual(baton.records(text)["malformed_records"], 0)
        rows = self.rows()
        observed = baton.records(text)["records"][0]
        observed["prefill_calls_before"] = 168
        rows[0]["layer_major_device_baton"]["records"] = [observed]
        self.assertFalse(baton.qualification(rows, 1)["passed"])

    def test_profile_requires_same_explicit_best_profile_and_no_other_experiment(self):
        args = SimpleNamespace(concurrent=False, unified_memory=True, spec=False, prefix_cache=False,
            repeat=1, plan="quick", gpu_telemetry=False, server_arg=[], mmproj="none", think=False)
        settings = {**{key: "1" for key in baton.REQUIRED_ON},
                    **{key: "0" for key in baton.REQUIRED_OFF},
                    **baton.REQUIRED_VALUES, baton.ENV: "1"}
        baton.validate_profile(args, settings)
        for key in tuple(settings):
            bad = dict(settings)
            del bad[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                baton.validate_profile(args, bad)
        for key in ("TS_HOST_MOE_FILE_SOURCE_VIEW", "TS_HOST_MOE_VMM_WEIGHTS",
                    "TS_GLM_DECODE_GRAPH_RESERVE", "TS_HOST_MOE_FILE_SOURCE_VIEW_PREFETCH"):
            bad = dict(settings, **{key: "1"})
            with self.assertRaises(ValueError):
                baton.validate_profile(args, bad)
        for key in ("TS_GLM_TRACE", "TS_GLM_TRACE_DUMP_DIR"):
            with self.assertRaises(ValueError):
                baton.validate_profile(args, settings, {key: "0"})


if __name__ == "__main__":
    unittest.main()
