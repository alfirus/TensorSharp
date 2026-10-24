import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("retain", Path(__file__).parents[1] / "glm_decode_retain_peer.py")
retain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retain)


class RetainedPeerTests(unittest.TestCase):
    def event(self):
        return dict(profile=1, layer=3, slots=291, hot=280, owned_bytes=2200000000,
                    decode_quota=2100000000, room_bytes=5000000000,
                    absent_floor_bytes=1000000000, absent_share_bytes=5100000000,
                    outcome="completed", scope=retain.SCOPE)

    def log(self, event=None):
        return retain.TAG + " " + " ".join(f"{key}={value}" for key, value in (event or self.event()).items())

    def row(self):
        return dict(name="decode", eval_tokens=128, prompt_tokens=33, cache_hits=0,
                    decode_retain_peer=retain.records(self.log()))

    def test_completed_request_has_retention_not_new_capacity(self):
        q = retain.qualification([self.row()])
        self.assertTrue(q["passed"])
        self.assertEqual(2200000000, q["requests"][0]["retained_payload_bytes"])
        self.assertEqual(0, q["requests"][0]["newly_granted_bytes"])
        self.assertFalse(q["physical_capacity_qualified"])

    def test_flag_requires_canonical_boolean(self):
        self.assertFalse(retain.enabled(None))
        self.assertFalse(retain.enabled("0"))
        self.assertTrue(retain.enabled("1"))
        for value in ("", "01", " 1", "true", 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                retain.enabled(value)

    def test_startup_cannot_credit_empty_request(self):
        row = self.row()
        row["decode_retain_peer"] = retain.records("[HOSTMOE-REBALANCE] layer=3")
        self.assertFalse(retain.qualification([row])["passed"])

    def test_conditional_activity_does_not_credit_empty_request(self):
        empty = self.row()
        empty["decode_retain_peer"] = retain.records("")
        q = retain.qualification([self.row(), empty])
        self.assertTrue(q["passed"])
        self.assertEqual(1, q["active_requests"])
        self.assertFalse(q["all_requests_active"])
        self.assertFalse(q["requests"][1]["passed"])
        self.assertTrue(q["requests"][1]["valid"])
        self.assertEqual("not_observed", q["requests"][1]["activity"])
        self.assertEqual(0, q["requests"][1]["retained_payload_bytes"])
        self.assertFalse(retain.qualification([empty, empty])["passed"])

    def test_bad_record_cannot_hide_beside_good_record(self):
        row = self.row()
        row["decode_retain_peer"] = retain.records(self.log() + "\n" + self.log() + " hot=1")
        self.assertEqual(1, row["decode_retain_peer"]["malformed_records"])
        self.assertFalse(retain.qualification([row])["passed"])

    def test_missing_unknown_duplicate_and_ambiguous_fields(self):
        for text in (self.log().replace(" hot=280", ""), self.log() + " extra=1",
                     self.log() + " hot=1", self.log() + " " + self.log()):
            self.assertEqual(1, retain.records(text)["malformed_records"])

    def test_geometry_requires_hot_eligible_peer_and_protected_floors(self):
        for key, value in (("hot", 0), ("hot", 292), ("slots", 0), ("profile", 2),
                           ("decode_quota", 2200000000), ("room_bytes", 5100000000),
                           ("absent_floor_bytes", 5000000001), ("outcome", "failed"),
                           ("scope", "physical_free"), ("owned_bytes", 1 << 64)):
            event = self.event()
            event[key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(1, retain.records(self.log(event))["malformed_records"])

    def test_noncanonical_values_fail(self):
        for value in ("01", "-1", "+1", "1.0", "18446744073709551616"):
            self.assertEqual(1, retain.records(self.log().replace("profile=1", "profile=" + value))["malformed_records"])

    def test_no_decode_prefix_hits_or_failed_request_refused(self):
        for key, value in (("eval_tokens", 0), ("eval_tokens", True), ("cache_hits", 1),
                           ("prompt_tokens", 0), ("error", "HTTP failed")):
            row = self.row()
            row[key] = value
            self.assertFalse(retain.qualification([row])["passed"])

    def test_mutated_report_is_revalidated(self):
        row = self.row()
        row["decode_retain_peer"]["records"][0]["hot"] = True
        self.assertFalse(retain.qualification([row])["passed"])

    def test_duplicate_peer_cannot_double_count(self):
        row = self.row()
        row["decode_retain_peer"]["records"].append(copy.deepcopy(row["decode_retain_peer"]["records"][0]))
        self.assertFalse(retain.qualification([row])["passed"])

    def test_empty_batch_cannot_pass(self):
        self.assertFalse(retain.qualification([])["passed"])


if __name__ == "__main__":
    unittest.main()
