import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("glm_file_read_workers",
    Path(__file__).resolve().parents[1] / "glm_file_read_workers.py")
workers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workers)


def line(**changes):
    values = dict(read_pool_requested=8, read_pool_workers=8, read_pool_maximum=15,
        read_pool_slots=9, file_read_calls=1, file_read_tasks=24,
        file_read_completed_tasks=24, file_read_failed_calls=0,
        file_read_source_bytes=40000000, file_bytes=40000000,
        layer=0, rows=1, total_ms=3.5, outcome="completed", scope=workers.SCOPE)
    values.update(changes)
    return workers.MARKER + " " + " ".join(str(key)+"="+str(value) for key, value in values.items())


def row(text):
    return dict(name="decode-a", file_read_workers=workers.records(text))


class FileWorkersTests(unittest.TestCase):
    def test_settings(self):
        self.assertIsNone(workers.setting(None))
        for number in range(17):
            self.assertEqual(workers.setting(str(number)), number)
        for invalid in ("", "01", "17", "-1", "+8", " 8", "8 ", "1.0", "0x8"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                workers.setting(invalid)

    def test_actual_completed(self):
        q = workers.qualification([row(line()), row(line())], 8)
        self.assertTrue(q["passed"])
        self.assertEqual(q["requests"][0]["tasks"], 24)
        self.assertFalse(q["performance_qualified"])
        self.assertFalse(q["physical_io_qualified"])

    def test_control_and_clamped(self):
        for requested in (0, 16):
            self.assertTrue(workers.qualification([row(line(read_pool_requested=requested,
                read_pool_workers=15, read_pool_slots=16))], requested)["passed"])

    def test_serial(self):
        self.assertTrue(workers.qualification([row(line(read_pool_requested=1,
            read_pool_workers=1, read_pool_slots=1))], 1)["passed"])

    def test_single_cpu_auto(self):
        for slots in (1, 2):
            self.assertTrue(workers.qualification([row(line(read_pool_requested=0,
                read_pool_workers=1, read_pool_maximum=1, read_pool_slots=slots))], 0)["passed"])

    def test_missing_and_duplicate_fields(self):
        for text in (line().replace(" file_read_tasks=24", ""), line()+" read_pool_workers=8"):
            self.assertEqual(workers.records(text)["malformed_records"], 1)
            self.assertFalse(workers.qualification([row(text)], 8)["passed"])

    def test_failed_and_partial_activity(self):
        for change in (dict(outcome="exception"), dict(outcome="not_completed"),
                       dict(file_read_failed_calls=1), dict(file_read_completed_tasks=23)):
            self.assertFalse(workers.qualification([row(line(**change))], 8)["passed"])

    def test_ready_ram_lease_is_not_source_read(self):
        self.assertFalse(workers.qualification([row(line(file_read_source_bytes=0))], 8)["passed"])

    def test_invalid_counters(self):
        for change in (dict(read_pool_workers=9), dict(read_pool_maximum=16),
            dict(read_pool_slots=8), dict(file_read_completed_tasks=25),
            dict(file_read_failed_calls=2), dict(file_read_calls=25),
            dict(file_read_source_bytes=40000001), dict(file_read_tasks=-1),
            dict(file_bytes=2**64), dict(scope="physical_io")):
            with self.subTest(change=change):
                self.assertEqual(workers.records(line(**change))["malformed_records"], 1)

    def test_no_activity_or_wrong_setting(self):
        self.assertFalse(workers.qualification([], 8)["passed"])
        self.assertFalse(workers.qualification([row("")], 8)["passed"])
        self.assertFalse(workers.qualification([row(line())], 4)["passed"])

    def test_each_request_needs_activity(self):
        self.assertFalse(workers.qualification([row(line()), row("")], 8)["passed"])

    def test_wide_reads_alone_do_not_qualify_n1(self):
        self.assertFalse(workers.qualification([row(line(rows=512))], 8)["passed"])
        self.assertTrue(workers.qualification([row(line(rows=512)+"\n"+line())], 8)["passed"])

    def test_worker_geometry_must_not_drift(self):
        self.assertFalse(workers.qualification([row(line()), row(line(read_pool_workers=7,
            read_pool_maximum=7, read_pool_slots=8))], 8)["passed"])

    def test_all_hit_call_supplies_no_activity(self):
        all_hit = line(**{key: 0 for key in workers.NUMBERS})
        self.assertEqual(workers.records(all_hit)["malformed_records"], 0)
        self.assertFalse(workers.qualification([row(all_hit)], 8)["passed"])
        self.assertTrue(workers.qualification([row(all_hit+"\n"+line())], 8)["passed"])

    def test_hint_setting(self):
        self.assertIsNone(workers.hint_setting(None))
        self.assertEqual(workers.hint_setting("0"), 0)
        self.assertEqual(workers.hint_setting("1"), 1)
        for value in ("", "01", "2", "+1", " 1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                workers.hint_setting(value)

    def test_actual_hint_required_for_control_and_candidate(self):
        for value in (0, 1):
            self.assertTrue(workers.qualification([row(line(file_read_random_access=value))], 8, value)["passed"])
            self.assertFalse(workers.qualification([row(line(file_read_random_access=1-value))], 8, value)["passed"])
            self.assertFalse(workers.qualification([row(line())], 8, value)["passed"])

    def test_mixed_hint_records_never_qualify(self):
        mixed = line(file_read_random_access=0)+"\n"+line(file_read_random_access=1)
        self.assertFalse(workers.qualification([row(mixed)], 8, 0)["passed"])
        self.assertFalse(workers.qualification([row(mixed)], 8, 1)["passed"])

    def test_invalid_hint_and_false_inactive_evidence(self):
        for value in (2, -1, "01", "+1"):
            self.assertEqual(workers.records(line(file_read_random_access=value))["malformed_records"], 1)
        inactive = {key: 0 for key in workers.NUMBERS}
        self.assertEqual(workers.records(line(**inactive, file_read_random_access=1))["malformed_records"], 1)


if __name__ == "__main__":
    unittest.main()
