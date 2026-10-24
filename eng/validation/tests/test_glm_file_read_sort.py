import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("glm_file_read_sort",
    Path(__file__).resolve().parents[1] / "glm_file_read_sort.py")
sort = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sort)


def line(layer=3, rows=16, requested=1, changed=True, **changes):
    active = requested == 1
    values = dict(layer=layer, rows=rows, bulk=int(type(rows) is int and rows > 1), file_bytes=4096,
        file_sort_requested=requested, file_sort_calls=int(active),
        file_sort_changed_calls=int(active and changed), file_sort_misses=2 if active else 0,
        file_sort_bytes=4096 if active else 0, file_sort_inversions_before=int(active and changed),
        file_sort_inversions_after=0, outcome="completed", scope=sort.SCOPE)
    values.update(changes)
    return sort.MARKER + " " + " ".join(f"{key}={value}" for key, value in values.items())


def complete_log(prompt=16, count=2, requested=1, wide_changed=True, decode_changed=True):
    result = [line(layer, min(512, prompt - start), requested, wide_changed)
        for layer in sorted(sort.LAYERS) for start in range(0, prompt, 512)]
    for position in range(prompt, prompt + count):
        result += [line(layer, 1, requested, decode_changed) for layer in sorted(sort.LAYERS)]
        result.append(sort.FORWARD_MARKER + f" kind=0 rows=1 p0={position} outcome=completed")
    return "\n".join(result)


def row(text, prompt=16, count=2, **changes):
    result = dict(name="fixture", prompt_tokens=prompt, eval_tokens=count, cache_hits=0,
                  file_read_sort=sort.records(text))
    result.update(changes)
    return result


class FileReadSortTests(unittest.TestCase):
    def test_canonical_settings(self):
        for value, expected in ((None, None), ("0", 0), ("1", 1)):
            self.assertEqual(sort.setting(value), expected)
        for value in ("", "01", "2", "+1", " 1", "1 ", "1.0", 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sort.setting(value)

    def test_complete_control_and_candidate(self):
        for requested in (0, 1):
            q = sort.qualification([row(complete_log(requested=requested))], requested)
            self.assertTrue(q["passed"])
            self.assertEqual(q["requests"][0]["phases"]["prefill"]["calls"], 42)
            self.assertEqual(q["requests"][0]["phases"]["decode"]["calls"], 84)
            self.assertFalse(q["physical_io_qualified"])
            self.assertFalse(q["performance_qualified"])
            self.assertFalse(q["numerical_quality_qualified"])

    def test_all_quick_prompt_widths_and_positions(self):
        for prompt, count in ((1872, 32), (33, 128)):
            q = sort.qualification([row(complete_log(prompt, count), prompt, count)], 1)
            self.assertTrue(q["passed"])

    def test_both_phases_and_every_request_need_actual_changes(self):
        good = row(complete_log())
        for text in (complete_log(wide_changed=False), complete_log(decode_changed=False)):
            self.assertFalse(sort.qualification([row(text)], 1)["passed"])
            self.assertFalse(sort.qualification([good, row(text)], 1)["passed"])
        self.assertFalse(sort.qualification([], 1)["passed"])
        self.assertFalse(sort.qualification([good, row("")], 1)["passed"])

    def test_missing_and_duplicate_fields_and_markers(self):
        for text in (line().replace(" file_sort_misses=2", ""), line()+" file_sort_calls=1",
                     line()+" partial", line()+" "+sort.MARKER):
            self.assertEqual(sort.records(text)["malformed_records"], 1)

    def test_counter_scope_status_and_geometry_refusals(self):
        for changes in (dict(file_sort_requested=2), dict(file_sort_calls=3),
            dict(file_sort_changed_calls=2), dict(file_sort_misses=289),
            dict(file_sort_bytes=4095), dict(file_sort_inversions_before=0),
            dict(file_sort_inversions_after=1), dict(file_sort_inversions_before=2),
            dict(file_sort_calls=0), dict(file_sort_requested=0), dict(outcome="failed"),
            dict(outcome="exception"), dict(scope="physical_file_order"), dict(layer=2),
            dict(layer=45), dict(rows=0), dict(rows=8), dict(rows=513), dict(bulk=0)):
            with self.subTest(changes=changes):
                self.assertEqual(sort.records(line(**changes))["malformed_records"], 1)
        self.assertEqual(sort.records(line(rows=1, file_sort_misses=9))["malformed_records"], 1)

    def test_uints_are_canonical_and_bounded(self):
        for key in sort.NUMBERS:
            for value in ("01", "+1", "-1", "1.0", "nan", str(2**64)):
                with self.subTest(key=key, value=value):
                    self.assertGreater(sort.records(line(**{key:value}))["malformed_records"], 0)

    def test_all_hit_frames_keep_requested_flag_without_false_activity(self):
        all_hit = {key:0 for key in sort.SORT_COUNTERS}
        text = complete_log().replace(line(), line(file_bytes=0, **all_hit), 1)
        self.assertTrue(sort.qualification([row(text)], 1)["passed"])
        no_read_activity = "\n".join(line(layer, 16, file_bytes=0, **all_hit) for layer in sort.LAYERS)
        self.assertFalse(sort.qualification([row(no_read_activity)], 1)["passed"])

    def test_unsorted_miss_path_cannot_hide_among_changed_calls(self):
        unsorted = {key:0 for key in sort.SORT_COUNTERS}
        text = complete_log().replace(line(), line(**unsorted), 1)
        self.assertFalse(sort.qualification([row(text)], 1)["passed"])

    def test_failed_outer_call_cannot_qualify_masked_zero_activity(self):
        zero = {key:0 for key in sort.SORT_COUNTERS}
        bad = line(outcome="exception", file_bytes=0, **zero)
        self.assertEqual(sort.records(bad)["malformed_records"], 0)
        self.assertFalse(sort.qualification([row(complete_log().replace(line(),bad,1))],1)["passed"])

    def test_missing_or_duplicate_layer_never_qualifies(self):
        text = complete_log()
        for bad in (text.replace(line(), "", 1), text.replace(line(), line(4), 1),
                    text.replace(line(rows=1), "", 1), text.replace(line(rows=1), line(4,1),1)):
            self.assertFalse(sort.qualification([row(bad)], 1)["passed"])

    def test_unknown_versioned_sort_field_and_reordered_n1_layers_refuse(self):
        self.assertEqual(sort.records(line()+" file_sort_future=1")["malformed_records"], 1)
        text=complete_log();a,b=line(rows=1),line(4,1)
        text=text.replace(a+"\n"+b,b+"\n"+a,1)
        self.assertFalse(sort.qualification([row(text)],1)["passed"])

    def test_postcall_positions_and_complete_n1_association_are_required(self):
        text = complete_log()
        for bad in (text.replace("p0=16", "p0=17"), text.replace("p0=17", "p0=18"),
                    text.replace(sort.FORWARD_MARKER+" kind=0 rows=1 p0=16 outcome=completed", ""),
                    text.replace("p0=16 outcome=completed", "p0=16 outcome=exception")):
            self.assertFalse(sort.qualification([row(bad)], 1)["passed"])

    def test_wrong_requested_flag_or_control_false_activity_refuses(self):
        self.assertFalse(sort.qualification([row(complete_log())], 0)["passed"])
        mixed = complete_log().replace("file_sort_requested=1", "file_sort_requested=0", 1)
        self.assertFalse(sort.qualification([row(mixed)], 1)["passed"])
        self.assertFalse(sort.qualification([row(complete_log())], True)["passed"])

    def test_aggregate_u64_overflow_is_not_a_qualification(self):
        text=complete_log().replace("file_bytes=4096",f"file_bytes={2**64-1}")
        text=text.replace("file_sort_bytes=4096",f"file_sort_bytes={2**64-1}")
        self.assertFalse(sort.qualification([row(text)],1)["passed"])

    def test_decoded_record_tampering_does_not_bypass_validation(self):
        valid=row(complete_log())
        for key,value in (("file_sort_calls",True),("file_sort_bytes",4095),
                          ("scope","physical_io"),("file_sort_inversions_after",1)):
            changed=copy.deepcopy(valid)
            changed["file_read_sort"]["records"][0][key]=value
            self.assertFalse(sort.qualification([changed],1)["passed"])
        for value in (None,False,1):
            changed=copy.deepcopy(valid);changed["file_read_sort"]["malformed_records"]=value
            self.assertFalse(sort.qualification([changed],1)["passed"])

    def test_cache_hits_empty_rows_invalid_shapes_and_scalar_tail_refuse(self):
        valid = row(complete_log())
        for changed in (dict(cache_hits=1), dict(cache_hits=False), dict(error="failed request"),
                        dict(prompt_tokens=0), dict(prompt_tokens=True),
                        dict(eval_tokens=0), dict(eval_tokens=2048), dict(eval_tokens=True)):
            self.assertFalse(sort.qualification([valid | changed], 1)["passed"])
        self.assertFalse(sort.qualification([row(complete_log(513),513)], 1)["passed"])


class FileReadSortProfileTests(unittest.TestCase):
    @staticmethod
    def settings():
        return ({key:"1" for key in sort.REQUIRED_ON}
            | {key:"0" for key in sort.REQUIRED_OFF}
            | sort.REQUIRED_VALUES | {sort.ENV:"1"})

    @staticmethod
    def args():
        return SimpleNamespace(concurrent=False, unified_memory=True, spec=False,
            prefix_cache=False, repeat=1, plan="quick", gpu_telemetry=False, server_arg=[],
            mmproj="none", think=False, require_file_read_workers=False,
            require_file_random_access=False, require_mapped_probation=False, require_vmm_pressure=False)

    def test_isolated_on_and_off_profiles(self):
        for requested in ("0","1"):
            sort.validate_profile(self.args(), self.settings() | {sort.ENV:requested})

    def test_every_required_setting_is_explicit(self):
        for key in self.settings():
            values = self.settings();del values[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                sort.validate_profile(self.args(),values)

    def test_confounded_modes_refuse(self):
        for key,value in dict(concurrent=True, unified_memory=False, spec=True, prefix_cache=True,
            repeat=2, plan="decode", gpu_telemetry=True, server_arg=["--context-size=1024"],
            mmproj="auto", think=True, require_file_read_workers=True,
            require_file_random_access=True, require_mapped_probation=True, require_vmm_pressure=True).items():
            args=self.args();setattr(args,key,value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                sort.validate_profile(args,self.settings())


if __name__ == "__main__":
    unittest.main()
