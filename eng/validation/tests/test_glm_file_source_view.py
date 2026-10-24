import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("glm_file_source_view",
    Path(__file__).resolve().parents[1] / "glm_file_source_view.py")
view = importlib.util.module_from_spec(spec)
spec.loader.exec_module(view)


def line(layer=3, rows=16, requested=1, **changes):
    values = dict(layer=layer, rows=rows, bulk=int(type(rows) is int and rows > 1),
        file_bytes=4096, file_read_completed_tasks=2, file_read_failed_calls=0,
        file_read_source_bytes=4096, file_view_requested=requested,
        file_view_calls=2 if requested == 1 else 0,
        file_view_bytes=4096 if requested == 1 else 0, file_view_fallbacks=0,
        file_view_buffered_bytes=0 if requested == 1 else 4096,
        outcome="completed", scope=view.SCOPE)
    values.update(changes)
    return view.MARKER + " " + " ".join(f"{k}={v}" for k, v in values.items())


def log(prompt=16, count=2, requested=1, **changes):
    result = [line(layer, min(512, prompt-start), requested, **changes)
        for layer in sorted(view.LAYERS) for start in range(0, prompt, 512)]
    for p0 in range(prompt, prompt+count):
        result += [line(layer, 1, requested, **changes) for layer in sorted(view.LAYERS)]
        result.append(view.FORWARD_MARKER + f" kind=0 rows=1 p0={p0} outcome=completed")
    return "\n".join(result)


def row(text, prompt=16, count=2, **changes):
    result = dict(name="fixture", prompt_tokens=prompt, eval_tokens=count, cache_hits=0,
        file_source_view=view.records(text))
    result.update(changes)
    return result


class FileSourceViewTests(unittest.TestCase):
    def test_canonical_settings(self):
        for value, expected in ((None, None), ("0", 0), ("1", 1)):
            self.assertEqual(view.setting(value), expected)
        for value in ("", "01", "+1", " 1", "1 ", "2", 1, True):
            with self.subTest(value=value), self.assertRaises(ValueError): view.setting(value)

    def test_complete_control_candidate_and_quick_shapes(self):
        for prompt, count in ((16,2), (1872,32), (33,128)):
            for requested in (0,1):
                q = view.qualification([row(log(prompt,count,requested),prompt,count)],requested)
                self.assertTrue(q["passed"])
                for k in ("physical_io_qualified","residency_qualified","performance_qualified","numerical_quality_qualified"):
                    self.assertFalse(q[k])

    def test_empty_and_each_request_must_qualify(self):
        good = row(log())
        self.assertFalse(view.qualification([],1)["passed"])
        self.assertFalse(view.qualification([good,row("")],1)["passed"])
        self.assertFalse(view.qualification([good],True)["passed"])

    def test_counter_conservation_and_bounds(self):
        for changes in (dict(file_view_bytes=4095),dict(file_view_buffered_bytes=1),
            dict(file_view_calls=3),dict(file_view_fallbacks=3),dict(file_view_calls=0),
            dict(file_read_failed_calls=1),dict(file_read_source_bytes=4095),
            dict(file_read_completed_tasks=0),dict(file_view_requested=0),
            dict(file_bytes=0,file_read_source_bytes=0,file_view_bytes=0),
            dict(file_bytes=(1<<64)),dict(file_view_bytes=-1)):
            with self.subTest(changes=changes):
                self.assertGreater(view.records(line(**changes))["malformed_records"],0)

    def test_fallback_preserves_logical_accounting_but_is_not_qualified(self):
        text=log(file_view_bytes=2048,file_view_buffered_bytes=2048,file_view_fallbacks=1)
        self.assertEqual(view.records(text)["malformed_records"],0)
        self.assertFalse(view.qualification([row(text)],1)["passed"])

    def test_disabled_generation_buffered_reads_are_not_qualified(self):
        text=log(file_view_calls=0,file_view_bytes=0,file_view_buffered_bytes=4096)
        self.assertEqual(view.records(text)["malformed_records"],0)
        self.assertFalse(view.qualification([row(text)],1)["passed"])

    def test_no_activity_cannot_qualify_candidate(self):
        text=log(file_bytes=0,file_read_completed_tasks=0,file_read_source_bytes=0,
                 file_view_calls=0,file_view_bytes=0)
        self.assertTrue(view.qualification([row(text.replace('file_view_requested=1','file_view_requested=0'))],0)["passed"])
        self.assertFalse(view.qualification([row(text)],1)["passed"])

    def test_missing_duplicate_unknown_fields_and_marker(self):
        for text in (line().replace(' file_view_calls=2',''),line()+' file_view_calls=2',
                     line()+' file_view_future=1',line()+' partial',line()+' '+view.MARKER):
            self.assertGreater(view.records(text)["malformed_records"],0)

    def test_noncanonical_native_numbers(self):
        for key in view.NUMBERS:
            for value in ('01','+1','-1','1.0',str(1<<64)):
                with self.subTest(key=key,value=value):
                    self.assertGreater(view.records(line(**{key:value}))["malformed_records"],0)

    def test_shapes_scope_status(self):
        for changes in (dict(layer=2),dict(layer=45),dict(rows=0),dict(rows=8),
            dict(rows=513),dict(bulk=0),dict(file_view_requested=2),dict(scope='physical_io'),dict(outcome='failed')):
            self.assertGreater(view.records(line(**changes))["malformed_records"],0)
        self.assertFalse(view.qualification([row(log(outcome='exception'))],1)["passed"])

    def test_every_layer_and_chunk_is_required(self):
        text=log(prompt=1872,count=2)
        lines=text.splitlines()
        self.assertFalse(view.qualification([row('\n'.join(lines[1:]),1872,2)],1)["passed"])
        self.assertFalse(view.qualification([row(text+'\n'+lines[0],1872,2)],1)["passed"])

    def test_exact_n1_positions_and_postcall_are_required(self):
        text=log()
        for bad in (text.replace('p0=17','p0=18'),text.replace('p0=16','p0=15'),
            text.replace(view.FORWARD_MARKER,'[ignored]'),text.replace('p0=16 outcome=completed','p0=16 outcome=exception')):
            self.assertFalse(view.qualification([row(bad)],1)["passed"])

    def test_n1_layer_order_is_required(self):
        lines=log().splitlines();lines[42],lines[43]=lines[43],lines[42]
        self.assertFalse(view.qualification([row('\n'.join(lines))],1)["passed"])

    def test_failed_cached_and_scalar_tail_requests_are_rejected(self):
        for changes in (dict(error='failed'),dict(cache_hits=1),dict(eval_tokens=True),dict(prompt_tokens='16')):
            self.assertFalse(view.qualification([row(log(),**changes)],1)["passed"])
        self.assertFalse(view.qualification([row(log(513,2),513,2)],1)["passed"])

    def test_decoded_data_cannot_bypass_validation(self):
        good=row(log())
        for value in ('4096',True,-1,1<<64):
            bad=copy.deepcopy(good);bad['file_source_view']['records'][0]['file_view_bytes']=value
            self.assertFalse(view.qualification([bad],1)["passed"])
        bad=copy.deepcopy(good);bad['file_source_view']['records'][-1].pop('forward_p0')
        self.assertFalse(view.qualification([bad],1)["passed"])

    def test_profile_pins_and_confounds(self):
        args=SimpleNamespace(concurrent=False,unified_memory=True,spec=False,prefix_cache=False,
            repeat=1,plan='quick',gpu_telemetry=False,server_arg=[],mmproj='none',think=False,
            require_file_read_workers=False,require_file_random_access=False,
            require_file_read_sort=False,require_mapped_probation=False,require_vmm_pressure=False)
        settings={**{key:'1' for key in view.REQUIRED_ON},**{key:'0' for key in view.REQUIRED_OFF},
            **view.REQUIRED_VALUES,view.ENV:'1'}
        view.validate_profile(args,settings)
        for key in view.REQUIRED_OFF:
            bad=dict(settings);bad[key]='1'
            with self.subTest(key=key),self.assertRaises(ValueError):view.validate_profile(args,bad)
        for key in view.REQUIRED_ON:
            bad=dict(settings);bad[key]='0'
            with self.subTest(key=key),self.assertRaises(ValueError):view.validate_profile(args,bad)
        args.require_file_read_sort=True
        with self.assertRaises(ValueError):view.validate_profile(args,settings)


def prefetch_log(prompt=16, count=2, requested_view=1, requested_prefetch=1, **changes):
    result = []
    for original in log(prompt, count, requested_view).splitlines():
        if view.MARKER in original:
            values = dict(file_prefetch_requested=requested_prefetch,
                file_prefetch_calls=1 if requested_prefetch else 0,
                file_prefetch_ranges=2 if requested_prefetch else 0,
                file_prefetch_bytes=4096 if requested_prefetch else 0,
                file_prefetch_failures=0)
            values.update(changes)
            original += " " + " ".join(f"{key}={value}" for key, value in values.items())
        result.append(original)
    return "\n".join(result)


def prefetch_row(text, prompt=16, count=2, **changes):
    result = dict(name="prefetch-fixture", prompt_tokens=prompt, eval_tokens=count,
        cache_hits=0, file_source_view_prefetch=view.prefetch_records(text))
    result.update(changes)
    return result


class FileSourceViewPrefetchTests(unittest.TestCase):
    def test_strict_setting(self):
        for value, expected in ((None,None),("0",0),("1",1)):
            self.assertEqual(view.prefetch_setting(value), expected)
        for value in ("", "01", "+1", " 1", "1 ", "2", 1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                view.prefetch_setting(value)

    def test_control_pure_view_and_combined_all_quick_shapes(self):
        for prompt, count in ((16,2),(1872,32),(33,128)):
            for pair in ((0,0),(1,0),(1,1)):
                q = view.prefetch_qualification([prefetch_row(
                    prefetch_log(prompt,count,*pair),prompt,count)],*pair)
                self.assertTrue(q["passed"])
                self.assertEqual(q["requested_prefetch"], pair[1])
                for key in ("physical_io_qualified","residency_qualified",
                            "performance_qualified","numerical_quality_qualified"):
                    self.assertFalse(q[key])

    def test_empty_and_every_request_and_exact_requested_types(self):
        good = prefetch_row(prefetch_log())
        for rows, pair in (([],(1,1)),([good,prefetch_row("")],(1,1)),
                           ([good],(True,1)),([good],(1,True)),([good],(0,1))):
            self.assertFalse(view.prefetch_qualification(rows,*pair)["passed"])

    def test_batch_bounds_and_partial_range_accounting(self):
        for changes in (dict(file_prefetch_calls=0),dict(file_prefetch_ranges=0),
            dict(file_prefetch_ranges=65),dict(file_prefetch_calls=3),
            dict(file_prefetch_bytes=4097),dict(file_prefetch_failures=2),
            dict(file_prefetch_requested=0),dict(file_prefetch_requested=2)):
            observed = view.prefetch_records(prefetch_log(**changes))
            self.assertGreater(observed["malformed_records"],0,changes)
        partial = prefetch_log(file_prefetch_bytes=2048,file_prefetch_ranges=1)
        self.assertEqual(view.prefetch_records(partial)["malformed_records"],0)
        self.assertFalse(view.prefetch_qualification([prefetch_row(partial)],1,1)["passed"])

    def test_advisory_failure_is_observed_but_cannot_qualify(self):
        failed = prefetch_log(file_prefetch_failures=1)
        self.assertEqual(view.prefetch_records(failed)["malformed_records"],0)
        self.assertFalse(view.prefetch_qualification([prefetch_row(failed)],1,1)["passed"])

    def test_exact_tasks_can_span_multiple_descriptor_batches(self):
        for tasks, calls in ((64,1),(65,2)):
            text=prefetch_log(file_prefetch_calls=calls,file_prefetch_ranges=tasks).replace(
                "file_read_completed_tasks=2",f"file_read_completed_tasks={tasks}").replace(
                "file_view_calls=2",f"file_view_calls={tasks}")
            self.assertTrue(view.prefetch_qualification([prefetch_row(text)],1,1)["passed"])

    def test_missing_duplicate_unknown_and_noncanonical_fields(self):
        text = prefetch_log()
        for bad in (text.replace(" file_prefetch_bytes=4096","",1),
                    text.replace(" file_prefetch_bytes=4096"," file_prefetch_bytes=4096 file_prefetch_bytes=4096",1),
                    text.replace(" file_prefetch_bytes=4096"," file_prefetch_future=1 file_prefetch_bytes=4096",1)):
            self.assertFalse(view.prefetch_qualification([prefetch_row(bad)],1,1)["passed"])
        for key in view.PREFETCH_NUMBERS:
            for value in ("01","+1","-1","1.0",str(1<<64)):
                with self.subTest(key=key,value=value):
                    self.assertGreater(view.prefetch_records(prefetch_log(**{key:value}))["malformed_records"],0)

    def test_no_prefetch_activity_in_either_phase_cannot_qualify(self):
        lines = prefetch_log().splitlines()
        for is_prefill in (False,True):
            bad = []
            for entry in lines:
                if view.MARKER in entry and (" rows=1 " not in entry) == is_prefill:
                    entry = entry.replace("file_prefetch_calls=1","file_prefetch_calls=0").replace(
                        "file_prefetch_ranges=2","file_prefetch_ranges=0").replace(
                        "file_prefetch_bytes=4096","file_prefetch_bytes=0")
                bad.append(entry)
            self.assertFalse(view.prefetch_qualification([prefetch_row("\n".join(bad))],1,1)["passed"])

    def test_all_hit_frame_allowed_without_claimed_prefetch(self):
        lines = prefetch_log().splitlines()
        for index in (0,42):
            for old, new in (("file_bytes=4096","file_bytes=0"),
                ("file_read_source_bytes=4096","file_read_source_bytes=0"),
                ("file_read_completed_tasks=2","file_read_completed_tasks=0"),
                ("file_view_calls=2","file_view_calls=0"),("file_view_bytes=4096","file_view_bytes=0"),
                ("file_prefetch_calls=1","file_prefetch_calls=0"),
                ("file_prefetch_ranges=2","file_prefetch_ranges=0"),
                ("file_prefetch_bytes=4096","file_prefetch_bytes=0")):
                lines[index] = lines[index].replace(old,new)
        self.assertTrue(view.prefetch_qualification([prefetch_row("\n".join(lines))],1,1)["passed"])

    def test_original_view_fallback_failure_and_postcall_guards_still_apply(self):
        text = prefetch_log()
        for bad in (text.replace("p0=17","p0=18"),
                    text.replace("file_view_fallbacks=0","file_view_fallbacks=1",1),
                    text.replace("outcome=completed","outcome=exception",1),
                    "\n".join(text.splitlines()[1:])):
            self.assertFalse(view.prefetch_qualification([prefetch_row(bad)],1,1)["passed"])
        for changes in (dict(error="failed"),dict(cache_hits=True),dict(cache_hits=1)):
            self.assertFalse(view.prefetch_qualification([prefetch_row(text,**changes)],1,1)["passed"])
        lines = text.splitlines(); lines[42],lines[43]=lines[43],lines[42]
        self.assertFalse(view.prefetch_qualification([prefetch_row("\n".join(lines))],1,1)["passed"])

    def test_decoded_counters_cannot_bypass_parser(self):
        good = prefetch_row(prefetch_log())
        for value in ("1",True,-1,1<<64):
            bad = copy.deepcopy(good)
            bad["file_source_view_prefetch"]["records"][0]["file_prefetch_calls"] = value
            self.assertFalse(view.prefetch_qualification([bad],1,1)["passed"])

    def test_aggregate_unsigned_overflow_cannot_qualify(self):
        size=1<<63
        text=prefetch_log(file_prefetch_bytes=size).replace("file_bytes=4096",f"file_bytes={size}").replace(
            "file_read_source_bytes=4096",f"file_read_source_bytes={size}").replace(
            "file_view_bytes=4096",f"file_view_bytes={size}")
        self.assertEqual(view.prefetch_records(text)["malformed_records"],0)
        self.assertFalse(view.prefetch_qualification([prefetch_row(text)],1,1)["passed"])

    def test_combined_profile_pair_and_pure_view_explicit_off(self):
        args=SimpleNamespace(concurrent=False,unified_memory=True,spec=False,prefix_cache=False,
            repeat=1,plan="quick",gpu_telemetry=False,server_arg=[],mmproj="none",think=False,
            require_file_read_workers=False,require_file_random_access=False,
            require_file_read_sort=False,require_mapped_probation=False,require_vmm_pressure=False,
            require_file_source_view=False)
        settings={**{key:"1" for key in view.REQUIRED_ON},
            **{key:"0" for key in view.REQUIRED_OFF},**view.REQUIRED_VALUES}
        for value in ("0","1"):
            valid = dict(settings,**{view.ENV:value,view.PREFETCH_ENV:value})
            view.validate_profile(args,valid,prefetch=True)
        for source, prefetch in (("0","1"),("1","0"),("1",None)):
            bad = dict(settings,**{view.ENV:source})
            if prefetch is None: bad.pop(view.PREFETCH_ENV)
            else: bad[view.PREFETCH_ENV]=prefetch
            with self.assertRaises(ValueError): view.validate_profile(args,bad,prefetch=True)
        for value in (None,"1"):
            bad=dict(settings,**{view.ENV:"1"})
            if value is None: bad.pop(view.PREFETCH_ENV)
            else: bad[view.PREFETCH_ENV]=value
            with self.assertRaises(ValueError): view.validate_profile(args,bad)
        args.require_file_source_view=True
        with self.assertRaises(ValueError):
            view.validate_profile(args,dict(settings,**{view.ENV:"1",view.PREFETCH_ENV:"1"}),prefetch=True)


if __name__ == '__main__': unittest.main()
