import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("glm_causal_route_neighbor",
    Path(__file__).resolve().parents[1] / "glm_causal_route_neighbor.py")
neighbor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neighbor)


def event_line(**changes):
    values = dict(profile=1, outcome="completed", epoch=1, p0=0, rows=1872,
        queries=0, committed=0, matches=0, no_matches=0, compared_anchors=0,
        min_completed_lag=0, history_rows_max=0, declared_host_bytes=65536,
        owned_slab_bytes=65536, held_ticket_bytes=65536, pending=0,
        extra_device_bytes=0, certificate=1, scope=neighbor.SCOPE)
    values.update(changes)
    return neighbor.MARKER + " " + " ".join(f"{key}={value}" for key, value in values.items())


def source_line(layer, rows, **changes):
    values = dict(layer=layer, rows=rows, bulk=int(type(rows) is int and rows > 1), file_bytes=4096,
        file_read_completed_tasks=1, file_read_failed_calls=0, file_read_source_bytes=4096,
        file_view_requested=0, file_view_calls=0, file_view_bytes=0,
        file_view_fallbacks=0, file_view_buffered_bytes=4096,
        file_prefetch_requested=0, file_prefetch_calls=0, file_prefetch_ranges=0,
        file_prefetch_bytes=0, file_prefetch_failures=0, outcome="completed", scope=neighbor._source.SCOPE)
    values.update(changes)
    return neighbor._source.MARKER + " " + " ".join(f"{key}={value}" for key, value in values.items())


def request_log(prompt, count, requested, timer_first=False):
    lines = [source_line(layer, min(512, prompt-start))
             for layer in range(3,45) for start in range(0,prompt,512)]
    epoch = 2 if prompt == 33 else 1
    if requested:
        lines.append(event_line(rows=prompt, epoch=epoch))
    for offset in range(count):
        lines += [source_line(layer,1) for layer in range(3,45)]
        timer = neighbor._source.FORWARD_MARKER + f" kind=0 rows=1 p0={prompt+offset} outcome=completed"
        matched = int(offset >= 9)
        event = event_line(rows=1, epoch=epoch, p0=prompt+offset, queries=42, committed=42,
            matches=matched, no_matches=42-matched, compared_anchors=matched,
            min_completed_lag=matched, history_rows_max=min(offset+1,64))
        if timer_first:
            lines.append(timer)
        if requested:
            lines.append(event)
        if not timer_first:
            lines.append(timer)
    return "\n".join(lines)


def request(name, prompt, count, text):
    return dict(name=name, round=0, prompt_tokens=prompt, eval_tokens=count, cache_hits=0,
                causal_route_neighbor=neighbor.records(text))


class CausalRouteNeighborTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = {flag: [request_log(prompt,count,flag) for _,prompt,count in neighbor.QUICK_REQUESTS]
                    for flag in (0,1)}
        cls.decoded = {flag: [request(name,prompt,count,text) for (name,prompt,count),text
                             in zip(neighbor.QUICK_REQUESTS,cls.logs[flag])] for flag in (0,1)}

    def rows(self, flag=1):
        return copy.deepcopy(self.decoded[flag])

    def test_strict_flag_unset_and_explicit_pair(self):
        for value,expected in ((None,None),("0",0),("1",1)):
            self.assertEqual(neighbor.setting(value),expected)
        for value in ("", "01", "+1", "true", " 1", "2", True, 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                neighbor.setting(value)

    def test_full_fixed_control_candidate_and_nonclaims(self):
        for flag in (0,1):
            q=neighbor.qualification(self.rows(flag),flag)
            self.assertTrue(q["passed"])
            self.assertEqual([r["completed_calls"] for r in q["requests"]], [33*flag,129*flag])
            self.assertEqual([r["committed"] for r in q["requests"]], [1344*flag,5376*flag])
            self.assertEqual(q["declared_host_bytes"],65536 if flag else None)
            for key in ("future_causality_qualified", "victim_policy_qualified", "saved_io_qualified",
                        "physical_io_qualified", "performance_qualified", "numerical_quality_qualified",
                        "all_process_metadata_qualified"):
                self.assertFalse(q[key])

    def test_event_may_bracket_same_postcall_timer_but_not_next_source_call(self):
        rows=[request(name,prompt,count,request_log(prompt,count,1,timer_first=True))
              for name,prompt,count in neighbor.QUICK_REQUESTS]
        self.assertTrue(neighbor.qualification(rows,1)["passed"])
        for key,value in (("decode_calls_before",41),("decode_calls_before",43),
                          ("prefill_calls_before",167),("prefill_calls_before",169)):
            bad=self.rows();bad[0]["causal_route_neighbor"]["records"][1][key]=value
            self.assertFalse(neighbor.qualification(bad,1)["passed"])

    def test_unknown_missing_duplicate_scope_and_trailing_tokens_rejected(self):
        line=event_line()
        for bad in (line+" extra=1",line+" epoch=1",line.replace(" epoch=1",""),
                    line+" trailing",line+" "+neighbor.MARKER,line.replace(neighbor.SCOPE,"other")):
            self.assertGreater(neighbor.records(bad)["malformed_records"],0)

    def test_canonical_u64_not_bool_float_signed_or_overflow(self):
        for value in ("-1","01","+1","1.0","nan","inf",str(2**64),"True"):
            self.assertEqual(neighbor.records(event_line(epoch=value))["malformed_records"],1)

    def test_positive_bounded_exact_host_owner_no_device_or_pending(self):
        for changes in (dict(profile=2),dict(epoch=0),dict(declared_host_bytes=0),
            dict(declared_host_bytes=1048577,owned_slab_bytes=1048577,held_ticket_bytes=1048577),
            dict(owned_slab_bytes=65535),dict(held_ticket_bytes=0),dict(extra_device_bytes=1),
            dict(certificate=0),dict(pending=1),dict(rows=0),dict(rows=2049),dict(p0=2048)):
            self.assertEqual(neighbor.records(event_line(**changes))["malformed_records"],1)

    def test_pp_event_required_even_if_no_queries_and_each_activity_refuses(self):
        rows=self.rows();rows[0]["causal_route_neighbor"]["records"].pop(0)
        self.assertFalse(neighbor.qualification(rows,1)["passed"])
        for key in neighbor.ACTIVITY:
            rows=self.rows();rows[0]["causal_route_neighbor"]["records"][0][key]=1
            self.assertFalse(neighbor.qualification(rows,1)["passed"])

    def test_every_token_queries_commits_and_accounts_all42(self):
        for key,value in (("queries",41),("committed",41),("matches",1),("no_matches",41)):
            rows=self.rows();rows[0]["causal_route_neighbor"]["records"][1][key]=value
            self.assertFalse(neighbor.qualification(rows,1)["passed"])

    def test_actual_match_required_in_each_request_not_only_aggregate(self):
        for request_index in (0,1):
            rows=self.rows()
            for event in rows[request_index]["causal_route_neighbor"]["records"][1:]:
                event.update(matches=0,no_matches=42,min_completed_lag=0,compared_anchors=0)
            q=neighbor.qualification(rows,1)
            self.assertFalse(q["passed"])
            self.assertTrue(q["requests"][1-request_index]["passed"])

    def test_past_lag_boundary56_and_history64_are_inclusive(self):
        rows=self.rows();event=rows[1]["causal_route_neighbor"]["records"][-1]
        event.update(min_completed_lag=56,history_rows_max=64,compared_anchors=2352)
        self.assertTrue(neighbor.qualification(rows,1)["passed"])
        for key,value in (("min_completed_lag",57),("min_completed_lag",0),
                          ("history_rows_max",65),("history_rows_max",8),("compared_anchors",2353),
                          ("compared_anchors",0)):
            bad=copy.deepcopy(rows);bad[1]["causal_route_neighbor"]["records"][-1][key]=value
            self.assertFalse(neighbor.qualification(bad,1)["passed"])

    def test_no_match_requires_zero_lag_but_may_have_compared_anchors(self):
        rows=self.rows();event=rows[0]["causal_route_neighbor"]["records"][11]
        event.update(matches=0,no_matches=42,min_completed_lag=0,compared_anchors=56)
        self.assertTrue(neighbor.qualification(rows,1)["passed"])
        event["min_completed_lag"]=1
        self.assertFalse(neighbor.qualification(rows,1)["passed"])

    def test_no_fabricated_early_history_match_or_uncompleted_anchor_suffix(self):
        for index,changes in ((1,dict(history_rows_max=9)),
            (1,dict(compared_anchors=1)),
            (1,dict(matches=1,no_matches=41,compared_anchors=1,min_completed_lag=1,history_rows_max=9)),
            (9,dict(matches=1,no_matches=41,compared_anchors=1,min_completed_lag=1,history_rows_max=9)),
            (10,dict(compared_anchors=43))):
            rows=self.rows();rows[0]["causal_route_neighbor"]["records"][index].update(changes)
            self.assertFalse(neighbor.qualification(rows,1)["passed"])

    def test_duplicate_missing_reordered_or_wrong_token_positions_refuse(self):
        for mutate in (lambda e:e.pop(4),lambda e:e.append(dict(e[-1])),
                       lambda e:e.reverse(),lambda e:e[2].update(p0=1872),
                       lambda e:e[1].update(rows=2)):
            rows=self.rows();mutate(rows[0]["causal_route_neighbor"]["records"])
            self.assertFalse(neighbor.qualification(rows,1)["passed"])

    def test_epoch_cannot_move_back_within_request_and_host_size_cannot_change(self):
        for changes in (dict(epoch=0),dict(declared_host_bytes=65537,owned_slab_bytes=65537,held_ticket_bytes=65537)):
            rows=self.rows();rows[1]["causal_route_neighbor"]["records"][-1].update(changes)
            self.assertFalse(neighbor.qualification(rows,1)["passed"])
        rows=self.rows();events=rows[0]["causal_route_neighbor"]["records"]
        events[1]["epoch"]=2
        self.assertFalse(neighbor.qualification(rows,1)["passed"])

    def test_next_prefill_must_strictly_reset_epoch_not_carry_prior_topic(self):
        for epoch in (1,2):
            rows=self.rows()
            for event in rows[0]["causal_route_neighbor"]["records"]:
                event["epoch"]=2
            for event in rows[1]["causal_route_neighbor"]["records"]:
                event["epoch"]=epoch
            self.assertFalse(neighbor.qualification(rows,1)["passed"])

    def test_bad_decoded_source_observation_refuses_without_exception(self):
        for value in (None,[],True):
            rows=self.rows();rows[0]["causal_route_neighbor"]["source_activity"]=value
            self.assertFalse(neighbor.qualification(rows,1)["passed"])

    def test_failed_or_off_events_never_qualify(self):
        rows=self.rows();rows[0]["causal_route_neighbor"]["records"][1]["outcome"]="failed"
        self.assertFalse(neighbor.qualification(rows,1)["passed"])
        self.assertFalse(neighbor.qualification(self.rows(),0)["passed"])

    def test_all42_source_layers_exact_bytes_no_fallback(self):
        for before,after in (("layer=3 rows=512","layer=4 rows=512"),
            ("file_read_source_bytes=4096","file_read_source_bytes=4095"),
            ("file_read_failed_calls=0","file_read_failed_calls=1"),
            ("file_view_fallbacks=0","file_view_fallbacks=1"),
            ("file_prefetch_requested=0","file_prefetch_requested=1"),
            ("kind=0 rows=1 p0=1872","kind=0 rows=1 p0=1873")):
            rows=self.rows();rows[0]["causal_route_neighbor"]=neighbor.records(self.logs[1][0].replace(before,after,1))
            self.assertFalse(neighbor.qualification(rows,1)["passed"])

    def test_prefill_layer_major_chronology_not_only_width_counts(self):
        rows=self.rows();events=rows[0]["causal_route_neighbor"]["source_activity"]["records"]
        events[0],events[4]=events[4],events[0]
        self.assertFalse(neighbor.qualification(rows,1)["passed"])

    def test_decoded_types_malformed_or_unrecognized_event_refuse(self):
        for field,value in (("queries",True),("pending","0"),("prefill_calls_before",True),("unknown",1)):
            rows=self.rows();rows[0]["causal_route_neighbor"]["records"][0][field]=value
            self.assertFalse(neighbor.qualification(rows,1)["passed"])
        for value in (True,-1,1,"0"):
            rows=self.rows();rows[0]["causal_route_neighbor"]["malformed_records"]=value
            self.assertFalse(neighbor.qualification(rows,1)["passed"])
        for value in (None,[],True):
            rows=self.rows();rows[0]["causal_route_neighbor"]=value
            self.assertFalse(neighbor.qualification(rows,1)["passed"])

    def test_fixed_request_identity_counts_and_no_error_or_cache_credit(self):
        for key,value in (("error","failed"),("cache_hits",1),("cache_hits",False),
            ("prompt_tokens",1871),("eval_tokens",31),("round",False),("name","startup")):
            rows=self.rows();rows[0][key]=value
            self.assertFalse(neighbor.qualification(rows,1)["passed"])
        rows=self.rows()
        for bad in ([],rows[:1],rows+[rows[1]],list(reversed(rows)),None,[None]):
            self.assertFalse(neighbor.qualification(bad,1)["passed"])
        for flag in (None,True,"1",2):
            self.assertFalse(neighbor.qualification(rows,flag)["passed"])

    def test_exact_profile_requires_explicit_flags_and_unset_trace(self):
        args=SimpleNamespace(concurrent=False,unified_memory=True,spec=False,prefix_cache=False,
            repeat=1,plan="quick",gpu_telemetry=False,server_arg=[],mmproj="none",think=False)
        settings={**{key:"1" for key in neighbor.REQUIRED_ON},
                  **{key:"0" for key in neighbor.REQUIRED_OFF},**neighbor.REQUIRED_VALUES,neighbor.ENV:"1"}
        neighbor.validate_profile(args,settings)
        for key in tuple(settings):
            bad=dict(settings);del bad[key]
            with self.subTest(key=key),self.assertRaises(ValueError):
                neighbor.validate_profile(args,bad)
        for key in ("TS_GLM_LAYER_MAJOR_DEVICE_BATON","TS_HOST_MOE_FILE_SOURCE_VIEW",
                    "TS_HOST_MOE_VMM_WEIGHTS","TS_HOST_MOE_EXPERT_CACHE_LFU"):
            with self.assertRaises(ValueError):
                neighbor.validate_profile(args,dict(settings,**{key:"1"}))
        for key in ("TS_GLM_TRACE","TS_GLM_TRACE_DUMP_DIR"):
            with self.assertRaises(ValueError):
                neighbor.validate_profile(args,settings,{key:"0"})


if __name__=="__main__":
    unittest.main()
