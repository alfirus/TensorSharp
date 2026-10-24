"""Independent tiny forced-admission/expiry/source/trace references, no GPU."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_gpu_rolling_frequency_replay as rolling

SOURCE = "source-generation-a"


class NaiveReference:
    """Full row histories, recomputed scores and oldest-first key list.

    No circular buffer, incremental Counter, slot ages or production Cache.
    """
    def __init__(self, capacity, window):
        self.capacity, self.window = capacity, window
        self.history, self.resident = {}, []

    def score(self, key):
        source, layer, expert = key
        return sum(expert in row for row in self.history.get((source, layer), [])[-self.window:])

    def visit(self, source, layer, ids):
        self.history.setdefault((source, layer), []).append(frozenset(ids))
        keys = [(source, layer, expert) for expert in ids]
        protected, misses = set(keys), []
        for key in keys:
            if key in self.resident:
                continue
            if len(self.resident) == self.capacity:
                victim = min((old for old in self.resident if old not in protected),
                             key=lambda old: (self.score(old), self.resident.index(old)))
                self.resident.remove(victim)
            self.resident.append(key)
            misses.append(key)
        for key in keys:
            self.resident.remove(key)
            self.resident.append(key)
        return misses


class ListLru:
    """Independent original-LRU trace writer, without rolling scores."""
    def __init__(self, capacity):
        self.capacity, self.resident = capacity, []

    def visit(self, layer, ids):
        keys, missing = [(layer, expert) for expert in ids], []
        protected = set(keys)
        for key in keys:
            if key not in self.resident:
                if len(self.resident) == self.capacity:
                    self.resident.remove(next(old for old in self.resident if old not in protected))
                self.resident.append(key)
                missing.append(key)
        for key in keys:
            self.resident.remove(key)
            self.resident.append(key)
        return missing


def registry():
    return {(SOURCE, 1): 8, (SOURCE, 2): 8, ("source-generation-b", 1): 8}


def metadata():
    groups = []
    for first, layers, hidden, ff in ((1, [1, 2], 2, 2), (3, [3], 2, 4), (4, [4], 4, 4)):
        groups.append(dict(id=first, layers=layers, expert_bytes=hidden * ff * 4 * 3,
            projections=[dict(type=0, shape=shape, expert_bytes=hidden * ff * 4)
                for shape in ([hidden, ff, 8], [hidden, ff, 8], [ff, hidden, 8])]))
    return dict(groups=groups, model_files=[dict(path="never-opened.gguf", bytes=12345, mtime_ns=1)])


def forward(position):
    fields = dict(kind=0, rows=1, p0=position, nout=1, wanth=0, logits=1, reused=0,
        outcome="completed", acquire_ms="0", input_ms="0", compute_ms="1", output_ms="0",
        total_ms="1", scope=rolling.cohort.retention._FORWARD_SCOPE)
    return rolling.cohort.retention._FORWARD + " " + " ".join(f"{key}={value}" for key, value in fields.items())


def trace(pairs=None):
    pairs = ((0, 1), (0, 2), (0, 1), (3, 4), (3, 4), (0, 1)) if pairs is None else pairs
    groups = metadata()["groups"]
    lookup = {layer: group for group in groups for layer in group["layers"]}
    caches = {group["id"]: ListLru(3) for group in groups}
    lines, byte_total = [], 0
    for index, ids in enumerate(pairs):
        for layer in sorted(lookup):
            group = lookup[layer]
            if not index and layer == group["id"]:
                lines.append(f"[HOSTMOE-ROUTE-CREATE] layer={layer} slots=3 experts=8")
            missed = caches[group["id"]].visit(layer, ids)
            size = len(missed) * group["expert_bytes"]
            byte_total += size
            lines.append(f"[HOSTMOE-ROUTE] layer={layer} ids=" + ",".join(map(str, ids)))
            lines.append(f"[HOSTMOE-CACHE-TIMING] layer={layer} rows=1 joint=0 bulk=0 outcome=completed "
                         f"file_bytes={size} scope=nested_host_wall_not_cuda_kernel_or_physical_io")
        lines.append(forward(33 + index))
    return "\n".join(lines), byte_total


def analyse(text=None, data=None, **changes):
    options = dict(capacities={1: 3, 3: 3, 4: 3}, start_position=33, count=6, top_k=2)
    options.update(changes)
    return rolling.analyse(trace()[0] if text is None else text,
                           metadata() if data is None else data, **options)


class RollingPolicyTests(unittest.TestCase):
    def test_strict_constructor_bounds_and_immutable_registry(self):
        for capacity in (0, -1, True, 1.0, rolling.adaptive.MAX_CAPACITY + 1):
            with self.assertRaises(ValueError):
                rolling.RollingFrequency(capacity, registry())
        for window in (0, 1, 31, 128, True, 32.0):
            with self.assertRaises(ValueError):
                rolling.RollingFrequency(3, registry(), window)
        for sources in ({}, {("", 1): 8}, {(SOURCE, True): 8}, {(SOURCE, 1): True},
                        {(SOURCE, 1): 0}, {(SOURCE, 1): 65537}, {SOURCE: 8},
                        {(SOURCE, n): 1 for n in range(rolling.MAX_SOURCES + 1)},
                        {(SOURCE, 1): 40000, (SOURCE, 2): 40000}):
            with self.subTest(sources=str(sources)[:80]), self.assertRaises(ValueError):
                rolling.RollingFrequency(3, sources)
        sources = registry()
        cache = rolling.RollingFrequency(3, sources)
        sources[(SOURCE, 1)] = 100
        self.assertEqual(cache.limits[(SOURCE, 1)], 8)

    def test_invalid_entire_row_changes_no_history_slots_clock_or_stats(self):
        cache = rolling.RollingFrequency(2, registry())
        cache.visit(SOURCE, 1, [0])
        before, stats = cache.snapshot(), dict(cache.stats)
        for args in ((SOURCE, 1, [1, 8]), (SOURCE, 1, [0, 1, 2]),
                     (SOURCE, 1, [1, True]), (SOURCE, 1, []),
                     (SOURCE, 1, [0] * 65), ("unregistered", 1, [0]),
                     (SOURCE, 3, [0]), (SOURCE, True, [0])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                cache.visit(*args)
            self.assertEqual(cache.snapshot(), before)
            self.assertEqual(cache.stats, stats)

    def test_all_original_hits_protected_before_first_miss(self):
        cache = rolling.RollingFrequency(2, registry())
        cache.visit(SOURCE, 1, [0, 1])
        self.assertEqual(cache.visit(SOURCE, 1, [2, 0]), [(SOURCE, 1, 2)])
        self.assertEqual(cache.resident, {(SOURCE, 1, 0), (SOURCE, 1, 2)})

    def test_miss_observations_survive_payload_eviction(self):
        cache = rolling.RollingFrequency(2, registry(), 16)
        for _ in range(8):
            cache.visit(SOURCE, 1, [0])
        cache.visit(SOURCE, 1, [1])
        cache.visit(SOURCE, 1, [2])
        self.assertNotIn((SOURCE, 1, 1), cache.resident)
        self.assertEqual(cache.score((SOURCE, 1, 1)), 1)
        self.assertEqual(cache.visit(SOURCE, 1, [1]), [(SOURCE, 1, 1)])
        self.assertEqual(cache.score((SOURCE, 1, 1)), 2)
        self.assertIn((SOURCE, 1, 0), cache.resident)
        self.assertNotIn((SOURCE, 1, 2), cache.resident)

    def test_high_frequency_cannot_bypass_any_current_union_miss(self):
        cache = rolling.RollingFrequency(2, registry())
        for _ in range(8):
            cache.visit(SOURCE, 1, [0])
        cache.visit(SOURCE, 1, [1])
        self.assertEqual(cache.visit(SOURCE, 1, [2, 3]), [(SOURCE, 1, 2), (SOURCE, 1, 3)])
        self.assertEqual(cache.resident, {(SOURCE, 1, 2), (SOURCE, 1, 3)})
        self.assertEqual(cache.score((SOURCE, 1, 0)), 8)  # history stays, payload must leave

    def test_rolling_frequency_can_lose_and_negative_net_must_be_preserved(self):
        rows = [[0]] * 8 + [[1], [2], [1], [2]]
        cache = rolling.RollingFrequency(2, registry())
        original = ListLru(2)
        misses = sum(len(cache.visit(SOURCE, 1, ids)) for ids in rows)
        baseline = sum(len(original.visit(1, ids)) for ids in rows)
        self.assertEqual((baseline, misses), (3, 5))
        self.assertLess(baseline - misses, 0)

    def test_empty_precedes_valid_zero_score_and_ties_use_original_lru(self):
        cache = rolling.RollingFrequency(3, registry(), 16)
        cache.visit(SOURCE, 1, [0])
        for _ in range(16):
            cache.visit(SOURCE, 1, [1])
        self.assertEqual(cache.score((SOURCE, 1, 0)), 0)
        cache.visit(SOURCE, 1, [2])
        self.assertIn((SOURCE, 1, 0), cache.resident)  # empty used before score0
        equal = rolling.RollingFrequency(2, registry())
        equal.visit(SOURCE, 1, [0]); equal.visit(SOURCE, 1, [1])
        equal.visit(SOURCE, 1, [2])
        self.assertNotIn((SOURCE, 1, 0), equal.resident)
        self.assertIn((SOURCE, 1, 1), equal.resident)

    def test_exact_equal_age_tie_prefers_increasing_slot(self):
        cache = rolling.RollingFrequency(2, registry())
        cache.visit(SOURCE, 1, [0, 1])
        cache.ages[:] = [7, 7]  # directly exercise native strict-less tie
        cache.visit(SOURCE, 1, [2])
        self.assertEqual(cache.slots[0], (SOURCE, 1, 2))
        self.assertEqual(cache.slots[1], (SOURCE, 1, 1))

    def test_duplicates_admit_and_observe_once_but_touch_original_occurrences(self):
        cache = rolling.RollingFrequency(3, registry())
        self.assertEqual(cache.visit(SOURCE, 1, [0, 1, 0]), [(SOURCE, 1, 0), (SOURCE, 1, 1)])
        self.assertEqual((cache.score((SOURCE, 1, 0)), cache.score((SOURCE, 1, 1))), (1, 1))
        self.assertEqual(cache.clock, 3)
        self.assertGreater(cache.ages[0], cache.ages[1])
        self.assertEqual(cache.stats["distinct_observations"], 2)
        self.assertEqual(cache.stats["resident_hits"], 0)

    def test_current_row_is_included_and_oldest_expires_at_exact_boundary(self):
        for window in rolling.WINDOWS:
            cache = rolling.RollingFrequency(2, registry(), window)
            cache.visit(SOURCE, 1, [0])
            self.assertEqual(cache.score((SOURCE, 1, 0)), 1)
            for _ in range(window - 1):
                cache.visit(SOURCE, 1, [1])
            self.assertEqual(cache.score((SOURCE, 1, 0)), 1)
            self.assertEqual(cache.cursors[(SOURCE, 1)], 0)
            cache.visit(SOURCE, 1, [1])
            self.assertEqual(cache.score((SOURCE, 1, 0)), 0)
            self.assertEqual(cache.score((SOURCE, 1, 1)), window)
            self.assertEqual(cache.cursors[(SOURCE, 1)], 1)
            self.assertNotIn(0, cache.counts[(SOURCE, 1)])

    def test_other_layers_and_generations_do_not_age_or_alias(self):
        cache = rolling.RollingFrequency(3, registry(), 16)
        cache.visit(SOURCE, 1, [0])
        for _ in range(40):
            cache.visit(SOURCE, 2, [1])
        cache.visit("source-generation-b", 1, [0])
        self.assertEqual(cache.score((SOURCE, 1, 0)), 1)
        self.assertEqual(cache.score((SOURCE, 2, 1)), 16)
        self.assertEqual(cache.score(("source-generation-b", 1, 0)), 1)
        self.assertEqual(cache.observations[(SOURCE, 1)], 1)
        self.assertEqual(cache.observations[(SOURCE, 2)], 40)

    def test_many_ring_wraps_exact_scores_not_epoch_halving_or_saturation(self):
        for window in rolling.WINDOWS:
            cache = rolling.RollingFrequency(2, registry(), window)
            for _ in range(3 * window + 7):
                cache.visit(SOURCE, 1, [0])
            self.assertEqual(cache.score((SOURCE, 1, 0)), window)
            for _ in range(window):
                cache.visit(SOURCE, 1, [1])
            self.assertEqual(cache.score((SOURCE, 1, 0)), 0)
            self.assertEqual(cache.score((SOURCE, 1, 1)), window)
            self.assertEqual(cache.stats["peak_score"], window)
            self.assertEqual(cache.ring_key_references, window)
            self.assertEqual(cache.active_counter_keys, 1)

    def test_independent_reference_matches_deterministic_multi_source_trace(self):
        identities = tuple(registry())
        for window in rolling.WINDOWS:
            for capacity in (2, 3, 5):
                cache, reference = rolling.RollingFrequency(capacity, registry(), window), NaiveReference(capacity, window)
                for step in range(160):
                    source, layer = identities[(step // 3) % len(identities)]
                    ids = [step % 7, (step + 3) % 7]
                    if step % 4 == 0:
                        ids.append(ids[0])
                    args = source, layer, ids
                    self.assertEqual(cache.visit(*args), reference.visit(*args))
                    self.assertEqual(cache.resident, set(reference.resident))
                    order = [cache.slots[i] for i in sorted(range(capacity), key=lambda i: cache.ages[i])
                             if cache.slots[i] is not None]
                    self.assertEqual(order, reference.resident)
                    for identity, experts in registry().items():
                        for expert in range(experts):
                            self.assertEqual(cache.score((*identity, expert)), reference.score((*identity, expert)))
                    self.assertLessEqual(cache.active_counter_keys, sum(registry().values()))
                    self.assertLessEqual(cache.ring_key_references, len(registry()) * window * 2)
                    self.assertLessEqual(len(cache.resident), capacity)

    def test_independent_exhaustive_tiny_rows(self):
        choices = [(n,) for n in range(3)] + list(itertools.combinations(range(3), 2))
        for capacity in (2, 3):
            for sequence in itertools.product(choices, repeat=4):
                cache, reference = rolling.RollingFrequency(capacity, {(SOURCE, 1): 3}), NaiveReference(capacity, 32)
                for ids in sequence:
                    self.assertEqual(cache.visit(SOURCE, 1, ids), reference.visit(SOURCE, 1, ids))
                    self.assertEqual(cache.resident, set(reference.resident))

    def test_future_suffix_cannot_change_prefix(self):
        prefix = [[0, 1], [0, 2], [3, 0]] * 12
        observed = []
        for suffix in ([[6, 7]] * 20, [[0, 1]] * 20):
            cache, snapshots = rolling.RollingFrequency(3, registry()), []
            for ids in prefix + suffix:
                cache.visit(SOURCE, 1, ids)
                snapshots.append(cache.snapshot())
            observed.append(snapshots[:len(prefix)])
        self.assertEqual(observed[0], observed[1])


class TraceAndCliTests(unittest.TestCase):
    def test_independent_cold_lru_every_layer_bytes_and_primary_are_exact(self):
        text, expected = trace()
        result = analyse(text)
        self.assertEqual(result["baseline"]["layer_timer_checks"], 24)
        self.assertEqual(result["policies"]["lru"]["logical_source_bytes"], expected)
        self.assertEqual(set(result["rolling"]), {"rolling_frequency_32"})
        self.assertEqual(result["primary_window"], 32)
        self.assertTrue(result["independent_lru_matched_every_row"])
        self.assertTrue(result["history_includes_current_row"])
        self.assertTrue(result["duplicates_count_once_per_row"])
        for flag in ("primary_uses_future", "cross_topic_history_carried", "windows_selected_from_results",
                     "native_implementation_qualified", "metadata_physical_admission_qualified",
                     "physical_io_qualified", "performance_qualified", "numerical_quality_qualified"):
            self.assertFalse(result[flag])
        policy = result["policies"]["rolling_frequency_32"]
        self.assertEqual(sum(policy["per_group"].values()), policy["logical_source_bytes"])
        self.assertEqual(sum(policy["per_position"].values()), policy["logical_source_bytes"])

    def test_diagnostic_windows_never_select_primary_or_hide_negative_results(self):
        result = analyse(diagnostic_windows=True)
        self.assertEqual(set(result["rolling"]), {"rolling_frequency_16", "rolling_frequency_32", "rolling_frequency_64"})
        self.assertEqual(result["primary_policy"], "rolling_frequency_32")
        self.assertFalse(result["windows_selected_from_results"])
        for name, item in result["rolling"].items():
            self.assertEqual(item["role"], "predeclared_primary" if name.endswith("_32") else "diagnostic_only_no_policy_selection")
            self.assertEqual(item["net_logical_bytes_saved"], result["policies"]["lru"]["logical_source_bytes"] - result["policies"][name]["logical_source_bytes"])
        with self.assertRaises(ValueError):
            analyse(diagnostic_windows=1)

    def test_each_cold_topic_resets_history_and_cannot_train_the_next(self):
        before = analyse()
        analyse(trace(((6, 7),) * 6)[0])
        after = analyse()
        self.assertEqual(before, after)
        self.assertTrue(all(rows == 6 for rows in after["rolling"]["rolling_frequency_32"]["layer_rows"].values()))

    def test_metadata_source_generation_is_qualified_and_does_not_alias(self):
        first = analyse()
        other = metadata(); other["model_files"][0]["mtime_ns"] += 1
        second = analyse(data=other)
        self.assertNotEqual(first["baseline"]["source_identity"], second["baseline"]["source_identity"])
        self.assertEqual(first["policies"], second["policies"])

    def test_equal_and_opposite_byte_errors_refuse_before_scoring(self):
        text = trace()[0].replace("file_bytes=96", "file_bytes=95", 1)
        text = text.replace("layer=2 rows=1 joint=0 bulk=0 outcome=completed file_bytes=96",
                            "layer=2 rows=1 joint=0 bulk=0 outcome=completed file_bytes=97", 1)
        with self.assertRaisesRegex(ValueError, "miss bytes"):
            analyse(text)

    def test_missing_seed_later_resize_or_interleaved_rows_fail_closed(self):
        text = trace()[0]
        bads = [text.replace("[HOSTMOE-ROUTE-CREATE] layer=1 slots=3 experts=8\n", "", 1),
                text.replace("slots=3 experts=8", "slots=4 experts=8", 1),
                text.replace("p0=34 ", "p0=33 ", 1), text.replace("outcome=completed", "outcome=failed", 1),
                text.replace("ids=0,1", "ids=0,0", 1), text + "\n[HOSTMOE-ROUTE] layer=1 ids=0,1"]
        marker = "[HOSTMOE-ROUTE] layer=1 ids=0,2"
        for event in ("[HOSTMOE-VMM-PRESSURE] profile=1", "[HOSTMOE-PREFETCH] issued=1",
                      "[HOSTMOE-REBALANCE] layer=1", "[HOSTMOE-ROUTE-CREATE] layer=1 slots=3 experts=8"):
            bads.append(text.replace(marker, event + "\n" + marker, 1))
        for bad in bads:
            with self.subTest(bad=bad[:80]), self.assertRaises(ValueError):
                analyse(bad)

    def test_v44_profiles_bound_complete_file_and_exact_independent_cold_windows(self):
        self.assertEqual(rolling.MAX_TRACE_BYTES, 64 << 20)
        self.assertLess(48582958, rolling.MAX_TRACE_BYTES)
        a, b = rolling.PROFILES["v44"]["windows"]
        self.assertEqual((a["begin_line"], a["end_line"], b["begin_line"], b["end_line"]), (637, 33675, 33786, 66824))
        self.assertEqual((a["count"], b["count"]), (384, 384))
        self.assertEqual(a["capacities"], {3: 291, 4: 729, 11: 85})
        self.assertEqual(b["capacities"], {3: 283, 4: 708, 11: 83})
        self.assertEqual((a["expected_misses"], b["expected_misses"]), (58824, 67147))
        self.assertEqual((a["expected_bytes"], b["expected_bytes"]), (408902172672, 466874728448))
        self.assertEqual(b["role"], "second_topic_fixed_policy_holdout")
        self.assertEqual(rolling.PRIMARY_WINDOW, 32)

    def test_cli_rejects_dependency_drift_before_input_loading(self):
        with mock.patch.dict(rolling.DEPENDENCIES, {"glm_gpu_adaptive_replay.py": "0" * 64}), \
             mock.patch.object(rolling.cohort, "_read_bounded") as read:
            with self.assertRaisesRegex(ValueError, "dependency"):
                rolling.main(["--profile", "v27", "--trace", "never-read.log", "--metadata", "never-read.json", "--output", "never-write.json"])
            read.assert_not_called()

    def test_cli_rejects_unpinned_trace_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "trace.log").write_text(trace()[0], encoding="utf-8")
            (path / "metadata.json").write_text(json.dumps(metadata()), encoding="utf-8")
            with self.assertRaises(ValueError):
                rolling.main(["--profile", "v44", "--trace", str(path / "trace.log"), "--metadata", str(path / "metadata.json"), "--output", str(path / "out.json")])
            self.assertFalse((path / "out.json").exists())

    def test_cli_both_topics_keep_fixed_primary_and_pinned_read_bound(self):
        specs = rolling.PROFILES["v44"]["windows"]
        def fake_analysis(text, data, **options):
            index = 0 if options["begin_line"] == 637 else 1
            return dict(baseline=dict(gpu_misses=specs[index]["expected_misses"], logical_source_bytes=specs[index]["expected_bytes"]),
                primary_window=32, rolling={"rolling_frequency_32": dict(net_logical_bytes_saved=-100),
                    "rolling_frequency_16": dict(net_logical_bytes_saved=100)}, windows_selected_from_results=False)
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(rolling.cohort, "_read_bounded", side_effect=[b"not-parsed", b"{}"]) as read, \
             mock.patch.object(rolling, "analyse", side_effect=fake_analysis) as analysis, \
             contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "out.json"
            rolling.main(["--profile", "v44", "--trace", "saved.log", "--metadata", "saved.json", "--output", str(output), "--diagnostic-windows"])
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["primary_window"], 32)
            self.assertFalse(result["policy_selected_from_results"])
            self.assertEqual(analysis.call_count, 2)
            self.assertEqual(read.call_args_list[0].args[1:], (64 << 20, rolling.PROFILES["v44"]["trace_sha256"]))
            self.assertEqual([item["name"] for item in result["chronological_windows"]], ["decode_a", "decode_b"])
            self.assertEqual(result["chronological_windows"][1]["topic_role"], "second_topic_fixed_policy_holdout")
            self.assertEqual(result["chronological_windows"][0]["rolling"]["rolling_frequency_32"]["net_logical_bytes_saved"], -100)
            self.assertFalse(result["provenance"]["checkpoint_opened"])


if __name__ == "__main__":
    unittest.main()
