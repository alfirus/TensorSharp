import importlib.util
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("glm_vmm_weights", Path(__file__).parents[1] / "glm_vmm_weights.py")
vmm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vmm)


class VmmProfileTests(unittest.TestCase):
    def args(self, **changes):
        values = dict(concurrent=False, unified_memory=True, spec=False, prefix_cache=False,
                      repeat=1, plan="quick", gpu_telemetry=False, server_arg=[], mmproj="none",
                      require_file_read_workers=False, require_file_random_access=False,
                      require_mapped_probation=False)
        return SimpleNamespace(**(values | changes))

    def settings(self):
        return ({key: "1" for key in vmm.REQUIRED_ON}
                | {key: "0" for key in vmm.REQUIRED_OFF} | vmm.REQUIRED_VALUES)

    def test_explicit_opt_in_and_malformed_values(self):
        self.assertFalse(vmm.enabled(None))
        self.assertFalse(vmm.enabled("0"))
        self.assertTrue(vmm.enabled("1"))
        for value in ("", "01", "true", "+1", "1 ", 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                vmm.enabled(value)

    def test_first_profile_accepts_only_explicit_isolated_requirements(self):
        vmm.validate_profile(self.args(), self.settings())
        for key in self.settings():
            with self.subTest(key=key):
                omitted = self.settings(); del omitted[key]
                with self.assertRaises(ValueError):
                    vmm.validate_profile(self.args(), omitted)
                changed = self.settings(); changed[key] = "invalid"
                with self.assertRaises(ValueError):
                    vmm.validate_profile(self.args(), changed)

    def test_other_runs_and_untracked_overrides_refused(self):
        for changes in (dict(concurrent=True), dict(unified_memory=False), dict(spec=True),
                        dict(prefix_cache=True), dict(repeat=2), dict(repeat=0), dict(plan="default"),
                        dict(gpu_telemetry=True), dict(server_arg=["--context=4096"]),
                        dict(mmproj="vision.gguf"), dict(require_file_read_workers=True),
                        dict(require_file_random_access=True), dict(require_mapped_probation=True)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                vmm.validate_profile(self.args(**changes), self.settings())


class VmmActivityTests(unittest.TestCase):
    @staticmethod
    def timing(layer=3, rows=1, **changes):
        event = dict(layer=layer, rows=rows, outcome="completed", vmm_profile=1, vmm_capacity=300,
                     vmm_physical_bytes=6000, vmm_pending_bytes=0, vmm_quarantine_bytes=0, vmm_certificate=1)
        return vmm.TIMING_TAG + " " + " ".join(f"{key}={value}" for key, value in (event | changes).items())

    @staticmethod
    def reshape(phase=1, **changes):
        event = dict(profile=1, phase=phase, outcome="completed", refunded_bytes=3000, grown_bytes=2000,
                     retained_hot=30, moved_slots=5, discarded_hot=10, budget=20000,
                     owned_before=16000, owned_after=15000, pending_bytes=0, quarantine_bytes=0,
                     certificate=1, scope=vmm.SCOPE)
        for group, values in ((3, (30, 20, 6000, 4000)), (4, (20, 30, 4000, 6000)),
                              (11, (20, 10, 3000, 2000))):
            for field, value in zip(("cap_before", "cap_after", "bytes_before", "bytes_after"), values):
                event[f"g{group}_{field}"] = value
        return vmm.RESHAPE_TAG + " " + " ".join(f"{key}={value}" for key, value in (event | changes).items())

    @staticmethod
    def pressure(mutation=1, **changes):
        event = dict(profile=1, mutation=mutation, outcome="completed", target=14500,
                     owned_before=16000, owned_after=14000, wide_refunded_bytes=1000,
                     weight_refunded_bytes=1000, retained_hot=25, moved_slots=4, discarded_hot=5,
                     active_owner_excluded=1, g3_cap_before=30, g3_cap_after=20,
                     g4_cap_before=20, g4_cap_after=20, g11_cap_before=0, g11_cap_after=0,
                     pending_bytes=0, quarantine_bytes=0, certificate=1, scope=vmm.PRESSURE_SCOPE)
        return vmm.PRESSURE_TAG + " " + " ".join(f"{key}={value}" for key, value in (event | changes).items())

    @classmethod
    def log(cls, phase=1, include_reshape=True, prompt=33, decode=2):
        widths = [512] * (prompt // 512) + ([prompt % 512] if prompt % 512 else []) + [1] * decode
        lines = [cls.timing(layer, rows) for layer in sorted(vmm.LAYERS) for rows in widths]
        if include_reshape:
            lines.append(cls.reshape(phase))
        return "\n".join(lines)

    @staticmethod
    def row(log):
        return dict(name="fixture", prompt_tokens=33, eval_tokens=2, cache_hits=0, vmm_weights=vmm.records(log))

    def test_all_tokens_layers_and_checked_deltas_qualify_without_invented_credit(self):
        q = vmm.qualification([self.row(self.log()), self.row(self.log(2))])
        self.assertTrue(q["passed"])
        self.assertEqual(q["reshape_events"], 2)
        self.assertTrue(all(row["decode_calls"] == 84 and row["prefill_calls"] == 42
                            and row["newly_granted_bytes"] == 0 for row in q["requests"]))
        self.assertFalse(q["physical_free_qualified"])

    def test_no_startup_or_requested_flag_credit_and_every_request_must_use_vmm(self):
        for rows in ([], [self.row("")], [self.row(self.log(include_reshape=False))],
                     [self.row(self.log()), self.row("")]):
            self.assertFalse(vmm.qualification(rows)["passed"])
        # A request can need no reshape when the other request performed one.
        self.assertTrue(vmm.qualification([self.row(self.log()), self.row(self.log(include_reshape=False))])["passed"])

    def test_missing_duplicate_bad_number_and_unknown_vmm_fields_refuse(self):
        invalid = [self.timing(vmm_profile=0), self.timing(vmm_pending_bytes=1),
                   self.timing(vmm_quarantine_bytes=1), self.timing(vmm_certificate=0),
                   self.timing(vmm_capacity=0), self.timing(vmm_physical_bytes=0),
                   self.timing(rows=513), self.timing(layer=2), self.timing(outcome="exception"),
                   self.timing() + " vmm_certificate=1", self.timing() + " vmm_unknown=1"]
        for key in vmm.TIMING_NUMBERS:
            for bad in ("-1", "+1", "01", "1.0", str(2**64)):
                invalid.append(self.timing(**{key: bad}))
            invalid.append(self.timing().replace(f"{key}=", f"missing_{key}="))
        for line in invalid:
            with self.subTest(line=line):
                row = self.row(self.log() + "\n" + line)
                self.assertEqual(row["vmm_weights"]["malformed_records"], 1)
                self.assertFalse(vmm.qualification([row])["passed"])

    def test_reshape_bytes_epochs_and_hot_metadata_must_reconcile(self):
        changes = (dict(profile=0), dict(phase=0), dict(outcome="failed"), dict(scope="total_vram"),
                   dict(pending_bytes=1), dict(quarantine_bytes=1), dict(certificate=0),
                   dict(refunded_bytes=3001), dict(grown_bytes=2001), dict(owned_after=15001),
                   dict(budget=14999), dict(owned_before=100), dict(retained_hot=0),
                   dict(retained_hot=1000), dict(moved_slots=31), dict(discarded_hot=41),
                   dict(g3_cap_after=31), dict(g11_bytes_after=4000), dict(g4_cap_after=20))
        for change in changes:
            with self.subTest(change=change):
                bad = self.row(self.log(include_reshape=False) + "\n" + self.reshape(**change))
                self.assertEqual(bad["vmm_weights"]["malformed_records"], 1)
                self.assertFalse(vmm.qualification([bad])["passed"])
        for key in vmm.RESHAPE_NUMBERS + ("scope", "outcome"):
            line = self.reshape()
            # Remove one complete key=value field.
            line = " ".join(field for field in line.split() if not field.startswith(key + "="))
            self.assertEqual(vmm.records(line)["malformed_records"], 1)
        self.assertFalse(vmm.qualification([self.row(self.log()), self.row(self.log())])["passed"])
        self.assertFalse(vmm.qualification([self.row(self.log(2)), self.row(self.log(1))])["passed"])

    def test_full_prompt_and_decode_coverage_is_required_at_every_routed_layer(self):
        row = self.row(self.log())
        for field, value in (("prompt_tokens", 34), ("eval_tokens", 3), ("cache_hits", 1), ("error", "failed")):
            self.assertFalse(vmm.qualification([row | {field: value}])["passed"])
        for calls in (row["vmm_weights"]["calls"][1:], row["vmm_weights"]["calls"] * 2):
            changed = copy.deepcopy(row); changed["vmm_weights"]["calls"] = calls
            self.assertFalse(vmm.qualification([changed])["passed"])
        changed = copy.deepcopy(row); changed["vmm_weights"]["calls"][0]["vmm_certificate"] = False
        self.assertFalse(vmm.qualification([changed])["passed"])

    def test_any_group_may_grow_or_donate_with_exact_total_deltas(self):
        events = [
            # Group4 donates to group3; group11 remains unchanged.
            dict(g3_cap_after=40, g3_bytes_after=8000, g4_cap_after=10, g4_bytes_after=2000,
                 g11_cap_after=20, g11_bytes_after=3000, refunded_bytes=2000, grown_bytes=2000,
                 owned_after=16000, retained_hot=40, moved_slots=5, discarded_hot=10),
            # Every group grows from an already-admitted free grant, with no refund.
            dict(g3_cap_after=40, g3_bytes_after=8000, g4_cap_after=30, g4_bytes_after=6000,
                 g11_cap_after=30, g11_bytes_after=4000, refunded_bytes=0, grown_bytes=5000,
                 owned_after=21000, budget=22000, retained_hot=30, moved_slots=0, discarded_hot=0),
            # Rounded plateaus allow fewer group3 slots and more group11 slots
            # without a corresponding physical delta. Net slots can increase
            # while real donor compaction still moves/discards old hot keys.
            dict(g3_cap_after=20, g3_bytes_after=6000, g11_cap_after=21, g11_bytes_after=3000,
                 refunded_bytes=0, grown_bytes=2000, owned_after=18000,
                 retained_hot=30, moved_slots=5, discarded_hot=10),
            dict(retained_hot=0, moved_slots=0, discarded_hot=0),
        ]
        for changes in events:
            with self.subTest(changes=changes):
                row = self.row(self.log(include_reshape=False) + "\n" + self.reshape(**changes))
                self.assertEqual(row["vmm_weights"]["malformed_records"], 0)
                self.assertTrue(vmm.qualification([row])["passed"])

    def test_general_reshape_rejects_direction_conflicts_and_no_physical_growth(self):
        invalid = [dict(g3_cap_after=30), dict(g4_cap_after=19), dict(g11_cap_after=21),
                   dict(g3_cap_after=20, g3_bytes_after=7000),
                   dict(retained_hot=51), dict(moved_slots=21), dict(discarded_hot=21),
                   dict(g4_cap_after=20, g4_bytes_after=4000, grown_bytes=0,
                        owned_after=13000)]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.assertEqual(vmm.records(self.reshape(**changes))["malformed_records"], 1)

    def test_pressure_and_scratch_only_retirement_record_real_owned_bytes(self):
        lines = [self.pressure(), self.pressure(3, weight_refunded_bytes=0, wide_refunded_bytes=2000,
                 g3_cap_after=30, moved_slots=0, discarded_hot=0, active_owner_excluded=0)]
        for line in lines:
            with self.subTest(line=line):
                row = self.row(self.log() + "\n" + line)
                result = vmm.qualification([row], require_pressure=True)
                self.assertTrue(result["passed"])
                self.assertEqual(result["pressure_events"], 1)
                self.assertEqual(result["requests"][0]["pressure_refunded_bytes"], 2000)
                self.assertFalse(result["physical_free_qualified"])

    def test_pressure_fields_and_real_retirement_bounds_fail_closed(self):
        invalid = [dict(profile=0), dict(mutation=0), dict(outcome="failed"), dict(scope=vmm.SCOPE),
                   dict(target=13999), dict(target=16000), dict(owned_after=0),
                   dict(wide_refunded_bytes=1001), dict(wide_refunded_bytes=0, weight_refunded_bytes=0),
                   dict(pending_bytes=1), dict(quarantine_bytes=1), dict(certificate=0),
                   dict(active_owner_excluded=2), dict(g3_cap_after=31), dict(g3_cap_after=0),
                   dict(g11_cap_before=1), dict(g3_cap_before=8193), dict(retained_hot=41),
                   dict(moved_slots=11), dict(discarded_hot=11),
                   dict(g3_cap_after=30, moved_slots=0, discarded_hot=0),
                   dict(g3_cap_before=0, g3_cap_after=0, g4_cap_before=0, g4_cap_after=0)]
        lines = [self.pressure(**changes) for changes in invalid]
        for key in vmm.PRESSURE_NUMBERS:
            for bad in ("-1", "+1", "01", "1.0", str(2**64)):
                lines.append(self.pressure(**{key: bad}))
        for key in vmm.PRESSURE_NUMBERS + ("scope", "outcome"):
            lines.append(" ".join(field for field in self.pressure().split() if not field.startswith(key + "=")))
        lines += [self.pressure() + " target=14500", self.pressure() + " unknown=1"]
        for line in lines:
            with self.subTest(line=line):
                row = self.row(self.log() + "\n" + line)
                self.assertEqual(row["vmm_weights"]["malformed_records"], 1)
                self.assertFalse(vmm.qualification([row])["passed"])

    def test_pressure_must_be_monotonic_across_requests_and_required_per_request(self):
        first = self.row(self.log() + "\n" + self.pressure(2))
        second = self.row(self.log(4) + "\n" + self.pressure(5))
        self.assertTrue(vmm.qualification([first, second], require_pressure=True)["passed"])
        for mutation in (1, 2):
            bad = self.row(self.log(4) + "\n" + self.pressure(mutation))
            self.assertFalse(vmm.qualification([first, bad])["passed"])
        missing = self.row(self.log(4))
        self.assertTrue(vmm.qualification([first, missing])["passed"])
        self.assertFalse(vmm.qualification([first, missing], require_pressure=True)["passed"])
        within = self.row(self.log() + "\n" + self.pressure(2) + "\n" + self.pressure(1))
        self.assertFalse(vmm.qualification([within])["passed"])

    def test_unknown_ambiguous_and_incomplete_native_vmm_tags_are_malformed(self):
        for line in ("[HOSTMOE-VMM-FUTURE] profile=1", "[HOSTMOE-VMM-PRESSURE-V2] profile=1",
                     "[HOSTMOE-VMM-RESHAPE", self.pressure() + " " + vmm.RESHAPE_TAG,
                     self.pressure() + " " + self.timing(), self.pressure() + " " + vmm.PRESSURE_TAG):
            with self.subTest(line=line):
                activity = vmm.records(line)
                self.assertEqual(activity["malformed_records"], 1)
                self.assertFalse(activity["pressures"])
                row = self.row(self.log() + "\n" + line)
                self.assertFalse(vmm.qualification([row])["passed"])

    def test_pressure_metadata_cannot_replace_actual_calls_or_reshape(self):
        self.assertFalse(vmm.qualification([self.row(self.pressure())], require_pressure=True)["passed"])
        row = self.row(self.log(include_reshape=False) + "\n" + self.pressure())
        self.assertFalse(vmm.qualification([row], require_pressure=True)["passed"])
        row = self.row(self.log() + "\n" + self.pressure())
        row["vmm_weights"]["pressures"][0]["mutation"] = True
        self.assertFalse(vmm.qualification([row])["passed"])
        # Previously saved activity without the additive list remains valid
        # only when the new activity requirement is not requested.
        legacy = self.row(self.log()); del legacy["vmm_weights"]["pressures"]
        self.assertTrue(vmm.qualification([legacy])["passed"])
        self.assertFalse(vmm.qualification([legacy], require_pressure=True)["passed"])


if __name__ == "__main__":
    unittest.main()
