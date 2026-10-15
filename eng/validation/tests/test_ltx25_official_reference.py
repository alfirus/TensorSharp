"""Offline identity/scope checks; these tests never load a model runtime."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("ltx25_reference", ROOT / "eng/validation/ltx25-official-denoiser-reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class OfficialReferenceScopeTests(unittest.TestCase):
    def test_accepts_only_explicit_forward_precision_and_bounded_depth(self):
        for precision in ["Float32", "Float64"]:
            for blocks in [1, 4]:
                with self.subTest(precision=precision, blocks=blocks):
                    self.assertEqual(reference.validate_probe(dict(completed=True, schedule="forward", output_kind="velocity",
                                                                  frequency_precision=precision, executed_blocks=blocks)), blocks)

    def test_sampled_latents_and_incomplete_or_ambiguous_metadata_are_rejected(self):
        valid = dict(completed=True, schedule="forward", output_kind="velocity", frequency_precision="Float64", executed_blocks=1)
        for changes in [dict(completed=False), dict(schedule="distilled"), dict(output_kind="sampled latent"),
                        dict(schedule=None), dict(frequency_precision=None), dict(frequency_precision="Float16"),
                        dict(executed_blocks=0), dict(executed_blocks=5), dict(executed_blocks=True)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                reference.validate_probe(dict(valid, **changes))

    def test_full_depth_requires_explicit_residency_override(self):
        full = dict(completed=True, schedule="forward", output_kind="velocity", frequency_precision="Float64", executed_blocks=48)
        with self.assertRaises(ValueError):
            reference.validate_probe(full)
        self.assertEqual(reference.validate_probe(full, 48), 48)
        for limit in [0, 129, True]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                reference.validate_probe(full, limit)


if __name__ == "__main__":
    unittest.main()
