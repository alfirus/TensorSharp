"""Negative input/lifecycle checks for the independent IMAGE-only diagnostic."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("image_reference", Path(__file__).with_name("embeddinggemma2-controlled-image-reference.py"))
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def fixture():
    parts = [dict(kind="image", value="/remote/exact.png")]
    engine = dict(completed=True, server_exit_code=0, benchmarks=dict(red=dict(parts=parts)))
    return dict(completed=True, model=dict(sha256="model"), projector=dict(sha256="projector"),
                inputs=[dict(name="red", parts=parts)], media=[dict(path="/remote/exact.png", sha256="image")],
                engines=dict(tensorsharp=copy.deepcopy(engine), llama=copy.deepcopy(engine)))


class ImageEvidenceTests(unittest.TestCase):
    def test_default_evidence_paths_are_unchanged(self):
        self.assertEqual(tool.evidence_resolver([])("/remote/exact.png"), Path("/remote/exact.png"))

    def test_explicit_prefix_matching_uses_path_boundaries_and_longest_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); local = root / "local"; nested = root / "nested"
            local.mkdir(); nested.mkdir()
            resolve = tool.evidence_resolver([f"/remote={local}", f"/remote/child={nested}"])
            self.assertEqual(resolve("/remote/exact.png"), local / "exact.png")
            self.assertEqual(resolve("/remote/child/exact.png"), nested / "exact.png")
            self.assertEqual(resolve("/remote-other/exact.png"), Path("/remote-other/exact.png"))

    def test_invalid_duplicate_traversing_and_escaping_bindings_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); local = root / "local"; local.mkdir()
            for bindings in ([f"relative={local}"], [f"/remote/../other={local}"],
                             [f"/remote={local}", f"/remote={local}"],
                             [f"/remote={root / 'missing'}"], ["missing-equals"]):
                with self.subTest(bindings=bindings), self.assertRaises(ValueError):
                    tool.evidence_resolver(bindings)
            resolve = tool.evidence_resolver([f"/remote={local}"])
            for recorded in ("relative.png", "/remote/../outside.png"):
                with self.assertRaises(ValueError):
                    resolve(recorded)
            (local / "escape").symlink_to(root)
            with self.assertRaises(ValueError):
                resolve("/remote/escape/input.png")

    def test_relocated_pixel_proof_keeps_original_records_and_every_hash_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rgb = np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3)
            rgba = np.concatenate((rgb, np.full((48, 48, 1), 255, np.uint8)), axis=2)
            tool.Image.fromarray(rgba).save(root / "image.png")
            rgba.tofile(root / "rgba.bin")
            (rgb.transpose(2, 0, 1).astype(np.float32) / np.float32(255)).astype("<f4").tofile(root / "pixels.f32")
            record = dict(image="/remote/image.png", rgba="/remote/rgba.bin", pixels="/remote/pixels.f32",
                          image_sha256=tool.digest(root / "image.png"), rgba_sha256=tool.digest(root / "rgba.bin"),
                          pixels_sha256=tool.digest(root / "pixels.f32"), source_width=48, source_height=48,
                          processed_width=48, processed_height=48, soft_tokens=1)
            original = copy.deepcopy(record)
            class IdentitySizing:
                @staticmethod
                def get_aspect_ratio_preserving_size(height, width, *_):
                    return height, width
            resolve = tool.evidence_resolver([f"/remote={root}"])
            pixels, proof = tool.pixel_proof(record, IdentitySizing, resolve)
            self.assertEqual(pixels.shape, (3, 48, 48))
            self.assertTrue(proof["all_processed_f32_pixels_bit_equal_to_no_resize"])
            self.assertEqual(record, original)
            for name in ("image.png", "rgba.bin", "pixels.f32"):
                path = root / name; data = path.read_bytes(); path.write_bytes(data + b"tampered")
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "evidence changed"):
                    tool.pixel_proof(record, IdentitySizing, resolve)
                path.write_bytes(data)

    def test_reused_oracle_accepts_current_per_assembly_pixel_proofs(self):
        case = dict(preprocessing_proofs=[dict(actual_processed_pixels_sha256="pixels") for _ in range(2)])
        self.assertEqual(tool.reused_pixel_hash(case), "pixels")

    def test_reused_oracle_accepts_immutable_original_single_pixel_proof(self):
        self.assertEqual(tool.reused_pixel_hash(dict(preprocessing_proof=dict(actual_processed_pixels_sha256="pixels"))), "pixels")

    def test_reused_oracle_rejects_absent_empty_or_disagreeing_pixel_proofs(self):
        for case in ({}, dict(preprocessing_proofs=[]), dict(preprocessing_proofs=[{}]),
                     dict(preprocessing_proofs=[dict(actual_processed_pixels_sha256=value) for value in ("a", "b")])):
            with self.subTest(case=case), self.assertRaises(ValueError):
                tool.reused_pixel_hash(case)

    def test_each_actual_assembly_needs_its_own_proof(self):
        first = dict(binary_artifacts=dict(tensorsharp=[dict(path="/first/TensorSharp.Models.dll",sha256="a")]))
        second = dict(binary_artifacts=dict(tensorsharp=[dict(path="/second/TensorSharp.Models.dll",sha256="b")]))
        proofs = [dict(completed=True,models_assembly_sha256="a")]
        tool.preprocessor_for_pair(first,proofs)
        with self.assertRaises(ValueError):
            tool.preprocessor_for_pair(second,proofs)

    def test_two_distinct_observed_assemblies_bind_both_pairs(self):
        proofs = [dict(completed=True,models_assembly_sha256=value) for value in ("a","b")]
        for value in ("a","b"):
            pair = dict(binary_artifacts=dict(tensorsharp=[dict(path="TensorSharp.Models.dll",sha256=value)]))
            self.assertEqual(tool.preprocessor_for_pair(pair,proofs)["models_assembly_sha256"],value)

    def test_incomplete_preprocessor_cannot_bind_pair(self):
        pair = dict(binary_artifacts=dict(tensorsharp=[dict(path="TensorSharp.Models.dll",sha256="a")]))
        with self.assertRaises(ValueError):
            tool.preprocessor_for_pair(pair,[dict(completed=False,models_assembly_sha256="a")])

    def test_exact_completed_image_sequence_accepted(self):
        tool.validate_pair(fixture(), "red", "image", "model", "projector")

    def reject(self, mutate):
        pair = fixture(); mutate(pair)
        with self.assertRaises(ValueError):
            tool.validate_pair(pair, "red", "image", "model", "projector")

    def test_fixture_change_rejected(self):
        self.reject(lambda p: p["media"][0].update(sha256="different"))

    def test_same_basename_different_path_cannot_bind_input(self):
        self.reject(lambda p: p["media"][0].update(path="/unrelated/exact.png"))

    def test_projector_change_rejected(self):
        self.reject(lambda p: p["projector"].update(sha256="different"))

    def test_incomplete_pair_rejected(self):
        self.reject(lambda p: p.update(completed=False))

    def test_bad_server_exit_rejected(self):
        self.reject(lambda p: p["engines"]["llama"].update(server_exit_code=1))

    def test_missing_server_exit_rejected(self):
        self.reject(lambda p: p["engines"]["tensorsharp"].pop("server_exit_code"))

    def test_audio_sequence_not_counted_as_image(self):
        self.reject(lambda p: p["inputs"][0]["parts"][0].update(kind="audio"))

    def test_text_prefix_rejected(self):
        self.reject(lambda p: p["inputs"][0]["parts"].insert(0, dict(kind="text", value="prefix")))

    def test_actual_engine_sequence_change_rejected(self):
        self.reject(lambda p: p["engines"]["llama"]["benchmarks"]["red"].update(parts=[dict(kind="image", value="other.png")]))

    def test_strict_numerical_failure_is_retained(self):
        expected = np.array([1., 0.]); actual = np.array([0., 1.])
        report = tool.compare(expected, actual, 260, 260, .9999, .002)
        self.assertFalse(report["passed"])
        self.assertEqual(report["cosine"], 0)
        self.assertEqual(report["max_absolute_error"], 1)

    def test_wrong_total_rows_rejected(self):
        with self.assertRaises(ValueError):
            tool.compare([1., 0.], [1., 0.], 264, 260, .9999, .002)


if __name__ == "__main__":
    unittest.main()
