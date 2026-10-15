import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "embeddinggemma2-default-hf-composite-recheck.py"
spec = importlib.util.spec_from_file_location("default_hf_recheck_tests", PATH)
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)


class SavedDefaultHfTests(unittest.TestCase):
    def oracle(self):
        value = np.zeros(768); value[0] = 1
        return dict(completed=True, passed=True, input_source_hashes_unchanged=True,
            before_hashes={"input": "sha"}, after_hashes={"input": "sha"},
            fresh_unique_vision_inputs=6, fresh_complete_text_cases=7,
            thresholds=dict(min_cosine=.9999, max_absolute_error=.002), pixel_max_absolute_error=2e-6,
            source_hashes={name: "sha" for name in (*tool.reference.SOURCES,
                "eng/validation/embeddinggemma2-default-hf-composite-reference.py")},
            cases=[dict(name=name, total_input_rows=13, embedding=value.tolist(),
                comparisons=[dict(protocols={p: dict(passed=True) for p in ("openai", "ollama")})])
                for name in tool.reference.CASE_NAMES])

    def test_complete_original_oracle(self):
        tool.validate_oracle(self.oracle())

    def test_rejects_failed_incomplete_and_changed_guard(self):
        for change in (dict(passed=False), dict(completed=False), dict(input_source_hashes_unchanged=False),
                dict(after_hashes={"input": "changed"}), dict(before_hashes={}, after_hashes={})):
            value = self.oracle(); value.update(change)
            with self.assertRaises(ValueError): tool.validate_oracle(value)

    def test_rejects_relaxed_limits_missing_source_or_coverage(self):
        for change in (dict(thresholds=dict(min_cosine=.99, max_absolute_error=.02)),
                dict(pixel_max_absolute_error=2e-5), dict(fresh_unique_vision_inputs=2),
                dict(fresh_complete_text_cases=6), dict(source_hashes={})):
            value = self.oracle(); value.update(change)
            with self.assertRaises(ValueError): tool.validate_oracle(value)

    def test_rejects_case_omission_duplicate_and_failed_protocol(self):
        value = self.oracle(); value["cases"].pop()
        with self.assertRaises(ValueError): tool.validate_oracle(value)
        value = self.oracle(); value["cases"][1]["name"] = "text"
        with self.assertRaises(ValueError): tool.validate_oracle(value)
        value = self.oracle(); value["cases"][0]["comparisons"][0]["protocols"]["ollama"]["passed"] = False
        with self.assertRaises(ValueError): tool.validate_oracle(value)

    def test_rejects_nonfinite_nonunit_and_wrong_dimension_vectors(self):
        for embedding in ([float("nan")] * 768, [0.] * 768, [1.]):
            value = self.oracle(); value["cases"][0]["embedding"] = embedding
            with self.assertRaises(ValueError): tool.validate_oracle(value)

    def test_rejects_invalid_token_count(self):
        for count in (0, -1, 13.0, True):
            value = self.oracle(); value["cases"][0]["total_input_rows"] = count
            with self.assertRaises(ValueError): tool.validate_oracle(value)

    def http(self):
        return dict(media=[dict(path="red.png", sha256="pixel")], manifest=dict(sha256="manifest", path="case.json"),
            cases=[dict(name="red", parts=[dict(kind="text", value="query"), dict(kind="image", value="red.png")])])

    def test_inputs_allow_new_report_path_but_preserve_actual_sequence(self):
        left = self.http(); right = copy.deepcopy(left); right["manifest"]["path"] = "new-case.json"
        tool.matching_http_inputs(left, right)
        right["cases"][0]["parts"].reverse()
        with self.assertRaises(ValueError): tool.matching_http_inputs(left, right)

    def test_inputs_reject_changed_media_or_manifest(self):
        left = self.http()
        for location in ("media", "manifest"):
            right = copy.deepcopy(left)
            if location == "media": right[location][0]["sha256"] = "changed"
            else: right[location]["sha256"] = "changed"
            with self.assertRaises(ValueError): tool.matching_http_inputs(left, right)

    def test_reused_proof_rejects_changed_preprocessing_assembly(self):
        report = dict(completed=True, passed=True, server_exit_code=0, require_embedding_default=True,
            audio_log_policy="HuggingFaceAddFloor", model=dict(sha256="m"), projector=dict(sha256="p"),
            binary_artifacts=[dict(path="TensorSharp.Models.dll", sha256="changed"),
                dict(path="TensorSharp.Runtime.dll", sha256="runtime")],
            environment_audit=dict(require_embedding_default=True, resolved_server_environment=True,
                scope="direct host child environment", audio_log_policy="HuggingFaceAddFloor",
                optimization_flags={k: None for k in tool.reference.environment.OPTIMIZATION_FLAGS},
                optimization_flag_sources={k: None for k in tool.reference.environment.OPTIMIZATION_FLAGS}))
        with self.assertRaises(ValueError): tool.reference.validate_http(report,
            dict(models_assembly_sha256="original", runtime_assembly_sha256="runtime"), "m", "p")

    def test_resized_proof_accepts_nonidentity_canvas_and_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            rgba = np.full((2, 3, 4), 255, dtype=np.uint8)
            pixels = np.zeros((3, 48, 48), dtype="<f4")
            rgba.tofile(parent / "rgba"); pixels.tofile(parent / "pixels")
            image = dict(source_width=3, source_height=2, processed_width=48, processed_height=48,
                soft_tokens=1, rgba="rgba", pixels="pixels", rgba_sha256=tool.sha(parent / "rgba"),
                pixels_sha256=tool.sha(parent / "pixels"))
            proof = dict(images=[image], videos=[])
            tool.check_proof_buffers(proof, parent)
            (parent / "pixels").write_bytes(b"changed")
            with self.assertRaises(ValueError): tool.check_proof_buffers(proof, parent)

    def test_resized_proof_rejects_nonfinite_buffer_even_with_current_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            np.full((2, 3, 4), 255, dtype=np.uint8).tofile(parent / "rgba")
            np.full((3, 48, 48), np.nan, dtype="<f4").tofile(parent / "pixels")
            image = dict(source_width=3, source_height=2, processed_width=48, processed_height=48,
                soft_tokens=1, rgba="rgba", pixels="pixels", rgba_sha256=tool.sha(parent / "rgba"),
                pixels_sha256=tool.sha(parent / "pixels"))
            with self.assertRaises(ValueError): tool.check_proof_buffers(dict(images=[image], videos=[]), parent)


if __name__ == "__main__":
    unittest.main()
