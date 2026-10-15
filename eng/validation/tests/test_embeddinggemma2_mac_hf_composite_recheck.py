import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "embeddinggemma2-mac-hf-composite-recheck.py"
spec = importlib.util.spec_from_file_location("mac_default_hf_recheck_tests", PATH)
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)


class SavedMacDefaultHfTests(unittest.TestCase):
    def oracle(self):
        value = np.zeros(768); value[0] = 1
        return dict(completed=True, passed=True, input_source_hashes_unchanged=True,
            before_hashes={"input": "sha", "/retained-decoder.json": "a" * 64},
            after_hashes={"input": "sha", "/retained-decoder.json": "a" * 64},
            fresh_unique_vision_inputs=6, fresh_complete_text_cases=7,
            thresholds=dict(min_cosine=.9999, max_absolute_error=.002), pixel_max_absolute_error=2e-6,
            source_hashes=dict(tool.EXPECTED_SOURCE_HASHES),
            codec_reference_kind=tool.CODEC_REFERENCE_KIND,
            ffmpeg_equivalence_qualified=False, independent_codec_implementation=False,
            decoder_proof=dict(path="/retained-decoder.json", sha256="a" * 64),
            mac_video_decoder=dict(proof_sha256="a" * 64, exact_sampled_rgba=True, same_platform_codec=True),
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

    def test_requires_exact_six_reference_digests(self):
        for name in tool.EXPECTED_SOURCE_HASHES:
            value = self.oracle(); value["source_hashes"][name] = "b" * 64
            with self.assertRaises(ValueError): tool.validate_oracle(value)
        value = self.oracle(); value["source_hashes"]["extra.py"] = "c" * 64
        with self.assertRaises(ValueError): tool.validate_oracle(value)

    def test_retains_same_codec_scope_and_failed_ffmpeg_equivalence(self):
        for change in (dict(codec_reference_kind="ffmpeg"), dict(ffmpeg_equivalence_qualified=True),
                dict(independent_codec_implementation=True), dict(codec_reference_kind=None)):
            value = self.oracle(); value.update(change)
            with self.assertRaises(ValueError): tool.validate_oracle(value)

    def test_requires_original_decoder_record_and_exact_sampled_proof(self):
        for key in ("decoder_proof", "mac_video_decoder"):
            value = self.oracle(); del value[key]
            with self.assertRaises(ValueError): tool.validate_oracle(value)
        for key in ("exact_sampled_rgba", "same_platform_codec"):
            value = self.oracle(); value["mac_video_decoder"][key] = False
            with self.assertRaises(ValueError): tool.validate_oracle(value)
        value = self.oracle(); value["decoder_proof"]["sha256"] = "b" * 64
        with self.assertRaises(ValueError): tool.validate_oracle(value)

    def test_single_current_report_only(self):
        tool.validate_report_selection([Path("one.json")])
        for paths in ([], [Path("one.json"), Path("one.json")], [Path("one.json"), Path("two.json")]):
            with self.assertRaises(ValueError): tool.validate_report_selection(paths)

    def test_original_http_mrl_summary_is_preserved(self):
        value = dict(base64_mrl=dict(dimensions=256, cosine=.999999))
        tool.validate_mrl_if_present(value, copy.deepcopy(value))
        tool.validate_mrl_if_present({}, {})
        for current in ({}, dict(base64_mrl=dict(dimensions=128, cosine=1)),
                dict(base64_mrl=dict(dimensions=256, cosine=.99998)),
                dict(base64_mrl=dict(dimensions=256, cosine=float("nan"))),
                dict(base64_mrl=dict(dimensions=True, cosine=1))):
            with self.assertRaises(ValueError): tool.validate_mrl_if_present(value, current)


class OriginalMacDecoderGuardTests(unittest.TestCase):
    def setUp(self):
        path = PATH.with_name("tests") / "test_embeddinggemma2_mac_hf_composite_reference.py"
        spec = importlib.util.spec_from_file_location("mac_decoder_tiny_fixture", path)
        fixture_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = fixture_module; spec.loader.exec_module(fixture_module)
        self.fixture = fixture_module.MacDecoderEvidenceTests("test_complete_same_codec_proof")
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.status = self.fixture.parent / "status.json"
        self.fixture.write(self.status, self.fixture.decoder)
        self.original = {str(path.resolve()): tool.sha(path)
            for path in self.fixture.parent.rglob("*") if path.is_file()}
        self.oracle = dict(before_hashes=self.original,
            decoder_proof=dict(path=str(self.status), sha256=tool.sha(self.status)),
            mac_video_decoder=dict(proof_sha256=tool.sha(self.status), exact_sampled_rgba=True, same_platform_codec=True))
        self.observed = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        digest = tool.sha(path)
        if expected is not None and digest != expected:
            raise ValueError("Pinned parsed bytes differ")
        self.observed[str(path)] = digest
        return json.loads(path.read_bytes())

    def bind(self, path, expected=None):
        if expected is None or tool.sha(path) != expected:
            raise ValueError("Original guard digest cannot be rebased")
        self.observed[str(Path(path).resolve())] = expected

    def check(self):
        tool.validate_original_decoder(self.oracle, self.fixture.proof, self.fixture.proof_path, self.read, self.bind)

    def test_full_tiny_original_decoder_chain(self):
        self.check()
        self.assertEqual(self.original[str((self.fixture.parent / "before.json").resolve())],
            self.observed[str((self.fixture.parent / "before.json").resolve())])

    def test_guard_manifest_must_be_in_original_oracle(self):
        del self.original[str((self.fixture.parent / "before.json").resolve())]
        with self.assertRaises(ValueError): self.check()

    def test_modified_manifest_cannot_be_rehashed_into_new_guard(self):
        for name in ("before.json", "after.json"):
            p = self.fixture.parent / name
            values = json.loads(p.read_bytes()); values["replacement"] = "a" * 64
            self.fixture.write(p, values)
        with self.assertRaises(ValueError): self.check()

    def test_generated_rgba_rehash_cannot_replace_original_digest(self):
        (self.fixture.decoded / "frame-0.rgba").write_bytes(bytes([99, 7, 13, 255]))
        self.fixture.refresh_raw()
        with self.assertRaises(ValueError): self.check()

    def test_validator_returned_digest_cannot_rebase_original(self):
        codec = Path(self.fixture.proof["video_codec_path"])
        with patch.object(tool.reference, "validate_decoder", return_value=({}, {codec: "b" * 64})):
            with self.assertRaises(ValueError): self.check()


class CandidateCodecRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.host = Path(self.temp.name)
        self.models = self.host / "TensorSharp.Models.dll"; self.models.write_bytes(b"models")
        self.managed = self.host / "OpenCvSharp.dll"; self.managed.write_bytes(b"opencv managed")
        self.native = self.host / "runtimes/osx-arm64/native/libOpenCvSharpExtern.dylib"
        self.native.parent.mkdir(parents=True); self.native.write_bytes(b"opencv native")
        self.report = dict(binary_artifacts=[dict(path=str(self.models), sha256=tool.sha(self.models))])
        self.proof = dict(codec_assemblies=[dict(path="/original/OpenCvSharp.dll", sha256=tool.sha(self.managed))],
            loaded_native_codecs=[dict(path="/original/runtimes/osx-arm64/native/libOpenCvSharpExtern.dylib",
                sha256=tool.sha(self.native))], video_codec_path="/original/runtimes/osx-arm64/native/libOpenCvSharpExtern.dylib",
            video_codec_sha256=tool.sha(self.native))
        before = {str(path): tool.sha(path) for path in (self.models, self.managed, self.native)}
        self.guards = [dict(artifacts_unchanged=True, before=before, after=dict(before))]
        self.bound = {}

    def bind(self, path, expected):
        if tool.sha(path) != expected:
            raise ValueError("Candidate codec bytes changed")
        self.bound[str(path)] = expected

    def check(self):
        tool.validate_codec_runtime(self.report, self.proof, self.guards, self.bind)

    def test_codec_assets_bound_to_same_actual_http_snapshot(self):
        self.check(); self.assertEqual(2, len(self.bound))
        self.assertIn(str(self.native), self.bound)

    def test_missing_codec_guard_rejected(self):
        for side in ("before", "after"):
            del self.guards[0][side][str(self.native)]
        with self.assertRaises(ValueError): self.check()

    def test_new_codec_bytes_and_rehashed_runtime_cannot_reuse_original_proof(self):
        self.native.write_bytes(b"different decoder")
        for side in ("before", "after"):
            self.guards[0][side][str(self.native)] = tool.sha(self.native)
        with self.assertRaises(ValueError): self.check()

    def test_codec_guard_cannot_come_from_an_unrelated_snapshot(self):
        codec_only = dict(artifacts_unchanged=True, before=self.guards[0]["before"].copy(), after=self.guards[0]["after"].copy())
        for side in ("before", "after"):
            del self.guards[0][side][str(self.native)]
            del codec_only[side][str(self.models)]
        self.guards.append(codec_only)
        with self.assertRaises(ValueError): self.check()


if __name__ == "__main__":
    unittest.main()
