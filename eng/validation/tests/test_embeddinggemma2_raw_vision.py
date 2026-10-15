"""Default-route evidence must not silently accept CPU vision overrides."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("raw_vision", Path(__file__).parents[1] / "embeddinggemma2-compare-raw-vision.py")
RAW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RAW)


class RawVisionDefaultGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.reference = root / "oracle.json"
        self.candidate = root / "candidate.json"
        weights, pixels, output = root / "weights", root / "pixels", root / "output.f32"
        weights.write_bytes(b"model fixture")
        pixels.write_bytes(b"pixel fixture")
        output.write_bytes(struct.pack("<2f", 1, 2))
        self.reference.with_suffix(".f32").write_bytes(output.read_bytes())
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        self.reference.write_text(json.dumps(dict(shape=[1, 2], model_sha256=sha(weights),
            pixels_sha256=sha(pixels), thresholds=dict(min_row_cosine=.9999, max_absolute_error=.01))))
        self.record = dict(rows=1, dimensions=2, completed=True, backend="GGML_CPU",
            projector=str(weights), projector_sha256=sha(weights), pixels=str(pixels), pixels_sha256=sha(pixels),
            output=str(output), output_sha256=sha(output), binary_artifacts=[], require_embedding_default=True,
            embedding_default_guard_version=4,cupti_activity_output=None)

    def compare(self):
        self.candidate.write_text(json.dumps(self.record))
        return RAW.compare(self.reference, self.candidate, False, True)

    def test_current_clean_record_passes(self):
        self.assertTrue(self.compare()["passed"])

    def test_cpu_and_diagnostic_selectors_cannot_claim_default(self):
        for name in ("embedding_linux_blas", "embedding_blas_library", "embedding_cpu_attention",
                "embedding_cpu_profile", "embedding_cpu_attention_profile", "embedding_cpu_attention_simd", "vision_stage", "vision_trace", "vision_profile", "cupti_activity_output"):
            for value in ("1", "0", ""):
                with self.subTest(name=name, value=value):
                    self.record[name] = value
                    with self.assertRaisesRegex(ValueError, "current guard"):
                        self.compare()
            self.record.pop(name)

    def test_old_guard_cannot_prove_cpu_or_profiler_selector_absence(self):
        for version in (None, 1, 2, 3):
            with self.subTest(version=version):
                if version is None:
                    self.record.pop("embedding_default_guard_version", None)
                else:
                    self.record["embedding_default_guard_version"] = version
                with self.assertRaisesRegex(ValueError, "current guard"):
                    self.compare()

    def test_current_guard_requires_explicit_profiler_absence_evidence(self):
        self.record.pop("cupti_activity_output")
        with self.assertRaisesRegex(ValueError,"current guard"):
            self.compare()

    def test_cuda_requires_embedding_precision_constructor(self):
        self.record["backend"] = "GGML_CUDA"
        with self.assertRaisesRegex(ValueError, "validated embedding constructor"):
            self.compare()
        self.record["require_validated_float32_precision"] = True
        self.assertTrue(self.compare()["passed"])


if __name__ == "__main__":
    unittest.main()
