"""Offline evidence policy tests, not native arithmetic/model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("owned_attention_quality", ROOT / "eng/validation/embeddinggemma2-owned-attention-quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class NativeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.library = Path(self.directory.name) / "libTensorSharpCpu.so"
        self.library.write_bytes(b"mock library bytes; never loaded or executed")
        self.log = Path(self.directory.name) / "run.log"
        self.record = dict(backend="CPU", graph="managed", native_math=True, successful_attention_tiles=6,
                           successful_attention_rows=41, managed_attention_tiles=0,
                           library_sha256=hashlib.sha256(self.library.read_bytes()).hexdigest())

    def write(self, **updates):
        self.log.write_text("unrelated ordinary model log\nEMBEDDING_CPU_MATH " + json.dumps(dict(self.record, **updates)) + "\n")

    def test_valid_actual_execution_metadata_accepted(self):
        self.write()
        self.assertEqual(quality.native_evidence(self.log, self.library), self.record)

    def test_fallback_noop_and_mixed_math_are_never_native_passes(self):
        for changes in (dict(native_math=False), dict(successful_attention_tiles=0), dict(successful_attention_rows=0),
                        dict(managed_attention_tiles=1), dict(backend="GGML_CPU"), dict(graph="native-ggml")):
            with self.subTest(changes=changes):
                self.write(**changes)
                with self.assertRaises(ValueError):
                    quality.native_evidence(self.log, self.library)

    def test_different_or_changed_library_rejected(self):
        self.write(library_sha256="0" * 64)
        with self.assertRaises(ValueError):
            quality.native_evidence(self.log, self.library)
        self.write()
        self.library.write_bytes(b"library replaced after inference")
        with self.assertRaises(ValueError):
            quality.native_evidence(self.log, self.library)

    def test_missing_or_duplicate_disposal_record_rejected(self):
        self.log.write_text("ordinary log only\n")
        with self.assertRaises(ValueError):
            quality.native_evidence(self.log, self.library)

        self.write()
        self.log.write_text(self.log.read_text() * 2)
        with self.assertRaises(ValueError):
            quality.native_evidence(self.log, self.library)

    def test_combined_requires_actual_projection_and_no_fallback(self):
        projection = dict(projection_math="owned-native-q8-q16-vnni", projection_library_sha256=self.record["library_sha256"],
                          projection_fallback_reason=None, successful_projection_calls=5, successful_projection_pack_rows=13,
                          successful_projection_groups=6, successful_projection_rows=13,
                          managed_q16_projection_calls=0, q16_float32_fallback_rows=0)
        self.write(**projection)
        quality.native_evidence(self.log, self.library, require_q16=True)
        failures = [dict(projection_math="managed"), dict(projection_library_sha256="0" * 64),
                    dict(projection_fallback_reason="unavailable"), dict(managed_q16_projection_calls=1),
                    dict(q16_float32_fallback_rows=1)]
        failures += [{key: 0} for key in ("successful_projection_calls", "successful_projection_pack_rows",
                                        "successful_projection_groups", "successful_projection_rows")]
        for failure in failures:
            with self.subTest(failure=failure):
                self.write(**dict(projection, **failure))
                with self.assertRaises(ValueError):
                    quality.native_evidence(self.log, self.library, require_q16=True)

    def test_fusion_request_cannot_pass_without_actual_work(self):
        good = dict(fused_gate_gelu_requested=True, successful_fused_gate_gelu_calls=24, successful_fused_gate_gelu_rows=31)
        self.write(**good)
        quality.native_evidence(self.log, self.library, require_fused=True)
        for failure in (dict(fused_gate_gelu_requested=False), dict(successful_fused_gate_gelu_calls=0), dict(successful_fused_gate_gelu_rows=0)):
            with self.subTest(failure=failure):
                self.write(**dict(good, **failure))
                with self.assertRaises(ValueError):
                    quality.native_evidence(self.log, self.library, require_fused=True)


if __name__ == "__main__":
    unittest.main()
