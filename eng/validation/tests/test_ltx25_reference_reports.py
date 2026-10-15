#!/usr/bin/env python3
"""Fail-closed matching of reference-conditioned benchmark reports."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COMPARE = ROOT / "eng/validation/ltx25-compare-generation.py"


class ReferenceReports(unittest.TestCase):
    def setUp(self):
        (ROOT / "artifacts").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="ltx-reference-report-test-", dir=ROOT / "artifacts")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "reference.png"
        self.image.write_bytes(b"synthetic image identity; comparator does not decode this fixture")
        digest = hashlib.sha256(self.image.read_bytes()).hexdigest()
        reference = {"path": str(self.image), "sha256": digest, "frame_index": 0, "strength": 1}
        execution = {"repeat": 0, "generation_ms": 1}
        self.native = {"completed": True, "prompt": "test", "settings": {"Width": 2, "Height": 2, "Frames": 2, "Fps": 24, "Seed": 42},
                       "sigmas": [1, 0], "reference_inputs": [reference], "model_factory_ms": 1, "executions": [execution]}
        self.baseline = {"completed": True, "prompt": "test", "settings": {"width": 2, "height": 2, "frames": 2, "fps": 24, "seed": 42, "sigmas": [1, 0]},
                         "reference_inputs": [dict(reference)], "model_factory_ms": 1, "executions": [execution]}
        pixels = np.linspace(0, 1, 24, dtype=np.float32)
        pixels.tofile(self.root / "native.json.0.rgb.f32")
        (pixels * 255).astype(np.uint8).tofile(self.root / "baseline.json.0.rgb.u8")
        audio = np.linspace(-0.1, 0.1, 16, dtype=np.float32).reshape(8, 2)
        for channel in range(2):
            audio[:, channel].tofile(self.root / f"native.json.0.audio-{channel}.f32")
        audio.tofile(self.root / "baseline.json.0.audio.f32")

    def run_comparison(self):
        for name, report in (("native", self.native), ("baseline", self.baseline)):
            (self.root / (name + ".json")).write_text(json.dumps(report))
        return subprocess.run([sys.executable, str(COMPARE), "--native", str(self.root / "native.json"),
                               "--baseline", str(self.root / "baseline.json"), "--output", str(self.root / "comparison.json")],
                              capture_output=True, text=True)

    def test_matching_contents_at_different_paths_are_accepted(self):
        other = self.root / "same-image.png"
        other.write_bytes(self.image.read_bytes())
        self.baseline["reference_inputs"][0]["path"] = str(other)
        result = self.run_comparison()
        self.assertEqual(0, result.returncode, result.stderr)

    def test_different_frame_positions_are_refused(self):
        self.baseline["reference_inputs"][0]["frame_index"] = 1
        result = self.run_comparison()
        self.assertNotEqual(0, result.returncode)
        self.assertIn("Reference images, target positions or strengths differ", result.stderr)

    def test_changed_source_image_is_refused(self):
        self.image.write_bytes(b"changed after generation")
        result = self.run_comparison()
        self.assertNotEqual(0, result.returncode)
        self.assertIn("Reference image changed since generation", result.stderr)


if __name__ == "__main__":
    unittest.main()
