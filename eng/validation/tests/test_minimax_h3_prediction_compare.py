"""Protect the sign/carrier conversion and matched-input evidence gate."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


TOOL = Path(__file__).resolve().parents[1] / "minimax-h3-prediction-compare.py"


class PredictionComparisonTests(unittest.TestCase):
    def run_case(self, *, changed_input=False, wrong_audio_sign=False, wrong_output_hash=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, reference, snapshots = [root / name for name in ("candidate", "reference", "oracle-snapshots")]
            for path in (candidate, reference, snapshots):
                path.mkdir()
            physical = np.array([.5, -2, 3], dtype="<f4")
            audio = np.array([-1, 2.5, -.125], dtype="<f4")
            video = np.array([1.5, -4, .25], dtype="<f4")
            physical_path = reference / "physical.f32"
            physical.tofile(physical_path)
            carrier = -3*physical - 4*audio
            (carrier if not wrong_audio_sign else -carrier).tofile(candidate / "carrier-audio-velocity.f32")
            (-video).tofile(candidate / "video-velocity.f32")
            audio.tofile(snapshots / "raw-audio-output.f32")
            video.tofile(snapshots / "raw-video-output.f32")
            metadata = [{"file": "physical.f32", "id": 1},
                        {"file": "projection-audio.f32", "id": 2, "sources": [
                            {"id": 0, "name": "audio_patch_proj.weight"}, {"id": 1, "name": "physical"}]}]
            for index, item in enumerate(metadata):
                (reference / f"{index}.f32.json").write_text(json.dumps(item))
            identities = [{"file": "physical.f32", "sha256": hashlib.sha256(physical_path.read_bytes()).hexdigest()}]
            oracle = {"completed": True, "actual_blocks": 50, "tf32": False, "inputs": identities}
            digest=hashlib.sha256((candidate / "video-velocity.f32").read_bytes()+
                                  (candidate / "carrier-audio-velocity.f32").read_bytes()).hexdigest()
            probe = {"completed": True, "blocks": 50, "sigma": 1, "inputs": identities, "trace": str(candidate),
                     "velocitySha256": "mismatch" if wrong_output_hash else digest}
            if changed_input:
                probe["inputs"] = [{"file": "physical.f32", "sha256": "mismatch"}]
            for name, item in [("oracle", oracle), ("probe", probe)]:
                (root / (name+".json")).write_text(json.dumps(item))
            result = subprocess.run([sys.executable, str(TOOL), "--probe-report", str(root / "probe.json"),
                                     "--oracle-report", str(root / "oracle.json"), "--reference", str(reference),
                                     "--output", str(root / "result.json")], capture_output=True, text=True)
            output = root / "result.json"
            return result, json.loads(output.read_text()) if output.exists() else None

    def test_recovers_raw_prediction_from_signed_carrier_velocity(self):
        process, report = self.run_case()
        self.assertEqual(0, process.returncode, process.stderr)
        self.assertTrue(report["passed"])
        for row in report["boundaries"].values():
            self.assertEqual(0, row["relative_l2"])

    def test_wrong_audio_sign_fails_the_bound(self):
        process, report = self.run_case(wrong_audio_sign=True)
        self.assertEqual(1, process.returncode)
        self.assertFalse(report["passed"])

    def test_mismatched_projection_input_cannot_produce_evidence(self):
        process, report = self.run_case(changed_input=True)
        self.assertNotEqual(0, process.returncode)
        self.assertIsNone(report)
        self.assertIn("inputs differ", process.stderr)

    def test_replaced_velocity_files_cannot_be_attributed_to_another_run(self):
        process, report = self.run_case(wrong_output_hash=True)
        self.assertNotEqual(0, process.returncode)
        self.assertIsNone(report)
        self.assertIn("velocities differ", process.stderr)


if __name__ == "__main__":
    unittest.main()
