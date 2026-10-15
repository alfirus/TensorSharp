"""Offline downloader policy tests with tiny synthetic files, never model execution."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("ltx25_download", ROOT / "eng/validation/ltx25-download-companions.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

TEXT = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
DIFFUSION = "vae/ltx-2.5-video-vae-bf16.safetensors"
CONV = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
AUDIO = "vae/ltx-2.5-audio-vae-bf16.safetensors"


class DownloadCompanionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.models = self.root / "models"
        self.models.mkdir()
        self.access = self.root / "access.json"
        self.output = self.root / "report.json"
        self.data = {name: ("synthetic fixture " + name).encode() for name in [TEXT, DIFFUSION, CONV, AUDIO]}
        self.manifest = dict(repository="test/fixture", revision="a" * 40, access_confirmed=True, credential_available=True,
                             files=[dict(path=name, bytes=len(data), lfs_sha256=hashlib.sha256(data).hexdigest(),
                                         etag=hashlib.sha256(data).hexdigest(), commit_hash="a" * 40, access_confirmed=True)
                                    for name, data in self.data.items()])

    def existing(self, *names):
        for name in names:
            path = self.models / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.data[name])

    def run_download(self, *arguments, downloader=None, free=100000):
        self.access.write_text(json.dumps(self.manifest))
        argv = ["ltx25-download-companions.py", "--hf", str(self.root / "hf"), "--access-manifest", str(self.access),
                "--models", str(self.models), "--output", str(self.output), "--reserve-gib", "0", *arguments]
        self.stdout = io.StringIO()
        with patch.object(sys, "argv", argv), \
                patch.object(download.shutil, "disk_usage", return_value=types.SimpleNamespace(free=free)), \
                patch.object(download.subprocess, "run", side_effect=downloader) as process, \
                contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stdout):
            result = download.main()
        return result, json.loads(self.output.read_text()), process

    def test_default_and_conv_select_distinct_pinned_decoder_files(self):
        self.existing(TEXT, DIFFUSION, CONV, AUDIO)
        for args, decoder in [((), DIFFUSION), (("--video-vae", "conv"), CONV)]:
            with self.subTest(decoder=decoder):
                result, report, process = self.run_download(*args)
                self.assertEqual(result, 0)
                self.assertTrue(report["completed"])
                self.assertEqual([item["path"] for item in report["files"]], [TEXT, decoder, AUDIO])
                self.assertTrue(all(item["completed"] for item in report["files"]))
                self.assertEqual(report["remaining_disk_growth_bytes"], 0)
                process.assert_not_called()

    def test_conv_requires_its_own_unique_confirmed_metadata(self):
        self.existing(TEXT, DIFFUSION, AUDIO)
        original = self.manifest["files"]
        for files in [[item for item in original if item["path"] != CONV], original + [original[2]]]:
            with self.subTest(files=len(files)):
                self.manifest["files"] = files
                with self.assertRaisesRegex(ValueError, "exactly one pinned metadata.*" + CONV):
                    self.run_download("--video-vae", "conv")
        self.manifest["files"] = original
        for field, value in [("access_confirmed", False), ("commit_hash", "b" * 40), ("etag", "c" * 64)]:
            with self.subTest(field=field):
                saved = original[2][field]
                original[2][field] = value
                with self.assertRaisesRegex(ValueError, "pinned official metadata"):
                    self.run_download("--video-vae", "conv")
                original[2][field] = saved

    def test_corrupt_existing_conv_is_force_downloaded_and_verified(self):
        self.existing(TEXT, CONV, AUDIO)
        (self.models / CONV).write_bytes(b"x" * len(self.data[CONV]))

        def repair(argv, **kwargs):
            self.assertIn("--force-download", argv)
            self.assertEqual(argv[3], CONV)
            self.assertEqual(kwargs["env"]["HF_HUB_DISABLE_XET"], "1")
            (self.models / CONV).write_bytes(self.data[CONV])
            return types.SimpleNamespace(returncode=0)

        result, report, process = self.run_download("--video-vae", "conv", downloader=repair)
        self.assertEqual(result, 0)
        self.assertTrue(report["completed"])
        self.assertEqual(process.call_count, 1)
        self.assertEqual(report["files"][1]["actual_sha256"], self.manifest["files"][2]["lfs_sha256"])

    def test_disk_preflight_credits_only_the_selected_component_resume_hash(self):
        self.existing(TEXT, AUDIO)
        item = self.manifest["files"][2]
        cache = self.models / ".cache/huggingface/download/vae"
        cache.mkdir(parents=True)
        (cache / ("partial." + item["lfs_sha256"] + ".incomplete")).write_bytes(b"x" * 7)
        (cache / ("other." + "f" * 64 + ".incomplete")).write_bytes(b"x" * 500)
        result, report, process = self.run_download("--video-vae", "conv", free=len(self.data[CONV]) - 8)
        self.assertEqual(result, 1)
        self.assertFalse(report["completed"])
        self.assertEqual(report["resumable_bytes"][CONV], 7)
        self.assertEqual(report["remaining_disk_growth_bytes"], len(self.data[CONV]) - 7)
        self.assertEqual(report["error_type"], "ValueError")
        process.assert_not_called()

    def test_transport_failure_does_not_log_captured_secret_output(self):
        self.existing(TEXT, AUDIO)
        secret = "temporary-signed-url-or-token-must-not-be-logged"
        result, report, _ = self.run_download("--video-vae", "conv", downloader=lambda *args, **kwargs:
                                            types.SimpleNamespace(returncode=9, stdout=secret, stderr=secret))
        self.assertEqual(result, 1)
        self.assertFalse(report["completed"])
        self.assertEqual(report["files"][-1]["exit_code"], 9)
        self.assertNotIn(secret, self.output.read_text() + self.stdout.getvalue())

    def test_successful_transport_with_wrong_bytes_never_counts_as_completed(self):
        self.existing(TEXT, AUDIO)

        def corrupt(argv, **kwargs):
            path = self.models / CONV
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * len(self.data[CONV]))
            return types.SimpleNamespace(returncode=0)

        result, report, _ = self.run_download("--video-vae", "conv", downloader=corrupt)
        self.assertEqual(result, 1)
        self.assertFalse(report["completed"])
        self.assertFalse(report["files"][-1]["completed"])
        self.assertEqual(report["error_type"], "ValueError")

    def test_invalid_reserve_cannot_bypass_disk_preflight(self):
        for value in ["-1", "nan", "inf"]:
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.run_download("--reserve-gib", value)


if __name__ == "__main__":
    unittest.main()
