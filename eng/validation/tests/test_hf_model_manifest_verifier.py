import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "model_verifier", Path(__file__).resolve().parents[1] / "verify-hf-model-manifest.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class ManifestVerifierTests(unittest.TestCase):
    def fixture(self, root, verified=False):
        files = []
        for index in range(3):
            data = bytes([index]) * (1024 + index)
            path = root / f"shard-{index}.gguf"
            path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            files.append(dict(path=path.name, size=len(data), sha256=digest,
                              actual_sha256=digest if verified else None, verified=verified))
        source = root / "source.json"
        source.write_text(json.dumps(dict(repo="test/model", revision="a" * 40, files=files, complete=False)))
        return source

    def run_verify(self, source, root, reuse=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return VERIFIER.verify_manifest(source, root, root / "verified.json", 4, reuse)

    def test_parallel_hashes_every_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_verify(self.fixture(root), root)
            self.assertTrue(result["complete"])
            self.assertTrue(all(f["verified"] and not f["verification_reused"] for f in result["files"]))

    def test_reuse_requires_verified_record_and_unchanged_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root, verified=True)
            result = self.run_verify(source, root, reuse=True)
            self.assertTrue(all(f["verification_reused"] for f in result["files"]))
            path = root / "shard-1.gguf"
            path.write_bytes(b"x" * path.stat().st_size)
            os.utime(path, ns=(source.stat().st_mtime_ns + 2_000_000_000,) * 2)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.run_verify(source, root, reuse=True)
            self.assertFalse(json.loads((root / "verified.json").read_text())["complete"])

    def test_unverified_matching_hash_record_does_not_skip_hashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            manifest = json.loads(source.read_text())
            for record in manifest["files"]:
                record["actual_sha256"] = record["sha256"]
            source.write_text(json.dumps(manifest))
            result = self.run_verify(source, root, reuse=True)
            self.assertTrue(all(not f["verification_reused"] for f in result["files"]))

    def test_size_and_path_failures_never_complete(self):
        for invalid in ("size", "path"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self.fixture(root)
                manifest = json.loads(source.read_text())
                if invalid == "size":
                    manifest["files"][0]["size"] += 1
                else:
                    manifest["files"][0]["path"] = "../outside.gguf"
                source.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    self.run_verify(source, root)
                self.assertFalse(json.loads((root / "verified.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
