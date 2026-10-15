import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
import contextlib
import io
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "embeddinggemma2-cuda-k8-projection-run.py"
SPEC = importlib.util.spec_from_file_location("k8_runner", PATH)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class K8RunnerControls(unittest.TestCase):
    def copied_fixture(self, root):
        (root / ".gitignore").write_text("/docs/validation/\nartifacts/\n")
        proof = dict(schema=1, checked_in_git_repository=True,
                     gitignore_sha256=TOOL.digest(root / ".gitignore"),
                     ignored_prefixes={"docs/validation": True, "artifacts": True},
                     limitations="Synthetic parser control; not an actual local Git proof")
        path = root / "policy.json"
        path.write_text(json.dumps(proof))
        return path, TOOL.digest(path), proof

    def test_explicit_copied_policy_accepts_without_git_and_retains_full_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, sha, proof = self.copied_fixture(root)
            output = root / "docs/validation/fresh"
            with patch.object(TOOL.subprocess, "run", side_effect=AssertionError("Copied route must not fake Git")):
                self.assertEqual(TOOL.fresh_ignored_output(output, root, copied_policy=path, expected_policy_sha256=sha), output)
            accepted = TOOL.copied_source_ignore_policy(path, sha, root, output)
            self.assertEqual(accepted["proof"], proof)
            self.assertEqual(accepted["sha256"], sha)

    def test_copied_policy_rejects_partial_flags_and_cli_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, sha, _ = self.copied_fixture(root)
            for policy, expected in ((path, None), (None, sha)):
                with self.assertRaises(ValueError):
                    TOOL.copied_source_ignore_policy(policy, expected, root, root / "artifacts/fresh")
            for flags in (("--copied-source-ignore-policy", str(path)),
                          ("--expected-copied-source-ignore-policy-sha256", sha)):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    TOOL.main(["--output", str(root / "artifacts/fresh"), *flags])
                self.assertEqual(error.exception.code, 2)

    def test_copied_policy_rejects_hash_schema_prefix_and_git_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, sha, proof = self.copied_fixture(root)
            output = root / "artifacts/fresh"
            with self.assertRaises(ValueError):
                TOOL.copied_source_ignore_policy(path, "0" * 64, root, output)
            for key, value in (("schema", True), ("schema", 2), ("checked_in_git_repository", False),
                               ("ignored_prefixes", {"docs/validation": True, "artifacts": 1}),
                               ("ignored_prefixes", {"docs/validation": True, "artifacts": True, "native": True})):
                bad = dict(proof)
                bad[key] = value
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    TOOL.copied_source_ignore_policy(path, TOOL.digest(path), root, output)
            path.write_text(json.dumps(proof))
            (root / ".git").mkdir()
            with self.assertRaises(ValueError):
                TOOL.copied_source_ignore_policy(path, TOOL.digest(path), root, output)

    def test_copied_policy_rejects_changed_root_bytes_and_new_nested_rules(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, sha, _ = self.copied_fixture(root)
            output = root / "docs/validation/fresh"
            original = (root / ".gitignore").read_bytes()
            (root / ".gitignore").write_bytes(original + b"!override\n")
            with self.assertRaises(ValueError):
                TOOL.copied_source_ignore_policy(path, sha, root, output)
            (root / ".gitignore").write_bytes(original)
            TOOL.copied_source_ignore_policy(path, sha, root, output)
            (root / "docs/validation").mkdir(parents=True)
            (root / "docs/.gitignore").write_text("!validation\n")
            with self.assertRaises(ValueError):
                TOOL.copied_source_ignore_policy(path, sha, root, output)

    def test_copied_policy_rejects_traversal_and_alias_override_or_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, sha, _ = self.copied_fixture(root)
            (root / "docs/validation").mkdir(parents=True)
            (root / "native").mkdir()
            (root / "docs/validation/native").symlink_to(root / "native")
            for output in (root / "docs/validation/../escaped", root / "docs/validation/native/fresh"):
                with self.assertRaises(ValueError):
                    TOOL.fresh_ignored_output(output, root, copied_policy=path, expected_policy_sha256=sha)
            (root / "artifacts/nested").mkdir(parents=True)
            (root / "artifacts/.gitignore").write_text("!nested\n")
            (root / "docs/validation/other").symlink_to(root / "artifacts/nested")
            with self.assertRaises(ValueError):
                TOOL.fresh_ignored_output(root / "docs/validation/other/fresh", root, copied_policy=path, expected_policy_sha256=sha)

    def test_copied_policy_detects_proof_change_on_revalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, sha, _ = self.copied_fixture(root)
            output = root / "artifacts/fresh"
            TOOL.copied_source_ignore_policy(path, sha, root, output)
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                TOOL.copied_source_ignore_policy(path, sha, root, output)

    def test_dependency_paths_bind_captured_cwd(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dep = root / "a.d"
            dep.write_text("probe: relative.h \\\n /absolute/header.h\n")
            self.assertEqual(TOOL.dependencies(dep, root), [root / "relative.h", Path("/absolute/header.h")])
            dep.write_text("wrong: header.h")
            with self.assertRaises(ValueError):
                TOOL.dependencies(dep, root)

    def test_symlink_alias_change_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, second, alias = root / "a", root / "b", root / "link"
            first.write_text("same")
            second.write_text("same")
            alias.symlink_to(first)
            before = TOOL.snapshot([alias])
            TOOL.require_same(before)
            alias.unlink()
            alias.symlink_to(second)
            with self.assertRaises(ValueError):
                TOOL.require_same(before)

    def test_bound_json_uses_exact_read_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "proof.json"
            path.write_bytes(b'{"passed":true}\n')
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertTrue(TOOL.read_bound(path, sha)["passed"])
            path.write_bytes(b'{"passed":true} \n')
            with self.assertRaises(ValueError):
                TOOL.read_bound(path, sha)

    def test_widening_guard_does_not_rebase_existing_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, extra = root / "old", root / "extra"
            old.write_text("original")
            extra.write_text("additional")
            before = TOOL.snapshot([old])
            widened = TOOL.extend_snapshot(before, [old, extra])
            self.assertEqual(widened[str(old)], before[str(old)])
            old.write_text("changed")
            with self.assertRaises(ValueError):
                TOOL.extend_snapshot(before, [old, extra])

    def test_child_negative_can_be_recorded_without_setup_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(TOOL.command([sys.executable, "-c", "raise SystemExit(7)"], root,
                                          root / "negative.log", allow_failure=True), 7)
            with self.assertRaises(RuntimeError):
                TOOL.command([sys.executable, "-c", "raise SystemExit(8)"], root, root / "setup.log")

    def test_ldd_missing_and_resolved_inputs(self):
        self.assertEqual(TOOL.ldd_files("libx.so => /lib/libx.so (0x123)\n /lib/ld.so (0x456)\nlinux-vdso.so (0x789)"),
                         [Path("/lib/libx.so"), Path("/lib/ld.so")])
        with self.assertRaises(ValueError):
            TOOL.ldd_files("libx.so => not found")

    def test_existing_or_unignored_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                TOOL.fresh_ignored_output(root, root)
            with patch.object(TOOL.subprocess, "run") as run:
                run.return_value.returncode = 1
                with self.assertRaises(ValueError):
                    TOOL.fresh_ignored_output(root / "docs/validation/fresh", root)
                run.return_value.returncode = 0
                self.assertEqual(TOOL.fresh_ignored_output(root / "docs/validation/fresh", root), root / "docs/validation/fresh")
                with self.assertRaises(ValueError):
                    TOOL.fresh_ignored_output(root / "ordinary-native-build/fresh", root)

    def test_instruction_review_requires_each_exact_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts = {name: root / name for name in ("source", "executable", "ptx", "sass")}
            for name, path in artifacts.items():
                path.write_text(name)
            review = dict(reviewed=True, reviewer="independent", k8_mma_and_external_rn_order=True,
                          **{name + "_sha256": TOOL.digest(path) for name, path in artifacts.items()})
            path = root / "review.json"
            path.write_text(json.dumps(review))
            TOOL.instruction_review(path, TOOL.digest(path), artifacts)
            for key in ("reviewed", "reviewer", "k8_mma_and_external_rn_order", "source_sha256", "executable_sha256", "ptx_sha256", "sass_sha256"):
                bad = dict(review)
                bad[key] = False if isinstance(bad[key], bool) else ""
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    TOOL.instruction_review(path, TOOL.digest(path), artifacts)


if __name__ == "__main__":
    unittest.main()
