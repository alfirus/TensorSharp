import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("rms_arm_candidate", Path(__file__).parents[1] / "embeddinggemma2-cpu-rms-arm-candidate.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class RmsArmCandidateControls(unittest.TestCase):
    def source(self):
        return '#include "embedding_cpu_rms.h"\nbool validated_cpu_geometry = true;\n' + TOOL.OLD

    def test_exact_scope_only(self):
        source = self.source()
        candidate = TOOL.scoped_source(source)
        self.assertEqual(candidate.replace(TOOL.NEW, TOOL.OLD), source)
        self.assertIn("defined(__APPLE__) && defined(__aarch64__)", candidate)
        self.assertIn("(threads > 0 ? threads : 4) == 7", candidate)
        self.assertIn("(threads > 0 ? threads : 4) == 8", candidate)
        self.assertNotIn("cpu_qkv_rms_minimum_tokens", candidate)

    def test_actual_current_encoder_supported(self):
        path = Path(__file__).parents[3] / "TensorSharp.GGML.Native/ggml_ops_embeddings.cpp"
        source = path.read_text()
        if TOOL.NEW in source:
            self.assertEqual(source.count(TOOL.NEW), 1)
            with self.assertRaisesRegex(ValueError, "exactly the known"):
                TOOL.scoped_source(source)
        else:
            self.assertIn(TOOL.NEW, TOOL.scoped_source(source))

    def test_changed_duplicate_and_reapplied_scope_reject(self):
        for source in (self.source().replace("== 7", "== 6"), self.source() + TOOL.OLD,
                       TOOL.scoped_source(self.source()), self.source().replace('#include "embedding_cpu_rms.h"', "")):
            with self.assertRaises(ValueError):
                TOOL.scoped_source(source)

    def test_install_name_is_not_a_response_file(self):
        command = TOOL.mac_argv("/usr/bin/c++ -dynamiclib -o libGgmlOps.dylib -install_name @rpath/libGgmlOps.dylib a.o")
        self.assertEqual(command[0], "/usr/bin/c++")
        for text in ("/usr/bin/c++ @inputs.rsp", "/usr/bin/c++ -install_name @elsewhere/x", "/usr/bin/c++ ; touch bad"):
            with self.assertRaises(ValueError):
                TOOL.mac_argv(text)

    def test_link_redirect_only_and_driver_spelling(self):
        base = TOOL.helper()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            original = root / "encoder.o"
            native = root / "libGgmlOps.dylib"
            command = ["/usr/bin/c++", "-dynamiclib", "-o", str(native), "-install_name", "@rpath/libGgmlOps.dylib", str(original), "other.o", "libggml.a", "-framework", "Metal"]
            candidate = TOOL.rewrite_link(command, root, original, native, root / "private.o", root / "private.dylib", base)
            self.assertEqual(candidate[0], "/usr/bin/c++")
            candidate[candidate.index(str(root / "private.o"))] = str(original)
            candidate[candidate.index(str(root / "private.dylib"))] = str(native)
            self.assertEqual(candidate, command)
            for suffix in (["-Wl,-map,unsafe"], ["@response"], ["-Xlinker", "-map"], [str(original)]):
                with self.assertRaises(ValueError):
                    TOOL.rewrite_link(command + suffix, root, original, native, root / "private.o", root / "private.dylib", base)

    def test_snapshot_detects_changed_bytes_and_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); first = root / "first"; second = root / "second"; alias = root / "alias"
            first.write_text("same"); second.write_text("same"); alias.symlink_to(first)
            before = TOOL.snapshot([alias]); TOOL.unchanged(before)
            alias.unlink(); alias.symlink_to(second)
            with self.assertRaises(ValueError):
                TOOL.unchanged(before)
            before = TOOL.snapshot([first]); first.write_text("different")
            with self.assertRaises(ValueError):
                TOOL.unchanged(before)

    def test_output_requires_ignored_physical_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); source = root / "native/encoder.cpp"; build = root / "native/build"
            source.parent.mkdir(); build.mkdir()
            with patch.object(TOOL.subprocess, "run") as run:
                run.return_value.returncode = 0
                self.assertEqual(TOOL.safe_output(root / "docs/validation/new", root, build, source), root / "docs/validation/new")
                for output in (root / "native/build/new", root / "docs/validation/../escaped", root):
                    with self.assertRaises(ValueError):
                        TOOL.safe_output(output, root, build, source)
                (root / "docs/validation").mkdir(parents=True)
                (root / "docs/validation/alias").symlink_to(build)
                with self.assertRaises(ValueError):
                    TOOL.safe_output(root / "docs/validation/alias/new", root, build, source)
                run.return_value.returncode = 1
                with self.assertRaises(ValueError):
                    TOOL.safe_output(root / "artifacts/new", root, build, source)

    def test_compile_rejects_precision_and_side_output_options(self):
        command = ["/usr/bin/c++", "-DGGML_USE_CPU", "-I/native", "-F/sdk/frameworks", "-O3", "-DNDEBUG", "-std=gnu++17", "-arch", "arm64", "-fPIC", "-MD", "-MT", "a.o", "-MF", "a.d", "-o", "a.o", "-c", "a.cpp"]
        TOOL.validate_compile(command)
        for option in ("-ffast-math", "-Ofast", "-fmodule-output=unbound", "-MJ", "-Xclang", "@response"):
            with self.assertRaises(ValueError):
                TOOL.validate_compile(command + [option])

    def test_dependency_alias_spelling_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); original = root / "original.h"; original.write_text("header")
            alias = root / "alias.h"; alias.symlink_to(original)
            dep = root / "encoder.d"; dep.write_text("encoder.o: alias.h\n")
            self.assertEqual(TOOL.dependency_spellings(dep, root), [alias])

    def test_generated_input_extension_cannot_rebase_prior_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); source = root / "source.cpp"; obj = root / "encoder.o"
            source.write_text("source"); obj.write_text("object")
            before = TOOL.snapshot([source]); derived = TOOL.extend_snapshot(before, [source, obj])
            self.assertEqual(derived[str(source)], before[str(source)])
            obj.write_text("changed")
            with self.assertRaises(ValueError):
                TOOL.extend_snapshot(derived, [source, obj])


if __name__ == "__main__":
    unittest.main()
