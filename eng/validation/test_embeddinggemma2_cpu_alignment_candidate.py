"""Build-isolation controls only; synthetic compiler is not native/model validation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("alignment_candidate", Path(__file__).with_name("embeddinggemma2-cpu-alignment-candidate.py"))
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class AlignmentIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.native = self.root / "native"
        self.build = self.native / "build"
        self.target = self.build / "CMakeFiles/GgmlOps.dir"
        self.target.mkdir(parents=True)
        self.source = self.native / "ggml_ops_embeddings.cpp"
        self.source.write_text("bool validated_cpu_geometry = true;\n"
                               "e->cpu_float_linears = std::all_of(...);\n"
                               "    if (cpu) {\n"
                               "        if (all_quantized_optimized) e->cpu_alignment = 4;\n    }\n" + tool.ANCHOR)
        self.header = self.native / "original.h"
        self.header.write_text("// original captured header\n")
        self.obj = self.target / "ggml_ops_embeddings.cpp.o"
        self.obj.write_bytes(b"ordinary-object")
        Path(str(self.obj) + ".d").write_text(f"{self.obj}: {self.source} {self.header}\n")
        self.other = self.target / "other.cpp.o"
        self.other.write_bytes(b"other-object")
        self.library = self.build / "libGgmlOps.so"
        self.library.write_bytes(b"ordinary-native")
        self.archive = self.build / "libggml.a"
        self.archive.write_bytes(b"unchanged-static")
        self.compiler = self.root / "fake-cxx"
        self.compiler.write_text("#!/usr/bin/env python3\nimport pathlib,sys\n"
                                 "a=sys.argv[1:]\n"
                                 "if any(x.startswith('-print-') for x in a): print('');sys.exit(0)\n"
                                 "o=pathlib.Path(a[a.index('-o')+1]);o.write_bytes(b'isolated-output')\n"
                                 "if '-c' in a:\n"
                                 " s=pathlib.Path(a[a.index('-c')+1]);h=pathlib.Path(a[a.index('-iquote')+1])/'original.h'\n"
                                 " pathlib.Path(a[a.index('-MF')+1]).write_text(str(o)+': '+str(s)+' '+str(h)+'\\n')\n")
        self.compiler.chmod(0o755)
        self.compile = [str(self.compiler), "-DTSG_EMBEDDING_CPU_TILED_GEMM=1", "-O3", "-fPIC", "-o", str(self.obj), "-c", str(self.source)]
        self.database = self.build / "compile_commands.json"
        self.database.write_text(json.dumps([{"directory": str(self.build), "file": str(self.source), "arguments": self.compile}]))
        self.link = [str(self.compiler), "-shared", "-o", str(self.library), str(self.obj), str(self.other), str(self.archive)]
        (self.target / "link.txt").write_text(" ".join(self.link))
        self.ggml = self.root / "ggml"
        self.ggml.mkdir()
        subprocess.run(["git", "init", "-q", str(self.ggml)], check=True)
        (self.ggml / "upstream.h").write_text("// unchanged upstream\n")
        subprocess.run(["git", "-C", str(self.ggml), "add", "upstream.h"], check=True)
        subprocess.run(["git", "-C", str(self.ggml), "-c", "user.name=IsolationTest", "-c", "user.email=isolation@example.invalid", "commit", "-qm", "fixture"], check=True)
        self.revision = subprocess.check_output(["git", "-C", str(self.ggml), "rev-parse", "HEAD"], text=True).strip()

    def args(self, execute=False):
        return SimpleNamespace(build_dir=self.build, source=self.source, native_library=self.library,
                               ggml_dir=self.ggml, output_dir=self.root / "artifacts/candidate",
                               expected_source_sha256=tool.sha(self.source), expected_native_sha256=tool.sha(self.library),
                               expected_ggml_revision=self.revision, execute=execute)

    def test_exact_scoped_insertion_preserves_original(self):
        original = self.source.read_text()
        candidate = tool.scoped_source(original)
        self.assertEqual(candidate.replace(tool.INSERTION, ""), original)
        self.assertEqual(candidate.count(tool.INSERTION), 1)
        self.assertEqual(self.source.read_text(), original)

    def test_duplicate_anchor_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            tool.scoped_source(self.source.read_text() + tool.ANCHOR)

    def test_insertion_before_optimized_loop_rejected(self):
        with self.assertRaisesRegex(ValueError, "optimized-weight"):
            tool.scoped_source(self.source.read_text().replace("if (all_quantized_optimized)", "if (other)"))

    def test_source_hash_mismatch_rejected_before_staging(self):
        a = self.args(); a.expected_source_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hash"):
            tool.build_candidate(a)
        self.assertFalse(a.output_dir.exists())

    def test_native_hash_mismatch_rejected(self):
        a = self.args(); a.expected_native_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "library hash"):
            tool.build_candidate(a)

    def test_nonignored_destination_rejected(self):
        a = self.args(); a.output_dir = self.root / "unsafe"
        with self.assertRaisesRegex(ValueError, "ignored"):
            tool.build_candidate(a)

    def test_existing_snapshot_rejected(self):
        a = self.args(); a.output_dir.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            tool.build_candidate(a)

    def test_response_and_shell_text_rejected(self):
        for value in ("c++ @objects.rsp", "c++ a && touch unwanted", "c++ $(whoami)"):
            with self.assertRaises(ValueError):
                tool.argv(value)

    def test_duplicate_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique -o"):
            tool.rewrite_compile(self.compile + ["-o", "other.o"], self.build, self.source, Path("copy.cpp"), Path("copy.o"), Path("copy.d"))

    def test_compiler_side_output_prefixes_rejected(self):
        for flag in ["-save-temps=obj", "--save-temps=cwd", "-fdump-tree-all", "-fprofile-generate=/unsafe", "-ftime-trace"]:
            with self.assertRaisesRegex(ValueError, "side outputs"):
                tool.rewrite_compile(self.compile + [flag], self.build, self.source, Path("copy.cpp"), Path("copy.o"), Path("copy.d"))

    def test_linker_dependency_side_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "side outputs"):
            tool.rewrite_link(self.link + ["-Wl,--dependency-file=/unsafe"], self.build, self.obj, self.library, Path("copy.o"), Path("copy.so"))

    def test_relative_dependencies_use_build_working_directory(self):
        dep = self.target / "relative.d"
        dep.write_text("target.o: ../original.h\n")
        self.assertEqual(tool.dependencies(dep, self.build), [self.header])

    def test_missing_link_object_rejected(self):
        command = [x for x in self.link if x != str(self.obj)]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            tool.rewrite_link(command, self.build, self.obj, self.library, Path("copy.o"), Path("copy.so"))

    def test_dependency_file_required(self):
        Path(str(self.obj) + ".d").unlink()
        with self.assertRaisesRegex(ValueError, "dependency file required"):
            tool.build_candidate(self.args())

    def test_relative_linker_script_input_is_guarded(self):
        script = self.build / "libcustom.so"
        concrete = self.build / "libconcrete.so.1"
        concrete.write_bytes(b"concrete-link-input")
        script.write_text("GROUP ( libconcrete.so.1 )\n")
        files = tool.library_files([str(self.compiler), str(script)], self.build, self.compiler)
        self.assertIn(script, files)
        self.assertIn(concrete, files)

    def test_upstream_changes_rejected(self):
        (self.ggml / "upstream.h").write_text("// modified\n")
        with self.assertRaisesRegex(ValueError, "checkout is modified"):
            tool.build_candidate(self.args())

    def test_prepare_only_never_builds(self):
        a = self.args(); report = tool.build_candidate(a)
        self.assertTrue(report["completed"])
        self.assertFalse(report["built"])
        self.assertEqual(report["runs"], [])
        self.assertFalse((a.output_dir / self.library.name).exists())
        self.assertTrue(report["guarded_inputs_unchanged"])
        self.assertFalse(report["quality_gates_executed"])

    def test_synthetic_compiler_outputs_are_isolated_and_guarded(self):
        a = self.args(True); original = self.library.read_bytes(); obj = self.obj.read_bytes()
        report = tool.build_candidate(a)
        self.assertTrue(report["built"] and report["guarded_inputs_unchanged"] and report["upstream_unchanged"])
        self.assertEqual(self.library.read_bytes(), original)
        self.assertEqual(self.obj.read_bytes(), obj)
        self.assertEqual(len(report["runs"]), 2)
        self.assertFalse(report["model_qualified"])
        self.assertFalse(report["quality_gates_executed"])
        self.assertEqual(report["compile_command"][report["compile_command"].index("-c") + 1], str(a.output_dir / self.source.name))

    def test_input_mutation_during_compile_prevents_link_and_qualification(self):
        self.compiler.write_text(self.compiler.read_text() + "if '-c' in a: h.write_text('// changed during compile\\n')\n")
        a = self.args(True)
        with self.assertRaisesRegex(ValueError, "changed before link"):
            tool.build_candidate(a)
        report = json.loads((a.output_dir / "manifest.json").read_text())
        self.assertFalse(report["completed"] or report["built"] or report["guarded_inputs_unchanged"])
        self.assertTrue(report["ordinary_native_unchanged"])
        self.assertEqual(len(report["runs"]), 1)
        self.assertFalse((a.output_dir / self.library.name).exists())


if __name__ == "__main__":
    unittest.main()
