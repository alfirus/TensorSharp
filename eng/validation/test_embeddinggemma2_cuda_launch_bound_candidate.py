"""Pure isolation/dispatch controls; synthetic compilers are not CUDA or model validation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("cuda_candidate", Path(__file__).with_name("embeddinggemma2-cuda-launch-bound-candidate.py"))
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class CudaIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.native = self.root / "native"; self.build = self.native / "build"
        self.target = self.build / "CMakeFiles/GgmlOps.dir"; self.target.mkdir(parents=True)
        self.source = self.native / "ggml_ops_attention_precision.cu"
        self.source.write_text(tool.KERNEL_START + "view q) { /* original body */ }\n\n" + tool.KERNEL_END + "\n" + tool.DISPATCH + "            vision_kernel<72>();\n")
        self.header = self.native / "original.h"; self.header.write_text("// original header\n")
        self.obj = self.target / "ggml_ops_attention_precision.cu.o"; self.obj.write_bytes(b"ordinary-object")
        self.dep = Path(str(self.obj) + ".d"); self.dep.write_text(f"{self.obj}: {self.source} {self.header}\n")
        self.other = self.target / "other.cpp.o"; self.other.write_bytes(b"other-object")
        self.library = self.build / "libGgmlOps.so"; self.library.write_bytes(b"ordinary-native")
        self.archive = self.build / "libggml.a"; self.archive.write_bytes(b"unchanged-upstream")
        self.cuda = self.root / "cuda"; (self.cuda / "bin").mkdir(parents=True)
        self.compiler = self.cuda / "bin/nvcc"
        self.compiler.write_text("#!/usr/bin/env python3\nimport pathlib,sys\na=sys.argv[1:]\n"
                                 "o=pathlib.Path(a[a.index('-o')+1]);o.write_bytes(b'isolated-cuda-object')\n"
                                 "s=pathlib.Path(a[a.index('-c')+1]);h=pathlib.Path(a[-1])/'original.h'\n"
                                 "pathlib.Path(a[a.index('-MF')+1]).write_text(str(o)+': '+str(s)+' '+str(h)+'\\n')\n")
        self.compiler.chmod(0o755)
        self.host = self.root / "fake-cxx"
        self.host.write_text("#!/usr/bin/env python3\nimport pathlib,sys\na=sys.argv[1:]\n"
                             "if any(x.startswith('-print-') for x in a): print('');sys.exit(0)\n"
                             "pathlib.Path(a[a.index('-o')+1]).write_bytes(b'isolated-library')\n")
        self.host.chmod(0o755)
        for relative in ("bin/ptxas", "bin/fatbinary", "bin/nvlink", "bin/cudafe++", "bin/bin2c", "bin/nvcc.profile", "nvvm/bin/cicc", "nvvm/libdevice/libdevice.10.bc"):
            p = self.cuda / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"synthetic-toolchain")
        self.rsp = self.target / "includes_CUDA.rsp"; self.rsp.write_text("-I" + str(self.native) + "\n")
        self.command = [str(self.compiler), "-forward-unknown-to-host-compiler", "-DGGML_USE_CUDA", "--options-file", str(self.rsp), "-O3", "-DNDEBUG", "-std=c++17", "--generate-code=arch=compute_86,code=[sm_86]", "-Xcompiler=-fPIC", "--ftz=false", "--prec-div=true", "--prec-sqrt=true", "-x", "cu", "-c", str(self.source), "-o", str(self.obj)]
        self.db = self.build / "compile_commands.json"
        self.entry = {"directory": str(self.build), "file": str(self.source), "arguments": self.command}
        self.db.write_text(json.dumps([self.entry]))
        self.link = [str(self.host), "-shared", "-o", str(self.library), str(self.obj), str(self.other), str(self.archive)]
        (self.target / "link.txt").write_text(" ".join(self.link))
        self.ggml = self.root / "ggml"; self.ggml.mkdir()
        subprocess.run(["git", "init", "-q", str(self.ggml)], check=True)
        (self.ggml / "upstream.h").write_text("// unchanged\n")
        subprocess.run(["git", "-C", str(self.ggml), "add", "upstream.h"], check=True)
        subprocess.run(["git", "-C", str(self.ggml), "-c", "user.name=IsolationTest", "-c", "user.email=isolation@example.invalid", "commit", "-qm", "fixture"], check=True)
        self.revision = subprocess.check_output(["git", "-C", str(self.ggml), "rev-parse", "HEAD"], text=True).strip()

    def args(self, execute=False):
        return SimpleNamespace(build_dir=self.build, source=self.source, native_library=self.library, ggml_dir=self.ggml,
                               output_dir=self.root / "artifacts/candidate", expected_source_sha256=tool.sha(self.source),
                               expected_native_sha256=tool.sha(self.library), expected_ggml_revision=self.revision, execute=execute)

    def run_builder(self, args):
        actual = tool.executable
        with patch.object(tool, "executable", side_effect=lambda x: self.host if x == "g++" else actual(x)), patch.object(tool.shutil, "which", wraps=tool.shutil.which) as which:
            original = which._mock_wraps
            which.side_effect = lambda x: str(self.host) if x == "g++" else original(x)
            return tool.build_candidate(args)

    def expanded(self):
        return tool.expand_includes(self.command, self.build)[0]

    def rewrite(self, command):
        return tool.rewrite_compile(command, self.build, self.source, Path("staged.cu"), Path("staged.o"), Path("staged.d"))

    def test_source_exact_duplicate_and_dispatch_preserve_original(self):
        original = self.source.read_text(); candidate = tool.scoped_source(original)
        duplicate = original[:original.index(tool.KERNEL_END)].replace("__launch_bounds__(128)", "__launch_bounds__(128, 3)").replace("void vision_kernel(", "void vision_kernel_minblocks3(")
        self.assertEqual(candidate.replace(duplicate, "").replace(tool.GUARDED_DISPATCH, tool.DISPATCH), original)
        self.assertEqual(self.source.read_text(), original)
        for clause in ("cc == 860", "q.ne[1] == 2304 || q.ne[1] == 2340", "k.ne[1] == q.ne[1]", "v.ne[1] == q.ne[1]", "q.ne[2] == 12", "q.ne[3] == 1", "v.type == GGML_TYPE_F32", "desc.f0 == 1.0f"):
            self.assertIn(clause, candidate)
        self.assertEqual(candidate.count("vision_kernel<72>();"), 1)

    def test_duplicate_or_changed_source_anchor_rejected(self):
        for source in (self.source.read_text() + tool.DISPATCH, self.source.read_text().replace("__launch_bounds__(128)", "__launch_bounds__(64)")):
            with self.assertRaises(ValueError): tool.scoped_source(source)

    def test_actual_owned_kernel_preserved_byte_for_byte(self):
        source = Path(__file__).resolve().parents[2] / "TensorSharp.GGML.Native/ggml_ops_attention_precision.cu"
        original = source.read_text(); candidate = tool.scoped_source(original)
        first, last = original.index(tool.KERNEL_START), original.index(tool.KERNEL_END)
        copied = original[first:last].replace("__launch_bounds__(128)", "__launch_bounds__(128, 3)").replace("void vision_kernel(", "void vision_kernel_minblocks3(")
        self.assertEqual(candidate.replace(copied, "").replace(tool.GUARDED_DISPATCH, tool.DISPATCH), original)

    def test_include_response_expanded_and_bound(self):
        command, path, digest = tool.expand_includes(self.command, self.build)
        self.assertNotIn("--options-file", command); self.assertIn("-I" + str(self.native), command)
        self.assertEqual(path, self.rsp); self.assertEqual(digest, tool.sha(self.rsp))

    def test_response_cannot_inject_compiler_flags(self):
        for value in ("-I" + str(self.native) + " --keep", "@recursive.rsp", "-include injected.h", "-isystem"):
            self.rsp.write_text(value)
            with self.assertRaises(ValueError): tool.expand_includes(self.command, self.build)

    def test_unknown_response_path_rejected(self):
        command = list(self.command); command[command.index("--options-file") + 1] = str(self.native / "other.rsp")
        with self.assertRaisesRegex(ValueError, "captured"): tool.expand_includes(command, self.build)

    def test_selects_exact_library_object_among_test_entries(self):
        other = dict(self.entry); other["arguments"] = list(self.command); other["arguments"][-1] = "CMakeFiles/Test.dir/ggml_ops_attention_precision.cu.o"
        self.db.write_text(json.dumps([other, self.entry]))
        command, _, _ = tool.compile_recipe(self.build, self.source)
        self.assertEqual(command[-1], str(self.obj))

    def test_ambiguous_library_compile_entry_rejected(self):
        self.db.write_text(json.dumps([self.entry, self.entry]))
        with self.assertRaisesRegex(ValueError, "one exact"): tool.compile_recipe(self.build, self.source)

    def test_rewrite_all_compile_outputs(self):
        command, original = self.rewrite(self.expanded())
        self.assertEqual(original, self.obj)
        for flag, value in (("-c", "staged.cu"), ("-o", "staged.o"), ("-MF", "staged.d"), ("-MT", "staged.o")):
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_side_output_and_unknown_flags_rejected(self):
        for flag in ("--keep", "--keep-dir=/unsafe", "-Xcompiler=-save-temps", "-Xptxas=-v", "--use_fast_math", "--maxrregcount=168", "--dryrun", "--generate-line-info", "-ccbin=/unsafe"):
            with self.assertRaises(ValueError): self.rewrite(self.expanded() + [flag])

    def test_architecture_and_precision_must_match(self):
        for old, new in (("--generate-code=arch=compute_86,code=[sm_86]", "--generate-code=arch=compute_80,code=[sm_80]"), ("--prec-div=true", "--prec-div=false")):
            with self.assertRaises(ValueError): self.rewrite([new if x == old else x for x in self.expanded()])

    def test_relative_dependencies_resolve_against_build(self):
        self.assertEqual(tool.dependencies(b"object.o: ../original.h\n", self.build), {self.header})

    def test_symlink_spelling_is_guarded_as_well_as_target(self):
        alias = self.root / "alias.h"; alias.symlink_to(self.header)
        data = ("object.o: " + str(alias) + "\n").encode()
        self.assertIn(alias, tool.dependency_aliases(data, self.build))
        self.assertIn(self.header, tool.dependencies(data, self.build))
        library_alias = self.root / "libalias.so"; library_alias.symlink_to(self.library)
        self.assertIn(library_alias, tool.explicit_library_aliases([str(self.host), str(library_alias)], self.build))

    def test_missing_toolkit_component_rejected(self):
        (self.cuda / "nvvm/libdevice/libdevice.10.bc").unlink()
        with self.assertRaisesRegex(ValueError, "bitcode"): tool.cuda_toolchain(self.compiler)

    def test_unknown_host_environment_rejected_before_staging(self):
        with patch.dict(tool.os.environ, {"NVCC_PREPEND_FLAGS": "--keep"}):
            with self.assertRaisesRegex(ValueError, "injection"): self.run_builder(self.args())
        self.assertFalse(self.args().output_dir.exists())

    def test_wrong_source_native_hash_rejected(self):
        for field in ("expected_source_sha256", "expected_native_sha256"):
            args = self.args(); setattr(args, field, "0" * 64)
            with self.assertRaisesRegex(ValueError, "identity"): self.run_builder(args)

    def test_prepare_only_has_no_build_or_math_claim(self):
        args = self.args(); report = self.run_builder(args)
        self.assertTrue(report["completed"] and report["guarded_inputs_unchanged"] and report["upstream_unchanged"])
        self.assertFalse(report["built"] or report["quality_gates_executed"] or report["model_qualified"])
        self.assertFalse((args.output_dir / "libGgmlOps.so").exists())

    def test_synthetic_compile_relink_preserves_normal_build(self):
        args = self.args(True); normal = self.library.read_bytes(); obj = self.obj.read_bytes(); report = self.run_builder(args)
        self.assertTrue(report["built"] and report["guarded_inputs_unchanged"])
        self.assertEqual(self.library.read_bytes(), normal); self.assertEqual(self.obj.read_bytes(), obj)
        self.assertEqual(len(report["runs"]), 2); self.assertFalse(report["model_qualified"])

    def test_changed_header_during_compile_blocks_relink(self):
        self.compiler.write_text(self.compiler.read_text() + "h.write_text('// changed during compile\\n')\n")
        args = self.args(True)
        with self.assertRaisesRegex(ValueError, "changed before link"): self.run_builder(args)
        report = json.loads((args.output_dir / "manifest.json").read_text())
        self.assertFalse(report["completed"] or report["built"] or report["guarded_inputs_unchanged"])
        self.assertEqual(len(report["runs"]), 1); self.assertTrue(report["ordinary_native_unchanged"])

    def test_new_unbound_include_blocks_relink(self):
        extra = self.root / "unbound.h"; extra.write_text("// not in original closure\n")
        self.compiler.write_text(self.compiler.read_text() + "d=pathlib.Path(a[a.index('-MF')+1]);d.write_text(d.read_text().rstrip()+' " + str(extra) + "\\n')\n")
        args = self.args(True)
        with self.assertRaisesRegex(ValueError, "unbound include"): self.run_builder(args)
        report = json.loads((args.output_dir / "manifest.json").read_text())
        self.assertFalse(report["completed"] or report["built"])
        self.assertEqual(len(report["runs"]), 1); self.assertTrue(report["guarded_inputs_unchanged"])


if __name__ == "__main__":
    unittest.main()
