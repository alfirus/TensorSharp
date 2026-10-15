import hashlib
import importlib.util
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

HERE = Path(__file__).parents[1]
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("rms_residual_candidate", HERE / "embeddinggemma2-cpu-rms-residual-candidate.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class RmsResidualSourceControls(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "TensorSharp.GGML.Native/ggml_ops_embeddings.cpp").read_text()
        self.component = TOOL.COMPONENT.read_text()

    def test_exact_arithmetic_extraction(self):
        injected = TOOL.injected_row(self.component)
        normalized = injected.removeprefix(TOOL.ATTRIBUTE).replace(TOOL.PRAGMA, "", 1)
        normalized = normalized.replace("void staged_rms_residual_row(", "void row(", 1).replace("staged_rms_residual_width", "width")
        self.assertEqual(normalized, TOOL.component_row(self.component))
        self.assertIn("sum += double(x[i] * x[i])", normalized)
        self.assertIn("y[i] = weighted + residual[i]", normalized)
        self.assertIn("#pragma clang fp contract(off)", injected)

    def test_three_sites_platform_precision_and_lifetime(self):
        result = TOOL.transform(self.source, self.component)
        for before, after in TOOL.CALLS.items():
            self.assertEqual(self.source.count(before), 1)
            self.assertNotIn(before, result)
            self.assertEqual(result.count(after), 1)
        self.assertIn("validated_cpu_geometry && e->cpu_float_linears", result)
        self.assertIn("#if defined(__APPLE__) && defined(__aarch64__)", TOOL.ADMISSION)
        self.assertNotIn("__linux__", TOOL.ADMISSION)
        self.assertIn("rms_residual_workers == 7 || rms_residual_workers == 8", result)
        self.assertLess(result.index("if (all_quantized_optimized) e->cpu_alignment = 4;"), result.index(TOOL.ADMISSION))
        self.assertIn("GGML_N_TASKS_MAX, const_cast<float *>(eps)", result)
        self.assertIn("ggml_map_custom3(ctx, input, gain, residual", result)
        self.assertIn("gain->ne[1] != 1", result)
        self.assertIn("value->ne[1] > 8192", result)
        self.assertIn("(size_t(-1) - 512 * sizeof(float)) / value->nb[1]", result)

    def test_scale_gelu_and_gpu_fallback_unchanged(self):
        result = TOOL.transform(self.source, self.component)
        for line in ['            x = ggml_mul(ctx, x, weight(p + "layer_output_scale.weight", 1, 1));',
                     '            x = ggml_mul(ctx, ggml_gelu(ctx, linear(ctx, x, p + "inp_gate")), layer_input);']:
            self.assertEqual(result.count(line), self.source.count(line))
        self.assertIn("if (!result) return ggml_add(ctx, rms(ctx, input, name), residual);", result)
        self.assertNotIn("#include", result[result.index(TOOL.BEGIN):result.index(TOOL.END)])
        self.assertNotIn("atomic", TOOL.FACTORY)

    def test_identity_rejects_stale_component_and_source(self):
        for source, component in [(self.source+"\n", self.component), (self.source, self.component+"\n")]:
            with self.assertRaisesRegex(ValueError, "exact reviewed"):
                TOOL.transform(source, component)

    def test_all_anchors_fail_closed_even_with_explicit_new_identity(self):
        for anchor in [*TOOL.CALLS, "struct source_file {\n", "    bool cpu_flat_qkv_rms = false;\n",
                       "        bool packed = false;\n", "    void validate() const {\n",
                       "            if (cpu_profile_plan::enabled()) {\n"]:
            for changed in [self.source.replace(anchor, "", 1), self.source+anchor]:
                digest = hashlib.sha256(changed.encode()).hexdigest()
                with patch.object(TOOL, "SOURCE_SHA", digest), self.assertRaises(ValueError):
                    TOOL.transform(changed, self.component)

    def test_reapplication_is_rejected(self):
        result = TOOL.transform(self.source, self.component)
        with patch.object(TOOL, "SOURCE_SHA", hashlib.sha256(result.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "not repeatable"):
            TOOL.transform(result, self.component)

    def test_default_mac_builder_byte_equivalence(self):
        builder = TOOL.builder()
        signature = inspect.signature(builder.build_candidate)
        self.assertIsNone(signature.parameters["source_transform"].default)
        self.assertIsNone(signature.parameters["scope"].default)
        self.assertEqual(signature.parameters["additional_guard_paths"].default, ())
        current = TOOL.BUILDER.read_text()
        original = current.replace("def build_candidate(args, *, source_transform=None, scope=None, additional_guard_paths=()):", "def build_candidate(args):")
        original = original.replace("candidate = (scoped_source if source_transform is None else source_transform)(original)", "candidate = scoped_source(original)")
        original = original.replace("    paths.update(Path(path).absolute() for path in additional_guard_paths)\n", "")
        original = original.replace('"scope": scope if scope is not None else ', '"scope": ')
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), "187dcc621464aed68dbc9af979709d026556da25720c1d88a8cceebc9c420254")


if __name__ == "__main__":
    unittest.main()
