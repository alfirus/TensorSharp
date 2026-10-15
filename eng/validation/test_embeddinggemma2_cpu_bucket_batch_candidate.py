"""Source/isolation and layout controls, not native or model qualification."""
import importlib.util
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "bucket_candidate", Path(__file__).with_name("embeddinggemma2-cpu-bucket-batch-candidate.py"))
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def original_source():
    return ("e->cpu_flat_qkv_rms = validated_cpu_geometry && workers;\n" +
            tool.DECLARATION +
            "        for (int i = 0; i < batch; ++i) {\n" +
            '            require(lengths[i] > 0 && lengths[i] <= context, "invalid");\n' +
            tool.LENGTH_UPDATE + "        }\n" + tool.PACKED +
            "            (gemma2 && !ggml_backend_is_cpu(backend) && total <= 1024));\n")


class BucketCandidateControls(unittest.TestCase):
    def test_only_three_scoped_replacements(self):
        source = original_source()
        candidate = tool.scoped_source(source)
        restored = candidate.replace(tool.BUCKET_DECLARATION, tool.DECLARATION).replace(
            tool.BUCKET_UPDATE, tool.LENGTH_UPDATE).replace(tool.BUCKET_PACKED, tool.PACKED)
        self.assertEqual(restored, source)
        self.assertIn("cpu_flat_qkv_rms && cpu_float_linears && cpu_alignment == 16 && pooling == 1", candidate)
        self.assertIn("bool same_cpu_bucket = false;\n#if defined(__linux__) && defined(__x86_64__)\n", candidate)

    def test_actual_current_owned_source_can_be_transformed(self):
        source = Path(__file__).resolve().parents[2] / "TensorSharp.GGML.Native/ggml_ops_embeddings.cpp"
        candidate = tool.scoped_source(source.read_text())
        self.assertIn(tool.BUCKET_PACKED, candidate)

    def test_already_staged_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "already present"):
            tool.scoped_source(tool.scoped_source(original_source()))

    def test_missing_or_ambiguous_anchor_rejected(self):
        for anchor in (tool.DECLARATION, tool.LENGTH_UPDATE, tool.PACKED):
            for source in (original_source().replace(anchor, ""), original_source() + anchor):
                with self.subTest(anchor=anchor, source=source):
                    with self.assertRaisesRegex(ValueError, "exactly one"):
                        tool.scoped_source(source)

    def test_missing_qualified_geometry_rejected(self):
        with self.assertRaisesRegex(ValueError, "qualified CPU"):
            tool.scoped_source(original_source().replace("cpu_flat_qkv_rms = validated_cpu_geometry", "other"))

    def test_unvalidated_or_reordered_lengths_rejected(self):
        source = original_source()
        require_line = '            require(lengths[i] > 0 && lengths[i] <= context, "invalid");\n'
        for changed in (source.replace(require_line, ""),
                        source.replace(require_line + tool.LENGTH_UPDATE, tool.LENGTH_UPDATE + require_line)):
            with self.assertRaisesRegex(ValueError, "validated|validation"):
                tool.scoped_source(changed)

    def test_build_uses_same_guarded_isolation_and_explicit_scope(self):
        args = object()
        with patch.object(tool.BUILD, "build_candidate", return_value={"sentinel": True}) as build:
            self.assertEqual(tool.build_candidate(args), {"sentinel": True})
        build.assert_called_once_with(args, source_transform=tool.scoped_source, scope=tool.SCOPE)

    def test_same_bucket_preserves_flat_rows_positions_masks_and_real_mean(self):
        # Exhaust every short length pair and representative 32-stream/tail
        # layouts. This is exact layout algebra, not simulated F32 accuracy.
        cases = [[a, b] for a in range(1, 17) for b in range(1, 17)]
        cases += [[13] * 10 + [14] * 22, [17, 31, 32], [8177, 8191, 8192]]
        for lengths in cases:
            bucket = min(8192, (max(lengths) + 15) // 16 * 16)
            old_sizes = [min(8192, (n + 15) // 16 * 16) for n in lengths]
            self.assertTrue(all(size == bucket for size in old_sizes))
            old_rows = [(s, q if q < n else None) for s, (n, size) in enumerate(zip(lengths, old_sizes)) for q in range(size)]
            new_rows = [(s, q if q < n else None) for s, n in enumerate(lengths) for q in range(bucket)]
            self.assertEqual(old_rows, new_rows)
            flat_values = [Fraction(11 * s + q * q - 3, 7)
                           for s, n in enumerate(lengths) for q in range(bucket)]
            start = 0
            for stream, n in enumerate(lengths):
                for q in (0, n - 1, bucket - 1):
                    # The old segmented mask addresses a slice in the flat
                    # stream; the new mask addresses a 4D batch plane.
                    keys_old = list(range(start + max(0, q - 512),
                                          start + min(n, q + 513)))
                    keys_new = [stream * bucket + k for k in range(bucket)
                                if k < n and abs(q - k) <= 512]
                    if not keys_old:
                        keys_old = [start]
                    if not keys_new:
                        keys_new = [stream * bucket]
                    self.assertEqual(keys_old, keys_new)
                # Old mean pooling multiplies the entire flat vector by a
                # sparse row. New pooling multiplies one contiguous stream.
                packed = sum((value / n if start <= i < start + n else Fraction())
                             for i, value in enumerate(flat_values))
                independent = sum(flat_values[stream * bucket:stream * bucket + n], Fraction()) / n
                self.assertEqual(independent, packed)
                start += old_sizes[stream]


if __name__ == "__main__":
    unittest.main()
