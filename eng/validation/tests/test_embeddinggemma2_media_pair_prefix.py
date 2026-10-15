import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("media_pair_prefix", ROOT / "embeddinggemma2-media-pair.py")
PAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIR)


class MediaPairPrefixTests(unittest.TestCase):
    def test_original_llama_mode_only_prefixes_candidate(self):
        parts = [dict(kind="text", value="caption"), dict(kind="video", value="clip.mp4")]
        self.assertEqual([parts[0], dict(kind="text", value="Video:"), parts[1]],
                         PAIR.controlled_parts(parts, "tensorsharp", True))
        self.assertEqual(parts, PAIR.controlled_parts(parts, "llama", True))

    def test_two_tensor_sharp_roles_receive_identical_ordered_parts(self):
        parts = [dict(kind="image", value="red.png"), dict(kind="video", value="a.mp4"),
                 dict(kind="audio", value="tone.wav"), dict(kind="video", value="b.mp4")]
        original = [dict(p) for p in parts]
        candidate = PAIR.controlled_parts(parts, "tensorsharp", True, True)
        reference = PAIR.controlled_parts(parts, "llama", True, True)
        self.assertEqual(candidate, reference)
        self.assertEqual(["image", "text", "video", "audio", "text", "video"],
                         [p["kind"] for p in candidate])
        self.assertEqual(original, parts)

    def test_default_hf_mode_adds_no_prefix(self):
        parts = [dict(kind="video", value="clip.mp4")]
        for name in ["tensorsharp", "llama"]:
            self.assertEqual(parts, PAIR.controlled_parts(parts, name, False))

    def test_reference_prefix_accepts_only_resolved_direct_tensor_sharp_host(self):
        PAIR.validate_reference_prefix(True, True,
            ["env", "OMP_NUM_THREADS=8", "dotnet", "/snapshot/TensorSharp.Server.Host.dll"], {})
        for command in [["/llama-server"], ["bash", "-lc", "launch"], ["dotnet", "Other.dll"]]:
            with self.assertRaises(ValueError):
                PAIR.validate_reference_prefix(True, True, command, {})
        with self.assertRaises(ValueError):
            PAIR.validate_reference_prefix(False, True,
                ["dotnet", "/snapshot/TensorSharp.Server.Host.dll"], {})

    def test_original_mode_does_not_restrict_reference_command(self):
        PAIR.validate_reference_prefix(True, False, ["/llama-server"], {})


if __name__ == "__main__":
    unittest.main()
