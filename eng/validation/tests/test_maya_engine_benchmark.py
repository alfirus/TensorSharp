"""Protocol failures must never turn into successful Maya benchmark evidence."""
import importlib.util
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "maya_engine_benchmark", Path(__file__).resolve().parents[1] / "run-maya-engine-benchmark.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class MayaBenchmarkProtocolTests(unittest.TestCase):
    def test_counts_and_reuse_are_checked(self):
        good = "DONE 2 4 20.0 50.0 length 0 0 1 10 12"
        result = BENCH.parse_done(good, [3, 4])
        self.assertEqual(result["native_fresh_prefill_tokens_per_second"], 150)
        for bad in (good.replace("DONE 2", "DONE 3"), good.replace("length", "cancel"),
                    good.replace("0 0 1 10", "0 0 5 10"), good.replace("20.0", "nan"),
                    "DONE malformed"):
            with self.subTest(record=bad), self.assertRaises(ValueError):
                BENCH.parse_done(bad, [3, 4])

    def test_actual_subprocess_protocol(self):
        program = (
            "import sys\n"
            "print('INFO loading', flush=True)\n"
            "print('READY 4096 stop', flush=True)\n"
            "for line in sys.stdin:\n"
            " if line == 'QUIT\\n': break\n"
            " assert line == 'GEN 2 temperature=0 spec=0 1,2,3,4\\n'\n"
            " print('PP 4 4 20 200 0', flush=True)\n"
            " print('T 3\\nSTAT pcie=0\\nT 4\\nDONE 2 4 20.0 50.0 length 0 0 0 10 12', flush=True)\n")
        with tempfile.TemporaryDirectory() as temporary:
            engine = BENCH.Engine([sys.executable, "-u", "-c", program], Path(temporary) / "engine.log", 5)
            try:
                engine.ready()
                result = engine.generate({"name": "sample", "ids": [1, 2, 3, 4], "max_new": 2})
                self.assertEqual(result["output_token_ids"], [3, 4])
                self.assertIsNotNone(result["time_to_first_token_ms"])
                self.assertEqual(result["finish_reason"], "length")
            finally:
                engine.close()

    def test_exited_engine_is_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = BENCH.Engine([sys.executable, "-c", "pass"], Path(temporary) / "engine.log", 5)
            try:
                with self.assertRaises(RuntimeError):
                    engine.ready()
            finally:
                engine.close()

    def test_capture_rows_are_verified_and_indexed(self):
        for size in (24, 20):
            with self.subTest(bytes=size), tempfile.TemporaryDirectory() as temporary:
                prefix = Path(temporary) / "logits"
                program = (
                    "import sys\nfrom pathlib import Path\nprint('READY 4096 stop', flush=True)\n"
                    "for line in sys.stdin:\n"
                    " if line == 'QUIT\\n': break\n"
                    f" Path({str(prefix)!r} + '.1.f32').write_bytes(bytes({size}))\n"
                    " print('T 3\\nT 4\\nDONE 2 4 20.0 50.0 length 0 0 0 10 12', flush=True)\n")
                engine = BENCH.Engine([sys.executable, "-u", "-c", program], Path(temporary) / "engine.log", 5,
                                      capture_prefix=prefix, vocab=3)
                try:
                    engine.ready()
                    case = {"name": "sample", "ids": [1, 2, 3, 4], "max_new": 2}
                    if size != 24:
                        with self.assertRaises(RuntimeError):
                            engine.generate(case)
                    else:
                        capture = engine.generate(case)["logits_capture"]
                        self.assertEqual((capture["request_index"], capture["row_count"], capture["vocab"]), (1, 2, 3))
                        self.assertEqual(capture["sha256"], hashlib.sha256(bytes(24)).hexdigest())
                finally:
                    engine.close()

    def test_timeout_is_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = BENCH.Engine([sys.executable, "-c", "import sys; sys.stdin.readline()"],
                                 Path(temporary) / "engine.log", .1)
            try:
                with self.assertRaises(TimeoutError):
                    engine.ready()
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
