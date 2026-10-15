import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "embeddinggemma2-cpu-phase-summary.py"
SPEC = importlib.util.spec_from_file_location("cpu_phases", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record():
    return dict(diagnostic=True, split_graph=True, step=1, length=32, batch=1, sequences=1,
        nodes=12, segments=6, wall_ms=11.2,
        category_ms=dict(projection=6., attention=4., other=1.),
        category_nodes=dict(projection=4, attention=2, other=6),
        category_segments=dict(projection=2, attention=2, other=2))


class CpuPhaseSummaryTests(unittest.TestCase):
    def test_accounting_and_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "profile.log"
            first, second = record(), record()
            second["step"] = 2
            second["category_ms"]["projection"] = 5.
            log.write_text("ordinary log\n" + "\n".join(MODULE.PREFIX + json.dumps(r) for r in (first, second)))
            result = MODULE.summarize([log], 1)
            group = result["groups"][0]
            self.assertEqual(group["selected_steps"], 1)
            self.assertEqual(group["categories"]["projection"]["median_ms"], 5.)
            self.assertAlmostEqual(sum(c["fraction"] for c in group["categories"].values()), 1.)

    def test_rejects_invalid_accounting(self):
        mutations = [
            lambda r: r.update(diagnostic=False),
            lambda r: r.update(split_graph=False),
            lambda r: r.update(length=0),
            lambda r: r.update(nodes=13),
            lambda r: r.update(segments=7),
            lambda r: r.update(wall_ms=10),
            lambda r: r["category_ms"].update(attention=float("nan")),
            lambda r: r["category_segments"].update(attention=-1),
            lambda r: r["category_ms"].update(missing=0),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value = copy.deepcopy(record())
                mutate(value)
                with self.assertRaises(ValueError):
                    MODULE.check(value)

    def test_rejects_abort_missing_and_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "profile.log"
            for text, discard in (("ordinary", 0), ("EMBEDDING_GGML_CPU_PROFILE_ABORT status=-1", 0),
                                  (MODULE.PREFIX + json.dumps(record()), 1)):
                log.write_text(text)
                with self.assertRaises(ValueError):
                    MODULE.summarize([log], discard)


if __name__ == "__main__":
    unittest.main()
