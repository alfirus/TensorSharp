"""Regressions for the pinned GLUE development TSV's literal quote semantics."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "embeddinggemma2_stsb", Path(__file__).resolve().parents[1] / "embeddinggemma2-stsb.py")
stsb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stsb)
FIXTURE = Path(__file__).with_name("fixtures") / "embeddinggemma2-stsb-quoted.tsv"


class DevelopmentTsvTests(unittest.TestCase):
    def test_official_unmatched_quote_preserves_pair_score_and_next_physical_row(self):
        pairs = stsb.read_development_pairs(FIXTURE.read_bytes(), expected_rows=2)
        self.assertEqual([row["index"] for row in pairs], ["764", "765"])
        self.assertTrue(pairs[0]["sentence1"].startswith('"The grass family'))
        self.assertEqual(pairs[0]["score"], "1.40")
        self.assertEqual(pairs[1]["score"], "1.20")
        self.assertIn('"HASTHINA"', pairs[1]["sentence2"])
        self.assertNotIn("\t", pairs[0]["sentence1"])
        self.assertNotIn("\n", pairs[0]["sentence1"])

    def test_wrong_row_count_cannot_claim_complete_development_coverage(self):
        with self.assertRaisesRegex(ValueError, "2 complete rows; expected 1500"):
            stsb.read_development_pairs(FIXTURE.read_bytes())

    def test_missing_score_reports_original_physical_line(self):
        lines = FIXTURE.read_text().splitlines()
        lines[1] = lines[1].rsplit("\t", 1)[0]
        with self.assertRaisesRegex(ValueError, "line 2: expected 10 tab-separated fields"):
            stsb.read_development_pairs("\n".join(lines).encode(), expected_rows=2)

    def test_nonfinite_or_out_of_range_score_is_rejected(self):
        source = FIXTURE.read_bytes()
        for score in (b"NaN", b"inf", b"-0.1", b"5.1", b"not-a-score"):
            with self.subTest(score=score):
                with self.assertRaisesRegex(ValueError, "line 2:.*gold score"):
                    stsb.read_development_pairs(source.replace(b"\t1.40\n", b"\t" + score + b"\n"), expected_rows=2)

    def test_coverage_label_distinguishes_complete_development_from_subset(self):
        self.assertEqual(stsb.selection_description(1500, 1500), "complete development split (1500 pairs)")
        self.assertEqual(stsb.selection_description(200, 1500), "first 200 rows in original file order")


if __name__ == "__main__":
    unittest.main()
