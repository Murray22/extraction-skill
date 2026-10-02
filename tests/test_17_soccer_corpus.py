"""Script 17 must know which competition every soccer match belongs to.

The corpus was described as "8 tournaments" until 2026-10-01. It is 13 competition-seasons, and
65% of the matches are women's club football, so a pooled board says little about the men's game.
The men's and women's results are only meaningful if every match is assigned.

Run from the repo root:  python -m unittest discover -s tests   (or: python -m pytest tests)
Needs no database: script 17 reads only data/derived/ and data/reference/.
"""
import importlib.util
import os
import tempfile
import unittest

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
spec = importlib.util.spec_from_file_location("s17", os.path.join(ROOT, "analysis", "17_soccer_replication_analysis.py"))
s17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s17)
SLIM = os.path.join(ROOT, "data", "derived", "soccer_events_slim.parquet")
COMPS = os.path.join(ROOT, "data", "reference", "soccer_match_competitions.csv")


class Corpus(unittest.TestCase):
    def setUp(self):
        s17.SLIM, s17.COMPS = SLIM, COMPS

    def test_composition(self):
        m = s17.load()
        n = m.groupby(["gender", "level"]).mid.nunique().to_dict()
        self.assertEqual(n, {("female", "club"): 771, ("female", "international"): 178, ("male", "international"): 230})
        self.assertEqual(m[["competition", "season"]].drop_duplicates().shape[0], 13)

    def test_a_match_with_no_competition_stops_the_run(self):
        comps = pd.read_csv(COMPS)
        with tempfile.TemporaryDirectory() as d:
            s17.COMPS = os.path.join(d, "comps.csv")
            comps.iloc[1:].to_csv(s17.COMPS, index=False)
            with self.assertRaises(SystemExit):
                s17.load()

    def test_published_numbers(self):
        m = s17.load()
        r, sb, n = s17.split_half(m)
        self.assertEqual((round(r, 3), round(sb, 3), n), (0.545, 0.705, 1397))
        r, sb, n = s17.split_half(m[m.gender == "male"])
        self.assertEqual((round(sb, 3), n), (0.636, 296))
        self.assertEqual(s17.board(m[m.gender == "male"]).nlargest(1, "ratio").name.iloc[0], "Eden Hazard")
        r, sb, n = s17.split_half(m[m.gender == "female"])
        self.assertEqual((round(sb, 3), n), (0.729, 1101))


if __name__ == "__main__":
    unittest.main()
