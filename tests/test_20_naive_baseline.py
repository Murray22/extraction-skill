"""Script 20 must compare the model against a baseline that never saw the held-out season.

The abstract says a player's prior raw draw rate predicts held-out totals as well as the full
model (0.72 both). That sentence is only honest if (a) the numbers reproduce from the committed
tables and (b) the baseline's one tuning constant, K, is chosen without the held-out season.

Run from the repo root:  python -m unittest discover -s tests   (or: python -m pytest tests)
Needs no database: script 20 reads only data/derived/.
"""
import importlib.util
import os
import subprocess
import sys
import unittest

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(HERE, "..", "analysis")
SCRIPT = os.path.join(ANALYSIS, "20_naive_baseline.py")
DERIVED = os.path.join(HERE, "..", "data", "derived")

spec = importlib.util.spec_from_file_location("s20", SCRIPT)
s20 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s20)


def _playergames():
    pg = pd.read_parquet(os.path.join(DERIVED, "l1_playergames.parquet"))
    pg["season"] = pg.season.astype(int)
    return pg


class HeadlineNumbers(unittest.TestCase):
    def test_output_matches_the_docstring(self):
        out = subprocess.run([sys.executable, "-W", "ignore", SCRIPT], cwd=ANALYSIS, capture_output=True, text=True, check=True).stdout
        for line in ("K chosen on the fit seasons: 15 hours",
                     "full model             r 0.723  deviance  969",
                     "prior raw rate (F/D)   r 0.721  deviance  974",
                     "model minus prior raw rate: +0.002, 95% CI [-0.003, +0.008]",
                     "forwards (n=392): ice time 0.457, context 0.459, model 0.713, prior raw rate 0.711"):
            self.assertIn(line, out)
            self.assertIn(line, s20.__doc__)


class KIsChosenWithoutTheHeldOutSeason(unittest.TestCase):
    def test_changing_the_held_out_season_does_not_move_k(self):
        pg = _playergames()
        k, dev = s20.choose_k(pg)
        spoiled = pg.copy()
        held = ~spoiled.season.isin(s20.TRAIN_SEASONS)
        self.assertGreater(held.sum(), 0)
        spoiled.loc[held, "drawn"] = spoiled.loc[held, "drawn"] * 10 + 3
        k2, dev2 = s20.choose_k(spoiled)
        self.assertEqual((k, dev), (k2, dev2))

    def test_changing_a_fit_season_does_move_the_deviances(self):
        """The check above can fail: choose_k does read the seasons it is supposed to read."""
        pg = _playergames()
        _, dev = s20.choose_k(pg)
        spoiled = pg.copy()
        fit = spoiled.season == s20.TRAIN_SEASONS[1]
        spoiled.loc[fit, "drawn"] = spoiled.loc[fit, "drawn"] * 10 + 3
        _, dev2 = s20.choose_k(spoiled)
        self.assertNotEqual(dev, dev2)


class PlayersWithNoRecord(unittest.TestCase):
    def test_a_player_with_no_fit_season_gets_his_position_average(self):
        pg = _playergames()
        oos = pd.read_csv(os.path.join(DERIVED, "l1_oos.csv"))
        m = s20.naive_predictions(pg, oos, 15)
        new = m[m.h == 0]
        self.assertEqual(len(new), 26)
        self.assertEqual(new.groupby("fwd").prior_rate.nunique().max(), 1)
        self.assertTrue((m.pred_naive > 0).all())


if __name__ == "__main__":
    unittest.main()
