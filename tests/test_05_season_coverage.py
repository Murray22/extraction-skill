"""Script 05 must use only seasons whose penalties carry a drawing player.

The discovery DB's 2025-26 season has 0 of 9,692 penalty rows with a drawing_player_id and
shift data for 30% of games. Pooling it into the team lever manufactured r=0.708 (n=425) out of
one cluster of 32 taken-only team-seasons; the covered seasons 2010-11..2024-25 give r=0.450
(n=393) and a best-to-worst spread of 139 net minors.

Run from the repo root:  python -m unittest discover -s tests   (or: python -m pytest tests)
The end-to-end tests need the (non-public) discovery DB and are skipped without it.
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "analysis", "05_fifteen_season_replication.py")
DB = "/home/steve_murray/projects/GameVibe/hockey/data/gamevibe_discovery.duckdb"


def _sandbox():
    """A temp copy of the script with its own ../data/derived, so no tracked file is written."""
    root = tempfile.mkdtemp(prefix="s05_")
    os.makedirs(os.path.join(root, "analysis"))
    os.makedirs(os.path.join(root, "data", "derived"))
    dst = os.path.join(root, "analysis", os.path.basename(SCRIPT))
    shutil.copy(SCRIPT, dst)
    return root, dst


def _run(args=()):
    root, dst = _sandbox()
    try:
        return subprocess.run([sys.executable, dst, *args], cwd=os.path.dirname(dst),
                              capture_output=True, text=True, timeout=600)
    finally:
        shutil.rmtree(root, ignore_errors=True)


class SelectSeasonsRule(unittest.TestCase):
    """The rule itself, on synthetic coverage numbers (no DB needed)."""

    @classmethod
    def setUpClass(cls):
        root, dst = _sandbox()
        cls._root = root
        cwd = os.getcwd()
        os.chdir(os.path.dirname(dst))   # importing must not run the analysis; if it does, contain it
        try:
            spec = importlib.util.spec_from_file_location("s05", dst)
            cls.m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.m)
        finally:
            os.chdir(cwd)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._root, ignore_errors=True)

    def cov(self):
        return pd.DataFrame({
            "season": [20232024, 20242025, 20252026],
            "drawn_share": [0.904, 0.911, 0.0],
            "shift_share": [1.0, 0.957, 0.298],
        })

    def test_default_excludes_season_without_drawing_players(self):
        used, excluded = self.m.select_seasons(self.cov())
        self.assertEqual(used, [20232024, 20242025])
        self.assertEqual(list(excluded.season), [20252026])

    def test_explicitly_requested_uncovered_season_raises(self):
        with self.assertRaises(SystemExit) as cm:
            self.m.select_seasons(self.cov(), requested=[20242025, 20252026])
        self.assertIn("20252026", str(cm.exception))

    def test_requested_season_missing_from_db_raises(self):
        with self.assertRaises(SystemExit):
            self.m.select_seasons(self.cov(), requested=[20302031])

    def test_partial_shift_coverage_alone_excludes(self):
        cov = self.cov()
        cov.loc[cov.season == 20252026, "drawn_share"] = 0.92
        used, excluded = self.m.select_seasons(cov)
        self.assertNotIn(20252026, used)


@unittest.skipUnless(os.path.exists(DB), "needs gamevibe_discovery.duckdb")
class EndToEnd(unittest.TestCase):
    """The committed script, run on the real DB."""

    def test_default_run_drops_2025_26_and_gives_the_covered_team_lever(self):
        out = _run()
        self.assertEqual(out.returncode, 0, out.stderr)
        m = re.search(r"team lever: levels r=([\d.]+) \(n=(\d+)\)", out.stdout)
        self.assertIsNotNone(m, out.stdout)
        self.assertEqual((m.group(1), int(m.group(2))), ("0.450", 393))
        self.assertIn("best-to-worst net-minor spread = 139", out.stdout)
        self.assertRegex(out.stdout + out.stderr, r"EXCLUDED.*20252026")

    def test_requesting_2025_26_fails_loudly(self):
        out = _run(["--seasons", "20242025,20252026"])
        self.assertNotEqual(out.returncode, 0, out.stdout[-500:])
        self.assertIn("20252026", out.stderr)


if __name__ == "__main__":
    unittest.main()
