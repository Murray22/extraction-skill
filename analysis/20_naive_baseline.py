#!/usr/bin/env python3
"""Naive baseline for the out-of-sample ladder: does the model beat the stat that already exists?

The ladder in 01 compares the skill model against ice time alone (0.250) and the context model
(0.527). Neither is what a reader already has: a player's own penalties drawn per hour, which
public stats sites have carried for years. This script predicts the same held-out 2025-26 draw
totals, for the same 606 players, from that prior raw rate (2023-24 + 2024-25), pulled toward the
forward or defence average when the sample is small.

The pull strength K (hours of league-average play added to each player's record) is chosen on the
two fit seasons only: 2023-24 rates predicting 2024-25 totals, lowest Poisson deviance. The
held-out season plays no part in choosing it.

Paper sections: 5 (validation). Reads only committed tables, so it runs without any database:
  ../data/derived/l1_playergames.parquet, ../data/derived/l1_oos.csv

Expected output (2026-10-01):
  K chosen on the fit seasons: 15 hours
  ice time only          r 0.250  deviance 1867
  context model          r 0.527  deviance 1379
  full model             r 0.723  deviance  969
  prior raw rate (F/D)   r 0.721  deviance  974
  model minus prior raw rate: +0.002, 95% CI [-0.003, +0.008]; the two predictions correlate 0.996
  forwards (n=392): ice time 0.457, context 0.459, model 0.713, prior raw rate 0.711
  defence  (n=214): ice time 0.413, context 0.417, model 0.609, prior raw rate 0.605
  held-out draws per hour by quartile of prior raw rate: 0.32 / 0.46 / 0.66 / 0.89

Reading: the context adjustments are nearly all position (within forwards, context adds 0.002 to
ice time), and the full model predicts no better than the shrunken raw rate. The skill claim does
not need the model; the model's job is the isolation tests (02, 09, L2b in 01), not prediction.
"""
import numpy as np, pandas as pd

DERIVED = "../data/derived"   # run from analysis/, like 01 and 05
TRAIN_SEASONS = (20232024, 20242025)   # 01's fit window; 2025-26 is the held-out season
K_GRID = (2, 5, 10, 15, 20, 30, 40, 60)
N_BOOT, SEED = 4000, 0


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def deviance(y, mu):
    """Poisson deviance, the loss 01 reports."""
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-9, None)
    return float(2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu)))


def totals(pg, seasons):
    """Drawn minors and EV hours per player over the given seasons, with forward/defence flag."""
    t = pg[pg.season.isin(seasons)].groupby('player_id').agg(d=('drawn', 'sum'), h=('hours', 'sum')).reset_index()
    t['fwd'] = t.player_id.map(pg.groupby('player_id').pos.first()).ne('D')
    return t


def shrunk_rate(d, h, fwd, group_rate, k):
    """Raw draws per hour pulled toward the player's forward/defence rate by k hours of average play."""
    return (d + fwd.map(group_rate) * k) / (h + k)


def choose_k(pg, grid=K_GRID):
    """Pick K using the fit seasons only: first fit season's rate predicting the second's totals."""
    first, second = TRAIN_SEASONS
    a, b = totals(pg, [first]), totals(pg, [second])
    group_rate = a.groupby('fwd').apply(lambda g: g.d.sum() / g.h.sum())
    m = b.merge(a[['player_id', 'd', 'h']], on='player_id', how='left', suffixes=('', '_prior')).fillna({'d_prior': 0, 'h_prior': 0})
    dev = {k: deviance(m.d, shrunk_rate(m.d_prior, m.h_prior, m.fwd, group_rate, k) * m.h) for k in grid}
    return min(dev, key=dev.get), dev


def naive_predictions(pg, oos, k):
    """Held-out totals predicted from each player's prior raw rate; players with no fit-season
    record get their position's average, as the model gives them theta = 1."""
    tr = totals(pg, TRAIN_SEASONS)
    group_rate = tr.groupby('fwd').apply(lambda g: g.d.sum() / g.h.sum())
    m = oos.merge(tr[['player_id', 'd', 'h']], on='player_id', how='left').fillna({'d': 0, 'h': 0})
    m['fwd'] = m.player_id.map(pg.groupby('player_id').pos.first()).ne('D')
    m['prior_rate'] = shrunk_rate(m.d, m.h, m.fwd, group_rate, k)
    m['pred_naive'] = m.prior_rate * m.hours
    return m


def main():
    pg = pd.read_parquet(f"{DERIVED}/l1_playergames.parquet"); pg['season'] = pg.season.astype(int)
    oos = pd.read_csv(f"{DERIVED}/l1_oos.csv")
    k, _ = choose_k(pg)
    print(f"K chosen on the fit seasons: {k} hours")
    m = naive_predictions(pg, oos, k)
    rows = [('ice time only', 'pred_flat'), ('context model', 'pred_ctx'), ('full model', 'pred_model'), ('prior raw rate (F/D)', 'pred_naive')]
    for label, col in rows:
        print(f"  {label:<22} r {corr(m.drawn, m[col]):.3f}  deviance {deviance(m.drawn, m[col]):4.0f}")
    rng = np.random.default_rng(SEED); y, a, b = m.drawn.values, m.pred_model.values, m.pred_naive.values
    diffs = [corr(y[i], a[i]) - corr(y[i], b[i]) for i in (rng.integers(0, len(m), len(m)) for _ in range(N_BOOT))]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  model minus prior raw rate: {corr(y, a) - corr(y, b):+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]; "
          f"the two predictions correlate {corr(a, b):.3f}")
    for label, sub in (('forwards', m[m.fwd]), ('defence ', m[~m.fwd])):
        print(f"  {label} (n={len(sub)}): ice time {corr(sub.drawn, sub.pred_flat):.3f}, context {corr(sub.drawn, sub.pred_ctx):.3f}, "
              f"model {corr(sub.drawn, sub.pred_model):.3f}, prior raw rate {corr(sub.drawn, sub.pred_naive):.3f}")
    q = pd.qcut(m.prior_rate, 4, labels=False)
    rates = m.groupby(q).apply(lambda g: g.drawn.sum() / g.hours.sum())
    print("  held-out draws per hour by quartile of prior raw rate: " + " / ".join(f"{r:.2f}" for r in rates))


if __name__ == "__main__":
    main()
