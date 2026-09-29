#!/usr/bin/env python3
"""Goals path for a drawn EV minor: goals in the 2-minute window after the call, against a
computed even-strength baseline (paper §6 triangulation; replaces the uncomputed "~8.5%").

The +0.118 goals/drawn minor cited through 2026-09-29 was 20.3% (drawing team scores in the
window) minus a "~8.5%" EV baseline that no script ever computed (hockey wiki note
CONTACT_WON_METRIC_TRIALS_2026-09-01, "v2 addendum"). This script computes both sides.

Minors: the same set as 03 (regular season, 2-minute penalty, manpower 5v5/4v4/3v3 at the call,
game_seconds <= 3600). Window: goals with game_seconds in (t, t+120]. Main estimate uses windows
that end inside regulation (t <= 3480) so every window is a full 120 s; the all-minors
conversion is printed too, to tie back to the 20.3%.
Baseline: each team's goals per second of even-strength (5v5/4v4/3v3) regulation time, by score
difference (clipped +/-3, from that team's side) and 10-minute bucket, the strata 03 uses; the
state of a goal is the goal row's own manpower_state (the state it was scored in); time in a
state is the gap to the next game_state row. Expected baseline goals = rate x 120 s.

Expected output (primary DB, regular seasons 2023-24..2025-26; run 2026-09-29):
  EV baseline: 3,213 EV hours, 16,211 EV goals -> 0.0841 goals per team per 120 s, P(>=1) 8.1%
  all 20,224 EV minors (03's n): drawing team scores in the window 20.3%
  full-window minors n=19,807: P(score) 20.5% vs state-matched baseline 8.2% (+0.123)
    goals for     +0.1266 +/- 0.0059 over baseline   <- replaces "+0.118"
    goals against -0.0367 +/- 0.0031 (shorthanded side scores less)
    NET goals     +0.1632 +/- 0.0068 per drawn minor
The goals->wins step (~0.16 wins/goal, i.e. ~6 goals per win) is a rule of thumb, not computed
here: goals-for x 0.16 = +0.020 and net x 0.16 = +0.026 wins, against 03's measured +0.0171.
Writes ../data/derived/l3_goals_path_windows.parquet (one row per minor window).
Requires: gamevibe_primary.duckdb.
"""
import duckdb, numpy as np, pandas as pd

DB = "/home/steve_murray/projects/GameVibe/hockey/data/active_db/gamevibe_primary.duckdb"
OUT_DIR = "../data/derived"
SEASONS = ('20232024', '20242025', '20252026')   # pinned: the live DB grows each season
EV = ('5v5', '4v4', '3v3')
W = 120

con = duckdb.connect(DB, read_only=True)
games = f"SELECT game_id FROM games_metadata WHERE season_type='2' AND season IN {SEASONS}"

# every game_state row in order, with the next row's time (for durations)
gs = con.execute(f"""
  SELECT gs.game_id, gs.game_seconds AS t, gs.manpower_state AS mp,
         gs.home_score AS hs, gs.away_score AS as_, LOWER(e.event_type) AS et, e.event_team_id,
         gm.home_team_abbr, gm.away_team_abbr, p.event_team_abbr, e.sort_order,
         LEAD(gs.game_seconds) OVER (PARTITION BY gs.game_id ORDER BY e.sort_order) AS t_next
  FROM game_state gs
  JOIN events e ON e.game_id=gs.game_id AND e.event_id=gs.event_id
  JOIN games_metadata gm ON CAST(gs.game_id AS VARCHAR)=gm.game_id
  LEFT JOIN play_by_play_raw p ON p.game_id=CAST(gs.game_id AS VARCHAR) AND TRY_CAST(p.event_id AS INT)=gs.event_id
  WHERE CAST(gs.game_id AS VARCHAR) IN ({games})""").df()
gs = gs.sort_values(['game_id', 'sort_order']).reset_index(drop=True)
gs['hs_prev'] = gs.groupby('game_id').hs.shift(1); gs['as_prev'] = gs.groupby('game_id').as_.shift(1)

# ---- baseline: EV goals per team-second, by (diff from the team's side, 10-min bucket)
reg = gs[(gs.t < 3600)].copy()
reg['dur'] = (reg.t_next.clip(upper=3600) - reg.t).clip(lower=0).fillna(0)
reg['hd'] = (reg.hs - reg.as_)
ev = reg[reg.mp.isin(EV)]
tb = (ev.t // 600).clip(0, 5)
secs = pd.concat([
    pd.DataFrame({'diff': ev.hd.clip(-3, 3), 'tb': tb, 'dur': ev.dur}),
    pd.DataFrame({'diff': (-ev.hd).clip(-3, 3), 'tb': tb, 'dur': ev.dur})]).groupby(['diff', 'tb']).dur.sum()
# goals: the goal row carries the post-goal score in some rows; use the score before it
gl = gs[(gs.et == 'goal') & gs.mp.isin(EV) & (gs.t <= 3600)].copy()
gl['home_scored'] = gl.event_team_abbr == gl.home_team_abbr
gl['pre_hd'] = gl.hs_prev.fillna(0) - gl.as_prev.fillna(0)
gl['diff'] = np.where(gl.home_scored, gl.pre_hd, -gl.pre_hd).clip(-3, 3)
gl['tb'] = (gl.t // 600).clip(0, 5)
goals = gl.groupby(['diff', 'tb']).size()
rate = (goals / secs).fillna(0).rename('rate').reset_index()   # goals per team-second
lam_all = len(gl) / (2 * ev.dur.sum())
print(f"EV regulation time {ev.dur.sum()/3600:,.0f} h, EV goals {len(gl):,}: "
      f"{lam_all*W:.4f} goals per team per 120 s -> P(>=1) {1-np.exp(-lam_all*W):.1%}")

# ---- minors (03's selection) and their windows
mn = con.execute(f"""
  SELECT p.game_id, e.game_seconds AS t, TRY_CAST(p.home_score AS INT) AS hs, TRY_CAST(p.away_score AS INT) AS as_,
         CASE WHEN p.event_team_abbr = gm.home_team_abbr THEN 0 ELSE 1 END AS home_drew,
         gm.home_team_abbr, gm.away_team_abbr
  FROM play_by_play_raw p
  JOIN game_state gs ON p.game_id=CAST(gs.game_id AS VARCHAR) AND TRY_CAST(p.event_id AS INTEGER)=gs.event_id
  JOIN events e ON CAST(e.game_id AS VARCHAR)=p.game_id AND CAST(e.event_id AS VARCHAR)=p.event_id
  JOIN games_metadata gm ON p.game_id=gm.game_id
  WHERE LOWER(p.event_type)='penalty' AND gs.manpower_state IN {EV}
    AND COALESCE(TRY_CAST(p.penalty_minutes AS DOUBLE), e.penalty_minutes)=2
    AND gm.season_type='2' AND gm.season IN {SEASONS}
    AND e.game_seconds IS NOT NULL AND e.game_seconds<=3600 AND p.home_score IS NOT NULL""").df()
mn['i'] = np.arange(len(mn))
g = gs[gs.et == 'goal'][['game_id', 't', 'event_team_abbr', 'home_team_abbr']].copy()
g['game_id'] = g.game_id.astype(str); g['home_goal'] = (g.event_team_abbr == g.home_team_abbr).astype(int)
j = mn[['i', 'game_id', 't', 'home_drew']].merge(g[['game_id', 't', 'home_goal']], on='game_id', suffixes=('', '_g'))
j = j[(j.t_g > j.t) & (j.t_g <= j.t + W)]
j['gf'] = (j.home_goal == j.home_drew).astype(int); j['ga'] = 1 - j.gf
cnt = j.groupby('i')[['gf', 'ga']].sum()
mn = mn.join(cnt, on='i').fillna({'gf': 0, 'ga': 0})
dd = np.where(mn.home_drew == 1, mn.hs - mn.as_, mn.as_ - mn.hs)
mn['diff'] = np.clip(dd, -3, 3); mn['tb'] = (mn.t // 600).clip(0, 5)
mn = mn.merge(rate, on=['diff', 'tb'], how='left').rename(columns={'rate': 'r_f'})
mn = mn.merge(rate.assign(diff=-rate['diff']), on=['diff', 'tb'], how='left').rename(columns={'rate': 'r_a'})
mn['base_gf'] = mn.r_f * W; mn['base_ga'] = mn.r_a * W

print(f"\nall {len(mn):,} EV minors: drawing team scores in the window {(mn.gf > 0).mean():.1%}")
full = mn[mn.t <= 3600 - W].copy()
n = len(full)
def est(x):
    return x.mean(), 1.96 * x.std() / np.sqrt(len(x))
p_obs = (full.gf > 0).mean(); p_base = (1 - np.exp(-full.base_gf)).mean()
print(f"full-window minors (t<=3480): n={n:,}")
print(f"  P(drawing team scores): {p_obs:.1%} vs state-matched EV baseline {p_base:.1%} -> {p_obs-p_base:+.3f}")
m, ci = est(full.gf - full.base_gf)
print(f"  goals for:     {full.gf.mean():.4f} vs baseline {full.base_gf.mean():.4f} -> {m:+.4f} +/- {ci:.4f}")
m2, ci2 = est(full.ga - full.base_ga)
print(f"  goals against: {full.ga.mean():.4f} vs baseline {full.base_ga.mean():.4f} -> {m2:+.4f} +/- {ci2:.4f}")
m3, ci3 = est((full.gf - full.ga) - (full.base_gf - full.base_ga))
print(f"  NET goals per drawn minor: {m3:+.4f} +/- {ci3:.4f} (95% CI)")
print(f"  unmatched league baseline {lam_all*W:.4f}/team: goals-for {full.gf.mean()-lam_all*W:+.4f}, "
      f"net {(full.gf-full.ga).mean():+.4f}")
full[['game_id', 't', 'home_drew', 'diff', 'tb', 'gf', 'ga', 'base_gf', 'base_ga']].to_parquet(
    f"{OUT_DIR}/l3_goals_path_windows.parquet", index=False)
