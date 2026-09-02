#!/usr/bin/env python3
"""Empirical win-probability value of a drawn EV minor (paper §4.4, §6), plus the
taking-side mirror (reported as a consistency check, not independent evidence).

Expected headline output (verified 2026-09-01):
  Delta-WP per drawn EV minor: +0.0171 +/- 0.0056 (95% CI), n=20224
  by time bucket: rises from +0.006 (first 10 min) to ~+0.02 late
  by score diff: peaks at tied/-1, shrinks at +/-3
  taking-side mirror: -0.0171 (construction identity)
Goals-path triangulation (+0.118 goals/drawn minor: 20.3% PP-window conversion vs ~8.5% EV
baseline over 22,024 minors) is documented in METHODS.md (v2 addendum).
"""
import duckdb, numpy as np, pandas as pd

DB = "/home/steve_murray/projects/GameVibe/hockey/data/active_db/gamevibe_primary.duckdb"
con = duckdb.connect(DB, read_only=True)
res = con.execute("""
  SELECT p.game_id, MAX(TRY_CAST(p.home_score AS INT)) AS hs, MAX(TRY_CAST(p.away_score AS INT)) AS as_
  FROM play_by_play_raw p JOIN games_metadata gm ON p.game_id=gm.game_id
  WHERE gm.season_type='2' GROUP BY 1""").df()
res['home_win']=(res.hs>res.as_).astype(int)

def side(perspective_home_col: str, label: str):
    ev = con.execute(f"""
      SELECT p.game_id, e.game_seconds,
             TRY_CAST(p.home_score AS INT) AS hs, TRY_CAST(p.away_score AS INT) AS as_,
             CASE WHEN p.event_team_abbr = gm.home_team_abbr THEN {'0' if label=='drawing' else '1'} ELSE {'1' if label=='drawing' else '0'} END AS home_side
      FROM play_by_play_raw p
      JOIN game_state gs ON p.game_id=CAST(gs.game_id AS VARCHAR) AND TRY_CAST(p.event_id AS INTEGER)=gs.event_id
      JOIN events e ON CAST(e.game_id AS VARCHAR)=p.game_id AND CAST(e.event_id AS VARCHAR)=p.event_id
      JOIN games_metadata gm ON p.game_id=gm.game_id
      WHERE LOWER(p.event_type)='penalty' AND gs.manpower_state IN ('5v5','4v4','3v3')
        AND COALESCE(TRY_CAST(p.penalty_minutes AS DOUBLE), e.penalty_minutes)=2
        AND gm.season_type='2' AND e.game_seconds IS NOT NULL AND e.game_seconds<=3600
        AND p.home_score IS NOT NULL""").df()
    ev = ev.merge(res[['game_id','home_win']], on='game_id')
    ev['diff'] = np.where(ev.home_side==1, ev.hs-ev.as_, ev.as_-ev.hs).clip(-3,3)
    ev['won']  = np.where(ev.home_side==1, ev.home_win, 1-ev.home_win)
    ev['tb'] = (ev.game_seconds//600).clip(0,5)
    return ev

base = con.execute("""
  SELECT p.game_id, e.game_seconds, TRY_CAST(p.home_score AS INT) AS hs, TRY_CAST(p.away_score AS INT) AS as_
  FROM play_by_play_raw p
  JOIN events e ON CAST(e.game_id AS VARCHAR)=p.game_id AND CAST(e.event_id AS VARCHAR)=p.event_id
  JOIN games_metadata gm ON p.game_id=gm.game_id
  WHERE gm.season_type='2' AND e.game_seconds IS NOT NULL AND e.game_seconds<=3600
    AND p.home_score IS NOT NULL""").df().merge(res[['game_id','home_win']], on='game_id')
b1 = pd.DataFrame({'diff':(base.hs-base.as_).clip(-3,3),'tb':(base.game_seconds//600).clip(0,5),'won':base.home_win,'home':1})
b2 = pd.DataFrame({'diff':(base.as_-base.hs).clip(-3,3),'tb':(base.game_seconds//600).clip(0,5),'won':1-base.home_win,'home':0})
wp = pd.concat([b1,b2]).groupby(['diff','tb','home']).won.mean().rename('wp_base').reset_index()

for label in ('drawing','taking'):
    ev = side('home_side', label)
    ev = ev.merge(wp, left_on=['diff','tb','home_side'], right_on=['diff','tb','home'])
    d = ev.won - ev.wp_base
    print(f"{label} side: dWP = {d.mean():+.4f} +/- {1.96*d.std()/np.sqrt(len(d)):.4f} (n={len(d)})")
    if label=='drawing':
        print("  by time bucket:"); print(ev.groupby('tb').apply(lambda g:(g.won-g.wp_base).mean(),include_groups=False).round(4).to_string())
        print("  by score diff:"); print(ev.groupby('diff').apply(lambda g:(g.won-g.wp_base).mean(),include_groups=False).round(4).to_string())
