#!/usr/bin/env python3
"""Market tests (paper §7.3 pre-salary): orthogonality to scoring, the baseline deployment
regression, and the embellishment mechanism check (paper §7.1 garnish).
Zone-start-controlled deployment variant: 08_deployment_zone_start_control.py.

Expected headline output (verified 2026-09-01/02):
  corr(theta2, points/60) = 0.067 (n=685, 15+ EV hrs)
  deployment: theta2 -0.279 min/game per SD, t=-3.65 (n=726, 60+ GP)
  embellishment: top-quartile drawers ~8x bottom (0.29 vs 0.04 mean count);
  HONESTY NOTE (post-swarm): among HIGH-VOLUME drawers (60+ drawn), 14/22 have zero calls
  and the group rate is 0.85/100 drawn — a single call is uninformative; never use N=1 as
  exoneration for any individual player.
"""
import duckdb, numpy as np, pandas as pd

DB = "/home/steve_murray/projects/GameVibe/hockey/data/active_db/gamevibe_primary.duckdb"
THETA_CSV = "../data/derived/l2b_theta2.csv"

con = duckdb.connect(DB, read_only=True)
theta = pd.read_csv(THETA_CSV, index_col=0)
pts = con.execute("""
  SELECT pid, COUNT(*) AS pts FROM (
    SELECT TRY_CAST(player_id_1 AS BIGINT) AS pid FROM events WHERE LOWER(event_type)='goal'
    UNION ALL SELECT TRY_CAST(player_id_2 AS BIGINT) FROM events WHERE LOWER(event_type)='goal' AND player_id_2 IS NOT NULL
    UNION ALL SELECT TRY_CAST(player_id_3 AS BIGINT) FROM events WHERE LOWER(event_type)='goal' AND player_id_3 IS NOT NULL)
  WHERE pid IS NOT NULL GROUP BY 1""").df().set_index('pid')
toi = con.execute("""
  SELECT player_id, SUM(toi_seconds)/3600.0 AS hrs, COUNT(DISTINCT game_id) AS gp,
         ANY_VALUE(position_code) AS pos
  FROM player_state_stats pss JOIN games_metadata gm USING(game_id)
  WHERE gm.season_type='2' AND position_code<>'G' GROUP BY 1""").df().set_index('player_id')

d = theta.join(pts).join(toi, how='inner')
d['pts'] = d.pts.fillna(0)

# --- orthogonality ---
o = d[d.hrs > 15].copy(); o['pts60'] = o.pts/o.hrs
print(f"orthogonality: corr(theta2, pts/60) = {np.corrcoef(o.theta2,o.pts60)[0,1]:.3f} (n={len(o)})")
for pos in ('C','L','R','D'):
    s = o[o.pos==pos]
    print(f"  {pos}: r={np.corrcoef(s.theta2,s.pts60)[0,1]:+.3f} (n={len(s)})")

# --- deployment baseline ---
m = d[d.gp >= 60].copy(); m['toi_pg'] = m.hrs*60/m.gp; m['pts60'] = m.pts/m.hrs
X = pd.get_dummies(m[['pos']], drop_first=True).astype(float)
X['pts60'] = (m.pts60-m.pts60.mean())/m.pts60.std()
X['theta2'] = (m.theta2-m.theta2.mean())/m.theta2.std()
X['const'] = 1.0
y = m.toi_pg.values
beta,_,_,_ = np.linalg.lstsq(X.values, y, rcond=None)
resid = y - X.values@beta; n,k = X.shape
cov = (resid@resid/(n-k))*np.linalg.inv(X.values.T@X.values); se = np.sqrt(np.diag(cov))
i = list(X.columns).index('theta2')
print(f"\ndeployment (n={n}): theta2 {beta[i]:+.3f} min/game per SD (t={beta[i]/se[i]:+.2f})")

# --- embellishment ---
emb = con.execute("""
  SELECT TRY_CAST(p.penalized_player_id AS BIGINT) AS pid, COUNT(*) AS emb
  FROM play_by_play_raw p JOIN games_metadata gm ON p.game_id=gm.game_id
  WHERE LOWER(p.event_type)='penalty' AND LOWER(COALESCE(p.penalty_type,'')) LIKE '%embell%'
    AND gm.season_type='2' GROUP BY 1""").df().set_index('pid')
e = theta.join(emb, how='left'); e['emb'] = e.emb.fillna(0)
hi = e[e.theta2 > e.theta2.quantile(.75)]; lo = e[e.theta2 < e.theta2.quantile(.25)]
print(f"\nembellishment: total={int(e.emb.sum())}; top-quartile mean {hi.emb.mean():.3f} vs bottom {lo.emb.mean():.3f}; corr={np.corrcoef(e.theta2,e.emb)[0,1]:.3f}")
print("(see docstring honesty note before quoting any individual player's count)")
