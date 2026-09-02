#!/usr/bin/env python3
"""Officiating by-products (paper §7.4): playoff whistle-swallowing test, skill transfer
into playoffs, make-up call sequencing, and close-and-late whistle timing.

Expected headline output (verified 2026-09-01/02):
  playoffs: EV minors 6.46/game vs 5.20 RS; per skater-TOI-unit +16% (0.648 vs 0.557/hr)
  skill transfer: RS theta -> playoff ratio r=0.348; top-quartile 1.38->1.352, bottom 0.64->0.660
  make-up: P(next minor on HOME) 0.380 after home penalty vs 0.584 after away
    (+0.204 swing, z=26.8); decays with gap (+0.221 at 2-5min -> +0.074 past 20);
    tied-score-only +0.217
  timing: draws in close-and-late = 0.137 of league draws vs 0.168 of play exposure (-12%
    relative); flat across drawer-skill quartiles (no clutch-timing skill)
"""
import duckdb, numpy as np, pandas as pd

DB = "/home/steve_murray/projects/GameVibe/hockey/data/active_db/gamevibe_primary.duckdb"
THETA_CSV = "../data/derived/l2b_theta2.csv"
con = duckdb.connect(DB, read_only=True)
theta = pd.read_csv(THETA_CSV, index_col=0)

PENS = """
  FROM play_by_play_raw p
  JOIN game_state gs ON p.game_id=CAST(gs.game_id AS VARCHAR) AND TRY_CAST(p.event_id AS INTEGER)=gs.event_id
  JOIN events e ON CAST(e.game_id AS VARCHAR)=p.game_id AND CAST(e.event_id AS VARCHAR)=p.event_id
  JOIN games_metadata gm ON p.game_id=gm.game_id
  WHERE LOWER(p.event_type)='penalty' AND gs.manpower_state IN ('5v5','4v4','3v3')
    AND COALESCE(TRY_CAST(p.penalty_minutes AS DOUBLE), e.penalty_minutes)=2"""

# ---- playoffs rate + transfer ----
r = con.execute(f"""SELECT gm.season_type, COUNT(DISTINCT p.game_id) g, COUNT(*) n {PENS}
  AND gm.season_type IN ('2','3') GROUP BY 1""").df()
print(r.assign(per_game=(r.n/r.g).round(2)).to_string(index=False))
po = con.execute(f"""SELECT TRY_CAST(p.drawing_player_id AS BIGINT) pid, COUNT(*) po_drawn {PENS}
  AND gm.season_type='3' AND p.drawing_player_id IS NOT NULL GROUP BY 1""").df().set_index('pid')
ptoi = con.execute("""
  SELECT player_id, ANY_VALUE(position_code) pos,
         SUM(CASE WHEN manpower_state IN ('5v5','4v4','3v3') THEN toi_seconds ELSE 0 END)/3600.0 hrs
  FROM player_state_stats pss JOIN games_metadata gm USING(game_id)
  WHERE gm.season_type='3' AND position_code<>'G' GROUP BY 1""").df().set_index('player_id')
j=ptoi.join(po).join(theta[['theta2']],how='inner'); j['po_drawn']=j.po_drawn.fillna(0); j=j[j.hrs>2]
er=j.groupby('pos').apply(lambda g:g.po_drawn.sum()/g.hrs.sum(),include_groups=False).rename('er')
j=j.join(er,on='pos'); j['po_ratio']=j.po_drawn/(j.er*j.hrs)
print(f"playoff transfer: r(theta2, po_ratio)={np.corrcoef(j.theta2,j.po_ratio)[0,1]:.3f} (n={len(j)})")
hi=j[j.theta2>=j.theta2.quantile(.75)]; lo=j[j.theta2<=j.theta2.quantile(.25)]
print(f"  top quartile theta {hi.theta2.mean():.2f} -> playoff {(hi.po_drawn.sum()/(hi.er*hi.hrs).sum()):.3f}; bottom {lo.theta2.mean():.2f} -> {(lo.po_drawn.sum()/(lo.er*lo.hrs).sum()):.3f}")

# ---- make-up sequencing ----
ev = con.execute(f"""SELECT p.game_id, e.game_seconds t, p.event_team_abbr pen_team,
    gm.home_team_abbr home, TRY_CAST(p.home_score AS INT)-TRY_CAST(p.away_score AS INT) hd
  {PENS} AND gm.season_type='2' AND e.game_seconds IS NOT NULL
  ORDER BY p.game_id, e.game_seconds""").df()
ev['pen_home']=(ev.pen_team==ev.home).astype(int)
rows=[]
for gid,g in ev.groupby('game_id'):
    g=g.sort_values('t'); ph=g.pen_home.values; tt=g.t.values; hd=g.hd.values
    for i in range(1,len(ph)): rows.append((ph[i-1],ph[i],tt[i]-tt[i-1],hd[i]))
P=pd.DataFrame(rows,columns=['prev_home','cur_home','gap','hd'])
a=P[P.prev_home==1].cur_home.mean(); b=P[P.prev_home==0].cur_home.mean()
n1=(P.prev_home==1).sum(); n0=(P.prev_home==0).sum()
se=np.sqrt(a*(1-a)/n1+b*(1-b)/n0)
print(f"\nmake-up: P(next on HOME) after-home={a:.3f} after-away={b:.3f} swing={b-a:+.3f} (z={(b-a)/se:.1f}, pairs={len(P)})")
P['gb']=pd.cut(P.gap,[0,120,300,600,1200,4000],labels=['<2m','2-5m','5-10m','10-20m','>20m'])
for gb,g2 in P.groupby('gb',observed=True):
    print(f"  gap {gb:>6}: swing {g2[g2.prev_home==0].cur_home.mean()-g2[g2.prev_home==1].cur_home.mean():+.3f} (n={len(g2)})")
z=P[P.hd==0]
print(f"  tied-only swing: {z[z.prev_home==0].cur_home.mean()-z[z.prev_home==1].cur_home.mean():+.3f} (n={len(z)})")

# ---- close-and-late timing (whistle-swallowing within games; flat across skill) ----
dv = con.execute(f"""SELECT TRY_CAST(p.drawing_player_id AS BIGINT) pid, e.game_seconds t,
    TRY_CAST(p.home_score AS INT) hs, TRY_CAST(p.away_score AS INT) as_,
    CASE WHEN p.event_team_abbr=gm.home_team_abbr THEN 0 ELSE 1 END home_drew
  {PENS} AND gm.season_type='2' AND p.drawing_player_id IS NOT NULL
  AND e.game_seconds IS NOT NULL AND e.game_seconds<=3600 AND p.home_score IS NOT NULL""").df()
dv['hl']=((abs(dv.hs-dv.as_)<=1)&(dv.t>2400)).astype(int)
st = con.execute("""SELECT e.game_seconds t, TRY_CAST(p.home_score AS INT)-TRY_CAST(p.away_score AS INT) d
  FROM play_by_play_raw p JOIN events e ON CAST(e.game_id AS VARCHAR)=p.game_id AND CAST(e.event_id AS VARCHAR)=p.event_id
  JOIN games_metadata gm ON p.game_id=gm.game_id
  WHERE gm.season_type='2' AND e.game_seconds<=3600 AND p.home_score IS NOT NULL USING SAMPLE 300000 ROWS""").df()
base=((st.d.abs()<=1)&(st.t>2400)).mean()
print(f"\nclose&late: draws {dv.hl.mean():.3f} vs exposure {base:.3f} (~-12% relative)")
jt=dv.merge(theta[['theta2']],left_on='pid',right_index=True)
q=pd.qcut(jt.theta2,4,labels=False)
print("close&late share by drawer-skill quartile:", jt.groupby(q).hl.mean().round(3).tolist(), "(flat = no clutch-timing skill)")
