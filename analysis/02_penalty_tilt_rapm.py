#!/usr/bin/env python3
"""Penalty-tilt ridge design: which team wins a penalty exchange while a player is on the
ice — deliberately a DIFFERENT construct from individual extraction (paper §4.3, §5).

Expected headline output (verified 2026-09-01):
  events=20369, players=858, chosen alpha=50, home-bias term ~+0.011
  split-half r=0.870 (SB 0.930)
  RAPM vs individual theta: pearson r=0.087  <- the two-construct result
  top of board = possession forwards (Panarin, Fiala, Robertson...), bottom = DZ defensemen
"""
import duckdb, numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge, RidgeCV

DB = "/home/steve_murray/projects/GameVibe/hockey/data/active_db/gamevibe_primary.duckdb"
THETA_CSV = "../data/derived/l2b_theta2.csv"   # from 01_nhl_skill_model.py

con = duckdb.connect(DB, read_only=True)
ev = con.execute("""
  WITH pens AS (
    SELECT p.game_id, p.event_id, p.event_team_abbr AS pen_team, gm.home_team_abbr, gm.season
    FROM play_by_play_raw p
    JOIN game_state gs ON p.game_id=CAST(gs.game_id AS VARCHAR) AND TRY_CAST(p.event_id AS INTEGER)=gs.event_id
    JOIN events e ON CAST(e.game_id AS VARCHAR)=p.game_id AND CAST(e.event_id AS VARCHAR)=p.event_id
    JOIN games_metadata gm ON p.game_id=gm.game_id
    WHERE LOWER(p.event_type)='penalty' AND gs.manpower_state IN ('5v5','4v4','3v3')
      AND COALESCE(TRY_CAST(p.penalty_minutes AS DOUBLE), e.penalty_minutes)=2
      AND gm.season_type='2' AND p.event_team_abbr IS NOT NULL)
  SELECT game_id, event_id, season,
         CASE WHEN pen_team = home_team_abbr THEN 0 ELSE 1 END AS home_drew
  FROM pens""").df()
oi = con.execute("""
  SELECT CAST(oi.game_id AS VARCHAR) AS game_id, oi.event_id, oi.player_id, oi.is_home
  FROM event_on_ice oi
  JOIN (SELECT DISTINCT player_id FROM player_state_stats WHERE position_code<>'G') sk USING(player_id)
""").df()
ev['key']=ev.game_id.astype(str)+'_'+ev.event_id.astype(str)
oi['key']=oi.game_id+'_'+oi.event_id.astype(str)
oi = oi[oi.key.isin(set(ev.key))]
cnt = oi.player_id.value_counts()
keep = set(cnt[cnt>=50].index)
oi_k = oi[oi.player_id.isin(keep)]
players = sorted(keep); pidx={p:i for i,p in enumerate(players)}
ev = ev.set_index('key').loc[sorted(set(oi_k.key))].reset_index()
eidx={k:i for i,k in enumerate(ev.key)}
rows=oi_k.key.map(eidx).values; cols=oi_k.player_id.map(pidx).values
vals=np.where(oi_k.is_home.values, 1.0, -1.0)
Xp = sparse.csr_matrix((vals,(rows,cols)), shape=(len(ev), len(players)))
y = ev.home_drew.values*2-1.0
Xh = sparse.hstack([sparse.csr_matrix(np.ones((len(ev),1))), Xp]).tocsr()
rcv=RidgeCV(alphas=[50,100,200,400,800,1600], fit_intercept=False); rcv.fit(Xh,y)
print(f"events={len(ev)} players={len(players)} alpha={rcv.alpha_} home-bias={float(rcv.coef_[0]):+.4f}")
coef=pd.Series(rcv.coef_[1:], index=players, name='rapm')
names = con.execute("SELECT player_id, ANY_VALUE(player_name) AS name FROM player_state_stats GROUP BY 1").df().set_index('player_id')
out = coef.to_frame().join(names)
print("\ntop-10:"); print(out.nlargest(10,'rapm').to_string())
print("\nbottom-10:"); print(out.nsmallest(10,'rapm').to_string())
theta=pd.read_csv(THETA_CSV,index_col=0)
jj=out.join(theta[['theta2']],how='inner').dropna()
print(f"\nRAPM vs individual theta: r={np.corrcoef(jj.rapm,jj.theta2)[0,1]:.3f} (n={len(jj)}) — two constructs")
half=np.arange(len(ev))%2==0
r1=Ridge(alpha=rcv.alpha_,fit_intercept=False); r1.fit(Xh[half],y[half])
r2=Ridge(alpha=rcv.alpha_,fit_intercept=False); r2.fit(Xh[~half],y[~half])
rr=np.corrcoef(r1.coef_[1:],r2.coef_[1:])[0,1]
print(f"split-half r={rr:.3f} SB={2*rr/(1+rr):.3f}")
