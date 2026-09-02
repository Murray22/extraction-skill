#!/usr/bin/env python3
"""Core NHL skill model: contextual Poisson expectation + empirical-Bayes skill layer,
with strict out-of-sample validation and the team-environment control.

Paper sections: 4.1, 4.2, 5 (validation), Methods L1/L2b in METHODS.md.
Expected headline output (verified 2026-09-01/02):
  OOS corr: pred_flat 0.250 / pred_ctx 0.527 / pred_model 0.723 (deviance 1867/1379/969)
  team-controlled variant: theta r=0.987 vs uncontrolled; OOS 0.738 / 924
  rate-based OOS (matched n=558): pearson 0.483; quartiles 0.41/0.50/0.62/0.81 per hr

Requires: gamevibe_primary.duckdb (NHL public-API-derived; see repo README for sources).
Outputs: l1_playergames.parquet, l1_theta.csv, l2b_theta2.csv into OUT_DIR.
"""
import duckdb, numpy as np, pandas as pd
from sklearn.linear_model import PoissonRegressor

DB = "/home/steve_murray/projects/GameVibe/hockey/data/active_db/gamevibe_primary.duckdb"
OUT_DIR = "."

con = duckdb.connect(DB, read_only=True)
df = con.execute("""
  WITH pens AS (
    SELECT p.game_id, TRY_CAST(p.drawing_player_id AS BIGINT) pid
    FROM play_by_play_raw p
    JOIN events e ON CAST(e.game_id AS VARCHAR)=p.game_id AND CAST(e.event_id AS VARCHAR)=p.event_id
    JOIN game_state gs ON p.game_id=CAST(gs.game_id AS VARCHAR) AND TRY_CAST(p.event_id AS INTEGER)=gs.event_id
    WHERE LOWER(p.event_type)='penalty' AND gs.manpower_state IN ('5v5','4v4','3v3')
      AND COALESCE(TRY_CAST(p.penalty_minutes AS DOUBLE), e.penalty_minutes)=2),
  d AS (SELECT game_id, pid, COUNT(*) AS drawn FROM pens WHERE pid IS NOT NULL GROUP BY 1,2),
  base AS (
    SELECT pss.player_id, ANY_VALUE(pss.player_name) AS name, ANY_VALUE(pss.position_code) AS pos,
           pss.game_id, pss.team_id, gm.season,
           SUM(CASE WHEN pss.manpower_state IN ('5v5','4v4','3v3') THEN pss.toi_seconds ELSE 0 END) AS toi_ev,
           SUM(CASE WHEN pss.score_state LIKE 'Trailing%' THEN pss.toi_seconds ELSE 0 END)/NULLIF(SUM(pss.toi_seconds),0) AS trail_share,
           SUM(CASE WHEN pss.score_state LIKE 'Leading%' THEN pss.toi_seconds ELSE 0 END)/NULLIF(SUM(pss.toi_seconds),0) AS lead_share
    FROM player_state_stats pss JOIN games_metadata gm USING(game_id)
    WHERE gm.season_type='2' AND pss.position_code<>'G'
    GROUP BY pss.player_id, pss.game_id, pss.team_id, gm.season),
  sides AS (SELECT CAST(g.game_id AS VARCHAR) AS gid, g.home_team_id, g.away_team_id FROM games g)
  SELECT b.*, d.drawn,
         CASE WHEN b.team_id = s.home_team_id THEN 1 ELSE 0 END AS is_home,
         CASE WHEN b.team_id = s.home_team_id THEN s.away_team_id ELSE s.home_team_id END AS opp_id
  FROM base b LEFT JOIN d ON b.game_id=d.game_id AND b.player_id=d.pid
  JOIN sides s ON b.game_id=s.gid
  WHERE b.toi_ev > 0""").df()
df['drawn'] = df['drawn'].fillna(0); df['hours'] = df.toi_ev/3600

# leave-one-out opponent discipline
tk = con.execute("""
  WITH pens AS (
    SELECT p.game_id, TRY_CAST(p.penalized_player_id AS BIGINT) pid
    FROM play_by_play_raw p
    JOIN events e ON CAST(e.game_id AS VARCHAR)=p.game_id AND CAST(e.event_id AS VARCHAR)=p.event_id
    JOIN game_state gs ON p.game_id=CAST(gs.game_id AS VARCHAR) AND TRY_CAST(p.event_id AS INTEGER)=gs.event_id
    WHERE LOWER(p.event_type)='penalty' AND gs.manpower_state IN ('5v5','4v4','3v3')
      AND COALESCE(TRY_CAST(p.penalty_minutes AS DOUBLE), e.penalty_minutes)=2)
  SELECT pss.team_id AS opp_id, gm.season, pens.game_id, COUNT(*) AS opp_taken_this_game
  FROM pens JOIN player_state_stats pss ON pens.game_id=pss.game_id AND pens.pid=pss.player_id
  JOIN games_metadata gm ON pens.game_id=gm.game_id WHERE gm.season_type='2'
  GROUP BY 1,2,3""").df()
ts = tk.groupby(['opp_id','season']).agg(opp_taken_season=('opp_taken_this_game','sum'),
                                         opp_games=('game_id','nunique')).reset_index()
df = df.merge(ts, on=['opp_id','season'], how='left').merge(tk, on=['opp_id','season','game_id'], how='left')
df['opp_taken_this_game']=df.opp_taken_this_game.fillna(0)
df['opp_disc']=((df.opp_taken_season-df.opp_taken_this_game)/(df.opp_games-1).clip(lower=1)).fillna(0)

# own-team leave-one-out draw environment (the L2b control)
tg = df.groupby(['team_id','season','game_id']).agg(team_drawn=('drawn','sum')).reset_index()
tss = tg.groupby(['team_id','season']).agg(team_drawn_season=('team_drawn','sum'),
                                           tgames=('game_id','nunique')).reset_index()
df = df.merge(tg,on=['team_id','season','game_id']).merge(tss,on=['team_id','season'])
df['team_draw_env'] = (df.team_drawn_season - df.team_drawn) / (df.tgames-1).clip(lower=1)

def fit(with_team_env: bool, label: str):
    X = pd.get_dummies(df[['pos']], columns=['pos'], drop_first=True).astype(float)
    X['is_home']=df.is_home
    X['opp_disc']=(df.opp_disc-df.opp_disc.mean())/df.opp_disc.std()
    if with_team_env:
        X['team_env']=(df.team_draw_env-df.team_draw_env.mean())/df.team_draw_env.std()
    X['trail_share']=df.trail_share.fillna(df.trail_share.mean())
    X['lead_share']=df.lead_share.fillna(df.lead_share.mean())
    for s in (20242025,20252026): X[f's{s}']=(df.season.astype(int)==s).astype(float)
    y=df.drawn/df.hours; w=df.hours.values
    train=df.season.astype(int).isin([20232024,20242025]).values
    m=PoissonRegressor(alpha=1e-6,max_iter=1000); m.fit(X[train],y[train],sample_weight=w[train])
    mu=m.predict(X)*df.hours
    print(f"\n== {label} ==")
    print('coefs:', {k: round(float(v),3) for k,v in zip(X.columns, m.coef_)})
    tr=df.assign(mu=mu)[train].groupby('player_id').agg(drawn=('drawn','sum'),mu=('mu','sum'))
    tr=tr[tr.mu>1]
    ratio=tr.drawn/tr.mu; mn,v=float(ratio.mean()),float(ratio.var())
    excess=v-float((1/tr.mu).mean()); k=max(mn**2/max(excess,1e-6),1.0)
    print(f"players={len(tr)} prior k={k:.2f} excess var={excess:.3f}")
    tr['theta']=(tr.drawn+k)/(tr.mu+k/mn)
    te=df.assign(mu=mu)[~train].groupby('player_id').agg(drawn=('drawn','sum'),mu=('mu','sum'),hours=('hours','sum'))
    te=te[te.hours>8].join(tr[['theta']],how='left'); matched=te.theta.notna().sum()
    te['theta']=te.theta.fillna(1.0)
    te['pred_model']=te.mu*te.theta; te['pred_ctx']=te.mu
    flat=df[train].drawn.sum()/df[train].hours.sum(); te['pred_flat']=flat*te.hours
    def dev(yv,muv):
        muv=np.clip(muv,1e-9,None)
        return 2*np.sum(np.where(yv>0,yv*np.log(yv/muv),0)-(yv-muv))
    for c in ('pred_flat','pred_ctx','pred_model'):
        print(f"OOS 25-26 {c:<10} corr={np.corrcoef(te.drawn,te[c])[0,1]:.3f} deviance={dev(te.drawn.values,te[c].values):.0f} (n={len(te)}, matched={matched})")
    # exposure-free rate test on matched players
    tm=te[te.index.isin(tr.index)].copy(); tm['rate']=tm.drawn/tm.hours
    tj=tm.join(tr[['theta']].rename(columns={'theta':'theta_tr'}))
    print(f"rate OOS: pearson={np.corrcoef(tj.theta_tr,tj.rate)[0,1]:.3f}")
    q=pd.qcut(tj.theta_tr,4,labels=False)
    print("held-out rate by prior-skill quartile:", tj.groupby(q).rate.mean().round(3).tolist())
    return tr

tr1 = fit(False, "L1: context model (no team env)")
tr2 = fit(True,  "L2b: + own-team draw environment")
j = tr1[['theta']].join(tr2[['theta']], lsuffix='_l1', rsuffix='_l2b').dropna()
print(f"\ntheta stability across team control: r={np.corrcoef(j.theta_l1,j.theta_l2b)[0,1]:.4f}")
tr2.rename(columns={'theta':'theta2'}).to_csv(f"{OUT_DIR}/l2b_theta2.csv")
df.to_parquet(f"{OUT_DIR}/l1_playergames.parquet")
