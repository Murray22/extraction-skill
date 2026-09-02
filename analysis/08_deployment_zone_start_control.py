import duckdb, numpy as np, pandas as pd
sp='../data/derived'
zs=pd.read_csv(f'{sp}/edge_zonestarts.csv').dropna(subset=['oz_starts'])
pz=zs.groupby('player_id').agg(oz_starts=('oz_starts','mean'),dz_starts=('dz_starts','mean')).reset_index()
theta=pd.read_csv(f'{sp}/l2b_theta2.csv',index_col=0)
con=duckdb.connect('/home/steve_murray/projects/GameVibe/hockey/data/active_db/gamevibe_primary.duckdb',read_only=True)
pts=con.execute("""
  SELECT pid, COUNT(*) AS pts FROM (
    SELECT TRY_CAST(player_id_1 AS BIGINT) AS pid FROM events WHERE LOWER(event_type)='goal'
    UNION ALL SELECT TRY_CAST(player_id_2 AS BIGINT) FROM events WHERE LOWER(event_type)='goal' AND player_id_2 IS NOT NULL
    UNION ALL SELECT TRY_CAST(player_id_3 AS BIGINT) FROM events WHERE LOWER(event_type)='goal' AND player_id_3 IS NOT NULL)
  WHERE pid IS NOT NULL GROUP BY 1""").df().set_index('pid')
toi=con.execute("""
  SELECT player_id, SUM(toi_seconds)/3600.0 AS hrs, COUNT(DISTINCT game_id) AS gp, ANY_VALUE(position_code) AS pos
  FROM player_state_stats pss JOIN games_metadata gm USING(game_id)
  WHERE gm.season_type='2' AND position_code<>'G' GROUP BY 1""").df().set_index('player_id')
d=theta.join(pts).join(toi,how='inner').merge(pz,left_index=True,right_on='player_id')
d['pts']=d.pts.fillna(0); d=d[d.gp>=60]
d['toi_pg']=d.hrs*60/d.gp; d['pts60']=d.pts/d.hrs
print(f"n={len(d)} (60+ GP with zone-start data)")
def reg(controls, label):
    X=pd.get_dummies(d[['pos']],drop_first=True).astype(float)
    X['pts60']=(d.pts60-d.pts60.mean())/d.pts60.std()
    for c in controls: X[c]=(d[c]-d[c].mean())/d[c].std()
    X['theta2']=(d.theta2-d.theta2.mean())/d.theta2.std()
    X['const']=1.0
    y=d.toi_pg.values
    beta,_,_,_=np.linalg.lstsq(X.values,y,rcond=None)
    resid=y-X.values@beta; n,k=X.shape
    cov=(resid@resid/(n-k))*np.linalg.inv(X.values.T@X.values); se=np.sqrt(np.diag(cov))
    i=list(X.columns).index('theta2')
    print(f"{label:<42} theta2: {beta[i]:+.3f} min/game per SD (t={beta[i]/se[i]:+.2f})")
reg([], "baseline (pos + pts60):")
reg(['oz_starts'], "+ OZ-start% control:")
reg(['oz_starts','dz_starts'], "+ OZ & DZ start% controls:")
