import duckdb, numpy as np, pandas as pd
sp='../data/derived'
edge=pd.read_csv(f'{sp}/edge_speed.csv').dropna(subset=['speed_max_mph'])
# player-level: max speed across seasons (trait), total bursts
pe=edge.groupby('player_id').agg(speed=('speed_max_mph','max'), speed_pctl=('speed_pctl','max'),
                                 bursts=('bursts20','sum'), oz=('oz_pct','mean')).reset_index()
con=duckdb.connect('/home/steve_murray/projects/GameVibe/hockey/data/active_db/gamevibe_primary.duckdb',read_only=True)
prov=('roughing','cross-checking','unsportsmanlike-conduct','slashing')
carry=('tripping','hooking','holding','interference','holding-the-stick')
d=con.execute(f"""
  SELECT TRY_CAST(p.drawing_player_id AS BIGINT) AS player_id,
         SUM(CASE WHEN LOWER(COALESCE(p.penalty_type,'')) IN {carry} THEN 1 ELSE 0 END) AS carry_n,
         SUM(CASE WHEN LOWER(COALESCE(p.penalty_type,'')) IN {prov} THEN 1 ELSE 0 END) AS prov_n,
         COUNT(*) AS drawn
  FROM play_by_play_raw p
  JOIN game_state gs ON p.game_id=CAST(gs.game_id AS VARCHAR) AND TRY_CAST(p.event_id AS INTEGER)=gs.event_id
  JOIN events e ON CAST(e.game_id AS VARCHAR)=p.game_id AND CAST(e.event_id AS VARCHAR)=p.event_id
  JOIN games_metadata gm ON p.game_id=gm.game_id
  WHERE LOWER(p.event_type)='penalty' AND gs.manpower_state IN ('5v5','4v4','3v3')
    AND COALESCE(TRY_CAST(p.penalty_minutes AS DOUBLE), e.penalty_minutes)=2
    AND gm.season_type='2' AND p.drawing_player_id IS NOT NULL GROUP BY 1""").df()
toi=con.execute("""
  SELECT player_id, SUM(CASE WHEN manpower_state IN ('5v5','4v4','3v3') THEN toi_seconds ELSE 0 END)/3600.0 AS hrs,
         ANY_VALUE(position_code) AS pos, ANY_VALUE(player_name) AS name
  FROM player_state_stats pss JOIN games_metadata gm USING(game_id)
  WHERE gm.season_type='2' AND position_code<>'G' GROUP BY 1""").df()
theta=pd.read_csv(f'{sp}/l2b_theta2.csv',index_col=0)
m=pe.merge(toi,on='player_id').merge(d,on='player_id',how='left').merge(theta[['theta2']],left_on='player_id',right_index=True)
m[['carry_n','prov_n','drawn']]=m[['carry_n','prov_n','drawn']].fillna(0)
m=m[m.hrs>=15]
m['carry60']=m.carry_n/m.hrs; m['prov60']=m.prov_n/m.hrs; m['b60']=m.bursts/m.hrs
print(f"n={len(m)} rated players with Edge speed (15+ EV hrs)")
def part(y,x,ctrl):
    # partial corr of y~x controlling ctrl cols (within-position handled by dummies)
    X=pd.get_dummies(m[['pos']],drop_first=True).astype(float)
    for c in ctrl: X[c]=m[c]
    X['const']=1.0
    bx=np.linalg.lstsq(X.values,m[x],rcond=None)[0]; rx=m[x]-X.values@bx
    by=np.linalg.lstsq(X.values,m[y],rcond=None)[0]; ry=m[y]-X.values@by
    return np.corrcoef(rx,ry)[0,1]
print("\n== THE MECHANISM TEST (position-controlled partial correlations) ==")
print(f"speed_max -> theta2 (overall skill):        r={part('theta2','speed',[]):+.3f}")
print(f"speed_max -> carrier-route draws/60:        r={part('carry60','speed',[]):+.3f}")
print(f"speed_max -> provocation-route draws/60:    r={part('prov60','speed',[]):+.3f}")
print(f"bursts/hr -> carrier-route draws/60:        r={part('carry60','b60',[]):+.3f}")
print(f"bursts/hr -> provocation-route draws/60:    r={part('prov60','b60',[]):+.3f}")
print("\n-- with offensive-zone% control (possession) --")
m2=m.dropna(subset=['oz'])
print(f"(n={len(m2)})")
def part2(y,x):
    X=pd.get_dummies(m2[['pos']],drop_first=True).astype(float); X['oz']=m2['oz']; X['const']=1.0
    bx=np.linalg.lstsq(X.values,m2[x],rcond=None)[0]; rx=m2[x]-X.values@bx
    by=np.linalg.lstsq(X.values,m2[y],rcond=None)[0]; ry=m2[y]-X.values@by
    return np.corrcoef(rx,ry)[0,1]
print(f"speed_max -> carrier/60 | pos+OZ%:          r={part2('carry60','speed'):+.3f}")
print(f"speed_max -> prov/60    | pos+OZ%:          r={part2('prov60','speed'):+.3f}")
print(f"bursts/hr -> carrier/60 | pos+OZ%:          r={part2('carry60','b60'):+.3f}")
# quartile table for the paper
q=pd.qcut(m.speed,4,labels=['Q1 slow','Q2','Q3','Q4 fast'])
print("\ncarrier-route draws/60 by measured speed quartile:")
print(m.groupby(q,observed=True)[['carry60','prov60','theta2']].mean().round(3).to_string())
m.to_csv(f'{sp}/edge_speed_merged.csv',index=False)
