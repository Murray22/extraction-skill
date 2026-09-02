#!/usr/bin/env python3
"""Route taxonomy, aging curves, route-specific aging (the mechanism's second leg), and
the disciplined nulls (hot hand; size moderator) — paper §4.5, §7.1.

Expected headline output (verified 2026-09-01/02):
  3-season: corr(theta2, prov share)=0.306; route split-half SB=0.734 (n=331)
  15-season: route adjacent-season r=0.457 (n=1034 pairs); league prov share ~0.30-0.33
    for a decade rising to 0.35 (2023-24) and 0.38 (2024-25)
  exemplars: M.Tkachuk prov 0.72 | Stuetzle 0.36 (AT league norm, NOT below) | McDavid 0.14 | Makar 0.12
  aging (real birthdates, 1137 players): 1.241 (age 20) -> 0.757 (37) monotone; FE slope -0.029/yr
  route-specific: carrier -4.0%/yr (boot SE .0011) vs provocation -2.6%/yr (SE .0005)
  hot hand NULL: within-game var-ratio 1.045; P(2+|1+) 0.082 obs vs 0.076 Poisson
  size: height r -0.05..-0.21 within position (D strongest); BMI ~0

Requires: gamevibe_primary.duckdb, gamevibe_discovery.duckdb,
          ../data/reference/player_birthdates.csv, ../data/reference/player_bios.csv,
          ../data/derived/repl_player_seasons.parquet (from 05), l1_playergames.parquet +
          l2b_theta2.csv (from 01).
"""
import duckdb, numpy as np, pandas as pd

PRIMARY = "/home/steve_murray/projects/GameVibe/hockey/data/active_db/gamevibe_primary.duckdb"
DISCOVERY = "/home/steve_murray/projects/GameVibe/hockey/data/gamevibe_discovery.duckdb"
REF = "../data/reference"; DER = "../data/derived"

PROV = ('roughing','cross-checking','unsportsmanlike-conduct','slashing')
CARRY = ('tripping','hooking','holding','interference','holding-the-stick')

# ---------- 3-season route mix + stability ----------
con = duckdb.connect(PRIMARY, read_only=True)
theta = pd.read_csv(f"{DER}/l2b_theta2.csv", index_col=0)
d = con.execute("""
  SELECT TRY_CAST(p.drawing_player_id AS BIGINT) AS pid, LOWER(COALESCE(p.penalty_type,'?')) AS ptype,
         (TRY_CAST(p.event_id AS INT)%2) AS half, COUNT(*) AS n
  FROM play_by_play_raw p
  JOIN game_state gs ON p.game_id=CAST(gs.game_id AS VARCHAR) AND TRY_CAST(p.event_id AS INTEGER)=gs.event_id
  JOIN events e ON CAST(e.game_id AS VARCHAR)=p.game_id AND CAST(e.event_id AS VARCHAR)=p.event_id
  JOIN games_metadata gm ON p.game_id=gm.game_id
  WHERE LOWER(p.event_type)='penalty' AND gs.manpower_state IN ('5v5','4v4','3v3')
    AND COALESCE(TRY_CAST(p.penalty_minutes AS DOUBLE), e.penalty_minutes)=2
    AND gm.season_type='2' AND p.drawing_player_id IS NOT NULL GROUP BY 1,2,3""").df()
d['cls'] = np.where(d.ptype.isin(PROV),'prov', np.where(d.ptype.isin(CARRY),'carry','other'))
piv = d.pivot_table(index='pid', columns='cls', values='n', aggfunc='sum', fill_value=0)
piv['tot'] = piv.sum(axis=1)
p20 = piv[piv.tot>=20].copy(); p20['prov_share'] = p20.prov/(p20.prov+p20.carry)
j = p20.join(theta[['theta2']], how='inner')
print(f"3-season: corr(theta2, prov_share)={np.corrcoef(j.theta2,j.prov_share)[0,1]:.3f} (n={len(j)})")
h = d.pivot_table(index=['pid','half'], columns='cls', values='n', aggfunc='sum', fill_value=0)
h['tot']=h.sum(axis=1); h=h[h.tot>=10]; h['ps']=h.prov/(h.prov+h.carry).clip(lower=1)
w = h.ps.unstack('half').dropna(); r = np.corrcoef(w[0],w[1])[0,1]
print(f"route split-half: r={r:.3f} SB={2*r/(1+r):.3f} (n={len(w)})")

# ---------- 15-season route stability + era trend ----------
dcon = duckdb.connect(DISCOVERY, read_only=True)
pt = dcon.execute("""
  SELECT gm.season, TRY_CAST(p.drawing_player_id AS BIGINT) AS player_id,
         LOWER(COALESCE(p.penalty_type,'?')) AS ptype, COUNT(*) AS n
  FROM play_by_play_raw p JOIN games_metadata gm ON p.game_id=CAST(gm.game_id AS VARCHAR)
  WHERE LOWER(p.event_type)='penalty' AND gm.season_type='2'
    AND p.strength_state IN ('1551','1441','1331') AND TRY_CAST(p.penalty_minutes AS INT)=2
    AND p.drawing_player_id IS NOT NULL AND CAST(gm.season AS INT)>=20102011 GROUP BY 1,2,3""").df()
pt['season']=pt.season.astype(int)
pt['cls']=np.where(pt.ptype.isin(PROV),'prov',np.where(pt.ptype.isin(CARRY),'carry','other'))
era=pt.groupby(['season','cls']).n.sum().unstack(fill_value=0)
era['prov_share']=era.prov/(era.prov+era.carry)
print("\nleague prov share by season:"); print(era['prov_share'].round(3).to_string())
ps=pt.pivot_table(index=['player_id','season'],columns='cls',values='n',aggfunc='sum',fill_value=0)
ps['tot']=ps.sum(axis=1); ps=ps[ps.tot>=12]; ps['share']=ps.prov/(ps.prov+ps.carry).clip(lower=1)
pairs=[]
for pid,g in ps.reset_index().groupby('player_id'):
    g=g.sort_values('season')
    for i in range(len(g)-1):
        if int(str(g.season.iloc[i+1])[:4])-int(str(g.season.iloc[i])[:4])==1:
            pairs.append((g.share.iloc[i],g.share.iloc[i+1]))
P=np.array(pairs)
print(f"route adjacent-season stability: r={np.corrcoef(P[:,0],P[:,1])[0,1]:.3f} (n={len(P)})")

# ---------- aging with real birthdates + route-specific FE slopes ----------
bd=pd.read_csv(f"{REF}/player_birthdates.csv"); bd=bd[bd.birth_date.notna()&(bd.birth_date!='')]
bd['by']=pd.to_datetime(bd.birth_date).dt.year + pd.to_datetime(bd.birth_date).dt.dayofyear/365
rep=pd.read_parquet(f"{DER}/repl_player_seasons.parquet").merge(bd,on='player_id')
rep['age']=(rep.season//10000+0.75)-rep.by
rep=rep[(rep.mu_drawn>=5)&(rep.age>=18)&(rep.age<=42)] if 'mu_drawn' in rep else rep[(rep.mu>=5)&(rep.age>=18)&(rep.age<=42)]
mu_col='mu_drawn' if 'mu_drawn' in rep else 'mu'
rep['ageb']=rep.age.round().astype(int)
cur=rep.groupby('ageb').apply(lambda g:pd.Series({'ratio':g.drawn.sum()/g[mu_col].sum(),'n':len(g)}),include_groups=False)
print("\ndraw ratio by age:"); print(cur[cur.n>=80].round(3).to_string())
g=rep.groupby('player_id'); rep['r']=rep.drawn/rep[mu_col]
rep['r_dm']=rep.r-g.r.transform('mean'); rep['a_dm']=rep.age-g.age.transform('mean')
m=g.player_id.transform('size')>=4
print(f"within-player age slope: {np.polyfit(rep[m].a_dm,rep[m].r_dm,1)[0]:+.4f}/yr")
rt=pt.pivot_table(index=['player_id','season'],columns='cls',values='n',aggfunc='sum',fill_value=0).reset_index()
rr=rt.merge(rep[['player_id','season','age','hrs',mu_col]],on=['player_id','season'])
for cls in ('carry','prov'):
    rr[f'{cls}_rate']=rr[cls]/rr.hrs
    sub=rr[rr[mu_col]>=8].copy(); gg=sub.groupby('player_id')
    sub['y']=sub[f'{cls}_rate']-gg[f'{cls}_rate'].transform('mean')
    sub['x']=sub.age-gg.age.transform('mean')
    mm=gg.player_id.transform('size')>=4
    sl=np.polyfit(sub[mm].x,sub[mm].y,1)[0]; mean_rate=sub[mm][f'{cls}_rate'].mean()
    print(f"{cls}-route FE age slope: {sl:+.4f}/hr/yr = {100*sl/mean_rate:+.1f}%/yr (n={int(mm.sum())})")

# ---------- nulls: hot hand + size ----------
pg=pd.read_parquet(f"{DER}/l1_playergames.parquet")
pr=pg.groupby(['player_id','season']).drawn.transform('sum')-pg.drawn
gh=pg.groupby(['player_id','season']).hours.transform('sum')-pg.hours
pg['own_rate']=(pr/gh).replace([np.inf,-np.inf],np.nan)
pg=pg.dropna(subset=['own_rate']); pg['exp_g']=pg.own_rate*pg.hours
obs_var=pg.groupby('player_id').apply(lambda g: ((g.drawn-g.exp_g)**2).mean(),include_groups=False)
exp_var=pg.groupby('player_id').exp_g.mean()
disp=(obs_var/exp_var).replace([np.inf,-np.inf],np.nan).dropna()
have1=pg[pg.drawn>=1]; lam=have1.exp_g
pexp=((1-np.exp(-lam)-lam*np.exp(-lam))/(1-np.exp(-lam))).mean()
print(f"\nhot hand: var-ratio={disp.mean():.3f} (1.0=Poisson); P(2+|1+) obs={(have1.drawn>=2).mean():.3f} vs exp={pexp:.3f}")
bio=pd.read_csv(f"{REF}/player_bios.csv").set_index('player_id')
car=rep.groupby('player_id').agg(dr=('drawn','sum'),mu=(mu_col,'sum'),pos=('pos','first'))
car=car[car.mu>=25]; car['ratio']=car.dr/car.mu
jb=car.join(bio,how='inner').dropna(subset=['height_cm'])
for p2,g2 in jb.groupby('pos'):
    if len(g2)>40: print(f"size ({p2}): height r={np.corrcoef(g2.height_cm,g2.ratio)[0,1]:+.3f} (n={len(g2)})")
