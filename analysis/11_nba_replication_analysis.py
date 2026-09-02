import json, glob
import numpy as np, pandas as pd
sp='../data/derived'
ft=pd.read_parquet(f'{sp}/nba_ft_trips.parquet')
bx=pd.read_parquet(f'{sp}/nba_box.parquet')
print('trips:',len(ft),'seasons:',sorted(ft.season.unique()))
trips=ft.groupby(['season','gid','pid']).size().rename('drawn').reset_index()
bx=bx[bx['min']>0]
d=bx.merge(trips,on=['season','gid','pid'],how='left'); d['drawn']=d.drawn.fillna(0)
# player modal position
pos=bx[bx.pos.isin(['F','C','G'])].groupby('pid').pos.agg(lambda s:s.mode().iat[0] if len(s.mode()) else 'U')
d['ppos']=d.pid.map(pos).fillna('U')
ps=d.groupby(['season','pid']).agg(drawn=('drawn','sum'),mins=('min','sum'),gp=('gid','nunique'),ppos=('ppos','first'),name=('name','first')).reset_index()
ps=ps[ps.mins>=300]
exp=ps.groupby(['season','ppos']).apply(lambda g:g.drawn.sum()/g.mins.sum(),include_groups=False).rename('er').reset_index()
ps=ps.merge(exp,on=['season','ppos']); ps['mu']=ps.er*ps.mins; ps['ratio']=ps.drawn/ps.mu
print('player-seasons (300+ min):',len(ps))
# split-half within season (odd/even games)
d['gh']=d.gid.astype(str).str[-1].astype(int)%2
h=d.groupby(['season','pid','gh']).agg(dr=('drawn','sum'),mn=('min','sum')).reset_index()
h=h[h.mn>=150].merge(d.groupby('pid').ppos.first().rename('ppos'),on='pid')
he=h.merge(exp,on=['season','ppos']); he['r']=he.dr/(he.er*he.mn)
w=he.pivot_table(index=['season','pid'],columns='gh',values='r').dropna()
sh=np.corrcoef(w[0],w[1])[0,1]
print(f"split-half (within season): r={sh:.3f} SB={2*sh/(1+sh):.3f} n={len(w)}")
# YoY
seasons=sorted(ps.season.unique()); yy=[]
for a,b in zip(seasons[:-1],seasons[1:]):
    A=ps[(ps.season==a)&(ps.mu>=15)][['pid','ratio']].rename(columns={'ratio':'ra'})
    B=ps[(ps.season==b)&(ps.mu>=15)][['pid','ratio']].rename(columns={'ratio':'rb'})
    j=A.merge(B,on='pid')
    yy.append((a,b,len(j),np.corrcoef(j.ra,j.rb)[0,1]))
print('\nYoY foul-drawing ratio:')
for r in yy: print(f"  {r[0]}->{r[1]}: n={r[2]} r={r[3]:.3f}")
# leaderboard, pooled
car=ps.groupby('pid').agg(dr=('drawn','sum'),mu=('mu','sum'),name=('name','first'),ppos=('ppos','first'))
car=car[car.mu>=100]; car['ratio']=car.dr/car.mu
print('\nPooled 2019-24 FT-trips-drawn above positional expectation (100+ expected):')
print(car.sort_values('ratio',ascending=False).head(12)[['name','ppos','dr','mu','ratio']].round(2).to_string())
print('\nBottom-6:')
print(car.sort_values('ratio').head(6)[['name','ppos','dr','mu','ratio']].round(2).to_string())
ps.to_parquet(f'{sp}/nba_pdae_playerseasons.parquet')
