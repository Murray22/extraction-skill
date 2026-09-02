import json, glob, re
import numpy as np, pandas as pd
sp='../data/derived'
files=sorted(glob.glob('/home/steve_murray/projects/GameVibe/beta-sports/basketball/wnba/data_raw/official_rapm_source/*/10*.json'))
rows=[]; box=[]
def mins(s):
    m=re.match(r'(\d+):(\d+)',str(s) or '')
    return int(m.group(1))+int(m.group(2))/60 if m else 0.0
for i,f in enumerate(files):
    try: d=json.load(open(f))
    except Exception: continue
    season=f.split('/')[-2]; gid=d.get('game_id')
    acts=(d.get('play_by_play_v3') or {}).get('game',{}).get('actions',[])
    for j,r in enumerate(acts):
        if r.get('actionType')=='Free Throw' and '1 of' in str(r.get('subType','')).lower():
            ok=False
            for k in range(j-1,max(j-7,-1),-1):
                if acts[k].get('actionType')=='Foul':
                    ok=acts[k].get('teamId') not in (None,r.get('teamId')); break
            if ok and r.get('personId'): rows.append((season,gid,r['personId']))
    bt=(d.get('box_score_traditional_v3') or {}).get('boxScoreTraditional',{})
    for side in ('homeTeam','awayTeam'):
        for p in bt.get(side,{}).get('players',[]):
            st=p.get('statistics') or {}
            box.append((season,gid,p.get('personId'),p.get('position') or 'B',mins(st.get('minutes')),
                        (p.get('firstName','')+' '+p.get('familyName','')).strip()))
    if i%300==0: print(i,flush=True)
ft=pd.DataFrame(rows,columns=['season','gid','pid'])
bx=pd.DataFrame(box,columns=['season','gid','pid','pos','min','name'])
print('trips:',len(ft),'box rows:',len(bx))
trips=ft.groupby(['season','gid','pid']).size().rename('drawn').reset_index()
bx=bx[bx['min']>0]
d=bx.merge(trips,on=['season','gid','pid'],how='left'); d['drawn']=d.drawn.fillna(0)
pos=bx[bx.pos.isin(['F','C','G'])].groupby('pid').pos.agg(lambda s:s.mode().iat[0] if len(s.mode()) else 'U')
d['ppos']=d.pid.map(pos).fillna('U')
ps=d.groupby(['season','pid']).agg(drawn=('drawn','sum'),mins=('min','sum'),ppos=('ppos','first'),name=('name','first')).reset_index()
ps=ps[ps.mins>=150]
exp=ps.groupby(['season','ppos']).apply(lambda g:g.drawn.sum()/g.mins.sum(),include_groups=False).rename('er').reset_index()
ps=ps.merge(exp,on=['season','ppos']); ps['mu']=ps.er*ps.mins; ps['ratio']=ps.drawn/ps.mu
print('player-seasons (150+ min):',len(ps),'seasons:',sorted(ps.season.unique()))
# split-half
d['gh']=d.gid.astype(str).str[-1].astype(int)%2
h=d.groupby(['season','pid','gh']).agg(dr=('drawn','sum'),mn=('min','sum')).reset_index()
h=h[h.mn>=75].merge(d.groupby('pid').ppos.first().rename('ppos'),on='pid').merge(exp,on=['season','ppos'])
h['r']=h.dr/(h.er*h.mn)
w=h.pivot_table(index=['season','pid'],columns='gh',values='r').dropna()
sh=np.corrcoef(w[0],w[1])[0,1]
print(f"split-half: r={sh:.3f} SB={2*sh/(1+sh):.3f} n={len(w)}")
seasons=sorted(ps.season.unique()); 
for a,b in zip(seasons[:-1],seasons[1:]):
    A=ps[(ps.season==a)&(ps.mu>=8)][['pid','ratio']].rename(columns={'ratio':'ra'})
    B=ps[(ps.season==b)&(ps.mu>=8)][['pid','ratio']].rename(columns={'ratio':'rb'})
    j=A.merge(B,on='pid')
    if len(j)>=30: print(f"YoY {a}->{b}: n={len(j)} r={np.corrcoef(j.ra,j.rb)[0,1]:.3f}")
car=ps.groupby('pid').agg(dr=('drawn','sum'),mu=('mu','sum'),name=('name','first'),ppos=('ppos','first'))
car=car[car.mu>=40]; car['ratio']=car.dr/car.mu
print('\nPooled 2020-25 WNBA fouls-drawn board (40+ expected):')
print(car.sort_values('ratio',ascending=False).head(10)[['name','ppos','dr','mu','ratio']].round(2).to_string())
print('\nBottom-5:')
print(car.sort_values('ratio').head(5)[['name','ppos','dr','mu','ratio']].round(2).to_string())
ps.to_parquet(f'{sp}/wnba_pdae_playerseasons.parquet')
