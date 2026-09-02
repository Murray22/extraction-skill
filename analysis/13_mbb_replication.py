import numpy as np, pandas as pd, glob
sp='../data/derived'
out=[]
for f in sorted(glob.glob('/home/steve_murray/projects/GameVibe/beta-sports/basketball/mbb/data_raw/gated_seasons/mbb_raw_20*_lossless.parquet')):
    season=int(f.split('_')[-2])
    df=pd.read_parquet(f,columns=['game_id','game_play_number','type_text','team_id','athlete_id_1','athlete_name_1'])
    df=df.sort_values(['game_id','game_play_number']).reset_index(drop=True)
    tt=df.type_text.values; tm=df.team_id.values; a1=df.athlete_id_1.values; gid=df.game_id.values
    trips={}
    idx=np.where(tt=='PersonalFoul')[0]
    ftmask=(tt=='MadeFreeThrow')
    for i in idx:
        g=gid[i]; t=tm[i]
        for k in range(i+1,min(i+5,len(df))):
            if gid[k]!=g: break
            if ftmask[k] and tm[k]!=t and not np.isnan(a1[k]):
                key=(a1[k],)
                trips[(g,a1[k])]=trips.get((g,a1[k]),0)+1
                break
    tr=pd.Series(trips).rename('drawn').reset_index()
    tr.columns=['game_id','pid','drawn']
    app=df[df.athlete_id_1.notna()].groupby(['game_id','athlete_id_1']).size().rename('rows').reset_index()
    app.columns=['game_id','pid','rows']
    m=app.merge(tr,on=['game_id','pid'],how='left'); m['drawn']=m.drawn.fillna(0); m['season']=season
    names=df[df.athlete_id_1.notna()].groupby('athlete_id_1').athlete_name_1.first()
    ps=m.groupby('pid').agg(drawn=('drawn','sum'),rows=('rows','sum'),gp=('game_id','nunique')).reset_index()
    ps['season']=season; ps['name']=ps.pid.map(names)
    out.append(ps)
    print(season,'players:',len(ps),'trips:',int(ps.drawn.sum()),flush=True)
allps=pd.concat(out)
allps=allps[allps.rows>=200]
rate=allps.groupby('season').apply(lambda g:g.drawn.sum()/g.rows.sum(),include_groups=False).rename('er')
allps=allps.join(rate,on='season'); allps['mu']=allps.er*allps.rows; allps['ratio']=allps.drawn/allps.mu
seasons=sorted(allps.season.unique())
print('\nYoY (MBB, drawn-per-appearance ratio, mu>=15):')
for a,b in zip(seasons[:-1],seasons[1:]):
    A=allps[(allps.season==a)&(allps.mu>=15)][['pid','ratio']].rename(columns={'ratio':'ra'})
    B=allps[(allps.season==b)&(allps.mu>=15)][['pid','ratio']].rename(columns={'ratio':'rb'})
    j=A.merge(B,on='pid')
    if len(j)>=100: print(f"  {a}->{b}: n={len(j)} r={np.corrcoef(j.ra,j.rb)[0,1]:.3f}")
car=allps.groupby('pid').agg(dr=('drawn','sum'),mu=('mu','sum'),name=('name','first'),ns=('season','nunique'))
car=car[car.mu>=60]; car['ratio']=car.dr/car.mu
print('\nMBB pooled board (60+ expected):')
print(car.sort_values('ratio',ascending=False).head(10)[['name','ns','dr','mu','ratio']].round(2).to_string())
allps.to_parquet(f'{sp}/mbb_pdae_playerseasons.parquet')
