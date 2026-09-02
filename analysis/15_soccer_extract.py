import json, glob
import numpy as np, pandas as pd
sp='../data/derived'
rows=[]
files=sorted(glob.glob('/home/steve_murray/projects/GameVibe/soccer/World_Cup/data/events_raw/*.json'))
print('files:',len(files),flush=True)
for i,f in enumerate(files):
    mid=int(f.split('/')[-1].split('.')[0])
    try: ev=json.load(open(f))
    except Exception: continue
    for e in ev:
        p=e.get('player') or {}
        if not p: continue
        t=e.get('type',{}).get('name')
        pos=(e.get('position') or {}).get('name','')
        rows.append((mid,p.get('id'),p.get('name'),pos,t=='Foul Won',t=='Foul Committed'))
    if i%200==0: print(i,flush=True)
df=pd.DataFrame(rows,columns=['mid','pid','name','pos','won','com'])
df.to_parquet(f'{sp}/soccer_events_slim.parquet')
print('rows:',len(df),'fouls won:',int(df.won.sum()),flush=True)
print('DONE',flush=True)
