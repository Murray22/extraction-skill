import json, urllib.request, csv, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
sp='../data/derived'
theta=pd.read_csv(f'{sp}/l2b_theta2.csv',index_col=0)
pids=sorted(int(p) for p in theta.index)
SEASONS=[20232024,20242025,20252026]
def get(args):
    pid,season=args
    for a in range(3):
        try:
            req=urllib.request.Request(f"https://api-web.nhle.com/v1/edge/skater-zone-time/{pid}/{season}/2",
                                       headers={"User-Agent":"Mozilla/5.0"})
            j=json.loads(urllib.request.urlopen(req,timeout=20).read())
            zs=j.get('zoneStarts') or {}
            return (pid,season,zs.get('offensiveZoneStartsPctg'),zs.get('defensiveZoneStartsPctg'))
        except urllib.error.HTTPError as e:
            if e.code==404: return (pid,season,None,None)
            time.sleep(1+a)
        except Exception: time.sleep(1+a)
    return (pid,season,None,None)
jobs=[(p,s) for p in pids for s in SEASONS]
rows=[]
with ThreadPoolExecutor(max_workers=6) as ex:
    for i,r in enumerate(ex.map(get,jobs)):
        rows.append(r)
        if i%400==0: print(i,flush=True)
with open(f'{sp}/edge_zonestarts.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['player_id','season','oz_starts','dz_starts']); w.writerows(rows)
print('DONE', sum(1 for r in rows if r[2] is not None),'/',len(rows),flush=True)
