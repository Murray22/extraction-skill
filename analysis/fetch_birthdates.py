import json, urllib.request, csv, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
sp='../data/derived'
d=pd.read_parquet(f'{sp}/repl_player_seasons.parquet')
pids=sorted(set(d[d.mu>=5].player_id.astype(int)))
print('players to fetch:', len(pids), flush=True)
def get(pid):
    for a in range(3):
        try:
            req=urllib.request.Request(f"https://api-web.nhle.com/v1/player/{pid}/landing", headers={"User-Agent":"Mozilla/5.0"})
            j=json.loads(urllib.request.urlopen(req,timeout=20).read())
            return pid, j.get('birthDate','')
        except Exception:
            time.sleep(1+a)
    return pid, ''
rows=[]
with ThreadPoolExecutor(max_workers=6) as ex:
    for i,(pid,bd) in enumerate(ex.map(get,pids)):
        rows.append((pid,bd))
        if i%200==0: print(i,flush=True)
with open(f'{sp}/player_birthdates.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['player_id','birth_date']); w.writerows(rows)
print('DONE', sum(1 for _,b in rows if b), '/', len(rows), flush=True)
