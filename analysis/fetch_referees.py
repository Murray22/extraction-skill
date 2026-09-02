import json, urllib.request, duckdb, csv, time
from concurrent.futures import ThreadPoolExecutor
sp='../data/derived'
con = duckdb.connect('/home/steve_murray/projects/GameVibe/hockey/data/active_db/gamevibe_primary.duckdb', read_only=True)
gids=[r[0] for r in con.execute("SELECT DISTINCT game_id FROM games_metadata WHERE season_type='2'").fetchall()]
con.close()
def get(gid):
    for attempt in range(3):
        try:
            req=urllib.request.Request(f"https://api-web.nhle.com/v1/gamecenter/{gid}/right-rail", headers={"User-Agent":"Mozilla/5.0"})
            d=json.loads(urllib.request.urlopen(req,timeout=30).read())
            info=d.get('gameInfo') or {}
            refs=[r['fullName']['default'] for r in (info.get('referees') or [])]
            # some payloads nest under summary
            if not refs:
                for v in d.values():
                    if isinstance(v,dict) and v.get('referees'):
                        refs=[r['fullName']['default'] for r in v['referees']]; break
            return gid, refs
        except Exception:
            time.sleep(2*(attempt+1))
    return gid, []
rows=[]
with ThreadPoolExecutor(max_workers=6) as ex:
    for i,(gid,refs) in enumerate(ex.map(get, gids)):
        rows.append((gid, '|'.join(refs)))
        if i%400==0: print(f"{i}/{len(gids)}", flush=True)
with open(f'{sp}/game_referees.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['game_id','referees']); w.writerows(rows)
missing=sum(1 for _,r in rows if not r)
print(f"DONE {len(rows)} games, {missing} missing refs", flush=True)
