# Reproducibility note (2026-09-04, added by the reproduction test -- see
# research_outputs/handoff_reports/2026-09-04_task13_extraction_skill_repro.md): this script's
# only input is public StatsBomb Open Data (github.com/statsbomb/open-data, non-commercial
# license with attribution) -- METHODS.md's Session 7 log names the corpus as "1,179 full
# StatsBomb event files (3.4GB, all 8 tournaments)" -- but neither this file nor METHODS.md
# names the exact 8 competition_id/season_id pairs, so a stranger cannot yet reconstruct the
# events_raw/ directory from the public StatsBomb repo alone: fetch statsbomb/open-data's
# data/matches/<competition_id>/<season_id>.json for each tournament to get match ids, then
# data/events/<match_id>.json for each -- this repo just doesn't say which competitions.
# Genuinely undocumented; not fixed here (would require re-deriving the exact tournament list
# from the local raw corpus and is a bigger change than a path/dependency fix).
# Closed 2026-10-01: ../data/reference/soccer_match_competitions.csv lists every match id with its
# StatsBomb competition_id and season_id. It is 13 competition-seasons (8 international tournaments
# and 5 women's club seasons), not 8 tournaments; see 17's docstring.
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
