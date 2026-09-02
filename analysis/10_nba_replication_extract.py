import json, glob, re, collections
import pandas as pd
sp='../data/derived'
files=sorted(glob.glob('/home/steve_murray/projects/GameVibe/beta-sports/basketball/nba/data_raw/official_rapm_source/*/00*.json'))
rows=[]; box_rows=[]
def mins(s):
    if not s: return 0.0
    m=re.match(r'(?:PT)?(\d+)M?([\d.]*)', str(s))
    if not m: return 0.0
    try: return float(m.group(1))+ (float(m.group(2))/60 if m.group(2) else 0)
    except: return 0.0
for i,f in enumerate(files):
    try: d=json.load(open(f))
    except Exception: continue
    season=f.split('/')[-2]; gid=d.get('game_id')
    pbp=d.get('pbp_v3') or []
    # FT trips: actionType Free Throw, subType startswith '1 of' or 'Free Throw 1' — attribute to shooter
    # link check: preceding foul by other team within 6 action rows
    for j,r in enumerate(pbp):
        if r.get('actionType')=='Free Throw' and '1 of' in str(r.get('subType','')).lower():
            pid=r.get('personId'); team=r.get('teamId')
            drawn_from_foul=False
            for k in range(j-1, max(j-7,-1), -1):
                p=pbp[k]
                if p.get('actionType')=='Foul':
                    if p.get('teamId') and p.get('teamId')!=team: drawn_from_foul=True
                    break
            if pid and drawn_from_foul:
                rows.append((season,gid,pid,team))
    for b in d.get('boxscore_traditional_v3') or []:
        box_rows.append((season,gid,b.get('personId'),b.get('position') or 'B', mins(b.get('minutes')), (b.get('firstName','')+' '+b.get('familyName','')).strip()))
    if i%500==0: print(f'{i}/{len(files)}',flush=True)
ft=pd.DataFrame(rows,columns=['season','gid','pid','team'])
bx=pd.DataFrame(box_rows,columns=['season','gid','pid','pos','min','name'])
ft.to_parquet(f'{sp}/nba_ft_trips.parquet'); bx.to_parquet(f'{sp}/nba_box.parquet')
print('trips:',len(ft),'box rows:',len(bx),flush=True)
print('DONE',flush=True)
