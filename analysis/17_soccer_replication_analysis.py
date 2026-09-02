#!/usr/bin/env python3
"""Soccer replication analysis (paper §7.2): fouls-won ratio via the identical template on
StatsBomb full-event tournament data. Extraction step: 15_soccer_extract.py (slim-parses
1,179 raw StatsBomb event files into soccer_events_slim.parquet).

Expected headline output (verified 2026-09-02):
  matches=1179, fouls won=25932
  split-half fouls-won ratio: r=0.545 SB=0.705 (n=1397)
  pooled board top includes Eden Hazard; leaders Milliet 3.71, Oliviero 3.55...
  corr(fouls-won ratio, fouls-committed ratio)=0.301 (n=758) — milder agitator tax than NHL (0.489)

StatsBomb attribution: this analysis uses StatsBomb data (Open Data license, non-commercial,
with attribution). Raw events not redistributed here; regenerate via 15_soccer_extract.py
against github.com/statsbomb/open-data.
"""
import numpy as np, pandas as pd

SLIM = "soccer_events_slim.parquet"   # produced by 15_soccer_extract.py

df = pd.read_parquet(SLIM)
m = df.groupby(['mid','pid']).agg(won=('won','sum'), com=('com','sum'), rows=('pid','size'),
                                  name=('name','first'), pos=('pos','first')).reset_index()
def pg(p):
    p = str(p)
    if 'Goalkeeper' in p: return 'G'
    if 'Back' in p: return 'D'
    if 'Midfield' in p: return 'M'
    return 'F'
m['pg'] = m.pos.map(pg); m = m[m.pg!='G']
er = m.groupby('pg').apply(lambda g: g.won.sum()/g.rows.sum(), include_groups=False).rename('er')
# split-half by match-id parity
m['gh'] = m.mid % 2
h = m.groupby(['pid','gh']).agg(w=('won','sum'), r=('rows','sum'), pg=('pg','first')).reset_index().join(er, on='pg')
h = h[h.r>=250]; h['rr'] = h.w/(h.er*h.r)
w = h.pivot_table(index='pid', columns='gh', values='rr').dropna()
r = np.corrcoef(w[0], w[1])[0,1]
print(f"split-half fouls-won ratio: r={r:.3f} SB={2*r/(1+r):.3f} (n={len(w)})")
ps = m.groupby('pid').agg(w=('won','sum'), c=('com','sum'), rows=('rows','sum'),
                          name=('name','first'), pg=('pg','first')).join(er, on='pg')
ps['mu'] = ps.er*ps.rows; ps = ps[ps.mu>=12]; ps['ratio'] = ps.w/ps.mu
print("\npooled board (12+ expected):")
print(ps.nlargest(10,'ratio')[['name','pg','w','ratio']].round(2).to_string())
erc = m.groupby('pg').apply(lambda g: g.com.sum()/g.rows.sum(), include_groups=False).rename('erc')
ps = ps.join(erc, on='pg'); ps['muc'] = ps.erc*ps.rows; ps['cratio'] = ps.c/ps.muc
print(f"\nagitator tax: corr(won ratio, committed ratio)={np.corrcoef(ps.ratio,ps.cratio)[0,1]:.3f} (n={len(ps)}; NHL comparison 0.489)")
