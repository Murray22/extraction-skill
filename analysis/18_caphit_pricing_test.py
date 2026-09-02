#!/usr/bin/env python3
"""Contact Won: literal market-pricing test — cap hit vs draw skill (theta2).

Usage: ./venv/bin/python3 analysis/forensics/caphit_pricing_test.py <caphits.csv>

<caphits.csv> columns (flexible): a player-name column and a cap-hit column
(named like 'cap_hit', 'caphit', 'aav'; dollar strings fine). Season assumed
2025-26 unless a 'season' column narrows it. Matches on normalized full name
against player_state_stats; prints match rate and unmatched names — inspect
that before trusting the regression.

Regression: log(cap_hit) ~ position + points/60 + TOI/gp + theta2.
H: if draw skill is unpriced, theta2 coefficient ~ 0 conditional on the rest.
Requires scratchpad l2b_theta2.csv from the 2026-09-01 research session, or
regenerate via wiki/plans/CONTACT_WON_RESEARCH_PROGRAM_2026-09-01.md.
"""
import sys, re, unicodedata
import duckdb, numpy as np, pandas as pd

SP='../data/derived'
def norm(s):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode().lower()
    return re.sub(r'[^a-z ]','',s).strip()

def main():
    caps=pd.read_csv(sys.argv[1])
    namecol=next(c for c in caps.columns if 'name' in c.lower() or 'player' in c.lower())
    capcol=next(c for c in caps.columns if re.search(r'cap|aav|salary',c,re.I))
    caps['cap']=caps[capcol].astype(str).str.replace(r'[$,]','',regex=True).astype(float)
    caps['key']=caps[namecol].map(norm)

    con=duckdb.connect('data/active_db/gamevibe_primary.duckdb',read_only=True)
    theta=pd.read_csv(f'{SP}/l2b_theta2.csv',index_col=0)
    base=con.execute("""
      SELECT player_id, ANY_VALUE(player_name) AS name, ANY_VALUE(position_code) AS pos,
             SUM(toi_seconds)/3600.0 AS hrs, COUNT(DISTINCT game_id) AS gp
      FROM player_state_stats pss JOIN games_metadata gm USING(game_id)
      WHERE gm.season_type='2' AND position_code<>'G' GROUP BY 1""").df().set_index('player_id')
    pts=con.execute("""
      SELECT pid, COUNT(*) AS pts FROM (
        SELECT TRY_CAST(player_id_1 AS BIGINT) AS pid FROM events WHERE LOWER(event_type)='goal'
        UNION ALL SELECT TRY_CAST(player_id_2 AS BIGINT) FROM events WHERE LOWER(event_type)='goal' AND player_id_2 IS NOT NULL
        UNION ALL SELECT TRY_CAST(player_id_3 AS BIGINT) FROM events WHERE LOWER(event_type)='goal' AND player_id_3 IS NOT NULL)
      WHERE pid IS NOT NULL GROUP BY 1""").df().set_index('pid')
    d=theta.join(base,how='inner').join(pts)
    d['pts']=d.pts.fillna(0); d=d[d.gp>=60]
    d['key']=d.name.map(norm)
    d=d.merge(caps[['key','cap']],on='key',how='left')
    print(f"matched {d.cap.notna().sum()}/{len(d)} rated players to cap hits")
    um=d[d.cap.isna()].name.tolist()
    if um: print("unmatched (first 20):", um[:20])
    d=d.dropna(subset=['cap'])
    d['pts60']=d.pts/d.hrs; d['toipg']=d.hrs*60/d.gp; d['logcap']=np.log(d.cap)
    X=pd.get_dummies(d[['pos']],drop_first=True).astype(float)
    for c in ('pts60','toipg','theta2'): X[c]=(d[c]-d[c].mean())/d[c].std()
    X['const']=1.0
    y=d.logcap.values
    beta,_,_,_=np.linalg.lstsq(X.values,y,rcond=None)
    resid=y-X.values@beta; n,k=X.shape
    cov=(resid@resid/(n-k))*np.linalg.inv(X.values.T@X.values); se=np.sqrt(np.diag(cov))
    print(f"\nlog(cap hit) regression, n={n}:")
    for nm,b,s in zip(X.columns,beta,se):
        print(f"  {nm:>8}: {b:+.4f}  (t={b/s:+.2f})")
    b=dict(zip(X.columns,beta))['theta2']
    print(f"\n+1 SD draw skill ≈ {100*(np.exp(b)-1):+.1f}% cap hit, conditional on scoring/TOI/position")

if __name__=='__main__': main()
