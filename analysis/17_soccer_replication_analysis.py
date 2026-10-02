#!/usr/bin/env python3
"""Soccer replication analysis (paper §7.2): fouls-won ratio via the identical template on
StatsBomb full-event data. Extraction step: 15_soccer_extract.py (slim-parses 1,179 raw
StatsBomb event files into soccer_events_slim.parquet).

Corpus (corrected 2026-10-01): the 1,179 matches are 13 StatsBomb competition-seasons, not "8
tournaments". 4 men's international tournaments (230 matches: World Cup 2018 and 2022, Euro 2020
and 2024), 4 women's international tournaments (178: World Cup 2019 and 2023, Euro 2022 and 2025)
and 5 women's club seasons (771: Liga F, NWSL, Frauen Bundesliga, FA WSL, Serie A Women). The
match-to-competition map is ../data/reference/soccer_match_competitions.csv, built from
statsbomb/open-data's competitions.json and matches/ files. Women's club matches are 65% of the
corpus, so the pooled board is led by women's club players and the pooled expectation mixes two
populations with different foul rates. The men's and women's results are therefore reported
separately, each with its own position expectation.

Expected headline output (pooled: verified 2026-09-02; by gender: 2026-10-01):
  corpus: 13 competition-seasons, 1179 matches (men international 230, women international 178, women club 771)
  split-half fouls-won ratio: r=0.545 SB=0.705 (n=1397)
  pooled board top includes Eden Hazard; leaders Milliet 3.71, Oliviero 3.55...
  corr(fouls-won ratio, fouls-committed ratio)=0.301 (n=758) — milder agitator tax than NHL (0.489)
  men   (230 matches): split-half r=0.467 SB=0.636 (n=296); board Hazard 2.98, Messi 2.25, Embolo 1.95, Kane 1.85, Neymar 1.84
  women (949 matches): split-half r=0.574 SB=0.729 (n=1101); board Milliet 3.89, Oliviero 3.72, Armengol 3.71, DeMelo 3.36, Oberdorf 3.26

StatsBomb attribution: this analysis uses StatsBomb data (Open Data license, non-commercial,
with attribution). Raw events not redistributed here; regenerate via 15_soccer_extract.py
against github.com/statsbomb/open-data.
"""
import numpy as np, pandas as pd

SLIM = "../data/derived/soccer_events_slim.parquet"   # produced by 15_soccer_extract.py
COMPS = "../data/reference/soccer_match_competitions.csv"   # match_id -> competition, gender, level


def pg(p):
    p = str(p)
    if 'Goalkeeper' in p: return 'G'
    if 'Back' in p: return 'D'
    if 'Midfield' in p: return 'M'
    return 'F'


def load():
    df = pd.read_parquet(SLIM)
    m = df.groupby(['mid','pid']).agg(won=('won','sum'), com=('com','sum'), rows=('pid','size'),
                                      name=('name','first'), pos=('pos','first')).reset_index()
    m['pg'] = m.pos.map(pg); m = m[m.pg!='G']
    comps = pd.read_csv(COMPS)
    m = m.merge(comps[['match_id','competition','season','gender','level']], left_on='mid', right_on='match_id', how='left')
    if m.gender.isna().any():
        raise SystemExit(f"{m[m.gender.isna()].mid.nunique()} matches have no competition in {COMPS}")
    return m


def split_half(m):
    """Fouls-won ratio (won / position-expected) in odd vs even matches; players with 250+ events in each half."""
    er = m.groupby('pg').apply(lambda g: g.won.sum()/g.rows.sum(), include_groups=False).rename('er')
    h = m.assign(gh=m.mid % 2).groupby(['pid','gh']).agg(w=('won','sum'), r=('rows','sum'), pg=('pg','first')).reset_index().join(er, on='pg')
    h = h[h.r>=250]; h['rr'] = h.w/(h.er*h.r)
    w = h.pivot_table(index='pid', columns='gh', values='rr').dropna()
    r = np.corrcoef(w[0], w[1])[0,1]
    return r, 2*r/(1+r), len(w)


def board(m):
    """Career fouls-won ratio for players with 12+ expected fouls won."""
    er = m.groupby('pg').apply(lambda g: g.won.sum()/g.rows.sum(), include_groups=False).rename('er')
    ps = m.groupby('pid').agg(w=('won','sum'), c=('com','sum'), rows=('rows','sum'),
                              name=('name','first'), pg=('pg','first')).join(er, on='pg')
    ps['mu'] = ps.er*ps.rows; ps = ps[ps.mu>=12].copy(); ps['ratio'] = ps.w/ps.mu
    return ps


def main():
    m = load()
    n = m.groupby(['gender','level']).mid.nunique()
    print(f"corpus: {m[['competition','season']].drop_duplicates().shape[0]} competition-seasons, {m.mid.nunique()} matches "
          f"(men international {n.get(('male','international'),0)}, women international {n.get(('female','international'),0)}, "
          f"women club {n.get(('female','club'),0)})")
    r, sb, k = split_half(m)
    print(f"split-half fouls-won ratio: r={r:.3f} SB={sb:.3f} (n={k})")
    ps = board(m)
    print("\npooled board (12+ expected):")
    print(ps.nlargest(10,'ratio')[['name','pg','w','ratio']].round(2).to_string())
    erc = m.groupby('pg').apply(lambda g: g.com.sum()/g.rows.sum(), include_groups=False).rename('erc')
    ps = ps.join(erc, on='pg'); ps['muc'] = ps.erc*ps.rows; ps['cratio'] = ps.c/ps.muc
    print(f"\nagitator tax: corr(won ratio, committed ratio)={np.corrcoef(ps.ratio,ps.cratio)[0,1]:.3f} (n={len(ps)}; NHL comparison 0.489)")
    # men's and women's competitions separately, each with its own position expectation
    for label, key in (('men  ', 'male'), ('women', 'female')):
        sub = m[m.gender == key]
        r, sb, k = split_half(sub)
        print(f"\n{label} ({sub.mid.nunique()} matches): split-half r={r:.3f} SB={sb:.3f} (n={k})")
        print(board(sub).nlargest(5,'ratio')[['name','pg','w','ratio']].round(2).to_string())


if __name__ == "__main__":
    main()
