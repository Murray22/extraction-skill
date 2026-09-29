#!/usr/bin/env python3
"""Fifteen-season NHL replication + career reliability + all-time board + the taken-side
skill and 2x2 archetype map + team-construction lever (paper §5, §6, §7.1, README table).

Expected headline output (seasons 2010-11..2024-25; 2025-26 excluded, see below; rerun 2026-09-29):
  YoY theta correlation positive in ALL 14 adjacent pairs, mean 0.587 (0.476-0.650)
  career split-half (odd vs even seasons, 20+ expected each): r=0.807 SB=0.893 n=597
  all-time board (40+ expected): B.Smith 2.72, Hathaway 2.49, Kadri 2.22, Dorsett, Downie,
    Cousins, Stuetzle 2.12 (7th of 641)
  taken-side YoY mean 0.622; corr(draw,take)=0.495 (n=593); 2x2 = 102/194/195/102
  team lever: prior-roster skill vs team net differential r=0.450 (n=393); changes r=0.134 (n=358)
  best-to-worst team-season net-minor spread = 139

Season coverage rule (2026-09-29): a season is used only if >=80% of its regular-season penalty
rows carry a drawing_player_id AND >=90% of its regular-season games have shifts_raw rows.
Seasons 2010-11..2024-25 sit at 90.4-92.3% drawn and 95.7-100% shifts; the DB's 2025-26 has
0 of 9,692 penalties with a drawing player and shifts for 29.8% of games, so it is excluded.
Including it gave r=0.708 (n=425) and a 163-minor spread (the numbers commit 8495185 adopted): with no
drawn side, all 32 of its team-seasons sit at a taken-only ~-67 and that cluster drove the r.
By default uncovered seasons are dropped with a warning; `--seasons a,b,...` that names an
uncovered season exits non-zero. Tests: tests/test_05_season_coverage.py.

IMPORTANT provenance note: exposure = shifts_raw (2010-11..2024-25 coverage; 2025-26 partial). The
`shifts`/`event_on_ice` tables in this DB cover 2023-25 only — a 2026-09-02 verification
flag was raised and resolved on exactly this point; do not "fix" the query to those tables.
Requires: gamevibe_discovery.duckdb.
"""
import argparse, sys
import duckdb, numpy as np, pandas as pd

DB = "/home/steve_murray/projects/GameVibe/hockey/data/gamevibe_discovery.duckdb"
OUT_DIR = "../data/derived"
EV_CODES = ('1551','1441','1331')   # situationCode EV states, goalies in
FIRST_SEASON = 20102011             # first season with shifts_raw
MIN_DRAWN_SHARE = 0.80              # share of penalty rows with a drawing_player_id
MIN_SHIFT_SHARE = 0.90              # share of games with shifts_raw rows
WP_PER_MINOR = 0.0171               # script 03: +0.0171 win probability per drawn EV minor


def season_coverage(con):
    """Per regular season: share of penalty rows naming a drawing player, share of games with shifts."""
    return con.execute("""
      WITH pen AS (
        SELECT gm.season, COUNT(*) AS penalties, COUNT(p.drawing_player_id) AS with_drawer
        FROM play_by_play_raw p JOIN games_metadata gm ON p.game_id=CAST(gm.game_id AS VARCHAR)
        WHERE LOWER(p.event_type)='penalty' AND gm.season_type='2' GROUP BY 1),
      gms AS (SELECT season, COUNT(*) AS games FROM games_metadata WHERE season_type='2' GROUP BY 1),
      sh AS (
        SELECT gm.season, COUNT(DISTINCT CAST(s.game_id AS VARCHAR)) AS shift_games
        FROM shifts_raw s JOIN games_metadata gm ON CAST(s.game_id AS VARCHAR)=CAST(gm.game_id AS VARCHAR)
        WHERE gm.season_type='2' GROUP BY 1)
      SELECT CAST(gms.season AS INT) AS season, COALESCE(penalties,0) AS penalties,
             COALESCE(with_drawer,0) AS with_drawer, games, COALESCE(shift_games,0) AS shift_games,
             COALESCE(with_drawer,0)*1.0/NULLIF(penalties,0) AS drawn_share,
             COALESCE(shift_games,0)*1.0/games AS shift_share
      FROM gms LEFT JOIN pen USING(season) LEFT JOIN sh USING(season) ORDER BY 1""").df()


def select_seasons(cov, requested=None, min_drawn=MIN_DRAWN_SHARE, min_shift=MIN_SHIFT_SHARE):
    """Return (seasons to use, rows of excluded seasons). A requested season without coverage is fatal."""
    cov = cov.copy()
    cov['season'] = cov.season.astype(int)
    ok = (cov.drawn_share.fillna(0) >= min_drawn) & (cov.shift_share.fillna(0) >= min_shift)
    if requested is not None:
        requested = sorted(int(s) for s in requested)
        missing = [s for s in requested if s not in set(cov.season)]
        bad = cov[cov.season.isin(requested) & ~ok]
        if missing or len(bad):
            detail = [f"{s}: not in DB" for s in missing] + [
                f"{r.season}: drawn {r.drawn_share or 0:.1%} (need {min_drawn:.0%}), "
                f"shifts {r.shift_share or 0:.1%} (need {min_shift:.0%})" for r in bad.itertuples()]
            raise SystemExit("ERROR: requested season(s) lack drawing-player/shift coverage -> "
                             + "; ".join(detail))
        return requested, cov.iloc[0:0]
    cand = cov[cov.season >= FIRST_SEASON]
    used = sorted(cand[ok.loc[cand.index]].season)
    if not used:
        raise SystemExit("ERROR: no season meets the coverage rule")
    return used, cand[~ok.loc[cand.index]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--seasons', help='comma-separated season ids, e.g. 20102011,20112012 '
                    '(default: every covered season from 2010-11)')
    a = ap.parse_args(argv)
    requested = [s for s in a.seasons.split(',') if s.strip()] if a.seasons else None

    con = duckdb.connect(DB, read_only=True)
    SEASONS, excluded = select_seasons(season_coverage(con), requested)
    for r in excluded.itertuples():
        print(f"WARNING: EXCLUDED season {r.season}: {r.with_drawer:,} of {r.penalties:,} penalties name a "
              f"drawing player ({r.drawn_share or 0:.1%}), shifts for {r.shift_share:.1%} of games "
              f"(rule: >={MIN_DRAWN_SHARE:.0%} and >={MIN_SHIFT_SHARE:.0%})", file=sys.stderr)
    print(f"seasons used: {SEASONS[0]}..{SEASONS[-1]} ({len(SEASONS)}); "
          f"excluded: {', '.join(str(s) for s in excluded.season) or 'none'}")
    drawn = con.execute(f"""
      SELECT gm.season, TRY_CAST(p.drawing_player_id AS BIGINT) AS player_id, COUNT(*) AS drawn
      FROM play_by_play_raw p JOIN games_metadata gm ON p.game_id=CAST(gm.game_id AS VARCHAR)
      WHERE LOWER(p.event_type)='penalty' AND gm.season_type='2'
        AND p.strength_state IN {EV_CODES}
        AND TRY_CAST(p.penalty_minutes AS INT)=2 AND p.drawing_player_id IS NOT NULL
      GROUP BY 1,2""").df()
    taken = con.execute(f"""
      SELECT gm.season, TRY_CAST(p.penalized_player_id AS BIGINT) AS player_id, COUNT(*) AS taken
      FROM play_by_play_raw p JOIN games_metadata gm ON p.game_id=CAST(gm.game_id AS VARCHAR)
      WHERE LOWER(p.event_type)='penalty' AND gm.season_type='2'
        AND p.strength_state IN {EV_CODES}
        AND TRY_CAST(p.penalty_minutes AS INT)=2 AND p.penalized_player_id IS NOT NULL
      GROUP BY 1,2""").df()
    toi = con.execute("""
      SELECT gm.season, TRY_CAST(s.player_id AS BIGINT) AS player_id,
             SUM(TRY_CAST(s.duration_seconds AS DOUBLE))/3600.0 AS hrs
      FROM shifts_raw s JOIN games_metadata gm ON CAST(s.game_id AS VARCHAR)=CAST(gm.game_id AS VARCHAR)
      WHERE gm.season_type='2' GROUP BY 1,2""").df()
    pos = con.execute("SELECT player_id, position_code AS pos, full_name AS name FROM players").df()
    d = toi.merge(drawn, on=['season','player_id'], how='left').merge(taken, on=['season','player_id'], how='left').merge(pos, on='player_id', how='left')
    d = d[(d.pos.notna()) & (d.pos!='G') & (d.hrs>3)]
    d[['drawn','taken']] = d[['drawn','taken']].fillna(0)
    d['season'] = d.season.astype(int); d = d[d.season.isin(SEASONS)]

    for col in ('drawn','taken'):
        exp = d.groupby(['season','pos']).apply(lambda g: g[col].sum()/g.hrs.sum(), include_groups=False).rename(f'er_{col}').reset_index()
        d = d.merge(exp, on=['season','pos'])
        d[f'mu_{col}'] = d[f'er_{col}']*d.hrs

    def fit_theta(g, col):
        g = g[g[f'mu_{col}']>0.5]
        ratio = g[col]/g[f'mu_{col}']; mn = ratio.mean()
        excess = ratio.var()-(1/g[f'mu_{col}']).mean(); k = max(mn**2/max(excess,1e-6),1.0)
        return g.assign(**{f'theta_{col}': (g[col]+k)/(g[f'mu_{col}']+k/mn)})

    for col in ('drawn','taken'):
        d2 = pd.concat([fit_theta(g,col) for _,g in d.groupby('season')])
        seasons = sorted(d2.season.unique()); rs=[]
        for a,b in zip(seasons[:-1],seasons[1:]):
            A=d2[(d2.season==a)&(d2[f'mu_{col}']>=8)][['player_id',f'theta_{col}']]
            B=d2[(d2.season==b)&(d2[f'mu_{col}']>=8)][['player_id',f'theta_{col}']]
            j=A.merge(B,on='player_id',suffixes=('_a','_b'))
            if len(j)>50: rs.append(np.corrcoef(j[f'theta_{col}_a'],j[f'theta_{col}_b'])[0,1])
        print(f"{col}-side YoY: {len(rs)} pairs, mean r={np.mean(rs):.3f} (min {min(rs):.3f} max {max(rs):.3f}), all positive={all(r>0 for r in rs)}")
        if col=='drawn': d2.to_parquet(f"{OUT_DIR}/repl_player_seasons.parquet")

    # career split-half (drawn)
    d['sy']=d.season//10000
    odd=d[d.sy%2==1].groupby('player_id').agg(dr=('drawn','sum'),mu=('mu_drawn','sum'))
    evn=d[d.sy%2==0].groupby('player_id').agg(dr=('drawn','sum'),mu=('mu_drawn','sum'))
    j=odd.join(evn,lsuffix='_o',rsuffix='_e'); j=j[(j.mu_o>=20)&(j.mu_e>=20)]
    ro=np.corrcoef(j.dr_o/j.mu_o, j.dr_e/j.mu_e)[0,1]
    print(f"career split-half: r={ro:.3f} SB={2*ro/(1+ro):.3f} n={len(j)}")

    # all-time board + 2x2
    car=d.groupby('player_id').agg(dr=('drawn','sum'),mu=('mu_drawn','sum'),tk=('taken','sum'),
                                   mut=('mu_taken','sum'),name=('name','first'))
    board=car[car.mu>=40].copy(); board['ratio']=board.dr/board.mu
    print("\nall-time board (40+ expected):"); print(board.nlargest(8,'ratio')[['name','ratio']].round(2).to_string())
    both=car[(car.mu>=40)&(car.mut>=40)].copy()
    both['draw_r']=both.dr/both.mu; both['take_r']=both.tk/both.mut
    print(f"\ncorr(draw_r, take_r)={np.corrcoef(both.draw_r,both.take_r)[0,1]:.3f} (n={len(both)})")
    hd=both.draw_r>both.draw_r.median(); ht=both.take_r>both.take_r.median()
    print(f"2x2 MAGNET/AGITATOR/INVISIBLE/LIABILITY = {((hd)&(~ht)).sum()}/{((hd)&(ht)).sum()}/{((~hd)&(~ht)).sum()}/{((~hd)&(ht)).sum()}")

    # team lever: prior-season cumulative skill -> team net differential
    team = con.execute("""
      SELECT gm.season, TRY_CAST(s.player_id AS BIGINT) AS player_id, s.team_abbr,
             SUM(TRY_CAST(s.duration_seconds AS DOUBLE))/3600.0 AS thrs
      FROM shifts_raw s JOIN games_metadata gm ON CAST(s.game_id AS VARCHAR)=CAST(gm.game_id AS VARCHAR)
      WHERE gm.season_type='2' GROUP BY 1,2,3""").df()
    team['season']=team.season.astype(int)
    modal=team.sort_values('thrs').groupby(['season','player_id']).tail(1)[['season','player_id','team_abbr']]
    dl=d.merge(modal,on=['season','player_id']).sort_values(['player_id','season'])
    g=dl.groupby('player_id')
    for col,mu in (('drawn','mu_drawn'),('taken','mu_taken')):
        dl[f'c_{col}']=g[col].cumsum()-dl[col]; dl[f'c_{mu}']=g[mu].cumsum()-dl[mu]
    k=6.0
    dl['sk_draw']=(dl.c_drawn+k)/(dl.c_mu_drawn+k); dl['sk_take']=(dl.c_taken+k)/(dl.c_mu_taken+k)
    dl.loc[dl.c_mu_drawn<=0,'sk_draw']=1.0; dl.loc[dl.c_mu_taken<=0,'sk_take']=1.0
    dl['act_net']=dl.drawn-dl.taken
    dl['pred_net']=dl.mu_drawn*dl.sk_draw - dl.mu_taken*dl.sk_take
    ts=dl.groupby(['season','team_abbr']).agg(act=('act_net','sum'),pred=('pred_net','sum'),n=('player_id','count')).reset_index()
    ts=ts[ts.n>=15]
    print(f"\nteam lever: levels r={np.corrcoef(ts.pred,ts.act)[0,1]:.3f} (n={len(ts)}); actual spread sd={ts.act.std():.1f}")
    hi, lo = ts.loc[ts.act.idxmax()], ts.loc[ts.act.idxmin()]
    spread = hi.act - lo.act
    print(f"team lever: best-to-worst net-minor spread = {spread:.0f} ({hi.team_abbr} {hi.season} {hi.act:+.0f}, "
          f"{lo.team_abbr} {lo.season} {lo.act:+.0f}) = ~{spread*WP_PER_MINOR:.1f} wins at +{WP_PER_MINOR} WP/minor; "
          f"actual net differential, not skill")
    ts=ts.sort_values(['team_abbr','season'])
    ts['d_act']=ts.groupby('team_abbr').act.diff(); ts['d_pred']=ts.groupby('team_abbr').pred.diff()
    v=ts.dropna()
    print(f"team lever: CHANGES r={np.corrcoef(v.d_pred,v.d_act)[0,1]:.3f} (n={len(v)}) — claim levels, not transactions")


if __name__ == '__main__':
    main()
