import numpy as np, pandas as pd
sp='../data/derived'
# game_referees.csv is fetch_referees.py's output path/name, but the committed reference pull
# lives at ../data/reference/game_referees_2023_2026.csv (same columns: game_id, referees) --
# fixed here rather than renaming the committed file, since data/reference/ is where every
# other NHL public-API reference pull in this repo lives (player_bios.csv, player_birthdates.csv).
refs=pd.read_csv('../data/reference/game_referees_2023_2026.csv',dtype={'game_id':str})
refs=refs[refs.referees.notna()&(refs.referees!='')]
print(f"games with refs: {len(refs)}; distinct referees: {len(set('|'.join(refs.referees).split('|')))}")
# NOTE (reproducibility fix): this script's only DB connection was unused dead code -- every
# value below comes from refs (data/reference/) and l1_playergames.parquet (data/derived/,
# both already committed), so the duckdb.connect() call was blocking the script for anyone
# without the private DB despite never actually reading from it. Removed.
df=pd.read_parquet(f'{sp}/l1_playergames.parquet')
df=df.merge(refs, left_on='game_id', right_on='game_id', how='inner')
print(f"player-games with refs: {len(df)} / 141711")
# 1) Referee heterogeneity: per-ref EV-minor rate per game
g=df.groupby(['game_id','referees']).agg(drawn=('drawn','sum')).reset_index()
long=[]
for _,r in g.iterrows():
    for ref in r.referees.split('|'): long.append((ref, r.drawn))
L=pd.DataFrame(long,columns=['ref','pens'])
rr=L.groupby('ref').agg(games=('pens','count'),mean_pens=('pens','mean'))
rr=rr[rr.games>=50]
print(f"\nreferee EV-minor-per-game rates ({len(rr)} refs, 50+ games): mean {rr.mean_pens.mean():.2f}, sd {rr.mean_pens.std():.2f}, range {rr.mean_pens.min():.2f}-{rr.mean_pens.max():.2f}")
# variance vs sampling expectation (Poisson): expected sd of mean = sqrt(lambda/n)
lam=rr.mean_pens.mean(); samp=np.sqrt((lam/rr.games).mean())
print(f"  sampling sd expectation ~{samp:.3f} -> excess referee heterogeneity: {max(rr.mean_pens.std()**2-samp**2,0)**.5:.3f} pens/game (real refs差)")
# 2) Referee-disjoint split-half of player skill
# assign each REF to half A/B by hash; a game goes to A if both refs in A, B if both in B, else dropped
# REPRODUCIBILITY FIX (2026-09-04): Python's built-in hash() on strings is randomized per
# process (PYTHONHASHSEED) unless explicitly disabled -- confirmed directly (two back-to-back
# `python3 -c "print(hash('Wes McCauley'))"` calls returned different values). That made this
# section's ref->half assignment, and therefore its printed r/SB/n, genuinely different on every
# run of this exact script against the exact same data -- not a data or logic bug, a hash-
# instability one. hashlib.md5 is stable across processes/machines/Python versions.
import hashlib
h={ref:(int(hashlib.md5(ref.encode()).hexdigest(),16)%2) for ref in set('|'.join(refs.referees).split('|'))}
def game_half(s):
    hs={h[r] for r in s.split('|')}
    return hs.pop() if len(hs)==1 else -1
df['rhalf']=df.referees.map(game_half)
sub=df[df.rhalf>=0]
print(f"\nreferee-disjoint games: A={len(sub[sub.rhalf==0].game_id.unique())} B={len(sub[sub.rhalf==1].game_id.unique())} (dropped mixed)")
A=sub[sub.rhalf==0].groupby('player_id').agg(dr=('drawn','sum'),mu=('mu','sum'))
B=sub[sub.rhalf==1].groupby('player_id').agg(dr=('drawn','sum'),mu=('mu','sum'))
j=A.join(B,lsuffix='_a',rsuffix='_b')
j=j[(j.mu_a>=6)&(j.mu_b>=6)]
ra=j.dr_a/j.mu_a; rb=j.dr_b/j.mu_b
r=np.corrcoef(ra,rb)[0,1]
print(f"REFEREE-DISJOINT split-half of draw ratio (6+ expected each side): r={r:.3f} SB={2*r/(1+r):.3f} n={len(j)}")
# compare to a random game split of same subset for calibration
rng=np.random.default_rng(7)
sub2=sub.copy(); sub2['gh']=rng.integers(0,2,len(sub2))
A2=sub2[sub2.gh==0].groupby('player_id').agg(dr=('drawn','sum'),mu=('mu','sum'))
B2=sub2[sub2.gh==1].groupby('player_id').agg(dr=('drawn','sum'),mu=('mu','sum'))
j2=A2.join(B2,lsuffix='_a',rsuffix='_b'); j2=j2[(j2.mu_a>=6)&(j2.mu_b>=6)]
r2=np.corrcoef(j2.dr_a/j2.mu_a,j2.dr_b/j2.mu_b)[0,1]
print(f"calibration random split same data: r={r2:.3f} SB={2*r2/(1+r2):.3f} n={len(j2)}")
