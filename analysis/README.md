# Analysis scripts

Consolidated from the research sessions of 2026-09-01/02. Each script's docstring states the
**expected headline output as verified when the research ran** — treat those numbers as the
regression targets when reproducing. Scripts are numbered by paper section, not run order.

## Environment & paths

Python 3.12+ with `duckdb, pandas, numpy, scikit-learn, scipy, pyarrow` — see
`requirements.txt` for a version floor confirmed to reproduce every script's docstring output
exactly (2026-09-04 reproduction test). Scripts that hit source databases carry absolute paths
to the research environment's DuckDB files at the top — **edit the constants for your
environment**. The source DBs are built from public APIs (see repo README), but are not
themselves distributed in this repo and cannot be rebuilt from it alone.

Correction (2026-09-04 reproduction test): most scripts still require one of those source
DuckDB files even to *verify* an already-published number, not just to regenerate one from
scratch — despite the line this replaces, the committed `../data/derived/` tables alone are
NOT sufficient for most scripts. As of that test, exactly three analysis scripts run end to
end from committed data with no database and no other private input:
`09_referee_disjoint_test.py`, `17_soccer_replication_analysis.py` (both needed a path fix —
see git history), and `18_caphit_pricing_test.py` (needs an external `caphits.csv`, not
committed, by design — see its own docstring). The four `fetch_*.py` reference-data collectors
(`fetch_bios.py`, `fetch_birthdates.py`, `fetch_edge_speed.py`, `fetch_edge_zonestarts.py`) also
run end to end against only public NHL API calls plus a committed derived table as input.
Every other numbered script requires a source DuckDB file (`gamevibe_primary.duckdb` or
`gamevibe_discovery.duckdb`) or private raw per-league event dumps (NBA/WNBA/MBB/WBB JSON or
parquet, StatsBomb soccer events) that are not part of this repository. Full per-script
ran/failed/needs-private-data status and docstring-match results:
`research_outputs/handoff_reports/2026-09-04_task13_extraction_skill_repro.md` in the parent
GameVibe workspace (not part of this public repo).

## Script map

| Script | Paper § | Produces / verifies |
|---|---|---|
| `01_nhl_skill_model.py` | 4.1-4.2, 5 | contextual Poisson + EB θ; OOS 0.25/0.53/0.72; team-env control; writes `l2b_theta2.csv`, `l1_playergames.parquet` |
| `02_penalty_tilt_rapm.py` | 4.3, 5 | tilt ridge (SB 0.93) vs individual θ (r=0.087) — two-construct result |
| `03_win_probability_value.py` | 4.4, 6 | ΔWP +0.0171±0.0056; time/score profiles; taking-side mirror |
| `04_market_orthogonality.py` | 7.1, 7.3 | pts/60 r=0.067; deployment −0.279 (t=−3.65); embellishment (with post-review honesty note) |
| `05_fifteen_season_replication.py` | 5, 6, 7.1 | 14/14 YoY pairs (0.587); career SB 0.893; all-time board; taken-side + 2×2; team lever r=0.450 (levels, n=393)/0.134 (changes), 139-minor spread; drops seasons without drawing-player coverage (2025-26); writes `repl_player_seasons.parquet` |
| `06_routes_aging_and_nulls.py` | 4.5, 7.1 | route stability; era trend; real-age curve; carrier −4.0 vs prov −2.6%/yr; hot-hand + size nulls |
| `07_edge_speed_mechanism.py` | 4.5, 7.1 | Edge speed → carrier-route partials (0.163-0.225 possession-controlled vs 0.037-0.051) |
| `08_deployment_zone_start_control.py` | 7.3 | deployment −0.286/−0.177 under OZ / OZ+DZ start controls |
| `09_referee_disjoint_test.py` | 4.3, 5 | disjoint-crew r=0.54 vs random-split 0.55; ref heterogeneity |
| `10-11_nba_replication_*.py` | 7.2 | NBA: FT-trip linkage; SB 0.940; YoY 0.86-0.88; Giannis 3.90× |
| `12_wnba_replication.py` | 7.2 | WNBA: SB 0.860; YoY 0.72-0.82; Wilson 2.34× |
| `13-14_{mbb,wbb}_replication.py` | 7.2 | NCAA: YoY 0.54-0.66 / 0.59-0.68 (see docstrings for feed caveats) |
| `15_soccer_extract.py` + `17_soccer_replication_analysis.py` | 7.2 | StatsBomb slim-parse; SB 0.705; Hazard top-10; tax 0.301 |
| `16_playoffs_makeup_whistles.py` | 7.4 | playoffs +16% & 1:1 transfer; make-up +0.204 (z=26.8) w/ decay; close-late −12%; timing flat across skill |
| `18_caphit_pricing_test.py` | 7.3 (pending) | staged salary regression — runs when contract data arrives |
| `19_goals_path_baseline.py` | 6 | goals path: +0.127 goals/drawn minor over a computed EV baseline (0.212 vs 0.086 goals in the window), +0.163 in goal differential; writes `l3_goals_path_windows.parquet` |
| `20_naive_baseline.py` | 5 | the ladder's missing rung: a player's prior raw draws per hour (pulled toward the F/D average) predicts held-out totals at 0.721, the full model 0.723 (difference +0.002, CI −0.003 to +0.008); within forwards, context adds 0.002 to ice time. Runs from committed tables, no database |
| `fetch_referees.py` / `fetch_birthdates.py` / `fetch_bios.py` / `fetch_edge_speed.py` / `fetch_edge_zonestarts.py` | 3 | reference-data collectors (NHL public API); outputs committed in `../data/reference/` |

## Provenance & honesty notes

- Numbered scripts were reconstructed from the original session's executed code with paths
  repointed; the docstring numbers ARE the session-verified outputs. Any material deviation
  on rerun should be investigated, not averaged away.
- Two documented traps preserved in docstrings: the 15-season exposure source is
  `shifts_raw` (NOT `shifts`/`event_on_ice`, which cover 2023-25 only — a wrong-table
  "could not reproduce" flag was raised and resolved on exactly this); and embellishment
  N=1 counts must never be used as individual exoneration (see `04`'s note).
- League feeds' known quirks (ESPN college missing missed FTs; basketball drawn-by inferred
  via foul→FT linkage; PWHL/NBA official feeds committer-only) are stated in the paper's
  limitations and the relevant docstrings.
