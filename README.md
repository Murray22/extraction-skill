# The Extraction Skill

**Winning contact is a persistent, individual, and unrewarded talent across six leagues.**

Research repository for the SSAC27 research paper competition submission by
[Steve Murray](https://gamevibeanalytics.com) (GameVibe Research).

Drawing penalties hands over a power play, yet hockey records the event only against the
offender — the player who caused it receives no statistic at all. This work shows that
penalty/foul drawing is a measurable, career-stable individual skill; values it in win
probability; isolates it from possession, team effects, and referee identity; identifies its
mechanism (two stylistic routes with different aging profiles, confirmed against NHL Edge
tracking data); and replicates the identical construction in six leagues (NHL, NBA, WNBA,
NCAA men's and women's basketball, international soccer).

## Headline results

| Claim | Evidence |
|---|---|
| The skill is real and persists out of sample | predicting held-out-season draw totals: r 0.25 (exposure) → 0.53 (context) → **0.72** (+skill); held-out rates monotone in prior skill quartiles (0.41→0.81/hr) |
| It is career-stable | positive in **14 of 14** adjacent season pairs since 2010 (mean r=0.59); career split-half 0.89 |
| It is individual, not possession or team | r=0.087 vs on-ice penalty "tilt" (which is possession); team-environment controls move skill ratings by nothing |
| It is not a referee artifact | reliability across disjoint referee crews (0.51) equals random-split calibration (0.49); full 46-official census |
| It has win value | +0.017 win probability per drawn EV minor (state-conditional), independently triangulated via goal conversion (+0.118 goals) |
| Mechanism: speed | two stable routes (provocation vs puck-carrier); carrier route ages faster (−4.0 vs −2.6%/yr); NHL Edge measured speed predicts carrier-route draws (partial r 0.17–0.23, possession-controlled) but not provocation draws (0.04) |
| It generalizes | same construction: NBA (reliability 0.94), WNBA (0.86), NCAA M/W (YoY 0.54–0.67), soccer (0.71) |
| Nobody rewards it | r=0.067 vs points/60; deployment −0.28 min/game per SD conditional on scoring, −0.18 after zone-start controls |

## Repository map

```
abstract/        SSAC27 abstract submission (PDF)
METHODS.md       full methods log: every model, test, coefficient, and session-by-session provenance
analysis/        analysis scripts (being populated; see METHODS.md for exact specifications)
figures/         paper figures
data/derived/    derived per-player tables produced by this research (see Data below)
data/reference/  reference data collected for this research (referee census, player bios, Edge tracking pulls)
```

## Data sources & licenses

All underlying data is public:

- **NHL**: official public API (`api-web.nhle.com`), including NHL Edge tracking endpoints.
  Raw play-by-play is re-derivable by anyone from the same API; this repo redistributes only
  derived aggregates and small reference pulls (referee assignments, bios, speed summaries).
- **NBA / WNBA / NCAA basketball**: official V3 play-by-play and ESPN feeds via the
  [sportsdataverse](https://sportsdataverse.org/) ecosystem and `nba_api`. Raw play-by-play is
  not redistributed here; derived per-player-season tables only, with fetch specifications in
  METHODS.md.
- **Soccer**: [StatsBomb Open Data](https://github.com/statsbomb/open-data). *This repository
  uses StatsBomb data. StatsBomb is the exclusive source of this data and it is used here
  under their open, non-commercial license with attribution.*

Code in this repository is MIT-licensed (see LICENSE). Data files remain subject to their
sources' terms above.

## Reproduction

Every model specification, filter, and coefficient is documented in `METHODS.md` (a complete,
dated research log). Analysis scripts are being consolidated into `analysis/` ahead of the
full-paper deadline; the derived tables in `data/derived/` are sufficient to verify every
number in the abstract directly.

## Contact

Steve Murray · stephenmurray22@gmail.com · [gamevibeanalytics.com](https://gamevibeanalytics.com)
