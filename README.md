# The Extraction Skill

**Drawing penalties belongs to the player, not his line or the referees.**

Research repository for the SSAC27 research paper competition submission by
[Steve Murray](https://gamevibeanalytics.com) (GameVibe Research).

When a player draws a penalty in hockey, his team gets a power play, and the official record
charges only the offender. That drawing penalties repeats from year to year has been public
knowledge for a decade, and public player-value models count it. This work tests three things
that had not been tested: whether the skill belongs to the player or to everyone on the ice,
whether it survives a change of referees, and whether coaches reward it with ice time. It also
values a drawn penalty in wins, ties part of the skill to measured skating speed (NHL Edge), and
repeats the construction where drawn fouls are already counted (NBA, WNBA, NCAA men's and women's
basketball, soccer).

## Headline results

| Claim | Evidence |
|---|---|
| The skill is real and persists out of sample | predicting held-out-season draw totals: r 0.25 (exposure) → 0.53 (context) → **0.72** (+skill); held-out rates monotone in prior skill quartiles (0.41→0.81/hr) |
| The raw rate carries it | a player's prior raw draws per hour, pulled toward the forward/defence average, predicts the same held-out totals at **0.72** (model minus raw rate +0.002, 95% CI −0.003 to +0.008; script 20). Adjusting for context changes almost nothing: the rate is the player's own |
| It is career-stable | positive in **14 of 14** adjacent season pairs since 2010 (mean r=0.59); career split-half 0.89 |
| It is individual, not possession or team | r=0.087 vs on-ice penalty "tilt" (which is possession); team-environment controls move skill ratings by nothing |
| It is not a referee artifact | reliability across disjoint referee crews (0.54) matches random-split calibration (0.55); full 46-official census |
| It has win value | +0.017 win probability per drawn EV minor (state-conditional), independently triangulated via goal conversion (+0.127 goals over a computed even-strength baseline) |
| Mechanism: speed | two stable routes (provocation vs puck-carrier); carrier route ages faster (−4.0 vs −2.6%/yr); NHL Edge measured speed predicts carrier-route draws (partial r 0.16–0.23, possession-controlled) but not provocation draws (0.04) |
| It generalizes | same construction: NBA (reliability 0.94), WNBA (0.86), NCAA M/W (YoY 0.54–0.68), soccer (0.71 pooled; men's international tournaments 0.64, women's competitions 0.73) |
| It comes with penalties taken | career draw and take ratios correlate 0.50 (n=593); about one player in six draws more than expected and takes fewer |
| Ice time does not rise with it | r=0.067 vs points/60; deployment −0.28 min/game per SD conditional on scoring, −0.18 after zone-start controls. This may reflect the roles drawers are cast in; there is no salary test yet |

## Repository map

```
abstract/        SSAC27 abstract submission (PDF)
METHODS.md       full methods log: every model, test, coefficient, and session-by-session provenance
analysis/        23 analysis scripts, one per paper section, with session-verified expected outputs (see analysis/README.md)
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
dated research log). The derived tables in `data/derived/` let you check many numbers directly;
most scripts need the source DuckDB files to rerun (see `analysis/README.md`).

## Contact

Steve Murray · stephenmurray22@gmail.com · [gamevibeanalytics.com](https://gamevibeanalytics.com)
