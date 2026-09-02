# Contact Won — Academic-Bar Research Program (2026-09-01)

Follow-on to `CONTACT_WON_METRIC_TRIALS_2026-09-01.md`. Steve's directive: lay out the full
research program in levels, then execute it level by level and see whether the finding clears an
academic research bar (CMSAC/Sloan shape). All numbers below are from real data
(`gamevibe_primary.duckdb`, 3 regular seasons 2023-24 → 2025-26, ~3,936 games, 141,711
player-games, 20-22k EV minors depending on filter), reproducible from this session
(206814c8). Analysis artifacts in session scratchpad: `l1_playergames.parquet`, `l1_theta.csv`,
`l2_rapm.csv`, `l2b_theta2.csv`, `l3_wp_events.parquet`.

## Measurand & hypotheses

**Measurand:** a player's contribution to the rate at which opponents are penalized while they
are on the ice, net of context (position, venue, opponent discipline, score state, own-team
environment) and net of penalties taken — valued in win-probability units.

- **H1 (skill/persistence):** individual draw rates persist out-of-sample beyond context.
- **H2 (value):** a drawn EV minor has measurable, non-trivial win value.
- **H3 (distinctness):** the skill is not an artifact of team quality, possession, or scoring
  talent — i.e., it is a separate (and plausibly unpriced) axis.
- **H4 (generality):** the construct ports across sports (FWAE lineage).

Scope: EV minors only (5v5/4v4/3v3, PIM=2), regular season, drawn attribution coverage 90%.

## L1 — Contextual expectation + empirical-Bayes skill (H1: CONFIRMED)

Weighted Poisson (sklearn `PoissonRegressor`, `sample_weight`=EV hours) on player-games,
fit on 2023-24 + 2024-25 only. Covariates: position dummies, venue, leave-one-out opponent
discipline (z), trailing/leading TOI share, season dummies (+ own-team draw environment in the
L2b variant).

Coefficients (log-rate): **D −0.69** vs C; **home +6.7%** (the documented referee home bias,
independently recovered); opponent discipline +10%/SD; **trailing +23% / leading +20%** vs tied
(penalties concentrate in non-tied states).

Skill layer: Gamma-Poisson empirical-Bayes multiplier θ = (drawn + k)/(μ + k/m), method-of-
moments k ≈ 6.6-7.3. Excess (non-Poisson) variance in player draw ratios = 0.146 — wide true
skill spread. θ range in practice ~0.37 (Boeser, Suter, Power) to ~2.45 (Hathaway; M. Tkachuk
2.17, Kadri 2.04, Stützle 2.00, B. Tkachuk 1.95, Marchand 1.96 — face validity exact).

**Out-of-sample test (predict 2025-26 totals, never seen in fit):**

| Predictor | corr | Poisson deviance |
|---|---|---|
| exposure only | 0.250 | 1867 |
| + context model | 0.527 | 1379 |
| + shrunken skill θ | **0.723** | **969** |

Exposure-free version (θ fit 23-25 vs raw drawn/hr in 25-26, matched n=558): pearson 0.483,
spearman 0.457; quartile means monotone **0.41 → 0.50 → 0.62 → 0.81 drawn/hr** — top-quartile
skill players draw at ~2× bottom-quartile in a season the model never saw.

## L2 — Isolation (H3: CONFIRMED, with a genuine surprise)

**L2a, penalty-tilt RAPM:** Ridge on ±1 on-ice design over 20,369 EV minors (858 skaters ≥50
events, `event_on_ice`, goalies excluded; home-intercept column recovers the event-level home
bias +0.011). Highly reliable (split-half SB 0.93) but its boards are possession boards: top =
Panarin, Fiala, Robertson, Raymond, MacKinnon; bottom = exclusively defensive-zone D (Larsson,
Parayko, Lindgren...). And it correlates with individual θ at only **r = 0.087**.
**Interpretation:** *which team wins a penalty exchange while you're on the ice* is mostly a
possession/territory phenomenon; *who personally extracts the call* is a different, individual
skill. These are two constructs, not one — worth a section of its own in any writeup.

**L2b, team-environment control:** adding own-team leave-one-out draw environment to L1 moves
θ by almost nothing (r = 0.987 with uncontrolled θ), excess skill variance survives
(0.131 vs 0.146), OOS improves slightly (0.738 / deviance 924). **Draw skill is individual,
not team-borne.** θ2 (team-controlled) is the consumption-grade skill number; stored in
`l2b_theta2.csv`.

## L3 — Value in win-probability units (H2: CONFIRMED)

Empirical ΔWP: for each of 20,224 EV minors, drawing team's actual win indicator minus the
empirical base win rate of the same (score diff ±3, 10-min time bucket, venue) state, base
surface built from all events in all games (both perspectives).

**ΔWP per drawn EV minor = +0.0171 (95% CI ±0.0056).** Stable across time buckets (+0.006
first 10 min → +0.019-0.023 late) and score states (peaks at tied/−1, shrinks at ±3 where WP
is saturated — exactly the leverage shape it should have).

Triangulation: the independent goals path (+0.118 goals/drawn minor measured earlier ×
~0.16 wins/goal) gives ≈ +0.019 — two routes, same answer.

Scale: Raymond-class net drawer (+17.7 net minors/season) ≈ **+0.30 wins/season** from the
penalty ledger alone; top-vs-bottom-decile net spread ≈ +0.4 wins. Small but real, and
concentrated in a skill nobody prices (see L4-3).

Caveat (stated in any paper): ΔWP is state-conditional but still observational — teams that
draw penalties may be better in ways the (score, time, venue) bucket doesn't capture. The
goals-path agreement and the symmetric taking-side estimate bound how much that can matter.

## L4 — Falsification battery

1. **Symmetry:** taking-team ΔWP = −0.0171 — mirrors the drawing side exactly. Honest note:
   with a two-sided outcome and perspective-symmetric baselines this is close to a construction
   identity; report it as a consistency check, not independent evidence.
2. **Embellishment:** high-θ players take ~8× more embellishment calls than low-θ players
   (top-quartile mean 0.29 vs bottom 0.04; r = 0.25) — mechanism evidence that the skill has a
   selling-contact component refs occasionally punish. But 110 total embellishment calls vs
   ~20k drawn minors ≈ 0.5% contamination — quantitatively negligible to the value story.
3. **Orthogonality to scoring (the unpriced claim):** corr(θ2, points/60) = **0.067**
   (n=685, 15+ EV hrs; by position: C 0.05, L 0.21, R −0.02, D 0.11). Draw skill is essentially
   uncorrelated with the thing the market pays for. Combined with L2 (uncorrelated with
   possession tilt) this is the distinctness result: penalty-drawing is its own axis.
4. **Known-bias recovery:** the model independently recovers the documented home-team
   officiating bias twice (player-level +6.7%, event-level +0.011) — a free external-validity
   anchor.

## L5/L6 — Limits & cross-sport (H4: partially supported, honestly bounded)

- Soccer: FWAE *is* this construct (fouls won above expected) — the original sibling; the
  Swish-selection provenance is the H4 anchor.
- NBA & PWHL: fouls/penalties currently export committer-only — drawn-by attribution is a
  pipeline extension, not a modeling gap; H4 untestable there until the feeds carry it.
- No officials table in-DB → referee-clustering robustness (makeup calls, per-ref propensity)
  is the single biggest known missing control. No zone coordinates on penalty events → no
  zone-of-infraction control. Both are data-acquisition items, both fixable from NHL API.
- 3 seasons is the corpus; a 10-season replication is the obvious strengthening move.

## Does it clear an academic bar? (updated end of session 2)

Of the four gaps named at end of session 1, three closed the same night: **referee controls**
(disjoint-crew reliability = random-split reliability), **multi-season replication** (15
seasons, 14/14 positive pairs, career SB 0.89), and a deployment-based **pricing test**
(θ gets *less* ice time conditional on scoring, t=−3.65). Novel claims now number four:
(1) individual extraction ≠ on-ice penalty tilt (two constructs), (2) orthogonal to scoring,
(3) the two-route taxonomy (provocation vs puck-carrier, each a stable style trait),
(4) the aging curve. Remaining for a full paper: literal cap-hit regression (external salary
data) and a second sport's drawn-by feed. **Assessment: this is now full-paper shape minus the
salary table; poster/short-paper submittable as-is.**

## Session 2 additions (same night): replication, referees, deployment pricing

### 15-season replication (discovery DB) — H1 at scale

`data/gamevibe_discovery.duckdb` (NOT drive_data/ — path in root AGENTS.md is stale) holds
2010-11 → 2024-25 with `drawing_player_id` at ~92% and full shift TOI. Per-season
position-expected Poisson + per-season EB shrinkage (EV = situationCode '1551'/'1441'/'1331',
PIM=2, exposure = all-situation TOI hours — noted denominator difference vs the 3-season study's
EV-TOI):

- **YoY θ persistence positive in all 14 adjacent season pairs: mean r = 0.587 (0.476–0.650).**
  No era where the skill vanishes; spread (excess var 0.11–0.24) present every season.
- **Career split-half (odd vs even seasons, 20+ expected draws each half): r = 0.807,
  SB = 0.893, n = 597.** Draw skill is close to a career-stable trait.
- All-time board 2010-25 (40+ expected): B. Smith 2.72, Hathaway 2.49, Kadri 2.22 (15 seasons),
  Dorsett, Downie, Cousins, Stützle, C. Miller, Bunting, M. Tkachuk… plus Jeff Skinner 1.96
  over 15 seasons — the era-spanning archetype plus the league's famous draw magnet.

### Deployment pricing test — "unpriced" with teeth

Regression: TOI/min-per-game ~ position + points/60 + θ2 (n=726, 60+ GP, 3-season corpus).
Points are heavily priced (+2.9 min/game per SD, t=32). **θ2 coefficient: −0.28 min/game per SD
(t = −3.65)** — conditional on scoring and position, higher draw skill gets *less* ice time.
Framing rule: this shows the skill is not positively priced in deployment (agitator-role
players are cast bottom-six regardless); it does NOT license "coaches are wrong" (θ correlates
with role/style and unmeasured defensive value). Cap-hit regression remains the literal
market-pricing test — needs external salary data.

### Referee robustness — CLOSED (the biggest limitation, resolved)

Officials are NOT in either DB but ARE in the NHL API: `api-web.nhle.com/v1/gamecenter/{id}/
right-rail` → `gameInfo.referees` (landing does NOT carry them). Fetched all 3,936
regular-season games (46 distinct referees, 0 missing; scratchpad `game_referees.csv`).

- **Referee-disjoint split-half:** hash each referee into set A or B, keep only games whose
  two refs fall in the same set (749 + 1,189 games), compute each player's draw ratio
  separately under crew-set A vs crew-set B. **r = 0.509 (SB 0.674, n=199)** vs a random game
  split of the same data **r = 0.485 (SB 0.653)**. Skill measured under completely disjoint
  referee crews agrees with itself exactly as well as any random split — draw skill is NOT a
  favored-by-specific-referees artifact. This retires the paper's largest stated limitation.
- Referee heterogeneity in call rates is real but modest: 41 refs (50+ games) range
  4.20–5.16 EV minors/game; excess sd beyond sampling ≈ 0.19/game (~4% of mean).

### Two-route mechanism taxonomy (novel)

Split drawn EV minors into **provocation** calls (roughing, cross-checking, slashing,
unsportsmanlike) vs **puck-carrier** calls (tripping, hooking, holding, interference,
holding-the-stick). Findings:

- High-θ players over-index on provocation (1.79× roughing, 1.33× cross-checking) and
  under-index on hooking (0.65×) vs low-θ.
- **The route is itself a stable trait:** season split-half SB = 0.734 (n=331); adjacent-season
  r = 0.457 over 1,034 season pairs across 15 seasons.
- It separates equal-skill players by craft: M. Tkachuk prov-share 0.72 vs McDavid 0.14 and
  Makar 0.12 at similar θ — two different skills producing the same call.
- Era wrinkle: league provocation share ~0.30–0.33 for a decade, rising to 0.35 (2023-24) and
  0.38 (2024-25).

### Aging curve

Debut-cohort (first corpus season ≥2011, μ≥5) draw ratio by career year: monotone decline
1.135 (rookie) → 0.913 (year 9); within-player fixed-effect slope **−0.027/yr** (4+ season
careers). Draw skill ages like a speed-dependent skill — peak extraction is young.

**Real-birthdate upgrade (2026-09-02):** fetched birthdates for all 1,137 replication-corpus
players (api-web `/v1/player/{id}/landing`, 100% success; CSV in `data/reference/`). True age
curve: monotone 1.241 (age 20) → 0.757 (age 37), no plateau — peak is the youngest observed
age. Within-player slope −0.029/yr.

**Route-specific aging — the mechanism test (preregistered prediction, confirmed):** if the
decline is speed-driven, the puck-carrier route should decay faster than the provocation
route. Within-player, cluster-bootstrapped: **carrier −4.0%/yr (SE 0.0011) vs provocation
−2.6%/yr (SE 0.0005)** — the speed-dependent route ages ~1.5× faster. You stop beating
defenders long before you stop getting under their skin.

## Session 3: H4 second sport — NBA, CONFIRMED (2026-09-02)

The "no second sport" wall had a door: `beta-sports/basketball/nba/data_raw/
official_rapm_source/` holds **5,830 cached official PlayByPlayV3 games (2019-20 → 2023-24)**
from the RAPM research cache. V3 foul rows name only the committer (parenthetical = referee),
but drawn-by is structurally inferable: attribute each FT trip (subType contains "1 of") to its
shooter when an opponent foul precedes it within 6 actions. 139,397 trips extracted.

Construction identical to hockey (trips drawn per minute vs position-season expectation):

- **Split-half SB = 0.940** (n=1,930 half-season pairs); **YoY r = 0.858–0.879 in all 4
  adjacent pairs** — more reliable than hockey (more events per player).
- Pooled 2019-24 board (100+ expected): Giannis 3.90×, Zion 3.00×, Luka/Embiid 2.79×, Butler
  2.68×, SGA, Trae, DeRozan, Morant, Harden. Bottom: Hauser 0.11×, Snell, Bullock, Ellington —
  all catch-and-shoot 3&D. Face validity exact at both ends.
- Honest caveats: basketball partially prices this already (FTA is a visible stat) so the
  "unpriced" leg is hockey-specific; non-shooting fouls only enter via penalty-situation FTs;
  technical-FT contamination minor. The generality claim is about the *construction* (contextual
  expectation → skill ratio → persistence) transferring, and it transfers with room to spare.
- Analysis: scratchpad `nba_pdae{,_analysis}.py`, `nba_pdae_playerseasons.parquet`.

**NBA value leg (same night):** empirical ΔWP per drawn foul trip, same same-state-baseline
method as hockey L3 (138,347 trips vs 254,558 sampled game states, score-band × 6-min bucket ×
venue): **+0.0102 ± 0.0021** — ~60% of a hockey minor, with correct leverage shape (peaks
+0.025 in the final 6 minutes: bonus + clock-stop). Key asymmetry to state in the paper: in
basketball this value flows through the drawer's own FT points (visible, priced); in hockey
the drawn penalty never touches the drawer's boxscore (hidden). Same skill, opposite market
visibility — which sharpens the hockey unpriced claim rather than diluting it.

Cross-sport standing: soccer (FWAE, original) + hockey (this program) + **NBA (replicated,
skill + value legs)**; PWHL blocked at feed level (committer-only).

## Session 5 (2026-09-02, "you sure it's exhaustive?"): three more threads, all landed

### WNBA — FOURTH league (H4 extended to a women's league)

`beta-sports/basketball/wnba/data_raw/official_rapm_source/` (1,306 games, 2020-2025; note
different nesting than NBA: `play_by_play_v3.game.actions`, `box_score_traditional_v3.
boxScoreTraditional`). Identical construction, 25,383 trips: **split-half SB = 0.860, YoY
r = 0.723–0.821 in all 5 pairs** despite 40-game seasons. Board: A'ja Wilson 2.34×, Cambage,
Stewart, Angel Reese, Chennedy Carter; bottom: Sue Bird 0.26×, Johannès — perimeter
specialists. Four leagues: soccer, NHL, NBA, WNBA.

### Playoff whistle-swallowing — folklore REFUTED

Playoff EV-minor rate is **+16% HIGHER** than regular season (0.648 vs 0.557 per skater-EV-hr;
256 playoff games) — referees do not pocket the whistle on minors. And skill transfers ~1:1:
top-quartile RS θ (1.38) draws at 1.352 in playoffs; bottom quartile (0.64) at 0.660;
RS θ → playoff ratio r = 0.348 (attenuated only by small playoff exposure). Draw skill is
playoff-robust.

### Even-up refereeing — replicated, and it's huge

Sequential EV minors within games (16,452 pairs): P(next minor on home) = 0.380 after a home
penalty vs 0.584 after an away penalty — **+0.204 swing, z = 26.8**; tied-score-only control
+0.217; decays with gap since last call (+0.221 at 2-5 min → +0.074 past 20 min), the temporal
signature of make-up calling rather than team-style persistence (style would push the swing
negative). Replicates Scorecasting's make-up-call result in-corpus. Program relevance: net
ΔWP (+0.0171) ≈ window-only goals path (+0.019) ⇒ the make-up tax on a drawn minor is real
but small (~0.002 WP) — the drawn penalty's value survives the ref's rebalancing.

## Session 6 (2026-09-02, "keep pulling"): leagues 5-6 + moderators + nulls + the 2×2

### MBB and WBB — leagues FIVE and SIX

`beta-sports/basketball/{mbb,wbb}/data_raw/gated_seasons/*_lossless.parquet` (ESPN
sportsdataverse pbp, ~2M rows/season, MBB 2017-2026, WBB 2016-2026). Feed quirks: committer
only on `PersonalFoul`; **missed FTs absent entirely** (trips identified via made FTs only,
~9% undercount, biased against poor FT shooters/hack-a targets); no minutes (exposure = pbp
row-appearances, a usage proxy). With those stated limits: **MBB YoY r = 0.54–0.66 across all
9 pairs (n≈1,500-1,800 each)**, pooled #1 = AJ Dybantsa (the #1 recruit). **WBB YoY
r = 0.59–0.67 across 8 pairs**, #1 = S'Mya Nichols 2.63×. Six leagues total:
soccer, NHL, NBA, WNBA, MBB, WBB.

### Size moderator (real birthdate+bio fetch, saved data/reference/player_bios.csv)

Shorter players draw slightly more within every position (height r −0.05 to −0.21, strongest
D at −0.21; weight similar; BMI null). Direction consistent with the speed mechanism +
refs protecting smaller players; magnitude modest — garnish, not headline.

### Hot hand — CLEAN NULL (strengthens the model)

Within-game overdispersion 1.045 vs 1.0 pure-Poisson given own rate (n=861);
P(2nd draw | 1st, same game) = 0.082 obs vs 0.076 Poisson-expected. Drawing is a rate skill,
not a streaky one — retroactively validates the Poisson machinery of L1.

### Discipline is its own equally-stable skill → the 2×2 archetype map

Taken-side EV minors, same construction, 15 seasons: **YoY mean r = 0.619** (14 pairs;
drawn-side was 0.587). Career draw-ratio × take-ratio correlate r = 0.489 (n=546, 40+
expected both sides). Median-split 2×2 (89/184/184/89):
- **MAGNET** (draw+, take−): Skinner, Gostisbehere, Ehlers, McDavid — the unpriced corner.
- **AGITATOR** (draw+, take+): Hathaway, Kadri, Bunting, Dorsett.
- **INVISIBLE** (draw−, take−): Garrison, Duncan Keith, Vrbata.
- **LIABILITY** (draw−, take+): Dowd, Verhaeghe, Goodrow, Byfuglien.
Scripts: scratchpad `mbb_pdae.py`/`wbb_pdae.py` (+ copies in each league's analysis/),
`fetch_bios.py`.

## Session 7 (2026-09-02, "keep pulling" ×2): uniform soccer numbers, team lever, timing nulls

### Soccer re-run through the identical template (six-league table now uniform)

Swish league parquets are carrier-extracts (no foul events — matches the 09-01 finding), but
`soccer/World_Cup/data/events_raw/` holds **1,179 full StatsBomb event files (3.4GB, all 8
tournaments)**. Slim-parsed (scratchpad `soccer_events_slim.parquet`): 25,932 Foul Won events.
Same construction (position-group expectation, exposure = event-appearances): **split-half
SB = 0.705 (n=1,397)** pooled across tournaments. Top-10 includes **Eden Hazard** (the
sport's canonical foul-winner) among women's-tournament leaders (Oberdorf, DeMelo). Agitator
tax exists but weaker than hockey: corr(won, committed) = 0.301 (NHL 0.489).

### Team construction — the GM lever, honestly bounded

Roster-aggregated prior-season player skill (cumulative shrunken draw & take ratios ×
current exposure) vs actual team-season net EV differential, 15 seasons, 393 team-seasons:
**levels r = 0.450**; actual spread sd ≈ 20 net minors/season, best-to-worst range 139 minors
≈ 16 goals ≈ **2.7 wins**. But YoY *changes* r = 0.134 — one-season diffs are mostly noise;
present the levels claim, not a churn-tracking claim.

### Leverage timing — null #3, with a gem inside

Draw timing is flat across skill quartiles (close-game share 0.66-0.68, close&late 0.13-0.15
for every quartile): elite drawers win by volume, not clutch timing. The gem: draws occur in
close&late states at 0.147 vs 0.168 exposure baseline — **referees swallow ~12% of whistles
late in close games** (within-game), even though playoff rates are UP 16%. The folklore was
right about the moment and wrong about the month.

## External walls (verified 2026-09-02, not guesses)

- **Cap-hit regression:** CapWages has the exact API needed (`capwages.com/api-docs`,
  `GET /players/:slug` returns contract detail incl. cap hits, JSON) but requires a paid
  subscription key. PuckPedia's data API is likewise commercial. → Steve decision: a CapWages
  key unlocks the literal "unpriced" test in an afternoon.
- **Second sport drawn-by:** verified against real raw feeds, not exports — NBA local raw
  events (181 games) carry committer + referee only (the parenthetical in foul descriptions is
  the REF, e.g. "Okani P.FOUL (P1.T1) (M.Kallio)"); PWHL raw penalty events duplicate the
  committer into `player_two_*` (checked game 10, Compher/Compher); WNBA primary DB is
  aggregate-only (3 tables, no pbp). NBA's stats API does carry a foul-drawn person id in its
  v3 pbp — a fresh-fetch pipeline extension remains the unlock, with the known Akamai risk.

## Reproduction notes

- All queries: EV = manpower_state IN ('5v5','4v4','3v3'), PIM=2 via
  `COALESCE(pbp.penalty_minutes, events.penalty_minutes)`, season_type='2', goalies excluded.
- Scores live on `play_by_play_raw` (NOT `events`); `score_state` vocabulary is
  `Leading_1/Leading_2+/Tied/Trailing_1/Trailing_2+`; DuckDB requires explicit `AS` aliases.
- statsmodels is not installed; weighted Poisson = sklearn `PoissonRegressor` with
  `sample_weight` = exposure hours on the rate.

## Session 8 (2026-09-02): The Whistle Ledger — referee dynamics as a section of the paper

Full study, scripts, and logs: `whistle_ledger_study/` (workspace root, committed). This session
turned the even-up robustness check (Session 5) into a paper section with its own claims. All
three leagues with penalty/foul sequences on disk were used: NHL (20,315 EV minors, 3 seasons,
referee pairs for every game), NBA (228,370 fouls, 5 seasons, calling official named on every
foul), WNBA (45,659 fouls, 6 seasons, no officials in feed).

**Claim A — the make-up effect is a cumulative ledger, not last-call memory.** Holding the
previous call fixed, the balance of all earlier calls still drives the next one: NHL minute-level
hazard −0.140 log-odds per net minor (z −20; 2.1× hazard between ledger +3 and −3; −0.144/−0.138/
−0.140 by season; survives >5 min after the last call), NBA −0.023 per net foul (z −18), WNBA
−0.031 (z −10). Direction rules out team-discipline persistence (which would push the sign
positive). Decay of the previous-call swing: NHL peaks +0.216 at 2-5 min, +0.06-0.09 past 20
min; NBA/WNBA +0.10 within 30 s, ~0 past 4 min. This is the generalisation of the Scorecasting
make-up result: refs carry a running account, not a one-call memory, and the account looks the
same in three leagues.

**Claim B — make-up propensity is a real but modest individual trait, and the norm has
tripled in strength since 2009.** In 3 seasons (39 refs, crew-level attribution) NHL excess
variance was zero — a power problem, not absence. With referees fetched for all 20,400
regular-season games 2009-10..2025-26 (49 refs, median 734 games): even-up rate 0.536-0.612,
χ² 130 on 48 df, split-half SB 0.64, crew-separated ridge SB 0.57, early-vs-late era r 0.25.
NBA (70 officials, call-level): split-half SB 0.50 on even-up rate, 0.56 on the per-official
ledger slope. Call *volume* is far more stable (NHL SB 0.78, NBA 0.95). Secular trend, NHL:
even-up rate 0.541 (2009-10) → 0.55-0.58 (2010s) → 0.600/0.613/0.606 (2023-26); previous-call
coefficient −0.17 → −0.51; cumulative-ledger coefficient −0.05 → −0.27. The referee-disjoint
reliability test in Session 2 still stands: individual propensities differ, but every crew
applies the ledger, so draw skill measured under disjoint crews agrees with itself.

**Claim C — a drawn penalty's value does not depend on the ledger (PDAE keeps a flat ΔWP).**
Same-state-baseline ΔWP rebuilt from every `game_state` row reproduces +0.0172 (SE 0.0029,
n=20,170). By the drawing team's ledger it runs 0.009-0.028 with no trend (slope +0.0035/net
minor, z 1.4). The components move and cancel: at drawer ledger +3 the next call lands on the
drawer only 24% of the time (vs 41% at −3), but the ensuing PP converts 15.8% vs 20-21%
mid-ledger. So the make-up tax and the conversion penalty offset; the +0.017 stands as a
state-independent number.

**Claim D — momentum and the whistle differ by sport, and in the NBA it is a league-wide norm
or a flow mechanism, not a referee trait.** NHL: exposure-based hazard is null once 2-minute
shot share is controlled (−0.008/SD, z −0.8; null in every season). The small event-level
side-selection residual (+0.068/SD) is a Corsi-vs-Fenwick artifact (loads on the possession
index = blocked attempts; hits differential z −0.3; defense index opposite sign). NBA: the
surging team is called *more* in all five seasons (hazard +0.047 to +0.065/SD, z 8-12 each;
+14% top vs bottom quintile), strongest in loose-ball/non-shooting fouls, *reversed* for
offensive fouls (−0.045). Two separation tests: (1) controlling the surging team's own steals +
blocks + forced turnovers and both teams' FGA in the prior 60 s leaves momentum unchanged
(0.054 → 0.052); (2) the per-official momentum slope has no excess variance (χ² 80.9 on 69 df,
split-half r 0.03) while the ledger slope on the same calls keeps its SB 0.54. So "the surging
team gets whistled" in the NBA is uniform across officials and not explained by countable
defensive events — consistent with a league-wide norm or a game-flow mechanism the pbp can't
see (set half-court defence after made baskets is the obvious candidate), and inconsistent with
individual referee bias. Never phrase as "refs punish the hot team."

**Claim E — draw skill is not timing.** Exposure-adjusted, the owed window (≤300 s after own
team's penalty, EV) doubles every skater's draw rate (0.838 vs 0.419/hr); θ2 quartile ratios
1.97/1.75/2.13/2.22; corr(θ2, log rate ratio) 0.16 (bootstrap CI 0.08-0.25). Elite drawers ride
the tide, they don't create it.

Placement in the paper: extends §10 "Officiating by-products" into a full section (ledger,
homogeneity, flat ΔWP, momentum split); Claim C also simplifies §06 (value is state-independent
w.r.t. the ledger); Claim B sharpens §05's "not the refs" leg.

## Session 8 (2026-09-02): NHL Edge tracking data confirms the speed mechanism DIRECTLY

Steve's suggestion: tie NHL Edge skating-speed data to the route taxonomy. Endpoint (found
via Zmalski/NHL-API-Reference): `api-web.nhle.com/v1/edge/skater-detail/{pid}/{season}/2` —
carries speedMax (mph + percentile), burstsOver20, zone-time %. Fetched all 888 rated players
× 3 seasons (2,362/2,664 player-seasons had data; CSV at data/reference/edge_speed_2023_2026.csv).

**The preregistered-style differential prediction: measured speed should predict carrier-route
draws and not provocation draws. CONFIRMED (n=664, position-controlled partial correlations):**

| Predictor | carrier-route draws/60 | provocation draws/60 |
|---|---|---|
| max speed | **+0.166** | +0.037 |
| bursts>20mph per hr | **+0.202** | +0.051 |
| max speed (+ OZ% possession control) | **+0.163** | +0.037 |
| bursts (+ OZ% control) | **+0.225** | — |

Carrier draws/60 by measured speed quartile: 0.242 → 0.322 → 0.335 → 0.377 (monotone, +56%
slow-to-fast); provocation flat. Overall θ2 vs speed only +0.084 — correct and expected:
provocation-route drawers dilute the aggregate; the mechanism is route-specific.

**The mechanism is now triangulated three independent ways:** penalty-type composition
(defenders' stick fouls), differential aging (carrier −4.0%/yr vs prov −2.6%/yr), and
league tracking data (speed loads on carrier route only, possession-controlled). Sloan
abstract updated. Coverage note: Edge data exists 2021-22+ only; 302 player-seasons missing
(mostly low-TOI).

## Session 9 (2026-09-02): zone-start deployment control — the pricing claim survives

Fetched NHL Edge zone-start data (skater-zone-time endpoint) for all rated players
(2,362/2,664 player-seasons; CSV at data/reference/edge_zonestarts_2023_2026.csv). Re-ran the
deployment regression (TOI/game ~ position + points/60 + θ2) with usage controls, n=726:

| Specification | θ2 coefficient | t |
|---|---|---|
| baseline (pos + pts/60) | −0.279 min/game per SD | −3.65 |
| + OZ-start% | −0.286 | −3.75 |
| + OZ & DZ start% | **−0.177** | **−2.44** |

~1/3 of the raw effect was deployment-flavored; a significant negative residual survives full
zone-usage controls. The "not positively priced in deployment" claim now withstands the
obvious usage-confound counterattack. Abstract updated (464/500 words) + PDF regenerated.
