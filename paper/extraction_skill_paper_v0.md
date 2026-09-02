# The Extraction Skill: Winning Contact Is a Persistent, Individual, and Unrewarded Talent Across Six Leagues

**Steve Murray** · GameVibe Research · stephenmurray22@gmail.com · gamevibeanalytics.com

*Draft v0.1 (2026-09-02) — full-paper draft accompanying the SSAC27 abstract submission.
One section (§7.3, contract pricing) awaits external salary data and is marked accordingly.
All other results are final and reproducible from this repository.*

---

## Abstract

We model even-strength penalties drawn as a weighted Poisson process with contextual
covariates and an empirical-Bayes player skill multiplier, fit on two NHL seasons and
validated on a third. The skill layer nearly triples out-of-sample predictive correlation
over exposure alone (0.25 → 0.72), persists across fifteen seasons (positive in 14 of 14
adjacent pairs; career reliability 0.89), survives team-environment controls and a
disjoint-referee-crew test, and replicates under one identical construction in five more
leagues — NBA, WNBA, men's and women's NCAA basketball, and international soccer. A drawn
NHL minor is worth +1.7 points of win probability, triangulated by an independent goals
path; the NBA analogue is +1.0. The skill decomposes into two stable stylistic routes —
provocation and puck-carrier extraction — whose differential aging (−2.6 vs −4.0%/yr) and
differential relationship to measured skating speed (NHL Edge tracking) confirm a speed
mechanism. It is orthogonal to scoring (r = 0.07), orthogonal to on-ice penalty tilt
(r = 0.09, which we show is possession), and negatively associated with deployment even
after zone-start controls. Hockey's accounting renders the skill structurally invisible;
nobody is paying for it.

## 1. Introduction

Every drawn penalty hands over a power play. Yet hockey's box score records the event only
against the offender: the player who *caused* the penalty — who forced a beaten defender
into a desperate stick foul, or provoked an opponent into retaliation — receives no
statistic of any kind. The value flows to the team; the credit flows to nobody.

This paper asks three questions about that invisible transaction. **Is drawing penalties a
skill** — a persistent individual trait rather than luck or a team byproduct? **What is it
worth**, in the currency teams actually compete for? And **does anyone pay for it** — in ice
time, in roster construction, in dollars?

The answers, in brief: it is among the most stable rate skills measurable from public
hockey data, persisting across entire careers and fifteen seasons of league history; a
drawn even-strength minor is worth about 1.7 points of win probability, making an elite
net-drawer worth roughly a third of a win per season through the penalty ledger alone; and
no — the skill is uncorrelated with what the market prices (scoring), and its carriers
receive slightly *less* ice time than their scoring predicts, a result that survives
deployment controls built from the NHL's own tracking data.

Two findings elevate this beyond a single-metric study. First, **individual extraction and
on-ice penalty "tilt" are different constructs**: which team wins the penalty exchange
while a player is on the ice is mostly a possession phenomenon, nearly uncorrelated
(r = 0.087) with who personally extracts calls. Any on-ice penalty-differential statistic —
including the penalty components of public WAR models — conflates a team effect with an
individual skill. Second, **the skill has a mechanism**: it decomposes into two stable
stylistic routes whose different aging profiles and different relationships to measured
skating speed confirm that one route runs on speed and the other on provocation. The same
construction, applied to raw event data in five other leagues, finds the same skill —
led by exactly the players folklore would nominate (Antetokounmpo, Wilson, Hazard).

## 2. Related work

Penalty differential is not an unknown quantity in hockey analytics. Behind The Net
published penalties drawn per sixty as early as the late 2000s; Tulsky (2011-2013) placed
a goal value on drawn penalties; Evolving-Hockey's WAR/GAR framework has carried an
explicit penalty component (draws and takes) for years, and penalty rates appear on every
major public stats site. In basketball, free-throw generation is a visible, priced part of
the sport's economy; in soccer, fouls won is a tracked and celebrated skill.

What this prior work has not done, to our knowledge: separated individual extraction from
on-ice tilt and shown they are nearly orthogonal constructs; decomposed drawing into
stylistic routes and shown the routes are themselves stable traits with distinct aging;
validated the skill against referee-crew identity using a full officials census; confirmed
a speed mechanism with league tracking data; applied one identical construction across six
leagues; or tested whether the skill is rewarded in deployment. Those are this paper's
contributions. Our value-per-penalty estimate (+0.118 goals, even-strength minors only)
sits slightly below older all-situations public estimates (~0.15-0.17), consistent with
scope differences.

## 3. Data

**NHL core corpus:** official public play-by-play (api-web.nhle.com), three regular
seasons 2023-24 → 2025-26: 3,936 games, 141,711 player-games, 20,369-22,024 even-strength
minor penalties depending on filter. Scope throughout: minors (2:00) at 5v5/4v4/3v3, with
drawn-player attribution present on 90% of penalty events. **NHL replication corpus:**
fifteen seasons 2010-11 → 2024-25 with drawn attribution (~92%) and full shift-level ice
time. **Officials census:** referee pairs for all 3,936 core-corpus games (46 referees,
100% coverage), fetched from the league's gamecenter feeds. **Tracking:** NHL Edge
skating-speed, burst, and zone-start data for all rated players (2,362 player-seasons).
**Replication leagues:** NBA official V3 play-by-play (5,830 games, 2019-20 → 2023-24);
WNBA (1,306 games, 2020-2025); NCAA men's and women's basketball (ESPN feeds via
sportsdataverse; 10 and 11 seasons); international soccer (StatsBomb open data; 1,179
tournament matches). In the basketball family, drawn-by attribution is inferred
structurally by linking each free-throw trip to the opponent foul immediately preceding it
(within six actions); soccer uses StatsBomb's labeled "Foul Won" events.

## 4. Methods

### 4.1 Contextual expectation

We model player *i*'s drawn even-strength minors in game *g* as Poisson with rate
λ_ig = exposure_ig · exp(x_ig'β), estimated as an exposure-weighted Poisson regression on
the drawn-per-hour rate. Covariates: position; venue; leave-one-out opponent discipline
(the opponent's season penalty-taking rate excluding the game in question, standardized);
trailing and leading time shares; season; and, in the controlled variant, the player's own
team's leave-one-out draw environment. The model is fit on 2023-24 and 2024-25 only.

Fitted context effects independently recover known officiating facts: home skaters draw
+6.7% more (the documented home bias, which the event-level design of §4.3 recovers again
as +0.011 on its intercept); penalties concentrate in non-tied states (trailing +23%,
leading +20% vs tied); defensemen draw at roughly half the forward rate.

### 4.2 Skill layer

Player skill is a Gamma-Poisson empirical-Bayes multiplier
θ_i = (drawn_i + k) / (μ_i + k/m), with prior strength k fit by method of moments
(k ≈ 6.6-7.3 across specifications) and m the population mean ratio. Excess (non-Poisson)
variance in player draw ratios is 0.13-0.15 — a wide true-talent spread. θ ranges in
practice from ~0.37 to ~2.45.

### 4.3 Isolation designs

**Tilt vs extraction.** A ridge regression over the 20,369 even-strength minors with full
on-ice data: outcome +1 if the home side drew the call, features ±1 for each on-ice skater
(858 skaters with ≥50 events; goalies excluded; home-intercept column). This estimates each
player's effect on *which team wins a penalty exchange while they are on the ice* —
deliberately a different question from personal extraction.

**Team environment.** The §4.1 model refit with own-team leave-one-out draw environment.

**Referee identity.** Each of 46 referees is hashed into crew-set A or B; games officiated
entirely within one set (749 and 1,189 games) yield two draw-ratio measurements per player
under disjoint judges. Cross-crew agreement is compared against a random game-split
calibration on the identical subset.

### 4.4 Valuation

For each even-strength minor we compute the drawing team's realized win indicator minus
the empirical win rate of identical (score differential ±3, ten-minute time bucket, venue)
states, with the baseline surface built from all game states in all games, both
perspectives. No model assumptions enter. An independent path — measured scoring rate
inside the two-minute drawn-penalty window (20.3%, which independently matches league
power-play conversion) against the even-strength baseline (~8.5%) — provides
triangulation.

### 4.5 Routes, aging, and mechanism

Drawn minors are classified by infraction: **provocation** (roughing, cross-checking,
slashing, unsportsmanlike) vs **puck-carrier** (tripping, hooking, holding, interference,
holding-the-stick). Aging uses real birthdates (fetched for all 1,137 replication-corpus
players) with within-player fixed-effects slopes and cluster-bootstrapped errors. The
mechanism prediction — the carrier route, and not the provocation route, should relate to
measured skating speed — is tested against NHL Edge maximum speed and bursts-over-20mph,
position- and possession-controlled (offensive-zone time share).

### 4.6 Cross-league replication

The identical three-step construction (contextual or positional expectation → shrunken
skill ratio → persistence test) is applied per league. College feeds omit missed free
throws (~9% trip undercount, biased against hack-a targets) and minutes (exposure =
event-row appearances); both limitations are carried explicitly.

## 5. Results: the skill is real, individual, and career-stable

**Out-of-sample validation.** Predicting 2025-26 player draw totals — a season never seen
in fitting — exposure alone achieves r = 0.250 (Poisson deviance 1867); the context model
0.527 (1379); context plus shrunken skill **0.723 (969)**. The exposure-free version:
players sorted into quartiles by θ fit on 2023-25 show perfectly monotone held-out draw
rates of **0.41 → 0.50 → 0.62 → 0.81 per hour** — the top quartile draws at twice the
bottom's rate in unseen data.

**Fifteen-season persistence.** Year-over-year θ correlation is positive in **all 14
adjacent season pairs** (mean r = 0.587, range 0.476-0.650), with excess skill variance
present in every season (0.11-0.24). Splitting careers into odd and even seasons yields
reliability **r = 0.807 (Spearman-Brown 0.893, n = 597)**. The fifteen-year leaderboard is
face-valid at both ends: Smith (2.72×), Hathaway (2.49×), Kadri (2.22× across fifteen
seasons), Dorsett, Downie, Cousins — career agitators — with the notable exception of a
24-year-old skill forward (Stützle, 2.12×, 7th of 641).

**Not possession.** The penalty-tilt ridge design is highly reliable (split-half SB 0.93)
but its leaderboard is a possession leaderboard — Panarin, Fiala, Robertson, MacKinnon on
top; defensive-zone defensemen at the bottom — and it correlates with individual θ at only
**r = 0.087**. On-ice tilt is territory; extraction is a skill. **Not the team:** adding
own-team draw environment moves θ by nothing (r = 0.987 with the uncontrolled version)
while slightly improving out-of-sample fit (0.738). **Not the referees:** draw ratios
measured under disjoint referee crews agree at r = 0.51, statistically identical to the
random-split calibration (0.49).

## 6. Results: value

The state-conditional estimate: **ΔWP = +0.0171 per drawn even-strength minor (95% CI
±0.0056)**, largest exactly where leverage theory predicts (tied or trailing by one;
shrinking toward zero at ±3), and in agreement with the independent goals path
(+0.118 goals × ~0.16 wins/goal ≈ +0.019). An elite net drawer (drawn minus taken, above
expectation) adds **~0.3 wins per season** through the penalty ledger; the best-to-worst
roster-level spread across fifteen seasons of team-seasons is ~139 net minors ≈ **2.7
wins**. Roster draw skill is acquirable: prior-season player skill aggregates predict a
team's next-season net differential at r = 0.45 (levels; year-over-year changes are far
noisier at 0.13, so the claim is about assembly, not transactions). In the NBA the
analogous estimate is **+0.0102 WP per drawn foul trip (±0.0021, n = 138,347)**, peaking
in the final six minutes where the bonus and stopped clock bind — with the important
asymmetry that basketball's version flows through the drawer's own visible free-throw
points, while hockey's never touches the drawer's stat line.

## 7. Results: mechanism, generality, and the market

### 7.1 Two routes, and why the skill ages

Route mix is a stable trait (season split-half SB 0.73; r = 0.46 across 1,034 season pairs
spanning fifteen seasons) that separates equal-skill players by craft: M. Tkachuk draws
72% of his calls by provocation; McDavid 14% and Makar 12%, almost entirely by forcing
defenders into stick fouls. With real birthdates, the age curve declines monotonically
from 1.24× (age 20) to 0.76× (37) with no plateau (within-player −0.029/yr). The
preregistered mechanism prediction holds: the carrier route decays at **−4.0%/yr** against
**−2.6%/yr** for provocation (cluster-bootstrapped SEs 0.0011/0.0005). NHL Edge tracking
closes the loop: measured maximum skating speed and bursts-over-20mph predict
carrier-route draws (partial r = 0.166-0.225, position- and possession-controlled) and not
provocation draws (r = 0.037-0.051), with carrier draws/60 monotone across measured speed
quartiles (0.242 → 0.377). Three independent evidence streams — infraction composition,
differential aging, and tracking data — converge on one mechanism. Consistent garnishes:
shorter players draw modestly more within every position (height r to −0.21); high-θ
players attract ~8× more embellishment calls, but at 110 calls against ~20k minors the
referee tax is ~0.5%. Two disciplined nulls close alternative stories: within-game draws
are indistinguishable from Poisson given a player's rate (variance ratio 1.045; no hot
hand), and elite drawers' calls are distributed across game states like everyone else's
(no clutch-timing skill; value accrues by volume).

### 7.2 Six leagues

| League | Corpus | Reliability (SB) | Year-over-year | Top of board |
|---|---|---|---|---|
| NHL (core) | 3 seasons · 3,936 games | 0.89 (career) | 0.59 mean × 14 pairs | Hathaway 2.45× |
| NBA | 5 seasons · 5,830 games | 0.94 | 0.86–0.88 × 4 | Antetokounmpo 3.90× |
| WNBA | 6 seasons · 1,306 games | 0.86 | 0.72–0.82 × 5 | Wilson 2.34× |
| NCAA M | 10 seasons | — | 0.54–0.66 × 9 | Dybantsa 2.16× |
| NCAA W | 11 seasons | — | 0.59–0.67 × 8 | Nichols 2.63× |
| Soccer | 8 tournaments · 1,179 matches | 0.71 | — | Hazard pooled top-10 |

One construction, both sexes, three levels of play; every leaderboard passes face validity
at both ends (the bottoms are catch-and-shoot specialists, stay-at-home defensemen, and
Sue Bird).

### 7.3 The market

Draw skill correlates with points per sixty at **r = 0.067** — nearly independent of what
the market prices. Deployment does not reward it: conditional on scoring and position, one
standard deviation of θ is associated with **−0.28 minutes per game (t = −3.65)**,
attenuating to **−0.177 (t = −2.44)** under full zone-start deployment controls from NHL
Edge — about a third of the raw association is usage-flavored; a significant negative
residual is not. This establishes non-reward in deployment, not coach error (θ correlates
with role and unmeasured attributes). *[PENDING EXTERNAL DATA: a cap-hit regression —
log salary on θ conditional on scoring, position, age — will make the pricing test
literal; the analysis script is staged and awaits contract data. Section to be completed
for the full-paper deadline.]*

### 7.4 Officiating by-products

Three findings double as external validation. The home bias is recovered twice
(player-level +6.7%; event-level +0.011). Make-up calling is large and real: after a home
penalty, the next even-strength minor goes to the home team 38.0% of the time vs 58.4%
after an away penalty (a 20.4-point swing, z = 26.8), surviving score controls and
decaying with time since the last call — rebalancing, not team style; the net ΔWP already
absorbs this, and agreement with the window-only goals path bounds the make-up tax near
0.002. And whistle-swallowing folklore is right about the moment, wrong about the month:
playoff even-strength minors are *up* 16% per skater-hour (with skill transferring ~1:1
into the postseason), while draws in close-and-late situations run 12% below exposure —
referees pocket the whistle in the dying minutes of close games, not in the playoffs.

## 8. Limitations

The ΔWP estimates are state-conditional but observational; the taking-side mirror is a
construction identity, not independent evidence. The fifteen-season replication uses
all-situation ice time as exposure where the core study uses even-strength time. College
feeds omit missed free throws and minutes; basketball drawn-by is a structural inference.
Soccer evidence is tournament-only (no club-season persistence yet). NHL penalty events
carry no zone coordinates. The market claim currently rests on orthogonality and
deployment; the literal salary test is pending (§7.3). PWHL and NBA league feeds carry
committer-only attribution — pipeline extensions, not modeling gaps.

## 9. Conclusion

Winning contact is a measurable, career-stable, win-valued talent that appears wherever
event data allows it to be measured — six leagues, both sexes, three levels of play — and
that hockey's accounting renders structurally invisible: unlike basketball, where drawn
fouls surface as free-throw points on the drawer's own line, a drawn penalty never touches
the ledger of the player who earned it. The skill has a confirmed mechanism, a stable
stylistic signature, and a price of approximately zero. For a league where playoff berths
turn on single points, the practical conclusion is uncomfortable in the way useful
findings are: teams already employ, and underplay, players who manufacture two dozen power
plays a season — and no one is bidding.

## Reproducibility

All data sources are public (§3); derived tables sufficient to verify every number are in
`data/` of this repository; complete model specifications with session-dated provenance
are in `METHODS.md`. Analysis scripts are being consolidated into `analysis/`.

## Acknowledgments

Data via the NHL's public APIs (including NHL Edge), the sportsdataverse ecosystem and
nba_api, and StatsBomb Open Data (StatsBomb is the exclusive source of the soccer event
data used here). This research was conducted on the GameVibe platform
(gamevibeanalytics.com) with AI-assisted analysis tooling; all modeling decisions,
validation standards, and claims are the author's.
