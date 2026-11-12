# Design decisions

Technical decisions behind the pipeline, numbered as referenced in the code. Decisions D-001,
D-002, D-003, D-009, D-011 and D-012 are now implemented in the shared data layer
[nbacore](https://github.com/yangzhou-tysportsanalytics/nbacore) (D-016); they are kept here
because the code and reports refer to them.

## D-001 Raw data source

The public 2015-16 SportVU archives (github.com/linouk23/NBA-Player-Movements) and the
play-by-play CSV (github.com/sumitrodatta/nba-alt-awards) are read directly rather than through
the Hugging Face script dataset, which drops `unix_ms` and events without a 1:1 pbp match.

## D-002 Deduplication at entity level

54–62 % of raw moments are cross-event duplicates; a few timestamps are split into a ball-only
and a players-only moment. Entity rows are unioned per `(game_id, period, unix_ms, team_id,
player_id)`, keeping the first occurrence.

## D-003 `unix_ms` is the time axis

About 25 % of events contain upward game-clock corrections; `unix_ms` is monotone. Sorting,
resampling and segmentation use `unix_ms`; the game clock is used only to align pbp events and
as a context feature.

## D-004 Tooling

Python >= 3.11 (3.12 used), uv for environments, ruff for lint and format.

## D-005 Scheme vs player effects are a definition, not an identification

In half a season nearly every player is nested in one team, so player and team effects are only
identified as (team mean) + (within-team deviation). Scheme effect := team mean of the deviation;
player effect := deviation from the own-team mean. Reported as a definition.

## D-006 Ghost conditioning for deviations is online

Conditioning on the whole possession lets a ghost see the offence's future, which is itself a
reaction to the real defence. Deviations use p(D_t | O_≤t, ball_≤t, context), with the defence
rolled out by the model; offline conditioning is used only for visualisation.

## D-007 Densities must be analytic or well sampled, and calibrated

The learned ghost has an analytic per-frame density (Gaussian-mixture head); if sampling is
needed, K >= 100. Calibration is checked (PIT histogram, coverage vs nominal level). Euclidean
deviation is kept as the control metric.

## D-008 Analysis order

Matchup HMM → rule ghost → Euclidean deviation → hierarchical model and reliability first; the
learned (generative) ghost and likelihood-based deviations follow.

## D-009 Possession bookkeeping

Attack direction per (game, team, period) from the ball at pbp shot events (halves swap after
period 2; overtime keeps period 4). A possession window starts when the ball crosses midcourt and
ends at the first terminal pbp event (FG attempt, turnover, foul, violation, timeout, period end)
+ 0.5 s; free-throw sequences are excluded; dead balls truncate windows; lineup changes inside a
window discard it. Windows are cropped to 24 s; transitions (shot within 4 s of the crossing) are
flagged and excluded from half-court analyses. A defensive possession of a player = a half-court
window in which he is on court on defence.

## D-010 Compute

Everything except full training of the learned ghost runs on a laptop CPU; model code is
device-agnostic and tested on CPU.

## D-011 Shot terminal time comes from the ball, not the pbp clock

pbp clock readings for field-goal attempts are late (median 2.0 s after the release). Shot
windows end at the ball-derived release. Refined by D-017.

## D-012 Ball-handler rule

Nearest offensive player within 3 ft of the ball (xy) while the ball is below 10 ft; a run of
identical candidates is accepted when it lasts >= 0.4 s or is followed by ball flight; shorter
runs inherit the previous handler.

## D-013 Two ghost masks

Team ghost: all five defenders masked and rolled out jointly (scheme-level questions).
Individual ghost: one defender masked, conditioned on the offence and the four real teammates up
to t (individual attribution). Both are trained with one masked-completion model.

## D-014 Testable form of the "under-helping" question

A league-average ghost is centred on the league by construction, so "do defenders help less than
their ghost" is ill-posed for it. Instead: (a) the distribution of help timing by team and player;
(b) whether, with the offence held fixed, earlier help lowers shot quality / expected points;
(c) a value-optimal ghost is future work.

## D-015 Shooter-aware xFG and breakdown events

xFG includes an empirical-Bayes shrunken shooter-by-zone make rate. Breakdown events: deviation
above the league 80th percentile sustained >= 1 s, the first in a cascade attributed to its owner.

## D-016 Shared data layer

Cleaning, events, the possession ledger and the fixed split are provided by nbacore (pinned
release `data-v1.6`). This repository keeps the half-court tensors, matchup HMM, switch / help /
closeout definitions, ghost models, deviations, hierarchical model and analyses.

## D-017 Corrected shot release

nbacore's corrected release (`shot_release.t_release_ms`) replaces D-011's in-hand rule, which
counted a ball in flight over a teammate as in hand (about 13 % of releases were 0.2–1.1 s late).
It is used for window ends, shot features and the handler check. `window_uid`
(`<game_id>:<terminal pbp event>`) is the key across versions.

## D-018 Event definitions

- Switch: the reported switch is definition A (screen switch): within 1 s of screen contact the
  screened defender and the screener's defender exchange attackers, each held >= 1 s. Definition
  B (any change of guarded attacker held >= 1 s) is kept as the reassignment total, classed as
  screen_switch / one_sided / rotation.
- Help: two model definitions are kept. A: the HMM help state. B (geometric, reported): a third
  defender at least 6 ft from his man, closing on the ball handler at >= 3 ft/s, handler within
  20 ft of the rim, for >= 0.4 s. They capture different behaviour (1–3 % overlap).
- Second-chance possessions are excluded in v1 (windows start at the midcourt crossing).
  Sensitivity: windows that start at the offensive-rebound control when the ball never leaves the
  frontcourt add 3.8 % half-court possessions; reliability, team shares, player effects and the
  team help / points association are unchanged
  (`reports/analysis/second_chance_sensitivity.json`).

## D-019 Data release policy

The raw SportVU files are not redistributed (licence unclear); this repository links the public
source and contains the code. Released data contain no player or ball tracks. The only point
coordinates are shot locations. Third-party metrics used for validation are not redistributed
except FiveThirtyEight RAPTOR (CC BY 4.0, attributed).

## D-020 Phases, help geometry and breakdown timing

- Phases per defender and step, in priority order: help (inside one of the defender's help
  events, definition B) > closeout (inside one of his closeouts) > recovery (2 s after the end of
  his help or closeout) > screen (0.5 s before to 1.0 s after an on-ball screen contact of the
  possession) > pre-screen (before the first on-ball screen's window) > other.
- Sightline cone: the defender is within 20° of the ball handler → rim direction and no farther
  from the handler than the handler is from the rim.
- Paint time: continuous stretches in the lane (19 ft from the baseline, 16 ft wide); steps in
  stretches of at least 2.5 s are flagged as approaching the defensive three-second limit.
- Wider alternative for sensitivity: screen window −1.0 s to +2.0 s, recovery 3 s, cone 30°.
- Breakdown events (D-015) are defined against the learned ghost only: against the rule ghost the
  80th-percentile, 1-s rule flags 97 % of possessions.

## D-021 Context term of the deviation model

The context adjustment of the deviations (and γ[context] of the hierarchical model) uses the base
covariates — mean man-to-rim and man-to-ball distance and their squares, strong-side share,
on-ball share, guarding more than one man, log steps — plus the D-020 phase shares (pre-screen,
screen, help, closeout, recovery; "other" is the reference). This is the main specification; the
base-only adjustment is reported alongside. Sightline-cone share and paint time are outcomes of
positioning and are reported as descriptors, not used as adjustments.

## D-022 Learned-ghost output parametrised on the rule ghost

Each mixture component mean of the learned ghost is an attention-weighted combination of the five
attackers' rule positions (man, ball and rim weights from the matchup model) plus a learned
offset. The weights come from attention between the defender and attacker tokens, plus two
learned priors: one ghost per attacker when all five defenders are hidden (team ghost), and a
preference for attackers far from every visible defender when four are visible (individual
ghost). The offset starts at zero, so the untrained model is the rule ghost and training learns
corrections to it. The unanchored head is kept as an ablation.

## D-023 Breakdown threshold at the possession level

A threshold on single defender-steps flags almost every possession (five defenders, about 100
autocorrelated steps each): 97 % with the rule ghost at the 80th percentile. The threshold is
therefore set per possession: the largest deviation level that some defender holds for at least
1 s, and its league 80th percentile across possessions, so the worst fifth of possessions contain
a breakdown. Breakdowns are runs of at least 1 s above that threshold; the first breakdown of a
cascade is attributed to its defender. The metric is the individual-ghost negative log density
(cross-fitted); the rule-ghost distance version is kept for reference.

## D-024 Published tags and release assets are immutable

Public tags (data-v1.0, data-v1.1, xfg-v1.0) are never moved and their release assets are never
replaced; any change gets a new tag. A later archive (e.g. Zenodo) re-publishes the same tag with
the same asset bytes.

## D-025 Matchup validation sample of 100 possessions (superseded by D-036)

The validation sample is 100 half-court possessions from the train and validation games, in the
same strata as the original 200-possession design (screen with switch 25, screen without switch
25, drive 18, ball movement 17, other 15), drawn at random within strata; 15 are annotated by a
second annotator. Accuracies are reported weighted by each stratum's share of all half-court
possessions.

## D-026 Learned ghost: anchored head and model choice

The learned ghost is the configuration with the lowest mean validation NLL of a pre-specified
sweep (d_model 64 or 128, learning rate 3e-4 or 1e-3): d_model 128, 4 blocks, learning rate 1e-3.
The rule-anchored head (D-022) is kept for both the team and the individual ghost. At full scale
the unanchored head has a better team NLL on validation (5.757 vs 6.079) and a slightly worse
individual NLL (4.332 vs 4.281); individual attribution uses the individual ghost, and the
anchor weights read as "whom the ghost guards". The unanchored head is reported as an ablation.

## D-027 Blind annotation subset (superseded by D-036)

Ten of the 100 validation possessions (two per stratum) are annotated first without any model
output, and again with the model pre-labels at least two days later by the same annotator. The
paired difference in matchup accuracy measures the anchoring of the pre-labels.

## D-028 Conditioned ghosts are not used as the reference

The lineup- and scheme-conditioned ghosts improve validation NLL (lineup by 0.020 / 0.035, scheme
by 0.030 / 0.060 for the team / individual ghost). A draft plan said such a ghost would then be
used for deviations. It is not: the lineup ghost includes the evaluated defender's own identity,
so his habits move into the reference and his effect is pulled toward zero by construction; the
scheme ghost measures deviations from the own team's habits and so absorbs part of the team
effect. The league ghost stays the reference; this is a judgement made after the results. On the
validation and test games, which both conditioned ghosts never saw, their deviations are
compared with the league ghost's.

## D-029 Still-frame figure of the learned ghost

The paper shows two possessions of held-out games (one on-ball screen with a switch, one
without) at screen contact, one second later and the shot release: real players, the team
ghost's most likely position for each defender, and the screened defender's individual ghost as
50 / 80 / 95 % regions (`paper/figures/learned_ghost_frames.png`, generated by
`scripts/make_figure_learned_ghost_frames.py`). The two possessions are drawn at random (seed 9)
among typical ones (individual-ghost NLL between the league quartiles), not chosen by hand; the
selection is recorded in `paper/figures/learned_ghost_frames.json`. It is a rendering of
positions at six moments, not a data table; no tracking data is distributed (D-019).

## D-030 Constant switching in the matchup model

A variant of the matchup HMM whose switching probability depends on screens, passes and drives
(`scripts/phase2_event_hmm.py`) raises the switching probability of the two defenders of a
screen about 1.7-fold, but improves held-out fit only slightly (0.001 nats per observation) and
leaves the decoded matchups almost unchanged (2.96 vs 2.94 reassignments per possession;
`reports/phase2/v1.6_all/event_hmm.json`). The constant-switching model is kept for all results;
the event-dependent model is a sensitivity analysis.

## Note: team-level test families

The six team-level tests reported as the first family (help rate within 3 s, switch rate and
median help delay, each against points per possession and per half-court possession) were
computed first and grouped as a family afterwards, once their results were known; they were not
pre-registered. Two later tests (early-help rate, rule-ghost expected points) form a second,
exploratory family. Re-estimates of the same associations are not counted as tests.

The event-dependent matchup model (D-030) is reported in the paper's supplement.

## D-031 Velocity features of the learned ghost

The velocity inputs are central differences and therefore include positions 0.2 s ahead. A
prototype trained with backward (strictly causal) differences gives the same validation NLL
(`reports/phase3/v1.6_all_v1nlla_g32_vel_{central,backward}_eval.json`), so the reported model
is kept; the causal option is available (`build_arrays(..., velocity="backward")`).

## D-032 Choices for the SSAC full paper

- Effects are stated in points per 100 possessions; the wins conversion (2.60 wins per point of
  net rating per 100 possessions, 2015-16, `analysis_wins_per_point.py`) is secondary. The main
  team-game number is the joint-model early − late contrast (5 percentage points of help moved
  from 1–3 s to within 1 s of the screen), stated as an association.
- Disclosed: the 1 s early-help threshold was chosen after the first results
  (`reports/analysis/help_threshold.json` gives the whole threshold curve), and dropping
  fourth-quarter garbage time weakens the team-game association (intervals include zero).
- Strong-side corner help is measured per drive opportunity
  (`analysis_corner_help_opportunity.py`); the reading of the result (no shooter dependence,
  uncertain scoring consequence) was fixed before the analysis was run.
- Example teams and players on the cards follow fixed rules: the team with the fewest points
  allowed and the team with the highest early-help rate; players with at least 1,000 defensive
  possessions whose intervals exclude zero.

## D-033 Stunts and clip lists

- The paper does not introduce a geometric definition of a stunt. Whether detected help is full
  help or a stunt is answered by the video check (D-036).
- The review-queue table in the paper names defenders by team and listed position; the release
  CSV keeps player names and IDs.

## D-034 Departure is the paper's player measure

Departure (the individual-ghost negative log-density) partly reflects how spread out the league
map is (correlation 0.47 per defender-step). Spread-free versions (departure minus the map's
entropy; PIT) are slightly more repeatable and give nearly the same ranking (rank correlation
0.95 / 0.93; `reports/analysis/departure_spread.json`). The negative log-density is kept as the
measure; the spread check is reported.

## D-035 Second annotation round (on hold)

A second, external annotation round was prepared and is on hold. The paper's numbers come from
the check of D-036.

## D-036 Video check on held-out games

The 100-possession annotation (D-025, D-027) was not carried out. It is replaced by a shorter
check, all on test games, so it is a held-out check of the matchup model and of the event rules
(`select_quick_check.py`, pages `make_quick_check.py` with `tools/quick_check/`, evaluation
`quick_check_eval.py`):

- Matchups: 40 half-court possessions stratified by play type (pick-and-roll with switch 10,
  without switch 10, drive 8, ball movement 6, other 6). At two random moments per possession
  (one in each half, from 0.6 s) the annotator names whom each defender guards (or nobody),
  without seeing the model's answer. Accuracy of the matchup model and three baselines, raw and
  weighted by the shares of the strata among test possessions; intervals by bootstrap of
  possessions within strata.
- Events: 90 short clips with the asked defender(s) ringed: 25 early help, 15 late help, 10
  strong-side corner help, 25 screen switches, and 15 clips without a detected event (8 help, 7
  switch) mixed in blind. Help answers: full help, stunt, already in a help spot and only sank
  toward the ball, no, can't tell. The main precision counts full help and stunts as confirmed;
  a lenient version also counts "already in a help spot". Wilson intervals. Recall (missed
  events) is not measured.
- Two analysts (not coaches) labelled independently; the clips on which they disagreed were
  reviewed together to a consensus (`make_quick_adjudication.py`). The release carries both
  analysts' answers and the consensus, without notes.
- Help events judged help on review were farther from the defender's man and closer to the ball
  handler. Section 4 is refitted keeping only such events (within 8 ft of the handler; at least
  12 ft from his man; both) as a sensitivity analysis (`analysis_help_strict_sensitivity.py`);
  the thresholds were read off the reviewed clips, so the in-sample precision of each variant is
  descriptive only.
- Possessions that start during a stopped game clock (e.g. a baseline inbound) are not cut by
  the segmentation; their share is measured (`analysis_stopped_clock_windows.py`).
