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
