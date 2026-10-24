#!/usr/bin/env bash
# Full rebuild on the pinned nbacore release and window rule (configs/data.yaml).
# Same steps as the make targets, for machines without make. Stops at the first failure.
#   PART=base        (default) `make pipeline-all`: possessions -> matchups -> rule ghost -> season
#                    analyses -> figures. Order: help timing reads the points-allowed table of
#                    analysis_team_validity; phase context reads the on-ball screens of
#                    analysis_team_screen_defense; the *--context phases* steps read it.
#   PART=robustness  `make robustness`: robustness and checks on the rule ghost (after base; ~2 h).
#   PART=learned     `make learned`: learned-ghost results on the laptop after the GPU steps
#                    (scripts/gpu.py; results unpacked into runs/ and the processed dir).
#                    SKIP_HEAVY=1 skips the long CPU steps (expected points ~9 h, conditioned
#                    ghosts on held-out games ~3 h, multimodality) and reuses their outputs.
#   PART=annotation  `make annotation-eval`: matchup / event accuracy, baselines, blind subset.
#   PART=everything  base, robustness, learned (the annotation evaluation needs the annotations).
# Usage: [PART=...] [SKIP_HEAVY=1] bash scripts/pipeline_all.sh [log file]
set -u
export PYTHONUTF8=1
PART=${PART:-base}
LOG="${1:-reports/logs/pipeline_${PART}.log}"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
G=learned:cf_gpu_d128_lr1e-3  # chosen learned ghost, cross-fitted deviations (D-026)
base=(
  "phase1_possessions.py --game-set all"
  "phase2_matchups.py --game-set all"
  "phase2_events.py --game-set all"
  "phase2_screen_events.py --game-set all"
  "phase2_screen_rates.py --game-set all"
  "phase4_rule_ghost.py --game-set all"
  "phase4_xfg_data.py"
  "phase4_xfg.py"
  "phase4_ghost_points.py"
  "analysis_team_screen_defense.py --game-set all"
  "analysis_team_validity.py"
  "analysis_help_timing.py"
  "analysis_help_value.py"
  "phase4_phase_context.py --defs A"
  "phase4_phase_context.py --defs B"
  "phase4_reliability.py --game-set all"
  "phase4_reliability.py --game-set all --context phases"
  "phase4_hier.py"
  "phase4_hier.py --context phases"
  "phase4_breakdowns.py"
  "phase4_breakdowns.py --quantile 0.95"
  "phase4_breakdowns.py --threshold step"
  "phase4_breakdowns.py --threshold step --quantile 0.95"
  "phase5_predictive_validity.py"
  "phase5_predictive_validity.py --context phases"
  "phase2_em_init_sensitivity.py --game-set all"
  "make_figures.py"
  "check_release.py"
)
robustness=(
  "analysis_player_validity.py"
  "phase4_opponent_skill.py"
  "analysis_team_game_panel.py"
  "phase4_context_time_robustness.py"
  "analysis_help_timing_matched.py"
  "analysis_help_timing_sensitivity.py"
  "analysis_meta_metrics.py"
  "analysis_xfg_subgroups.py"
  "analysis_xfg_defender_features.py"
  "phase5_predictive_validity.py --context phases --xfg-variant m3d"
  "phase5_predictive_validity.py --context phases --xfg-variant m3h"
  "phase5_predictive_validity.py --context phases --xfg-variant m3c"
  "analysis_help_left_man.py"
  "phase2_event_hmm.py --game-set all"
)
learned=(
  "phase3_rule_baselines.py"
  "phase3_compute_summary.py"
  "phase4_reliability.py --ghost $G --context phases"
  "phase4_hier.py --ghost $G --context phases"
  "phase4_breakdowns.py --ghost $G"
  "phase4_breakdowns.py --ghost $G --threshold step"
  "phase4_breakdowns.py --ghost $G --threshold step --quantile 0.95"
  "analysis_meta_metrics.py --ghost $G"
  "analysis_learned_vs_rule.py"
  "phase5_predictive_validity.py --context phases --ghost $G"
  "analysis_release_density_baseline.py"
)
learned_heavy=(
  "phase3_multimodality.py --model-run gpu_d128_lr1e-3_final --split test"
  "phase4_learned_ghost_points.py --crossfit gpu_d128_lr1e-3 --device cpu"
  "phase4_learned_ghost_dev.py --model-run gpu_d128_lr1e-3_scheme --splits val,test --device cpu --resume"
  "phase4_learned_ghost_dev.py --model-run gpu_d128_lr1e-3_lineup --splits val,test --device cpu --resume"
)
learned_after=(
  "analysis_learned_points_reliability.py"
  "analysis_conditioned_ghosts.py"
  "make_figures.py"
)
annotation=(
  "phase2_annotation_eval.py --bundle gd_v2_core100 --blind gd_v2_blind10"
)
steps=()
case "$PART" in
  base) steps=("${base[@]}") ;;
  robustness) steps=("${robustness[@]}") ;;
  learned)
    steps=("${learned[@]}")
    [ "${SKIP_HEAVY:-0}" = 1 ] || steps+=("${learned_heavy[@]}")
    steps+=("${learned_after[@]}") ;;
  annotation) steps=("${annotation[@]}") ;;
  everything)
    steps=("${base[@]}" "${robustness[@]}" "${learned[@]}")
    [ "${SKIP_HEAVY:-0}" = 1 ] || steps+=("${learned_heavy[@]}")
    steps+=("${learned_after[@]}") ;;
  *) echo "unknown PART $PART (base | robustness | learned | annotation | everything)"; exit 1 ;;
esac
for s in "${steps[@]}"; do
  echo "=== $(date '+%H:%M:%S') $s" >> "$LOG"
  # shellcheck disable=SC2086
  if ! uv run python scripts/$s >> "$LOG" 2>&1; then
    echo "FAILED $s" >> "$LOG"
    exit 1
  fi
done
echo "=== $(date '+%H:%M:%S') ALL DONE ($PART)" >> "$LOG"
