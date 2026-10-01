#!/usr/bin/env bash
# Full rebuild on the pinned nbacore release and window rule (configs/data.yaml).
# Same steps as `make pipeline-all`, for machines without make. Stops at the first failure.
# Order: help timing reads the points-allowed table of analysis_team_validity; phase context reads
# the on-ball screens of analysis_team_screen_defense; the *--context phases* steps read it.
# Usage: bash scripts/pipeline_all.sh [log file]
set -u
export PYTHONUTF8=1
LOG="${1:-reports/logs/pipeline_all.log}"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
steps=(
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
  "phase5_predictive_validity.py"
  "phase5_predictive_validity.py --context phases"
  "phase2_em_init_sensitivity.py --game-set all"
  "make_figures.py"
  "check_release.py"
)
for s in "${steps[@]}"; do
  echo "=== $(date '+%H:%M:%S') $s" >> "$LOG"
  # shellcheck disable=SC2086
  if ! uv run python scripts/$s >> "$LOG" 2>&1; then
    echo "FAILED $s" >> "$LOG"
    exit 1
  fi
done
echo "=== $(date '+%H:%M:%S') ALL DONE" >> "$LOG"
