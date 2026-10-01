#!/usr/bin/env bash
# Second-chance sensitivity (decision D-018): rerun the analysis chain on possession windows that
# also start at an offensive-rebound control when the ball never leaves the frontcourt, in a
# separate directory (runs/sc_sensitivity, git-ignored), then compare with the main results.
# Needs the main results first (bash scripts/pipeline_all.sh). About 1 h on a laptop CPU.
# Usage: bash scripts/second_chance_sensitivity.sh
set -u
export PYTHONUTF8=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN="$ROOT/runs/sc_sensitivity"
mkdir -p "$RUN/reports/logs" && cp -r "$ROOT/configs" "$RUN/"
sed -i 's|^windows_release: .*|windows_release: l2_sc|; s|^processed_dir: .*|processed_dir: data/processed/nbacore-v1.6-l2sc|' \
  "$RUN/configs/data.yaml"
mkdir -p "$RUN/data/processed/nbacore-v1.6-l2sc"
LOG="$RUN/reports/logs/sc_pipeline.log"
: > "$LOG"
steps=(
  "build_windows_second_chance.py --workers 3"
  "phase1_possessions.py --game-set all --workers 3"
  "phase2_matchups.py --game-set all"
  "phase2_events.py --game-set all"
  "phase2_screen_events.py --game-set all"
  "phase2_screen_rates.py --game-set all"
  "phase4_rule_ghost.py --game-set all"
  "analysis_team_screen_defense.py --game-set all"
  "analysis_team_validity.py"
  "analysis_help_timing.py"
  "phase4_phase_context.py --defs A"
  "phase4_reliability.py --game-set all"
  "phase4_reliability.py --game-set all --context phases"
  "phase4_hier.py"
  "phase4_hier.py --context phases"
  "phase4_xfg_data.py"
  "phase4_xfg.py"
  "phase5_predictive_validity.py"
)
cd "$RUN" || exit 1
for s in "${steps[@]}"; do
  echo "=== $(date '+%H:%M:%S') $s" >> "$LOG"
  # shellcheck disable=SC2086
  if ! uv run python "$ROOT"/scripts/$s >> "$LOG" 2>&1; then
    echo "FAILED $s" >> "$LOG"
    exit 1
  fi
done
cd "$ROOT" && uv run python scripts/compare_second_chance.py >> "$LOG" 2>&1
echo "=== $(date '+%H:%M:%S') ALL DONE" >> "$LOG"
