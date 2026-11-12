#!/usr/bin/env bash
# Departure vs map spread (SSAC final review R4 P1): score the spread of the cross-fitted
# individual ghost for every defender-step, build the spread-free versions and run the player
# pipeline on them. Resumable: the scoring keeps games already done.
#   bash scripts/run_departure_spread.sh
set -euo pipefail
export PYTHONUTF8=1
TAG=gpu_d128_lr1e-3
uv run python scripts/phase4_learned_ghost_spread.py --crossfit "$TAG" --resume
uv run python scripts/phase4_learned_ghost_spread.py --crossfit "$TAG" --variants
for V in excess pit; do
  G="learned:cf_${TAG}_${V}"
  echo "=== $(date +%T) $G"
  uv run python scripts/phase4_reliability.py --ghost "$G" --context phases
  uv run python scripts/analysis_meta_metrics.py --ghost "$G"
  uv run python scripts/phase4_hier.py --ghost "$G" --context phases --cell-variance game_clustered
  uv run python scripts/phase5_predictive_validity.py --context phases --ghost "$G"
done
uv run python scripts/analysis_departure_spread.py
echo "=== $(date +%T) ALL DONE (departure spread)"
