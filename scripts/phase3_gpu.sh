#!/usr/bin/env bash
# Learned ghost on a GPU machine (decisions D-006, D-007, D-013, D-022). Needs only this code and
# the packed arrays of scripts/phase3_export_arrays.py (no tracking data, no nbacore data root).
#
# Usage: bash scripts/phase3_gpu.sh <stage> [arrays dir]
#   check     GPU visible to torch; array checksums match manifest.json
#   smoke     2 train / 2 val games, 1 epoch on the GPU (~2 min)
#   sweep     4 configs on train / val: (d64 x 2 blocks, d128 x 4 blocks) x lr (3e-4, 1e-3)
#   final     CONFIG=<sweep tag> : load the chosen sweep model (chosen on val) and score it once
#             on the test games; nothing is retrained or tuned on test
#   ablation  CONFIG=<sweep tag> : same config with the unanchored head and with the L2 loss
#   conditions CONFIG=<sweep tag>: lineup- and scheme-conditioned ghosts (Phase 3 task 4; shrunk
#             identity embeddings), compared with the league ghost on val
#   crossfit  CONFIG=<sweep tag> : 5 models, each trained without one nbacore fold (all splits)
#   deviations CONFIG=<sweep tag>: score every possession with its out-of-fold model
#             (phase4_learned_ghost_dev.py; outputs contain no coordinates)
# Outputs: runs/phase3/<version>_all_<tag>/ (model, history, eval) and reports/phase3/*_eval.json;
# logs in reports/logs/gpu_<stage>.log. Copy runs/phase3 and reports/phase3 back afterwards.
set -u
export PYTHONUTF8=1
STAGE=${1:?stage: check | smoke | sweep | final | ablation | conditions | crossfit | deviations}
ARR=${2:-data/ghost_arrays}
LOG=reports/logs/gpu_${STAGE}.log
mkdir -p reports/logs
COMMON=(--game-set all --arrays-dir "$ARR" --device cuda --schedule mixed --loss nll
        --anchor rule --batch-size 64)
EPOCHS=${EPOCHS:-30}

run() {  # tag, extra args...
  local tag=$1; shift
  echo "=== $(date '+%F %T') $tag $*" | tee -a "$LOG"
  uv run python scripts/phase3_train.py "${COMMON[@]}" --epochs "$EPOCHS" --tag "$tag" "$@" \
    >> "$LOG" 2>&1 || { echo "FAILED $tag" | tee -a "$LOG"; exit 1; }
}

config_args() {  # sweep tag -> model / lr arguments
  case "$1" in
    gpu_d64_lr3e-4)  echo "--d-model 64 --blocks 2 --n-heads 4 --lr 3e-4" ;;
    gpu_d64_lr1e-3)  echo "--d-model 64 --blocks 2 --n-heads 4 --lr 1e-3" ;;
    gpu_d128_lr3e-4) echo "--d-model 128 --blocks 4 --n-heads 8 --lr 3e-4" ;;
    gpu_d128_lr1e-3) echo "--d-model 128 --blocks 4 --n-heads 8 --lr 1e-3" ;;
    *) echo "unknown CONFIG $1" >&2; exit 1 ;;
  esac
}

case "$STAGE" in
  check)
    nvidia-smi | tee "$LOG"
    uv run python - "$ARR" <<'EOF' | tee -a "$LOG"
import hashlib, json, sys
from pathlib import Path
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch does not see the GPU"
print("device", torch.cuda.get_device_name(0))
root = Path(sys.argv[1])
man = json.loads((root / "manifest.json").read_text())
for split, info in man["splits"].items():
    for f, want in info["files"].items():
        h = hashlib.sha256()
        with open(root / split / f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        assert h.hexdigest() == want, f"checksum mismatch: {split}/{f}"
    print(split, info["n_possessions"], "possessions: checksums ok")
EOF
    ;;
  smoke)
    EPOCHS=1 run gpu_smoke --train-games 2 --val-games 2 ;;
  sweep)
    for tag in gpu_d64_lr3e-4 gpu_d64_lr1e-3 gpu_d128_lr3e-4 gpu_d128_lr1e-3; do
      # shellcheck disable=SC2046
      run "$tag" $(config_args "$tag")
    done ;;
  final)
    : "${CONFIG:?set CONFIG to the chosen sweep tag}"
    # shellcheck disable=SC2046
    V=$(uv run python -c "from ghost import data as D; print(D.DataConfig.load().version)")
    run "${CONFIG}_final" $(config_args "$CONFIG") --eval-test       --load-model "runs/phase3/${V}_all_${CONFIG}/model.pt" ;;
  ablation)
    : "${CONFIG:?set CONFIG to the chosen sweep tag}"
    # shellcheck disable=SC2046
    run "${CONFIG}_noanchor" $(config_args "$CONFIG") --anchor none
    # shellcheck disable=SC2046
    run "${CONFIG}_l2" $(config_args "$CONFIG") --loss l2 ;;
  conditions)
    : "${CONFIG:?set CONFIG to the chosen sweep tag}"
    for c in lineup scheme; do
      # shellcheck disable=SC2046
      run "${CONFIG}_${c}" $(config_args "$CONFIG") --condition "$c"
    done ;;
  crossfit)
    : "${CONFIG:?set CONFIG to the chosen sweep tag}"
    for k in 0 1 2 3 4; do
      # shellcheck disable=SC2046
      run "${CONFIG}_fold$k" $(config_args "$CONFIG") --fold-out "$k"
    done ;;
  deviations)
    : "${CONFIG:?set CONFIG to the chosen sweep tag}"
    uv run python scripts/phase4_learned_ghost_dev.py --crossfit "$CONFIG" --arrays-dir "$ARR"       --device cuda >> "$LOG" 2>&1 || { echo "FAILED deviations" | tee -a "$LOG"; exit 1; } ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "=== $(date '+%F %T') done $STAGE" | tee -a "$LOG"
