#!/usr/bin/env bash
# Learned ghost on a GPU machine (decisions D-006, D-007, D-013, D-022). Needs only this code and
# the packed arrays of scripts/phase3_export_arrays.py (no tracking data, no nbacore data root).
#
# Usage: bash scripts/phase3_gpu.sh <stage> [arrays dir]
#   check       GPU visible to torch; array checksums match manifest.json
#   smoke       2 train / 2 val games, 1 epoch on the GPU (~2 min)
#   sweep       4 configs on train / val: (d64 x 2 blocks, d128 x 4 blocks) x lr (3e-4, 1e-3)
#   summary     validation NLL of the sweep runs and the suggested CONFIG
#   final      CONFIG=<sweep tag> : load the chosen sweep model (chosen on val) and score it once
#               on the test games; nothing is retrained or tuned on test
#   calibration CONFIG=<sweep tag> : calibration by game situation of the final model on test
#   ablation    CONFIG=<sweep tag> : same config with the unanchored head and with the L2 loss
#   conditions  CONFIG=<sweep tag> : lineup- and scheme-conditioned ghosts (Phase 3 task 4; shrunk
#               identity embeddings), compared with the league ghost on val
#   crossfit    CONFIG=<sweep tag> : 5 models, each trained without one nbacore fold (all splits)
#   deviations  CONFIG=<sweep tag> : score every possession with its out-of-fold model
#               (phase4_learned_ghost_dev.py; outputs contain no coordinates)
#   pack        ghost_gpu_results.zip with everything to bring back (no coordinates)
# Outputs: runs/phase3/<version>_all_<tag>/ (model, history, eval) and reports/phase3/*_eval.json;
# logs in reports/logs/gpu_<stage>.log. Copy runs/, reports/ and the learned_ghost outputs back.
#
# Environment (defaults for the real run): DEVICE=cuda, EPOCHS=30, FOLDS="0 1 2 3 4".
# After an interruption, rerun the same stage with RESUME=1 (finished configs are not retrained).
# Dry run of the whole chain on a CPU (seconds per stage, numbers meaningless):
#   DEVICE=cpu EPOCHS=1 EXTRA_ARGS="--train-games 2 --val-games 2 --max-batches 3" \
#   CF_ARGS="--max-batches 3" FOLDS="0" DEV_ARGS="--games 1" CAL_ARGS="--games 1" ...
set -u
export PYTHONUTF8=1
STAGE=${1:?stage: check | smoke | sweep | summary | final | calibration | ablation | conditions | crossfit | deviations | pack}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-30}
FOLDS=${FOLDS:-0 1 2 3 4}
EXTRA_ARGS=${EXTRA_ARGS:-}
CF_ARGS=${CF_ARGS:-}
DEV_ARGS=${DEV_ARGS:-}
CAL_ARGS=${CAL_ARGS:-}
RESUME_ARG=""  # RESUME=1: continue from checkpoints after an interruption
[ "${RESUME:-0}" = 1 ] && RESUME_ARG="--resume"
V=$(uv run python -c "from ghost import data as D; print(D.DataConfig.load().version)")
ARR=${2:-$(uv run python -c "from ghost import data as D; print((D.DataConfig.load().processed_dir / 'all' / 'ghost_arrays').as_posix())")}
LOG=reports/logs/gpu_${STAGE}.log
mkdir -p reports/logs
COMMON=(--game-set all --arrays-dir "$ARR" --device "$DEVICE" --schedule mixed --loss nll
        --anchor rule --batch-size 64)

run() {  # tag, extra args...
  local tag=$1; shift
  echo "=== $(date '+%F %T') $tag $*" | tee -a "$LOG"
  # shellcheck disable=SC2086
  uv run python scripts/phase3_train.py "${COMMON[@]}" --epochs "$EPOCHS" --tag "$tag" "$@" $RESUME_ARG \
    >> "$LOG" 2>&1 || { echo "FAILED $tag (see $LOG)" | tee -a "$LOG"; exit 1; }
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
    if [ "$DEVICE" = cuda ]; then nvidia-smi | tee "$LOG"; else : > "$LOG"; fi
    uv run python - "$ARR" "$DEVICE" <<'EOF' | tee -a "$LOG"
import hashlib, json, sys
from pathlib import Path
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if sys.argv[2] == "cuda":
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
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED check"; exit 1; } ;;
  smoke)
    EPOCHS=1 run gpu_smoke --train-games 2 --val-games 2 ;;
  sweep)
    for tag in gpu_d64_lr3e-4 gpu_d64_lr1e-3 gpu_d128_lr3e-4 gpu_d128_lr1e-3; do
      # shellcheck disable=SC2046,SC2086
      run "$tag" $(config_args "$tag") $EXTRA_ARGS
    done ;;
  final)
    : "${CONFIG:?set CONFIG to the chosen sweep tag}"
    # shellcheck disable=SC2046,SC2086
    run "${CONFIG}_final" $(config_args "$CONFIG") --eval-test \
      --load-model "runs/phase3/${V}_all_${CONFIG}/model.pt" $EXTRA_ARGS ;;
  calibration)
    : "${CONFIG:?set CONFIG to the chosen sweep tag}"
    # shellcheck disable=SC2086
    uv run python scripts/phase3_calibration_by_context.py --model-run "${CONFIG}_final" \
      --split test --device "$DEVICE" $CAL_ARGS >> "$LOG" 2>&1 \
      || { echo "FAILED calibration (see $LOG)" | tee -a "$LOG"; exit 1; } ;;
  ablation)
    : "${CONFIG:?set CONFIG to the chosen sweep tag}"
    # shellcheck disable=SC2046,SC2086
    run "${CONFIG}_noanchor" $(config_args "$CONFIG") --anchor none $EXTRA_ARGS
    # shellcheck disable=SC2046,SC2086
    run "${CONFIG}_l2" $(config_args "$CONFIG") --loss l2 $EXTRA_ARGS ;;
  conditions)
    : "${CONFIG:?set CONFIG to the chosen sweep tag}"
    for c in lineup scheme; do
      # shellcheck disable=SC2046,SC2086
      run "${CONFIG}_${c}" $(config_args "$CONFIG") --condition "$c" $EXTRA_ARGS
    done ;;
  crossfit)
    : "${CONFIG:?set CONFIG to the chosen sweep tag}"
    for k in $FOLDS; do
      # shellcheck disable=SC2046,SC2086
      run "${CONFIG}_fold$k" $(config_args "$CONFIG") --fold-out "$k" $CF_ARGS
    done ;;
  deviations)
    : "${CONFIG:?set CONFIG to the chosen sweep tag}"
    # shellcheck disable=SC2086
    uv run python scripts/phase4_learned_ghost_dev.py --crossfit "$CONFIG" --arrays-dir "$ARR" \
      --device "$DEVICE" $DEV_ARGS >> "$LOG" 2>&1 \
      || { echo "FAILED deviations (see $LOG)" | tee -a "$LOG"; exit 1; } ;;
  summary)
    # validation NLL of the sweep runs; suggests CONFIG = lowest mean of team and individual NLL
    uv run python - "$V" <<'EOF' | tee "$LOG"
import json, sys
from pathlib import Path
v, rows = sys.argv[1], []
for tag in ("gpu_d64_lr3e-4", "gpu_d64_lr1e-3", "gpu_d128_lr3e-4", "gpu_d128_lr1e-3"):
    p = Path("reports/phase3") / f"{v}_all_{tag}_eval.json"
    if not p.exists():
        print(f"{tag}: not run"); continue
    e = json.loads(p.read_text())
    t, i = e["val_team"]["slot_mixture_nll_ft"], e["val_individual"]["nll_ft"]
    rows.append(((t + i) / 2, tag))
    print(f"{tag}: val team NLL {t:.4f}  individual NLL {i:.4f}  mean {(t + i) / 2:.4f}")
if rows:
    print("suggested CONFIG =", min(rows)[1])
EOF
    ;;
  pack)
    # results to bring back (no coordinates): runs, reports, learned-ghost deviations
    uv run python - "$V" <<'EOF'
import sys, zipfile
from pathlib import Path
from ghost import data as D
all_dir = D.DataConfig.load().processed_dir / "all"
items = [Path("runs/phase3"), Path("reports/phase3"), Path("reports/logs"), all_dir / "learned_ghost"]
items += sorted((all_dir / "analysis").glob("learned_ghost_dev_*.parquet"))
z = Path("ghost_gpu_results.zip")
with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as f:
    for it in items:
        files = [it] if it.is_file() else (sorted(it.rglob("*")) if it.exists() else [])
        for p in files:
            if p.is_file() and p.name != "checkpoint.pt":
                f.write(p, arcname=p.as_posix())
print(f"{z.resolve()}: {z.stat().st_size:,} bytes")
EOF
    ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "=== $(date '+%F %T') done $STAGE" | tee -a "$LOG"
