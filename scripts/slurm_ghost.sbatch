#!/bin/bash
# Learned ghost in short Slurm jobs (scripts/gpu.py). Submit the same file again and again from
# the bundle directory; each job continues where the previous one stopped.
#   sbatch scripts/slurm_ghost.sbatch                  # one job
#   CHAIN=1 sbatch scripts/slurm_ghost.sbatch          # each job submits the next one itself
#   sbatch scripts/slurm_ghost.sbatch run crossfit:fold3   # one step (parallel jobs)
# Set the account before the first submission (sacctmgr -n show assoc user=$USER format=account%30).
#SBATCH --job-name=ghost
#SBATCH --account=def-CHANGEME
#SBATCH --time=02:55:00
# GPU type is required on Alliance clusters: a 3g.40gb MIG slice is enough for this small model
# (shorter queue); a full card is --gpus-per-node=h100:1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-ghost-%j.out

cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the bundle directory}" || exit 1
export PYTHONUTF8=1
unset PYTHONPATH  # Alliance sets it to its own site-packages, ahead of .venv
[ $# -gt 0 ] || set -- next
.venv/bin/python scripts/gpu.py "$@"
rc=$?
case $rc in
  0) echo "finished (exit 0)" ;;
  3) echo "time is up, progress was made (exit 3): submit again"
     if [ "${CHAIN:-0}" = 1 ] && [ "$1" = next ]; then
       sbatch --export=ALL,CHAIN=1 scripts/slurm_ghost.sbatch && echo "next job submitted"
     fi ;;
  4) echo "waiting for the config choice: .venv/bin/python scripts/gpu.py choose --suggested" ;;
  5) echo "no progress in this job (time too short, or other jobs hold the steps); not resubmitted" ;;
  *) echo "FAILED (exit $rc): see runs/gpu_state/journal.log and reports/logs/gpu_*.log" ;;
esac
case $rc in 0|3|4) exit 0 ;; *) exit $rc ;; esac  # 3 and 4 are normal stops, not failures
