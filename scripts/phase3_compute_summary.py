"""Compute used for the learned ghost (public source for the compute figures; writing v0.10).

From the cluster job outputs (reports/logs/cluster/slurm-ghost-*.out, lines "<time> [<job>]
start|done <step>") and the training histories (runs/phase3/<version>_all_<tag>/history.json):
wall time of every completed step, the job time of every job (first to last log line, including
failed attempts), seconds per epoch by configuration (median of the logged training time per
epoch; validation each epoch excluded), and the device.

Output: reports/phase3/<version>_all_gpu_compute.json

Usage:
    uv run python scripts/phase3_compute_summary.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from ghost import data as D

LINE = re.compile(r"^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8}) \[([0-9]+)\] (.*)$")


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    steps, jobs, device = {}, {}, set()
    for f in sorted(Path("reports/logs/cluster").glob("slurm-ghost-*.out")):
        open_steps = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            m = LINE.match(line)
            if not m:
                continue
            t, job, msg = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S"), m[2], m[3]
            j = jobs.setdefault(job, [t, t])
            j[0], j[1] = min(j[0], t), max(j[1], t)
            if msg.startswith("node "):
                device.add(msg.split("GPU ", 1)[1].split(",")[0])
            elif msg.startswith("start "):
                open_steps[msg[6:]] = t
            elif msg.startswith("done ") and msg[5:] in open_steps:
                s = msg[5:]
                steps[s] = steps.get(s, 0.0) + (t - open_steps.pop(s)).total_seconds() / 60
    per_epoch = {}
    for h in sorted(Path("runs/phase3").glob(f"{cfg.version}_all_gpu_*/history.json")):
        hist = json.loads(h.read_text())
        if hist:
            tag = h.parent.name.removeprefix(f"{cfg.version}_all_")
            per_epoch[tag] = {"epochs": len(hist),
                              "median_s": float(np.median([r["seconds"] for r in hist]))}  # fmt: skip
    job_min = {k: (v[1] - v[0]).total_seconds() / 60 for k, v in jobs.items()}
    rep = {
        "device": sorted(device),
        "n_jobs": len(jobs),
        "step_wall_minutes": {k: round(v, 1) for k, v in sorted(steps.items())},
        "sum_step_wall_hours": round(sum(steps.values()) / 60, 2),
        "sum_job_hours_incl_failed_attempts": round(sum(job_min.values()) / 60, 2),
        "training_seconds_per_epoch": per_epoch,
        "note": "One MIG slice (3/7 of an H100) per job; jobs under 3 h with checkpoint "
        "resumption. Step wall time includes data loading and evaluation.",
    }
    p = Path("reports/phase3") / f"{cfg.version}_all_gpu_compute.json"
    p.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
