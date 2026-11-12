"""Half-court windows that start during a stopped game clock (quick check, 2026-11-09).

One analyst flagged a possession whose ball sat behind the baseline for the first 3.6 s: a
baseline inbound with the clock stopped. The dead-ball rule of Phase 1 cuts a window at a clock
freeze inside it, not at its start. This counts, per half-court window, the leading run of 5 Hz
steps whose game clock equals the clock at step 0 (stopped clock at the start), from the frame
tables.

Output: reports/analysis/stopped_clock_windows.json (shares with leading stopped clock > 1 s and
> 3 s, quantiles of its length)

Usage:
    uv run python scripts/analysis_stopped_clock_windows.py
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ghost import data as D

EPS = 1e-3


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    keep = P.select(["game_id", "possession_id"])
    F = (pl.scan_parquet(sorted((base / "frames").glob("*.parquet")))
         .filter((pl.col("role") == "ball") & pl.col("valid"))
         .select(["game_id", "possession_id", "step", "game_clock"])
         .collect()
         .join(keep, on=["game_id", "possession_id"], how="semi")
         .sort(["game_id", "possession_id", "step"]))  # fmt: skip
    F = F.with_columns(
        (pl.col("game_clock") - pl.col("game_clock").first().over(["game_id", "possession_id"]))
        .abs()
        .gt(EPS)
        .cum_max()
        .over(["game_id", "possession_id"])
        .alias("running")
    )
    lead = F.group_by(["game_id", "possession_id"]).agg(
        ((~pl.col("running")).sum() * 0.2).alias("stopped_s")
    )
    s = lead["stopped_s"]
    rep = {
        "n_halfcourt_windows": lead.height,
        "share_stopped_start_gt_1s": float((s > 1.0).mean()),
        "share_stopped_start_gt_3s": float((s > 3.0).mean()),
        "quantiles_s_among_gt_1s": {
            str(q): float(s.filter(s > 1.0).quantile(q)) for q in (0.25, 0.5, 0.75, 0.95)
        },
        "definition": "leading run of valid 5 Hz steps whose game clock equals the clock at the "
        "first step",
    }
    out = Path("reports/analysis/stopped_clock_windows.json")
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
