"""Reliability of the expected points conceded relative to the learned ghost (dEP).

From learned_ghost_points_<ghost>.parquet (phase4_learned_ghost_points.py; one row per shot and
defender slot): split-half reliability by game parity of the player mean individual-ghost dEP
(players with >= 100 defender-shots in each half) and of the team mean team-ghost dEP (one row
per shot); league means; spread of player means (>= 200 defender-shots).

Output: reports/phase4/<version>_all/learned_ghost_points_reliability_<ghost>.json

Usage:
    uv run python scripts/analysis_learned_points_reliability.py [--ghost cf_gpu_d128_lr1e-3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D


def split_half(d: pl.DataFrame, key: str, col: str, min_n: int) -> tuple[float, int]:
    pm = (
        d.group_by([key, "parity"]).agg(pl.col(col).mean(), pl.len()).filter(pl.col("len") >= min_n)
    )
    w = pm.pivot(on="parity", index=key, values=col).drop_nulls()
    return float(np.corrcoef(w["0"], w["1"])[0, 1]), w.height


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghost", default="cf_gpu_d128_lr1e-3")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    g = pl.read_parquet(
        cfg.processed_dir / "all" / "analysis" / f"learned_ghost_points_{args.ghost}.parquet"
    )
    shots = g.unique(subset=["game_id", "event_uid"])
    r_p, n_p = split_half(g, "def_id", "dEP_ind", 100)
    r_t, n_t = split_half(shots, "defense_team_id", "dEP_team", 1)
    pm = g.group_by("def_id").agg(pl.col("dEP_ind").mean(), pl.len()).filter(pl.col("len") >= 200)
    rep = {
        "ghost": args.ghost,
        "n_shots": shots.height,
        "n_defender_shots": g.height,
        "league_mean_xfg_real": float(shots["xfg_real"].mean()),
        "league_mean_xfg_individual_ghost": float(g["xfg_ind_ghost"].mean()),
        "league_mean_xfg_team_ghost": float(shots["xfg_team_ghost"].mean()),
        "league_mean_dEP_individual_per_defender_shot": float(g["dEP_ind"].mean()),
        "league_mean_dEP_team_per_shot": float(shots["dEP_team"].mean()),
        "player_dEP_individual_split_half_r": r_p,
        "player_n_ge_100_per_half": n_p,
        "player_dEP_individual_sd_ge_200": float(pm["dEP_ind"].std()),
        "team_dEP_team_split_half_r": r_t,
        "n_teams": n_t,
        "note": "team-ghost slots are drawn independently (the model gives each slot's distribution, "
        "not their joint), so the team ghost's defence is less coordinated than a real one and its "
        "xFG is biased upward; the team dEP level is not interpretable, its ranking is",
    }
    p = (
        Path("reports/phase4")
        / f"{cfg.version}_all"
        / f"learned_ghost_points_reliability_{args.ghost}.json"
    )
    p.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
