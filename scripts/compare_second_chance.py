"""Second-chance sensitivity (D-018): headline results without vs with second-chance possessions.

The main results (this repository's reports/ and processed_dir) against the same chain rerun on
windows that include second chances (``windows_release: l2_sc``; by default in
runs/sc_sensitivity/, see its run.sh). Also compares the rule-ghost deviations of second-chance
possessions with the others.

Output: reports/analysis/second_chance_sensitivity.json

Usage:
    uv run python scripts/compare_second_chance.py [--sc runs/sc_sensitivity]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D


def jload(p: Path) -> dict:
    return json.loads(p.read_text())


def pick(d: dict, *keys):
    for k in keys:
        d = d[k]
    return d


def headline(root: Path, version: str) -> dict:
    r4 = root / "reports/phase4" / f"{version}_all"
    ra = root / "reports/analysis"
    out: dict = {}
    st = jload(root / "reports/phase1" / f"{version}_all" / "possessions_stats.json")
    out["n_possessions"] = st["n_possessions"]
    out["n_halfcourt"] = st["n_halfcourt_v1"]
    fit = jload(root / "reports/phase2" / f"{version}_all" / "matchup_fit.json")
    out["hmm_strat_help_gamma"] = pick(fit, "models", "hmm_strat_help", "params", "gamma")
    for ctx, sfx in (("base", ""), ("phases", "_phases")):
        rel = jload(r4 / f"reliability{sfx}.json")["metrics"]
        wp = jload(r4 / f"reliability_within_position{sfx}.json")
        hier = jload(r4 / f"hier{sfx}.json")["metrics"]
        for y in ("sag_ft", "dev_ft"):
            h = hier[y]
            out[f"{y}_{ctx}"] = {
                "n_players_ge300": rel[y]["n_players"],
                "split_half_r_adjusted": rel[y]["split_half_r_adjusted"],
                "split_half_r_within_position": wp[y]["split_half_r_within_position"],
                "hier_team_share": h["prior_sensitivity"]["1.0"]["team_share"],
                "share_player_intervals_excluding_zero": h["share_player_intervals_excluding_zero"],
            }
    tv = jload(ra / "team_validity.json")
    ht = jload(ra / "help_timing.json")
    tsd = jload(ra / "team_screen_defense.json")["league"]
    out["team"] = {
        "n_on_ball_screens": tsd["n_on_ball_screen_groups"],
        "league_help_rate_3s": tsd["help_rate_3s"],
        "league_switch_rate": tsd["switch_rate"],
        "league_help_delay_median_s": tsd["help_delay_median_s"],
        "help_rate_3s_vs_pts": {
            "r": tv["help_rate_3s__vs__pts_allowed_per_poss"]["pearson_r"],
            "ci95": tv["inference_pts_allowed_per_poss"]["help_r_ci95"],
        },
        "switch_rate_vs_pts": {
            "r": tv["switch_rate__vs__pts_allowed_per_poss"]["pearson_r"],
            "ci95": tv["inference_pts_allowed_per_poss"]["switch_r_ci95"],
        },
        "early_help_rate_vs_pts": ht["teams"]["early_help_rate_vs_pts"],
        "league_early_help_rate": ht["teams"]["league_early_help_rate"],
    }
    pv = jload(root / "reports/phase5" / f"{version}_all" / "predictive_validity.json")
    pl_ = pv["player_level"]
    out["predictive_validity"] = {
        "n_shot_possessions": pv["n_shot_possessions"],
        "space_per_ft_sag": pick(pv, "possession_level", "def1_ft", "pre_sag_ft"),
        "xfg_m3_per_ft_sag": pick(pv, "possession_level", "xfg_m3_cf", "pre_sag_ft"),
        "player_rho": {
            k: [pl_[p][k]["rho"] for p in sorted(pl_)]
            for k in (
                "def1_ft",
                "xfg_m3_cf",
                "def1_ft__within_position",
                "xfg_m3_cf__within_position",
            )
        },
    }
    return out


def player_effect_corr(main: Path, sc: Path, version: str, sfx: str) -> dict:
    res = {}
    for y in ("sag_ft", "dev_ft"):
        f = f"hier_players_{y}{sfx}.csv"
        a = pl.read_csv(main / "reports/phase4" / f"{version}_all" / f)
        b = pl.read_csv(sc / "reports/phase4" / f"{version}_all" / f)
        j = a.join(b, on=["player_id", "team"], suffix="_sc")
        j = j.filter((pl.col("n_poss") >= 300) & (pl.col("n_poss_sc") >= 300))
        res[y] = {
            "pearson_r": float(np.corrcoef(j["effect_mean"], j["effect_mean_sc"])[0, 1]),
            "max_abs_change_ft": float((j["effect_mean"] - j["effect_mean_sc"]).abs().max()),
            "n_player_cells": j.height,
        }
    return res


def second_chance_deviation(sc: Path) -> dict:
    """Rule-ghost deviations of second-chance possessions vs the others (sensitivity build)."""
    cfg = D.DataConfig.load(path=sc / "configs/data.yaml", game_set="all")
    proc = sc / cfg.processed_dir
    flag = pl.read_parquet(proc / "windows_l2_sc.parquet").select(
        ["window_uid", "is_second_chance"]
    )
    P = pl.read_parquet(proc / "all" / "possessions.parquet").select(
        ["game_id", "possession_id", "window_uid"]
    )
    dev = pl.read_parquet(proc / "all" / "analysis" / "rule_ghost_dev.parquet").join(
        P.join(flag, on="window_uid"), on=["game_id", "possession_id"]
    )
    g = dev.group_by("is_second_chance").agg(
        pl.len().alias("n_possession_defender"),
        pl.struct(["game_id", "possession_id"]).n_unique().alias("n_possessions"),
        pl.col("sag_ft").mean(),
        pl.col("dev_ft").mean(),
    )
    return {("second_chance" if r["is_second_chance"] else "other"): r for r in g.to_dicts()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", default="runs/sc_sensitivity")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    main_root, sc = Path("."), Path(args.sc)
    rep = {
        "note": "Main = v1 windows (second chances excluded, D-018); with_sc = the same chain on "
        "windows that start at the offensive-rebound control when the ball never leaves the "
        "frontcourt (nbacore include_second_chance). Second chances shot within 4 s of the "
        "rebound are flagged transition and stay excluded, as in v1.",
        "windows": jload(sc / "data/processed/nbacore-v1.6-l2sc/windows_l2_sc_stats.json"),
        "main": headline(main_root, cfg.version),
        "with_sc": headline(sc, cfg.version),
        "player_effects_main_vs_with_sc": {
            "base": player_effect_corr(main_root, sc, cfg.version, ""),
            "phases": player_effect_corr(main_root, sc, cfg.version, "_phases"),
        },
        "deviation_by_possession_type_with_sc": second_chance_deviation(sc),
    }
    out = Path("reports/analysis/second_chance_sensitivity.json")
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
