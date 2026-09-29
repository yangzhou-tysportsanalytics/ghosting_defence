"""Phase 5 task 2: face validity of player-level measures against public defensive metrics.

Our measures (players with >= 300 defensive half-court possessions):
  sag_effect, dev_effect  within-team player effects from phase4_hier.py (rule ghost)
  help_b_rate, help_a_rate  help onsets per defensive possession (phase2_events.py)
  indiv_dep  mean expected points conceded vs the individual rule ghost, per shot defended
             (phase4_ghost_points.py; defenders with >= 100 shots)
External (nbacore v1.5 L5; local research use; RAPTOR CC BY 4.0, credit FiveThirtyEight):
  dbpm (Basketball-Reference), raptor_defense, predator_defense, all_defense (2015-16 teams),
  dpoy vote share.
Spearman correlations with p-values; All-Defensive vs other players (Mann-Whitney).
Expectation (analysis plan): moderate at best; sag is a positioning style, not a value.

Output: reports/analysis/player_validity.json

Usage:
    uv run python scripts/analysis_player_validity.py [--source data/processed/nbacore-v1.0/all]
        [--hier reports/phase4/v1.2_all]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ghost import data as D

MIN_POSS = 300


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="processed game-set dir (default: config)")
    ap.add_argument("--hier", default=None, help="phase-4 report dir (default: config version)")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    src = Path(args.source) if args.source else cfg.processed_dir / "all"
    hier = Path(args.hier) if args.hier else Path("reports/phase4") / f"{cfg.version}_all"

    dev = pl.read_parquet(src / "analysis" / "rule_ghost_dev.parquet")
    n_poss = dev.group_by("def_id").len().rename({"def_id": "player_id", "len": "n_def_poss"})
    help_ = pl.read_parquet(src / "events" / "help.parquet")
    hr = (
        help_.group_by(["def_id", "definition"])
        .len()
        .pivot(on="definition", index="def_id", values="len")
        .rename({"def_id": "player_id"})
    )
    hr = hr.rename({c: f"n_help_{c.lower()}" for c in hr.columns if c != "player_id"})
    P = (
        n_poss.join(hr, on="player_id", how="left")
        .fill_null(0)
        .with_columns(
            (pl.col("n_help_b") / pl.col("n_def_poss")).alias("help_b_rate"),
            (pl.col("n_help_a") / pl.col("n_def_poss")).alias("help_a_rate"),
        )
        .filter(pl.col("n_def_poss") >= MIN_POSS)
    )
    for m in ("sag_ft", "dev_ft"):
        f = hier / f"hier_players_{m}.csv"
        if f.exists():
            h = (
                pl.read_csv(f)
                .group_by("player_id")
                .agg((pl.col("effect_mean") * pl.col("n_poss")).sum() / pl.col("n_poss").sum())
                .rename({"effect_mean": f"{m.split('_')[0]}_effect"})
            )
            P = P.join(h, on="player_id", how="left")
    gp = src / "analysis" / "ghost_points.parquet"
    if gp.exists():
        g = pl.read_parquet(gp).filter(
            pl.col("dEP_indiv").is_not_null() & (pl.col("shooter_def_ids").list.len() == 1)
        )
        g = g.with_columns(pl.col("shooter_def_ids").list.first().alias("player_id"))
        gi = g.group_by("player_id").agg(
            pl.col("dEP_indiv").mean().alias("indiv_dep"), pl.len().alias("n_shots_def")
        )
        P = P.join(gi.filter(pl.col("n_shots_def") >= 100), on="player_id", how="left")

    adv = D.L.advanced(cfg.version)
    # one row per player: the season total row when traded, otherwise the single team row
    adv = adv.sort("is_total", descending=True).unique(subset=["player_id"], keep="first")
    rap = D.L.raptor(cfg.version).select(
        ["player_id", "raptor_defense", "predator_defense", "poss"]
    )
    aw = D.L.awards(cfg.version)
    alld = (
        aw.filter(pl.col("award") == "all_defense")
        .select("player_id")
        .unique()
        .with_columns(pl.lit(True).alias("all_defense"))
    )
    dpoy = (
        aw.filter(pl.col("award") == "dpoy")
        .select(["player_id", "share"])
        .rename({"share": "dpoy_share"})
    )
    X = (
        P.join(adv.select(["player_id", "dbpm", "pos"]), on="player_id", how="left")
        .join(rap, on="player_id", how="left")
        .join(alld, on="player_id", how="left")
        .join(dpoy, on="player_id", how="left")
        .with_columns(pl.col("all_defense").fill_null(False), pl.col("dpoy_share").fill_null(0.0))
    )

    ours = [
        c
        for c in ("sag_effect", "dev_effect", "help_b_rate", "help_a_rate", "indiv_dep")
        if c in X.columns
    ]
    ext = ["dbpm", "raptor_defense", "predator_defense", "dpoy_share"]
    res = {
        "n_players": X.height,
        "n_all_defense": int(X["all_defense"].sum()),
        "spearman": {},
        "all_defense": {},
    }
    for a in ours:
        for b in ext:
            d = X.select([a, b]).drop_nulls()
            if d.height < 20:
                continue
            rho, p = stats.spearmanr(d[a], d[b])
            res["spearman"][f"{a}__{b}"] = {"rho": float(rho), "p": float(p), "n": d.height}
        d = X.select([a, "all_defense"]).drop_nulls()
        yes, no = (
            d.filter(pl.col("all_defense"))[a].to_numpy(),
            d.filter(~pl.col("all_defense"))[a].to_numpy(),
        )
        if yes.size >= 3:
            u = stats.mannwhitneyu(yes, no)
            res["all_defense"][a] = {
                "mean_all_defense": float(yes.mean()),
                "mean_others": float(no.mean()),
                "n_all_defense": int(yes.size),
                "p_mannwhitney": float(u.pvalue),
            }
    res["sources"] = {
        "basketball_reference": "local research use only; do not redistribute",
        "raptor": "FiveThirtyEight, CC BY 4.0 (attribution required)",
    }
    res["note"] = (
        "face validity, descriptive; sag is a positioning style; help rates use the "
        "unvalidated help-B definition; rule-ghost measures are a CPU-only stand-in for "
        "the learned ghost"
    )
    out = Path("reports/analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "player_validity.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    np.set_printoptions(precision=3)


if __name__ == "__main__":
    main()
