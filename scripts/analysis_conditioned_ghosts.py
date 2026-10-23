"""Conditioned ghosts vs the league ghost on games they never saw (decision D-028).

The lineup and scheme ghosts were trained on the train games only, so on the validation and test
games their deviations are out of sample, like the league ghost's cross-fitted ones. On the same
(possession, defender) rows of those games, for the individual- and team-ghost NLL:
* league means and the possession-level correlation;
* player effects (main specification: context-adjusted with the D-021 phase shares, within
  team; players with >= 100 defensive possessions in these games): correlation with the league
  ghost's, their spread (SD) relative to the league ghost's, and their split-half reliability
  (game parity) - conditioning on identity is expected to shrink the lineup ghost's player
  effects (the defender's own identity is an input);
* team means of the context-adjusted values: SD and correlation (the scheme ghost is expected to
  absorb part of the team effect).

Output: reports/phase4/<version>_all/conditioned_ghosts_valtest.json

Usage:
    uv run python scripts/analysis_conditioned_ghosts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_reliability import PHASE_COVS, adjust, with_phases  # noqa: E402

LEAGUE = "cf_gpu_d128_lr1e-3"
CONDITIONED = {"lineup": "gpu_d128_lr1e-3_lineup_valtest",
               "scheme": "gpu_d128_lr1e-3_scheme_valtest"}  # fmt: skip
METRICS = ("nll_ind", "nll_team")
MIN_POSS = 100
KEYS = ["game_id", "possession_id", "def_id"]


def effects(d: pl.DataFrame, col: str, players: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Player effects within team (and by parity) and team means of the adjusted values."""
    res, _ = adjust(d, col, tuple(PHASE_COVS))
    d = d.with_columns(pl.Series("adj", res))
    tm = d.group_by("defense_team_id").agg(pl.col("adj").mean().alias("tm"))
    d = d.join(tm, on="defense_team_id").with_columns((pl.col("adj") - pl.col("tm")).alias("w"))
    d = d.filter(pl.col("def_id").is_in(players))
    pe = d.group_by("def_id").agg(pl.col("w").mean().alias("all"))
    halves = d.group_by(["def_id", "parity"]).agg(pl.col("w").mean())
    halves = halves.pivot(on="parity", index="def_id", values="w")
    return pe.join(halves, on="def_id").drop_nulls().sort("def_id"), tm.sort("defense_team_id")


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    rule = pl.read_parquet(base / "analysis" / "rule_ghost_dev.parquet")
    league = pl.read_parquet(base / "analysis" / f"learned_ghost_dev_{LEAGUE}.parquet")
    rep: dict = {"league_ghost": LEAGUE, "min_possessions": MIN_POSS, "comparisons": {}}
    for label, name in CONDITIONED.items():
        cond = pl.read_parquet(base / "analysis" / f"learned_ghost_dev_{name}.parquet")
        d = rule.join(league.select(KEYS + list(METRICS)), on=KEYS, how="inner").join(
            cond.select(KEYS + list(METRICS)), on=KEYS, how="inner", suffix="_cond"
        )
        d = with_phases(d, base)
        players = d.group_by("def_id").len().filter(pl.col("len") >= MIN_POSS)["def_id"].to_list()
        out = {"n_rows": d.height, "n_games": d["game_id"].n_unique(), "n_players": len(players)}
        for m in METRICS:
            pl_l, tm_l = effects(d, m, players)
            pl_c, tm_c = effects(d, f"{m}_cond", players)
            sh = lambda x: float(np.corrcoef(x["0"], x["1"])[0, 1])  # noqa: E731
            out[m] = {
                "league_mean": float(d[m].mean()),
                "conditioned_mean": float(d[f"{m}_cond"].mean()),
                "corr_possession_defender": float(np.corrcoef(d[m], d[f"{m}_cond"])[0, 1]),
                "player_effects": {
                    "corr_league_vs_conditioned": float(np.corrcoef(pl_l["all"], pl_c["all"])[0, 1]),
                    "sd_league": float(pl_l["all"].std()),
                    "sd_conditioned": float(pl_c["all"].std()),
                    "sd_ratio_conditioned_to_league": float(pl_c["all"].std() / pl_l["all"].std()),
                    "split_half_r_league": sh(pl_l),
                    "split_half_r_conditioned": sh(pl_c),
                },
                "team_means": {
                    "corr_league_vs_conditioned": float(np.corrcoef(tm_l["tm"], tm_c["tm"])[0, 1]),
                    "sd_league": float(tm_l["tm"].std()),
                    "sd_conditioned": float(tm_c["tm"].std()),
                    "sd_ratio_conditioned_to_league": float(tm_c["tm"].std() / tm_l["tm"].std()),
                },
            }  # fmt: skip
            v = out[m]
            print(f"{label:>6} {m}: mean {v['league_mean']:.3f} -> {v['conditioned_mean']:.3f}; "
                  f"players r {v['player_effects']['corr_league_vs_conditioned']:.3f}, SD ratio "
                  f"{v['player_effects']['sd_ratio_conditioned_to_league']:.2f}, split-half "
                  f"{v['player_effects']['split_half_r_league']:.2f} -> "
                  f"{v['player_effects']['split_half_r_conditioned']:.2f}; teams SD ratio "
                  f"{v['team_means']['sd_ratio_conditioned_to_league']:.2f}", flush=True)  # fmt: skip
        rep["comparisons"][label] = out
    p = Path("reports/phase4") / f"{cfg.version}_all" / "conditioned_ghosts_valtest.json"
    p.write_text(json.dumps(rep, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
