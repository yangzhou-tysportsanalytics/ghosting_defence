"""Robustness of the player findings to score / clock context and to a split by date.

1. Score and clock in the context term (the charter's γ[context] lists score and clock bins;
   D-021 adopted the phase shares): offence score margin bins (≤ −16, −15..−6, 6..15, ≥ 16;
   reference −5..5), period (2, 3, 4, overtime; reference 1), clutch (period ≥ 4, ≤ 5 min left,
   |margin| ≤ 5) added to the main specification. Also: possessions with |margin| > 20
   (garbage time) dropped.
2. Split-half by date instead of game parity: games before vs on / after the median game date
   (players need possessions in both halves).

For sag and |dev|: context R², split-half reliability (adjusted, within team, within listed
position), team share, and the correlation of player means with the main specification.

Output: reports/phase4/<version>_all/context_time_robustness.json

Usage:
    uv run python scripts/phase4_context_time_robustness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_reliability import (  # noqa: E402
    MIN_POSS,
    PHASE_COVS,
    adjust,
    listed_positions,
    split_half,
    variance_shares,
    with_phases,
    within_position,
)

SCORE_CLOCK = ["m_le_m16", "m_m15_m6", "m_6_15", "m_ge_16", "per2", "per3", "per4", "per_ot",
               "clutch"]  # fmt: skip


def add_context(df: pl.DataFrame, poss: pl.DataFrame) -> pl.DataFrame:
    p = poss.select(["game_id", "possession_id", "period", "game_clock_start", "game_date"])
    m = pl.col("score_margin_offense").fill_null(0)
    return df.join(p, on=["game_id", "possession_id"], how="left").with_columns(
        (m <= -16).cast(pl.Float64).alias("m_le_m16"),
        ((m >= -15) & (m <= -6)).cast(pl.Float64).alias("m_m15_m6"),
        ((m >= 6) & (m <= 15)).cast(pl.Float64).alias("m_6_15"),
        (m >= 16).cast(pl.Float64).alias("m_ge_16"),
        (pl.col("period") == 2).cast(pl.Float64).alias("per2"),
        (pl.col("period") == 3).cast(pl.Float64).alias("per3"),
        (pl.col("period") == 4).cast(pl.Float64).alias("per4"),
        (pl.col("period") >= 5).cast(pl.Float64).alias("per_ot"),
        ((pl.col("period") >= 4) & (pl.col("game_clock_start") <= 300) & (m.abs() <= 5))
        .cast(pl.Float64)
        .alias("clutch"),
    )


def metrics(df: pl.DataFrame, y: str, extra: tuple[str, ...], players, pos, ref=None) -> dict:
    r, r2 = adjust(df, y, extra)
    d = df.with_columns(pl.Series("adj", r))
    tm = d.group_by("defense_team_id").agg(pl.col("adj").mean().alias("tm"))
    dw = d.join(tm, on="defense_team_id").with_columns(
        (pl.col("adj") - pl.col("tm")).alias("adj_w")
    )
    out = {
        "n_rows": d.height,
        "context_r2": r2,
        "split_half_r_adjusted": split_half(d, "adj", players)[0],
        "split_half_r_player_within_team": split_half(dw, "adj_w", players)[0],
        "split_half_r_within_position": within_position(d, "adj", players, pos)[
            "split_half_r_within_position"
        ],
        "team_share_of_systematic": variance_shares(d, "adj", players)["share_team_of_systematic"],
    }
    pm = d.filter(pl.col("def_id").is_in(players)).group_by("def_id").agg(pl.col("adj").mean())
    if ref is not None:
        j = pm.join(ref, on="def_id", suffix="_ref")
        out["corr_player_means_with_main_spec"] = float(np.corrcoef(j["adj"], j["adj_ref"])[0, 1])
    return out, pm


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    poss = pl.read_parquet(base / "possessions.parquet")
    df = with_phases(pl.read_parquet(base / "analysis" / "rule_ghost_dev.parquet"), base)
    df = add_context(df, poss)
    players = df.group_by("def_id").len().filter(pl.col("len") >= MIN_POSS)["def_id"].to_list()
    pos = listed_positions(cfg)
    main_spec = tuple(PHASE_COVS)
    cutoff = df.select(pl.col("game_id"), pl.col("game_date")).unique()["game_date"].sort()
    cutoff = cutoff[len(cutoff) // 2]
    rep: dict = {"n_rows": df.height, "n_players": len(players), "date_cutoff": str(cutoff),
                 "metrics": {}}  # fmt: skip
    for y in ("sag_ft", "dev_ft"):
        res = {}
        ref_m, ref_pm = metrics(df, y, main_spec, players, pos)
        res["main_specification"] = ref_m
        res["with_score_and_clock"], _ = metrics(df, y, (*main_spec, *SCORE_CLOCK), players, pos,
                                                 ref_pm)  # fmt: skip
        g = df.filter(pl.col("score_margin_offense").fill_null(0).abs() <= 20)
        res["garbage_time_dropped"], _ = metrics(g, y, main_spec, players, pos, ref_pm)
        res["garbage_time_dropped"]["share_rows_dropped"] = 1 - g.height / df.height
        t = df.with_columns((pl.col("game_date") >= cutoff).cast(pl.Int8).alias("parity"))
        res["split_by_date"], _ = metrics(t, y, main_spec, players, pos, ref_pm)
        rep["metrics"][y] = res
    out = Path("reports/phase4") / f"{cfg.version}_all" / "context_time_robustness.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
