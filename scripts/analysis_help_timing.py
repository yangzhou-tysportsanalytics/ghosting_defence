"""Help timing after on-ball screens (Phase 5 task 3a) and screen-confidence sensitivity.

Input: onball_screens.parquet (analysis_team_screen_defense.py): one row per on-ball screen group
with the first third-defender help (help B) within 3 s, its delay and helper, the group's highest
nbacore ``screen_confidence`` (v1.6) and the defensive lineup.

(a) Teams: median delay of help, share of screens with help within 1 s ("early"), and their
    association with points allowed per possession (tracked games).
(b) Players as potential helpers: opportunities = screens where the player is on defence and is
    neither the screened nor the screener's defender; help propensity = first helps /
    opportunities; median delay of the player's helps. Players with >= 300 opportunities;
    split-half reliability by game parity.
(c) Sensitivity of the team results to screen confidence: all groups, confidence >= 0.5, top half.

Descriptive; the play-level caveat of analysis_help_value.py applies (help responds to threat).
Output: reports/analysis/help_timing.json, help_timing_teams.csv, help_timing_players.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ghost import data as D

EARLY_S = 1.0
MIN_OPP = 300


def corr(x: np.ndarray, y: np.ndarray) -> dict:
    r, p = stats.pearsonr(x, y)
    z, se = np.arctanh(r), 1 / np.sqrt(len(x) - 3)
    return {
        "r": float(r),
        "p": float(p),
        "ci95": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))],
    }


def team_table(G: pl.DataFrame) -> pl.DataFrame:
    return (
        G.group_by("defense_team_id")
        .agg(
            pl.len().alias("n_screens"),
            pl.col("switched").mean().alias("switch_rate"),
            pl.col("helped_3s").mean().alias("help_rate_3s"),
            (pl.col("help_delay_s") <= EARLY_S).fill_null(False).mean().alias("early_help_rate"),
            pl.col("help_delay_s").median().alias("help_delay_median_s"),
        )
        .sort("defense_team_id")
    )


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    G = pl.read_parquet(cfg.processed_dir / "all" / "analysis" / "onball_screens.parquet")
    pts = pl.read_csv("reports/analysis/team_points_allowed.csv").rename(
        {"team_id": "defense_team_id"}
    )
    abbr = pl.read_csv("reports/analysis/team_screen_defense.csv").select(
        pl.col("team_id").alias("defense_team_id"), "team"
    )
    rep: dict = {"n_screens": G.height, "early_s": EARLY_S}

    # (a) teams
    T = team_table(G).join(pts, on="defense_team_id").join(abbr, on="defense_team_id")
    y = T["pts_allowed_per_poss"].to_numpy()
    rep["teams"] = {
        "league_median_delay_s": float(G["help_delay_s"].median()),
        "league_early_help_rate": float((G["help_delay_s"] <= EARLY_S).fill_null(False).mean()),
        "team_median_delay_range_s": [
            float(T["help_delay_median_s"].min()),
            float(T["help_delay_median_s"].max()),
        ],
        "team_early_help_range": [
            float(T["early_help_rate"].min()),
            float(T["early_help_rate"].max()),
        ],
        "most_aggressive_by_early_rate": T.sort("early_help_rate", descending=True)
        .head(5)["team"]
        .to_list(),
        "least_aggressive_by_early_rate": T.sort("early_help_rate").head(5)["team"].to_list(),
        "early_help_rate_vs_pts": corr(T["early_help_rate"].to_numpy(), y),
        "median_delay_vs_pts": corr(T["help_delay_median_s"].to_numpy(), y),
        "help_rate_3s_vs_pts": corr(T["help_rate_3s"].to_numpy(), y),
    }
    T.sort("early_help_rate", descending=True).write_csv("reports/analysis/help_timing_teams.csv")

    # (b) players as potential helpers
    par = D.splits(cfg).select(["game_id", "parity"])
    opp = (
        G.select(
            [
                "screen_group",
                "game_id",
                "screened_def_id",
                "screener_def_id",
                "defense_player_ids",
                "first_helper_id",
                "help_delay_s",
            ]
        )  # fmt: skip
        .explode("defense_player_ids")
        .rename({"defense_player_ids": "def_id"})
        .filter(
            (pl.col("def_id") != pl.col("screened_def_id"))
            & (pl.col("def_id") != pl.col("screener_def_id"))
        )
        .join(par, on="game_id", how="left")
        .with_columns(
            (pl.col("first_helper_id") == pl.col("def_id")).fill_null(False).alias("helped")
        )
    )
    Pl = (
        opp.group_by("def_id")
        .agg(
            pl.len().alias("n_opp"),
            pl.col("helped").mean().alias("help_propensity"),
            pl.col("help_delay_s").filter(pl.col("helped")).median().alias("median_delay_s"),
            pl.col("helped").sum().alias("n_helps"),
        )
        .filter(pl.col("n_opp") >= MIN_OPP)
    )
    halves = (
        opp.filter(pl.col("def_id").is_in(Pl["def_id"].implode()))
        .group_by(["def_id", "parity"])
        .agg(pl.col("helped").mean())
        .pivot(on="parity", index="def_id", values="helped")
        .drop_nulls()
    )
    rep["players"] = {
        "n_players": Pl.height,
        "help_propensity_quantiles": {
            q: float(Pl["help_propensity"].quantile(q)) for q in (0.1, 0.5, 0.9)
        },
        "split_half_r_help_propensity": float(np.corrcoef(halves["0"], halves["1"])[0, 1]),
        "note": "opportunity = on-ball screen with the player on defence, not the screened or screener's defender",
    }
    Pl.sort("help_propensity", descending=True).write_csv(
        "reports/analysis/help_timing_players.csv"
    )

    # (c) sensitivity to screen confidence
    med = float(G["screen_confidence"].median())
    rep["confidence_sensitivity"] = {}
    for lab, sub in (
        ("all", G),
        ("confidence_ge_0.5", G.filter(pl.col("screen_confidence") >= 0.5)),
        (f"top_half_ge_{med:.2f}", G.filter(pl.col("screen_confidence") >= med)),
    ):
        Ts = team_table(sub).join(pts, on="defense_team_id")
        ys = Ts["pts_allowed_per_poss"].to_numpy()
        rep["confidence_sensitivity"][lab] = {
            "n_screens": sub.height,
            "league_switch_rate": float(sub["switched"].mean()),
            "league_help_rate_3s": float(sub["helped_3s"].mean()),
            "help_rate_3s_vs_pts": corr(Ts["help_rate_3s"].to_numpy(), ys),
            "switch_rate_vs_pts": corr(Ts["switch_rate"].to_numpy(), ys),
            "corr_team_help_rate_with_all_groups": float(
                np.corrcoef(
                    Ts["help_rate_3s"].to_numpy(),
                    T.sort("defense_team_id")["help_rate_3s"].to_numpy(),
                )[0, 1]
            ),
        }
    out = Path("reports/analysis")
    (out / "help_timing.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
