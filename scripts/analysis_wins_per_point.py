"""Wins per point of net rating, 2015-16 (the wins conversion of the SSAC27 paper, D-031).

Full regular season (1,230 games; nbacore box scores, not only the tracked games). Per team:
wins, and net rating = 100 x (points scored - points allowed) / possessions, possessions per
team-game estimated from the player box scores as FGA + 0.44 FTA - OREB + TOV (team rebounds and
team turnovers are not in the player rows), averaged over the two teams of the game. OLS of wins
(82 games) on net rating over the 30 teams; slope = wins per point of net rating per 100
possessions.

Output: reports/analysis/wins_per_point.json

Usage:
    uv run python scripts/analysis_wins_per_point.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D


def main() -> None:
    G = D.L.box_games()
    Bx = D.L.box_players()
    tg = Bx.group_by(["game_id", "team_id"]).agg(
        pl.col("pts").sum(),
        (pl.col("fga").sum() + 0.44 * pl.col("fta").sum() - pl.col("oreb").sum()
         + pl.col("tov").sum()).alias("poss_est"),  # fmt: skip
    )
    poss = tg.group_by("game_id").agg(pl.col("poss_est").mean().alias("poss"))
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        rows.append(G.select(pl.col("game_id"), pl.col(f"{side}_team_id").alias("team_id"),
                             pl.col(f"{side}_score").cast(pl.Int32).alias("pf"),
                             pl.col(f"{opp}_score").cast(pl.Int32).alias("pa")))  # fmt: skip
    T = (pl.concat(rows).join(poss, on="game_id", how="left")
         .group_by("team_id").agg(pl.len().alias("games"), (pl.col("pf") > pl.col("pa")).sum().alias("wins"),
                                  pl.col("pf").sum(), pl.col("pa").sum(), pl.col("poss").sum())
         .with_columns((100 * (pl.col("pf") - pl.col("pa")) / pl.col("poss")).alias("net_rating"))
         .sort("team_id"))  # fmt: skip
    x, y = T["net_rating"].to_numpy(), T["wins"].to_numpy().astype(float)
    slope, icept = np.polyfit(x, y, 1)
    rep = {
        "season": "2015-16 regular season",
        "source": "NBA.com box scores, via the eoinamoore Kaggle dataset 'NBA Dataset: Box Scores "
                  "and Stats (1947 - Today)', version 515; team season totals only",
        "n_games": G.height, "n_teams": T.height,
        "games_per_team": sorted(set(T["games"].to_list())),
        "missing_possession_estimates": int(poss["poss"].null_count()),
        "wins_per_point_net_rating": float(slope), "intercept_wins": float(icept),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "mean_possessions_per_team_game": float(poss["poss"].mean()),
        "teams": T.to_dicts(),
    }  # fmt: skip
    out = Path("reports/analysis/wins_per_point.json")
    out.write_text(json.dumps(rep, indent=1))
    print({k: v for k, v in rep.items() if k != "teams"})


if __name__ == "__main__":
    main()
