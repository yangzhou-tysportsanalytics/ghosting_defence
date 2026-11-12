"""Out-of-sample team check (review A3): early help in the tracked games vs defensive rating in
the rest of the 2015-16 regular season.

* Early-help rate per defence (tracking period, games through 2016-01-23): share of on-ball
  screens with third-defender help within 1 s of contact (as in the team analysis).
* Defensive rating after 2016-01-23 (all games to 2016-04-13; no tracking): points allowed per
  100 possessions. Points from the nbacore box scores; possessions estimated from the
  play-by-play of every game, FGA - OREB + TOV + 0.44 FTA per team, averaged over the two teams
  of a game (the usual estimate). The same rating for the tracking period is given for reference.
* Pearson r (30 teams) with a team-bootstrap 95 % interval and a permutation p; and the
  regression of the later rating on early help and the tracking-period rating (does early help
  predict later defence beyond the defence's level so far?).
Only nbacore data (the full-season play-by-play and box scores); nothing external.

Output: reports/analysis/team_out_of_sample.json

Usage:
    uv run python scripts/analysis_team_out_of_sample.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D

CUT = date(2016, 1, 23)
EARLY_S = 1.0


def team_game_possessions(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per (game, team): FGA, FTA, TOV, OREB and the possession estimate."""
    team = pl.coalesce(
        "p1_team_id",
        pl.when(pl.col("p1_id") >= 1610612737).then(pl.col("p1_id")),  # team rebounds / turnovers
    )
    desc = pl.concat_str(["home_desc", "visitor_desc"], separator=" ", ignore_nulls=True)
    p = pbp.sort(["game_id", "event_num"]).with_columns(team.alias("team"), desc.alias("desc"))
    miss = (pl.col("msg_type") == 2) | (
        (pl.col("msg_type") == 3) & pl.col("desc").str.contains("MISS")
    )
    p = p.with_columns(
        pl.when(miss).then(pl.col("team")).forward_fill().over("game_id").alias("last_miss_team")
    )
    p = p.with_columns(
        (
            (pl.col("msg_type") == 4)
            # player rebounds only: team rebounds include dead-ball placeholders after missed FTs
            & pl.col("p1_team_id").is_not_null()
            & (pl.col("team") == pl.col("last_miss_team").shift(1).over("game_id"))
        ).alias("oreb")
    )
    agg = (
        p.filter(pl.col("team").is_not_null())
        .group_by(["game_id", "team"])
        .agg(
            pl.col("msg_type").is_in([1, 2]).sum().alias("fga"),
            (pl.col("msg_type") == 3).sum().alias("fta"),
            (pl.col("msg_type") == 5).sum().alias("tov"),
            pl.col("oreb").sum().alias("oreb"),
        )
        .with_columns(
            (pl.col("fga") - pl.col("oreb") + pl.col("tov") + 0.44 * pl.col("fta")).alias(
                "poss_est"
            )
        )
    )
    game_poss = agg.group_by("game_id").agg(pl.col("poss_est").mean().alias("poss"))
    return agg.join(game_poss, on="game_id")


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    S = pl.read_parquet(base / "analysis" / "onball_screens.parquet")
    early = (
        S.group_by("defense_team_id")
        .agg(
            (pl.col("help_delay_s").fill_null(99) <= EARLY_S).mean().alias("early_help"),
            pl.len().alias("n_screens"),
        )  # fmt: skip
        .sort("defense_team_id")
    )
    box = D.L.box_games(cfg.version)
    poss = team_game_possessions(D.L.pbp(cfg.version))
    games = box.join(poss.select(["game_id", "poss"]).unique(), on="game_id", how="inner")
    rows = pl.concat([
        games.select(pl.col("home_team_id").alias("team"), pl.col("away_score").alias("pts_allowed"),
                     "poss", "game_date", "game_id"),
        games.select(pl.col("away_team_id").alias("team"), pl.col("home_score").alias("pts_allowed"),
                     "poss", "game_date", "game_id"),
    ])  # fmt: skip
    rating = (
        rows.with_columns((pl.col("game_date") > CUT).alias("after"))
        .group_by(["team", "after"])
        .agg(pl.col("pts_allowed").sum(), pl.col("poss").sum(), pl.len().alias("games"))
        .with_columns((100 * pl.col("pts_allowed") / pl.col("poss")).alias("drtg"))
    )
    wide = (
        early.join(rating.filter(pl.col("after")).select(pl.col("team").alias("defense_team_id"),
                   pl.col("drtg").alias("drtg_after"), pl.col("games").alias("games_after")),
                   on="defense_team_id")
        .join(rating.filter(~pl.col("after")).select(pl.col("team").alias("defense_team_id"),
              pl.col("drtg").alias("drtg_before"), pl.col("games").alias("games_before")),
              on="defense_team_id")
        .sort("defense_team_id")
    )  # fmt: skip
    x, y, z = (
        wide[c].to_numpy().astype(float) for c in ("early_help", "drtg_after", "drtg_before")
    )
    n = len(x)
    rng = np.random.default_rng(9)
    r = float(np.corrcoef(x, y)[0, 1])
    boot = [np.corrcoef(x[i], y[i])[0, 1] for i in (rng.integers(0, n, n) for _ in range(5000))]
    perm = np.mean([abs(np.corrcoef(x, rng.permutation(y))[0, 1]) >= abs(r) for _ in range(10000)])
    X = np.column_stack([np.ones(n), (x - x.mean()) / x.std(), z])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    bb = []
    for i in (rng.integers(0, n, n) for _ in range(5000)):
        Xi = np.column_stack([np.ones(n), (x[i] - x[i].mean()) / x[i].std(), z[i]])
        bb.append(np.linalg.lstsq(Xi, y[i], rcond=None)[0][1])
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]  # noqa: E731
    rep = {
        "cut_date": str(CUT),
        "n_teams": n,
        "games_after_per_team": [int(wide["games_after"].min()), int(wide["games_after"].max())],
        "possession_estimate": "FGA - OREB + TOV + 0.44 FTA per team, mean of the two teams",
        "r_early_help_vs_drtg_after": {
            "r": r,
            "ci95_team_bootstrap": q(boot),
            "p_permutation": float(perm),
        },  # fmt: skip
        "r_early_help_vs_drtg_before": float(np.corrcoef(x, z)[0, 1]),
        "r_drtg_before_vs_after": float(np.corrcoef(z, y)[0, 1]),
        "regression_drtg_after": {
            "early_help_per_sd": float(beta[1]),
            "ci95_team_bootstrap": q(bb),
            "drtg_before_coef": float(beta[2]),
            "note": "later rating on standardised early help and the tracking-period rating",
        },  # fmt: skip
        "mean_drtg_after": float(y.mean()),
        "note": "points allowed per 100 possessions; nbacore full-season play-by-play and box "
        "scores; correlations only",
    }
    out = Path("reports/analysis/team_out_of_sample.json")
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
