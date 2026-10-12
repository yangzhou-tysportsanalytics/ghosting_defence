"""Team-game panel for the help / points-allowed association (strengthens the 30-team result).

Per (defence team, game): early-help rate (third-defender help within 1 s of on-ball screen
contact), help within 3 s, switch rate, and points allowed per possession in that game (nbacore
ledger, possessions without data gaps). Regressions of points allowed on each rate:
  (1) pooled, (2) with defence-team fixed effects (does a team allow fewer points in the games in
  which it helps earlier?), (3) with defence-team and opponent fixed effects.
WLS weighted by possessions; standard errors clustered by defence team (CR1) and a game bootstrap.
Coefficients are reported per 10 percentage points of the rate, in points per 100 possessions.

Descriptive: within a team, game-to-game help and scoring share causes (opponent style, effort,
score state), so the fixed effects remove stable team quality, not all confounding.

Output: reports/analysis/team_game_panel.json

Usage:
    uv run python scripts/analysis_team_game_panel.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D

EARLY_S = 1.0
RATES = ("early_help_rate", "help_rate_3s", "switch_rate")


def panel(cfg: D.DataConfig) -> pl.DataFrame:
    base = cfg.processed_dir / "all"
    S = pl.read_parquet(base / "analysis" / "onball_screens.parquet")
    rates = S.group_by(["game_id", "defense_team_id"]).agg(
        pl.len().alias("n_screens"),
        (pl.col("help_delay_s").fill_null(99) <= EARLY_S).mean().alias("early_help_rate"),
        pl.col("helped_3s").cast(pl.Float64).mean().alias("help_rate_3s"),
        pl.col("switched").cast(pl.Float64).mean().alias("switch_rate"),
    )
    led = D.L.ledger(cfg.version).filter(~pl.col("data_gap"))
    pts = led.group_by(["game_id", "defense_team_id", "offense_team_id"]).agg(
        pl.col("points").sum().alias("points"), pl.len().alias("n_poss")
    )
    return (
        rates.join(pts, on=["game_id", "defense_team_id"], how="inner")
        .with_columns((pl.col("points") / pl.col("n_poss")).alias("pts_per_poss"))
        .sort(["game_id", "defense_team_id"])
    )


def design(df: pl.DataFrame, x: str, fe: tuple[str, ...]) -> np.ndarray:
    cols = [np.ones(df.height), df[x].to_numpy()]
    for f in fe:
        levels = sorted(df[f].unique().to_list())[1:]  # drop one level per factor
        v = df[f].to_numpy()
        cols += [(v == lv).astype(float) for lv in levels]
    return np.column_stack(cols)


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    XtWX_inv = np.linalg.pinv((X * w[:, None]).T @ X)
    beta = XtWX_inv @ (X * w[:, None]).T @ y
    return beta, XtWX_inv


def cluster_se(X, y, w, beta, XtWX_inv, groups) -> float:
    """CR1 standard error of the coefficient on column 1."""
    e = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    ug = np.unique(groups)
    for g in ug:
        i = groups == g
        s = (X[i] * (w[i] * e[i])[:, None]).sum(0)
        meat += np.outer(s, s)
    n, k, G = len(y), X.shape[1], len(ug)
    V = XtWX_inv @ meat @ XtWX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return float(np.sqrt(V[1, 1]))


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    P = panel(cfg)
    y, w = P["pts_per_poss"].to_numpy(), P["n_poss"].to_numpy().astype(float)
    games = P["game_id"].to_numpy()
    ug = np.unique(games)
    rows_of = {g: np.flatnonzero(games == g) for g in ug}
    rng = np.random.default_rng(9)
    boot_idx = [np.concatenate([rows_of[g] for g in rng.choice(ug, ug.size)]) for _ in range(500)]
    rep: dict = {
        "n_team_games": P.height,
        "n_games": int(ug.size),
        "n_teams": P["defense_team_id"].n_unique(),
        "mean_screens_per_team_game": float(P["n_screens"].mean()),
        "units": "points per 100 possessions per 10 percentage points of the rate",
        "models": {},
    }
    for x in RATES:
        res = {}
        for name, fe in (("pooled", ()), ("team_fe", ("defense_team_id",)),
                         ("team_and_opponent_fe", ("defense_team_id", "offense_team_id"))):  # fmt: skip
            X = design(P, x, fe)
            beta, inv = wls(X, y, w)
            se = cluster_se(X, y, w, beta, inv, P["defense_team_id"].to_numpy())
            bs = []
            for idx in boot_idx:
                b, _ = wls(X[idx], y[idx], w[idx])
                bs.append(b[1])
            scale = 100 * 0.10  # per 10 pp, per 100 possessions
            res[name] = {
                "coef": float(beta[1] * scale),
                "se_cluster_team": se * scale,
                "ci95_cluster": [
                    float((beta[1] - 1.96 * se) * scale),
                    float((beta[1] + 1.96 * se) * scale),
                ],  # fmt: skip
                "ci95_game_bootstrap": [
                    float(np.quantile(bs, 0.025) * scale),
                    float(np.quantile(bs, 0.975) * scale),
                ],  # fmt: skip
            }
        res["rate_sd_within_team"] = float(
            P.with_columns((pl.col(x) - pl.col(x).mean().over("defense_team_id")).alias("d"))[
                "d"
            ].std()
        )
        rep["models"][x] = res
    out = Path("reports/analysis/team_game_panel.json")
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
