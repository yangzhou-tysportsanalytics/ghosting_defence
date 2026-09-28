"""Directional value test (Phase 5 task 3b, D-014): is third-defender help after an on-ball screen
associated with fewer points on the possession?

Unit: on-ball screen group (analysis_team_screen_defense.py) in a half-court possession.
Outcomes: points on the ledger possession segment; xFG of the possession's shot (if any).
Treatment: helped_3s; secondary: early help (<= 1 s).
Controls: screen distance to the rim and lateral position (bins), period, offence team fixed
effects, and (model B) defence team fixed effects, so the comparison is within the same defence.
Uncertainty: game-clustered bootstrap (200 draws).

This is observational: help is triggered by threat (e.g. a turned corner), which biases the naive
contrast toward "help looks bad". Stated as an association.

Output: reports/analysis/help_value.json

Usage:
    uv run python scripts/analysis_help_value.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.court import HOOP_LEFT, LENGTH, WIDTH


def ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def build_X(df: pl.DataFrame, defense_fe: bool) -> tuple[np.ndarray, list[str]]:
    cols = [
        np.ones(df.height),
        df["helped_3s"].to_numpy().astype(float),
        df["early_help"].to_numpy().astype(float),
    ]
    names = ["const", "helped_3s", "early_help_le_1s"]
    for b in (15, 20, 25, 30):
        cols.append((df["screen_dist_ft"].to_numpy() >= b).astype(float))
        names.append(f"dist_ge_{b}")
    cols.append(np.abs(df["screen_y"].to_numpy() - 25) / 25)
    names.append("lateral")
    for p in (2, 3, 4):
        cols.append((df["period"].to_numpy() == p).astype(float))
        names.append(f"period_{p}")
    fes = ["offense_team_id"] + (["defense_team_id"] if defense_fe else [])
    for fe in fes:
        levels = sorted(df[fe].unique().to_list())[1:]
        v = df[fe].to_numpy()
        for lv in levels:
            cols.append((v == lv).astype(float))
            names.append(f"{fe}_{lv}")
    return np.column_stack(cols), names


def fit_with_bootstrap(df: pl.DataFrame, y: str, defense_fe: bool, B: int, rng) -> dict:
    X, names = build_X(df, defense_fe)
    yy = df[y].to_numpy().astype(float)
    beta = ols(yy, X)
    games = df["game_id"].to_numpy()
    ug = np.unique(games)
    idx_by = {g: np.flatnonzero(games == g) for g in ug}
    bs = []
    for _ in range(B):
        pick = rng.choice(ug, size=ug.size, replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        bs.append(ols(yy[ii], X[ii])[1:3])
    bs = np.array(bs)
    return {
        "n": df.height,
        "mean_outcome": float(yy.mean()),
        "helped_3s": {
            "coef": float(beta[1]),
            "ci95": [float(np.quantile(bs[:, 0], q)) for q in (0.025, 0.975)],
        },
        "early_help_le_1s": {
            "coef": float(beta[2]),
            "ci95": [float(np.quantile(bs[:, 1], q)) for q in (0.025, 0.975)],
        },
    }


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    G = pl.read_parquet(base / "analysis" / "onball_screens.parquet")
    P = pl.read_parquet(base / "possessions.parquet").select(
        ["game_id", "possession_id", "period", "poss_uid"]
    )
    led = D.L.ledger(cfg.version).select(["poss_uid", "points"])
    shots = (
        pl.read_parquet(base / "analysis" / "ghost_points.parquet").select(
            ["game_id", "possession_id", "xfg_real"]
        )
        if (base / "analysis" / "ghost_points.parquet").exists()
        else None
    )
    df = G.join(P, on=["game_id", "possession_id"], how="left").join(led, on="poss_uid", how="left")
    if shots is not None:
        df = df.join(shots, on=["game_id", "possession_id"], how="left")
    # screen location in the offence-attacks-left frame
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    left = df["attacks_left"].to_numpy()
    xl = np.where(left, x, LENGTH - x)
    yl = np.where(left, y, WIDTH - y)
    df = df.with_columns(
        pl.Series("screen_dist_ft", np.hypot(xl - HOOP_LEFT[0], yl - HOOP_LEFT[1])),
        pl.Series("screen_y", yl),
        (pl.col("help_delay_s").fill_null(99) <= 1.0).alias("early_help"),
    ).filter(pl.col("points").is_not_null())
    rng = np.random.default_rng(9)
    res = {
        "points_on_possession": {
            "A_offense_fe": fit_with_bootstrap(df, "points", False, 200, rng),
            "B_offense_and_defense_fe": fit_with_bootstrap(df, "points", True, 200, rng),
        },
        "raw_means": {
            "points_helped": float(df.filter(pl.col("helped_3s"))["points"].mean()),
            "points_not_helped": float(df.filter(~pl.col("helped_3s"))["points"].mean()),
        },
        "caveat": "observational; help responds to threat, so estimates are biased toward "
        "'help looks costly'. Help B unvalidated; screen candidates 74 % precise on-ball.",
    }
    if shots is not None:
        ds = df.filter(pl.col("xfg_real").is_not_null())
        res["xfg_of_possession_shot"] = {
            "A_offense_fe": fit_with_bootstrap(ds, "xfg_real", False, 200, rng),
            "B_offense_and_defense_fe": fit_with_bootstrap(ds, "xfg_real", True, 200, rng),
        }
    out = Path("reports/analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "help_value.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
