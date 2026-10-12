"""Is sag a defender's style or the shooting skill of the man he is assigned to?

A defender who is usually assigned to poor shooters may sag more by design (leave the
non-shooter, help elsewhere). This script adds the guarded attacker's three-point threat to the
context adjustment of the deviation metrics and checks whether the player findings survive.

Shooting profile of every attacker (all field-goal attempts of the season):
  three_pct_shrunk  empirical-Bayes (beta-binomial, method of moments over players with >= 50
                    attempts) three-point percentage
  three_rate        three-point attempts per 100 offensive possessions on court
Per (possession, defender): the step-weighted mean of the guarded attacker's profile over the
steps with a man (matchup HMM, state 0-4).

Then, for sag and |dev|, with the base and the main (phases) context: context R², split-half
reliability (adjusted, within team, within listed position), team share of systematic variance,
the correlation of player means with and without the skill terms, and the skill coefficients.

Outputs: <processed_dir>/all/analysis/guarded_skill.parquet;
         reports/phase4/<version>_all/opponent_skill.json

Usage:
    uv run python scripts/phase4_opponent_skill.py
"""

from __future__ import annotations

import json
import sys
import time
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

SKILL = ["g_three_pct", "g_three_rate", "g_three_pct_x_rate"]
MIN_3PA_PRIOR = 50


def shooting_profiles(shots: pl.DataFrame, poss: pl.DataFrame) -> pl.DataFrame:
    """Per attacker: shrunk three-point percentage and three-point attempts per 100 possessions."""
    t = (
        shots.filter(pl.col("is_three"))
        .group_by("shooter_id")
        .agg(pl.len().alias("n3"), pl.col("made").sum().alias("m3"))
    )
    big = t.filter(pl.col("n3") >= MIN_3PA_PRIOR)
    p = (big["m3"] / big["n3"]).to_numpy()
    mean, var = float(p.mean()), float(p.var())
    within = float((p * (1 - p) / big["n3"].to_numpy()).mean())  # binomial noise
    tau2 = max(var - within, 1e-6)
    k = mean * (1 - mean) / tau2 - 1  # prior strength (alpha + beta)
    a, b = mean * k, (1 - mean) * k
    on = (
        poss.select(pl.col("offense_player_ids").alias("pid")).explode("pid")
        .group_by("pid").len().rename({"pid": "shooter_id", "len": "n_poss"})
    )  # fmt: skip
    prof = on.join(t, on="shooter_id", how="left").with_columns(
        pl.col("n3").fill_null(0), pl.col("m3").fill_null(0)
    )
    return prof.with_columns(
        ((pl.col("m3") + a) / (pl.col("n3") + a + b)).alias("three_pct_shrunk"),
        (100 * pl.col("n3") / pl.col("n_poss")).alias("three_rate"),
    ).select(["shooter_id", "three_pct_shrunk", "three_rate", "n3", "n_poss"]), {
        "prior_mean": mean, "prior_strength": k, "n_players_prior": big.height
    }  # fmt: skip


def guarded_skill(base: Path, games: list[str], prof: pl.DataFrame) -> pl.DataFrame:
    rows, t0 = [], time.time()
    pr = prof.select(
        pl.col("shooter_id").alias("off_id"),
        pl.col("three_pct_shrunk").alias("p3"),
        pl.col("three_rate").alias("r3"),
    )
    for gi, gid in enumerate(games, 1):
        M = pl.read_parquet(base / "matchups" / "hmm_strat_help" / f"{gid}.parquet").filter(
            pl.col("state") < 5
        )
        g = (
            M.join(pr, on="off_id", how="left")
            .group_by(["game_id", "possession_id", "def_id"])
            .agg(
                pl.col("p3").mean().alias("g_three_pct"),
                pl.col("r3").mean().alias("g_three_rate"),
                (pl.col("p3") * pl.col("r3")).mean().alias("g_three_pct_x_rate"),
            )
        )
        rows.append(g)
        if gi % 100 == 0 or gi == len(games):
            print(f"[{gi}/{len(games)}] {time.time() - t0:.0f}s", flush=True)
    return pl.concat(rows)


def ols_coefs(df: pl.DataFrame, y: str, extra: tuple[str, ...]) -> dict:
    """Coefficients (per SD of each skill term) of the full context regression."""
    from phase4_reliability import COVS

    cols = [np.ones(df.height)] + [df[c].to_numpy() for c in COVS]
    cols += [df["man_to_hoop_ft"].to_numpy() ** 2, df["man_to_ball_ft"].to_numpy() ** 2]
    cols += [(df["n_men"].to_numpy() > 1).astype(float), np.log(df["n_steps"].to_numpy())]
    cols += [df[c].to_numpy() for c in extra]
    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X, df[y].to_numpy(), rcond=None)[0]
    k0 = X.shape[1] - len(extra)
    return {c: float(beta[k0 + j] * df[c].std()) for j, c in enumerate(extra)}


def player_level(df: pl.DataFrame, y: str, players: list[int], pos: dict) -> dict:
    """Player means of the main-specification adjusted metric (phases, no skill terms) against
    the mean skill of the men they guarded: R² pooled and within listed position."""
    from phase4_reliability import _center

    r, _ = adjust(df, y, tuple(PHASE_COVS))
    d = df.with_columns(pl.Series("adj", r)).filter(
        pl.col("def_id").is_in(players) & pl.col("def_id").is_in(list(pos))
    )
    pm = d.group_by("def_id").agg(
        pl.col("adj").mean(), pl.col("g_three_pct").mean(), pl.col("g_three_rate").mean()
    )
    grp = [pos[p] for p in pm["def_id"].to_list()]
    sk = [pm["g_three_pct"].to_numpy(), pm["g_three_rate"].to_numpy()]

    def r2(v, cols):
        X = np.column_stack([np.ones(len(v))] + cols)
        f = X @ np.linalg.lstsq(X, v, rcond=None)[0]
        return float(1 - ((v - f) ** 2).sum() / ((v - v.mean()) ** 2).sum())

    a = pm["adj"].to_numpy()
    ac, skc = _center(a, grp), [_center(c, grp) for c in sk]
    return {
        "n_players": pm.height,
        "r2_pooled": r2(a, sk),
        "r2_within_position": r2(ac, skc),
        "corr_within_position_guarded_three_rate": float(np.corrcoef(ac, skc[1])[0, 1]),
        "corr_within_position_guarded_three_pct": float(np.corrcoef(ac, skc[0])[0, 1]),
    }


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    shots = pl.read_parquet(base / "analysis" / "shots.parquet")
    poss = pl.read_parquet(base / "possessions.parquet")
    prof, prior = shooting_profiles(shots, poss)
    games = sorted(poss.filter(~pl.col("is_transition"))["game_id"].unique().to_list())
    G = guarded_skill(base, games, prof)
    G.write_parquet(base / "analysis" / "guarded_skill.parquet")

    dev = pl.read_parquet(base / "analysis" / "rule_ghost_dev.parquet")
    df = dev.join(G, on=["game_id", "possession_id", "def_id"], how="inner").drop_nulls(SKILL)
    df = with_phases(df, base)
    players = df.group_by("def_id").len().filter(pl.col("len") >= MIN_POSS)["def_id"].to_list()
    pos = listed_positions(cfg)
    rep: dict = {"shooting_prior": prior, "n_rows": df.height, "n_rows_rule_ghost_dev": dev.height,
                 "n_players": len(players), "metrics": {}}  # fmt: skip
    for y in ("sag_ft", "dev_ft"):
        out = {}
        for ctx, phase in (("base", ()), ("phases", tuple(PHASE_COVS))):
            res = {}
            for lab, extra in (("without_skill", phase), ("with_skill", (*phase, *SKILL))):
                r, r2 = adjust(df, y, extra)
                d = df.with_columns(pl.Series("adj", r))
                tm = d.group_by("defense_team_id").agg(pl.col("adj").mean().alias("tm"))
                dw = d.join(tm, on="defense_team_id").with_columns(
                    (pl.col("adj") - pl.col("tm")).alias("adj_w")
                )
                res[lab] = {
                    "context_r2": r2,
                    "split_half_r_adjusted": split_half(d, "adj", players)[0],
                    "split_half_r_player_within_team": split_half(dw, "adj_w", players)[0],
                    "split_half_r_within_position": within_position(d, "adj", players, pos)[
                        "split_half_r_within_position"
                    ],
                    "team_share_of_systematic": variance_shares(d, "adj", players)[
                        "share_team_of_systematic"
                    ],
                    "_player_means": d.filter(pl.col("def_id").is_in(players))
                    .group_by("def_id").agg(pl.col("adj").mean()),
                }  # fmt: skip
            a = res["without_skill"].pop("_player_means")
            b = res["with_skill"].pop("_player_means")
            j = a.join(b, on="def_id", suffix="_s")
            res["corr_player_means_without_vs_with_skill"] = float(
                np.corrcoef(j["adj"], j["adj_s"])[0, 1]
            )
            res["skill_coefficients_per_sd_ft"] = ols_coefs(df, y, (*phase, *SKILL))
            out[ctx] = res
        # how much of the between-player spread in sag is the skill of the men guarded
        pm = (
            df.filter(pl.col("def_id").is_in(players))
            .group_by("def_id")
            .agg(pl.col(y).mean(), *[pl.col(c).mean() for c in SKILL])
        )
        X = np.column_stack([np.ones(pm.height)] + [pm[c].to_numpy() for c in SKILL])
        yy = pm[y].to_numpy()
        fit = X @ np.linalg.lstsq(X, yy, rcond=None)[0]
        out["share_of_player_mean_variance_explained_by_guarded_skill"] = float(
            1 - ((yy - fit) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        )
        out["player_level_main_spec"] = player_level(df, y, players, pos)
        rep["metrics"][y] = out
    p = Path("reports/phase4") / f"{cfg.version}_all" / "opponent_skill.json"
    p.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
