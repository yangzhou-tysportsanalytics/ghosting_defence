"""Phase 5 task 1: predictive validity of the rule-ghost deviation.

Possession level: for half-court possessions ending in a field-goal attempt, the shooter's primary
defender over the 2 s before the release (the defender whose stable matchup is the shooter for
most of those 10 steps, at least 5): mean sag and |deviation| relative to the rule ghost (same
definitions as phase4_rule_ghost.py). Outcomes at the release: nearest-defender distance and the
cross-fitted league-level xFG (M2, each fold predicted by a model fitted without it). OLS with
zone dummies, distance and distance^2; 95 % intervals from a game bootstrap.

Player level, out of sample: context-adjusted player mean sag on the games of one parity (>= 150
defensive possessions) against the player's mean location-adjusted openness / xFG allowed as
primary defender on the other parity (>= 30 shots); Spearman, player-bootstrap intervals; both
directions.

Output: reports/phase5/<version>_all/predictive_validity{,_phases}.json. ``--context phases``
adds the D-020 phase shares to the context adjustment of the player trait (D-021 main
specification); the possession-level regressions do not use the adjustment and are unchanged.

Usage:
    uv run python scripts/phase5_predictive_validity.py [--context base|phases]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ghost import data as D
from ghost.court import HOOP_LEFT
from ghost.matchup.events import EventConfig, stable_path
from ghost.tensors import load_game_set
from ghost.xfg import api

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_reliability import (  # noqa: E402
    PHASE_COVS,
    _center,
    adjust,
    listed_positions,
    with_phases,
)

PRE_STEPS = 10  # 2 s at 5 Hz
MIN_PRE = 5
MIN_POSS_TRAIT = 150
MIN_SHOTS_OUTCOME = 30
ZONES = ["paint", "mid", "corner3", "above3"]  # rim = reference


def cross_fit(shots: pl.DataFrame, version: str) -> np.ndarray:
    games = D.games(D.DataConfig.load(game_set="all")).select(["game_id", "fold"])
    S = shots.join(games, on="game_id", how="left")
    out = np.full(S.height, np.nan)
    for k in sorted(S["fold"].drop_nulls().unique().to_list()):
        other = S.filter(pl.col("fold").is_not_null() & (pl.col("fold") != k))
        m = api.fit(other, other["game_id"].unique().to_list(), version)
        idx = (S["fold"] == k).fill_null(False).to_numpy()
        out[idx] = api.predict(S.filter(pl.col("fold") == k), m)
    return out


def design(df: pl.DataFrame, x_cols: list[str]) -> np.ndarray:
    z = df["zone"].to_numpy()
    d = df["dist_ft"].to_numpy()
    cols = [np.ones(df.height)] + [df[c].to_numpy() for c in x_cols]
    cols += [(z == q).astype(float) for q in ZONES] + [d, d**2 / 100]
    return np.column_stack(cols)


def ols_boot(df: pl.DataFrame, y: str, x_cols: list[str], reps: int, rng) -> dict:
    X, Y = design(df, x_cols), df[y].to_numpy()
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    X0 = design(df, [])
    r2 = 1 - np.var(Y - X @ beta) / np.var(Y)
    r2_0 = 1 - np.var(Y - X0 @ np.linalg.lstsq(X0, Y, rcond=None)[0]) / np.var(Y)
    gids = df["game_id"].to_numpy()
    ug = np.unique(gids)
    rows_of = {g: np.flatnonzero(gids == g) for g in ug}
    bs = []
    for _ in range(reps):
        idx = np.concatenate([rows_of[g] for g in rng.choice(ug, ug.size)])
        bs.append(np.linalg.lstsq(X[idx], Y[idx], rcond=None)[0][1 : 1 + len(x_cols)])
    bs = np.array(bs)
    sd = {c: float(df[c].std()) for c in x_cols}
    return {
        c: {
            "coef": float(beta[1 + j]),
            "ci95": [float(np.quantile(bs[:, j], 0.025)), float(np.quantile(bs[:, j], 0.975))],
            "per_sd": float(beta[1 + j] * sd[c]),
        }
        for j, c in enumerate(x_cols)
    } | {"r2_location_only": float(r2_0), "r2_with_predictors": float(r2), "n": df.height}


def spearman_boot(a: np.ndarray, b: np.ndarray, reps: int, rng) -> dict:
    rho = stats.spearmanr(a, b)[0]
    bs = [
        stats.spearmanr(a[i], b[i])[0]
        for i in (rng.integers(0, a.size, a.size) for _ in range(reps))
    ]
    return {
        "rho": float(rho),
        "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        "n_players": int(a.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="base", choices=["base", "phases"], help="D-021")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    prm = json.loads((base / "matchups" / "hmm_strat_help_params.json").read_text())
    g_strong, g_weak = np.array(prm["gamma"]["strong"]), np.array(prm["gamma"]["weak"])
    hold = EventConfig().steps(EventConfig().min_hold_s)
    rng = np.random.default_rng(9)

    shots_all = pl.read_parquet(base / "analysis" / "shots.parquet")
    shots_all = shots_all.with_columns(
        pl.Series("xfg_m2_cf", cross_fit(shots_all, "M2")),
        pl.Series("xfg_m3_cf", cross_fit(shots_all, "M3")),
    )
    S = shots_all.filter(
        pl.col("possession_id").is_not_null()
        & ~pl.col("window_is_transition").fill_null(True)
        & pl.col("release_method").is_in(api.TRUSTED_METHODS)
        & pl.col("def1_ft").is_not_null()
    )
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    rows, t0 = [], time.time()
    games = sorted(S["game_id"].unique().to_list())
    for gi, gid in enumerate(games, 1):
        Sg = S.filter(pl.col("game_id") == gid)
        Pg = P.filter(pl.col("game_id") == gid).join(
            Sg.select("possession_id"), on="possession_id", how="semi"
        )
        if Pg.height == 0:
            continue
        pt = load_game_set(base / "frames", [gid], Pg)
        M = pl.read_parquet(base / "matchups" / "hmm_strat_help" / f"{gid}.parquet")
        key = {int(p): i for i, p in enumerate(pt.possession_id)}
        M = M.filter(pl.col("possession_id").is_in(list(key)))
        n, T = pt.valid.shape
        path = np.full((n, 5, T), -1, dtype=np.int8)
        ri = np.array([key[int(p)] for p in M["possession_id"]])
        path[ri, M["def_slot"].to_numpy(), M["step"].to_numpy()] = M["state"].to_numpy()
        for r in Sg.iter_rows(named=True):
            i = key.get(int(r["possession_id"]))
            if i is None:
                continue
            valid = np.flatnonzero(pt.valid[i])
            rel = int(valid[np.argmin(np.abs(pt.t[i, valid] - r["t_release_ms"]))])
            pre = np.arange(max(0, rel - PRE_STEPS), rel)
            ks = np.flatnonzero(pt.off_ids[i] == r["shooter_id"])
            if ks.size == 0 or pre.size < MIN_PRE:
                continue
            k = int(ks[0])
            s = stable_path(path[i], pt.valid[i], hold)
            cnt = (s[:, pre] == k).sum(1)
            d = int(np.argmax(cnt))
            if cnt[d] < MIN_PRE:
                continue
            tt = pre[s[d, pre] == k]
            om, B = pt.off_xy[i, tt, k], pt.ball[i, tt, :2]
            strong = np.sign(om[:, 1] - 25.0) == np.sign(B[:, 1] - 25.0)
            g = np.where(strong[:, None], g_strong, g_weak)
            mu = g[:, 0:1] * om + g[:, 1:2] * B + g[:, 2:3] * HOOP_LEFT
            dv = pt.def_xy[i, tt, d] - mu
            u = HOOP_LEFT - om
            u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
            rows.append(
                {
                    "game_id": gid,
                    "possession_id": int(r["possession_id"]),
                    "event_uid": r["event_uid"],
                    "shooter_id": r["shooter_id"],
                    "primary_def_id": int(pt.def_ids[i, d]),
                    "pre_steps": int(tt.size),
                    "pre_sag_ft": float((dv * u).sum(1).mean()),
                    "pre_dev_ft": float(np.linalg.norm(dv, axis=1).mean()),
                    "def1_ft": r["def1_ft"],
                    "xfg_m2_cf": r["xfg_m2_cf"],
                    "xfg_m3_cf": r["xfg_m3_cf"],
                    "made": bool(r["made"]),
                    "zone": r["zone"],
                    "dist_ft": r["dist_ft"],
                    "parity": r["parity"],
                }
            )
        if gi % 100 == 0 or gi == len(games):
            print(f"[{gi}/{len(games)}] {time.time() - t0:.0f}s rows={len(rows)}", flush=True)
    V = pl.DataFrame(rows)

    rep: dict = {"n_shot_possessions": V.height, "possession_level": {}}
    outcomes = ("def1_ft", "xfg_m2_cf", "xfg_m3_cf")
    for y in outcomes:
        rep["possession_level"][y] = ols_boot(V, y, ["pre_sag_ft", "pre_dev_ft"], 500, rng)

    # player level, out of sample across game parity; outcomes adjusted for shot location
    X0 = design(V, [])
    for y in outcomes:
        Y = V[y].to_numpy()
        V = V.with_columns(pl.Series(f"{y}_resid", Y - X0 @ np.linalg.lstsq(X0, Y, rcond=None)[0]))
    pos = listed_positions(cfg)
    dev = pl.read_parquet(base / "analysis" / "rule_ghost_dev.parquet")
    extra: tuple[str, ...] = ()
    if args.context == "phases":
        dev, extra = with_phases(dev, base), tuple(PHASE_COVS)
    adj, _ = adjust(dev, "sag_ft", extra)
    dev = dev.with_columns(pl.Series("sag_adj", adj))
    rep["player_level"] = {}
    for trait_par, out_par in ((0, 1), (1, 0)):
        tr = (
            dev.filter(pl.col("parity") == trait_par)
            .group_by("def_id")
            .agg(pl.col("sag_adj").mean().alias("trait"), pl.len().alias("n_poss"))
            .filter(pl.col("n_poss") >= MIN_POSS_TRAIT)
        )
        oc = (
            V.filter(pl.col("parity") == out_par)
            .group_by("primary_def_id")
            .agg([pl.col(f"{y}_resid").mean() for y in outcomes] + [pl.len().alias("n_shots")])
            .filter(pl.col("n_shots") >= MIN_SHOTS_OUTCOME)
            .rename({"primary_def_id": "def_id"})
        )
        J = tr.join(oc, on="def_id").filter(pl.col("def_id").is_in(list(pos)))
        grp = [pos[p] for p in J["def_id"].to_list()]
        a = J["trait"].to_numpy()
        res_pl = {}
        for y in outcomes:  # M3 knows the shooter: tests "sagging off poor shooters"
            b = J[f"{y}_resid"].to_numpy()
            res_pl[y] = spearman_boot(a, b, 1000, rng)
            res_pl[y + "__within_position"] = spearman_boot(
                _center(a, grp), _center(b, grp), 1000, rng
            )
        rep["player_level"][f"trait_parity{trait_par}_outcome_parity{out_par}"] = res_pl
    out = Path("reports/phase5") / f"{cfg.version}_all"
    out.mkdir(parents=True, exist_ok=True)
    rep["context"] = args.context
    sfx = "" if args.context == "base" else "_phases"
    (out / f"predictive_validity{sfx}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
