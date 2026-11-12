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
    rates = rates.with_columns(  # help after 1 s but within 3 s (review A2.2)
        (pl.col("help_rate_3s") - pl.col("early_help_rate")).alias("late_help_rate")
    )
    led = D.L.ledger(cfg.version).filter(~pl.col("data_gap"))
    # possessions with at least one on-ball screen, and half-court possessions (review A2.3)
    P = pl.read_parquet(base / "possessions.parquet").select(
        ["poss_uid", "game_id", "possession_id", "is_transition"]
    )
    scr = S.select(["game_id", "possession_id"]).unique().with_columns(pl.lit(True).alias("scr"))
    tag = (
        P.join(scr, on=["game_id", "possession_id"], how="left")
        .with_columns(pl.col("scr").fill_null(False), (~pl.col("is_transition")).alias("hc"))
        .select(["poss_uid", "scr", "hc"])
        .unique(subset=["poss_uid"], keep="first", maintain_order=True)
    )
    led = led.join(tag, on="poss_uid", how="left").with_columns(
        pl.col("scr").fill_null(False), pl.col("hc").fill_null(False)
    )
    subsets = {
        "": pl.col("points").is_not_null(),  # all possessions (a per-row mask, not a literal)
        "_screen": pl.col("scr"),
        "_noscreen_halfcourt": ~pl.col("scr") & pl.col("hc"),
        "_noscreen_all": ~pl.col("scr"),
    }
    pts = led.group_by(["game_id", "defense_team_id", "offense_team_id"]).agg(
        *[
            e
            for s, c in subsets.items()
            for e in (
                pl.col("points").filter(c).sum().alias(f"points{s}"),
                c.sum().alias(f"n_poss{s}"),
            )
        ],  # fmt: skip
    )
    return (
        rates.join(pts, on=["game_id", "defense_team_id"], how="inner")
        .with_columns(
            *[
                (pl.col(f"points{s}") / pl.col(f"n_poss{s}")).alias(f"pts_per_poss{s}")
                for s in subsets
            ]  # fmt: skip
        )
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


def cluster_vcov(X, y, w, beta, XtWX_inv, groups) -> np.ndarray:
    """CR1 covariance matrix for one clustering."""
    e = y - X @ beta
    s = X * (w * e)[:, None]
    ug, inv = np.unique(groups, return_inverse=True)
    S = np.zeros((ug.size, X.shape[1]))
    np.add.at(S, inv, s)
    n, k, G = len(y), X.shape[1], ug.size
    return XtWX_inv @ (S.T @ S) @ XtWX_inv * (G / (G - 1)) * ((n - 1) / (n - k))


def twoway_se(X, y, w, beta, inv, g1, g2) -> float:
    """Cameron-Gelbach-Miller two-way clustered SE of column 1 (g1, g2 and their intersection)."""
    g12 = np.array([f"{a}|{b}" for a, b in zip(g1, g2, strict=True)])
    V = (cluster_vcov(X, y, w, beta, inv, g1) + cluster_vcov(X, y, w, beta, inv, g2)
         - cluster_vcov(X, y, w, beta, inv, g12))  # fmt: skip
    return float(np.sqrt(max(V[1, 1], 0.0)))


WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def wild_cluster_p(X, y, w, groups, rng, B: int = 999) -> float:
    """Wild cluster bootstrap-t p-value for column 1 = 0 with the null imposed (WCR; Webb
    six-point weights, CR1 t statistic); suited to few clusters (Cameron et al. 2008)."""
    beta, inv = wls(X, y, w)
    t0 = beta[1] / np.sqrt(cluster_vcov(X, y, w, beta, inv, groups)[1, 1])
    Xr = np.delete(X, 1, axis=1)
    br, _ = wls(Xr, y, w)
    fit, er = Xr @ br, y - Xr @ br
    ug, gi = np.unique(groups, return_inverse=True)
    hits = 0
    for _ in range(B):
        ys = fit + er * WEBB[rng.integers(0, 6, ug.size)][gi]
        b, _ = wls(X, ys, w)  # same X, so inv is unchanged
        t = b[1] / np.sqrt(cluster_vcov(X, ys, w, b, inv, groups)[1, 1])
        hits += abs(t) >= abs(t0)
    return (hits + 1) / (B + 1)


def if_cluster_sums(X, y, w, beta, inv, groups) -> np.ndarray:
    """Per-cluster sums of the influence function of column 1 (for covariances across models)."""
    e = y - X @ beta
    infl = (X * (w * e)[:, None]) @ inv[:, 1]
    ug, gi = np.unique(groups, return_inverse=True)
    out = np.zeros(ug.size)
    np.add.at(out, gi, infl)
    return out


FE_SPECS = (("pooled", ()), ("team_fe", ("defense_team_id",)),
            ("team_and_opponent_fe", ("defense_team_id", "offense_team_id")))  # fmt: skip


def design_multi(df: pl.DataFrame, xs: list[str], fe: tuple[str, ...]) -> np.ndarray:
    X = design(df, xs[0], fe)
    return np.insert(X, [2] * (len(xs) - 1), df.select(xs[1:]).to_numpy(), axis=1)


def extend(rep: dict, P: pl.DataFrame, y, w, boot_idx) -> None:
    """Review A2 (2026-10-30): crossed-dependence inference, early vs 3 s help, outcome split by
    screen possessions. Separate random streams, so the earlier results are unchanged."""
    rng = np.random.default_rng(10)
    scale = 100 * 0.10
    gd, go = P["defense_team_id"].to_numpy(), P["offense_team_id"].to_numpy()
    ci = lambda b, s: [float((b - 1.96 * s) * scale), float((b + 1.96 * s) * scale)]  # noqa: E731
    # 1. two-way clustering (defence team x opponent) and wild cluster bootstrap (defence team)
    for x in RATES:
        for name, fe in FE_SPECS:
            X = design(P, x, fe)
            beta, inv = wls(X, y, w)
            se2 = twoway_se(X, y, w, beta, inv, gd, go)
            rep["models"][x][name].update({
                "se_twoway_team_opponent": se2 * scale,
                "ci95_twoway_team_opponent": ci(beta[1], se2),
                "p_wild_cluster_team": wild_cluster_p(X, y, w, gd, rng),
                "p_cluster_team_normal": float(2 * (1 - _phi(abs(beta[1]) / np.sqrt(
                    cluster_vcov(X, y, w, beta, inv, gd)[1, 1])))),
            })  # fmt: skip
    # 2. early vs 3 s help: (a) difference of the two separate coefficients; (b) joint model with
    # early (<= 1 s) and late (1-3 s) help rates, difference of their coefficients
    rep["early_vs_3s_help"] = {"note": "(a) b_early - b_3s from the separate regressions "
                               "(same rows), CR1 by defence team via influence functions and the "
                               "same game-bootstrap draws; (b) joint regression on early and late "
                               "(1-3 s) help rates, b_early - b_late"}  # fmt: skip
    G = np.unique(gd).size
    for name, fe in FE_SPECS:
        est, sums, boots = [], [], []
        for x in ("early_help_rate", "help_rate_3s"):
            X = design(P, x, fe)
            beta, inv = wls(X, y, w)
            est.append(beta[1])
            sums.append(if_cluster_sums(X, y, w, beta, inv, gd))
            boots.append([wls(X[i], y[i], w[i])[0][1] for i in boot_idx])
        d = est[0] - est[1]
        se = float(np.sqrt(((sums[0] - sums[1]) ** 2).sum() * G / (G - 1)))
        bd = np.array(boots[0]) - np.array(boots[1])
        X = design_multi(P, ["early_help_rate", "late_help_rate"], fe)
        beta, inv = wls(X, y, w)
        V = cluster_vcov(X, y, w, beta, inv, gd)
        sej = float(np.sqrt(V[1, 1] + V[2, 2] - 2 * V[1, 2]))
        rep["early_vs_3s_help"][name] = {
            "a_difference": float(d * scale), "a_ci95_cluster_team": ci(d, se),
            "a_p_cluster_team": float(2 * (1 - _phi(abs(d) / se))),
            "a_ci95_game_bootstrap": [float(np.quantile(bd, q) * scale) for q in (0.025, 0.975)],
            "b_coef_early": float(beta[1] * scale), "b_coef_late": float(beta[2] * scale),
            "b_difference": float((beta[1] - beta[2]) * scale),
            "b_ci95_cluster_team": ci(beta[1] - beta[2], sej),
            "b_p_cluster_team": float(2 * (1 - _phi(abs(beta[1] - beta[2]) / sej))),
            "b_ci95_early_cluster_team": ci(beta[1], float(np.sqrt(V[1, 1]))),
            "b_ci95_late_cluster_team": ci(beta[2], float(np.sqrt(V[2, 2]))),
        }  # fmt: skip
    # 3. points allowed in possessions with / without an on-ball screen, early-help rate
    rep["by_screen_possessions"] = {"predictor": "early_help_rate",
                                    "units": rep["units"]}  # fmt: skip
    for s in ("_screen", "_noscreen_halfcourt", "_noscreen_all"):
        Q = P.filter(pl.col(f"n_poss{s}") > 0)
        yq, wq = Q[f"pts_per_poss{s}"].to_numpy(), Q[f"n_poss{s}"].to_numpy().astype(float)
        g = Q["game_id"].to_numpy()
        ug = np.unique(g)
        rows = {k: np.flatnonzero(g == k) for k in ug}
        bidx = [np.concatenate([rows[k] for k in rng.choice(ug, ug.size)]) for _ in range(500)]
        qd, qo = Q["defense_team_id"].to_numpy(), Q["offense_team_id"].to_numpy()
        out = {"n_team_games": Q.height, "mean_poss_per_team_game": float(wq.mean())}
        for name, fe in FE_SPECS:
            X = design(Q, "early_help_rate", fe)
            beta, inv = wls(X, yq, wq)
            se = np.sqrt(cluster_vcov(X, yq, wq, beta, inv, qd)[1, 1])
            bs = [wls(X[i], yq[i], wq[i])[0][1] for i in bidx]
            out[name] = {
                "coef": float(beta[1] * scale), "ci95_cluster": ci(beta[1], se),
                "ci95_twoway_team_opponent": ci(beta[1], twoway_se(X, yq, wq, beta, inv, qd, qo)),
                "ci95_game_bootstrap": [float(np.quantile(bs, q) * scale) for q in (0.025, 0.975)],
                "p_wild_cluster_team": wild_cluster_p(X, yq, wq, qd, rng),
            }  # fmt: skip
        rep["by_screen_possessions"][s.lstrip("_")] = out


def _phi(z: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1 + erf(z / sqrt(2)))


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
    extend(rep, P, y, w, boot_idx)
    out = Path("reports/analysis/team_game_panel.json")
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
