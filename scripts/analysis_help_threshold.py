"""Early-help threshold curve, censoring and garbage-time checks (SSAC27 review, 2026-11-02).

The team-game panel (analysis_team_game_panel.py) defines early help as third-defender help
within 1 s of on-ball screen contact. The 1 s threshold was chosen after the first results, so:

(a) Threshold curve: for tau = 0.4 ... 3.0 s, the within-team coefficient of the share of screens
    with help within tau s (points allowed per 100 possessions per 10 pp), with defence-team FE
    and with defence-team + opponent FE; CR1 by defence team and two-way (team x opponent)
    intervals. Also the joint model with help <= tau and help in (tau, 3] s and their difference.
(b) Censoring: a screen with less than 3 s of tracked possession left after contact cannot show
    help within 3 s (or within 1 s if < 1 s is left), so it counts as "no help". Refit with only
    screens followed by >= 3 s (and >= 1 s) of possession.
    Garbage time: refit without fourth-period garbage time (score / clock part of the Cleaning
    the Glass rule: margin >= 25 with 12-9 min left, >= 20 with 9-6, >= 10 with 6 or less), in
    both the screens and the points allowed. Overtime is never garbage time.

Descriptive, like the panel itself.
Output: reports/analysis/help_threshold.json

Usage:
    uv run python scripts/analysis_help_threshold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_team_game_panel as TG  # noqa: E402
from ghost import data as D  # noqa: E402

TAUS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0]
HELP_WINDOW_S = 3.0
SCALE = 100 * 0.10  # per 10 pp, per 100 possessions
FE = {"team_fe": ("defense_team_id",), "team_and_opponent_fe": ("defense_team_id",
      "offense_team_id")}  # fmt: skip


def garbage(period: pl.Expr, clock_s: pl.Expr, margin: pl.Expr) -> pl.Expr:
    m = margin.abs()
    return (period == 4) & (
        ((clock_s > 540) & (m >= 25))
        | ((clock_s > 360) & (clock_s <= 540) & (m >= 20))
        | ((clock_s <= 360) & (m >= 10))
    )


def screens(cfg: D.DataConfig) -> pl.DataFrame:
    base = cfg.processed_dir / "all"
    S = pl.read_parquet(base / "analysis" / "onball_screens.parquet")
    P = pl.read_parquet(base / "possessions.parquet").select(
        [
            "game_id",
            "possession_id",
            "period",
            "t_start",
            "t_end",
            "game_clock_start",
            "score_margin_offense",
        ]  # fmt: skip
    )
    S = S.join(P, on=["game_id", "possession_id"], how="left")
    clock = pl.col("game_clock_start") - (pl.col("t_contact_ms") - pl.col("t_start")) / 1000
    return S.with_columns(
        ((pl.col("t_end") - pl.col("t_contact_ms")) / 1000).alias("left_s"),
        garbage(pl.col("period"), clock, pl.col("score_margin_offense")).alias("garbage"),
    ).sort(["game_id", "t_contact_ms", "screen_group"])


def points(cfg: D.DataConfig, drop_garbage: bool) -> pl.DataFrame:
    led = D.L.ledger(cfg.version).filter(~pl.col("data_gap"))
    if drop_garbage:
        # game clock of a ledger possession: from the nearest tracked possession start in the
        # same game and period (as-of join), advanced by the elapsed wall time (approximate)
        P = (
            pl.read_parquet(cfg.processed_dir / "all" / "possessions.parquet")
            .select(["game_id", "period", "t_start", "game_clock_start"])
            .sort("t_start")
        )
        led = (
            led.sort("t_start_ms")
            .join_asof(
                P,
                left_on="t_start_ms",
                right_on="t_start",
                by=["game_id", "period"],
                strategy="nearest",
            )  # fmt: skip
            .with_columns(
                (pl.col("game_clock_start") - (pl.col("t_start_ms") - pl.col("t_start")) / 1000)
                .clip(0, 720)
                .fill_null(720.0)
                .alias("clock")  # fmt: skip
            )
            .filter(
                ~garbage(pl.col("period"), pl.col("clock"), pl.col("score_margin_offense_start"))
            )
        )
    return (
        led.group_by(["game_id", "defense_team_id", "offense_team_id"])
        .agg(pl.col("points").sum().alias("points"), pl.len().alias("n_poss"))
        .with_columns((pl.col("points") / pl.col("n_poss")).alias("pts_per_poss"))
    )


def rates(S: pl.DataFrame, tau: float) -> pl.DataFrame:
    d = pl.col("help_delay_s").fill_null(99)
    return S.group_by(["game_id", "defense_team_id"]).agg(
        pl.len().alias("n_screens"),
        (d <= tau).mean().alias("early"),
        ((d > tau) & (d <= HELP_WINDOW_S)).mean().alias("late"),
        (d <= HELP_WINDOW_S).mean().alias("total"),
    )


def fit(Pn: pl.DataFrame, xs: list[str], fe: tuple[str, ...]) -> dict:
    Pn = Pn.sort(["game_id", "defense_team_id"])
    y, w = Pn["pts_per_poss"].to_numpy(), Pn["n_poss"].to_numpy().astype(float)
    gd, go = Pn["defense_team_id"].to_numpy(), Pn["offense_team_id"].to_numpy()
    X = TG.design(Pn, xs[0], fe) if len(xs) == 1 else TG.design_multi(Pn, xs, fe)
    beta, inv = TG.wls(X, y, w)
    V = TG.cluster_vcov(X, y, w, beta, inv, gd)
    ci = lambda b, s: [float((b - 1.96 * s) * SCALE), float((b + 1.96 * s) * SCALE)]  # noqa: E731
    out = {"n_team_games": Pn.height}
    for j, x in enumerate(xs, start=1):
        out[x] = {
            "coef": float(beta[j] * SCALE),
            "ci95_cluster_team": ci(beta[j], np.sqrt(V[j, j])),
        }
    if len(xs) == 1:
        out[xs[0]]["ci95_twoway"] = ci(beta[1], TG.twoway_se(X, y, w, beta, inv, gd, go))
    else:
        # two-way (defense, opponent) intervals for each coefficient (SSAC Figure 5b)
        g12 = np.array([f"{a}|{b}" for a, b in zip(gd, go, strict=True)])
        V2 = (TG.cluster_vcov(X, y, w, beta, inv, gd) + TG.cluster_vcov(X, y, w, beta, inv, go)
              - TG.cluster_vcov(X, y, w, beta, inv, g12))  # fmt: skip
        for j, x in enumerate(xs, start=1):
            out[x]["ci95_twoway"] = ci(beta[j], np.sqrt(max(V2[j, j], 0.0)))
        se = float(np.sqrt(V[1, 1] + V[2, 2] - 2 * V[1, 2]))
        # early - late = the early coefficient when the regressors are early and early + late
        Xd = TG.design_multi(Pn, ["early", "total"], fe)
        bd, invd = TG.wls(Xd, y, w)
        out["difference"] = {"coef": float((beta[1] - beta[2]) * SCALE),
                             "ci95_cluster_team": ci(beta[1] - beta[2], se),
                             "ci95_twoway": ci(bd[1], TG.twoway_se(Xd, y, w, bd, invd, gd, go))}  # fmt: skip
    return out


def curve(S: pl.DataFrame, pts: pl.DataFrame, taus=TAUS) -> dict:
    res = {}
    for tau in taus:
        Pn = rates(S, tau).join(pts, on=["game_id", "defense_team_id"], how="inner")
        res[f"{tau:.1f}"] = {
            "share_of_screens": float((S["help_delay_s"].fill_null(99) <= tau).mean()),
            # at tau = 3 s there is no "late" help left, so no joint model
            **{
                name: {
                    "single": fit(Pn, ["early"], fe),
                    "joint": fit(Pn, ["early", "late"], fe) if tau < HELP_WINDOW_S else None,
                }
                for name, fe in FE.items()
            },  # fmt: skip
        }
    return res


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    S = screens(cfg)
    pts = points(cfg, drop_garbage=False)
    rep: dict = {
        "units": "points per 100 possessions per 10 percentage points of the rate",
        "n_screens": S.height,
        "share_screens_left_lt_1s": float((S["left_s"] < 1).mean()),
        "share_screens_left_lt_3s": float((S["left_s"] < 3).mean()),
        "share_screens_garbage": float(S["garbage"].mean()),
        "share_left_lt_1s_by_early_help": {
            "early": float(S.filter(pl.col("help_delay_s") <= 1)["left_s"].lt(1).mean()),
        },
    }
    rep["threshold_curve"] = curve(S, pts)
    rep["censoring_left_ge_3s"] = curve(S.filter(pl.col("left_s") >= 3), pts, [1.0])
    rep["censoring_left_ge_1s"] = curve(S.filter(pl.col("left_s") >= 1), pts, [1.0])
    rep["no_garbage_time"] = curve(S.filter(~pl.col("garbage")), points(cfg, True), [1.0])
    rep["no_garbage_left_ge_3s"] = curve(S.filter(~pl.col("garbage") & (pl.col("left_s") >= 3)),
                                         points(cfg, True), [1.0])  # fmt: skip
    out = Path("reports/analysis/help_threshold.json")
    out.write_text(json.dumps(rep, indent=1))
    for tau, r in rep["threshold_curve"].items():
        a, b = r["team_fe"], r["team_and_opponent_fe"]
        if a["joint"] is None:
            print(
                f"tau {tau}  FE {a['single']['early']['coef']:+.2f}  FE+opp "
                f"{b['single']['early']['coef']:+.2f}"
            )
            continue
        print(f"tau {tau}  share {r['share_of_screens']:.3f}  "
              f"FE {a['single']['early']['coef']:+.2f} {a['single']['early']['ci95_twoway']}  "
              f"FE+opp {b['single']['early']['coef']:+.2f} {b['single']['early']['ci95_twoway']}  "
              f"joint diff {a['joint']['difference']['coef']:+.2f} "
              f"{a['joint']['difference']['ci95_twoway']}  FE+opp diff "
              f"{b['joint']['difference']['coef']:+.2f} {b['joint']['difference']['ci95_twoway']}")  # fmt: skip
    for k in ("censoring_left_ge_3s", "censoring_left_ge_1s", "no_garbage_time",
              "no_garbage_left_ge_3s"):  # fmt: skip
        r = rep[k]["1.0"]
        print(k, json.dumps({n: {"single": r[n]["single"]["early"], "diff": r[n]["joint"]
                                 ["difference"]} for n in FE}))  # fmt: skip
    print({k: v for k, v in rep.items() if k.startswith("share")})


if __name__ == "__main__":
    main()
