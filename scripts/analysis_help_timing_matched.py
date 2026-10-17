"""Early help and the possession outcome, adjusted for the state at screen contact (Phase 5 task 3b).

The naive contrast is biased: help is triggered by threat. Here the comparison conditions on the
state of the play at the moment of on-ball screen contact (nothing after the contact is used):
handler distance to the rim, speed and velocity toward the rim, lateral position; screen
location; screener distance to the rim and velocity toward it (roll); distances of the screened
defender and of the screener's defender to the handler; the nearest third defender's distance to
the handler and to the rim; attackers beyond the arc; the handler's three-point profile (shrunk
3P%, 3PA rate); shot clock, game clock, period, score margin; offence and defence team indicators.

Treatment: help by a third defender within 1 s of contact (early help).
Comparison A: early help vs everything else (later help or none within 3 s).
Comparison B: early help vs help between 1 and 3 s (help happened; only timing differs).
Outcomes: points on the possession; xFG of the possession's shot (shot possessions).
Estimators: overlap weights (ATO, main: exact covariate balance by construction) and a doubly
robust ATT on the common support (propensity 0.05-0.95). Propensity: logistic regression on
standardised covariates, their decile indicators and team indicators; outcome model OLS.
Balance: standardised mean differences before / after weighting. 95 % intervals: game bootstrap.

Observational: unmeasured threat (e.g. a defender beaten a step before contact, play calls)
can remain. Reported as an adjusted association.

Output: reports/analysis/help_timing_matched.json

Usage:
    uv run python scripts/analysis_help_timing_matched.py [--boot 100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.court import HOOP_LEFT
from ghost.tensors import fill_shot_clock, load_game_set
from ghost.xfg.model import Logistic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_opponent_skill import shooting_profiles  # noqa: E402

NUM = ["h_dist_rim", "h_speed", "h_v_to_rim", "h_lateral", "scr_dist_rim", "scr_v_to_rim",
       "d_screened_h", "d_screener_def_h", "d_third_h", "d_third_rim", "n_beyond_arc",
       "h_three_pct", "h_three_rate", "shot_clock", "shot_clock_missing", "game_clock",
       "margin", "screen_dist_rim", "screen_lateral"]  # fmt: skip


def state_features(base: Path, S: pl.DataFrame, P: pl.DataFrame, cfg) -> pl.DataFrame:
    rows, t0 = [], time.time()
    games = sorted(S["game_id"].unique().to_list())
    for gi, gid in enumerate(games, 1):
        Sg = S.filter(pl.col("game_id") == gid)
        Pg = P.filter(pl.col("game_id") == gid).join(Sg.select("possession_id").unique(),
                                                      on="possession_id", how="semi")  # fmt: skip
        if Pg.height == 0:
            continue
        pt = load_game_set(base / "frames", [gid], Pg)
        fill_shot_clock(pt, {gid: D.shot_clock(gid, cfg)})
        key = {int(p): i for i, p in enumerate(pt.possession_id)}
        for r in Sg.iter_rows(named=True):
            i = key.get(int(r["possession_id"]))
            if i is None:
                continue
            k = int(round((r["t_contact_ms"] - pt.t[i, 0]) / 200))
            if not (0 <= k < pt.valid.shape[1]) or not pt.valid[i, k]:
                continue
            h = int(pt.handler[i, k])
            if h < 0:
                continue
            ids = [int(x) for x in pt.def_ids[i]]
            if r["screened_def_id"] not in ids or r["screener_def_id"] not in ids:
                continue
            ds, dsc = ids.index(r["screened_def_id"]), ids.index(r["screener_def_id"])
            off, dfn = pt.off_xy[i, k].astype(float), pt.def_xy[i, k].astype(float)
            hv = pt.off_v[i, k, h].astype(float)
            to_rim = HOOP_LEFT - off[h]
            dr = max(float(np.linalg.norm(to_rim)), 1e-6)
            # screener: the attacker (not the handler) nearest the screened defender
            cand = [j for j in range(5) if j != h]
            sc = cand[int(np.argmin([np.linalg.norm(off[j] - dfn[ds]) for j in cand]))]
            sv = pt.off_v[i, k, sc].astype(float)
            to_rim_s = HOOP_LEFT - off[sc]
            ds_r = max(float(np.linalg.norm(to_rim_s)), 1e-6)
            third = [j for j in range(5) if j not in (ds, dsc)]
            dh = [float(np.linalg.norm(dfn[j] - off[h])) for j in third]
            jt = third[int(np.argmin(dh))]
            beyond = np.linalg.norm(off - HOOP_LEFT, axis=1) > 23.0
            sclk = float(pt.shot_clock[i, k])
            rows.append({
                "game_id": gid, "possession_id": int(r["possession_id"]),
                "screen_group": r["screen_group"], "handler_id": int(pt.off_ids[i, h]),
                "h_dist_rim": dr, "h_speed": float(np.linalg.norm(hv)),
                "h_v_to_rim": float(hv @ to_rim / dr), "h_lateral": abs(off[h, 1] - 25.0),
                "scr_dist_rim": ds_r, "scr_v_to_rim": float(sv @ to_rim_s / ds_r),
                "d_screened_h": float(np.linalg.norm(dfn[ds] - off[h])),
                "d_screener_def_h": float(np.linalg.norm(dfn[dsc] - off[h])),
                "d_third_h": min(dh), "d_third_rim": float(np.linalg.norm(dfn[jt] - HOOP_LEFT)),
                "n_beyond_arc": float(beyond.sum()),
                "shot_clock": 0.0 if np.isnan(sclk) else sclk,
                "shot_clock_missing": float(np.isnan(sclk)),
                "game_clock": float(pt.game_clock[i, k]),
            })  # fmt: skip
        if gi % 100 == 0 or gi == len(games):
            print(f"[{gi}/{len(games)}] {time.time() - t0:.0f}s rows={len(rows)}", flush=True)
    return pl.DataFrame(rows)


TRIM = (0.05, 0.95)  # secondary ATT: common support


def design(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Propensity / outcome design: each numeric covariate standardised and as decile
    indicators (non-linear threat), plus offence and defence team indicators.
    Returns (design, standardised linear covariates for the balance check)."""
    lin = np.column_stack([df[c].to_numpy().astype(float) for c in NUM])
    lin = (lin - lin.mean(0)) / (lin.std(0) + 1e-9)
    cols = [lin]
    for j in range(lin.shape[1]):
        edges = np.unique(np.quantile(lin[:, j], np.linspace(0.1, 0.9, 9)))
        b = np.searchsorted(edges, lin[:, j], side="right")
        cols.append(np.column_stack([(b == k).astype(float) for k in range(1, len(edges) + 1)]))
    for f in ("offense_team_id", "defense_team_id"):
        v = df[f].to_numpy()
        cols.append(
            np.column_stack([(v == lv).astype(float) for lv in sorted(set(v.tolist()))[1:]])
        )
    return np.column_stack(cols), lin


def estimates(X: np.ndarray, t: np.ndarray, y: np.ndarray) -> dict:
    """Overlap-weighted effect (ATO; exact mean balance on the propensity terms by construction)
    and a doubly robust ATT on the trimmed common support."""
    e = Logistic(C=10.0, tol=1e-8, max_iter=300).fit(X, t).predict_proba(X)[:, 1]
    w1, w0 = (1 - e) * t, e * (1 - t)
    ato = float((w1 @ y) / w1.sum() - (w0 @ y) / w0.sum())
    keep = (e >= TRIM[0]) & (e <= TRIM[1])
    Xk, tk, yk, ek = X[keep], t[keep], y[keep], e[keep]
    Xc = np.column_stack([np.ones(keep.sum()), Xk])
    m0 = Xc @ np.linalg.lstsq(Xc[tk == 0], yk[tk == 0], rcond=None)[0]
    wk = ek / (1 - ek)
    att = float(np.mean(tk * (yk - m0) - (1 - tk) * wk * (yk - m0)) / tk.mean())
    return {"ato": ato, "att_trimmed": att, "share_kept_trim": float(keep.mean()), "e": e}


def smd(X: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Standardised mean differences (treated − control), optionally weighted in both arms."""
    w = np.ones(len(t)) if w is None else w
    m1 = np.average(X[t == 1], axis=0, weights=w[t == 1])
    m0 = np.average(X[t == 0], axis=0, weights=w[t == 0])
    s = np.sqrt((X[t == 1].var(0) + X[t == 0].var(0)) / 2) + 1e-9
    return (m1 - m0) / s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=100)
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    S = pl.read_parquet(base / "analysis" / "onball_screens.parquet")
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    cache = base / "analysis" / "screen_state.parquet"  # state at contact (no coordinates)
    if cache.exists():
        st = pl.read_parquet(cache)
    else:
        st = state_features(base, S, P, cfg)
        st.write_parquet(cache)
    prof, _ = shooting_profiles(pl.read_parquet(base / "analysis" / "shots.parquet"),
                                pl.read_parquet(base / "possessions.parquet"))  # fmt: skip
    led = D.L.ledger(cfg.version).select(["poss_uid", "points"])
    xfg = (
        pl.read_parquet(base / "analysis" / "ghost_points.parquet")
        .select(["game_id", "possession_id", "xfg_real"])
        .unique(subset=["game_id", "possession_id"], keep="first")
    )
    from ghost.court import LENGTH, WIDTH

    df = (
        S.join(st, on=["game_id", "possession_id", "screen_group"], how="inner")
        .join(
            P.select(
                ["game_id", "possession_id", "poss_uid", "score_margin_offense", "offense_team_id"]
            ),
            on=["game_id", "possession_id"],
            how="left",
        )  # fmt: skip
        .join(led, on="poss_uid", how="left")
        .join(xfg, on=["game_id", "possession_id"], how="left")
        .join(
            prof.select(
                pl.col("shooter_id").alias("handler_id"),
                pl.col("three_pct_shrunk").alias("h_three_pct"),
                pl.col("three_rate").alias("h_three_rate"),
            ),
            on="handler_id",
            how="left",
        )  # fmt: skip
        .filter(pl.col("points").is_not_null())
    )
    xl = np.where(df["attacks_left"].to_numpy(), df["x"].to_numpy(), LENGTH - df["x"].to_numpy())
    yl = np.where(df["attacks_left"].to_numpy(), df["y"].to_numpy(), WIDTH - df["y"].to_numpy())
    df = df.with_columns(
        pl.Series("screen_dist_rim", np.hypot(xl - HOOP_LEFT[0], yl - HOOP_LEFT[1])),
        pl.Series("screen_lateral", np.abs(yl - 25.0)),
        pl.col("score_margin_offense").fill_null(0).cast(pl.Float64).alias("margin"),
        pl.col("h_three_pct").fill_null(pl.col("h_three_pct").mean()),
        pl.col("h_three_rate").fill_null(0.0),
        (pl.col("help_delay_s").fill_null(99) <= 1.0).cast(pl.Int8).alias("early"),
        ((pl.col("help_delay_s").fill_null(99) > 1.0) & pl.col("helped_3s")).cast(pl.Int8)
        .alias("late"),
    ).sort(["game_id", "screen_group"])  # fmt: skip
    rng = np.random.default_rng(9)
    rep: dict = {"n_screens": df.height, "share_early": float(df["early"].mean()),
                 "covariates": NUM,
                 "indicators": ["offense_team_id", "defense_team_id"],
                 "propensity_terms": "standardised covariates, their decile indicators, team indicators",
                 "comparisons": {}}  # fmt: skip
    for comp, sub in (("A_early_vs_rest", df),
                      ("B_early_vs_late_help", df.filter((pl.col("early") == 1) | (pl.col("late") == 1)))):  # fmt: skip
        out = {"n": sub.height, "n_treated": int(sub["early"].sum())}
        for yname, d in (
            ("points", sub),
            ("xfg_of_shot", sub.filter(pl.col("xfg_real").is_not_null())),
        ):
            Xd, lin = design(d)
            td = d["early"].to_numpy().astype(float)
            yd = (d["points"] if yname == "points" else d["xfg_real"]).to_numpy().astype(float)
            est = estimates(Xd, td, yd)
            e = est.pop("e")
            naive = float(yd[td == 1].mean() - yd[td == 0].mean())
            games = d["game_id"].to_numpy()
            ug = np.unique(games)
            rows_of = {g: np.flatnonzero(games == g) for g in ug}
            bs = []
            for _ in range(args.boot):
                ii = np.concatenate([rows_of[g] for g in rng.choice(ug, ug.size)])
                b = estimates(Xd[ii], td[ii], yd[ii])
                bs.append((b["ato"], b["att_trimmed"]))
            bs = np.array(bs)
            w_ato = np.where(td == 1, 1 - e, e)
            q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]  # noqa: E731
            out[yname] = {
                "n": d.height,
                "naive_difference": naive,
                "ato_overlap_weighted": est["ato"],
                "ato_ci95_game_bootstrap": q(bs[:, 0]),
                "att_trimmed_aipw": est["att_trimmed"],
                "att_ci95_game_bootstrap": q(bs[:, 1]),
                "share_kept_by_trimming": est["share_kept_trim"],
                "propensity_quantiles_treated": [float(x) for x in np.quantile(e[td == 1], [0.05, 0.5, 0.95])],
                "propensity_quantiles_control": [float(x) for x in np.quantile(e[td == 0], [0.05, 0.5, 0.95])],
                "max_abs_smd_before": float(np.abs(smd(lin, td)).max()),
                "max_abs_smd_overlap_weighted": float(np.abs(smd(lin, td, w_ato)).max()),
                "smd_before": dict(zip(NUM, np.round(smd(lin, td), 3).tolist(), strict=True)),
                "smd_overlap_weighted": dict(zip(NUM, np.round(smd(lin, td, w_ato), 3).tolist(), strict=True)),
            }  # fmt: skip
        rep["comparisons"][comp] = out
    p = Path("reports/analysis/help_timing_matched.json")
    p.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "covariates"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
