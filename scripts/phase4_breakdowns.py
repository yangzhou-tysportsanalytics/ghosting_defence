"""Phase 4 task 1: discrete breakdown events on the rule-ghost deviation (Le et al. 2017 80-20 rule).

Per 5 Hz step, a defender with a stable man (help state and steps without a man excluded) has
deviation |d| from the rule ghost (definitions as in phase4_rule_ghost.py). The league 80th
percentile of |d| over all such defender-steps is the threshold. A breakdown is a run of >= 5
consecutive steps (1 s) above it. Breakdowns of one possession form a cascade when one starts no
later than ``gap`` after the previous one of the cascade ends; the earliest breakdown of a cascade
is its initiator. gap = 1 s (reported), 0.5 s and 2 s (sensitivity).

Outputs:
    <processed_dir>/all/analysis/breakdowns.parquet   one row per breakdown (no coordinates)
    reports/phase4/<version>_all/breakdowns.json       threshold, counts, player rates and their
                                                       split-half reliability, possession outcomes
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.court import HOOP_LEFT
from ghost.matchup.events import EventConfig, stable_path
from ghost.tensors import load_game_set

MIN_RUN = 5  # 1 s at 5 Hz
GAPS = {"0.5s": 2, "1s": 5, "2s": 10}
REPORT_GAP = "1s"
MIN_POSS = 300


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of runs of True of length >= MIN_RUN."""
    out, start = [], None
    for t, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = t
        elif not v and start is not None:
            if t - start >= MIN_RUN:
                out.append((start, t - 1))
            start = None
    return out


def cascades(events: list[tuple[int, int]], gap: int) -> tuple[list[int], list[bool]]:
    """Cascade id and initiator flag for (start, end) events of one possession."""
    order = sorted(range(len(events)), key=lambda j: events[j][0])
    cid, init = [0] * len(events), [False] * len(events)
    c, last_end = -1, None
    for j in order:
        s, e = events[j]
        if last_end is None or s > last_end + gap:
            c += 1
            init[j] = True
            last_end = e
        else:
            last_end = max(last_end, e)
        cid[j] = c
    return cid, init


def step_deviation(pt, i: int, s: np.ndarray, g_strong, g_weak) -> np.ndarray:
    """(5, T) |d| for steps with a stable man, NaN elsewhere."""
    T = pt.valid.shape[1]
    out = np.full((5, T), np.nan)
    ball = pt.ball[i, :, :2]
    for d in range(5):
        m = s[d].astype(int)
        man = (m >= 0) & (m < 5) & pt.valid[i]
        tt = np.flatnonzero(man)
        if tt.size == 0:
            continue
        om, B = pt.off_xy[i, tt, m[tt]], ball[tt]
        strong = np.sign(om[:, 1] - 25.0) == np.sign(B[:, 1] - 25.0)
        g = np.where(strong[:, None], g_strong, g_weak)
        mu = g[:, 0:1] * om + g[:, 1:2] * B + g[:, 2:3] * HOOP_LEFT
        out[d, tt] = np.linalg.norm(pt.def_xy[i, tt, d] - mu, axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantile", type=float, default=0.8, help="league threshold (D-015: 0.8)")
    args = ap.parse_args()
    tag = "" if args.quantile == 0.8 else f"_q{round(args.quantile * 100)}"
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    prm = json.loads((base / "matchups" / "hmm_strat_help_params.json").read_text())
    g_strong, g_weak = np.array(prm["gamma"]["strong"]), np.array(prm["gamma"]["weak"])
    hold = EventConfig().steps(EventConfig().min_hold_s)
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    games = sorted(P["game_id"].unique().to_list())

    # pass 1: per-step deviations of every game (kept in memory as per-possession arrays)
    store, all_dev, t0 = {}, [], time.time()
    for gi, gid in enumerate(games, 1):
        Pg = P.filter(pl.col("game_id") == gid)
        pt = load_game_set(base / "frames", [gid], Pg)
        M = pl.read_parquet(base / "matchups" / "hmm_strat_help" / f"{gid}.parquet")
        n, T = pt.valid.shape
        key = {int(p): i for i, p in enumerate(pt.possession_id)}
        path = np.full((n, 5, T), -1, dtype=np.int8)
        ri = np.array([key[int(p)] for p in M["possession_id"]])
        path[ri, M["def_slot"].to_numpy(), M["step"].to_numpy()] = M["state"].to_numpy()
        for i in range(n):
            dev = step_deviation(pt, i, stable_path(path[i], pt.valid[i], hold), g_strong, g_weak)
            store[(gid, int(pt.possession_id[i]))] = (dev.astype(np.float32), pt.def_ids[i].copy())
            all_dev.append(dev[~np.isnan(dev)])
        if gi % 100 == 0 or gi == len(games):
            print(f"[pass 1 {gi}/{len(games)}] {time.time() - t0:.0f}s", flush=True)
    thr = float(np.quantile(np.concatenate(all_dev), args.quantile))
    del all_dev

    # pass 2: events, cascades
    rows = []
    for (gid, pid), (dev, def_ids) in store.items():
        ev = []
        for d in range(5):
            for s, e in runs(np.nan_to_num(dev[d], nan=-1.0) > thr):
                ev.append((d, s, e, float(np.nanmax(dev[d, s : e + 1]))))
        if not ev:
            continue
        flags = {k: cascades([(s, e) for _, s, e, _ in ev], g) for k, g in GAPS.items()}
        for j, (d, s, e, peak) in enumerate(ev):
            rows.append(
                {
                    "game_id": gid,
                    "possession_id": pid,
                    "def_id": int(def_ids[d]),
                    "start_step": s,
                    "end_step": e,
                    "duration_s": (e - s + 1) / 5,
                    "peak_dev_ft": peak,
                    **{f"cascade_{k}": flags[k][0][j] for k in GAPS},
                    **{f"initiator_{k}": flags[k][1][j] for k in GAPS},
                }
            )
    B = pl.DataFrame(rows)
    B.write_parquet(base / "analysis" / f"breakdowns{tag}.parquet")

    # player rates per 100 defensive possessions, split-half by game parity
    dev_tab = pl.read_parquet(base / "analysis" / "rule_ghost_dev.parquet").select(
        ["game_id", "possession_id", "def_id", "parity", "defense_team_id"]
    )
    par = dev_tab.select(["game_id", "parity"]).unique()
    Bp = B.join(par, on="game_id", how="left")
    rep = {"threshold_ft": thr, "n_breakdowns": B.height, "gaps": {}}
    for k in GAPS:
        ini = Bp.filter(pl.col(f"initiator_{k}"))
        cnt = ini.group_by(["def_id", "parity"]).len().rename({"len": "n_init"})
        poss = dev_tab.group_by(["def_id", "parity"]).len().rename({"len": "n_poss"})
        R = poss.join(cnt, on=["def_id", "parity"], how="left").with_columns(
            (100 * pl.col("n_init").fill_null(0) / pl.col("n_poss")).alias("rate")
        )
        tot = R.group_by("def_id").agg(pl.col("n_poss").sum(), pl.col("n_init").fill_null(0).sum())
        elig = tot.filter(pl.col("n_poss") >= MIN_POSS)["def_id"].to_list()
        W = (
            R.filter(pl.col("def_id").is_in(elig))
            .pivot(on="parity", index="def_id", values="rate")
            .drop_nulls()
        )
        n_cas = Bp.select(["game_id", "possession_id", f"cascade_{k}"]).unique().height
        rep["gaps"][k] = {
            "n_cascades": n_cas,
            "share_breakdowns_in_multi_cascades": float(
                Bp.group_by(["game_id", "possession_id", f"cascade_{k}"])
                .len()
                .filter(pl.col("len") > 1)["len"]
                .sum()
                / B.height
            ),
            "league_initiated_per_100_poss": float(100 * ini.height / dev_tab.height),
            "split_half_r_player_rate": float(np.corrcoef(W["0"], W["1"])[0, 1]),
            "n_players": W.height,
        }
    # possession outcomes with vs without a breakdown (descriptive)
    led = D.L.ledger(cfg.version).select(["poss_uid", "points"])
    Pp = P.select(["game_id", "possession_id", "poss_uid"]).join(led, on="poss_uid", how="left")
    has = B.select(["game_id", "possession_id"]).unique().with_columns(pl.lit(True).alias("has_bd"))
    Pp = Pp.join(has, on=["game_id", "possession_id"], how="left").with_columns(
        pl.col("has_bd").fill_null(False)
    )
    rep["possession_outcomes"] = {
        "share_possessions_with_breakdown": float(Pp["has_bd"].mean()),
        "points_with": float(Pp.filter(pl.col("has_bd"))["points"].mean()),
        "points_without": float(Pp.filter(~pl.col("has_bd"))["points"].mean()),
        "note": "descriptive; breakdowns and scoring share causes (penetration, mismatches)",
    }
    # player table (initiated breakdowns per 100, gap 1 s)
    ini = (
        Bp.filter(pl.col(f"initiator_{REPORT_GAP}"))
        .group_by("def_id")
        .len()
        .rename({"len": "n_init"})
    )
    poss = dev_tab.group_by("def_id").len().rename({"len": "n_poss"})
    PT = (
        poss.join(ini, on="def_id", how="left")
        .with_columns(
            pl.col("n_init").fill_null(0),
            (100 * pl.col("n_init").fill_null(0) / pl.col("n_poss")).alias("per_100"),
        )
        .filter(pl.col("n_poss") >= MIN_POSS)
        .sort("per_100")
    )
    out = Path("reports/phase4") / f"{cfg.version}_all"
    PT.write_csv(out / f"breakdown_rates_players{tag}.csv")
    rep["player_rate_quantiles_per_100"] = {
        q: float(PT["per_100"].quantile(q)) for q in (0.1, 0.5, 0.9)
    }
    rep["quantile"] = args.quantile
    (out / f"breakdowns{tag}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
