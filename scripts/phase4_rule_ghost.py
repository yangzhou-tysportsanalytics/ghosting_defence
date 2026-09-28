"""Phase 4 on the minimal path, CPU only: deviations from the *rule ghost*.

Rule ghost (until a learned ghost is trained): for a defender whose stable matchup at step t is
attacker k, the league-average position is the HMM emission mean
    mu = g_o O_k + g_b B + g_h H   (strong/weak-side weights of the fitted hmm_strat_help model).
Deviation d = z - mu. Two summaries:
    dev_ft   = |d|
    sag_ft   = d . u, u = unit vector from the attacker to the hoop
               (> 0: further toward the hoop than the league-average defender of that man;
                < 0: tighter / overplaying)
Help-state steps and steps without a stable man are excluded (they are not "guarding a man").

Aggregated per (possession, defender) with context covariates, so the table holds no raw
coordinates. Output: <processed_dir>/<game_set>/analysis/rule_ghost_dev.parquet

Usage:
    uv run python scripts/phase4_rule_ghost.py [--game-set all] [--model hmm_strat_help]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from ghost import data as D
from ghost.court import HOOP_LEFT
from ghost.matchup.events import EventConfig, stable_path
from ghost.tensors import load_game_set

MIN_STEPS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-set", default=None)
    ap.add_argument("--model", default="hmm_strat_help")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set=args.game_set)
    base = cfg.processed_dir / cfg.game_set
    prm = json.loads((base / "matchups" / f"{args.model}_params.json").read_text())
    g_strong = np.array(prm["gamma"]["strong"])
    g_weak = np.array(prm["gamma"]["weak"])
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    hold = EventConfig().steps(EventConfig().min_hold_s)
    rows = []
    t0 = time.time()
    games = sorted(P["game_id"].unique().to_list())
    for gi, gid in enumerate(games, 1):
        Pg = P.filter(pl.col("game_id") == gid)
        pt = load_game_set(base / "frames", [gid], Pg)
        M = pl.read_parquet(base / "matchups" / args.model / f"{gid}.parquet")
        n, T = pt.valid.shape
        key = {int(p): i for i, p in enumerate(pt.possession_id)}
        path = np.full((n, 5, T), -1, dtype=np.int8)
        ri = np.array([key[int(p)] for p in M["possession_id"]])
        path[ri, M["def_slot"].to_numpy(), M["step"].to_numpy()] = M["state"].to_numpy()
        meta = Pg.select(
            [
                "possession_id",
                "window_uid",
                "defense_team_id",
                "offense_team_id",
                "score_margin_offense",
            ]
        )
        mrow = {r["possession_id"]: r for r in meta.iter_rows(named=True)}
        for i in range(n):
            s = stable_path(path[i], pt.valid[i], hold)  # (5, T)
            ball = pt.ball[i, :, :2]
            for d in range(5):
                m = s[d].astype(int)
                man = (m >= 0) & (m < 5) & pt.valid[i]
                if man.sum() < MIN_STEPS:
                    continue
                tt = np.flatnonzero(man)
                k = m[tt]
                om = pt.off_xy[i, tt, k]  # (n,2)
                B = ball[tt]
                strong = np.sign(om[:, 1] - 25.0) == np.sign(B[:, 1] - 25.0)
                g = np.where(strong[:, None], g_strong, g_weak)
                mu = g[:, 0:1] * om + g[:, 1:2] * B + g[:, 2:3] * HOOP_LEFT
                z = pt.def_xy[i, tt, d]
                dv = z - mu
                u = HOOP_LEFT - om
                u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
                sag = (dv * u).sum(1)
                h = pt.handler[i, tt]
                r = mrow[int(pt.possession_id[i])]
                rows.append(
                    {
                        "game_id": gid,
                        "possession_id": int(pt.possession_id[i]),
                        "window_uid": r["window_uid"],
                        "def_id": int(pt.def_ids[i, d]),
                        "defense_team_id": int(r["defense_team_id"]),
                        "offense_team_id": int(r["offense_team_id"]),
                        "n_steps": int(tt.size),
                        "dev_ft": float(np.linalg.norm(dv, axis=1).mean()),
                        "sag_ft": float(sag.mean()),
                        "man_to_hoop_ft": float(np.linalg.norm(om - HOOP_LEFT, axis=1).mean()),
                        "man_to_ball_ft": float(np.linalg.norm(om - B, axis=1).mean()),
                        "share_strong": float(strong.mean()),
                        "share_on_ball": float((h == k).mean()),
                        "n_men": int(np.unique(k).size),
                        "score_margin_offense": r["score_margin_offense"],
                    }
                )
        if gi % 100 == 0 or gi == len(games):
            print(f"[{gi}/{len(games)}] {time.time() - t0:.0f}s rows={len(rows)}", flush=True)
    out = base / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).join(
        D.splits(cfg).select(["game_id", "split", "parity", "fold"]), on="game_id", how="left"
    )
    df.write_parquet(out / "rule_ghost_dev.parquet")
    print(df.describe())


if __name__ == "__main__":
    main()
