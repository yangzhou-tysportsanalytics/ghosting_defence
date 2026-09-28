"""Convert rule-ghost deviations into expected points at the shot (Phase 4 task 3, minimal path).

For every half-court possession ending in a field-goal attempt, at the release step of the 5 Hz
grid (all features recomputed on the grid so real and ghost are compared consistently):
  real          actual defender positions
  team ghost    every defender with a stable man moved to the rule-ghost position
                mu = g_o O_man + g_b B + g_h H (help / unmatched defenders stay where they are)
  indiv ghost   only the defender(s) matched to the shooter moved; the other four stay real (D-013)
Expected points = xFG x (3 if three else 2). Delta = EP_real - EP_ghost
(> 0: the real defence conceded more than the rule ghost would have).

Outputs: <processed_dir>/all/analysis/ghost_points.parquet;
         reports/phase4/<version>_all/ghost_points.json

Usage:
    uv run python scripts/phase4_ghost_points.py
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.court import HOOP_LEFT, is_three_point_location
from ghost.matchup.events import EventConfig, stable_path
from ghost.tensors import load_game_set


def _zone(dist, three, y):
    if three:
        return "corner3" if abs(y - 25.0) >= 21.0 else "above3"
    return "rim" if dist < 4 else ("paint" if dist < 14 else "mid")


def _defense_feats(s: np.ndarray, dfn: np.ndarray) -> tuple[float, float]:
    dd = np.linalg.norm(dfn - s, axis=1)
    to_rim = HOOP_LEFT - s
    n = max(np.linalg.norm(to_rim), 1e-6)
    cos = ((dfn - s) @ to_rim) / np.maximum(np.linalg.norm(dfn - s, axis=1) * n, 1e-6)
    front = dd[cos >= np.cos(np.pi / 4)]
    return float(dd.min()), float(front.min()) if front.size else 30.0


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    with open(base / "analysis" / "xfg_model.pkl", "rb") as fp:
        xfg = pickle.load(fp)
    prm = json.loads((base / "matchups" / "hmm_strat_help_params.json").read_text())
    g_s, g_w = np.array(prm["gamma"]["strong"]), np.array(prm["gamma"]["weak"])
    S = pl.read_parquet(base / "analysis" / "shots.parquet").filter(
        pl.col("possession_id").is_not_null() & ~pl.col("window_is_transition").fill_null(True)
    )
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    hold = EventConfig().steps(EventConfig().min_hold_s)
    rows = []
    t0 = time.time()
    games = sorted(S["game_id"].unique().to_list())
    for gi, gid in enumerate(games, 1):
        Sg = S.filter(pl.col("game_id") == gid)
        Pg = P.filter(pl.col("game_id") == gid).join(
            Sg.select(["possession_id"]), on="possession_id", how="semi"
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
            k = int(round((r["t_release_ms"] - pt.t[i, 0]) / 200))
            k = int(np.clip(k, 0, pt.valid[i].sum() - 1))
            slots = np.flatnonzero(pt.off_ids[i] == r["shooter_id"])
            if slots.size == 0:
                continue
            sh = int(slots[0])
            st = stable_path(path[i], pt.valid[i], hold)[:, k].astype(int)
            s = pt.off_xy[i, k, sh].astype(float)
            B = pt.ball[i, k, :2].astype(float)
            real = pt.def_xy[i, k].astype(float)
            ghost = real.copy()
            for d in range(5):
                m = st[d]
                if 0 <= m < 5:
                    om = pt.off_xy[i, k, m]
                    strong = np.sign(om[1] - 25) == np.sign(B[1] - 25)
                    g = g_s if strong else g_w
                    ghost[d] = g[0] * om + g[1] * B + g[2] * HOOP_LEFT
            ind = real.copy()
            matched = np.flatnonzero(st == sh)
            for d in matched:
                ind[d] = ghost[d]
            dist = float(np.linalg.norm(HOOP_LEFT - s))
            three = bool(is_three_point_location(np.array([s[0]]), np.array([s[1]]), True)[0])
            common = {
                "shooter_id": r["shooter_id"],
                "zone": _zone(dist, three, s[1]),
                "made": r["made"],
                "dist_ft": dist,
                "angle_deg": float(
                    np.degrees(np.arctan2(abs(s[1] - 25), max(s[0] - HOOP_LEFT[0], 1e-3)))
                ),
                "is_three": three,
                "n_dribbles": r["n_dribbles"],
                "catch_and_shoot": r["catch_and_shoot"],
                "is_transition": False,
            }
            feats = []
            for pos in (real, ghost, ind):
                d1, df_ = _defense_feats(s, pos)
                feats.append({**common, "def1_ft": d1, "def_front_ft": df_})
            X = pl.DataFrame(feats)
            p = xfg.predict(X)
            val = 3.0 if three else 2.0
            rows.append(
                {
                    "game_id": gid,
                    "possession_id": int(r["possession_id"]),
                    "t_release_ms": r["t_release_ms"],
                    "shooter_id": r["shooter_id"],
                    "defense_team_id": r["defense_team_id"],
                    "shooter_def_ids": [int(pt.def_ids[i, d]) for d in matched],
                    "made": r["made"],
                    "is_three": three,
                    "def1_real": feats[0]["def1_ft"],
                    "def1_team_ghost": feats[1]["def1_ft"],
                    "xfg_real": float(p[0]),
                    "xfg_team_ghost": float(p[1]),
                    "xfg_indiv_ghost": float(p[2]),
                    "dEP_team": float((p[0] - p[1]) * val),
                    "dEP_indiv": float((p[0] - p[2]) * val) if matched.size else None,
                    "split": r["split"],
                    "parity": r["parity"],
                }
            )
        if gi % 100 == 0 or gi == len(games):
            print(f"[{gi}/{len(games)}] {time.time() - t0:.0f}s rows={len(rows)}", flush=True)
    G = pl.DataFrame(rows)
    G.write_parquet(base / "analysis" / "ghost_points.parquet")

    # summaries
    abbr = {}
    for r in D.games(cfg).iter_rows(named=True):
        abbr[r["home_team_id"]] = r["home_abbr"]
        abbr[r["visitor_team_id"]] = r["visitor_abbr"]
    team = (
        G.group_by("defense_team_id")
        .agg(
            pl.col("dEP_team").mean().alias("dEP_team_per_shot"),
            pl.len().alias("n_shots"),
            pl.col("xfg_real").mean(),
            pl.col("xfg_team_ghost").mean(),
        )
        .sort("dEP_team_per_shot")
    )
    tp = (
        G.group_by(["defense_team_id", "parity"])
        .agg(pl.col("dEP_team").mean())
        .pivot(on="parity", index="defense_team_id", values="dEP_team")
        .drop_nulls()
    )
    ind = G.filter(pl.col("dEP_indiv").is_not_null() & (pl.col("shooter_def_ids").list.len() == 1))
    ind = ind.with_columns(pl.col("shooter_def_ids").list.first().alias("def_id"))
    cnt = ind.group_by("def_id").len()
    elig = cnt.filter(pl.col("len") >= 100)["def_id"].to_list()
    ip = (
        ind.filter(pl.col("def_id").is_in(elig))
        .group_by(["def_id", "parity"])
        .agg(pl.col("dEP_indiv").mean())
        .pivot(on="parity", index="def_id", values="dEP_indiv")
        .drop_nulls()
    )
    summary = {
        "n_shots": G.height,
        "mean_xfg_real": float(G["xfg_real"].mean()),
        "mean_xfg_team_ghost": float(G["xfg_team_ghost"].mean()),
        "mean_dEP_team_per_shot": float(G["dEP_team"].mean()),
        "mean_def1_real_ft": float(G["def1_real"].mean()),
        "mean_def1_team_ghost_ft": float(G["def1_team_ghost"].mean()),
        "team_dEP_range": [
            float(team["dEP_team_per_shot"].min()),
            float(team["dEP_team_per_shot"].max()),
        ],
        "team_split_half_r": float(np.corrcoef(tp["0"], tp["1"])[0, 1]),
        "indiv_n_defenders_ge_100_shots": len(elig),
        "indiv_split_half_r": float(np.corrcoef(ip["0"], ip["1"])[0, 1]) if ip.height > 5 else None,
        "teams": [{"team": abbr.get(r["defense_team_id"]), **r} for r in team.to_dicts()],
        "note": "rule ghost = matchup-conditional league-average position; at the release real "
        "defenders close out, so the ghost is not expected to be neutral at the shot.",
    }
    out = Path("reports/phase4") / f"{cfg.version}_all"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ghost_points.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "teams"}, indent=2))


if __name__ == "__main__":
    main()
