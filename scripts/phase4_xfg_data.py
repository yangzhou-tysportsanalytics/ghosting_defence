"""Shot table for the shot-quality model (Phase 4 task 3).

One row per field-goal attempt of the tracked games (nbacore v1.1 ``shot_release`` events with
``offense_match``). Geometry is taken from the 25 Hz L1 frame at the release, mapped so that the
offence attacks the left basket. No raw trajectories are stored, only per-shot features.

Columns: game_id, t_release_ms, period, shooter_id, offense/defense team, made, blocked, zone,
dist_ft, angle_deg, is_three, def1_ft, def2_ft, def_front_ft (closest defender within 45 deg of
the shooter-rim line), n_dribbles, touch_s, catch_and_shoot, poss_uid, is_transition,
possession_id (our half-court window if the shot ends one), split, parity.

Output: <processed_dir>/all/analysis/shots.parquet

Usage:
    uv run python scripts/phase4_xfg_data.py
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from ghost import data as D
from ghost.court import HOOP_LEFT, is_three_point_location, to_left_half
from ghost.io.raw import BALL_ID


def zone_of(dist: np.ndarray, three: np.ndarray, y: np.ndarray) -> np.ndarray:
    corner = three & (np.abs(y - 25.0) >= 21.0)
    z = np.where(dist < 4, "rim", np.where(dist < 14, "paint", "mid"))
    z = np.where(three, np.where(corner, "corner3", "above3"), z)
    return z


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    led = D.L.ledger(cfg.version).select(["poss_uid", "is_transition", "defense_team_id"])
    P = (
        pl.read_parquet(base / "possessions.parquet")
        .filter(pl.col("terminal_msg_type").is_in([1, 2]))
        .select(["game_id", "possession_id", "t_terminal", "is_transition"])
        # a few windows share their terminal time; attach each shot to one window (the lower id)
        .sort(["game_id", "t_terminal", "possession_id"])
        .unique(subset=["game_id", "t_terminal"], keep="first", maintain_order=True)
    )
    splits = D.splits(cfg).select(["game_id", "split", "parity"])
    rows = []
    t0 = time.time()
    games = D.game_ids(cfg)
    for gi, gid in enumerate(games, 1):
        sh = D.events(gid, "shot_release", cfg)
        if sh.height == 0:
            continue
        touch = D.events(gid, "possession_touch", cfg, offense_only=False).select(
            ["actor_id", "t_start_ms", "t_end_ms", "n_dribbles"]
        )
        f = D.frames_at(gid, sh["t_release_ms"].to_list(), cfg)
        for r in sh.iter_rows(named=True):
            fr = f.filter(
                (pl.col("unix_ms") == r["t_release_ms"]) & (pl.col("period") == r["period"])
            )
            if fr.height < 10:
                continue
            sid = r["shooter_id"]
            srow = fr.filter(pl.col("player_id") == sid)
            if srow.height == 0:
                continue
            left = bool(r["attacks_left"])
            sx, sy = to_left_half(np.array([srow["x"][0]]), np.array([srow["y"][0]]), left)
            dfn = fr.filter((pl.col("team_id") != BALL_ID) & (pl.col("team_id") != r["team_id"]))
            dx, dy = to_left_half(dfn["x"].to_numpy(), dfn["y"].to_numpy(), left)
            s = np.array([sx[0], sy[0]])
            dd = np.hypot(dx - s[0], dy - s[1])
            order = np.sort(dd)
            to_rim = HOOP_LEFT - s
            dist = float(np.linalg.norm(to_rim))
            vecs = np.column_stack([dx - s[0], dy - s[1]])
            cosang = (vecs @ to_rim) / np.maximum(
                np.linalg.norm(vecs, axis=1) * max(dist, 1e-6), 1e-6
            )
            front = dd[cosang >= np.cos(np.pi / 4)]
            three = bool(is_three_point_location(np.array([s[0]]), np.array([s[1]]), True)[0])
            tc = touch.filter(
                (pl.col("actor_id") == sid)
                & (pl.col("t_start_ms") <= r["t_release_ms"])
                & (pl.col("t_start_ms") >= r["t_release_ms"] - 15000)
                & (pl.col("t_end_ms") >= r["t_release_ms"] - 1000)
            ).sort("t_start_ms")
            n_drib = int(tc["n_dribbles"][-1]) if tc.height else None
            touch_s = (r["t_release_ms"] - int(tc["t_start_ms"][-1])) / 1000 if tc.height else None
            rows.append(
                {
                    "game_id": gid,
                    "period": r["period"],
                    "t_release_ms": r["t_release_ms"],
                    "shooter_id": sid,
                    "offense_team_id": r["team_id"],
                    "poss_uid": r["poss_uid"],
                    "made": bool(r["made"]),
                    "release_method": r["method"],
                    "blocked": bool(r["blocked"]) if r["blocked"] is not None else False,
                    "x": float(s[0]),
                    "y": float(s[1]),
                    "dist_ft": dist,
                    "angle_deg": float(
                        np.degrees(np.arctan2(abs(s[1] - 25.0), max(s[0] - HOOP_LEFT[0], 1e-3)))
                    ),
                    "is_three": three,
                    "def1_ft": float(order[0]) if order.size else None,
                    "def2_ft": float(order[1]) if order.size > 1 else None,
                    "def_front_ft": float(front.min()) if front.size else 30.0,
                    "n_dribbles": n_drib,
                    "touch_s": touch_s,
                }
            )
        if gi % 100 == 0 or gi == len(games):
            print(f"[{gi}/{len(games)}] {time.time() - t0:.0f}s shots={len(rows)}", flush=True)
    S = pl.DataFrame(rows)
    S = S.with_columns(
        pl.Series(
            "zone", zone_of(S["dist_ft"].to_numpy(), S["is_three"].to_numpy(), S["y"].to_numpy())
        ),
        ((pl.col("n_dribbles").fill_null(0) == 0) & (pl.col("touch_s").fill_null(9) < 2.0)).alias(
            "catch_and_shoot"
        ),
    )
    S = S.join(led, on="poss_uid", how="left").join(splits, on="game_id", how="left")
    S = S.join(
        P.rename({"t_terminal": "t_release_ms", "is_transition": "window_is_transition"}),
        on=["game_id", "t_release_ms"],
        how="left",
    )
    out = base / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    S.write_parquet(out / "shots.parquet")
    print(S.select(["made", "is_three", "def1_ft", "n_dribbles", "catch_and_shoot"]).describe())
    print(S["zone"].value_counts().sort("zone"))
    print(
        "linked to our half-court windows:", S["possession_id"].is_not_null().sum(), "of", S.height
    )


if __name__ == "__main__":
    main()
