"""Shot table for the shot-quality model (Phase 4 task 3).

One row per field-goal attempt of the tracked games (nbacore v1.1 ``shot_release`` events with
``offense_match``). Geometry is taken from the 25 Hz L1 frame at the release, mapped so that the
offence attacks the left basket. No raw trajectories are stored, only per-shot features.

Columns: game_id, t_release_ms, period, shooter_id, offense/defense team, made, blocked, zone,
dist_ft, angle_deg, is_three, def1_ft, def2_ft, def_front_ft, n_dribbles (so far), touch_s,
catch_and_shoot, event_uid, pbp_event_num, poss_uid, is_transition (definitions:
``ghost.xfg.features``),
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
from ghost.xfg.features import catch_and_shoot, geometry, touch_state, zone_of


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
        touch = D.events(gid, "possession_touch", cfg, offense_only=False)
        drib = D.events(gid, "dribble", cfg, offense_only=False)
        tk = [touch[c].to_numpy() for c in ("actor_id", "t_start_ms", "t_end_ms", "event_uid")]
        dk = (
            [drib[c].to_numpy() for c in ("touch_uid", "t_start_ms")]
            if drib.height
            else [
                np.array([], dtype=object),
                np.array([], dtype=np.int64),
            ]
        )
        f = D.frames_at(gid, sh["t_release_ms"].to_list(), cfg)
        for r in sh.iter_rows(named=True):
            fr = f.filter(
                (pl.col("unix_ms") == r["t_release_ms"]) & (pl.col("period") == r["period"])
            )
            g = geometry(
                fr["team_id"].to_numpy(),
                fr["player_id"].to_numpy(),
                fr["x"].to_numpy(),
                fr["y"].to_numpy(),
                r["shooter_id"],
                r["team_id"],
                bool(r["attacks_left"]),
            )
            if g is None:
                continue
            n_drib, touch_s = touch_state(*tk, *dk, r["shooter_id"], r["t_release_ms"])
            rows.append(
                {
                    "game_id": gid,
                    "event_uid": r["event_uid"],
                    "pbp_event_num": r["pbp_event_num"],
                    "period": r["period"],
                    "t_release_ms": r["t_release_ms"],
                    "shooter_id": r["shooter_id"],
                    "offense_team_id": r["team_id"],
                    "poss_uid": r["poss_uid"],
                    "made": bool(r["made"]),
                    "release_method": r["method"],
                    "blocked": bool(r["blocked"]) if r["blocked"] is not None else False,
                    **g,
                    "n_dribbles": n_drib,
                    "touch_s": touch_s,
                    "catch_and_shoot": catch_and_shoot(n_drib, touch_s),
                }
            )
        if gi % 100 == 0 or gi == len(games):
            print(f"[{gi}/{len(games)}] {time.time() - t0:.0f}s shots={len(rows)}", flush=True)
    S = pl.DataFrame(rows)
    S = S.with_columns(
        pl.Series(
            "zone", zone_of(S["dist_ft"].to_numpy(), S["is_three"].to_numpy(), S["y"].to_numpy())
        )
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
