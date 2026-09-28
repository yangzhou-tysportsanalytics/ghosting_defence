"""Phase 2 task 3 (second half): switch A classification and closeouts with nbacore v1.1 events.

Inputs: switch-B events and matchup paths from phase2_events.py / phase2_matchups.py; nbacore
``screen_candidate`` and ``pass`` / ``handoff`` events filtered to ``offense_match`` (nbacore
recommendation). Screens are attached to a possession when their contact time falls inside the
window [t_start - 1 s, t_end] of the same period; passes when the catch does.

Outputs (under <processed_dir>/<game_set>/events/): switches.parquet (switch-B events with
``event_class`` screen_switch / one_sided / rotation and ``screen_uid``), closeouts.parquet;
reports/phase2/<version>_<game_set>/screen_events_summary.json

Usage:
    uv run python scripts/phase2_screen_events.py [--game-set small] [--model hmm_strat_help]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.matchup.events import (
    PASS_COLUMNS,
    SCREEN_COLUMNS,
    EventConfig,
    classify_switches,
    closeout_events,
    stable_path,
)
from ghost.tensors import load_game_set


def attach(ev: pl.DataFrame, P: pl.DataFrame, t_col: str, pad_ms: int) -> pl.DataFrame:
    """Attach events to possessions by period and time (first matching window)."""
    w = P.select(["game_id", "possession_id", "period", "t_start", "t_end"])
    j = ev.join(w, on=["game_id", "period"], how="inner").filter(
        (pl.col(t_col) >= pl.col("t_start") - pad_ms) & (pl.col(t_col) <= pl.col("t_end"))
    )
    return j.sort(["event_uid", "possession_id"]).unique(subset=["event_uid"], keep="first")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-set", default=None)
    ap.add_argument("--model", default="hmm_strat_help")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set=args.game_set)
    ecfg = EventConfig()
    base = cfg.processed_dir / cfg.game_set
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    SWB = pl.read_parquet(base / "events" / "switches_b.parquet")
    hold = ecfg.steps(ecfg.min_hold_s)
    t0 = time.time()
    out_sw, out_co = [], []
    n_screens = n_on_ball = n_passes = 0
    screen_rows = []
    games = sorted(P["game_id"].unique().to_list())
    for gi, gid in enumerate(games, 1):
        Pg = P.filter(pl.col("game_id") == gid)
        sc = D.events(gid, "screen_candidate", cfg)
        ps = D.events(gid, ["pass", "handoff"], cfg)
        sc = sc.with_columns(
            pl.coalesce("user_def_id", "screened_def_id").alias("screened_def_id")
        )  # v1.3: user_def_id = the user's defender in the 1 s before contact (nbacore)
        sc = attach(
            sc.select([*SCREEN_COLUMNS, "period", "dup_group", "screen_group_uid"]),
            Pg,
            "t_contact_ms",
            1000,
        )
        ps = attach(
            ps.filter(pl.col("completed")).select([*PASS_COLUMNS, "period", "kind"]),
            Pg,
            "t_catch_ms",
            0,
        )
        n_screens += sc.height
        n_on_ball += int(sc["on_ball"].sum()) if sc.height else 0
        n_passes += ps.height
        swg = SWB.filter(pl.col("game_id") == gid)
        pt = load_game_set(base / "frames", [gid], Pg)
        M = pl.read_parquet(base / "matchups" / args.model / f"{gid}.parquet")
        n, T = pt.valid.shape
        key = {int(p): i for i, p in enumerate(pt.possession_id)}
        path = np.full((n, 5, T), -1, dtype=np.int8)
        rows = np.array([key[int(p)] for p in M["possession_id"]])
        path[rows, M["def_slot"].to_numpy(), M["step"].to_numpy()] = M["state"].to_numpy()
        for pid, i in key.items():
            sw = swg.filter(pl.col("possession_id") == pid).drop(["game_id", "possession_id"])
            scp = sc.filter(pl.col("possession_id") == pid)
            if sw.height:
                cls = classify_switches(sw, scp, ecfg).with_columns(
                    pl.lit(gid).alias("game_id"), pl.lit(pid).alias("possession_id")
                )
                out_sw.append(cls)
            # per-screen outcome (for the switch rate per screen)
            if scp.height:
                cl_uids = set(cls["screen_uid"].drop_nulls().to_list()) if sw.height else set()
                screen_rows.append(
                    scp.select(["game_id", "event_uid", "on_ball", "dup_group"]).with_columns(
                        pl.col("event_uid").is_in(list(cl_uids)).alias("switched_or_one_sided")
                    )
                )
            psp = ps.filter(pl.col("possession_id") == pid)
            if psp.height:
                s_ = stable_path(path[i], pt.valid[i], hold)
                for e in closeout_events(
                    psp,
                    s_,
                    pt.off_xy[i],
                    pt.def_xy[i],
                    pt.valid[i],
                    pt.t[i],
                    pt.off_ids[i],
                    pt.def_ids[i],
                    ecfg,
                ):
                    out_co.append({"game_id": gid, "possession_id": pid, **e})
        if gi % 100 == 0 or gi == len(games):
            print(f"[{gi}/{len(games)}] {time.time() - t0:.0f}s", flush=True)

    SW = pl.concat(out_sw, how="diagonal") if out_sw else pl.DataFrame()
    CO = pl.DataFrame(out_co)
    SR = pl.concat(screen_rows) if screen_rows else pl.DataFrame()
    out = base / "events"
    SW.write_parquet(out / "switches.parquet")
    CO.write_parquet(out / "closeouts.parquet")
    N = P.height
    cls_counts = SW["event_class"].value_counts().sort("event_class") if SW.height else None
    screen_switch_pairs = SW.filter(pl.col("event_class") == "screen_switch")
    # a real screen typically has 3-4 candidates (dup_group); count distinct screen groups
    groups = SR.select(["game_id", "dup_group"]).unique() if SR.height else SR
    switched_groups = (
        SR.filter(pl.col("switched_or_one_sided")).select(["game_id", "dup_group"]).unique().height
        if SR.height
        else 0
    )
    summary = {
        "model": args.model,
        "n_halfcourt_possessions": N,
        "screen_candidates_offense_match_in_windows": n_screens,
        "screen_candidates_on_ball": n_on_ball,
        "screen_groups (dup_group)": groups.height if SR.height else 0,
        "completed_passes_in_windows": n_passes,
        "switch_B_by_class": dict(zip(*cls_counts.to_dict(as_series=False).values(), strict=True))
        if cls_counts is not None
        else {},
        "screen_switch_events_per_possession": screen_switch_pairs.height / 2 / N,
        "screen_groups_with_switch_or_one_sided": switched_groups,
        "closeouts": {
            "n": CO.height,
            "per_possession": CO.height / N,
            "per_completed_pass": CO.height / max(n_passes, 1),
            "matched_defender_share": float(CO["matched_defender"].mean()) if CO.height else None,
            "closing_speed_fts_median": float(CO["closing_speed_fts"].median())
            if CO.height
            else None,
        },
        "seconds": round(time.time() - t0, 1),
    }
    rep = Path("reports/phase2") / f"{cfg.version}_{cfg.game_set}"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "screen_events_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
