"""Phase 2 task 3: derive switch / help events from the matchup paths.

Computed now: switch B (reassignments), help B (geometric), help A (help-state posterior).
Waiting for nbacore v1.1 (screens, passes): switch A classification and closeouts
(``ghost.matchup.events.classify_switches`` / ``closeout_events`` are ready and tested).

Outputs (under <processed_dir>/<game_set>/events/): switches_b.parquet, help.parquet;
reports/phase2/<version>_<game_set>/events_summary.json

Usage:
    uv run python scripts/phase2_events.py [--game-set small] [--model hmm_strat_help]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.matchup.events import (
    EventConfig,
    help_a_events,
    help_b_events,
    stable_path,
    switch_b_events,
)
from ghost.tensors import load_game_set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-set", default=None)
    ap.add_argument("--model", default="hmm_strat_help")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set=args.game_set)
    ecfg = EventConfig()
    base = cfg.processed_dir / cfg.game_set
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    sw, hb, ha = [], [], []
    hold = ecfg.steps(ecfg.min_hold_s)
    has_help = False
    N = 0
    for gid in sorted(P["game_id"].unique().to_list()):  # game by game: bounded memory
        pt = load_game_set(base / "frames", [gid], P.filter(pl.col("game_id") == gid))
        M = pl.read_parquet(base / "matchups" / args.model / f"{gid}.parquet")
        n, T = pt.valid.shape
        N += n
        key = {int(p): i for i, p in enumerate(pt.possession_id)}
        path = np.full((n, 5, T), -1, dtype=np.int8)
        p_help = np.zeros((n, 5, T), dtype=np.float32)
        rows = np.array([key[int(p)] for p in M["possession_id"]])
        path[rows, M["def_slot"].to_numpy(), M["step"].to_numpy()] = M["state"].to_numpy()
        has_help = M["p_help"].null_count() < M.height
        if has_help:
            p_help[rows, M["def_slot"].to_numpy(), M["step"].to_numpy()] = M["p_help"].to_numpy()
        for i in range(n):
            s_ = stable_path(path[i], pt.valid[i], hold)
            meta = {"game_id": str(pt.game_id[i]), "possession_id": int(pt.possession_id[i])}
            for e in switch_b_events(s_, pt.t[i], pt.off_ids[i], pt.def_ids[i]):
                sw.append({**meta, **e})
            for e in help_b_events(
                s_,
                pt.handler[i],
                pt.off_xy[i],
                pt.def_xy[i],
                pt.valid[i],
                pt.t[i],
                pt.off_ids[i],
                pt.def_ids[i],
                ecfg,
            ):
                hb.append({**meta, **e})
            if has_help:
                for e in help_a_events(
                    p_help[i],
                    pt.handler[i],
                    pt.off_xy[i],
                    pt.def_xy[i],
                    pt.valid[i],
                    pt.t[i],
                    pt.def_ids[i],
                    pt.off_ids[i],
                    ecfg,
                ):
                    ha.append({**meta, **e})
    out = base / "events"
    out.mkdir(parents=True, exist_ok=True)
    SW = pl.DataFrame(sw)
    H = pl.concat([pl.DataFrame(hb), pl.DataFrame(ha)], how="diagonal") if ha else pl.DataFrame(hb)
    SW.write_parquet(out / "switches_b.parquet")
    H.write_parquet(out / "help.parquet")

    def per_poss(df: pl.DataFrame) -> dict:
        c = df.group_by(["game_id", "possession_id"]).len()["len"] if df.height else pl.Series([0])
        return {
            "n_events": df.height,
            "per_possession": df.height / N,
            "share_possessions_with_any": c.len() / N if df.height else 0.0,
        }

    hb_df = H.filter(pl.col("definition") == "B") if H.height else H
    ha_df = H.filter(pl.col("definition") == "A") if H.height and has_help else pl.DataFrame()
    # overlap of help A and help B: same defender, onsets within 0.6 s
    overlap = None
    if hb_df.height and ha_df.height:
        j = hb_df.join(ha_df, on=["game_id", "possession_id", "def_id"], suffix="_a")
        j = j.filter((pl.col("t_start_ms") - pl.col("t_start_ms_a")).abs() <= 600)
        overlap = {
            "help_B_also_A": j.select(["game_id", "possession_id", "def_id", "t_start_ms"])
            .unique()
            .height
            / hb_df.height,
            "help_A_also_B": j.select(["game_id", "possession_id", "def_id", "t_start_ms_a"])
            .unique()
            .height
            / ha_df.height,
        }
    summary = {
        "model": args.model,
        "n_possessions": N,
        "switch_B": per_poss(SW),
        "help_B": per_poss(hb_df),
        "help_A": per_poss(ha_df) if ha_df.height else None,
        "help_A_B_overlap": overlap,
        "switch_A_and_closeouts": "see phase2_screen_events.py (needs nbacore screens / passes)",
        "config": ecfg.__dict__,
    }
    rep = Path("reports/phase2") / f"{cfg.version}_{cfg.game_set}"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "events_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
