"""Second-chance sensitivity (D-018): possession windows including second chances.

Per game, nbacore's ``ghost_v1_windows`` with the same configuration as the pinned release's
``ghost_v1_l2release`` (FG windows end at the corrected shot release) plus
``include_second_chance=True``: a window whose ball never leaves the frontcourt starts at the last
offensive-rebound control inside it instead of being rejected. Windows of the release are
unchanged (checked: with the option off the build reproduces the release exactly on
``--check-games`` games); the added windows are flagged ``is_second_chance``.

Output: ``<processed_dir>/windows_l2_sc.parquet`` (read by ``ghost.data.ghost_v1`` when
``windows_release: l2_sc``) and ``<processed_dir>/windows_l2_sc_stats.json``.

Usage (from a directory whose configs/data.yaml sets windows_release: l2_sc):
    uv run python <repo>/scripts/build_windows_second_chance.py [--workers 3] [--check-games 5]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import nbacore.load as L
import polars as pl
from nbacore.possession.shots import ShotTimeConfig
from nbacore.views.ghost_v1 import SegmentConfig, ghost_v1_windows

from ghost import data as D

BASE_CFG = SegmentConfig(shot_cfg=ShotTimeConfig(shooter_only=True))  # = ghost_v1_l2release


def build_game(gid: str, version: str, include_sc: bool) -> tuple[pl.DataFrame, dict]:
    fr = L.frames(gid, version)
    fi = L.frame_index(gid, version).sort("period", "unix_ms")
    home = int(L.games(version).filter(pl.col("game_id") == gid)["home_team_id"][0])
    shots = L.events(gid, ["shot_release"], version)
    cfg = replace(BASE_CFG, include_second_chance=include_sc)
    return ghost_v1_windows(
        fr,
        fi,
        L.pbp(version, game_id=gid),
        L.attack_direction(version, game_id=gid),
        home,
        L.ledger(version, gid),
        shots,
        cfg,
    )


def run_game(gid: str, version: str, out_dir: str) -> dict:
    from pathlib import Path

    part = Path(out_dir) / f"{gid}.parquet"
    t0 = time.time()
    if not part.exists():
        win, rej = build_game(gid, version, True)
        win.write_parquet(part)
    else:
        rej = {}
    return {"game_id": gid, "seconds": round(time.time() - t0, 1), "rejects": rej}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--check-games", type=int, default=5)
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    assert cfg.windows_release == "l2_sc", "run from a config with windows_release: l2_sc"
    base = L.ghost_v1(cfg.version, release="l2")
    gids = sorted(base["game_id"].unique().to_list())

    # 1. the option off reproduces the release (same code path, same inputs)
    for gid in gids[: args.check_games]:
        win, _ = build_game(gid, cfg.version, False)
        ref = base.filter(pl.col("game_id") == gid)
        cols = [c for c in ref.columns if c in win.columns]
        a, b = win.select(cols).sort("window_uid"), ref.select(cols).sort("window_uid")
        if not a.equals(b):
            raise SystemExit(f"{gid}: rebuild without second chances differs from the release")
        print(f"check {gid}: identical to the release ({ref.height} windows)", flush=True)

    # 2. with second chances
    parts = cfg.processed_dir / "_windows_l2_sc"
    parts.mkdir(parents=True, exist_ok=True)
    t0, stats = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_game, g, cfg.version, str(parts)) for g in gids]
        for i, f in enumerate(as_completed(futs), 1):
            stats.append(f.result())
            if i % 25 == 0 or i == len(gids):
                print(f"[{i}/{len(gids)}] {time.time() - t0:.0f}s", flush=True)
    win = pl.concat([pl.read_parquet(parts / f"{g}.parquet") for g in gids], how="vertical")
    win = win.with_columns(
        (~pl.col("window_uid").is_in(base["window_uid"].implode())).alias("is_second_chance")
    )
    # release windows must be carried over unchanged
    kept = win.filter(~pl.col("is_second_chance"))
    cols = [c for c in base.columns if c in win.columns and c != "possession_id"]
    same = kept.select(cols).sort("window_uid").equals(base.select(cols).sort("window_uid"))
    win.write_parquet(cfg.processed_dir / "windows_l2_sc.parquet")
    sc = win.filter(pl.col("is_second_chance"))
    rep = {
        "nbacore_version": cfg.version,
        "n_games": len(gids),
        "windows_release": base.height,
        "windows_with_second_chances": win.height,
        "added_second_chance": sc.height,
        "added_halfcourt": sc.filter(~pl.col("is_transition")).height,
        "release_windows_missing": base.height - kept.height,
        "release_windows_unchanged": bool(same),
        "wall_seconds": round(time.time() - t0),
    }
    (cfg.processed_dir / "windows_l2_sc_stats.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
