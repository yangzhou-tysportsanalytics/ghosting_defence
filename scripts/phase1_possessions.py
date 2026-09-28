"""Phase 1 steps 3-5 on the shared nbacore release: possessions, 5 Hz left-half frames, ball
handler, pbp validation. L1 inputs and the fixed split come from nbacore (D-016); the
segmentation rules (D-009, D-011, D-012) are this project's own and unchanged.

Writes under ``<processed_dir>/<game_set>/`` (default data/processed/nbacore-v1.0/<game_set>/):
    possessions.parquet                  one row per accepted possession (+ split / fold / parity)
    possessions/<game_id>.parquet        same, per game (lets an interrupted run resume)
    frames/<game_id>.parquet             11 rows x 121 steps per possession (5 Hz, left half)
    splits.parquet                       fixed split of the games in the set (from games.parquet)
and reports/phase1/<version>_<game_set>/possessions_stats.json.

Usage:
    uv run python scripts/phase1_possessions.py [--game-set tiny|small|all] [--workers 1]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.ballhandler.infer import (
    HandlerConfig,
    handler_on_grid,
    infer_handler_raw,
    last_handler_before,
)
from ghost.possession.resample import ResampleConfig, arrays_to_long, resample_possession
from ghost.possession.segment import POSSESSION_SCHEMA, SHOT_TYPES, SegmentConfig, segment_game


def process_game(
    gid: str, cfg: D.DataConfig, out_dir: str, force: bool, windows: str = "nbacore"
) -> dict:
    """Windows for one game (from nbacore ghost_v1, or re-segmented locally); write per-game
    possessions and 5 Hz frames; return per-game statistics."""
    out = Path(out_dir)
    poss_path = out / "possessions" / f"{gid}.parquet"
    stats_path = out / "per_game_stats" / f"{gid}.json"
    if poss_path.exists() and stats_path.exists() and not force:
        return json.loads(stats_path.read_text())
    t0 = time.time()
    ent = D.frames(gid, cfg)
    fi = D.frame_index(gid, cfg)
    direction = D.attack_direction(cfg, gid)
    if windows == "nbacore":
        # nbacore v1.2 ghost_v1 reproduces segment_game exactly (all 631 games, 2026-09-26);
        # reject counts live in nbacore's reports.
        poss = (
            D.ghost_v1(cfg)
            .filter(pl.col("game_id") == gid)
            .sort("possession_id")
            .select(list(POSSESSION_SCHEMA))
        )
        rej: dict[str, int] = {}
    else:
        poss, rej = segment_game(
            ent, fi, D.pbp(cfg, gid), direction, D.home_team_id(gid, cfg), SegmentConfig()
        )
    rs_cfg, h_cfg = ResampleConfig(), HandlerConfig()
    frames_out, checks, coverage = [], [], []
    for row in poss.iter_rows(named=True):
        pa = resample_possession(ent, fi, row, rs_cfg)
        t_raw, h_raw = infer_handler_raw(
            ent, row["period"], row["t_start"], row["t_end"], row["offense_player_ids"], h_cfg
        )
        h_grid = np.where(pa.valid, handler_on_grid(t_raw, h_raw, pa.t), -1).astype(np.int8)
        frames_out.append(arrays_to_long(pa, h_grid))
        coverage.append(float((h_grid[pa.valid] >= 0).mean()))
        if row["terminal_msg_type"] in SHOT_TYPES and row["terminal_player_id"] is not None:
            slot = last_handler_before(t_raw, h_raw, row["t_end"])
            matched = slot >= 0 and int(pa.off_ids[slot]) == int(row["terminal_player_id"])
            checks.append((slot >= 0, bool(matched)))
    if frames_out:
        pl.concat(frames_out).write_parquet(out / "frames" / f"{gid}.parquet", compression="zstd")
    poss.write_parquet(poss_path)
    n_hc = poss.filter(~pl.col("is_transition")).height
    st = {
        "game_id": gid,
        "n_terminal_events": sum(rej.values()) + poss.height,
        "n_possessions": poss.height,
        "n_halfcourt_v1": n_hc,
        "rejects": rej,
        "handler_checks": len(checks),
        "handler_has": sum(c[0] for c in checks),
        "handler_matched": sum(c[1] for c in checks),
        "handler_coverage_sum": float(np.sum(coverage)),
        "seconds": round(time.time() - t0, 1),
    }
    stats_path.write_text(json.dumps(st))
    return st


def _dist(series: pl.Series) -> dict:
    vc = series.value_counts().sort(series.name)
    return dict(zip(vc[series.name].to_list(), vc["count"].to_list(), strict=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-set", default=None, help="tiny | small | all (default: config)")
    ap.add_argument("--workers", type=int, default=1, help="parallel games (<= 3 on 16 GB)")
    ap.add_argument("--force", action="store_true", help="recompute games already done")
    ap.add_argument(
        "--windows",
        default="nbacore",
        choices=["nbacore", "local"],
        help="possession windows from nbacore ghost_v1 (default) or local segment_game",
    )
    ap.add_argument(
        "--keys-only",
        action="store_true",
        help="do not recompute; rebuild possessions.parquet with nbacore stable keys",
    )
    args = ap.parse_args()

    cfg = D.DataConfig.load(game_set=args.game_set)
    ids = D.game_ids(cfg)
    out = cfg.processed_dir / cfg.game_set
    for sub in ("frames", "possessions", "per_game_stats"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    report_dir = Path("reports/phase1") / f"{cfg.version}_{cfg.game_set}"
    report_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"nbacore {cfg.version}, game set {cfg.game_set}: {len(ids)} games, {args.workers} worker(s)"
    )

    t0 = time.time()
    per_game = []
    if args.keys_only:
        per_game = [json.loads((out / "per_game_stats" / f"{g}.json").read_text()) for g in ids]
    elif args.workers <= 1:
        for i, gid in enumerate(ids, 1):
            per_game.append(process_game(gid, cfg, str(out), args.force, args.windows))
            st = per_game[-1]
            print(
                f"[{i}/{len(ids)}] {gid}: halfcourt={st['n_halfcourt_v1']} "
                f"rejects={sum(st['rejects'].values())} [{st['seconds']}s]",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(process_game, g, cfg, str(out), args.force, args.windows): g for g in ids
            }
            for i, fut in enumerate(as_completed(futs), 1):
                st = fut.result()
                per_game.append(st)
                print(
                    f"[{i}/{len(ids)}] {st['game_id']}: halfcourt={st['n_halfcourt_v1']} "
                    f"[{st['seconds']}s]",
                    flush=True,
                )
    per_game.sort(key=lambda s: s["game_id"])

    splits = D.splits(cfg).filter(pl.col("game_id").is_in(ids))
    splits.write_parquet(out / "splits.parquet")
    keys = D.ghost_v1(cfg).select(["game_id", "possession_id", *D.WINDOW_KEYS])
    P = (
        pl.concat([pl.read_parquet(out / "possessions" / f"{g}.parquet") for g in ids])
        .select(list(POSSESSION_SCHEMA))
        .join(
            splits.select(["game_id", "game_date", "split", "fold", "parity"]),
            on="game_id",
            how="left",
        )
        .join(keys, on=["game_id", "possession_id"], how="left")
    )
    if P["window_uid"].null_count():
        raise RuntimeError("possessions without an nbacore window_uid: windows diverged")
    P.write_parquet(out / "possessions.parquet")

    hc = P.filter(~pl.col("is_transition"))
    shots = P.filter(pl.col("terminal_msg_type").is_in(SHOT_TYPES))
    rejects: dict[str, int] = {}
    for st in per_game:
        for k, v in st["rejects"].items():
            rejects[k] = rejects.get(k, 0) + v
    top_rejects: dict[str, int] = {}
    for k, v in rejects.items():
        key = ":".join(k.split(":")[:2])
        top_rejects[key] = top_rejects.get(key, 0) + v
    n_chk = sum(s["handler_checks"] for s in per_game)
    n_valid_poss = sum(s["n_possessions"] for s in per_game)
    stats = {
        "nbacore_version": cfg.version,
        "nbacore_manifest_commit": D.manifest(cfg).get("code_commit"),
        "game_set": cfg.game_set,
        "n_games": len(ids),
        "wall_seconds": round(time.time() - t0, 1),
        "games_by_split": _dist(splits["split"]),
        "n_possessions": P.height,
        "n_halfcourt_v1": hc.height,
        "n_transition": P.height - hc.height,
        "per_game_halfcourt_mean": hc.height / max(1, len(ids)),
        "halfcourt_by_split": _dist(hc["split"]),
        "rejects_total": dict(sorted(top_rejects.items())),
        "rejects_detail": dict(sorted(rejects.items())),
        "terminal_type_counts": _dist(hc["terminal_type"]),
        "duration_s_quantiles": {
            q: float(hc["duration_s"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "cropped_frac": float(hc["cropped"].mean()) if hc.height else None,
        "handler_frame_coverage_mean": sum(s["handler_coverage_sum"] for s in per_game)
        / max(1, n_valid_poss),
        "shot_time": {
            "n_shots": shots.height,
            "method_counts": _dist(shots["shot_time_method"]),
            "release_minus_pbp_s_quantiles": {
                q: float(((shots["t_terminal"] - shots["t_pbp_terminal"]) / 1000).quantile(q))
                for q in (0.05, 0.25, 0.5, 0.75, 0.95)
            },
            "inferred_shooter_match_frac": float(
                (shots["shooter_inferred_id"] == shots["terminal_player_id"])
                .fill_null(False)
                .mean()
            ),
        },
        "handler_vs_shooter": {
            "n_shot_possessions": n_chk,
            "has_handler_frac": sum(s["handler_has"] for s in per_game) / max(1, n_chk),
            "match_frac": sum(s["handler_matched"] for s in per_game) / max(1, n_chk),
        },
        "per_game": per_game,
    }
    (report_dir / "possessions_stats.json").write_text(json.dumps(stats, indent=2))
    print(
        json.dumps(
            {k: v for k, v in stats.items() if k not in ("per_game", "rejects_detail")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
