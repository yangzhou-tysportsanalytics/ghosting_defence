"""Draw the stratified annotation sample (docs/annotation_guide.md §5; D-018).

Sample v1 (full season, release-corrected windows, nbacore screens):
  pnr_switch      the possession has a screen switch (switch A)                 50
  pnr_no_switch   it has an on-ball screen group but no screen switch             50
  drive           the handler gains >= 8 ft toward the rim within 2 s              35
  ball_movement   >= 3 handler changes                                             35
  other           none of the above                                               30
Strata are assigned in that order. Only TRAIN and VAL games of the fixed split, half-court.
30 possessions (spread over strata) are flagged for double annotation. Keys: ``window_uid``.

Output: data/annotations/sample_v1.parquet (ids only, safe to commit) + summary JSON.

Usage:
    uv run python scripts/select_annotation_sample.py [--game-set all] [--seed 9]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.court import HOOP_LEFT
from ghost.tensors import load_game_set

ALLOC = {"pnr_switch": 50, "pnr_no_switch": 50, "drive": 35, "ball_movement": 35, "other": 30}


def proxies(pt) -> tuple[np.ndarray, np.ndarray]:
    N, T = pt.valid.shape
    changes = np.zeros(N, dtype=int)
    drive = np.zeros(N, dtype=bool)
    for i in range(N):
        h = pt.handler[i][pt.valid[i]]
        hh = h[h >= 0]
        changes[i] = int((np.diff(hh) != 0).sum()) if hh.size else 0
        hv = (pt.handler[i] >= 0) & pt.valid[i]
        xy = pt.off_xy[i, np.arange(T), np.clip(pt.handler[i], 0, 4)]
        dh = np.linalg.norm(xy - HOOP_LEFT, axis=-1)
        for k in range(T - 10):
            if (
                hv[k]
                and hv[k + 10]
                and pt.handler[i, k] == pt.handler[i, k + 10]
                and dh[k] - dh[k + 10] >= 8
            ):
                drive[i] = True
                break
    return changes, drive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-set", default="all")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--version-tag", default="v1")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set=args.game_set)
    base = cfg.processed_dir / cfg.game_set
    P = pl.read_parquet(base / "possessions.parquet").filter(
        ~pl.col("is_transition") & pl.col("split").is_in(["train", "val"])
    )
    SW = pl.read_parquet(base / "events" / "switches.parquet")
    sw_poss = (
        SW.filter(pl.col("event_class") == "screen_switch")
        .select(["game_id", "possession_id"])
        .unique()
    )
    OB = (
        pl.read_parquet(base / "analysis" / "onball_screens.parquet")
        .select(["game_id", "possession_id"])
        .unique()
    )
    rows = []
    for gid in sorted(P["game_id"].unique().to_list()):
        Pg = P.filter(pl.col("game_id") == gid)
        pt = load_game_set(base / "frames", [gid], Pg)
        ch, dr = proxies(pt)
        rows.append(
            pl.DataFrame(
                {
                    "game_id": pt.game_id,
                    "possession_id": pt.possession_id,
                    "handler_changes": ch,
                    "drive": dr,
                }
            )
        )
    S = pl.concat(rows).join(
        P.select(
            ["game_id", "possession_id", "window_uid", "split", "terminal_type", "duration_s"]
        ),
        on=["game_id", "possession_id"],
    )
    S = S.join(
        sw_poss.with_columns(pl.lit(True).alias("has_switch")),
        on=["game_id", "possession_id"],
        how="left",
    )
    S = S.join(
        OB.with_columns(pl.lit(True).alias("has_onball")),
        on=["game_id", "possession_id"],
        how="left",
    )
    S = S.with_columns(pl.col("has_switch").fill_null(False), pl.col("has_onball").fill_null(False))
    S = S.with_columns(
        pl.when(pl.col("has_switch"))
        .then(pl.lit("pnr_switch"))
        .when(pl.col("has_onball"))
        .then(pl.lit("pnr_no_switch"))
        .when(pl.col("drive"))
        .then(pl.lit("drive"))
        .when(pl.col("handler_changes") >= 3)
        .then(pl.lit("ball_movement"))
        .otherwise(pl.lit("other"))
        .alias("stratum")
    )
    rng = np.random.default_rng(args.seed)
    picks = []
    for s, k in ALLOC.items():
        pool = S.filter(pl.col("stratum") == s).sort(["game_id", "possession_id"])
        idx = rng.choice(pool.height, size=min(k, pool.height), replace=False)
        picks.append(pool[np.sort(idx)])
    sample = pl.concat(picks)
    # 30 double-annotated possessions spread proportionally over strata
    dbl = np.zeros(sample.height, dtype=bool)
    for s, k in ALLOC.items():
        ix = np.flatnonzero(sample["stratum"].to_numpy() == s)
        dbl[rng.choice(ix, size=round(30 * k / 200), replace=False)] = True
    sample = sample.with_columns(
        pl.Series("double_annotate", dbl),
        pl.lit(f"{args.version_tag}-{cfg.version}-{cfg.windows_release}").alias("sample_version"),
    )
    out = Path("data/annotations")
    out.mkdir(parents=True, exist_ok=True)
    sample.drop(["handler_changes", "drive"]).write_parquet(
        out / f"sample_{args.version_tag}.parquet"
    )

    def dist(col: pl.Series) -> dict:
        vc = col.value_counts().sort(col.name)
        return dict(zip(vc[col.name].to_list(), vc["count"].to_list(), strict=True))

    summary = {
        "pool": dist(S["stratum"]),
        "sample": dist(sample["stratum"]),
        "by_split": dist(sample["split"]),
        "n_games": sample["game_id"].n_unique(),
        "double_annotate": int(dbl.sum()),
        "data": {"nbacore": cfg.version, "windows_release": cfg.windows_release},
    }
    (out / f"sample_{args.version_tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
