"""Pack the ghost model's inputs into plain arrays, so training can run on a machine without the
nbacore data layer (e.g. a rented GPU).

For each split (train / val / test of the fixed game split) and every half-court possession:
``feats`` (N, 11, T, 8) float32, ``ctx`` (N, T, 5), ``target`` (N, 5, T, 2) feet, ``valid``
(N, T), ``sc_imputed`` (N, T) — exactly what ``ghost.ghost.dataset.build_arrays`` returns — plus
``keys.parquet`` (game_id, possession_id, offence / defence player ids, row order). Arrays are
written game by game into memory-mapped .npy files, so peak memory stays at one game.

These files contain player coordinates: keep them private (never in the repository or a public
bucket; D-019).

Output: <processed_dir>/all/ghost_arrays/<split>/{feats,ctx,target,valid,sc_imputed}.npy,
        keys.parquet, and <processed_dir>/all/ghost_arrays/manifest.json

Usage:
    uv run python scripts/phase3_export_arrays.py [--splits train,val,test]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time

import numpy as np
import polars as pl

from ghost import data as D
from ghost.ghost.dataset import build_arrays
from ghost.tensors import fill_shot_clock, load_game_set

KEYS = ("feats", "ctx", "target", "valid", "sc_imputed")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    out_root = base / "ghost_arrays"
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    manifest = {"nbacore_version": cfg.version, "windows_release": cfg.windows_release,
                "processed_dir": str(cfg.processed_dir), "splits": {}}  # fmt: skip
    for split in args.splits.split(","):
        Ps = P.filter(pl.col("split") == split)
        games = sorted(Ps["game_id"].unique().to_list())
        n_tot = Ps.height
        out = out_root / split
        out.mkdir(parents=True, exist_ok=True)
        t0, row, mm, keys = time.time(), 0, {}, []
        for gi, gid in enumerate(games, 1):
            Pg = Ps.filter(pl.col("game_id") == gid)
            pt = load_game_set(base / "frames", [gid], Pg)
            fill_shot_clock(pt, {gid: D.shot_clock(gid, cfg)})
            a = build_arrays(pt, Pg)
            n = len(a["valid"])
            if not mm:  # allocate once the per-row shapes are known
                for k in KEYS:
                    mm[k] = np.lib.format.open_memmap(
                        out / f"{k}.npy",
                        mode="w+",
                        dtype=a[k].dtype,
                        shape=(n_tot, *a[k].shape[1:]),
                    )
            for k in KEYS:
                mm[k][row : row + n] = a[k]
            keys.append(
                pl.DataFrame(
                    {
                        "row": np.arange(row, row + n),
                        "game_id": pt.game_id.astype(str),
                        "possession_id": pt.possession_id.astype(np.int64),
                        "off_ids": pt.off_ids.astype(np.int64).tolist(),
                        "def_ids": pt.def_ids.astype(np.int64).tolist(),
                    }
                )
            )
            row += n
            if gi % 50 == 0 or gi == len(games):
                print(f"[{split} {gi}/{len(games)}] rows={row} {time.time() - t0:.0f}s", flush=True)
        for k in KEYS:
            mm[k].flush()
        if row != n_tot:  # possessions without frames: truncate the files to the rows written
            for k in KEYS:
                arr = np.load(out / f"{k}.npy", mmap_mode="r")[:row].copy()
                del mm[k]
                gc.collect()  # release the memory map before overwriting (Windows)
                np.save(out / f"{k}.npy", arr)
        K = pl.concat(keys).join(Ps.select(["game_id", "possession_id", "split", "fold", "defense_team_id"]),
                                 on=["game_id", "possession_id"], how="left")  # fmt: skip
        K.write_parquet(out / "keys.parquet")
        manifest["splits"][split] = {
            "n_games": len(games),
            "n_possessions": row,
            "files": {f"{k}.npy": sha256(out / f"{k}.npy") for k in KEYS},
            "bytes": sum((out / f"{k}.npy").stat().st_size for k in KEYS),
        }
        print(json.dumps({split: manifest["splits"][split]["n_possessions"]}), flush=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
