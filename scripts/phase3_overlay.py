"""Phase 3 task 6: real defence vs learned-ghost overlay animations for on-ball screens.

Picks likely-real on-ball screens (screen_confidence >= 0.81) in held-out games, half with a
switch and half without (seeded), and renders the whole possession: team ghost (grey circles)
and the screened defender's individual ghost (red dashed circles). Videos show player positions:
they stay local (reports/**/*.mp4 is git-ignored; D-019).

Output: reports/phase3/overlays/<tag>/NN_<game>_<possession>_<switch|noswitch>.mp4 and index.csv

Usage:
    uv run python scripts/phase3_overlay.py --model-run v1nlla_g32 [--n 10] [--split val]
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ghost import data as D
from ghost.ghost.dataset import load_arrays, mask_defenders
from ghost.viz.overlay import animate_ghost

_p = Path(__file__).resolve().parent / "phase4_learned_ghost_dev.py"
_spec = importlib.util.spec_from_file_location("lgd", _p)
lgd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lgd)


@torch.no_grad()
def mixture(model, feats, ctx, masked) -> dict[str, np.ndarray]:
    out = model(mask_defenders(feats, masked), ctx)
    return {
        "w": torch.softmax(out["logits"][0], -1).numpy(),
        "mu": out["mu"][0].numpy(),
        "sigma": out["sigma"][0].numpy(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--min-confidence", type=float, default=0.81)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--arrays-dir", default=None)
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    root = Path(args.arrays_dir) if args.arrays_dir else base / "ghost_arrays"
    keys = pl.read_parquet(root / args.split / "keys.parquet")
    P = pl.read_parquet(base / "possessions.parquet").select(
        ["game_id", "possession_id", "t_start"]
    )
    S = (
        pl.read_parquet(base / "analysis" / "onball_screens.parquet")
        .filter(pl.col("screen_confidence") >= args.min_confidence)
        .join(keys.select(["game_id", "possession_id", "def_ids"]), on=["game_id", "possession_id"])
        .join(P, on=["game_id", "possession_id"])
        .with_columns(
            ((pl.col("t_contact_ms") - pl.col("t_start")) / 200).round().cast(pl.Int32).alias("c")
        )
        .filter(pl.col("c") >= 10)  # at least 2 s of play before the contact
        .sort(["game_id", "possession_id", "t_contact_ms"])
        .unique(subset=["game_id", "possession_id"], keep="first", maintain_order=True)
    )
    rng = np.random.default_rng(args.seed)
    pick = []
    for sw in (True, False):
        g = S.filter(pl.col("switched") == sw)
        pick.append(
            g[rng.choice(g.height, size=min(args.n // 2, g.height), replace=False).tolist()]
        )
    pick = pl.concat(pick)
    A, K = load_arrays(root, args.split, sorted(pick["game_id"].unique().to_list()))
    K = K.with_row_index("i")
    pick = pick.join(K.select(["game_id", "possession_id", "i"]), on=["game_id", "possession_id"])
    model = lgd.load_model(
        Path("runs/phase3") / f"{cfg.version}_all_{args.model_run}", A["valid"].shape[1], "cpu"
    )
    out_dir = Path("reports/phase3/overlays") / args.model_run
    index = []
    for n, r in enumerate(pick.iter_rows(named=True)):
        i = r["i"]
        ids = list(r["def_ids"])
        d = ids.index(r["screened_def_id"]) if r["screened_def_id"] in ids else None
        f, x = torch.from_numpy(A["feats"][i : i + 1]), torch.from_numpy(A["ctx"][i : i + 1])
        team = mixture(model, f, x, torch.ones(1, 5, dtype=torch.bool))
        ind = None
        if d is not None:
            m = torch.zeros(1, 5, dtype=torch.bool)
            m[0, d] = True
            ind = mixture(model, f, x, m)
        kind = "switch" if r["switched"] else "noswitch"
        c = int(r["c"])
        marks = {t: "screen contact" for t in range(c - 1, c + 3)}
        name = f"{n:02d}_{r['game_id']}_{r['possession_id']}_{kind}.mp4"
        title = f"{r['game_id']} possession {r['possession_id']} | on-ball screen, {kind}"
        animate_ghost(A["feats"][i], A["target"][i], A["valid"][i], team, ind, d,
                      out_dir / name, title, marks)  # fmt: skip
        index.append({"file": name, "game_id": r["game_id"], "possession_id": r["possession_id"],
                      "switched": r["switched"], "contact_step": c,
                      "screen_confidence": r["screen_confidence"]})  # fmt: skip
        print(name, flush=True)
    pl.DataFrame(index).write_csv(out_dir / "index.csv")


if __name__ == "__main__":
    main()
