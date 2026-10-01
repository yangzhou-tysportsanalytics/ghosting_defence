"""Phase 4 task 1: deviations of every defender from the learned ghost (D-006, D-007, D-013, D-015).

For each half-court possession (packed arrays of phase3_export_arrays.py), with the online model:
* team ghost (all five defenders hidden): identity-free NLL of each real defender, -log of the
  equal-weight mixture over the five predicted slots (``nll_team``);
* individual ghost (only defender d hidden, the other four real defenders visible up to t):
  NLL of d under his own slot's mixture (``nll_ind``), distance to the mixture mean (``d_mean``)
  and to the nearest component mean with weight >= 0.1 (``d_mode``, the "nearest mode").

Cross-fitting (``--crossfit <config tag>``): the games of fold k are scored by the model trained
without fold k (runs <tag>_fold<k>, phase3_gpu.sh crossfit), so no deviation is in-sample.
``--model-run <tag>`` scores everything with one model (prototype only: train games in-sample).

Outputs (no coordinates):
    <processed_dir>/all/learned_ghost/<name>/<game_id>.parquet   per (possession, defender, step)
    <processed_dir>/all/analysis/learned_ghost_dev_<name>.parquet per (possession, defender) means
    reports/phase4/<version>_all/learned_ghost_dev_<name>.json    league summaries

Usage:
    uv run python scripts/phase4_learned_ghost_dev.py --crossfit gpu_d64_lr3e-4 --device cuda
    uv run python scripts/phase4_learned_ghost_dev.py --model-run v1nlla_g32 --games 8   # CPU
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ghost import data as D
from ghost.ghost.dataset import mask_defenders
from ghost.ghost.model import GhostConfig, GhostModel, mixture_mean, mixture_nll

MODE_MIN_WEIGHT = 0.1


def load_model(run: Path, n_steps: int, device: str) -> GhostModel:
    cfg = json.loads((run / "eval.json").read_text())["config"]["model"]
    cfg["gamma"] = tuple(cfg["gamma"])
    m = GhostModel(GhostConfig(**cfg), n_steps=n_steps).to(device)
    m.load_state_dict(torch.load(run / "model.pt", map_location=device))
    vp = run / "vocab.json"  # lineup / scheme ghosts
    m.vocab = json.loads(vp.read_text()) if vp.exists() else None
    return m.eval()


def ids_for(model: GhostModel, keys: pl.DataFrame, device: str) -> dict | None:
    """Identity indices of the rows in ``keys`` for a conditioned model (0 = unknown)."""
    v = getattr(model, "vocab", None)
    if v is None or model.cfg.condition == "league":
        return None
    players = [[v["players"].get(str(p), 0) for p in ids] for ids in keys["def_ids"].to_list()]
    teams = [v["teams"].get(str(t), 0) for t in keys["defense_team_id"].to_list()]
    return {"players": torch.tensor(players, device=device),
            "team": torch.tensor(teams, device=device)}  # fmt: skip


@torch.no_grad()
def score(model: GhostModel, b: dict, ids: dict | None = None) -> dict[str, torch.Tensor]:
    """Per (possession, defender, step) metrics for one batch of tensors."""
    feats, ctx, z = b["feats"], b["ctx"], b["target"]
    B = feats.shape[0]
    team = torch.ones(B, 5, dtype=torch.bool, device=feats.device)
    cost = mixture_nll(model(mask_defenders(feats, team), ctx, ids), z)  # (B, 5 slots, 5 def, T)
    nll_team = -(torch.logsumexp(-cost, dim=1) - math.log(5.0))  # (B, 5, T)
    nll_ind, d_mean, d_mode = (torch.empty_like(nll_team) for _ in range(3))
    for d in range(5):
        one = torch.zeros(B, 5, dtype=torch.bool, device=feats.device)
        one[:, d] = True
        out = model(mask_defenders(feats, one), ctx, ids)
        nll_ind[:, d] = mixture_nll(out, z)[:, d, d]
        zd = z[:, d]  # (B, T, 2)
        d_mean[:, d] = torch.linalg.vector_norm(mixture_mean(out)[:, d] - zd, dim=-1)
        w = torch.softmax(out["logits"][:, d], -1)  # (B, T, M)
        dist = torch.linalg.vector_norm(out["mu"][:, d] - zd[:, :, None], dim=-1)  # (B, T, M)
        dist = torch.where(w >= MODE_MIN_WEIGHT, dist, torch.full_like(dist, float("inf")))
        d_mode[:, d] = dist.amin(-1)
    return {"nll_team": nll_team, "nll_ind": nll_ind, "d_mean": d_mean, "d_mode": d_mode}


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--crossfit", help="config tag; runs <tag>_fold0 .. <tag>_fold4")
    g.add_argument("--model-run", help="one run tag (prototype; train games in-sample)")
    ap.add_argument("--arrays-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--games", type=int, default=None, help="first N games per split (CPU test)")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    root = Path(args.arrays_dir) if args.arrays_dir else base / "ghost_arrays"
    runs = Path("runs/phase3")
    name = f"cf_{args.crossfit}" if args.crossfit else args.model_run
    step_dir = base / "learned_ghost" / name
    step_dir.mkdir(parents=True, exist_ok=True)

    T = None
    models: dict = {}
    agg, t0 = [], time.time()
    for split in ("train", "val", "test"):
        K = pl.read_parquet(root / split / "keys.parquet").sort("row")
        games = sorted(K["game_id"].unique().to_list())[: args.games]
        A = {k: np.load(root / split / f"{k}.npy", mmap_mode="r")
             for k in ("feats", "ctx", "target", "valid")}  # fmt: skip
        T = T or A["valid"].shape[1]
        for gi, gid in enumerate(games, 1):
            Kg = K.filter(pl.col("game_id") == gid)
            rows = Kg["row"].to_numpy()
            fold = int(Kg["fold"][0])
            key = fold if args.crossfit else -1
            if key not in models:
                run = runs / f"{cfg.version}_all_{args.crossfit}_fold{fold}" if args.crossfit \
                    else runs / f"{cfg.version}_all_{args.model_run}"  # fmt: skip
                models[key] = load_model(run, T, args.device)
            m = models[key]
            res = {k: [] for k in ("nll_team", "nll_ind", "d_mean", "d_mode")}
            for a in range(0, len(rows), args.batch_size):
                r = rows[a : a + args.batch_size]
                b = {k: torch.from_numpy(np.array(A[k][r])).to(args.device)
                     for k in ("feats", "ctx", "target")}  # fmt: skip
                ids = ids_for(m, Kg[a : a + args.batch_size], args.device)
                for k, v in score(m, b, ids).items():
                    res[k].append(v.float().cpu().numpy())
            res = {k: np.concatenate(v) for k, v in res.items()}  # (n, 5, T)
            valid = np.array(A["valid"][rows])  # (n, T)
            n = len(rows)
            ii, dd, tt = np.nonzero(np.broadcast_to(valid[:, None, :], (n, 5, T)))
            def_ids = np.array(Kg["def_ids"].to_list())  # (n, 5)
            steps = pl.DataFrame(
                {
                    "game_id": gid,
                    "possession_id": Kg["possession_id"].to_numpy()[ii],
                    "def_slot": dd.astype(np.int8),
                    "def_id": def_ids[ii, dd],
                    "step": tt.astype(np.int16),
                    **{k: v[ii, dd, tt].astype(np.float32) for k, v in res.items()},
                }
            )
            steps.write_parquet(step_dir / f"{gid}.parquet")
            agg.append(
                steps.group_by(["game_id", "possession_id", "def_id"]).agg(
                    *[pl.col(k).mean() for k in res], pl.len().alias("n_steps_ghost")
                )
            )
            if gi % 50 == 0 or gi == len(games):
                print(f"[{split} {gi}/{len(games)}] {time.time() - t0:.0f}s", flush=True)
    dev = pl.concat(agg).with_columns(pl.lit(name).alias("ghost"))
    out = base / "analysis" / f"learned_ghost_dev_{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    dev.write_parquet(out)
    rep = {
        "ghost": name,
        "cross_fitted": bool(args.crossfit),
        "n_possession_defender": dev.height,
        "league_mean": {
            k: float(dev[k].mean()) for k in ("nll_team", "nll_ind", "d_mean", "d_mode")
        },
        "corr_possession_defender": {
            f"{a}__{b}": float(np.corrcoef(dev[a], dev[b])[0, 1])
            for a, b in (("nll_ind", "nll_team"), ("nll_ind", "d_mean"), ("d_mean", "d_mode"))
        },
        "seconds": round(time.time() - t0),
    }
    rdir = Path("reports/phase4") / f"{cfg.version}_all"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"learned_ghost_dev_{name}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
