"""How often two team-ghost slots have their most likely positions on top of each other.

Test games, team ghost (all five defenders hidden), final model on CPU. At every valid step each
slot's most likely position is its mixture mode (ghost.ghost.calibration.mixture_mode: the
component mean with the highest mixture density). Reported: share of steps in which at least one
pair of slot modes is within 3 ft (and 2 / 5 ft), share of slot pairs within 3 ft, and the same
for the five real defenders as a reference (double teams, crossing screens).

Output: reports/phase3/<version>_all_<run>_slot_coincidence.json

Usage:
    uv run python scripts/phase3_slot_coincidence.py [--model-run gpu_d128_lr1e-3_final]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from ghost import data as D
from ghost.ghost.calibration import mixture_mode
from ghost.ghost.dataset import apply_mask, load_arrays

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lgd", HERE / "phase4_learned_ghost_dev.py")
lgd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lgd)

RADII = (2.0, 3.0, 5.0)
PAIRS = [(a, b) for a in range(5) for b in range(a + 1, 5)]


def pair_dist(xy: np.ndarray) -> np.ndarray:
    """xy (..., 5, 2) -> (..., 10) pairwise distances."""
    return np.stack([np.linalg.norm(xy[..., a, :] - xy[..., b, :], axis=-1) for a, b in PAIRS], -1)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-run", default="gpu_d128_lr1e-3_final")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    A, K = load_arrays(cfg.processed_dir / "all" / "ghost_arrays", "test")
    T = A["valid"].shape[1]
    model = lgd.load_model(Path("runs/phase3") / f"{cfg.version}_all_{args.model_run}", T, "cpu")
    ghost_d, real_d = [], []
    for a in range(0, len(A["valid"]), 32):
        sl = slice(a, a + 32)
        f = torch.from_numpy(np.ascontiguousarray(A["feats"][sl]))
        x = torch.from_numpy(np.ascontiguousarray(A["ctx"][sl]))
        fm, _ = apply_mask(f, "team")
        o = model(fm, x, None)
        mode = mixture_mode(o["logits"], o["mu"], o["sigma"]).numpy()  # (B,5 slots,T,2)
        v = A["valid"][sl]
        ghost_d.append(pair_dist(mode.transpose(0, 2, 1, 3))[v])  # (steps, 10)
        real_d.append(pair_dist(A["target"][sl].transpose(0, 2, 1, 3))[v])
    G, R = np.concatenate(ghost_d), np.concatenate(real_d)
    rep = {"model_run": args.model_run, "n_steps": int(len(G)), "n_possessions": len(K)}
    for name, Dm in (("team_ghost_slot_modes", G), ("real_defenders", R)):
        rep[name] = {
            f"share_steps_any_pair_within_{r:g}ft": float((Dm < r).any(1).mean()) for r in RADII
        } | {f"share_pairs_within_{r:g}ft": float((Dm < r).mean()) for r in RADII}
        rep[name]["median_min_pair_distance_ft"] = float(np.median(Dm.min(1)))
    p = Path("reports/phase3") / f"{cfg.version}_all_{args.model_run}_slot_coincidence.json"
    p.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
