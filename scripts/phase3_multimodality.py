"""Phase 3 task 5: is the learned ghost multimodal where it should be (switch or stay at a screen)?

For on-ball screens in held-out games, the screened defender alone is hidden (individual ghost,
online) and his mixture is read for 1.4 s after screen contact. Each component with weight
>= W_MIN is attached to the attacker whose rule position is nearest to its mean. A step is
*bimodal* when components of at least two different attackers carry weight >= W_MIN each and
their means are >= SEP_FT apart. Compared across switched screens, screens without a switch and
steps of the same possessions away from any screen (>= 2 s before or >= 3 s after a contact).
Also: weight the ghost gives, at the end of the window, to the attacker the real defender guards
then (nearest rule position), and how the mean (D-007: not the ghost) lands between the modes.

Output: reports/phase3/<version>_all_<tag>_multimodality.json (aggregates only)

Usage:
    uv run python scripts/phase3_multimodality.py --model-run v1nlla_g32 [--split val] [--games N]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ghost import data as D
from ghost.ghost.dataset import load_arrays, mask_defenders

_p = Path(__file__).resolve().parent / "phase4_learned_ghost_dev.py"
_spec = importlib.util.spec_from_file_location("lgd", _p)
lgd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lgd)

W_MIN, SEP_FT, WINDOW = 0.15, 5.0, 7  # weight, mode separation, steps after contact (1.4 s)


@torch.no_grad()
def modes(model, feats, ctx, d: int) -> dict[str, np.ndarray]:
    """Individual ghost of defender slot d for a batch: weights (B,T,M), means (B,T,M,2), the
    attacker attached to each component (B,T,M) and the attackers' rule positions (B,5,T,2)."""
    m = torch.zeros(feats.shape[0], 5, dtype=torch.bool)
    m[:, d] = True
    out = model(mask_defenders(feats, m), ctx)
    anchors = model.rule_anchors(feats)  # (B, 5k, T, 2)
    mu = out["mu"][:, d]  # (B, T, M, 2)
    w = torch.softmax(out["logits"][:, d], -1)
    dist = torch.linalg.vector_norm(
        mu[:, :, :, None] - anchors.permute(0, 2, 1, 3)[:, :, None], dim=-1
    )
    return {
        "w": w.numpy(),
        "mu": mu.numpy(),
        "att": dist.argmin(-1).numpy(),  # (B, T, M)
        "anchors": anchors.numpy(),
    }


def bimodal(w: np.ndarray, mu: np.ndarray, att: np.ndarray) -> bool:
    """One step: >= 2 attackers with weight >= W_MIN each, means >= SEP_FT apart."""
    tot: dict[int, float] = {}
    for k, a in enumerate(att):
        tot[int(a)] = tot.get(int(a), 0.0) + float(w[k])
    big = [a for a, v in tot.items() if v >= W_MIN]
    if len(big) < 2:
        return False
    # weighted mean position of each attacker's components
    cen = {a: (w[att == a, None] * mu[att == a]).sum(0) / w[att == a].sum() for a in big}
    return max(np.linalg.norm(cen[a] - cen[b]) for a in big for b in big if a < b) >= SEP_FT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--arrays-dir", default=None)
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    root = Path(args.arrays_dir) if args.arrays_dir else base / "ghost_arrays"
    keys = pl.read_parquet(root / args.split / "keys.parquet")
    games = sorted(keys["game_id"].unique().to_list())[: args.games]
    A, K = load_arrays(root, args.split, games)
    K = K.with_row_index("i")
    P = pl.read_parquet(base / "possessions.parquet").select(
        ["game_id", "possession_id", "t_start"]
    )
    S = (
        pl.read_parquet(base / "analysis" / "onball_screens.parquet")
        .filter(pl.col("game_id").is_in(games))
        .join(
            K.select(["game_id", "possession_id", "i", "def_ids"]), on=["game_id", "possession_id"]
        )
        .join(P, on=["game_id", "possession_id"])
        .with_columns(
            ((pl.col("t_contact_ms") - pl.col("t_start")) / 200).round().cast(pl.Int32).alias("c")
        )
    )
    T = A["valid"].shape[1]
    model = lgd.load_model(Path("runs/phase3") / f"{cfg.version}_all_{args.model_run}", T, "cpu")

    res = {"switch": [], "no_switch": [], "away": []}
    p_real, mean_gap = {"switch": [], "no_switch": []}, []
    contacts: dict[int, list[int]] = {}
    for r in S.iter_rows(named=True):
        contacts.setdefault(r["i"], []).append(r["c"])
    for r in S.iter_rows(named=True):
        ids = list(r["def_ids"])
        if r["screened_def_id"] not in ids:
            continue
        d, i, c = ids.index(r["screened_def_id"]), r["i"], r["c"]
        steps = [t for t in range(c, c + WINDOW + 1) if 0 <= t < T and A["valid"][i, t]]
        if len(steps) < 3:
            continue
        f = torch.from_numpy(A["feats"][i : i + 1])
        x = torch.from_numpy(A["ctx"][i : i + 1])
        mo = modes(model, f, x, d)
        kind = "switch" if r["switched"] else "no_switch"
        res[kind].append(
            np.mean([bimodal(mo["w"][0, t], mo["mu"][0, t], mo["att"][0, t]) for t in steps])
        )
        # end of window: weight on the attacker the real defender guards then; mean vs modes
        t = steps[-1]
        z = A["target"][i, d, t]
        man = int(np.linalg.norm(mo["anchors"][0, :, t] - z, axis=-1).argmin())
        p_real[kind].append(float(mo["w"][0, t][mo["att"][0, t] == man].sum()))
        if bimodal(mo["w"][0, t], mo["mu"][0, t], mo["att"][0, t]):
            mean = (mo["w"][0, t][:, None] * mo["mu"][0, t]).sum(0)
            near = np.linalg.norm(mo["mu"][0, t] - mean, axis=-1).min()
            mean_gap.append(float(near))
        # away from screens: steps >= 2 s before or >= 3 s after every contact of the possession
        cs = contacts[i]
        away = [
            t for t in range(T) if A["valid"][i, t] and all(t < cc - 10 or t > cc + 15 for cc in cs)
        ]
        if away:
            res["away"].append(
                np.mean([bimodal(mo["w"][0, t], mo["mu"][0, t], mo["att"][0, t]) for t in away])
            )
    rep = {
        "model_run": args.model_run,
        "split": args.split,
        "n_games": len(games),
        "definition": {"w_min": W_MIN, "sep_ft": SEP_FT, "window_steps_after_contact": WINDOW},
        "share_bimodal_steps": {k: (float(np.mean(v)) if v else None) for k, v in res.items()},
        "n": {k: len(v) for k, v in res.items()},
        "weight_on_real_man_end_of_window": {
            k: (float(np.mean(v)) if v else None) for k, v in p_real.items()
        },
        "bimodal_steps_mean_to_nearest_component_ft": float(np.median(mean_gap))
        if mean_gap
        else None,
        "note": "screened defender's individual ghost; 'away' uses the same defender's ghost at "
        "steps far from any on-ball screen of the possession",
    }
    out = Path("reports/phase3") / f"{cfg.version}_all_{args.model_run}_multimodality.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
