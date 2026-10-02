"""Calibration of the learned ghost by game situation (D-007 check behind the shot-value sign).

Each 5 Hz step of a held-out possession gets one context, by priority:
  release  the last 0.4 s up to the shot release (field-goal attempts with a trusted release)
  screen   0 to 1.4 s after an on-ball screen contact
  early    the first 2 s after the midcourt crossing
  other    the rest
Per context: highest-density coverage at 0.5 / 0.8 / 0.9 / 0.95 and mean NLL, for the individual
ghost (each defender hidden alone, his own slot) and the identity-free team ghost (all hidden;
equal-weight mixture over the five slots). At the release step: the share of individual-ghost
draws farther from the shooter than the real defender, over all defenders and for the defender
whose ghost is nearest to the shooter (0.5 if calibrated; > 0.5 = real defenders closer than the
ghost expects). Defenders are never selected by their real position (selection bias).

Output (aggregates only): reports/phase3/<version>_all_<tag>_calibration_by_context.json

Usage:
    uv run python scripts/phase3_calibration_by_context.py --model-run v1nlla_g32 --games 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ghost import data as D
from ghost.ghost.calibration import coverage_curve, hpd_pit, mixture_log_density, sample_mixture
from ghost.ghost.dataset import load_arrays, mask_defenders
from ghost.xfg import api

_p = Path(__file__).resolve().parent / "phase4_learned_ghost_dev.py"
_spec = importlib.util.spec_from_file_location("lgd", _p)
lgd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lgd)

CONTEXTS = ("release", "screen", "early", "other")
LEVELS = (0.5, 0.8, 0.9, 0.95)


def step_contexts(n: int, release: int | None, contacts: list[int]) -> np.ndarray:
    """(n,) context index per step, by priority release > screen > early > other."""
    c = np.full(n, CONTEXTS.index("other"))
    c[: min(10, n)] = CONTEXTS.index("early")
    for s in contacts:
        c[max(0, s) : max(0, min(n, s + 8))] = CONTEXTS.index("screen")
    if release is not None and 0 <= release < n:
        c[max(0, release - 2) : release + 1] = CONTEXTS.index("release")
    return c


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    root = base / "ghost_arrays"
    games = sorted(
        pl.read_parquet(root / args.split / "keys.parquet")["game_id"].unique().to_list()
    )
    games = games[: args.games]
    A, K = load_arrays(root, args.split, games)
    K = K.with_row_index("i")
    P = pl.read_parquet(base / "possessions.parquet").select(
        ["game_id", "possession_id", "t_start"]
    )
    K = K.join(P, on=["game_id", "possession_id"], how="left")
    shots = api.trusted(pl.read_parquet(base / "analysis" / "shots.parquet")).filter(
        pl.col("game_id").is_in(games) & pl.col("possession_id").is_not_null()
    )
    rel = {(r["game_id"], r["possession_id"]): (r["t_release_ms"], r["shooter_id"])
           for r in shots.iter_rows(named=True)}  # fmt: skip
    scr = pl.read_parquet(base / "analysis" / "onball_screens.parquet").filter(
        pl.col("game_id").is_in(games)
    )
    contacts: dict = {}
    for r in scr.iter_rows(named=True):
        contacts.setdefault((r["game_id"], r["possession_id"]), []).append(r["t_contact_ms"])
    T = A["valid"].shape[1]
    model = lgd.load_model(
        Path("runs/phase3") / f"{cfg.version}_all_{args.model_run}", T, args.device
    )
    g = torch.Generator().manual_seed(0)
    u_ind, u_team, nll_ind, nll_team, ctx_ind, ctx_team, farther = [], [], [], [], [], [], []
    for r in K.iter_rows(named=True):
        i = r["i"]
        n = int(A["valid"][i].sum())
        key = (r["game_id"], r["possession_id"])
        k_rel, shooter = None, None
        if key in rel:
            k_rel = int(round((rel[key][0] - r["t_start"]) / 200))
            shooter = rel[key][1]
        cs = [int(round((t - r["t_start"]) / 200)) for t in contacts.get(key, [])]
        cx = step_contexts(n, k_rel if k_rel is not None and k_rel < n else None, cs)
        f = torch.from_numpy(np.array(A["feats"][i : i + 1]))
        x = torch.from_numpy(np.array(A["ctx"][i : i + 1]))
        z = torch.from_numpy(np.array(A["target"][i, :, :n]))  # (5, n, 2)
        ids = lgd.ids_for(model, K.filter(pl.col("i") == i), args.device)
        # team ghost: equal-weight mixture over the five slots, evaluated for every defender
        out = model(mask_defenders(f, torch.ones(1, 5, dtype=torch.bool)), x, ids)
        lg = out["logits"][0, :, :n]  # (5 slots, n, M)
        M = lg.shape[-1]
        lw = (torch.log_softmax(lg, -1) - math.log(5.0)).permute(1, 0, 2).reshape(n, 5 * M)
        mu = out["mu"][0, :, :n].permute(1, 0, 2, 3).reshape(n, 5 * M, 2)
        sg = out["sigma"][0, :, :n].permute(1, 0, 2).reshape(n, 5 * M)
        for d in range(5):
            u_team.append(hpd_pit(lw, mu, sg, z[d], args.n_samples, g))
            nll_team.append(-mixture_log_density(lw, mu, sg, z[d][:, None])[:, 0])
            ctx_team.append(cx)
        for d in range(5):  # individual ghost of d
            m = torch.zeros(1, 5, dtype=torch.bool)
            m[0, d] = True
            o = model(mask_defenders(f, m), x, ids)
            lg_d, mu_d, sg_d = o["logits"][0, d, :n], o["mu"][0, d, :n], o["sigma"][0, d, :n]
            u_ind.append(hpd_pit(lg_d, mu_d, sg_d, z[d], args.n_samples, g))
            nll_ind.append(-mixture_log_density(lg_d, mu_d, sg_d, z[d][:, None])[:, 0])
            ctx_ind.append(cx)
            if k_rel is not None and k_rel < n and shooter in r["off_ids"]:
                # no selection on the real positions (choosing the defender who is really
                # nearest would bias the comparison: the minimum of five is small by selection)
                s = int(r["off_ids"].index(shooter))
                sxy = torch.tensor([(A["feats"][i, 1 + s, k_rel, 0] + 0.5) * 47.0,
                                    (A["feats"][i, 1 + s, k_rel, 1] + 0.5) * 50.0])  # fmt: skip
                real = float(torch.linalg.vector_norm(z[d, k_rel] - sxy))
                smp = sample_mixture(lg_d[k_rel], mu_d[k_rel], sg_d[k_rel], 512, g)
                dist = torch.linalg.vector_norm(smp - sxy, dim=-1)
                # ghost-side distance used to pick "the" defender per shot: expected ghost distance
                farther.append(
                    (
                        f"{key[0]}:{key[1]}",
                        d,
                        float((dist > real).float().mean()),
                        float(dist.mean()),
                    )
                )
    rep = {"model_run": args.model_run, "split": args.split, "n_games": len(games),
           "n_possessions": K.height, "contexts": {}}  # fmt: skip
    for name, us, ls, cs in (("individual", u_ind, nll_ind, ctx_ind),
                             ("team", u_team, nll_team, ctx_team)):  # fmt: skip
        u, nl, c = torch.cat(us), torch.cat(ls), np.concatenate(cs)
        rep["contexts"][name] = {}
        for j, cname in enumerate(CONTEXTS):
            sel = torch.from_numpy(c == j)
            if sel.sum() == 0:
                continue
            rep["contexts"][name][cname] = {
                "n_defender_steps": int(sel.sum()),
                "coverage": coverage_curve(u[sel], LEVELS),
                "mean_nll": float(nl[sel].mean()),
            }
    F = pl.DataFrame(farther, schema=["key", "d", "share", "ghost_dist"], orient="row")
    if F.height:
        by_ghost = F.sort("ghost_dist").group_by("key", maintain_order=True).first()
    rep["release_distance_to_shooter"] = {
        "all_defenders": {"n": F.height,
                          "share_ghost_draws_farther": float(F["share"].mean()) if F.height else None},
        "defender_nearest_by_ghost": {
            "n_shots": by_ghost.height if F.height else 0,
            "share_ghost_draws_farther": float(by_ghost["share"].mean()) if F.height else None,
        },
        "note": "0.5 if calibrated; > 0.5: real defenders are closer to the shooter than the ghost "
        "expects. The defender is chosen by his ghost, never by his real position (selecting "
        "the really nearest of five would bias the comparison by construction).",
    }  # fmt: skip
    out_p = (
        Path("reports/phase3") / f"{cfg.version}_all_{args.model_run}_calibration_by_context.json"
    )
    out_p.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
