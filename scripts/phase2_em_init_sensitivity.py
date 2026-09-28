"""Initialisation sensitivity of the matchup-HMM EM (does it recover the published weights?).

The production fit starts at the published (man, ball, rim) weights (0.62, 0.11, 0.27) and
converges in a few iterations, so "recovering" them needs a check from other starting points and
with a tighter convergence criterion. Fits the pooled model ``hmm`` (the Table 1 headline row) from
several initial weight vectors on a seeded subset of train possessions.

Output: reports/phase2/<version>_<game_set>/em_init_sensitivity.json

Usage:
    uv run python scripts/phase2_em_init_sensitivity.py [--fit-max 5000] [--tol 1e-8]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.matchup.hmm import FRANKS_GAMMA, HMMParams, MatchupInputs, fit_em
from ghost.tensors import load_game_set

INITS = {
    "published": FRANKS_GAMMA,
    "uniform": (1 / 3, 1 / 3, 1 / 3),
    "man_heavy": (0.90, 0.05, 0.05),
    "rim_heavy": (0.30, 0.10, 0.60),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-set", default=None)
    ap.add_argument("--fit-max", type=int, default=5000)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set=args.game_set)
    base = cfg.processed_dir / cfg.game_set
    P = pl.read_parquet(base / "possessions.parquet").filter(
        ~pl.col("is_transition") & (pl.col("split") == "train")
    )
    rng = np.random.default_rng(args.seed)
    if P.height > args.fit_max:
        P = P[np.sort(rng.choice(P.height, size=args.fit_max, replace=False))]
    pt = load_game_set(base / "frames", sorted(P["game_id"].unique().to_list()), P)
    inp = MatchupInputs.from_tensors(pt)
    n_obs = int(inp.valid.sum()) * 5
    del pt
    rep = {
        "nbacore_version": cfg.version,
        "model": "hmm (pooled, no help state)",
        "n_train_possessions": P.height,
        "tol": args.tol,
        "fits": {},
    }
    for name, g in INITS.items():
        t0 = time.time()
        p0 = HMMParams.initial()
        p0.gamma = {"all": np.array(g, dtype=np.float64)}
        fit = fit_em(inp, p0, max_iter=args.max_iter, tol=args.tol)
        rep["fits"][name] = {
            "init": list(g),
            "gamma": [round(float(x), 4) for x in fit.params.gamma["all"]],
            "sigma2": float(fit.params.sigma2["all"]),
            "rho": float(fit.params.rho),
            "loglik_per_obs": fit.loglik[-1] / n_obs,
            "iterations": fit.n_iter,
            "converged": fit.converged,
            "seconds": round(time.time() - t0, 1),
        }
        print(name, json.dumps(rep["fits"][name]), flush=True)
    G = np.array([f["gamma"] for f in rep["fits"].values()])
    rep["max_abs_spread_across_inits"] = [float(x) for x in G.max(0) - G.min(0)]
    out = Path("reports/phase2") / f"{cfg.version}_{cfg.game_set}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "em_init_sensitivity.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
