"""Phase 2 steps 1-2: min-residual baseline and Franks HMM matchups on half-court possessions.

Two stages with bounded memory (the full season does not fit in RAM at once):
1. **Fit**: EM on the TRAIN games of the fixed split (nbacore C-009); if the train set has more
   than ``--fit-max`` possessions, a seeded random subset is used (the models have ~10 parameters).
   Held-out log-likelihood on (a subset of) the VAL games.
2. **Inference**: game by game, posteriors + Viterbi paths for every half-court possession; one
   parquet per game and model; label-free diagnostics accumulated as sums.

Outputs (under <processed_dir>/<game_set>/):
    matchups/<model>/<game_id>.parquet   game_id, possession_id, step, def_slot, def_id, state,
                                         off_id, post_max, p_help, baseline_state
    matchups/<model>_params.json
    reports/phase2/<version>_<game_set>/matchup_fit.json

Usage:
    uv run python scripts/phase2_matchups.py [--game-set tiny|small|all] [--fit-max 20000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.matchup.hmm import (
    FRANKS_GAMMA,
    HMMParams,
    MatchupInputs,
    fit_em,
    loglik,
    min_residual,
    posteriors,
)
from ghost.tensors import PossessionTensors, load_game_set

MODELS = {
    "hmm": dict(stratify=False, help_state=False),
    "hmm_strat": dict(stratify=True, help_state=False),
    "hmm_strat_help": dict(stratify=True, help_state=True),
}
SUM_KEYS = (
    "n_obs",
    "agree",
    "switches",
    "n_poss",
    "double_steps",
    "valid_steps",
    "help",
    "post_max_sum",
)


def _load(
    base: Path, P: pl.DataFrame, game_ids: list[str]
) -> tuple[PossessionTensors, pl.DataFrame]:
    pt = load_game_set(base / "frames", game_ids, P.filter(pl.col("game_id").is_in(game_ids)))
    order = pl.DataFrame(
        {"game_id": pt.game_id, "possession_id": pt.possession_id, "_row": np.arange(len(pt))}
    )
    Pg = P.join(order, on=["game_id", "possession_id"]).sort("_row")
    return pt, Pg


def diag_sums(pt, inp: MatchupInputs, path, post, base, Pg: pl.DataFrame) -> tuple[dict, list]:
    valid = inp.valid[:, None, :]
    s = {k: 0.0 for k in SUM_KEYS}
    s["n_obs"] = float(valid.sum()) * 5
    s["agree"] = float(((path == base) & valid).sum())
    ch = (path[:, :, 1:] != path[:, :, :-1]) & (inp.valid[:, None, 1:] & inp.valid[:, None, :-1])
    s["switches"] = float(ch.sum())
    s["n_poss"] = float(len(pt))
    counts = np.stack([((path == k) & valid).sum(axis=1) for k in range(5)], axis=-1)
    s["double_steps"] = float(((counts >= 2).any(-1) & inp.valid).sum())
    s["valid_steps"] = float(inp.valid.sum())
    s["help"] = float(((path >= 5) & valid).sum())
    s["post_max_sum"] = float((post.max(-1) * valid).sum())
    # at the last step with a ball handler of shot possessions: who guards him?
    has_h = (pt.handler >= 0) & inp.valid
    T = pt.valid.shape[1]
    last = np.where(has_h.any(1), T - 1 - np.argmax(has_h[:, ::-1], axis=1), 0)
    idx = np.arange(len(pt))
    h = np.where(has_h.any(1), pt.handler[idx, last], -1).astype(int)
    is_shot = np.isin(Pg["terminal_type"].to_numpy(), ["fg_made", "fg_missed"]) & (h >= 0)
    shots = []
    for i in idx[is_shot]:
        k, hh = last[i], h[i]
        dist = np.linalg.norm(inp.def_xy[i, k] - inp.off_xy[i, k, hh], axis=-1)
        assigned = path[i, :, k] == hh
        shots.append(
            (
                bool(assigned.any()),
                bool(assigned[dist.argmin()]),
                float(dist[assigned].min()) if assigned.any() else np.nan,
                float(dist.min()),
            )
        )
    return s, shots


def summarize(s: dict, shots: list) -> dict:
    a = np.array(shots, dtype=float) if shots else np.zeros((0, 4))
    has = a[:, 0] > 0 if len(a) else np.zeros(0, bool)
    return {
        "agreement_with_baseline": s["agree"] / max(s["n_obs"], 1),
        "switches_per_possession": s["switches"] / max(s["n_poss"], 1),
        "double_team_step_share": s["double_steps"] / max(s["valid_steps"], 1),
        "help_state_share": s["help"] / max(s["n_obs"], 1),
        "mean_max_posterior": s["post_max_sum"] / max(s["n_obs"], 1),
        "at_shot": {
            "n_shot_possessions": int(len(a)),
            "shooter_has_assigned_defender": float(has.mean()) if len(a) else None,
            "assigned_is_nearest": float(a[:, 1].mean()) if len(a) else None,
            "assigned_dist_ft_median": float(np.nanmedian(a[has, 2])) if has.any() else None,
            "nearest_dist_ft_median": float(np.median(a[:, 3])) if len(a) else None,
        },
    }


def to_table(pt, path, post, base, help_state: bool) -> pl.DataFrame:
    N, Dd, T = path.shape
    off_ids = np.concatenate([pt.off_ids, np.full((N, 1), -1)], axis=1)
    off_id = np.take_along_axis(
        np.repeat(off_ids[:, None, :], Dd, 1).reshape(N * Dd, -1),
        path.reshape(N * Dd, T).astype(np.int64),
        axis=1,
    ).reshape(-1)
    df = pl.DataFrame(
        {
            "game_id": np.repeat(pt.game_id, Dd * T),
            "possession_id": np.repeat(pt.possession_id, Dd * T),
            "step": np.tile(np.arange(T, dtype=np.int16), N * Dd),
            "def_slot": np.tile(np.repeat(np.arange(Dd, dtype=np.int8), T), N),
            "def_id": np.repeat(pt.def_ids.reshape(-1), T),
            "state": path.reshape(-1),
            "off_id": off_id,
            "post_max": post.max(-1).reshape(-1).astype(np.float32),
            "p_help": post[..., 5].reshape(-1).astype(np.float32) if help_state else None,
            "baseline_state": base.reshape(-1),
            "valid": np.repeat(pt.valid[:, None, :], Dd, 1).reshape(-1),
        }
    )
    return df.filter(pl.col("valid")).drop("valid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-set", default=None)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--fit-max", type=int, default=20000, help="max train possessions for EM")
    ap.add_argument("--val-max", type=int, default=5000, help="max val possessions for loglik")
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    cfg = D.DataConfig.load(game_set=args.game_set)
    base = cfg.processed_dir / cfg.game_set
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    def sample(split: str, cap: int) -> pl.DataFrame:
        S = P.filter(pl.col("split") == split)
        if S.height > cap:
            S = S[np.sort(rng.choice(S.height, size=cap, replace=False))]
        return S

    Ptr, Pva = sample("train", args.fit_max), sample("val", args.val_max)
    pt_tr, _ = _load(base, Ptr, sorted(Ptr["game_id"].unique().to_list()))
    inp_tr = MatchupInputs.from_tensors(pt_tr)
    del pt_tr
    inp_va = None
    if Pva.height:
        pt_va, _ = _load(base, Pva, sorted(Pva["game_id"].unique().to_list()))
        inp_va = MatchupInputs.from_tensors(pt_va)
        del pt_va
    n_tr_obs = int(inp_tr.valid.sum()) * 5
    n_va_obs = int(inp_va.valid.sum()) * 5 if inp_va is not None else 0
    print(
        f"{cfg.game_set}: fit on {Ptr.height} train / eval on {Pva.height} val possessions "
        f"(of {P.height} half-court), loaded in {time.time() - t0:.0f}s",
        flush=True,
    )

    out_dir = base / "matchups"
    report = {
        "nbacore_version": cfg.version,
        "game_set": cfg.game_set,
        "n_halfcourt_possessions": P.height,
        "n_fit_train": Ptr.height,
        "n_eval_val": Pva.height,
        "models": {},
    }
    params = {}
    for name in args.models.split(","):
        t1 = time.time()
        fit = fit_em(inp_tr, HMMParams.initial(**MODELS[name]), max_iter=args.max_iter)
        params[name] = fit.params
        report["models"][name] = {
            "params": fit.params.to_dict(),
            "em_iterations": fit.n_iter,
            "converged": fit.converged,
            "train_loglik_per_obs": fit.loglik[-1] / n_tr_obs,
            "val_loglik_per_obs": loglik(inp_va, fit.params) / n_va_obs if n_va_obs else None,
            "fit_seconds": round(time.time() - t1, 1),
        }
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}_params.json").write_text(json.dumps(fit.params.to_dict(), indent=2))
        print(f"fit {name}: {json.dumps(report['models'][name])}", flush=True)
    del inp_tr, inp_va

    # inference game by game
    sums = {n: {k: 0.0 for k in SUM_KEYS} for n in params}
    shots = {n: [] for n in params}
    games = sorted(P["game_id"].unique().to_list())
    t2 = time.time()
    for gi, gid in enumerate(games, 1):
        pt, Pg = _load(base, P, [gid])
        inp = MatchupInputs.from_tensors(pt)
        base_path = min_residual(inp, FRANKS_GAMMA)
        for name, prm in params.items():
            post, path = posteriors(inp, prm)
            to_table(pt, path, post, base_path, prm.help_state).write_parquet(
                out_dir / name / f"{gid}.parquet", compression="zstd"
            )
            s, sh = diag_sums(pt, inp, path, post, base_path, Pg)
            for k in SUM_KEYS:
                sums[name][k] += s[k]
            shots[name].extend(sh)
        if gi % 50 == 0 or gi == len(games):
            print(f"[{gi}/{len(games)}] inference {time.time() - t2:.0f}s", flush=True)
    for name in params:
        report["models"][name].update(summarize(sums[name], shots[name]))
    report["baseline"] = {"gamma": list(FRANKS_GAMMA)}
    report["total_seconds"] = round(time.time() - t0, 1)
    rep_dir = Path("reports/phase2") / f"{cfg.version}_{cfg.game_set}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "matchup_fit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
