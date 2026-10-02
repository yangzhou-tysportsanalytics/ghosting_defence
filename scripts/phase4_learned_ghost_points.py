"""Phase 4 task 3 with the learned ghost: expected points conceded relative to the ghost at the
shot (D-006, D-007, D-013, D-015).

For every half-court field-goal attempt with a trusted release, at the release step of the 5 Hz
grid (real defence re-measured on the same grid):
* individual ghost of defender d (only d hidden, the other four real, online): K positions are
  drawn from d's mixture; with d moved there, the defensive features (nearest, second-nearest and
  nearest-in-front defender) and the shooter-aware xFG (M3) are recomputed; the expected value
  over the draws is the ghost's. dEP_d = (xFG_real - E[xFG_ghost_d]) x shot value (> 0: the real
  defender conceded more than his ghost would have).
* team ghost (all five hidden): the five slots are drawn independently (the model gives each
  slot's distribution, not their joint) -> dEP_team.
xFG is cross-fitted over the nbacore folds (each fold scored by a model fitted without it); the
ghost models are the cross-fitted ones (``--crossfit``) or one model (``--model-run``, prototype).

Outputs (no coordinates):
    <processed_dir>/all/analysis/learned_ghost_points_<name>.parquet  per shot, per defender slot
    reports/phase4/<version>_all/learned_ghost_points_<name>.json      league / team summaries

Usage:
    uv run python scripts/phase4_learned_ghost_points.py --crossfit gpu_d64_lr3e-4 --device cuda
    uv run python scripts/phase4_learned_ghost_points.py --model-run v1nlla_g32 --games 3  # CPU
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ghost import data as D
from ghost.court import HOOP_LEFT
from ghost.ghost.dataset import mask_defenders
from ghost.xfg import api
from ghost.xfg.features import FRONT_CONE_RAD, NO_FRONT_FT

_p = Path(__file__).resolve().parent / "phase4_learned_ghost_dev.py"
_spec = importlib.util.spec_from_file_location("lgd", _p)
lgd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lgd)

K_SAMPLES = 100  # D-007: K >= 100 when sampling


def defense_geometry(s: np.ndarray, dfn: np.ndarray) -> tuple[np.ndarray, ...]:
    """Shot location s (2,), defenders dfn (..., 5, 2) -> def1, def2, def_front (...), the same
    definitions as ghost.xfg.features.geometry."""
    v = dfn - s
    dd = np.linalg.norm(v, axis=-1)
    srt = np.sort(dd, axis=-1)
    to_rim = HOOP_LEFT - s
    n = max(float(np.linalg.norm(to_rim)), 1e-6)
    cos = (v @ to_rim) / np.maximum(dd * n, 1e-6)
    front = np.where(cos >= np.cos(FRONT_CONE_RAD), dd, np.inf).min(-1)
    return srt[..., 0], srt[..., 1], np.where(np.isfinite(front), front, NO_FRONT_FT)


def sample_mixture(w, mu, sigma, k: int, rng) -> np.ndarray:
    """K draws from one isotropic Gaussian mixture: w (M,), mu (M, 2), sigma (M,) -> (K, 2)."""
    c = rng.choice(len(w), size=k, p=w / w.sum())
    return mu[c] + rng.normal(size=(k, 2)) * sigma[c, None]


@torch.no_grad()
def mixtures(model, feats, ctx, ids) -> dict[str, dict[str, np.ndarray]]:
    """Team mixture and the five individual mixtures of one possession (batch of 1), each as
    w (5, T, M), mu (5, T, M, 2), sigma (5, T, M); individual: row d = defender d hidden alone."""

    def pack(out):
        return {"w": torch.softmax(out["logits"][0], -1).cpu().numpy(),
                "mu": out["mu"][0].cpu().numpy(), "sigma": out["sigma"][0].cpu().numpy()}  # fmt: skip

    team = pack(model(mask_defenders(feats, torch.ones(1, 5, dtype=torch.bool)), ctx, ids))
    ind = {k: np.empty_like(v) for k, v in team.items()}
    for d in range(5):
        m = torch.zeros(1, 5, dtype=torch.bool)
        m[0, d] = True
        o = pack(model(mask_defenders(feats, m.to(feats.device)), ctx, ids))
        for k in ind:
            ind[k][d] = o[k][d]
    return {"team": team, "ind": ind}


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--crossfit")
    g.add_argument("--model-run")
    ap.add_argument("--arrays-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--games", type=int, default=None, help="first N games per split (CPU test)")
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    root = Path(args.arrays_dir) if args.arrays_dir else base / "ghost_arrays"
    name = f"cf_{args.crossfit}" if args.crossfit else args.model_run
    rng = np.random.default_rng(args.seed)

    shots = pl.read_parquet(base / "analysis" / "shots.parquet")
    S = api.trusted(shots).filter(
        pl.col("possession_id").is_not_null() & ~pl.col("window_is_transition").fill_null(True)
    )
    folds = D.games(cfg).select(["game_id", "fold"])
    S = S.join(folds, on="game_id", how="left")
    P = pl.read_parquet(base / "possessions.parquet").select(
        ["game_id", "possession_id", "t_start"]
    )
    S = S.join(P, on=["game_id", "possession_id"]).with_columns(
        ((pl.col("t_release_ms") - pl.col("t_start")) / 200).round().cast(pl.Int32).alias("k")
    )
    xfg_models = {}  # cross-fitted M3, one per fold
    for f in sorted(S["fold"].drop_nulls().unique().to_list()):
        other = shots.join(folds, on="game_id", how="left").filter(pl.col("fold") != f)
        xfg_models[f] = api.fit(other, other["game_id"].unique().to_list(), "M3")

    T, models, rows, t0 = None, {}, [], time.time()
    for split in ("train", "val", "test"):
        K = pl.read_parquet(root / split / "keys.parquet").sort("row")
        games = sorted(K["game_id"].unique().to_list())[: args.games]
        A = {k: np.load(root / split / f"{k}.npy", mmap_mode="r")
             for k in ("feats", "ctx", "target", "valid")}  # fmt: skip
        T = T or A["valid"].shape[1]
        Sk = S.join(K.select(["game_id", "possession_id", "row", "def_ids", "defense_team_id"]),
                    on=["game_id", "possession_id"]).filter(pl.col("game_id").is_in(games))  # fmt: skip
        for gi, (gid,), Sg in (
            (i, *x) for i, x in enumerate(Sk.group_by(["game_id"], maintain_order=True), 1)
        ):
            fold = int(Sg["fold"][0])
            key = fold if args.crossfit else -1
            if key not in models:
                run = (Path("runs/phase3") / f"{cfg.version}_all_{args.crossfit}_fold{fold}"
                       if args.crossfit else Path("runs/phase3") / f"{cfg.version}_all_{args.model_run}")  # fmt: skip
                models[key] = lgd.load_model(run, T, args.device)
            m = models[key]
            feats_rows, meta = [], []
            for r in Sg.iter_rows(named=True):
                i, k = int(r["row"]), int(r["k"])
                if not (0 <= k < T) or not A["valid"][i, k]:
                    continue
                f = torch.from_numpy(np.array(A["feats"][i : i + 1])).to(args.device)
                x = torch.from_numpy(np.array(A["ctx"][i : i + 1])).to(args.device)
                ids = lgd.ids_for(m, Sg.filter(pl.col("row") == i), args.device)
                mix = mixtures(m, f, x, ids)
                real = np.array(A["target"][i, :, k], dtype=np.float64)  # (5, 2)
                s = np.array([r["x"], r["y"]], dtype=np.float64)
                configs = [real[None]]  # (1, 5, 2) real
                for d in range(5):  # individual ghost of d
                    smp = sample_mixture(mix["ind"]["w"][d, k], mix["ind"]["mu"][d, k],
                                         mix["ind"]["sigma"][d, k], K_SAMPLES, rng)  # fmt: skip
                    c = np.repeat(real[None], K_SAMPLES, 0)
                    c[:, d] = smp
                    configs.append(c)
                team = np.stack([sample_mixture(mix["team"]["w"][j, k], mix["team"]["mu"][j, k],
                                                mix["team"]["sigma"][j, k], K_SAMPLES, rng)
                                 for j in range(5)], 1)  # fmt: skip
                configs.append(team)
                C = np.concatenate(configs)  # (1 + 6K, 5, 2)
                d1, d2, fr = defense_geometry(s, C)
                feats_rows.append(pl.DataFrame({"def1_ft": d1, "def2_ft": d2, "def_front_ft": fr})
                                  .with_columns(pl.lit(len(meta)).alias("shot_ix")))  # fmt: skip
                meta.append(r)
            if not meta:
                continue
            F = pl.concat(feats_rows)
            base_cols = [
                c for c in api.FEATURE_COLUMNS if c not in ("def1_ft", "def2_ft", "def_front_ft")
            ]
            M = pl.DataFrame([{c: r[c] for c in [*base_cols, "shooter_id", "made"]} for r in meta])
            M = M.with_row_index("shot_ix").with_columns(pl.col("shot_ix").cast(pl.Int32))
            X = F.with_columns(pl.col("shot_ix").cast(pl.Int32)).join(M, on="shot_ix", how="left")
            p = api.predict(X, xfg_models[fold]).reshape(len(meta), 1 + 6 * K_SAMPLES)
            for j, r in enumerate(meta):
                val = 3.0 if r["is_three"] else 2.0
                pr = p[j, 0]
                ind = [p[j, 1 + d * K_SAMPLES : 1 + (d + 1) * K_SAMPLES].mean() for d in range(5)]
                tm = p[j, 1 + 5 * K_SAMPLES :].mean()
                ids = list(r["def_ids"])
                for d in range(5):
                    rows.append({"game_id": gid, "possession_id": r["possession_id"],
                                 "event_uid": r["event_uid"], "split": split, "fold": r["fold"],
                                 "parity": r["parity"], "defense_team_id": r["defense_team_id"],
                                 "shooter_id": r["shooter_id"], "made": r["made"],
                                 "is_three": r["is_three"], "def_slot": d, "def_id": int(ids[d]),
                                 "xfg_real": float(pr), "xfg_ind_ghost": float(ind[d]),
                                 "xfg_team_ghost": float(tm),
                                 "dEP_ind": float((pr - ind[d]) * val),
                                 "dEP_team": float((pr - tm) * val)})  # fmt: skip
            if gi % 25 == 0 or gi == len(games):
                print(
                    f"[{split} {gi}/{len(games)}] {time.time() - t0:.0f}s shots={len(rows) // 5}",
                    flush=True,
                )
    G = pl.DataFrame(rows)
    out = base / "analysis" / f"learned_ghost_points_{name}.parquet"
    G.write_parquet(out)
    shot = G.unique(subset=["event_uid"], keep="first")
    team = shot.group_by("defense_team_id").agg(pl.col("dEP_team").mean(), pl.len().alias("n"))
    rep = {
        "ghost": name,
        "k_samples": K_SAMPLES,
        "n_shots": shot.height,
        "mean_xfg_real": float(shot["xfg_real"].mean()),
        "mean_xfg_team_ghost": float(shot["xfg_team_ghost"].mean()),
        "mean_dEP_team_per_shot": float(shot["dEP_team"].mean()),
        "mean_dEP_ind_per_defender_shot": float(G["dEP_ind"].mean()),
        "team_dEP_range": [float(team["dEP_team"].min()), float(team["dEP_team"].max())],
        "note": "team ghost slots drawn independently; xFG M3 cross-fitted by fold",
        "seconds": round(time.time() - t0),
    }
    rdir = Path("reports/phase4") / f"{cfg.version}_all"
    (rdir / f"learned_ghost_points_{name}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
