"""Departure vs the spread of the ghost's map (SSAC final review R4 P1, 2026-11-02).

Departure (nll_ind) is the negative log-density of the real position under the individual
ghost. A spread-out map gives every position a low density, so departure partly measures how
spread the map is. For every defender-step, with the same cross-fitted models as
phase4_learned_ghost_dev.py (each fold scored by the model trained without it):
* ``nll``: -log f(z) at the real position (equals nll_ind of phase4_learned_ghost_dev.py);
* ``entropy``: the map's own expected NLL, E_f[-log f(Z)], by Monte Carlo (``--draws``);
* ``pit``: the coverage level of the smallest highest-density region containing the real
  position, u = P_f(f(Z) >= f(z)) (the quantity used for calibration);
* ``excess`` = nll - entropy: departure with the spread removed.

Outputs (no coordinates):
    <processed_dir>/all/learned_ghost/<name>_spread/<game_id>.parquet   per defender-step
    with --variants: learned_ghost/<name>_excess and <name>_pit (per-step files in the layout of
    phase4_learned_ghost_dev.py, with nll_ind replaced by excess or pit; the other metrics copied
    from <name>) and analysis/learned_ghost_dev_<name>_{excess,pit}.parquet, so the existing
    reliability, meta-metric, hierarchical and predictive scripts run on them with
    --ghost learned:<name>_excess (or _pit).

Usage:
    uv run python scripts/phase4_learned_ghost_spread.py --crossfit gpu_d128_lr1e-3 --resume
    uv run python scripts/phase4_learned_ghost_spread.py --crossfit gpu_d128_lr1e-3 --variants
    uv run python scripts/phase4_learned_ghost_spread.py --crossfit gpu_d128_lr1e-3 --games 2  # test
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ghost import data as D  # noqa: E402
from ghost.ghost.dataset import mask_defenders  # noqa: E402
from phase4_learned_ghost_dev import _per_defender, ids_for, load_model  # noqa: E402

SPLITS = ("train", "val", "test")


def log_density(logits: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """log f(x) of an isotropic Gaussian mixture. logits, sigma (..., M); mu (..., M, 2);
    x (..., K, 2) -> (..., K)."""
    lw = torch.log_softmax(logits, -1)[..., None, :]  # (..., 1, M)
    d2 = ((x[..., :, None, :] - mu[..., None, :, :]) ** 2).sum(-1)  # (..., K, M)
    s = sigma[..., None, :]
    logn = -0.5 * d2 / s**2 - 2 * torch.log(s) - math.log(2 * math.pi)
    return torch.logsumexp(lw + logn, -1)


@torch.no_grad()
def spread(model, b: dict, ids, draws: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    """nll, entropy and pit per (possession, defender, step) for the individual ghost."""
    feats, ctx, z = b["feats"], b["ctx"], b["target"]
    B = feats.shape[0]
    out = {k: torch.empty(B, 5, z.shape[2]) for k in ("nll", "entropy", "pit")}
    for d in range(5):
        one = torch.zeros(B, 5, dtype=torch.bool)
        one[:, d] = True
        o = model(mask_defenders(feats, one), ctx, ids)
        lg, mu, sg = o["logits"][:, d], o["mu"][:, d], o["sigma"][:, d]  # (B,T,M), (B,T,M,2)
        lf_real = log_density(lg, mu, sg, z[:, d][:, :, None, :])[..., 0]  # (B, T)
        comp = torch.multinomial(torch.softmax(lg, -1).reshape(-1, lg.shape[-1]), draws,
                                 replacement=True, generator=gen).reshape(*lg.shape[:2], draws)  # fmt: skip
        m_k = torch.gather(mu, 2, comp[..., None].expand(-1, -1, -1, 2))  # (B, T, K, 2)
        s_k = torch.gather(sg, 2, comp)  # (B, T, K)
        eps = torch.randn(m_k.shape, generator=gen)
        lf_draw = log_density(lg, mu, sg, m_k + s_k[..., None] * eps)  # (B, T, K)
        out["nll"][:, d] = -lf_real
        out["entropy"][:, d] = -lf_draw.mean(-1)
        out["pit"][:, d] = (lf_draw >= lf_real[..., None]).float().mean(-1)
    return out


def score_games(args, base: Path, name: str) -> None:
    root = base / "ghost_arrays"
    out_dir = base / "learned_ghost" / f"{name}_spread"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = Path("runs/phase3")
    cfg = D.DataConfig.load(game_set="all")
    models: dict = {}
    t0, n_done = time.time(), 0
    for split in SPLITS:
        K = pl.read_parquet(root / split / "keys.parquet").sort("row")
        games = sorted(K["game_id"].unique().to_list())[: args.games]
        A = {k: np.load(root / split / f"{k}.npy", mmap_mode="r")
             for k in ("feats", "ctx", "target", "valid")}  # fmt: skip
        T = A["valid"].shape[1]
        for gi, gid in enumerate(games, 1):
            done = out_dir / f"{gid}.parquet"
            if args.resume and done.exists():
                continue
            Kg = K.filter(pl.col("game_id") == gid)
            rows = Kg["row"].to_numpy()
            fold = int(Kg["fold"][0])
            if fold not in models:
                models[fold] = load_model(runs / f"{cfg.version}_all_{args.crossfit}_fold{fold}", T,
                                          "cpu")  # fmt: skip
            m = models[fold]
            gen = torch.Generator().manual_seed(int(gid[-6:]))
            res = {k: [] for k in ("nll", "entropy", "pit")}
            for a in range(0, len(rows), args.batch_size):
                r = rows[a : a + args.batch_size]
                b = {k: torch.from_numpy(np.array(A[k][r])) for k in ("feats", "ctx", "target")}
                for k, v in spread(m, b, ids_for(m, Kg[a : a + args.batch_size], "cpu"),
                                   args.draws, gen).items():  # fmt: skip
                    res[k].append(v.numpy())
            res = {k: np.concatenate(v) for k, v in res.items()}
            valid = np.array(A["valid"][rows])
            n = len(rows)
            ii, dd, tt = np.nonzero(np.broadcast_to(valid[:, None, :], (n, 5, T)))
            def_ids = np.array(Kg["def_ids"].to_list())
            pl.DataFrame({
                "game_id": gid, "possession_id": Kg["possession_id"].to_numpy()[ii],
                "def_slot": dd.astype(np.int8), "def_id": def_ids[ii, dd],
                "step": tt.astype(np.int16),
                **{k: v[ii, dd, tt].astype(np.float32) for k, v in res.items()},
            }).write_parquet(out_dir / f"{gid}.parquet.tmp")  # fmt: skip
            (out_dir / f"{gid}.parquet.tmp").replace(done)
            n_done += 1
            if n_done % 25 == 0:
                print(f"[{split} {gi}/{len(games)}] {n_done} games {time.time() - t0:.0f}s", flush=True)
    print(f"scored {n_done} games in {time.time() - t0:.0f}s", flush=True)


def variants(base: Path, name: str) -> None:
    """Per-step files and per-defender means with nll_ind replaced by excess or pit."""
    src = base / "learned_ghost" / f"{name}_spread"
    keys = ["game_id", "possession_id", "def_slot", "step"]
    agg = {"excess": [], "pit": []}
    for v in agg:
        (base / "learned_ghost" / f"{name}_{v}").mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.parquet")):
        S = pl.read_parquet(f).with_columns((pl.col("nll") - pl.col("entropy")).alias("excess"))
        L = pl.read_parquet(base / "learned_ghost" / name / f.name)
        J = L.join(S.select([*keys, "excess", "pit"]), on=keys, how="inner")
        for v in agg:
            out = J.with_columns(pl.col(v).alias("nll_ind")).drop(["excess", "pit"])
            out.write_parquet(base / "learned_ghost" / f"{name}_{v}" / f.name)
            agg[v].append(_per_defender(out))
    for v, parts in agg.items():
        dev = pl.concat(parts).with_columns(pl.lit(f"{name}_{v}").alias("ghost"))
        dev.write_parquet(base / "analysis" / f"learned_ghost_dev_{name}_{v}.parquet")
        print(f"learned_ghost_dev_{name}_{v}.parquet: {dev.height} possession-defenders")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossfit", required=True, help="config tag; runs <tag>_fold0 .. _fold4")
    ap.add_argument("--draws", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--games", type=int, default=None, help="first N games per split (test)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--variants", action="store_true", help="build the _excess / _pit inputs")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    base = D.DataConfig.load(game_set="all").processed_dir / "all"
    name = f"cf_{args.crossfit}"
    if args.variants:
        variants(base, name)
    else:
        score_games(args, base, name)


if __name__ == "__main__":
    main()
