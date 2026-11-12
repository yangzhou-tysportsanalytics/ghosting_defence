"""Velocity look-ahead and sharpness of the final learned ghost on the test games (SSAC27 review
d, e; 2026-11-02).

(d) Look-ahead. The inputs carry central-difference velocities, which reach 0.2 s ahead (D-031).
    The final model is scored on the test games twice, as phase3_test_uncertainty.py replays
    the evaluation (same batches and mask draws): with the stored inputs, and with every
    velocity replaced by the backward difference (x_t - x_{t-1}) / 0.2 s, recomputed from the
    positions in the features (dataset.backward_velocity). The model was not retrained, so the
    NLL increase bounds how much of the fit leans on the 0.2 s look-ahead. Game bootstrap of the
    difference.
(e) Sharpness. On a random subset of test possessions (individual schedule, one hidden defender
    each, every 5th valid step = 1 s), the area of the 80 % highest-density region of the
    learned individual ghost (0.5 ft grid) and whether the real defender is inside it; the same
    for the individual rule baseline (an isotropic Gaussian with the validation sigma around the
    left-over attacker's rule position: area pi * sigma^2 * 2 ln 5).

Output: reports/phase3/<version>_all_<run>_lookahead_sharpness.json

Usage:
    uv run python scripts/phase3_lookahead_sharpness.py [--model-run gpu_d128_lr1e-3_final]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ghost import data as D
from ghost.ghost.dataset import apply_mask, backward_velocity, load_arrays
from ghost.ghost.model import V_SCALE, X_SCALE, Y_SCALE

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("tu", HERE / "phase3_test_uncertainty.py")
tu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tu)

LEVEL = 0.8
CELL = 0.5
GX = np.arange(-5.0, 60.0, CELL) + CELL / 2
GY = np.arange(-5.0, 55.0, CELL) + CELL / 2


def with_backward_velocity(A: dict) -> dict:
    f = A["feats"]  # (N, 11, T, F), positions scaled
    xy = np.stack([(f[..., 0] + 0.5) * X_SCALE, (f[..., 1] + 0.5) * Y_SCALE], -1)  # (N,11,T,2)
    xy = np.transpose(xy, (0, 2, 1, 3)).astype(np.float64)  # (N,T,11,2)
    v = backward_velocity(xy, A["valid"]).astype(np.float32)
    g = f.copy()
    g[..., 2:4] = np.transpose(v, (0, 2, 1, 3)) / V_SCALE
    g[~np.broadcast_to(A["valid"][:, None, :], g.shape[:3])] = f[
        ~np.broadcast_to(A["valid"][:, None, :], g.shape[:3])]  # padding unchanged  # fmt: skip
    return {**A, "feats": g}


def boot_diff(Gc: pl.DataFrame, Gb: pl.DataFrame, num: str, den: str, B: int = 2000) -> dict:
    J = Gc.select(["game_id", num, den]).join(
        Gb.select(["game_id", pl.col(num).alias("nb"), pl.col(den).alias("db")]), on="game_id"
    ).sort("game_id")
    a, b, c, d = (J[k].to_numpy() for k in (num, den, "nb", "db"))
    rng = np.random.default_rng(9)
    W = np.stack([np.bincount(rng.integers(0, len(a), len(a)), minlength=len(a)) for _ in range(B)])
    diff = (W @ c) / (W @ d) - (W @ a) / (W @ b)
    return {"central": float(a.sum() / b.sum()), "backward": float(c.sum() / d.sum()),
            "difference": float(c.sum() / d.sum() - a.sum() / b.sum()),
            "ci95": [float(np.quantile(diff, q)) for q in (0.025, 0.975)]}  # fmt: skip


@torch.no_grad()
def sharpness(model, A: dict, gamma, sigma_left: float, n_poss: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(A["valid"]), size=n_poss, replace=False))
    gx = torch.tensor(GX, dtype=torch.float32)
    gy = torch.tensor(GY, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)  # (G,2)
    areas, inside, rule_inside = [], [], []
    g = torch.Generator(device="cpu").manual_seed(seed)
    r2 = 2 * math.log(1 / (1 - LEVEL))  # chi-square(2) quantile / sigma^2
    for a in range(0, n_poss, 16):
        sl = idx[a : a + 16]
        sub = {k: A[k][sl] for k in ("feats", "ctx", "target", "valid")}
        f = torch.from_numpy(np.ascontiguousarray(sub["feats"]))
        fm, masked = apply_mask(f, "individual", g)
        o = model(fm, torch.from_numpy(np.ascontiguousarray(sub["ctx"])), None)
        mu_rule = tu._rule_positions(sub, gamma)  # (n,5k,T,2)
        for b in range(len(sl)):
            d = int(torch.nonzero(masked[b])[0, 0])
            steps = np.flatnonzero(sub["valid"][b])[::5]
            if steps.size == 0:
                continue
            w = torch.softmax(o["logits"][b, d, steps], -1)  # (S,M)
            mu, sg = o["mu"][b, d, steps], o["sigma"][b, d, steps]  # (S,M,2), (S,M)
            d2 = ((grid[None, :, None] - mu[:, None]) ** 2).sum(-1)  # (S,G,M)
            dens = (w[:, None] * torch.exp(-0.5 * d2 / sg[:, None] ** 2)
                    / (2 * math.pi * sg[:, None] ** 2)).sum(-1)  # (S,G)  # fmt: skip
            mass = dens * CELL**2
            srt, _ = torch.sort(mass, -1, descending=True)
            cum = torch.cumsum(srt, -1) / mass.sum(-1, keepdim=True)
            k = (cum < LEVEL).sum(-1)  # cells needed
            thr = srt.gather(1, k[:, None])[:, 0] / CELL**2
            areas.append((k + 1).float().numpy() * CELL**2)
            z = torch.from_numpy(sub["target"][b, d, steps])  # (S,2)
            dz = ((z[:, None] - mu) ** 2).sum(-1)
            pz = (w * torch.exp(-0.5 * dz / sg**2) / (2 * math.pi * sg**2)).sum(-1)
            inside.append((pz >= thr).numpy())
            # rule: left-over attacker after the best assignment of the other four defenders
            zr = sub["target"][b][:, steps]  # (5d,S,2)
            mr = mu_rule[b][:, steps]  # (5k,S,2)
            dist = np.sqrt(((zr[:, None] - mr[None]) ** 2).sum(-1))  # (5d,5k,S)
            per = dist[np.arange(5)[None, :], tu.PERMS]  # (120,5,S)
            best = (per.sum(1) - per[:, d]).argmin(0)  # (S,)
            kk = tu.PERMS[best, d]
            e2 = dist[d, kk, np.arange(steps.size)] ** 2
            rule_inside.append(e2 <= r2 * sigma_left**2)
    areas = np.concatenate(areas)
    return {
        "level": LEVEL, "n_possessions": n_poss, "n_defender_steps": int(areas.size),
        "learned_area_sqft": {"mean": float(areas.mean()), "median": float(np.median(areas)),
                              "p10_p90": [float(np.quantile(areas, q)) for q in (0.1, 0.9)]},
        "learned_coverage": float(np.concatenate(inside).mean()),
        "rule_area_sqft": float(math.pi * sigma_left**2 * r2), "rule_sigma_ft": sigma_left,
        "rule_coverage": float(np.concatenate(rule_inside).mean()),
        "learned_equivalent_radius_ft": float(np.sqrt(areas.mean() / math.pi)),
        "rule_equivalent_radius_ft": float(sigma_left * math.sqrt(r2)),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-run", default="gpu_d128_lr1e-3_final")
    ap.add_argument("--sharp-poss", type=int, default=2000)
    ap.add_argument("--skip-lookahead", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    stem = f"{cfg.version}_all_{args.model_run}"
    ev = json.loads(Path(f"reports/phase3/{stem}_eval.json").read_text())
    A, K = load_arrays(base / "ghost_arrays", "test")
    T = A["valid"].shape[1]
    model = tu.lgd.load_model(Path("runs/phase3") / stem, T, "cpu")
    model.eval()
    rep: dict = {"model_run": args.model_run, "n_test_possessions": len(A["valid"])}
    t0 = time.time()
    sig_left = ev["baseline_individual_leftover"]["sigma_ft"]
    rep["sharpness"] = sharpness(model, A, ev["baseline_attacker_mixture"]["gamma"], sig_left,
                                 args.sharp_poss)  # fmt: skip
    print(json.dumps(rep["sharpness"], indent=1), f"{time.time() - t0:.0f}s", flush=True)
    if not args.skip_lookahead:
        Gc = pl.read_parquet(f"reports/phase3/{stem}_test_per_game.parquet")
        S = tu.learned_sums(model, with_backward_velocity(A))
        Gb = (pl.DataFrame({"game_id": K["game_id"].to_list(), **S}).group_by("game_id").sum()
              .sort("game_id"))  # fmt: skip
        rep["lookahead"] = {
            "note": "same final model; inputs with backward-difference velocities, not retrained",
            "team_nll": boot_diff(Gc, Gb, "team_nll", "team_n"),
            "individual_nll": boot_diff(Gc, Gb, "ind_nll", "ind_n"),
            "team_mse": boot_diff(Gc, Gb, "team_sq", "team_n"),
            "individual_mse": boot_diff(Gc, Gb, "ind_sq", "ind_n"),
            "rule_individual_nll_test": float(Gc["rind_nll"].sum() / Gc["rind_n"].sum()),
        }  # fmt: skip
        print(json.dumps(rep["lookahead"], indent=1), f"{time.time() - t0:.0f}s")
    out = Path(f"reports/phase3/{stem}_lookahead_sharpness.json")
    out.write_text(json.dumps(rep, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
