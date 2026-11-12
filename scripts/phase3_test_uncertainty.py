"""Game-bootstrap intervals for the learned-ghost test results (review A4, Table 4).

Replays the test evaluation of phase3_train.py exactly (same batches of 32, same mask draws with
a generator seeded 0 per schedule, the final model on CPU) but keeps per-possession sums, so that
the totals reproduce Table 4 and the 126 test games can be resampled:
  * learned team ghost: identity-free slot-mixture NLL and RMSE of the best permutation (as the
    set loss), over all five defenders;
  * learned individual ghost: NLL and RMSE of the one randomly hidden defender per possession;
  * rule ghost: equal-weight attacker-mixture NLL (sigma fitted on validation) and RMSE of the
    best permutation; individual rule baseline (left-over attacker, sigma fitted on validation),
    over all five defenders.
Each quantity is a ratio of game sums (sum / count), so the bootstrap resamples games and
recomputes the ratios; differences learned - rule use the same draws. Design effect = bootstrap
variance / the variance if every defender-step were independent; effective sample size =
defender-steps / design effect.

Outputs: reports/phase3/<version>_all_<run>_test_uncertainty.json (aggregates only) and the
per-game sums reports/phase3/<version>_all_<run>_test_per_game.parquet (game_id, sums, counts).

Usage:
    uv run python scripts/phase3_test_uncertainty.py [--model-run gpu_d128_lr1e-3_final]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ghost import data as D
from ghost.ghost.dataset import apply_mask, load_arrays
from ghost.ghost.model import mixture_mean, mixture_nll

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase3_train import _rule_positions  # noqa: E402

_spec = importlib.util.spec_from_file_location("lgd", HERE / "phase4_learned_ghost_dev.py")
lgd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lgd)

PERMS = np.array(list(permutations(range(5))))  # (120, 5)
TPERMS = torch.tensor(PERMS, dtype=torch.long)
BS = 32  # as phase3_train.evaluate


def best_perm_sum(cost: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """cost (B, 5 slots, 5 defenders, T): per-possession total of the best permutation."""
    c = (cost * valid[:, None, None, :]).sum(-1)  # (B,5,5)
    tot = c[:, torch.arange(5)[None, :], TPERMS].sum(-1)  # (B,120)
    return tot.min(-1).values


@torch.no_grad()
def learned_sums(model, A: dict) -> dict[str, np.ndarray]:
    """Per-possession sums of the learned ghost, replaying evaluate() for both schedules."""
    N = len(A["valid"])
    out = {k: np.zeros(N) for k in ("team_nll", "team_sq", "team_n", "ind_nll", "ind_sq", "ind_n")}
    for sch in ("team", "individual"):
        g = torch.Generator(device="cpu").manual_seed(0)
        for a in range(0, N, BS):
            sl = slice(a, a + BS)
            f = torch.from_numpy(np.ascontiguousarray(A["feats"][sl]))
            x = torch.from_numpy(np.ascontiguousarray(A["ctx"][sl]))
            z = torch.from_numpy(np.ascontiguousarray(A["target"][sl]))
            v = torch.from_numpy(np.ascontiguousarray(A["valid"][sl]))
            fm, masked = apply_mask(f, sch, g)
            o = model(fm, x, None)
            cost = mixture_nll(o, z)  # (B,5 slots,5 defenders,T)
            sq = ((mixture_mean(o)[:, :, None] - z[:, None]) ** 2).sum(-1)  # (B,5,5,T)
            vf = v.float()
            if sch == "team":
                nll = -(torch.logsumexp(-cost, dim=1) - math.log(5.0))  # (B,5d,T)
                out["team_nll"][sl] = (nll * vf[:, None]).sum((1, 2)).numpy()
                out["team_sq"][sl] = best_perm_sum(sq, vf).numpy()
                out["team_n"][sl] = (vf.sum(-1) * 5).numpy()
            else:
                diag_nll = torch.diagonal(cost, dim1=1, dim2=2).permute(0, 2, 1)  # (B,5,T)
                diag_sq = torch.diagonal(sq, dim1=1, dim2=2).permute(0, 2, 1)
                w = masked[:, :, None].float() * vf[:, None, :]
                out["ind_nll"][sl] = (diag_nll * w).sum((1, 2)).numpy()
                out["ind_sq"][sl] = (diag_sq * w).sum((1, 2)).numpy()
                out["ind_n"][sl] = w.sum((1, 2)).numpy()
    return out


def rule_sums(A: dict, gamma, sigma_mix: float, sigma_left: float) -> dict[str, np.ndarray]:
    """Per-possession sums of the rule baselines (phase3_train.baseline_nll / baseline_rmse /
    individual_baseline, with the validation sigmas)."""
    N = len(A["valid"])
    out = {k: np.zeros(N) for k in ("rteam_nll", "rteam_sq", "rteam_n", "rind_d2", "rind_n")}
    s2 = sigma_mix**2
    for a in range(0, N, 512):
        sl = slice(a, a + 512)
        sub = {k: A[k][sl] for k in ("feats", "target", "valid")}
        mu = _rule_positions(sub, gamma)  # (n,5k,T,2)
        z, v = sub["target"], sub["valid"]
        d2 = ((z[:, :, None] - mu[:, None]) ** 2).sum(-1)  # (n,5d,5k,T)
        logn = -0.5 * d2 / s2 - math.log(s2) - math.log(2 * math.pi)
        m = logn.max(2, keepdims=True)
        ll = m[:, :, 0] + np.log(np.exp(logn - m).mean(2))  # (n,5d,T)
        out["rteam_nll"][sl] = -(ll * v[:, None]).sum((1, 2))
        cost = (d2 * v[:, None, None, :]).sum(-1)  # (n,5d,5k)
        out["rteam_sq"][sl] = cost[:, np.arange(5)[None, :], PERMS].sum(-1).min(1)
        out["rteam_n"][sl] = v.sum(-1) * 5
        dist = np.sqrt(d2.transpose(0, 3, 1, 2))  # (n,T,5d,5k)
        per = dist[:, :, np.arange(5)[None, :], PERMS]  # (n,T,120,5)
        tot = per.sum(-1)
        d2t = d2.transpose(0, 3, 1, 2)
        acc = np.zeros(len(v))
        for d in range(5):
            best = (tot - per[..., d]).argmin(-1)
            k = PERMS[best, d]
            e = np.take_along_axis(d2t[:, :, d, :], k[..., None], -1)[..., 0]  # (n,T)
            acc += (e * v).sum(-1)
        out["rind_d2"][sl] = acc
        out["rind_n"][sl] = v.sum(-1) * 5
    out["rind_nll"] = 0.5 * out["rind_d2"] / sigma_left**2 + out["rind_n"] * (
        math.log(sigma_left**2) + math.log(2 * math.pi)
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-run", default="gpu_d128_lr1e-3_final")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    ev = json.loads(
        Path(f"reports/phase3/{cfg.version}_all_{args.model_run}_eval.json").read_text()
    )
    A, K = load_arrays(base / "ghost_arrays", "test")
    T = A["valid"].shape[1]
    model = lgd.load_model(Path("runs/phase3") / f"{cfg.version}_all_{args.model_run}", T, "cpu")
    model.eval()
    gamma = ev["baseline_attacker_mixture"]["gamma"]
    sig_mix = ev["baseline_attacker_mixture"]["sigma_ft"]
    sig_left = ev["baseline_individual_leftover"]["sigma_ft"]
    S = {**learned_sums(model, A), **rule_sums(A, gamma, sig_mix, sig_left)}
    G = (
        pl.DataFrame({"game_id": K["game_id"].to_list(), **S})
        .group_by("game_id")
        .sum()
        .sort("game_id")
    )
    out_dir = Path("reports/phase3")
    G.write_parquet(out_dir / f"{cfg.version}_all_{args.model_run}_test_per_game.parquet")

    metrics = {  # name: (numerator, denominator, transform)
        "learned_team_nll": ("team_nll", "team_n", None),
        "learned_team_rmse_ft": ("team_sq", "team_n", "sqrt"),
        "learned_individual_nll": ("ind_nll", "ind_n", None),
        "learned_individual_rmse_ft": ("ind_sq", "ind_n", "sqrt"),
        "rule_team_nll": ("rteam_nll", "rteam_n", None),
        "rule_team_rmse_ft": ("rteam_sq", "rteam_n", "sqrt"),
        "rule_individual_nll": ("rind_nll", "rind_n", None),
        "rule_individual_rmse_ft": ("rind_d2", "rind_n", "sqrt"),
    }
    cols = {c: G[c].to_numpy() for c in G.columns if c != "game_id"}

    def value(w: np.ndarray, name: str) -> float:
        a, b, tr = metrics[name]
        r = (cols[a] * w).sum() / (cols[b] * w).sum()
        return float(np.sqrt(r) if tr else r)

    ng = G.height
    rng = np.random.default_rng(9)
    W = np.stack([np.bincount(rng.integers(0, ng, ng), minlength=ng) for _ in range(args.boot)])
    one = np.ones(ng)
    rep = {"model_run": args.model_run, "n_test_games": ng, "n_test_possessions": len(K),
           "bootstrap": f"games, B={args.boot}, seed 9", "metrics": {}, "differences": {},
           "check_against_eval": {}}  # fmt: skip
    boots = {}
    for name in metrics:
        est = value(one, name)
        b = np.array([value(w, name) for w in W])
        boots[name] = b
        a, d, tr = metrics[name]
        n_obs = float(cols[d].sum())
        out = {"estimate": est, "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))],
               "se_bootstrap": float(b.std(ddof=1)), "n_defender_steps": n_obs}  # fmt: skip
        rep["metrics"][name] = out
    for kind in ("team_nll", "team_rmse_ft", "individual_nll", "individual_rmse_ft"):
        d = boots[f"learned_{kind}"] - boots[f"rule_{kind}"]
        e = (
            rep["metrics"][f"learned_{kind}"]["estimate"]
            - rep["metrics"][f"rule_{kind}"]["estimate"]
        )
        rep["differences"][f"learned_minus_rule_{kind}"] = {
            "estimate": e, "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}  # fmt: skip
    t = ev["test"]
    rep["check_against_eval"] = {
        "learned_team_nll": t["team"]["slot_mixture_nll_ft"],
        "learned_team_rmse_ft": t["team"]["rmse_ft"],
        "learned_individual_nll": t["individual"]["nll_ft"],
        "learned_individual_rmse_ft": t["individual"]["rmse_ft"],
        "rule_team_nll": t["baseline_attacker_mixture"]["nll_ft"],
        "rule_team_rmse_ft": t["baseline_attacker_mixture"]["rmse_ft_best_permutation"],
        "rule_individual_nll": t["baseline_individual_leftover"]["nll_ft"],
        "rule_individual_rmse_ft": t["baseline_individual_leftover"]["rmse_ft"],
    }
    rep["max_abs_deviation_from_eval"] = max(
        abs(rep["metrics"][k]["estimate"] - v) for k, v in rep["check_against_eval"].items()
    )
    # possession-level design effect: bootstrap variance of the game ratio vs the variance if
    # possessions were independent (possession ratio estimator, delta method)
    P = pl.DataFrame({"game_id": K["game_id"].to_list(), **S})
    for name, (a, d, tr) in metrics.items():
        if tr is not None:
            continue
        x, n = P[a].to_numpy(), P[d].to_numpy()
        keep = n > 0
        x, n = x[keep], n[keep]
        r = x.sum() / n.sum()
        var_iid_poss = ((x - r * n) ** 2).sum() / n.sum() ** 2
        deff = float(boots[name].var(ddof=1) / var_iid_poss)
        rep["metrics"][name]["design_effect_vs_independent_possessions"] = deff
        rep["metrics"][name]["effective_possessions"] = float(keep.sum() / deff)
    p = out_dir / f"{cfg.version}_all_{args.model_run}_test_uncertainty.json"
    p.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
