"""Single-moment conditional density baselines at the shot release (writing request B3).

NFL Ghosts (Yurko et al. 2024) model a defender's position at one moment from features of that
moment. Here, at the release of every half-court shot (shots.parquet, trusted releases), a
mixture density network (MLP, isotropic Gaussian mixture, fitted on the train games, early
stopping on validation) sees only the snapshot at the release - no history:
* "ind_snapshot": attackers, ball, the four other defenders (sorted by distance to the shooter)
  and the context vector -> the remaining defender (one row per defender; individual task);
* "team_snapshot": attackers and ball and context -> any defender (identity-free; team task);
* "team_shot_features": shooter location, distance, angle, three-point flag, ball and context
  only -> any defender (the few-features version).
Scored on the test games by NLL (ft^-2 units, as the ghosts) against, on the same defenders at
the same step: the learned ghost (nll_ind / nll_team of phase4_learned_ghost_dev.py; by default
cross-fitted, out of fold; ``--learned gpu_d128_lr1e-3_final_test`` = the final model, trained on
the same train games as the MDNs with validation for selection: the like-for-like comparison,
review R3 2026-11-02) and the rule ghost (individual: left-over attacker, sigma fitted on
validation; team: equal-weight attacker mixture). Differences with game-bootstrap intervals.

Output: reports/phase3/<version>_all_release_density_baseline[_<learned>].json (suffix when
--learned is not the default)

Usage:
    uv run python scripts/analysis_release_density_baseline.py [--epochs 40]
    uv run python scripts/analysis_release_density_baseline.py --learned gpu_d128_lr1e-3_final_test
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import permutations
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from ghost import data as D
from ghost.court import HOOP_LEFT
from ghost.xfg import api

LEARNED = "cf_gpu_d128_lr1e-3"
K_COMP = 8
PERMS = np.array(list(permutations(range(5))))


def snapshots(base: Path, split: str) -> dict:
    """Positions (ft) at the release step of every trusted half-court shot of a split."""
    root = base / "ghost_arrays" / split
    K = pl.read_parquet(root / "keys.parquet").join(
        pl.read_parquet(base / "possessions.parquet").select(["game_id", "possession_id", "t_start"]),
        on=["game_id", "possession_id"], how="left")  # fmt: skip
    S = api.trusted(pl.read_parquet(base / "analysis" / "shots.parquet")).filter(
        pl.col("possession_id").is_not_null()
    ).select(["game_id", "possession_id", "t_release_ms", "shooter_id", "is_three", "dist_ft",
              "angle_deg"]).unique(subset=["game_id", "possession_id"], keep="first")  # fmt: skip
    J = K.join(S, on=["game_id", "possession_id"], how="inner").sort("row")
    F = np.load(root / "feats.npy", mmap_mode="r")
    C = np.load(root / "ctx.npy", mmap_mode="r")
    Z = np.load(root / "target.npy", mmap_mode="r")
    V = np.load(root / "valid.npy", mmap_mode="r")
    out = {k: [] for k in ("off", "ball", "def", "ctx", "shooter", "three", "dist", "angle",
                           "game", "poss", "step")}  # fmt: skip
    T = V.shape[1]
    for r in J.iter_rows(named=True):
        k = int(round((r["t_release_ms"] - r["t_start"]) / 200))
        i = r["row"]
        if not (0 <= k < T) or not V[i, k] or r["shooter_id"] not in r["off_ids"]:
            continue
        f = np.asarray(F[i, :, k, :2], dtype=np.float64)
        out["off"].append(np.stack([(f[1:6, 0] + 0.5) * 47.0, (f[1:6, 1] + 0.5) * 50.0], -1))
        out["ball"].append([(f[0, 0] + 0.5) * 47.0, (f[0, 1] + 0.5) * 50.0])
        out["def"].append(np.asarray(Z[i, :, k], dtype=np.float64))
        out["ctx"].append(np.asarray(C[i, k], dtype=np.float64))
        out["shooter"].append(r["off_ids"].index(r["shooter_id"]))
        out["three"].append(float(r["is_three"]))
        out["dist"].append(r["dist_ft"])
        out["angle"].append(r["angle_deg"])
        out["game"].append(r["game_id"])
        out["poss"].append(r["possession_id"])
        out["step"].append(k)
    return {k: np.asarray(v) for k, v in out.items()}


def features(s: dict, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, target z, defender slot) with one row per defender."""
    n = len(s["step"])
    sh = s["off"][np.arange(n), s["shooter"]]  # (n, 2)
    scale = np.array([47.0, 50.0])
    base = [sh / scale, s["ball"] / scale, s["ctx"],
            np.stack([s["three"], s["dist"] / 30, s["angle"] / 90], -1)]  # fmt: skip
    if kind != "team_shot_features":
        o = np.argsort(np.linalg.norm(s["off"] - sh[:, None], axis=-1), axis=1)
        base.append((np.take_along_axis(s["off"], o[..., None], 1) / scale).reshape(n, 10))
    X0 = np.concatenate(base, 1)
    rows, zs, slots = [], [], []
    for d in range(5):
        x = X0
        if kind == "ind_snapshot":
            others = np.delete(s["def"], d, axis=1)  # (n, 4, 2)
            o = np.argsort(np.linalg.norm(others - sh[:, None], axis=-1), axis=1)
            x = np.concatenate([X0, (np.take_along_axis(others, o[..., None], 1) / scale)
                                .reshape(n, 8)], 1)  # fmt: skip
        rows.append(x)
        zs.append(s["def"][:, d])
        slots.append(np.full(n, d))
    return np.concatenate(rows), np.concatenate(zs), np.concatenate(slots)


class MDN(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, K_COMP * 4))  # fmt: skip

    def nll(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.net(x).view(-1, K_COMP, 4)
        logw = torch.log_softmax(h[..., 0], -1)
        mu = torch.stack([(h[..., 1] + 0.5) * 47.0, (h[..., 2] + 0.5) * 50.0], -1)
        logs = h[..., 3].clamp(-1, 4) + 1.0  # sigma in feet, e^0 .. e^5
        d2 = ((z[:, None] - mu) ** 2).sum(-1)
        logn = -0.5 * d2 / torch.exp(2 * logs) - 2 * logs - math.log(2 * math.pi)
        return -torch.logsumexp(logw + logn, -1)


def fit(Xtr, ztr, Xva, zva, epochs: int, seed: int = 9) -> tuple[MDN, dict, list]:
    torch.manual_seed(seed)
    mean, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    tr = [
        torch.tensor((Xtr - mean) / sd, dtype=torch.float32),
        torch.tensor(ztr, dtype=torch.float32),
    ]
    va = [
        torch.tensor((Xva - mean) / sd, dtype=torch.float32),
        torch.tensor(zva, dtype=torch.float32),
    ]
    m = MDN(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    g = torch.Generator().manual_seed(seed)
    best, state, hist = float("inf"), None, []
    for _ in range(epochs):
        m.train()
        for idx in torch.randperm(len(tr[0]), generator=g).split(512):
            loss = m.nll(tr[0][idx], tr[1][idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = float(m.nll(*va).mean())
        hist.append(v)
        if v < best:
            best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
    m.load_state_dict(state)
    return m, {"mean": mean, "sd": sd}, hist


def rule_nll(s: dict, gamma, sigma_team: float, sigma_ind: float) -> tuple[np.ndarray, np.ndarray]:
    """(team, individual) rule-ghost NLL per (shot, defender) at the release."""
    mu = gamma[0] * s["off"] + gamma[1] * s["ball"][:, None] + gamma[2] * HOOP_LEFT  # (n, 5k, 2)
    d2 = ((s["def"][:, :, None] - mu[:, None]) ** 2).sum(-1)  # (n, 5d, 5k)
    st = sigma_team**2
    logn = -0.5 * d2 / st - math.log(st) - math.log(2 * math.pi)
    team = -(np.logaddexp.reduce(logn, axis=2) - math.log(5))
    dist = np.sqrt(d2)
    per = dist[:, np.arange(5)[None, :], PERMS]  # (n, 120, 5)
    tot = per.sum(-1)
    ind = np.empty_like(team)
    si = sigma_ind**2
    for d in range(5):
        best = (tot - per[..., d]).argmin(-1)
        k = PERMS[best, d]
        e = d2[np.arange(len(k)), d, k]
        ind[:, d] = 0.5 * e / si + math.log(si) + math.log(2 * math.pi)
    return team, ind


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--learned", default=LEARNED, help="learned_ghost/<name> to compare with")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    snap = {sp: snapshots(base, sp) for sp in ("train", "val", "test")}
    rb = json.loads((Path("reports/phase3") / f"{cfg.version}_all_rule_baselines.json").read_text())
    gamma = rb["gamma"]
    te = snap["test"]
    n = len(te["step"])
    # learned ghost at the same (possession, defender slot, step)
    lg = pl.concat([
        pl.read_parquet(base / "learned_ghost" / args.learned / f"{g}.parquet")
        for g in np.unique(te["game"])
    ]).select(["game_id", "possession_id", "def_slot", "step", "nll_ind", "nll_team"])  # fmt: skip
    key = pl.DataFrame({"game_id": np.repeat(te["game"], 5), "possession_id": np.repeat(te["poss"], 5),
                        "def_slot": np.tile(np.arange(5), n).astype(np.int8),
                        "step": np.repeat(te["step"], 5).astype(np.int16)})  # fmt: skip
    lgj = key.join(lg, on=["game_id", "possession_id", "def_slot", "step"], how="left")
    learned = {"ind": lgj["nll_ind"].to_numpy().reshape(n, 5),
               "team": lgj["nll_team"].to_numpy().reshape(n, 5)}  # fmt: skip
    r_team, r_ind = rule_nll(
        te, gamma, rb["team_sigma_ft"], rb["val"]["individual_leftover"]["sigma_ft"]
    )
    rep: dict = {"n_test_shots": n, "n_train_shots": len(snap["train"]["step"]),
                 "learned_ghost": args.learned, "mdn_components": K_COMP, "models": {}}  # fmt: skip
    games = np.repeat(te["game"], 1)
    ug, inv = np.unique(games, return_inverse=True)
    rng = np.random.default_rng(9)
    bidx = rng.integers(0, len(ug), size=(args.boot, len(ug)))

    def boot_diff(a: np.ndarray, b: np.ndarray) -> list[float]:
        dd = (a - b).mean(1)  # per shot, mean over defenders
        s_, c_ = np.bincount(inv, weights=dd), np.bincount(inv).astype(float)
        bs = s_[bidx].sum(1) / c_[bidx].sum(1)
        return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]

    for kind in ("ind_snapshot", "team_snapshot", "team_shot_features"):
        Xtr, ztr, _ = features(snap["train"], kind)
        Xva, zva, _ = features(snap["val"], kind)
        Xte, zte, slot = features(te, kind)
        m, norm, hist = fit(Xtr, ztr, Xva, zva, args.epochs)
        with torch.no_grad():
            nll = m.nll(torch.tensor((Xte - norm["mean"]) / norm["sd"], dtype=torch.float32),
                        torch.tensor(zte, dtype=torch.float32)).numpy()  # fmt: skip
        mdn = np.empty((n, 5))
        mdn[np.tile(np.arange(n), 5), slot] = nll  # rows are defender-major
        task = "ind" if kind == "ind_snapshot" else "team"
        rule = r_ind if task == "ind" else r_team
        ok = ~np.isnan(learned[task]).any(1)
        res = {
            "task": task, "n_features": Xtr.shape[1], "best_epoch": int(np.argmin(hist)),
            "val_nll_by_epoch": [round(v, 4) for v in hist],
            "test_nll": {"mdn_snapshot": float(mdn[ok].mean()),
                         "learned_ghost": float(learned[task][ok].mean()),
                         "rule_ghost": float(rule[ok].mean())},
            "n_shots_compared": int(ok.sum()),
            "difference_learned_minus_mdn_ci95": boot_diff(learned[task][ok], mdn[ok])
            if ok.all() else None,
            "difference_rule_minus_mdn": float((rule[ok] - mdn[ok]).mean()),
        }  # fmt: skip
        rep["models"][kind] = res
        t = res["test_nll"]
        print(f"{kind:>19}: test NLL  MDN {t['mdn_snapshot']:.3f}  learned {t['learned_ghost']:.3f}"
              f"  rule {t['rule_ghost']:.3f}  (n {res['n_shots_compared']}, best epoch "
              f"{res['best_epoch']})", flush=True)  # fmt: skip
    trained = ("cross-fitted (out of fold; trained on four fifths of all games)"
               if args.learned.startswith("cf_") else "trained on the train games")
    rep["note"] = ("Single-moment baselines see the snapshot at the release only; the learned ghost "
                   f"sees the possession up to the release (online). Learned values: {trained}; the "
                   "MDNs are fitted on train games. Same test defenders and step.")  # fmt: skip
    suffix = "" if args.learned == LEARNED else f"_{args.learned}"
    p = Path("reports/phase3") / f"{cfg.version}_all_release_density_baseline{suffix}.json"
    p.write_text(json.dumps(rep, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
