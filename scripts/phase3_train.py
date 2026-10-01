"""Phase 3 minimal ghost model: train on the train games, evaluate on val (CPU-capable).

Compares the ghost's conditional NLL (feet) against the rule baseline "defender stands near
an attacker" = equal-weight mixture over the five attackers of the fitted HMM emission
N(g_o O_k + g_b B + g_h H, sigma^2 I) (Phase 3 acceptance criterion).

Outputs: runs/phase3/<version>_<game_set>_<tag>/{model.pt, history.json, eval.json, overlay.png}
(runs/ is git-ignored); a copy of eval.json under reports/phase3/.

Usage:
    uv run python scripts/phase3_train.py --game-set tiny --epochs 5 [--loss nll|l2]
        [--schedule mixed|team|individual] [--mode online|offline] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from ghost import data as D  # noqa: E402
from ghost.court import HOOP_LEFT  # noqa: E402
from ghost.ghost.dataset import build_arrays, load_arrays  # noqa: E402
from ghost.ghost.model import GhostConfig, GhostModel  # noqa: E402
from ghost.ghost.train import (  # noqa: E402
    TrainConfig,
    calibration_eval,
    config_dict,
    evaluate,
    fit,
    predict_mean,
)
from ghost.tensors import fill_shot_clock, load_game_set  # noqa: E402
from ghost.viz.court import draw_court  # noqa: E402


def baseline_nll(arrs: dict, gamma, sigma_ft: float) -> float:
    """Mean NLL (feet) of defender positions under the equal-weight attacker mixture."""
    f = arrs["feats"]  # (N,11,T,F) normalised
    off = np.stack([(f[:, 1:6, :, 0] + 0.5) * 47.0, (f[:, 1:6, :, 1] + 0.5) * 50.0], -1)
    ball = np.stack([(f[:, 0, :, 0] + 0.5) * 47.0, (f[:, 0, :, 1] + 0.5) * 50.0], -1)
    g_o, g_b, g_h = gamma
    mu = g_o * off + g_b * ball[:, None] + g_h * HOOP_LEFT  # (N,5k,T,2)
    z = arrs["target"]  # (N,5d,T,2)
    d2 = ((z[:, :, None] - mu[:, None]) ** 2).sum(-1)  # (N,5d,5k,T)
    s2 = sigma_ft**2
    logn = -0.5 * d2 / s2 - math.log(s2) - math.log(2 * math.pi)
    m = logn.max(2, keepdims=True)
    ll = m[:, :, 0] + np.log(np.exp(logn - m).mean(2))  # (N,5d,T)
    v = arrs["valid"][:, None, :]
    return float(-(ll * v).sum() / (v.sum() * 5))


def baseline_rmse(arrs: dict, gamma) -> float:
    """RMSE (ft) of the rule positions "one defender at the emission mean of each attacker",
    matched to the five real defenders by the best permutation per possession (the same hindsight
    matching as the team-ghost set loss)."""
    from itertools import permutations

    f = arrs["feats"]
    off = np.stack([(f[:, 1:6, :, 0] + 0.5) * 47.0, (f[:, 1:6, :, 1] + 0.5) * 50.0], -1)
    ball = np.stack([(f[:, 0, :, 0] + 0.5) * 47.0, (f[:, 0, :, 1] + 0.5) * 50.0], -1)
    g_o, g_b, g_h = gamma
    mu = g_o * off + g_b * ball[:, None] + g_h * HOOP_LEFT  # (N, 5 attackers, T, 2)
    z, v = arrs["target"], arrs["valid"]  # (N, 5 defenders, T, 2), (N, T)
    d2 = ((z[:, :, None] - mu[:, None]) ** 2).sum(-1)  # (N, 5d, 5k, T)
    cost = (d2 * v[:, None, None, :]).sum(-1)  # (N, 5d, 5k) summed over valid steps
    perms = np.array(list(permutations(range(5))))  # (120, 5): defender d -> attacker perm[d]
    tot = cost[:, np.arange(5)[None, :], perms].sum(-1)  # (N, 120)
    return float(np.sqrt(tot.min(1).sum() / (v.sum() * 5)))


def _rule_positions(arrs: dict, gamma) -> np.ndarray:
    f = arrs["feats"]
    off = np.stack([(f[:, 1:6, :, 0] + 0.5) * 47.0, (f[:, 1:6, :, 1] + 0.5) * 50.0], -1)
    ball = np.stack([(f[:, 0, :, 0] + 0.5) * 47.0, (f[:, 0, :, 1] + 0.5) * 50.0], -1)
    g_o, g_b, g_h = gamma
    return g_o * off + g_b * ball[:, None] + g_h * HOOP_LEFT  # (N, 5 attackers, T, 2)


def individual_baseline(arrs: dict, gamma, sigma_ft: float | None = None, chunk: int = 512) -> dict:
    """Rule baseline of the individual ghost: with defender d hidden, the four visible defenders
    are assigned to attackers per step (minimum total distance to the attackers' rule positions)
    and d is predicted at the left-over attacker's rule position. The 120 permutations of five
    attackers enumerate every (assignment of four, left-over attacker) pair. Returns RMSE and the
    NLL of an isotropic Gaussian there; ``sigma_ft`` None = MLE on these data."""
    from itertools import permutations

    perms = np.array(list(permutations(range(5))))  # (120, 5): defender -> attacker
    d2_all = []
    for a in range(0, len(arrs["valid"]), chunk):
        sub = {k: v[a : a + chunk] for k, v in arrs.items()}
        mu = _rule_positions(sub, gamma).transpose(0, 2, 1, 3)  # (n, T, 5k, 2)
        z = sub["target"].transpose(0, 2, 1, 3)  # (n, T, 5d, 2)
        d2 = ((z[:, :, :, None] - mu[:, :, None]) ** 2).sum(-1)  # (n, T, 5d, 5k)
        dist = np.sqrt(d2)
        per = dist[:, :, np.arange(5)[None, :], perms]  # (n, T, 120, 5): cost of d -> perm[d]
        tot = per.sum(-1)
        v = sub["valid"]
        for d in range(5):
            best = (tot - per[..., d]).argmin(-1)  # (n, T): assignment of the other four
            k = perms[best, d]  # left-over attacker
            e = np.take_along_axis(d2[:, :, d, :], k[..., None], -1)[..., 0]
            d2_all.append(e[v])
    d2s = np.concatenate(d2_all)
    s2 = d2s.mean() / 2 if sigma_ft is None else sigma_ft**2
    nll = float((0.5 * d2s / s2 + np.log(s2) + np.log(2 * np.pi)).mean())
    return {"rmse_ft": float(np.sqrt(d2s.mean())), "sigma_ft": float(np.sqrt(s2)), "nll_ft": nll}


def pick_games(games: dict, k_train: int | None, k_val: int | None, seed: int) -> dict:
    """Seeded whole-game subsets: one stream per split, so the val games do not depend on the
    train size, and a larger train subset contains the smaller ones (prefix of one permutation).
    ``games``: split -> sorted game ids. None = all games of the split."""
    keep = {}
    for j, (split_name, k) in enumerate((("train", k_train), ("val", k_val))):
        rng = np.random.default_rng([seed, j])
        g = games[split_name]
        keep[split_name] = list(rng.permutation(g)[: min(k, len(g))]) if k else list(g)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-set", default="tiny")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--loss", default="nll")
    ap.add_argument("--schedule", default="mixed")
    ap.add_argument("--mode", default="online")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="v1min")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA")
    ap.add_argument("--resume", action="store_true", help="resume from runs/.../checkpoint.pt")
    ap.add_argument("--train-games", type=int, default=None, help="seeded subset of train games")
    ap.add_argument("--val-games", type=int, default=None, help="seeded subset of val games")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--arrays-dir", default=None, help="packed arrays (phase3_export_arrays.py)")
    ap.add_argument("--eval-test", action="store_true", help="final run only: also score test")
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--load-model", default=None, help="model.pt to evaluate instead of training")
    ap.add_argument("--fold-out", type=int, default=None, help="cross-fitting: hold out this fold")
    ap.add_argument("--condition", default="league", choices=["league", "lineup", "scheme"])
    ap.add_argument(
        "--anchor", default="none", choices=["none", "rule"], help="rule: residual on rule ghost"
    )
    args = ap.parse_args()

    cfg = D.DataConfig.load(game_set=args.game_set)
    base = cfg.processed_dir / cfg.game_set
    test = None
    if args.fold_out is not None:  # cross-fitting: train on the other folds (all splits)
        if not args.arrays_dir or args.eval_test or args.train_games or args.val_games:
            raise SystemExit("--fold-out needs --arrays-dir, without --eval-test / game subsets")
        root, parts = Path(args.arrays_dir), {"train": [], "val": []}
        for s in ("train", "val", "test"):
            K = pl.read_parquet(root / s / "keys.parquet")
            for role, keep_fold in (("train", False), ("val", True)):
                sel = (K["fold"] == args.fold_out) if keep_fold else (K["fold"] != args.fold_out)
                g = sorted(K.filter(sel)["game_id"].unique().to_list())
                if g:
                    parts[role].append(load_arrays(root, s, g))
        cat = lambda ps: {k: np.concatenate([p[0][k] for p in ps]) for k in ps[0][0]}  # noqa: E731
        train, val = cat(parts["train"]), cat(parts["val"])  # "val" = the held-out fold
        keys = {r: pl.concat([p[1] for p in parts[r]]) for r in ("train", "val")}
    elif args.arrays_dir:  # packed arrays (phase3_export_arrays.py): no nbacore data needed
        root = Path(args.arrays_dir)
        games = {
            s: sorted(pl.read_parquet(root / s / "keys.parquet")["game_id"].unique().to_list())
            for s in ("train", "val")
        }
        keep = pick_games(games, args.train_games, args.val_games, cfg.seed)
        train, ktr = load_arrays(root, "train", keep["train"] if args.train_games else None)
        val, kva = load_arrays(root, "val", keep["val"] if args.val_games else None)
        keys = {"train": ktr, "val": kva}
        if args.eval_test:
            test, keys["test"] = load_arrays(root, "test")
    else:
        P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
        if args.train_games or args.val_games:  # CPU prototypes: whole games, fixed seed
            games = {
                s: sorted(P.filter(pl.col("split") == s)["game_id"].unique().to_list())
                for s in ("train", "val")
            }
            keep = pick_games(games, args.train_games, args.val_games, cfg.seed)
            P = P.filter(pl.col("game_id").is_in(keep["train"] + keep["val"]))
        if args.eval_test:
            raise SystemExit("--eval-test needs --arrays-dir (final runs use the packed arrays)")
        pt = load_game_set(base / "frames", sorted(P["game_id"].unique().to_list()), P)
        fill_shot_clock(pt, {g: D.shot_clock(g, cfg) for g in np.unique(pt.game_id)})
        arrs = build_arrays(pt, P)
        smap = dict(zip(P["game_id"].to_list(), P["split"].to_list(), strict=False))
        split = np.array([smap[g] for g in pt.game_id])
        tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
        sub = lambda ix: {k: v[ix] for k, v in arrs.items()}  # noqa: E731
        train, val = sub(tr), sub(va)
    tr, va = np.arange(len(train["valid"])), np.arange(len(val["valid"]))
    print(
        f"{cfg.game_set}: train {len(tr)} / val {len(va)} half-court possessions; device {args.device}"
    )

    # HMM emission fitted on the same game set (Phase 2): rule baseline and anchored head
    fitp = Path("reports/phase2") / f"{cfg.version}_{cfg.game_set}" / "matchup_fit.json"
    gamma, sigma = (0.62, 0.11, 0.27), 3.0
    if fitp.exists():
        prm = json.loads(fitp.read_text())["models"]["hmm"]["params"]
        gamma, sigma = prm["gamma"]["all"], prm["sigma_ft"]["all"]
    vocab = None
    if args.condition != "league":  # identities seen in the training games; 0 = unknown
        if not args.arrays_dir:
            raise SystemExit("--condition lineup / scheme needs --arrays-dir")
        vp = Path(args.load_model).parent / "vocab.json" if args.load_model else None
        if vp is not None and vp.exists():
            vocab = json.loads(vp.read_text())
        else:
            kt = keys["train"]
            players = sorted({int(p) for ids in kt["def_ids"].to_list() for p in ids})
            teams = sorted(int(t) for t in kt["defense_team_id"].unique().to_list())
            vocab = {"players": {str(p): i + 1 for i, p in enumerate(players)},
                     "teams": {str(t): i + 1 for i, t in enumerate(teams)}}  # fmt: skip
        for name, arrs in (("train", train), ("val", val), ("test", test)):
            if arrs is None:
                continue
            k = keys[name]
            arrs["player_idx"] = np.array(
                [[vocab["players"].get(str(p), 0) for p in ids] for ids in k["def_ids"].to_list()],
                dtype=np.int64,
            )
            arrs["team_idx"] = np.array(
                [vocab["teams"].get(str(t), 0) for t in k["defense_team_id"].to_list()],
                dtype=np.int64,
            )
    mcfg = GhostConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.blocks,
        mode=args.mode,
        anchor=args.anchor,
        gamma=tuple(gamma),
        condition=args.condition,
        n_players=1 + (len(vocab["players"]) if vocab else 0),
        n_teams=1 + (len(vocab["teams"]) if vocab else 0),
    )
    tcfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss=args.loss,
        schedule=args.schedule,
        device=args.device,
        max_batches_per_epoch=args.max_batches,
        amp=args.amp,
        **({"lr": args.lr} if args.lr else {}),
        checkpoint_path=str(
            Path("runs/phase3") / f"{cfg.version}_{cfg.game_set}_{args.tag}" / "checkpoint.pt"
        ),
    )
    ck = Path(tcfg.checkpoint_path)
    if ck.exists() and not args.resume:
        ck.unlink()  # fresh run unless --resume
    init_eval = None
    if args.anchor == "rule" and len(va):  # the untrained anchored model should be the rule ghost
        torch.manual_seed(tcfg.seed)  # same initial weights as fit()
        m0 = GhostModel(mcfg, n_steps=train["valid"].shape[1]).to(args.device)
        init_eval = {s: evaluate(m0, val, s, args.device) for s in ("team", "individual")}
        print("untrained anchored model:", json.dumps(init_eval))
    if args.load_model:  # score a trained model (e.g. the chosen sweep run on the test games)
        model = GhostModel(mcfg, n_steps=train["valid"].shape[1]).to(args.device)
        model.load_state_dict(torch.load(args.load_model, map_location=args.device))
        hist = []
    else:
        model, hist = fit(train, val, mcfg, tcfg)

    result = {
        "config": config_dict(mcfg, tcfg),
        "n_params": sum(p.numel() for p in model.parameters()),
        "val_team": evaluate(model, val, "team", args.device),
        "val_individual": evaluate(model, val, "individual", args.device),
        "val_untrained": init_eval,
        # sensitivity: drop steps whose shot clock was imputed by rule (nbacore v1.2)
        "val_team_excl_imputed_sc": evaluate(model, val, "team", args.device, exclude_imputed=True),
        "share_steps_sc_imputed_val": float(val["sc_imputed"][val["valid"]].mean())
        if "sc_imputed" in val
        else None,
        "calibration_val": calibration_eval(model, val, args.device) if len(va) else None,
        "baseline_attacker_mixture": {
            "gamma": gamma,
            "sigma_ft": sigma,
            "val_nll_ft": baseline_nll(val, gamma, sigma),
            "val_rmse_ft_best_permutation": baseline_rmse(val, gamma),
        },
        # sigma fitted on val (MLE): favourable to the baseline on val
        "baseline_individual_leftover": individual_baseline(val, gamma),
        "note": "NLL = -log density of the true defender position in ft^-2 units, mean per "
        "defender-step; lower is better.",
    }
    if test is not None:  # final run only: the test games are scored once, nothing is tuned
        ind_val = result["baseline_individual_leftover"]
        result["test"] = {
            "n_possessions": int(len(test["valid"])),
            "team": evaluate(model, test, "team", args.device),
            "individual": evaluate(model, test, "individual", args.device),
            "calibration": calibration_eval(model, test, args.device),
            "baseline_attacker_mixture": {
                "nll_ft": baseline_nll(test, gamma, sigma),
                "rmse_ft_best_permutation": baseline_rmse(test, gamma),
            },
            "baseline_individual_leftover": individual_baseline(test, gamma, ind_val["sigma_ft"]),
        }
    run = Path("runs/phase3") / f"{cfg.version}_{cfg.game_set}_{args.tag}"
    run.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run / "model.pt")
    if vocab is not None:
        (run / "vocab.json").write_text(json.dumps(vocab))
    (run / "history.json").write_text(json.dumps(hist, indent=2))
    (run / "eval.json").write_text(json.dumps(result, indent=2))
    rep = Path("reports/phase3")
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{cfg.version}_{cfg.game_set}_{args.tag}_eval.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))

    # static overlay: one val possession, real defence vs team-ghost mean, a few time steps
    if len(va):
        one = {k: v[:1] for k, v in val.items()}
        gh = predict_mean(model, one, "team", args.device)[0]  # (5,T,2)
        n = int(one["valid"][0].sum())
        fig, ax = plt.subplots(figsize=(6, 6.4))
        draw_court(ax, color="0.4")
        f0 = one["feats"][0]
        for k in range(5):
            ox = (f0[1 + k, :n, 0] + 0.5) * 47
            oy = (f0[1 + k, :n, 1] + 0.5) * 50
            ax.plot(ox, oy, color="#1f77b4", lw=1)
            dz = one["target"][0, k, :n]
            ax.plot(dz[:, 0], dz[:, 1], color="#d62728", lw=1)
            ax.plot(gh[k, :n, 0], gh[k, :n, 1], color="#d62728", lw=1, ls="--", alpha=0.6)
        ax.set_xlim(0, 47)
        ax.set_title("offence (blue), real defence (red), team ghost mean (dashed)", fontsize=9)
        fig.savefig(run / "overlay.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
