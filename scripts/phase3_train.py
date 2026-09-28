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
from ghost.ghost.dataset import build_arrays  # noqa: E402
from ghost.ghost.model import GhostConfig  # noqa: E402
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
    args = ap.parse_args()

    cfg = D.DataConfig.load(game_set=args.game_set)
    base = cfg.processed_dir / cfg.game_set
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    pt = load_game_set(base / "frames", sorted(P["game_id"].unique().to_list()), P)
    fill_shot_clock(pt, {g: D.shot_clock(g, cfg) for g in np.unique(pt.game_id)})
    arrs = build_arrays(pt, P)
    split = pt.game_id.astype(object)
    smap = dict(zip(P["game_id"].to_list(), P["split"].to_list(), strict=False))
    split = np.array([smap[g] for g in pt.game_id])
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    sub = lambda ix: {k: v[ix] for k, v in arrs.items()}  # noqa: E731
    train, val = sub(tr), sub(va)
    print(
        f"{cfg.game_set}: train {len(tr)} / val {len(va)} half-court possessions; device {args.device}"
    )

    mcfg = GhostConfig(d_model=args.d_model, n_blocks=args.blocks, mode=args.mode)
    tcfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss=args.loss,
        schedule=args.schedule,
        device=args.device,
        max_batches_per_epoch=args.max_batches,
        amp=args.amp,
        checkpoint_path=str(
            Path("runs/phase3") / f"{cfg.version}_{cfg.game_set}_{args.tag}" / "checkpoint.pt"
        ),
    )
    ck = Path(tcfg.checkpoint_path)
    if ck.exists() and not args.resume:
        ck.unlink()  # fresh run unless --resume
    model, hist = fit(train, val, mcfg, tcfg)

    # baseline with the HMM emission fitted on the same game set (Phase 2)
    fitp = Path("reports/phase2") / f"{cfg.version}_{cfg.game_set}" / "matchup_fit.json"
    gamma, sigma = (0.62, 0.11, 0.27), 3.0
    if fitp.exists():
        prm = json.loads(fitp.read_text())["models"]["hmm"]["params"]
        gamma, sigma = prm["gamma"]["all"], prm["sigma_ft"]["all"]
    result = {
        "config": config_dict(mcfg, tcfg),
        "n_params": sum(p.numel() for p in model.parameters()),
        "val_team": evaluate(model, val, "team", args.device),
        "val_individual": evaluate(model, val, "individual", args.device),
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
        },
        "note": "NLL = -log density of the true defender position in ft^-2 units, mean per "
        "defender-step; lower is better.",
    }
    run = Path("runs/phase3") / f"{cfg.version}_{cfg.game_set}_{args.tag}"
    run.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run / "model.pt")
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
