"""SSAC27 Figure 1: the ghost explained in one screen.

Possession (a) of Figure C1 (make_figure_learned_ghost_frames.py: same fixed selection
rule and seed; an on-ball screen with a switch in a held-out test game), at screen contact, one
second later and the shot release. Each panel: the ten players and the ball; for every defender,
his individual ghost (the learned ghost with only him hidden, his four teammates real): its most
likely position (open circle) joined to the real defender by a line, labelled with the gap in
feet when it is at least GAP_LABEL_FT; for the screened defender also the 50 / 80 / 95 % regions.
The screened defender is the focus (full colour, gap labels); the
other four defenders and their ghost spots are drawn light and grey, without labels.
The team ghost is not drawn (its slots can coincide; Appendix C).

Output: paper/ssac27/figures/ghost_explainer.png and .json (window, steps, gaps). Positions at
three moments are rendered; no tracking data is written (D-019, D-029).

Usage:
    uv run python scripts/ssac_figure1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_figure_learned_ghost_frames as lgf  # noqa: E402
from ghost import data as D  # noqa: E402
from ghost.ghost.dataset import load_arrays  # noqa: E402
from ghost.viz.court import draw_court  # noqa: E402
from ghost.viz.overlay import BALL, DEF, OFF  # noqa: E402
from ssac_figures import STYLE  # noqa: E402

OUT = ROOT / "paper/ssac27/figures"
GAP_LABEL_FT = 3.0
REGION_COLORS = ["#fcd5d5", "#f4a3a3", "#e06666"]


def main() -> None:
    torch.set_grad_enabled(False)
    plt.rcParams.update(STYLE)
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    rng = np.random.default_rng(lgf.SEED)
    chosen, rule = lgf.select(base, rng)
    c = chosen[0]  # the possession with a switch (row a of Figure C1)
    shots = pl.read_parquet(base / "analysis" / "shots.parquet").filter(
        (pl.col("game_id") == c["game_id"]) & (pl.col("possession_id") == c["possession_id"])
    )
    A, K = load_arrays(base / "ghost_arrays", "test", [c["game_id"]])
    K = K.with_row_index("i")
    model = lgf.ovl.lgd.load_model(ROOT / "runs/phase3" / f"{cfg.version}_all_{lgf.MODEL_RUN}",
                                   A["valid"].shape[1], "cpu")  # fmt: skip
    r = K.filter(pl.col("possession_id") == c["possession_id"]).row(0, named=True)
    i, ids = r["i"], list(r["def_ids"])
    d = ids.index(c["screened_def_id"])
    f, x = torch.from_numpy(A["feats"][i : i + 1]), torch.from_numpy(A["ctx"][i : i + 1])
    ind = []
    for dd in range(5):
        m = torch.zeros(1, 5, dtype=torch.bool)
        m[0, dd] = True
        ind.append(lgf.ovl.mixture(model, f, x, m))
    feats, target = A["feats"][i], A["target"][i]
    off = np.stack([(feats[1:6, :, 0] + 0.5) * 47.0, (feats[1:6, :, 1] + 0.5) * 50.0], -1)
    ball = np.stack([(feats[0, :, 0] + 0.5) * 47.0, (feats[0, :, 1] + 0.5) * 50.0], -1)
    handler = feats[1:6, :, 4] > 0.5
    k0, last = c["contact_step"], c["last_step"]
    k_rel = int(round((shots["t_release_ms"][0] - c["t_start"]) / 200)) if shots.height else None
    k3 = k_rel if k_rel is not None and k0 + lgf.AFTER + 2 <= k_rel <= last else last
    sw = "the defense switched; " if c["switched"] else ""
    steps = [(k0, "1  The screen", "screened defender (black edge):\n{g:.0f} ft from the league's spot"),
             (k0 + lgf.AFTER, "2  One second later", sw + "he is\n{g:.0f} ft from the league's spot"),
             (k3, "3  The shot" if k3 == k_rel else "3  End of possession",
              "at the shot he is\n{g:.0f} ft from the league's spot")]  # fmt: skip
    h0 = int(np.argmax(handler[:, k0])) if handler[:, k0].any() else -1
    cand = [j for j in range(5) if j != h0]
    screener = cand[int(np.argmin([np.linalg.norm(off[j, k0] - target[d, k0]) for j in cand]))]
    gx, gy = np.meshgrid(np.linspace(0, 47, 236), np.linspace(0, 50, 251))
    grid = np.stack([gx, gy], -1)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.3))
    gaps = []
    for ax, (t, title, note) in zip(axs, steps, strict=True):
        draw_court(ax, color="0.6", lw=0.6)
        ax.set_xlim(0, 47)
        ax.set_ylim(0, 50)
        ax.set_aspect("equal")
        ax.axis("off")
        w, mu, sg = ind[d]["w"][d, t], ind[d]["mu"][d, t], ind[d]["sigma"][d, t]
        lev = lgf.hpd_levels(w, mu, sg, rng)
        dens = lgf.mixture_density(w, mu, sg, grid)
        ax.contourf(gx, gy, dens, levels=[lev[0], lev[1], lev[2], dens.max() + 1],
                    colors=REGION_COLORS, alpha=0.8, zorder=1)  # fmt: skip
        row = {}
        for dd in range(5):
            mo = lgf.slot_mode(ind[dd]["w"][dd, t], ind[dd]["mu"][dd, t], ind[dd]["sigma"][dd, t])
            real = target[dd, t]
            gap = float(np.linalg.norm(real - mo))
            row[str(dd)] = round(gap, 2)
            focus = dd == d
            ax.plot([real[0], mo[0]], [real[1], mo[1]], color=DEF if focus else "0.7",
                    lw=1.2 if focus else 0.6, ls="--", zorder=3)  # fmt: skip
            ax.scatter(*mo, s=40 if focus else 16, facecolor="white", ec=DEF if focus else "0.65",
                       lw=1.3 if focus else 0.7, zorder=3)  # fmt: skip
            if focus and gap >= GAP_LABEL_FT:
                mid = (real + mo) / 2
                ax.text(mid[0] + 1.0, mid[1] + 1.0, f"{gap:.0f} ft", fontsize=7, color=DEF,
                        fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))  # fmt: skip
        gaps.append({"step": int(t), "gap_ft_by_defender": row})
        for j in range(5):
            ax.scatter(*off[j, t], s=30, c=OFF, ec="k" if handler[j, t] else OFF, lw=1.1,
                       marker="s" if j == screener else "o", zorder=4)  # fmt: skip
            ax.scatter(*target[j, t], s=42 if j == d else 22, c=DEF, ec="k" if j == d else DEF,
                       lw=1.3, alpha=1.0 if j == d else 0.4, zorder=5)  # fmt: skip
        ax.scatter(*ball[t], s=12, c=BALL, zorder=6)
        ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
        ax.text(0.0, -0.03, note.format(g=row[str(d)], m=max(row.values())),
                transform=ax.transAxes, fontsize=7, va="top", color="0.25")  # fmt: skip
    handles = [
        Line2D([], [], marker="o", ls="", color=OFF, label="offense (black edge: ball handler)"),
        Line2D([], [], marker="s", ls="", color=OFF, label="screener"),
        Line2D([], [], marker="o", ls="", color=DEF, mec="k", label="screened defender"),
        Line2D([], [], marker="o", ls="", color=DEF, alpha=0.4, label="other defenders"),
        Line2D([], [], marker="o", ls="--", color=DEF, mfc="white", label="where a league-average "
               "defender would stand, and the gap (grey: the other four)"),
        plt.Rectangle((0, 0), 1, 1, fc="#e06666", alpha=0.8, label="screened defender: the league's "
                      "50 / 80 / 95 % regions"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.1))  # fmt: skip
    fig.subplots_adjust(wspace=0.05, bottom=0.12)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "ghost_explainer.png", bbox_inches="tight")
    plt.close(fig)
    (OUT / "ghost_explainer.json").write_text(json.dumps(
        {"model_run": lgf.MODEL_RUN, "selection_rule": rule, "window_uid": c["window_uid"],
         "game_id": c["game_id"], "possession_id": int(c["possession_id"]),
         "switched": bool(c["switched"]), "steps": [s for s, _, _ in steps],
         "screened_slot": d, "gaps": gaps}, indent=1))  # fmt: skip
    print("ghost_explainer.png", gaps)


if __name__ == "__main__":
    main()
