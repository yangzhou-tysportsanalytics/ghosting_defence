"""Figure C: the learned ghost at on-ball screens, two held-out possessions (writing request
2026-10-23; D-029).

Selection (fixed rule, no manual choice):
* on-ball screens of the test games with screen_confidence >= 0.81 (as the overlay animations);
* the possession runs at least 1.4 s after the screen contact;
* the screened defender's individual-ghost NLL on the possession (cross-fitted deviations,
  phase4_learned_ghost_dev.py) lies between the league 25th and 75th percentiles of all
  (possession, defender) rows - a typical, not a striking, possession;
* one random draw (seed 9) among the eligible screens with a switch and one among those without.
Panels per possession: screen contact; contact + 1 s; shot release (or the last step of the
possession if no tracked release, or if the release is less than 0.4 s after the second panel).
Each panel: real positions of the ten players and the ball; the screened defender's individual
ghost (other four defenders real) as 50 / 80 / 95 % highest-density regions; the team ghost's
mode for each of the five defender slots (D-007: the most likely position, not the mean); the
screener (the attacker other than the handler nearest the screened defender at contact) marked.
Model: the final learned ghost (trained on the train games, so the test games are held out).

Outputs: paper/figures/learned_ghost_frames.png and paper/figures/learned_ghost_frames.json (the
chosen window_uid values, rule, seed, steps). Positions at six moments of two possessions are
rendered; no tracking data is written (D-019, D-029).

Usage:
    uv run python scripts/make_figure_learned_ghost_frames.py [--out paper/figures]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ghost import data as D  # noqa: E402
from ghost.ghost.dataset import load_arrays  # noqa: E402
from ghost.viz.court import draw_court  # noqa: E402
from ghost.viz.overlay import BALL, DEF, OFF  # noqa: E402

_p = Path(__file__).resolve().parent / "phase3_overlay.py"
_spec = importlib.util.spec_from_file_location("ovl", _p)
ovl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ovl)

MODEL_RUN = "gpu_d128_lr1e-3_final"
CROSSFIT = "cf_gpu_d128_lr1e-3"
SEED = 9
MIN_CONF = 0.81
AFTER = 5  # steps (1 s) from contact to the second panel
LEVELS = (0.95, 0.8, 0.5)


def mixture_density(w, mu, sigma, xy: np.ndarray) -> np.ndarray:
    """Isotropic Gaussian mixture density at points xy (..., 2)."""
    d2 = ((xy[..., None, :] - mu) ** 2).sum(-1)
    return (w * np.exp(-0.5 * d2 / sigma**2) / (2 * np.pi * sigma**2)).sum(-1)


def hpd_levels(w, mu, sigma, rng, n: int = 20000) -> list[float]:
    """Density thresholds of the 95 / 80 / 50 % highest-density regions (Monte Carlo)."""
    comp = rng.choice(len(w), size=n, p=w / w.sum())
    pts = mu[comp] + sigma[comp, None] * rng.standard_normal((n, 2))
    dens = mixture_density(w, mu, sigma, pts)
    return [float(np.quantile(dens, 1 - q)) for q in LEVELS]


def slot_mode(w, mu, sigma) -> np.ndarray:
    """Component mean with the highest mixture density (D-007)."""
    return mu[int(np.argmax(mixture_density(w, mu, sigma, mu)))]


def select(base: Path, rng) -> tuple[list[dict], dict]:
    root = base / "ghost_arrays" / "test"
    K = pl.read_parquet(root / "keys.parquet")
    P = pl.read_parquet(base / "possessions.parquet").select(
        ["game_id", "possession_id", "t_start", "window_uid"]
    )
    dev = pl.read_parquet(base / "analysis" / f"learned_ghost_dev_{CROSSFIT}.parquet")
    q25, q75 = float(dev["nll_ind"].quantile(0.25)), float(dev["nll_ind"].quantile(0.75))
    valid = np.load(root / "valid.npy", mmap_mode="r")
    last = {(r["game_id"], r["possession_id"]): int(np.flatnonzero(valid[r["row"]])[-1])
            for r in K.iter_rows(named=True)}  # fmt: skip
    S = (
        pl.read_parquet(base / "analysis" / "onball_screens.parquet")
        .filter(pl.col("game_id").is_in(K["game_id"].unique().to_list())
                & (pl.col("screen_confidence") >= MIN_CONF))
        .join(P, on=["game_id", "possession_id"], how="inner")
        .join(dev.select(["game_id", "possession_id", "def_id", "nll_ind"])
              .rename({"def_id": "screened_def_id"}),
              on=["game_id", "possession_id", "screened_def_id"], how="inner")
        .with_columns(((pl.col("t_contact_ms") - pl.col("t_start")) / 200).round().cast(pl.Int64)
                      .alias("contact_step"))
    )  # fmt: skip
    S = S.with_columns(
        pl.struct(["game_id", "possession_id"])
        .map_elements(
            lambda s: last.get((s["game_id"], s["possession_id"]), -1), return_dtype=pl.Int64
        )
        .alias("last_step")
    )
    elig = S.filter(pl.col("nll_ind").is_between(q25, q75)
                    & (pl.col("last_step") >= pl.col("contact_step") + AFTER + 2)
                    & (pl.col("contact_step") >= 0))  # fmt: skip
    elig = elig.sort(["game_id", "possession_id", "t_contact_ms"])
    chosen = []
    counts = {}
    for sw in (True, False):
        pool = elig.filter(pl.col("switched") == sw)
        counts["switch" if sw else "no_switch"] = pool.height
        chosen.append(pool.row(int(rng.integers(pool.height)), named=True))
    rule = {"test_games": True, "screen_confidence_min": MIN_CONF,
            "nll_ind_between_league_quantiles": [0.25, 0.75], "nll_ind_bounds": [q25, q75],
            "min_steps_after_contact": AFTER + 2, "seed": SEED, "eligible_screens": counts,
            "nll_source": f"learned_ghost_dev_{CROSSFIT} (cross-fitted)"}  # fmt: skip
    return chosen, rule


def render(out_dir="paper/figures") -> Path:
    """Write learned_ghost_frames.png / .json to ``out_dir``; returns the figure path."""
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    rng = np.random.default_rng(SEED)
    chosen, rule = select(base, rng)
    shots = pl.read_parquet(base / "analysis" / "shots.parquet").filter(
        pl.col("possession_id").is_not_null()
    )
    root = base / "ghost_arrays"
    A, K = load_arrays(root, "test", sorted({c["game_id"] for c in chosen}))
    K = K.with_row_index("i")
    model = ovl.lgd.load_model(Path("runs/phase3") / f"{cfg.version}_all_{MODEL_RUN}",
                               A["valid"].shape[1], "cpu")  # fmt: skip
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 9.4))
    meta = []
    for row, c in enumerate(chosen):
        r = K.filter((pl.col("game_id") == c["game_id"])
                     & (pl.col("possession_id") == c["possession_id"])).row(0, named=True)  # fmt: skip
        i = r["i"]
        ids = list(r["def_ids"])
        d = ids.index(c["screened_def_id"])
        f, x = torch.from_numpy(A["feats"][i : i + 1]), torch.from_numpy(A["ctx"][i : i + 1])
        team = ovl.mixture(model, f, x, torch.ones(1, 5, dtype=torch.bool))
        m = torch.zeros(1, 5, dtype=torch.bool)
        m[0, d] = True
        ind = ovl.mixture(model, f, x, m)
        feats, target = A["feats"][i], A["target"][i]
        off = np.stack([(feats[1:6, :, 0] + 0.5) * 47.0, (feats[1:6, :, 1] + 0.5) * 50.0], -1)
        ball = np.stack([(feats[0, :, 0] + 0.5) * 47.0, (feats[0, :, 1] + 0.5) * 50.0], -1)
        handler = feats[1:6, :, 4] > 0.5
        k0 = c["contact_step"]
        h0 = int(np.argmax(handler[:, k0])) if handler[:, k0].any() else -1
        cand = [j for j in range(5) if j != h0]
        screener = cand[int(np.argmin([np.linalg.norm(off[j, k0] - target[d, k0]) for j in cand]))]
        sh = shots.filter((pl.col("game_id") == c["game_id"])
                          & (pl.col("possession_id") == c["possession_id"]))  # fmt: skip
        k_rel = int(round((sh["t_release_ms"][0] - c["t_start"]) / 200)) if sh.height else None
        last = c["last_step"]
        k3 = k_rel if k_rel is not None and k0 + AFTER + 2 <= k_rel <= last else last
        third = "shot release" if k3 == k_rel else "end of possession"
        steps = [(k0, "screen contact"), (k0 + AFTER, "contact + 1 s"), (k3, third)]
        gx, gy = np.meshgrid(np.linspace(0, 47, 236), np.linspace(0, 50, 251))
        grid = np.stack([gx, gy], -1)
        for col, (t, lab) in enumerate(steps):
            ax = axs[row, col]
            draw_court(ax, color="0.6")
            ax.set_xlim(0, 47)
            ax.set_ylim(0, 50)
            ax.set_aspect("equal")
            ax.axis("off")
            w, mu, sg = ind["w"][d, t], ind["mu"][d, t], ind["sigma"][d, t]
            lev = hpd_levels(w, mu, sg, rng)
            dens = mixture_density(w, mu, sg, grid)
            ax.contourf(gx, gy, dens, levels=[lev[0], lev[1], lev[2], dens.max() + 1],
                        colors=["#fcd5d5", "#f4a3a3", "#e06666"], alpha=0.75, zorder=1)  # fmt: skip
            for s in range(5):  # team ghost: each slot's mode
                mo = slot_mode(team["w"][s, t], team["mu"][s, t], team["sigma"][s, t])
                ax.scatter(*mo, s=120, facecolor="none", ec="0.35", lw=1.3, zorder=3)
            for j in range(5):
                ax.scatter(*off[j, t], s=60, c=OFF, ec="k" if handler[j, t] else OFF, lw=1.4,
                           marker="s" if j == screener else "o", zorder=4)  # fmt: skip
                ax.scatter(*target[j, t], s=60 if j == d else 45, c=DEF,
                           ec="k" if j == d else DEF, lw=1.4, zorder=5)  # fmt: skip
            ax.scatter(*ball[t], s=22, c=BALL, zorder=6)
            ax.set_title(f"({'ab'[row]}) {'switch' if c['switched'] else 'no switch'}: {lab}",
                         fontsize=10)  # fmt: skip
        meta.append({"panel_row": "ab"[row], "switched": bool(c["switched"]),
                     "window_uid": c["window_uid"], "game_id": c["game_id"],
                     "possession_id": int(c["possession_id"]), "contact_step": k0,
                     "steps": [s for s, _ in steps], "third_panel": third,
                     "screened_defender_nll_ind": float(c["nll_ind"]),
                     "screen_confidence": float(c["screen_confidence"])})  # fmt: skip
    handles = [
        Line2D([], [], marker="o", ls="", color=OFF, label="offense (black edge: ball handler)"),
        Line2D([], [], marker="s", ls="", color=OFF, label="screener"),
        Line2D([], [], marker="o", ls="", color=DEF, label="real defense (black edge: screened)"),
        Line2D([], [], marker="o", ls="", color=BALL, label="ball"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="0.35", label="team ghost: most likely "
               "position of each defender slot"),
        plt.Rectangle((0, 0), 1, 1, fc="#e06666", alpha=0.75, label="screened defender's individual "
                      "ghost: 50 / 80 / 95 % regions"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.subplots_adjust(bottom=0.1, wspace=0.04, hspace=0.12)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "learned_ghost_frames.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (out / "learned_ghost_frames.json").write_text(
        json.dumps({"model_run": MODEL_RUN, "selection_rule": rule, "possessions": meta}, indent=1)
    )
    print(json.dumps(meta, indent=1))
    return out / "learned_ghost_frames.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    render(ap.parse_args().out)


if __name__ == "__main__":
    main()
