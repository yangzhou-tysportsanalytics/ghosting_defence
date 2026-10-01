"""Regenerate all paper figures and tables from saved results (no heavy computation).

Reads the JSON/CSV outputs of the analysis scripts and writes
    paper/figures/*.png  and  paper/tables/*.tex
Each item is skipped with a message if its input does not exist yet.

Usage:
    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

FIG = Path("paper/figures")
TAB = Path("paper/tables")
from ghost import data as D  # noqa: E402

V = D.DataConfig.load().version


def _load(p: str):
    path = Path(p)
    if not path.exists():
        return None
    return json.loads(path.read_text()) if path.suffix == ".json" else pl.read_csv(path)


def fig_team_screen() -> str:
    T = _load("reports/analysis/team_screen_defense.csv")
    if T is None:
        return "skip team_screen (run analysis_team_screen_defense.py)"
    x, y = T["switch_rate"].to_numpy(), T["help_rate_3s"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.errorbar(
        x,
        y,
        xerr=[x - T["switch_lo"].to_numpy(), T["switch_hi"].to_numpy() - x],
        yerr=[y - T["help_lo"].to_numpy(), T["help_hi"].to_numpy() - y],
        fmt="none",
        ecolor="0.8",
        lw=0.8,
    )
    ax.scatter(x, y, s=18, color="#1f77b4")
    for t, xi, yi in zip(T["team"], x, y, strict=True):
        ax.annotate(t, (xi, yi), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("switch rate on on-ball screens")
    ax.set_ylabel("third-defender help within 3 s")
    fig.savefig(FIG / "team_screen_defense.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return "team_screen_defense.png"


def fig_help_vs_points() -> str:
    T = _load("reports/analysis/team_screen_defense.csv")
    v = _load("reports/analysis/team_validity.json")
    pts = _load("reports/analysis/team_points_allowed.csv")
    if T is None or v is None or pts is None:
        return "skip help_vs_points (needs team_points_allowed.csv from analysis_team_validity.py)"
    T = T.join(pts, on="team_id")
    x, y = T["help_rate_3s"].to_numpy(), T["pts_allowed_per_poss"].to_numpy()
    r = v["help_rate_3s__vs__pts_allowed_per_poss"]
    rob = v["help_rate_3s__vs__pts_allowed_per_poss__robustness"]["without_SAS"]
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    # 95 % game-bootstrap intervals: help rate (analysis_team_screen_defense.py) and points
    # allowed (analysis_team_validity.py)
    ax.errorbar(
        x,
        y,
        xerr=[x - T["help_lo"].to_numpy(), T["help_hi"].to_numpy() - x],
        yerr=[y - T["pts_lo"].to_numpy(), T["pts_hi"].to_numpy() - y],
        fmt="none",
        ecolor="0.82",
        lw=0.8,
        zorder=1,
    )
    ax.scatter(x, y, s=18, color="#1f77b4", zorder=2)
    xx = np.linspace(x.min(), x.max(), 10)
    b1, b0 = np.polyfit(x, y, 1)
    ax.plot(
        xx, b0 + b1 * xx, color="#d62728", lw=1.2, label=f"all 30 teams, r = {r['pearson_r']:.2f}"
    )
    keep = (T["team"] != "SAS").to_numpy()
    c1, c0 = np.polyfit(x[keep], y[keep], 1)
    ax.plot(
        xx, c0 + c1 * xx, color="#d62728", lw=1.0, ls="--",
        label=f"without SAS, r = {rob['pearson_r']:.2f}",
    )  # fmt: skip
    for t, xi, yi in zip(T["team"], x, y, strict=True):
        ax.annotate(t, (xi, yi), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("share of on-ball screens with third-defender help within 3 s")
    ax.set_ylabel("points allowed per possession (tracked games)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"r = {r['pearson_r']:.2f} (p = {r['p']:.3f}), 30 teams", fontsize=9)
    fig.savefig(FIG / "help_vs_points.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return "help_vs_points.png"


def fig_reliability(sfx: str = "") -> str:
    """sfx "" = base context adjustment (SSAC abstract); "_phases" = main specification (D-021)."""
    rel = _load(f"reports/phase4/{V}_all/reliability{sfx}.json")
    if rel is None:
        return "skip reliability (run phase4_reliability.py)"
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for key, lab, col in (
        ("sag_ft", "sag toward the rim", "#1f77b4"),
        ("dev_ft", "|deviation|", "#ff7f0e"),
    ):
        for curve, ls, suffix in (
            ("stability_curve_adjusted", "-", "all positions"),
            ("stability_curve_within_position", "--", "within listed position"),
        ):
            c = rel["metrics"][key].get(curve)
            if not c:
                continue
            n = sorted(int(k) for k in c)
            ax.plot(n, [c[str(k)]["r"] for k in n], marker="o", ms=3, color=col, ls=ls,
                    label=f"{lab}, {suffix}")  # fmt: skip
    ax.axhline(0.3, color="0.6", ls=":", lw=0.8)
    ax.text(
        305, 0.31, "decision threshold r = 0.3", fontsize=7, color="0.4", ha="right", va="bottom"
    )
    ax.set_xlabel("defensive possessions per player in each half (odd / even games)")
    ax.set_ylabel("split-half correlation across players")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.savefig(FIG / f"reliability_curve{sfx}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return f"reliability_curve{sfx}.png"


def fig_team_effects(sfx: str = "") -> str:
    h = _load(f"reports/phase4/{V}_all/hier{sfx}.json")
    if h is None:
        return "skip team_effects (run phase4_hier.py)"
    te = h["metrics"]["sag_ft"]["team_effects"]
    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    for i, r in enumerate(te):
        ax.plot([r["lo95"], r["hi95"]], [i, i], color="0.6")
        ax.scatter([r["mean"]], [i], color="#1f77b4", s=12)
    ax.set_yticks(range(len(te)), [r["team"] for r in te], fontsize=7)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("team mean sag relative to the rule ghost (ft), 95% interval")
    fig.savefig(FIG / f"team_sag_effects{sfx}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return f"team_sag_effects{sfx}.png"


def tab_franks() -> str:
    m = _load(f"reports/phase2/{V}_all/matchup_fit.json")
    if m is None:
        return "skip franks table"
    g = m["models"]["hmm"]["params"]["gamma"]["all"]
    gs = m["models"]["hmm_strat_help"]["params"]["gamma"]
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"source & man & ball & rim \\",
        r"\hline",
        r"Franks et al. (2015) & 0.62 & 0.11 & 0.27 \\",
        rf"This work & {g[0]:.3f} & {g[1]:.3f} & {g[2]:.3f} \\",
        rf"This work, strong side & {gs['strong'][0]:.3f} & {gs['strong'][1]:.3f} & {gs['strong'][2]:.3f} \\",
        rf"This work, weak side & {gs['weak'][0]:.3f} & {gs['weak'][1]:.3f} & {gs['weak'][2]:.3f} \\",
        r"\hline",
        r"\end{tabular}",
    ]
    (TAB / "franks_weights.tex").write_text("\n".join(lines) + "\n")
    return "franks_weights.tex"


def tab_xfg() -> str:
    x = _load(f"reports/phase4/{V}_all/xfg.json")
    if x is None:
        return "skip xfg table (run phase4_xfg.py)"
    lines = [r"\begin{tabular}{lccc}", r"\hline", r"model & log loss & Brier & AUC \\", r"\hline"]
    for name, m in x["models"].items():
        t = m["test"]
        lines.append(
            rf"{name.replace('_', ' ')} & {t['log_loss']:.4f} & {t['brier']:.4f} & {t['auc']:.3f} \\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    (TAB / "xfg_models.tex").write_text("\n".join(lines) + "\n")
    return "xfg_models.tex"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    for name, f in (
        ("fig_team_screen", fig_team_screen),
        ("fig_help_vs_points", fig_help_vs_points),
        ("fig_reliability", fig_reliability),
        ("fig_reliability_phases", lambda: fig_reliability("_phases")),
        ("fig_team_effects", fig_team_effects),
        ("fig_team_effects_phases", lambda: fig_team_effects("_phases")),
        ("tab_franks", tab_franks),
        ("tab_xfg", tab_xfg),
    ):
        print(name, "->", f())


if __name__ == "__main__":
    main()
