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


def _ghost_eval(tag: str):
    return _load(f"reports/phase3/{V}_all_{tag}_eval.json")


def fig_ghost_calibration(tag: str, fig_dir: Path) -> str:
    """Highest-density coverage vs nominal and PIT histograms (test games if scored, else val)."""
    e = _ghost_eval(tag)
    if e is None:
        return f"skip ghost calibration ({tag} not evaluated)"
    where = "test" if e.get("test") else "val"
    cal = e["test"]["calibration"] if where == "test" else e["calibration_val"]
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for key, lab, col in (
        ("individual", "individual ghost", "#d62728"),
        ("team", "team ghost", "0.3"),
    ):
        c = cal[key]["coverage"]
        nom = sorted(c, key=float)
        axs[0].plot([float(k) for k in nom], [c[k] for k in nom], marker="o", color=col, label=lab)
        h = np.asarray(cal[key]["pit_hist"])
        edges = np.linspace(0, 1, len(h) + 1)
        axs[1].step(edges, np.append(h, h[-1]), where="post", color=col, label=lab)
    axs[0].plot([0.4, 1], [0.4, 1], color="0.6", ls=":", lw=0.8)
    axs[0].set_xlabel("nominal coverage")
    axs[0].set_ylabel("observed coverage")
    axs[0].legend(fontsize=8, frameon=False)
    axs[1].axhline(0.1, color="0.6", ls=":", lw=0.8)
    axs[1].set_ylim(0, 0.2)  # uniform = 0.1; a zero-based axis does not exaggerate noise
    fig.subplots_adjust(wspace=0.35)
    axs[1].set_xlabel("HPD probability integral transform")
    axs[1].set_ylabel("share of defender-steps")
    fig.suptitle(f"learned ghost calibration ({where} games)", fontsize=9)
    out = fig_dir / f"ghost_calibration_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_ghost_multimodality(tag: str, fig_dir: Path) -> str:
    m = _load(f"reports/phase3/{V}_all_{tag}_multimodality.json")
    if m is None:
        return f"skip multimodality ({tag})"
    sh = m["share_bimodal_steps"]
    labs = [
        ("switch", "after a screen,\nswitch"),
        ("no_switch", "after a screen,\nno switch"),
        ("away", "away from\nscreens"),
    ]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.bar([b for _, b in labs], [sh[a] for a, _ in labs], color=["#d62728", "#ff9896", "0.6"])
    for i, (a, _) in enumerate(labs):
        ax.text(i, sh[a] + 0.02, f"{sh[a]:.0%}\n(n = {m['n'][a]:,})", ha="center", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of steps with two modes")
    ax.set_title("screened defender's ghost (held-out games)", fontsize=9)
    out = fig_dir / f"ghost_multimodality_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.name


def tab_ghost_vs_baseline(tag: str, tab_dir: Path) -> str:
    """Learned ghost vs rule baselines: NLL and RMSE of the mixture mean (test if scored)."""
    e = _ghost_eval(tag)
    if e is None:
        return f"skip ghost table ({tag})"
    nan = float("nan")
    if e.get("test"):
        t, where = e["test"], "test"
        b, bi = t["baseline_attacker_mixture"], t["baseline_individual_leftover"]
        team = (t["team"]["slot_mixture_nll_ft"], b["nll_ft"], t["team"]["rmse_ft"],
                b["rmse_ft_best_permutation"])  # fmt: skip
        ind = (t["individual"]["nll_ft"], bi["nll_ft"], t["individual"]["rmse_ft"], bi["rmse_ft"])
    else:
        where = "val"
        b, bi = e["baseline_attacker_mixture"], e.get("baseline_individual_leftover") or {}
        team = (e["val_team"]["slot_mixture_nll_ft"], b["val_nll_ft"], e["val_team"]["rmse_ft"],
                b["val_rmse_ft_best_permutation"])  # fmt: skip
        ind = (e["val_individual"]["nll_ft"], bi.get("nll_ft", nan),
               e["val_individual"]["rmse_ft"], bi.get("rmse_ft", nan))  # fmt: skip
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        rf"{where} games & NLL ghost & NLL rule & RMSE ghost (ft) & RMSE rule (ft) \\",
        r"\hline",
    ]
    for lab, (a, b_, c, d) in (("team ghost (identity-free)", team), ("individual ghost", ind)):
        lines.append(rf"{lab} & {a:.3f} & {b_:.3f} & {c:.2f} & {d:.2f} \\")
    lines += [r"\hline", r"\end{tabular}"]
    out = tab_dir / f"ghost_vs_baseline_{tag}.tex"
    out.write_text("\n".join(lines) + "\n")
    return out.name


LEARNED_FINAL = "gpu_d128_lr1e-3_final"  # chosen model scored on test (D-026)
LEARNED_CF = "cf_gpu_d128_lr1e-3"  # its cross-fitted deviations


def fig_ghost_calibration_context(tag: str, fig_dir: Path) -> str:
    """Coverage of the 80 % and 95 % regions by game situation (test games)."""
    c = _load(f"reports/phase3/{V}_all_{tag}_calibration_by_context.json")
    if c is None:
        return f"skip calibration by context ({tag})"
    ctxs = [("release", "shot\nrelease"), ("screen", "around\nscreens"), ("early", "first 2 s"),
            ("other", "other")]  # fmt: skip
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    for ax, (ghost, title) in zip(axs, (("individual", "individual ghost"),
                                         ("team", "team ghost")), strict=True):  # fmt: skip
        x = np.arange(len(ctxs))
        for q, off, col in (("0.8", -0.18, "#ff9896"), ("0.95", 0.18, "#d62728")):
            v = [c["contexts"][ghost][k]["coverage"][q] for k, _ in ctxs]
            # points, not bars: coverage has no meaningful zero on this scale
            ax.scatter(x + off, v, s=36, color=col, zorder=3, label=f"nominal {float(q):.0%}")
            for xi, vi in zip(x + off, v, strict=True):
                ax.text(xi, vi + 0.006, f"{vi:.3f}", ha="center", fontsize=6.5)
            ax.axhline(float(q), color=col, ls=":", lw=0.9)
        ax.set_xticks(x, [lab for _, lab in ctxs], fontsize=8)
        ax.set_xlim(-0.6, len(ctxs) - 0.4)
        ax.set_ylim(0.74, 1.0)
        ax.set_title(title, fontsize=9)
    axs[0].set_ylabel("observed coverage")
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("learned ghost: coverage by game situation (test games)", fontsize=9)
    out = fig_dir / f"ghost_calibration_by_context_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_ghost_ablations(fig_dir: Path) -> str:
    """Validation NLL of the chosen model, its ablations and the conditioned ghosts."""
    variants = [("gpu_d128_lr1e-3", "chosen\n(anchored, NLL)"), ("gpu_d128_lr1e-3_noanchor",
                "unanchored"), ("gpu_d128_lr1e-3_l2", "L2 loss"), ("gpu_d128_lr1e-3_lineup",
                "+ lineup"), ("gpu_d128_lr1e-3_scheme", "+ team")]  # fmt: skip
    ev = [(lab, _ghost_eval(t)) for t, lab in variants]
    if any(e is None for _, e in ev):
        return "skip ablations (not all variants evaluated)"
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, (key, sub, base, title) in zip(axs, (
            ("val_team", "slot_mixture_nll_ft", ("baseline_attacker_mixture", "val_nll_ft"), "team ghost"),
            ("val_individual", "nll_ft", ("baseline_individual_leftover", "nll_ft"), "individual ghost"),
    ), strict=True):  # fmt: skip
        v = [e[key][sub] for _, e in ev]
        cols = ["#d62728"] + ["0.45"] * (len(v) - 1)
        ax.scatter(range(len(v)), v, s=40, color=cols, zorder=3)  # NLL has no natural zero
        rule = ev[0][1][base[0]][base[1]]
        ax.axhline(rule, color="0.2", ls="--", lw=0.9, label=f"rule ghost {rule:.2f}")
        span = max(max(v), rule) - min(v)
        ax.set_ylim(min(v) - 0.08 * span, max(max(v), rule) + 0.1 * span)
        ax.set_xlim(-0.5, len(v) - 0.5)
        for i, y in enumerate(v):
            ax.text(i, y + 0.03 * span, f"{y:.3f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(v)), [lab for lab, _ in ev], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="center right")
    axs[0].set_ylabel("validation NLL (lower is better)")
    out = fig_dir / "ghost_ablations.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_release_baseline(fig_dir: Path) -> str:
    """B3: NLL at the shot release - learned ghost vs single-moment density networks vs rule."""
    r = _load(f"reports/phase3/{V}_all_release_density_baseline.json")
    if r is None:
        return "skip release baseline (run analysis_release_density_baseline.py)"
    m = r["models"]
    rows = [("individual task", [m["ind_snapshot"]["test_nll"]["learned_ghost"],
                                 m["ind_snapshot"]["test_nll"]["mdn_snapshot"], None,
                                 m["ind_snapshot"]["test_nll"]["rule_ghost"]]),
            ("team task", [m["team_snapshot"]["test_nll"]["learned_ghost"],
                           m["team_snapshot"]["test_nll"]["mdn_snapshot"],
                           m["team_shot_features"]["test_nll"]["mdn_snapshot"],
                           m["team_snapshot"]["test_nll"]["rule_ghost"]])]  # fmt: skip
    labs = ["learned ghost\n(possession history)", "single-moment\nnetwork",
            "few shot\nfeatures", "rule ghost"]  # fmt: skip
    cols = ["#d62728", "#1f77b4", "#9ecae1", "0.55"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, (title, vals) in zip(axs, rows, strict=True):
        xs = [i for i, v in enumerate(vals) if v is not None]
        ax.scatter(xs, [vals[i] for i in xs], s=44, color=[cols[i] for i in xs], zorder=3)
        lo, hi = min(vals[i] for i in xs), max(vals[i] for i in xs)
        for i in xs:
            ax.text(i, vals[i] + 0.05 * (hi - lo), f"{vals[i]:.2f}", ha="center", fontsize=7)
        ax.set_xticks(range(4), labs, fontsize=7)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.15 * (hi - lo))  # NLL: points, no zero baseline
        ax.set_title(title, fontsize=9)
    axs[0].set_ylabel("test NLL at the release")
    fig.suptitle(f"shot release, {r['n_test_shots']:,} test shots", fontsize=9)
    out = fig_dir / "release_density_baseline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_learned_vs_rule_reliability(fig_dir: Path) -> str:
    """Reliability (within-team split-half r) and independence of position / size / guarded
    threat, for the rule-ghost and learned-ghost player measures (main specification)."""
    rel = _load(f"reports/phase4/{V}_all/reliability_learned_{LEARNED_CF}_phases.json")
    mm = _load(f"reports/analysis/meta_metrics_learned_{LEARNED_CF}.json")
    if rel is None or mm is None:
        return "skip learned reliability (run phase4_reliability / analysis_meta_metrics)"
    items = [("sag_ft", "sag\n(rule)"), ("dev_ft", "deviation\n(rule)"),
             ("nll_ind", "individual NLL\n(learned)"), ("nll_team", "team NLL\n(learned)"),
             ("d_mean", "distance to\nmean (learned)")]  # fmt: skip
    r = [rel["metrics"][k]["split_half_r_player_within_team"] for k, _ in items]
    ind = [mm["metrics"][f"{k}_within_team"]["independence"]["position_height_guarded_threat"]
           ["independence"] for k, _ in items]  # fmt: skip
    x = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    ax.bar(x - 0.18, r, width=0.34, color="0.45", label="split-half reliability (within team)")
    ax.bar(x + 0.18, ind, width=0.34, color="#d62728",
           label="independence of position, height, guarded threat")  # fmt: skip
    for i in range(len(items)):
        ax.text(i - 0.18, r[i] + 0.02, f"{r[i]:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.18, ind[i] + 0.02, f"{ind[i]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, [lab for _, lab in items], fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
    ax.set_title("player measures: reliability and what they add beyond role", fontsize=9)
    out = fig_dir / "learned_vs_rule_reliability.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.name


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ghost-tag", default=LEARNED_FINAL, help="learned-ghost run tag (final model)"
    )
    ap.add_argument("--out-dir", default=None, help="ghost figures go here (prototypes only)")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    if args.ghost_tag:
        fd = Path(args.out_dir) if args.out_dir else FIG
        td = Path(args.out_dir) if args.out_dir else TAB
        fd.mkdir(parents=True, exist_ok=True)
        print("fig_ghost_calibration ->", fig_ghost_calibration(args.ghost_tag, fd))
        print("fig_ghost_multimodality ->", fig_ghost_multimodality(args.ghost_tag, fd))
        print("tab_ghost_vs_baseline ->", tab_ghost_vs_baseline(args.ghost_tag, td))
        print("fig_ghost_calibration_context ->", fig_ghost_calibration_context(args.ghost_tag, fd))
        if args.out_dir:
            return  # prototypes: only the ghost figures, kept out of paper/
    for name, f in (
        ("fig_team_screen", fig_team_screen),
        ("fig_help_vs_points", fig_help_vs_points),
        ("fig_reliability", fig_reliability),
        ("fig_reliability_phases", lambda: fig_reliability("_phases")),
        ("fig_team_effects", fig_team_effects),
        ("fig_team_effects_phases", lambda: fig_team_effects("_phases")),
        ("tab_franks", tab_franks),
        ("tab_xfg", tab_xfg),
        ("fig_ghost_ablations", lambda: fig_ghost_ablations(FIG)),
        ("fig_release_baseline", lambda: fig_release_baseline(FIG)),
        ("fig_learned_vs_rule_reliability", lambda: fig_learned_vs_rule_reliability(FIG)),
    ):
        print(name, "->", f())


if __name__ == "__main__":
    main()
