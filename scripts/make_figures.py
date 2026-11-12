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
    p_perm = v["inference_pts_allowed_per_poss"]["help_r_permutation_p"]  # as in the text
    ax.set_title(f"r = {r['pearson_r']:.2f} (permutation p = {p_perm:.3f}), 30 teams", fontsize=9)
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
    (TAB / "franks_weights.tex").write_text("\n".join(lines) + "\n", newline="\n")
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
    (TAB / "xfg_models.tex").write_text("\n".join(lines) + "\n", newline="\n")
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
    out.write_text("\n".join(lines) + "\n", newline="\n")
    return out.name


def _n(x: float, d: int = 2, sign: bool = False) -> str:
    """Number for LaTeX: true minus sign, optional explicit plus."""
    s = f"{abs(x):.{d}f}"
    return ("$-$" if x < 0 else ("+" if sign else "")) + s


def _ci(lo: float, hi: float, d: int = 2) -> str:
    return f"({_n(lo, d)} to {_n(hi, d)})"


def _tex(name: str, cols: str, header: str, rows: list[str], note: str = "") -> str:
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\hline", header + r" \\", r"\hline"]
    lines += [r + r" \\" for r in rows] + [r"\hline", r"\end{tabular}"]
    if note:
        lines.append(r"\par\smallskip{\footnotesize " + note + "}")
    (TAB / name).write_text("\n".join(lines) + "\n", newline="\n")  # LF on every platform
    return name


def tab_screen_confidence() -> str:
    """Table 4: team help / switch rate vs points allowed by screen confidence."""
    h = _load("reports/analysis/help_timing.json")
    if h is None:
        return "skip screen confidence table"
    cs = h["confidence_sensitivity"]
    rows = []
    for key, lab in (("all", "all"), ("confidence_ge_0.5", r"confidence $\geq$ 0.5"),
                     ("top_half_ge_0.81", r"top half ($\geq$ 0.81)")):  # fmt: skip
        c = cs[key]
        a, s = c["help_rate_3s_vs_pts"], c["switch_rate_vs_pts"]
        rows.append(f"{lab} & {c['n_screens']:,} & {100 * c['league_switch_rate']:.1f}\\% & "
                    f"{_n(a['r'])} {_ci(*a['ci95'])} & {_n(s['r'])} {_ci(*s['ci95'])}")  # fmt: skip
    return _tex("screen_confidence.tex", "lrrcc",
                r"screens & $n$ & league switch rate & help $r$ (95\% CI) & switch $r$ (95\% CI)",
                rows)  # fmt: skip


def tab_team_game_panel() -> str:
    """Table 5: points allowed per 100 possessions per 10 pp of each rate (team-game panel)."""
    p = _load("reports/analysis/team_game_panel.json")
    if p is None:
        return "skip team-game panel table"
    labs = {"early_help_rate": r"early help ($\leq$ 1 s)", "help_rate_3s": "help within 3 s",
            "switch_rate": "switch rate"}  # fmt: skip
    rows = []
    for k, lab in labs.items():
        cells = []
        for spec in ("pooled", "team_fe", "team_and_opponent_fe"):
            m = p["models"][k][spec]
            cell = f"{_n(m['coef'], sign=True)} {_ci(*m['ci95_cluster'])}"
            if k == "early_help_rate" and spec == "team_and_opponent_fe":
                cell += "; game bootstrap " + _ci(*m["ci95_game_bootstrap"])[1:-1]
            cells.append(cell)
        rows.append(" & ".join([lab, *cells]))
    return _tex("team_game_panel.tex", "lccc",
                r"rate & pooled & team fixed effects & team + opponent fixed effects", rows)  # fmt: skip


def tab_team_game_panel_inference() -> str:
    """Review A2: team-game panel with two-way clustered intervals and wild-cluster p, the
    early-vs-late help contrast, and early help vs points in screen / non-screen possessions."""
    p = _load("reports/analysis/team_game_panel.json")
    if p is None or "early_vs_3s_help" not in p:
        return "skip team-game panel inference table"
    specs = (
        ("pooled", "pooled"),
        ("team_fe", "team FE"),
        ("team_and_opponent_fe", "team + opp. FE"),
    )
    labs = {"early_help_rate": r"early help ($\leq$ 1 s)", "help_rate_3s": "help within 3 s",
            "switch_rate": "switch rate"}  # fmt: skip
    rows = []
    for k, lab in labs.items():
        for s, sl in specs:
            m = p["models"][k][s]
            rows.append(f"{lab} & {sl} & {_n(m['coef'], sign=True)} & {_ci(*m['ci95_cluster'])} & "
                        f"{_ci(*m['ci95_twoway_team_opponent'])} & {m['p_wild_cluster_team']:.3f}")  # fmt: skip
    for s, sl in specs:
        d = p["early_vs_3s_help"][s]
        rows.append(f"early $-$ late help (joint model) & {sl} & {_n(d['b_difference'], sign=True)}"
                    f" & {_ci(*d['b_ci95_cluster_team'])} & & ")  # fmt: skip
    for k, kl in (("screen", "possessions with an on-ball screen"),
                  ("noscreen_halfcourt", "half-court possessions without one"),
                  ("noscreen_all", "all possessions without one")):  # fmt: skip
        for s, sl in specs:
            m = p["by_screen_possessions"][k][s]
            rows.append(f"early help, {kl} & {sl} & {_n(m['coef'], sign=True)} & "
                        f"{_ci(*m['ci95_cluster'])} & {_ci(*m['ci95_twoway_team_opponent'])} & "
                        f"{m['p_wild_cluster_team']:.3f}")  # fmt: skip
    return _tex("team_game_panel_inference.tex", "llcccc",
                "predictor / outcome & model & coefficient & 95\\% CI (team clusters) & "
                "95\\% CI (team $\\times$ opponent) & wild-cluster $p$", rows,
                note="Points allowed per 100 possessions per 10 percentage points of the rate; "
                "late help = help after 1 s and within 3 s.")  # fmt: skip


def tab_help_matched() -> str:
    """Table 6: early help adjusted for the state at screen contact (overlap weights)."""
    m = _load("reports/analysis/help_timing_matched.json")
    if m is None:
        return "skip matched table"
    c = m["comparisons"]
    rows = []
    for k, lab in (("A_early_vs_rest", "A: early help vs all other screens ({n:,} screens; "
                    "{t:,} early)"), ("B_early_vs_late_help", "B: early vs late help ({n:,} "
                    "helped screens)")):  # fmt: skip
        cells = [f"{_n(c[k][y]['ato_overlap_weighted'], 3, True)} "
                 f"{_ci(*c[k][y]['ato_ci95_game_bootstrap'], 3)}"
                 for y in ("points", "xfg_of_shot")]  # fmt: skip
        rows.append(" & ".join([lab.format(n=c[k]["n"], t=c[k]["n_treated"]), *cells]))
    return _tex("help_timing_matched.tex", "lcc",
                r"comparison & points per possession & xFG of the possession's shot", rows)  # fmt: skip


def tab_meta_metrics() -> str:
    """Table 7: meta-metrics (Franks et al. 2016) of the rule-ghost player measures."""
    mm = _load("reports/analysis/meta_metrics.json")
    if mm is None:
        return "skip meta-metrics table"
    rows = []
    for k, lab in (("sag_ft_within_team", "sag"), ("dev_ft_within_team", "deviation"),
                   ("help_b_rate", "help rate (third-defender)")):  # fmt: skip
        v = mm["metrics"][k]
        ind = v["independence"]
        ext = v["external_correlations"]
        rows.append(" & ".join([
            lab, f"{v['discrimination']:.3f}", f"{v['stability_parity']['stability']:.3f}",
            f"{v['stability_date']['stability']:.3f}",
            *[f"{ind[s]['independence']:.3f}" for s in
              ("position", "position_height", "position_height_guarded_threat")],
            _n(ext["raptor_defense"]["pearson"], sign=True), _n(ext["dbpm"]["pearson"], sign=True),
        ]))  # fmt: skip
    return _tex("meta_metrics.tex", "lcccccccc",
                r"measure & discrimination & stability (parity) & stability (date) & "
                r"independence: position & + height & + guarded threat & $r$ RAPTOR D & $r$ DBPM",
                rows)  # fmt: skip


def tab_help_sensitivity() -> str:
    """Table 8: E-values, robustness values and bounds for a confounder k = 1, 2, 3 times as
    strong as a benchmark covariate: the handler's velocity toward the rim, the covariate most
    related to the treatment and the one most related to the outcome (review A5)."""
    s = _load("reports/analysis/help_timing_sensitivity.json")
    if s is None:
        return "skip sensitivity table"
    label = {"h_v_to_rim": "handler velocity to rim", "h_dist_rim": "handler distance to rim",
             "shot_clock": "shot clock", "margin": "score margin"}  # fmt: skip
    rows = []
    for comp, cl in (("A_early_vs_rest", "A"), ("B_early_vs_late_help", "B")):
        for y, yl in (("points", "points"), ("xfg_of_shot", "xFG")):
            r = s["comparisons"][comp][y]
            marks = {}
            for nm, tag in ((r.get("strongest_observed_covariate"), "T"),
                            (r.get("strongest_observed_covariate_with_outcome"), "Y")):  # fmt: skip
                if nm:
                    marks.setdefault(nm, []).append(tag)
            first = True
            for nm in dict.fromkeys(["h_v_to_rim", *marks]):
                cells = ([f"{r['e_value_estimate']:.2f} / {r['e_value_ci_bound']:.2f}",
                          f"{100 * r['rv_q1']:.1f}\\%"] if first else ["", ""])  # fmt: skip
                bl = label.get(nm, nm.replace("_", " "))
                if marks.get(nm):
                    bl += " (" + ", ".join(marks[nm]) + ")"
                cells.append(bl)
                for b in r["bounds"][nm]:
                    if "adjusted_estimate" not in b:
                        cells.append("undefined")
                        continue
                    flip = (b["adjusted_estimate"] > 0) != (r["estimate_ols_adjusted"] > 0)
                    dag = r"$^\dagger$" if flip else ""
                    cells.append(f"{_n(b['adjusted_estimate'], 3, True)}{dag} "
                                 f"{_ci(*b['adjusted_ci95'], 3)}")  # fmt: skip
                rows.append(" & ".join([f"{cl}, {yl}" if first else "", *cells]))
                first = False
    return _tex("help_timing_sensitivity.tex", "lcclccc",
                r"comparison, outcome & E-value (estimate / bound) & RV & benchmark & $1\times$ & "
                r"$2\times$ & $3\times$", rows,
                note=r"$^\dagger$ sign reversed relative to the unadjusted estimate. Benchmarks: "
                r"T = largest partial $R^2$ with early help, Y = with the outcome.")  # fmt: skip


def tab_learned_reliability() -> str:
    """Table 10: reliability of the learned-ghost player measures and the rule-ghost ones."""
    R4 = f"reports/phase4/{V}_all"
    rl = _load(f"{R4}/reliability_learned_{LEARNED_CF}_phases.json")
    hl = _load(f"{R4}/hier_learned_{LEARNED_CF}_phases.json")
    rr, hr = _load(f"{R4}/reliability_phases.json"), _load(f"{R4}/hier_phases.json")
    if None in (rl, hl, rr, hr):
        return "skip learned reliability table"

    def cells(rel, hier, k):
        m = rel["metrics"][k]
        return (m["split_half_r_player_within_team"],
                m["within_position"]["split_half_r_within_position"],
                m["variance_shares_adjusted"]["share_team_of_systematic"],
                hier["metrics"][k]["share_player_intervals_excluding_zero"])  # fmt: skip

    rows = []
    for k, lab in (("nll_ind", "individual-ghost NLL"), ("nll_team", "team-ghost NLL"),
                   ("d_mean", "distance to individual-ghost mean")):  # fmt: skip
        a, b, c, d = cells(rl, hl, k)
        rows.append(f"{lab} & {a:.3f} & {b:.3f} & {c:.2f} & {100 * d:.0f}\\%")
    s, v = cells(rr, hr, "sag_ft"), cells(rr, hr, "dev_ft")
    rows.append(f"rule ghost: sag / deviation & {s[0]:.3f} / {v[0]:.3f} & {s[1]:.3f} / {v[1]:.3f} "
                f"& {s[2]:.2f} / {v[2]:.2f} & {100 * s[3]:.0f}\\% / {100 * v[3]:.0f}\\%")  # fmt: skip
    return _tex("learned_reliability.tex", "lcccc",
                r"metric & split-half $r$, within team & within listed position & team share of "
                r"systematic variance (descriptive) & player intervals excluding 0", rows)  # fmt: skip


def tab_ghost_coverage() -> str:
    """Review A4: HPD coverage of the learned ghosts on all test possessions, overall and by game
    situation, with game-bootstrap 95 % intervals."""
    c = _load(f"reports/phase3/{V}_all_{LEARNED_FINAL}_calibration_by_context.json")
    if c is None or "game_bootstrap" not in c:
        return "skip coverage table (run phase3_calibration_by_context.py)"
    g = c["game_bootstrap"]["coverage"]
    rows = []
    for ghost in ("individual", "team"):
        for ctx, lab in (("all", "all steps"), ("release", "shot release"), ("screen", "screen"),
                         ("early", "first 2 s"), ("other", "other")):  # fmt: skip
            v = g[ghost][ctx]
            cells = [f"{100 * v[q]['coverage']:.1f} ({100 * v[q]['ci95'][0]:.1f}--"
                     f"{100 * v[q]['ci95'][1]:.1f})" for q in ("0.5", "0.8", "0.9", "0.95")]  # fmt: skip
            rows.append(" & ".join([ghost if ctx == "all" else "", lab, f"{v['n_defender_steps']:,}",
                                    *cells]))  # fmt: skip
    return _tex("ghost_coverage.tex", "llrcccc",
                r"ghost & situation & defender-steps & 50\% & 80\% & 90\% & 95\%", rows,
                note="Coverage (\\%) of the highest-density regions on all test possessions; "
                "95\\% intervals from a game bootstrap.")  # fmt: skip


def tab_rule_reliability() -> str:
    """Split-half reliability of sag and dev (348 players), main and base specification."""
    R4 = f"reports/phase4/{V}_all"
    rel = {s: _load(f"{R4}/reliability{s}.json") for s in ("_phases", "")}
    pos = {s: _load(f"{R4}/reliability_within_position{s}.json") for s in ("_phases", "")}
    if None in (*rel.values(), *pos.values()):
        return "skip rule reliability table"
    rows = []
    for k, lab in (("sag_ft", "sag"), ("dev_ft", "dev")):
        for s, spec in (("_phases", "main (with phases)"), ("", "base")):
            m = rel[s]["metrics"][k]
            rows.append(f"{lab} & {spec if k == 'sag_ft' else spec.split(' ')[0]} & "
                        f"{m['split_half_r_adjusted']:.3f} & {m['split_half_r_player_within_team']:.3f} & "
                        f"{pos[s][k]['split_half_r_within_position']:.3f}")  # fmt: skip
    return _tex("rule_reliability.tex", "llccc",
                "metric & specification & adjusted & within team & within listed position", rows)  # fmt: skip


def tab_predictive_validity_player() -> str:
    """Player-level predictive validity (main specification), both parity directions."""
    pv = _load(f"reports/phase5/{V}_all/predictive_validity_phases.json")
    if pv is None:
        return "skip player predictive validity table"
    pl_ = pv["player_level"]
    dirs = ("trait_parity0_outcome_parity1", "trait_parity1_outcome_parity0")
    rows = []
    for y, lab in (("def1_ft", "space allowed"), ("xfg_m3_cf", "shooter-aware xFG allowed")):
        pooled = " / ".join(f"{pl_[d][y]['rho']:.2f}" for d in dirs)
        within = " / ".join(
            f"{pl_[d][y + '__within_position']['rho']:.2f} "
            f"({pl_[d][y + '__within_position']['ci95'][0]:.2f}--"
            f"{pl_[d][y + '__within_position']['ci95'][1]:.2f})"
            for d in dirs
        )
        rows.append(f"{lab} & {pooled} & {within}")
    return _tex("predictive_validity_player.tex", "lcc",
                r"outcome & pooled $\rho$ & within listed position $\rho$ (95\% CI)", rows)  # fmt: skip


def tab_second_chance() -> str:
    """Main results with and without second-chance possessions (D-018)."""
    s = _load("reports/analysis/second_chance_sensitivity.json")
    if s is None:
        return "skip second-chance table"

    def cells(m):
        t = m["team"]
        rr = [" / ".join(f"{x:.3f}" for x in m["hmm_strat_help_gamma"]["strong"]),
              f"{m['sag_ft_base']['split_half_r_within_position']:.3f} / "
              f"{m['sag_ft_phases']['split_half_r_within_position']:.3f}",
              f"{100 * m['sag_ft_base']['hier_team_share']['mean']:.1f}\\% / "
              f"{100 * m['sag_ft_phases']['hier_team_share']['mean']:.1f}\\%"]  # fmt: skip
        for k in ("help_rate_3s_vs_pts", "early_help_rate_vs_pts", "switch_rate_vs_pts"):
            rr.append(f"{_n(t[k]['r'])} {_ci(*t[k]['ci95'])}")
        rr.append(f"{m['predictive_validity']['space_per_ft_sag']['coef']:.2f} ft")
        return rr

    labs = ["strong-side weights (man, ball, rim)", "sag within-position reliability, base / main",
            "team share of sag, base / main", "help rate vs points allowed",
            "early-help rate vs points allowed", "switch rate vs points allowed",
            "space per ft of pre-release sag"]  # fmt: skip
    rows = [" & ".join(r) for r in zip(labs, cells(s["main"]), cells(s["with_sc"]), strict=True)]
    return _tex("second_chance.tex", "lcc", "quantity & main & with second chances", rows)


def tab_ghost_ablations() -> str:
    """Validation NLL of the chosen learned ghost, its ablations and the conditioned ghosts."""
    variants = (("", "chosen (anchored, NLL loss)"), ("_noanchor", "unanchored head"),
                ("_l2", "L2 loss"), ("_lineup", "+ defending lineup (player identities)"),
                ("_scheme", "+ defending team (scheme)"))  # fmt: skip
    ev = [(lab, _ghost_eval(f"gpu_d128_lr1e-3{t}")) for t, lab in variants]
    if any(e is None for _, e in ev):
        return "skip ablation table"
    rows = [f"{lab} & {e['val_team']['slot_mixture_nll_ft']:.3f} & "
            f"{e['val_individual']['nll_ft']:.3f}" for lab, e in ev]  # fmt: skip
    return _tex("ghost_ablations.tex", "lcc", "variant & team NLL & individual NLL", rows)


def tab_release_density_baseline() -> str:
    """NLL at the shot release: learned ghost vs single-moment networks vs rule ghost (B3)."""
    r = _load(f"reports/phase3/{V}_all_release_density_baseline.json")
    if r is None:
        return "skip release baseline table"
    m = r["models"]
    ind, team = m["ind_snapshot"]["test_nll"], m["team_snapshot"]["test_nll"]
    few = m["team_shot_features"]["test_nll"]["mdn_snapshot"]
    rows = [f"individual & {ind['learned_ghost']:.3f} & {ind['mdn_snapshot']:.3f} & --- & "
            f"{ind['rule_ghost']:.3f}",
            f"team & {team['learned_ghost']:.3f} & {team['mdn_snapshot']:.3f} & {few:.3f} & "
            f"{team['rule_ghost']:.3f}"]  # fmt: skip
    return _tex("release_density_baseline.tex", "lcccc",
                f"task ({r['n_test_shots']:,} test shots) & learned ghost (online, possession "
                "history) & single-moment network & few-features network & rule ghost", rows)  # fmt: skip


def tab_stability_curves() -> str:
    """Split-half r at n defensive possessions per player per half (main specification)."""
    rel = _load(f"reports/phase4/{V}_all/reliability_phases.json")
    if rel is None:
        return "skip stability curve table"
    m = rel["metrics"]
    ns = sorted(int(k) for k in m["sag_ft"]["stability_curve_adjusted"])
    rows = [" & ".join([str(n)] + [f"{m[k][c][str(n)]['r']:.2f}" for k in ("sag_ft", "dev_ft")
                                   for c in ("stability_curve_adjusted",
                                             "stability_curve_within_position")])
            for n in ns]  # fmt: skip
    return _tex("stability_curves.tex", "rcccc",
                "$n$ & sag, pooled & sag, within position & dev, pooled & dev, within position",
                rows)  # fmt: skip


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


def fig_sensitivity_contours(fig_dir: Path) -> str:
    """Review A5: Cinelli-Hazlett contours of the adjusted early-help estimate over the partial R2
    of a confounder with the treatment and with the outcome; benchmarks = k x the strongest
    observed covariates (with the treatment / with the outcome)."""
    s = _load("reports/analysis/help_timing_sensitivity.json")
    if s is None or "contour" not in s["comparisons"]["A_early_vs_rest"]["points"]:
        return "skip sensitivity contours (run analysis_help_timing_sensitivity.py)"
    fig, axs = plt.subplots(2, 2, figsize=(9, 7.2))
    for ax, (comp, y, title) in zip(axs.ravel(), (
            ("A_early_vs_rest", "points", "A, points"), ("A_early_vs_rest", "xfg_of_shot", "A, xFG"),
            ("B_early_vs_late_help", "points", "B, points"),
            ("B_early_vs_late_help", "xfg_of_shot", "B, xFG")), strict=True):  # fmt: skip
        r = s["comparisons"][comp][y]
        c = r["contour"]
        X, Y = np.meshgrid(c["r2_treatment"], c["r2_outcome"], indexing="ij")
        Z = np.array(c["adjusted_estimate"])
        lim = np.abs(Z).max()
        # symmetric levels around zero, so the colour of 0 is the neutral middle of the map
        lev = np.linspace(-lim, lim, 22)
        cs = ax.contourf(X, Y, Z, levels=lev, cmap="RdBu_r")
        ax.contour(X, Y, Z, levels=[0.0], colors="k", linewidths=1.4)
        names = {"h_v_to_rim": "handler velocity to rim", "h_dist_rim": "handler distance to rim",
                 "shot_clock": "shot clock", "margin": "score margin"}  # fmt: skip
        bench = dict.fromkeys(["h_v_to_rim", r["strongest_observed_covariate"],
                               r["strongest_observed_covariate_with_outcome"]])  # fmt: skip
        ytop = c["r2_outcome"][-1]
        for name, mk in zip(bench, ("^", "o", "s"), strict=False):
            pts = [b for b in r["bounds"][name] if "r2_confounder_treatment" in b]
            ax.scatter([b["r2_confounder_treatment"] for b in pts],
                       [min(b["r2_confounder_outcome"], ytop) for b in pts],
                       marker=mk, s=24, facecolor="k" if mk != "^" else "w", edgecolor="k",
                       zorder=3, label=f"{names.get(name, name)}, 1-3x")  # fmt: skip
        ax.set_title(f"{title}: OLS-adjusted estimate "
                     f"{_n(r['estimate_ols_adjusted'], 3, True)}".replace("$", ""), fontsize=9)  # fmt: skip
        ax.set_xlabel("partial $R^2$ of confounder with early help", fontsize=8)
        ax.set_ylabel("partial $R^2$ with outcome", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(cs, ax=ax, shrink=0.85).ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    out = fig_dir / "help_timing_sensitivity_contours.png"
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


def fig_learned_ghost_frames() -> str:
    """Figure C (D-029): two held-out possessions at three moments around a screen; needs the
    packed arrays and the final learned-ghost model (tracking-derived inputs; skipped without)."""
    import importlib.util

    run = Path("runs/phase3") / f"{V}_all_{LEARNED_FINAL}" / "model.pt"
    arr = D.DataConfig.load().processed_dir / "all" / "ghost_arrays" / "test" / "feats.npy"
    if not (run.exists() and arr.exists()):
        return "skip learned ghost frames (needs the packed arrays and the final model)"
    spec = importlib.util.spec_from_file_location(
        "frames", Path(__file__).parent / "make_figure_learned_ghost_frames.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.render(FIG).name


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
        ("tab_screen_confidence", tab_screen_confidence),
        ("tab_team_game_panel", tab_team_game_panel),
        ("tab_team_game_panel_inference", tab_team_game_panel_inference),
        ("tab_help_matched", tab_help_matched),
        ("tab_meta_metrics", tab_meta_metrics),
        ("tab_help_sensitivity", tab_help_sensitivity),
        ("tab_learned_reliability", tab_learned_reliability),
        ("tab_ghost_coverage", tab_ghost_coverage),
        ("tab_rule_reliability", tab_rule_reliability),
        ("tab_predictive_validity_player", tab_predictive_validity_player),
        ("tab_second_chance", tab_second_chance),
        ("tab_ghost_ablations", tab_ghost_ablations),
        ("tab_release_density_baseline", tab_release_density_baseline),
        ("tab_stability_curves", tab_stability_curves),
        ("fig_ghost_ablations", lambda: fig_ghost_ablations(FIG)),
        ("fig_release_baseline", lambda: fig_release_baseline(FIG)),
        ("fig_sensitivity_contours", lambda: fig_sensitivity_contours(FIG)),
        ("fig_learned_vs_rule_reliability", lambda: fig_learned_vs_rule_reliability(FIG)),
        ("fig_learned_ghost_frames", fig_learned_ghost_frames),
    ):
        print(name, "->", f())


if __name__ == "__main__":
    main()
