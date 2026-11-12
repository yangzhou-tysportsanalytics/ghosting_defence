"""Figures for the SSAC27 paper (impact-first version), in Cambria at print width.

Written to paper/ssac27/figures/ (the journal figures in paper/figures/ are not touched):
  help_sources.png        where help comes from: help rate per 1,000 guarding steps by the
                          location and side of the man left, and corner help by the shooter's
                          three-point volume (A1)
  help_timing_team.png    team early help vs points allowed (tracked games), and the within-team
                          early / late contrast in points per 100 possessions and wins (A2)
  learned_ghost_frames.png, learned_vs_rule_reliability.png
                          the journal figures re-rendered in Cambria

Usage:
    uv run python scripts/ssac_figures.py
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/ssac27/figures"
WINS_PER_POINT = 2.61  # wins per 82 games per point of net rating per 100 possessions (D-031)
STYLE = {"font.family": "Cambria", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300}  # fmt: skip
STRONG, WEAK = "#c0392b", "#2c6fbb"


def _load(p: str):
    f = ROOT / p
    return json.loads(f.read_text()) if f.exists() else None


def fig_help_sources() -> str:
    """Figure 4 (review 2026-11-02): per drive opportunity (analysis_corner_help_opportunity.py).
    (a) how often the defender of a strong-side man beyond the arc helps, by the man's location
    and three-point volume; (b) points per possession when he helped vs stayed, by the corner
    man's volume. Game-bootstrap intervals."""
    h = _load("reports/analysis/corner_help_opportunity.json")
    if h is None:
        return "skip help_sources (run analysis_corner_help_opportunity.py)"
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    t = np.arange(3)
    for loc, col, off, lab in (("corner_strict", STRONG, -0.18, "corner"),
                               ("above_break", WEAK, 0.18, "above the break")):  # fmt: skip
        e = [h["help_rate"][loc][f"threat{j}"] for j in range(3)]
        v = 100 * np.array([d["mean"] for d in e])
        lo = 100 * np.array([d["ci95"][0] for d in e])
        hi = 100 * np.array([d["ci95"][1] for d in e])
        a.bar(t + off, v, width=0.34, color=col, label=lab, yerr=[v - lo, hi - v], capsize=2,
              error_kw={"lw": 0.8})  # fmt: skip
        for xx, yy in zip(t + off, v, strict=True):
            a.text(xx, yy + 3.5, f"{yy:.0f}%", ha="center", fontsize=7)
    a.set_xticks(t, ["low", "middle", "high"])
    a.set_xlabel("his man's three-point attempt rate (tercile)")
    a.set_ylabel("% of drives on his side\nwith a help by his defender")
    a.set_ylim(0, 90)
    a.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    a.set_title("(a) Help rates differ little by the man's three-point rate", fontsize=9, loc="left",
                y=1.08)  # fmt: skip
    o = h["outcomes"]["corner_strict"]
    for j in range(3):
        d = o[f"points_threat{j}"]
        for flag, col, off in (("helped", STRONG, -0.18), ("not_helped", "0.6", 0.18)):
            b.bar(j + off, 100 * d[flag]["mean"], width=0.34, color=col,
                  label=("helped" if flag == "helped" else "stayed") if j == 0 else None)  # fmt: skip
        ci = d["difference"]["ci95"]
        top = 100 * max(d["helped"]["mean"], d["not_helped"]["mean"])
        b.text(j, top + 3, f"{100 * d['difference']['mean']:+.0f}\n"
               f"({100 * ci[0]:+.0f} to {100 * ci[1]:+.0f})", ha="center", fontsize=6.5)  # fmt: skip
    b.set_xticks(t, ["low", "middle", "high"])
    b.set_xlabel("corner man's three-point attempt rate (tercile)")
    b.set_ylabel("points per 100 possessions")
    b.set_ylim(0, 140)  # bars start at zero (final review R6)
    b.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    b.set_title("(b) Helped vs stayed: the scoring difference is uncertain", fontsize=9,
                loc="left", y=1.08)  # fmt: skip
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "help_sources.png", bbox_inches="tight")
    plt.close(fig)
    return "help_sources.png"


def fig_help_timing_team() -> str:
    """Figure 3 (review 2026-11-02): (a) within team, game by game: early help vs points allowed,
    both centred on the team's own season mean: means of the eighths of early help with 95 %
    intervals and the pooled slope (single games scatter widely; their SD is given).
    (b) the within-team coefficient by early-help threshold (two-way intervals)."""
    th = _load("reports/analysis/help_threshold.json")
    if th is None:
        return "skip help_timing_team (run analysis_help_threshold.py)"
    sys.path.insert(0, str(ROOT / "scripts"))
    import analysis_team_game_panel as TG
    from ghost import data as D

    cache = ROOT / "reports/analysis/team_game_early_help.csv"  # shipped: rates and points only
    cfg = D.DataConfig.load(game_set="all")
    if (cfg.processed_dir / "all" / "analysis" / "onball_screens.parquet").exists():
        P = TG.panel(cfg)
        P.select(["game_id", "defense_team_id", "early_help_rate", "pts_per_poss"]).write_csv(cache)
    P = pl.read_csv(cache, schema_overrides={"game_id": pl.String})
    teams = pl.read_csv(ROOT / "reports/ssac27/team_cards.csv").select(["team", "team_id"])
    P = P.join(teams, left_on="defense_team_id", right_on="team_id", how="left").with_columns(
        (100 * (pl.col("early_help_rate") - pl.col("early_help_rate").mean().over("team"))).alias("dx"),
        (100 * (pl.col("pts_per_poss") - pl.col("pts_per_poss").mean().over("team"))).alias("dy"),
    )  # fmt: skip
    x, y = P["dx"].to_numpy(), P["dy"].to_numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.1, 1]})
    edges = np.quantile(x, np.linspace(0, 1, 9))
    k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 7)
    bx = np.array([x[k == i].mean() for i in range(8)])
    by = np.array([y[k == i].mean() for i in range(8)])
    se = np.array([y[k == i].std(ddof=1) / np.sqrt((k == i).sum()) for i in range(8)])
    a.errorbar(bx, by, yerr=1.96 * se, fmt="-o", color=STRONG, ms=4, lw=1.4, capsize=2)
    b1 = np.polyfit(x, y, 1)[0]
    xx = np.array([x.min(), x.max()])
    a.plot(xx, b1 * xx, color="0.4", lw=0.8, ls="--")
    a.axhline(0, color="0.5", lw=0.5)
    a.axvline(0, color="0.5", lw=0.5)
    a.set_xlim(-16, 16)
    a.set_ylim(-8, 8)
    nk = np.bincount(k, minlength=8)
    nk_txt = f"{nk.min()}" if nk.min() == nk.max() else f"{nk.min()}–{nk.max()}"
    a.text(0.03, 0.04, f"each point: the {nk_txt} team-games in one eighth\n"
           f"single games vary by \u00b1{y.std():.0f} points (SD)", transform=a.transAxes,
           fontsize=6.5, color="0.35")  # fmt: skip
    a.set_xlabel("early help vs the team's own season average (pp)")
    a.set_ylabel("points allowed per 100 poss.\nvs the team's own season average")
    a.set_title("(a) On its early-help nights, a team\nallows slightly fewer points", fontsize=9,
                loc="left")  # fmt: skip
    # (b) the joint model at 1 s: early, late and early - late, two-way intervals (Table D1)
    j1 = th["threshold_curve"]["1.0"]
    rows = [("early help\n(within 1 s)", "early"), ("late help\n(1–3 s)", "late"),
            ("early − late", "difference")]  # fmt: skip
    for spec, col, lab, dy in (("team_fe", STRONG, "within team", -0.12),
                               ("team_and_opponent_fe", "0.35", "within team, opponent fixed", 0.12)):  # fmt: skip
        for r, (_, key) in enumerate(rows):
            e = j1[spec]["joint"][key]
            lo, hi = e["ci95_twoway"]
            b.errorbar(e["coef"], r + dy, xerr=[[e["coef"] - lo], [hi - e["coef"]]], fmt="o",
                       color=col, ms=4, lw=1.2, capsize=2, label=lab if r == 0 else None)  # fmt: skip
            b.text(max(hi, 0.0) + 0.12, r + dy, f"{e['coef']:+.2f}".replace("-", "−"), va="center", fontsize=6.5, color=col)
    b.axvline(0, color="0.4", lw=0.7)
    b.set_yticks(range(3), [t for t, _ in rows])
    b.set_ylim(2.5, -1.4)
    b.set_xlim(-4.2, 2.8)
    b.set_xlabel("points allowed per 100 poss.\nper +10 pp of the rate (per on-ball screen)")
    b.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2, handletextpad=0.2,
             columnspacing=0.8)  # fmt: skip
    b.set_title("(b) Early and late help in the joint model", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "help_timing_team.png", bbox_inches="tight")
    plt.close(fig)
    return "help_timing_team.png"


def fig_help_threshold() -> str:
    """Appendix Figure D2 (writing request 2026-11-02c): the single-rate within-team association
    as the threshold for "early" runs from 0.4 to 3 s, alone and with opponents fixed, two-way
    intervals (analysis_help_threshold.py)."""
    th = _load("reports/analysis/help_threshold.json")
    if th is None:
        return "skip help_threshold (run analysis_help_threshold.py)"
    fig, b = plt.subplots(figsize=(4.6, 3.0))
    cur = th["threshold_curve"]
    taus = sorted(float(t) for t in cur)
    for spec, col, lab, dx in (("team_fe", STRONG, "within team", -0.02),
                               ("team_and_opponent_fe", "0.35", "within team, opponent fixed", 0.02)):  # fmt: skip
        e = [cur[f"{t:.1f}"][spec]["single"]["early"] for t in taus]
        est = np.array([v["coef"] for v in e])
        lo = np.array([v["ci95_twoway"][0] for v in e])
        hi = np.array([v["ci95_twoway"][1] for v in e])
        b.errorbar(np.array(taus) + dx, est, yerr=[est - lo, hi - est], fmt="o-", color=col,
                   ms=3.5, lw=1, capsize=1.5, label=lab)  # fmt: skip
    b.axhline(0, color="0.4", lw=0.7)
    b.axvline(1.0, color="0.6", lw=0.7, ls=":")
    b.text(0.97, 0.04, "threshold\nused in\nthe text", fontsize=6.5, color="0.4", ha="right",
           va="bottom", transform=b.get_xaxis_transform())  # fmt: skip
    b.set_xlabel("help counted as early if within ... s of the screen")
    b.set_ylabel("points allowed per 100 poss.\nper +10 pp of early help")
    b.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "help_threshold.png", bbox_inches="tight")
    plt.close(fig)
    return "help_threshold.png"

def fig_player_measures() -> str:
    """Each player measure as a point: reliability (split-half, within team) against independence
    from role (position, height, threat of the man guarded); upper right = stable and new."""
    cf = "cf_gpu_d128_lr1e-3"
    rel = _load(f"reports/phase4/v1.6_all/reliability_learned_{cf}_phases.json")
    mm = _load(f"reports/analysis/meta_metrics_learned_{cf}.json")
    if rel is None or mm is None:
        return "skip player_measures"
    items = [("sag_ft", "sag (rule ghost)", "0.3"), ("dev_ft", "deviation (rule ghost)", "0.3"),
             ("nll_ind", "deviation from the\nindividual learned ghost", STRONG),
             ("nll_team", "deviation from the\nteam learned ghost", STRONG),
             ("d_mean", "distance to the\nlearned ghost", STRONG)]  # fmt: skip
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhspan(0.8, 1.0, xmin=(0.6 - 0.2) / 0.85, xmax=1.0, color="#eaf4ea", zorder=0)
    ax.text(1.04, 0.995, "stable and new information", ha="right", va="top", fontsize=8,
            color="#2e7d32")  # fmt: skip
    offs = {"sag_ft": (8, -4), "dev_ft": (-8, -6), "nll_ind": (-8, -8), "nll_team": (8, 6),
            "d_mean": (-8, -18)}  # fmt: skip
    for k, lab, col in items:
        x = mm["metrics"][f"{k}_within_team"]["independence"]["position_height_guarded_threat"][
            "independence"
        ]
        y = rel["metrics"][k]["split_half_r_player_within_team"]
        ax.scatter(x, y, s=40, color=col, zorder=3, edgecolor="k", linewidth=0.5)
        dx, dy = offs[k]
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    ha="left" if dx > 0 else "right", va="center")  # fmt: skip
    ax.set_xlim(0.2, 1.05)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("independence from role (position, height, threat of the man guarded)")
    ax.set_ylabel("reliability (split-half, within team)")
    ax.set_title("What the player measures add", fontsize=10, loc="left")
    fig.savefig(OUT / "player_measures.png", bbox_inches="tight")
    plt.close(fig)
    return "player_measures.png"


def fig_measure_guide() -> str:
    """Figure 5 (review 2026-11-02): a guide to two player measures, three questions each:
    repeatable (split-half within team), independent of role (share not explained by position,
    height and the threat of the man guarded), and predictive (Spearman correlation of the trait
    on one half of the games with the shooter-aware xFG allowed on the other half, both
    directions, 95 % bootstrap intervals; phase5_predictive_validity.py)."""
    cf = "cf_gpu_d128_lr1e-3"
    rel = _load(f"reports/phase4/v1.6_all/reliability_learned_{cf}_phases.json")
    mm = _load(f"reports/analysis/meta_metrics_learned_{cf}.json")
    pv = {"sag_ft": _load("reports/phase5/v1.6_all/predictive_validity_phases.json"),
          "nll_ind": _load(f"reports/phase5/v1.6_all/predictive_validity_phases_learned_{cf}.json")}  # fmt: skip
    if rel is None or mm is None or None in pv.values():
        return "skip measure_guide"
    # compact (2026-11-02c layout pass): drawn at the printed width, so text is not scaled down;
    # the use of each measure is stated in the caption
    rows = [("sag_ft", "Sag", ""), ("nll_ind", "Departure\nfrom the ghost", "")]
    fig, axs = plt.subplots(len(rows), 3, figsize=(5.6, 1.25),
                            gridspec_kw={"wspace": 0.3, "hspace": 1.0})  # fmt: skip
    heads = ["Repeatable?\n(odd vs even games)", "Share unexplained by measured\nrole (position, height, man)",
             "Predicts xFG allowed\nin other games? (rank corr.)"]  # fmt: skip
    for i, (k, lab, _) in enumerate(rows):
        r = rel["metrics"][k]["split_half_r_player_within_team"]
        ind = mm["metrics"][f"{k}_within_team"]["independence"]["position_height_guarded_threat"]["independence"]
        for j, v in enumerate((r, ind)):
            ax = axs[i, j]
            ax.barh([0], [1], color="0.92", height=0.5)
            ax.barh([0], [v], color=STRONG if v >= 0.7 else "0.55", height=0.5)
            ax.text(min(v, 0.97), 0, f"{v:.2f}", ha="right", va="center", fontsize=8, color="white",
                    fontweight="bold")  # fmt: skip
        ax = axs[i, 2]
        pl_ = pv[k]["player_level"]
        ests = [pl_[d]["xfg_m3_cf"] for d in ("trait_parity0_outcome_parity1", "trait_parity1_outcome_parity0")]
        for yy, e, tag in zip((0.15, -0.15), ests, ("odd" + "→even", "even" + "→odd"), strict=True):
            ax.errorbar(e["rho"], yy, xerr=[[e["rho"] - e["ci95"][0]], [e["ci95"][1] - e["rho"]]],
                        fmt="o", color="0.25", ms=4, capsize=2, lw=1)  # fmt: skip
            if i == 0:  # which half of the games predicts which (first row only)
                ax.text(e["ci95"][1] + 0.02, yy, tag, va="center", fontsize=6, color="0.35")
        ax.axvline(0, color="0.5", lw=0.6)
        ax.set_xlim(-0.15, 0.6)
        for j in range(3):
            a = axs[i, j]
            a.set_yticks([])
            a.set_ylim(-0.5, 0.5)
            for sp in ("left", "top", "right"):
                a.spines[sp].set_visible(False)
            a.tick_params(axis="x", labelsize=7, length=2)
            if j < 2:
                a.set_xlim(0, 1)
            if i == 0:
                a.set_title(heads[j], fontsize=7.5, loc="left")
        axs[i, 0].text(-0.06, 0, lab, transform=axs[i, 0].get_yaxis_transform(), ha="right",
                       va="center", fontsize=8, fontweight="bold")  # fmt: skip
    fig.savefig(OUT / "measure_guide.png", bbox_inches="tight")
    plt.close(fig)
    return "measure_guide.png"


def fig_pipeline() -> str:
    """Figure 2: a navigation strip, five labelled steps in one row (writing request 2026-11-02c:
    about 1-1.2 in tall at 5-6 in wide, little text)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    steps = [("Tracking data", "631 games,\n10 players + ball", "0.35"),
             ("Matchups", "who guards whom;\nscreens, help, switches", "0.35"),
             ("League ghost", "where a league-average\ndefender would stand", "0.35"),
             ("Measures", "sag, departure,\nhelp timing", "0.35"),
             ("Team uses", "player and team cards,\nreview queue", STRONG)]  # fmt: skip
    fig, ax = plt.subplots(figsize=(5.8, 1.05))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    w, gap = 1.72, 0.33
    for i, (head, sub, col) in enumerate(steps):
        x0 = i * (w + gap) + 0.02
        ax.add_patch(FancyBboxPatch((x0, 0.12), w, 0.76, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=col, ec=col, lw=1.2))  # fmt: skip
        ax.text(x0 + w / 2, 0.66, head, ha="center", va="center", color="white", fontsize=7,
                fontweight="bold")  # fmt: skip
        ax.text(x0 + w / 2, 0.34, sub, ha="center", va="center", color="white", fontsize=5.6,
                linespacing=1.15)  # fmt: skip
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w + 0.03, 0.5), (x0 + w + gap - 0.03, 0.5),
                                         arrowstyle="-|>", mutation_scale=9, color="0.3", lw=1))  # fmt: skip
    fig.savefig(OUT / "pipeline.png", bbox_inches="tight")
    plt.close(fig)
    return "pipeline.png"


def rerender_journal_figures() -> list[str]:
    """The learned-ghost frames re-rendered in the SSAC style (same code, Cambria, 300 dpi)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_figure_learned_ghost_frames as lgf
    from ghost import data as D

    arrays = D.DataConfig.load(game_set="all").processed_dir / "all" / "ghost_arrays"
    if not arrays.exists():
        return ["skip learned_ghost_frames.png (needs the tracking arrays, not distributed)"]
    plt.rcParams.update(STYLE)
    lgf.render(OUT)
    return ["learned_ghost_frames.png"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    for f in (fig_help_sources, fig_help_timing_team, fig_help_threshold, fig_player_measures,
              fig_measure_guide, fig_pipeline):  # fmt: skip
        plt.rcParams.update(STYLE)
        print(f.__name__, "->", f())
    plt.rcParams.update(STYLE)
    print("re-rendered:", rerender_journal_figures())


if __name__ == "__main__":
    main()
