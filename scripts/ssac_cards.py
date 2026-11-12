"""Worked examples for Section 2 of the SSAC27 paper ("What a team can do with this").

1. Team defense cards: per team (tracked games) points allowed per 100 possessions, help within
   3 s, early help (<= 1 s), switch rate, median help delay, strong- and weak-side corner help
   per 1,000 guarding steps, team sag effect; with the league percentile of each.
   reports/ssac27/team_cards.csv; figure paper/ssac27/figures/team_card_<TEAM>.png
2. Player cards: per player with >= 300 defensive possessions: within-team effects (posterior
   mean and 95 % interval) of sag, deviation and the individual learned-ghost deviation,
   strong-side corner help per 1,000 guarding steps, percentiles within listed position.
   reports/ssac27/player_cards.csv; candidates by a stated rule in
   reports/ssac27/player_card_candidates.csv; figure paper/ssac27/figures/player_card_<id>.png
3. Clip list: the breakdowns of one game against the learned individual ghost (D-023), with
   period, game clock, defender, duration and whether it started a cascade - what a video
   coordinator would cut. reports/ssac27/clip_list_<game>.csv
Counts and effects only; no coordinates.

Usage:
    uv run python scripts/ssac_cards.py [--team SAS] [--players 1626169,2571] [--game 0021500569]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ghost import data as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase4_reliability import listed_positions  # noqa: E402
from ssac_figures import STRONG, STYLE  # noqa: E402

OUT_R, OUT_F = ROOT / "reports/ssac27", ROOT / "paper/ssac27/figures"
R4 = ROOT / "reports/phase4/v1.6_all"
CF = "cf_gpu_d128_lr1e-3"
MIN_POSS_CARD, MIN_POSS_CANDIDATE = 300, 1000
MIN_CORNER_STEPS = 2000  # 400 s guarding a strong-side corner man
# player and team effect intervals: hierarchical model with game-clustered cell variances
# (review 2026-11-02: the pooled intervals, about +/- 0.03 ft, were too narrow)
HIER_SUFFIX = "_phases_gclust"
TEAM_COLORS = ("#2c6fbb", "#e67e22")


def _ordinal(pct: float) -> str:
    """Percentile as words: 'lowest', 'highest', or '96th percentile'."""
    k = int(round(pct))
    if k <= 0:
        return "lowest"
    if k >= 100:
        return "highest"
    suf = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suf} percentile"


def _pct(v: np.ndarray) -> np.ndarray:
    """Percentile (0-100) of each value within the array."""
    order = v.argsort().argsort()
    return 100 * order / max(len(v) - 1, 1)


def team_cards(base: Path) -> pl.DataFrame:
    scr = pl.read_csv(ROOT / "reports/analysis/team_screen_defense.csv")
    pts = pl.read_csv(ROOT / "reports/analysis/team_points_allowed.csv")
    S = pl.read_parquet(base / "analysis" / "onball_screens.parquet")
    early = (
        S.group_by("defense_team_id")
        .agg((pl.col("help_delay_s").fill_null(99) <= 1.0).mean().alias("early_help_rate"))
        .rename({"defense_team_id": "team_id"})
    )
    side = pl.read_csv(ROOT / "reports/analysis/help_side_by_team.csv")
    corner = side.filter(pl.col("loc") == "corner3").pivot(
        on="side", index="team", values="rate_per_1000"
    ).select(pl.col("team").alias("team_id"), pl.col("strong").alias("corner_help_strong_per_1000"),
             pl.col("weak").alias("corner_help_weak_per_1000"))  # fmt: skip
    hier = json.loads((R4 / "hier_phases.json").read_text())
    sag_team = pl.DataFrame(hier["metrics"]["sag_ft"]["team_effects"]).select(
        pl.col("team"), pl.col("mean").alias("team_sag_effect_ft"))  # fmt: skip
    # review 2026-11-02: intervals for early help (game bootstrap) and team sag (game-clustered
    # hierarchical model), corner help per drive opportunity (analysis_corner_help_opportunity)
    rng = np.random.default_rng(12)
    eb = []
    for tid, d in S.sort(["game_id", "t_contact_ms"]).group_by("defense_team_id", maintain_order=True):
        g = d.group_by("game_id", maintain_order=True).agg(
            (pl.col("help_delay_s").fill_null(99) <= 1.0).sum().alias("k"), pl.len().alias("n"))
        k, n = g["k"].to_numpy(), g["n"].to_numpy()
        idx = rng.integers(0, len(k), size=(1000, len(k)))
        m = k[idx].sum(1) / n[idx].sum(1)
        eb.append({"team_id": tid[0], "early_lo": float(np.quantile(m, 0.025)),
                   "early_hi": float(np.quantile(m, 0.975))})  # fmt: skip
    gcl = json.loads((R4 / f"hier{HIER_SUFFIX}.json").read_text())
    sag_ci = pl.DataFrame(gcl["metrics"]["sag_ft"]["team_effects"]).select(
        pl.col("team"), pl.col("lo95").alias("sag_lo"), pl.col("hi95").alias("sag_hi"),
        pl.col("mean").alias("team_sag_effect_ft_gclust"))  # fmt: skip
    copp = pl.read_csv(ROOT / "reports/analysis/corner_help_opportunity_by_team.csv").select(
        pl.col("team").alias("team_id"), pl.col("corner_help_per_opp"),
        pl.col("n").alias("corner_opportunities"))  # fmt: skip
    T = (scr.join(pts, on="team_id").join(early, on="team_id").join(corner, on="team_id")
         .join(sag_team, on="team", how="left").join(pl.DataFrame(eb), on="team_id", how="left")
         .join(sag_ci, on="team", how="left").join(copp, on="team_id", how="left")
         .with_columns((100 * pl.col("pts_allowed_per_poss")).alias("pts_allowed_per_100"),
                       (100 * pl.col("pts_lo")).alias("pts_allowed_lo"),
                       (100 * pl.col("pts_hi")).alias("pts_allowed_hi"))
         .sort("team"))  # fmt: skip
    cols = ["pts_allowed_per_100", "help_rate_3s", "early_help_rate", "switch_rate",
            "help_delay_median_s", "corner_help_strong_per_1000", "corner_help_weak_per_1000",
            "team_sag_effect_ft"]  # fmt: skip
    T = T.with_columns(*[pl.Series(f"{c}_pct", _pct(T[c].to_numpy())) for c in cols])
    return T.select(["team", "team_id", "n_games", "n_on_ball_screens", *cols,
                     "pts_allowed_lo", "pts_allowed_hi", "help_lo", "help_hi", "switch_lo",
                     "switch_hi", *[f"{c}_pct" for c in cols], "early_lo", "early_hi",
                     "team_sag_effect_ft_gclust", "sag_lo", "sag_hi", "corner_help_per_opp",
                     "corner_opportunities"])  # fmt: skip


MAIN_TEAM_ROWS = ("pts_allowed_per_100", "early_help_rate", "help_rate_3s",
                  "team_sag_effect_ft_gclust")  # the slim main-text card (writing request 11-02c)


def team_pair_figure(T: pl.DataFrame, teams: list[str], rows: tuple[str, ...] | None = None,
                     suffix: str = "") -> str:  # fmt: skip
    """Review 2026-11-02: the league (30 teams, grey; median tick) with two teams and their 95 %
    intervals: the league's best defense by points allowed and the team with the highest early
    help. Selection rule stated in the caption. ``rows``: value columns to keep (all six by
    default); ``suffix`` is added to the file name."""
    plt.rcParams.update(STYLE)
    p = lambda r: np.sqrt(r["corner_help_per_opp"] * (1 - r["corner_help_per_opp"])  # noqa: E731
                          / r["corner_opportunities"])  # fmt: skip
    spec = [  # label, value column, (lo, hi) or callable, scale, format, sign note
        ("points allowed per 100 possessions", "pts_allowed_per_100", ("pts_allowed_lo", "pts_allowed_hi"), 1, "{:.1f}", "lower = better"),
        ("early help: within 1 s of the screen (% of screens)", "early_help_rate", ("early_lo", "early_hi"), 100, "{:.0f}%", ""),
        ("any help within 3 s (% of screens)", "help_rate_3s", ("help_lo", "help_hi"), 100, "{:.0f}%", ""),
        ("switches (% of screens)", "switch_rate", ("switch_lo", "switch_hi"), 100, "{:.0f}%", ""),
        ("strong-side corner defender helps (% of drives)", "corner_help_per_opp", p, 100, "{:.0f}%", ""),
        ("team sag vs league norm (ft)", "team_sag_effect_ft_gclust", ("sag_lo", "sag_hi"), 1, "{:+.2f}", "+ = off the man"),
    ]  # fmt: skip
    if rows is not None:
        spec = [r for r in spec if r[1] in rows]
    fig, axs = plt.subplots(len(spec), 1, figsize=(6.6, 0.62 * len(spec) + 0.7))
    for ax, (lab, c, ci, k, fmt, note) in zip(axs, spec, strict=True):
        dist = k * T[c].to_numpy()
        ax.scatter(dist, np.zeros_like(dist), s=12, color="0.8", zorder=2)
        ax.plot([np.median(dist)] * 2, [-0.5, 0.5], color="0.45", lw=0.8, zorder=1)
        txt = []
        for (t, col), dy in zip(zip(teams, TEAM_COLORS, strict=False), (0.22, -0.22), strict=False):
            r = T.filter(pl.col("team") == t).row(0, named=True)
            v = k * r[c]
            if callable(ci):
                lo, hi = v - 1.96 * k * ci(r), v + 1.96 * k * ci(r)
            else:
                lo, hi = k * r[ci[0]], k * r[ci[1]]
            ax.errorbar([v], [dy], xerr=[[v - lo], [hi - v]], fmt="o", color=col, ms=5, mec="k",
                        mew=0.5, capsize=2, lw=1.2, zorder=3)  # fmt: skip
            txt.append((fmt.format(v), col))
        ends = []
        for t in teams:
            r = T.filter(pl.col("team") == t).row(0, named=True)
            ends += ([k * (r[c] - 1.96 * ci(r)), k * (r[c] + 1.96 * ci(r))] if callable(ci)
                     else [k * r[ci[0]], k * r[ci[1]]])  # fmt: skip
        lo_all, hi_all = min(dist.min(), *ends), max(dist.max(), *ends)
        pad = 0.06 * (hi_all - lo_all)
        ax.set_xlim(lo_all - pad, hi_all + pad)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        for sp in ("left", "top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="x", labelsize=7, length=2)
        ax.text(0.0, 1.05, lab + (f"  ({note})" if note else ""), transform=ax.transAxes,
                fontsize=8, va="bottom")  # fmt: skip
        for i, (s, col) in enumerate(txt):
            ax.text(1.02, 0.7 - 0.45 * i, s, transform=ax.transAxes, fontsize=8, color=col,
                    va="center", fontweight="bold")  # fmt: skip
    hand = [plt.Line2D([], [], marker="o", ls="", color=c, mec="k", mew=0.5, label=t)
            for t, c in zip(teams, TEAM_COLORS, strict=False)]  # fmt: skip
    hand += [plt.Line2D([], [], marker="o", ls="", color="0.8", label="other teams"),
             plt.Line2D([], [], color="0.45", lw=0.8, label="league median")]  # fmt: skip
    fig.legend(handles=hand, loc="upper left", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.0, 1.0))  # fmt: skip
    fig.subplots_adjust(left=0.02, right=0.88, hspace=1.35, top=1 - 0.55 / (0.62 * len(spec) + 0.7))
    out = OUT_F / f"team_card_{'_'.join(teams)}{suffix}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out.name


def strip_card(rows: list[tuple[str, np.ndarray, float, str]], title: str, out: Path) -> None:
    """One row per metric: the league distribution (grey) and the unit (red), with its value."""
    plt.rcParams.update(STYLE)
    fig, axs = plt.subplots(len(rows), 1, figsize=(5.4, 0.48 * len(rows) + 0.6))
    for ax, (lab, dist, val, txt) in zip(axs, rows, strict=True):
        lo, hi = np.nanmin(dist), np.nanmax(dist)
        pad = 0.05 * (hi - lo or 1)
        ax.scatter(dist, np.zeros_like(dist), s=10, color="0.75", zorder=2)
        ax.scatter([val], [0], s=40, color=STRONG, zorder=3, edgecolor="k", linewidth=0.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_yticks([])
        for sp in ("left", "top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="x", labelsize=7, length=2)
        ax.text(-0.02, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=8)
        ax.text(1.02, 0.5, txt, transform=ax.transAxes, ha="left", va="center", fontsize=8,
                color=STRONG)  # fmt: skip
    fig.suptitle(title, fontsize=10, x=0.02, ha="left")
    fig.subplots_adjust(left=0.42, right=0.82, hspace=1.1, top=1 - 0.45 / (0.48 * len(rows) + 0.6))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def team_card_figure(T: pl.DataFrame, team: str) -> str:
    r = T.filter(pl.col("team") == team).row(0, named=True)
    spec = [("points allowed / 100 poss.", "pts_allowed_per_100", "{:.1f}", 1),
            ("help within 3 s (% of screens)", "help_rate_3s", "{:.0f}%", 100),
            ("early help, ≤ 1 s (%)", "early_help_rate", "{:.0f}%", 100),
            ("switch rate (%)", "switch_rate", "{:.0f}%", 100),
            ("median help delay (s)", "help_delay_median_s", "{:.2f} s", 1),
            ("strong-side corner help / 1,000", "corner_help_strong_per_1000", "{:.1f}", 1),
            ("team sag vs league (ft)", "team_sag_effect_ft", "{:+.2f} ft", 1)]  # fmt: skip
    rows = []
    n = T.height
    for lab, c, fmt, k in spec:
        pct = r[c + "_pct"]
        where = (f"lowest of {n}" if pct == 0 else f"highest of {n}" if pct == 100
                 else f"{pct:.0f}th pct.")  # fmt: skip
        rows.append((lab, k * T[c].to_numpy(), k * r[c], f"{fmt.format(k * r[c])}  ({where})"))
    out = OUT_F / f"team_card_{team}.png"
    strip_card(rows, f"Team defense card: {team} ({r['n_games']} tracked games)", out)
    return out.name


def player_cards(cfg) -> pl.DataFrame:
    def eff(f: str, name: str) -> pl.DataFrame:
        d = pl.read_csv(R4 / f)
        return d.select(pl.col("player_id"), pl.col("team"), pl.col("n_poss"),
                        pl.col("firstname"), pl.col("lastname"),
                        pl.col("effect_mean").alias(name), pl.col("effect_lo95").alias(f"{name}_lo"),
                        pl.col("effect_hi95").alias(f"{name}_hi"))  # fmt: skip

    sag = eff(f"hier_players_sag_ft{HIER_SUFFIX}.csv", "sag_effect_ft")
    dev = eff(f"hier_players_dev_ft{HIER_SUFFIX}.csv", "dev_effect_ft").drop(
        ["n_poss", "firstname", "lastname"]
    )
    nll = eff(f"hier_players_nll_ind_learned_{CF}{HIER_SUFFIX}.csv", "learned_dev_effect").drop(
        ["n_poss", "firstname", "lastname"])  # fmt: skip
    side = pl.read_csv(ROOT / "reports/analysis/help_side_by_player.csv")
    corner = (side.filter((pl.col("loc") == "corner3") & (pl.col("side") == "strong"))
              .select(pl.col("def_id").alias("player_id"), pl.col("steps").alias("strong_corner_steps"),
                      pl.col("rate_per_1000").alias("strong_corner_help_per_1000")))  # fmt: skip
    # help onsets per 100 s of guarding (5 Hz steps), Poisson 95 % interval
    sec = pl.col("steps") * 0.2
    allh = side.group_by("def_id").agg(pl.col("helps").sum(), pl.col("steps").sum()).select(
        pl.col("def_id").alias("player_id"), (1000 * pl.col("helps") / pl.col("steps")).alias("help_per_1000"),
        (100 * pl.col("helps") / sec).alias("help_per_100s"),
        (100 * (pl.col("helps") - 1.96 * pl.col("helps").sqrt()) / sec).alias("help_per_100s_lo"),
        (100 * (pl.col("helps") + 1.96 * pl.col("helps").sqrt()) / sec).alias("help_per_100s_hi"))  # fmt: skip
    P = (sag.join(dev, on=["player_id", "team"]).join(nll, on=["player_id", "team"], how="left")
         .join(corner, on="player_id", how="left").join(allh, on="player_id", how="left")
         .filter(pl.col("n_poss") >= MIN_POSS_CARD))  # fmt: skip
    pos = listed_positions(cfg)
    P = P.with_columns(pl.col("player_id").map_elements(lambda p: pos.get(int(p), "?"),
                                                        return_dtype=pl.Utf8).alias("position"))  # fmt: skip
    pcts = []
    for c in (
        "sag_effect_ft",
        "dev_effect_ft",
        "learned_dev_effect",
        "strong_corner_help_per_1000",
        "help_per_100s",
    ):
        P = P.with_columns(pl.col(c).rank("average").over("position").alias("_r"),
                           pl.len().over("position").alias("_n"))  # fmt: skip
        P = P.with_columns((100 * (pl.col("_r") - 1) / (pl.col("_n") - 1)).alias(f"{c}_pos_pct"))
        pcts.append(f"{c}_pos_pct")
    return P.drop(["_r", "_n"]).sort(["team", "lastname"])


def candidates(P: pl.DataFrame) -> pl.DataFrame:
    """Rule: >= 1,000 defensive possessions with the player's team; effect intervals excluding
    zero; top 3 by each criterion. Illustrates the range of the measures, not a ranking."""
    Q = P.filter(pl.col("n_poss") >= MIN_POSS_CANDIDATE)
    out = []
    for crit, col, desc in (("sags the most vs league norm", "sag_effect_ft", True),
                            ("sags the least vs league norm", "sag_effect_ft", False),
                            ("departs most from the learned ghost", "learned_dev_effect", True),
                            ("departs least from the learned ghost", "learned_dev_effect", False)):  # fmt: skip
        excl = (pl.col(f"{col}_lo") > 0) | (pl.col(f"{col}_hi") < 0)
        top = Q.filter(excl).sort(col, descending=desc).head(3)
        out.append(top.with_columns(pl.lit(crit).alias("criterion")))
    return pl.concat(out).select(["criterion", "player_id", "firstname", "lastname", "team",
                                  "position", "n_poss", "sag_effect_ft", "learned_dev_effect",
                                  "strong_corner_help_per_1000"])  # fmt: skip


def player_card_figure(P: pl.DataFrame, pid: int) -> str:
    r = P.filter(pl.col("player_id") == pid).row(0, named=True)
    same = P.filter(pl.col("position") == r["position"])
    spec = [("sag vs league norm (ft)", "sag_effect_ft", "{:+.2f} ft"),
            ("deviation vs league (ft)", "dev_effect_ft", "{:+.2f} ft"),
            ("departure from learned ghost", "learned_dev_effect", "{:+.2f}"),
            ("strong-side corner help / 1,000", "strong_corner_help_per_1000", "{:.1f}"),
            ("all help / 1,000 guarding steps", "help_per_1000", "{:.1f}")]  # fmt: skip
    rows = []
    for lab, c, fmt in spec:
        v = r[c]
        if v is None:
            continue
        # corner rates need exposure: skip players who rarely guard the strong-side corner
        if (
            c == "strong_corner_help_per_1000"
            and (r["strong_corner_steps"] or 0) < MIN_CORNER_STEPS
        ):
            continue
        ci = f" [{r[c + '_lo']:+.2f}, {r[c + '_hi']:+.2f}]" if f"{c}_lo" in r else ""
        rows.append((lab, same[c].drop_nulls().to_numpy(), v, fmt.format(v) + ci))
    name = f"{r['firstname']} {r['lastname']}"
    out = OUT_F / f"player_card_{pid}.png"
    strip_card(rows, f"Player card: {name} ({r['team']}, {r['position']}; {r['n_poss']:,} "
               f"defensive possessions; grey: other {r['position']}s)", out)  # fmt: skip
    return out.name


POSITION_WORDS = {"G": "guard", "F": "forward", "C": "center", "G-F": "guard-forward",
                  "F-G": "forward-guard", "F-C": "forward-center", "C-F": "center-forward"}  # fmt: skip
CLIP_MIN_CLOCK_S = 10.0  # end-of-period moments (heaves, intentional plays) left out
CLIP_MAX_DURATION_S = 6.0  # longer stretches are mostly matchup or ghost failures
CLIP_TOP = 10
CLIP_NO_MATCHUP_TOP = 3


def player_group_figure(P: pl.DataFrame, pids: list[int], name: str) -> str:
    """Review 2026-11-02: several players of one team on one card. Grey: all players with >= 300
    defensive possessions; each player with the 95 % interval (game-clustered hierarchical
    model; Poisson for the help rate) and his percentile within his listed position."""
    plt.rcParams.update(STYLE)
    cols = ("#c0392b", "#2c6fbb", "#e67e22", "#27ae60")
    spec = [("sag vs league norm (ft)", "+ = plays farther off his man, toward the rim",
             "sag_effect_ft", "{:+.1f} ft"),
            ("departure from the learned ghost", "+ = stands where the league rarely stands",
             "learned_dev_effect", "{:+.2f}"),
            ("help moves per 100 s of guarding", "moving off his man toward the ball",
             "help_per_100s", "{:.1f}")]  # fmt: skip
    rows = [P.filter(pl.col("player_id") == p).row(0, named=True) for p in pids]
    row_in, top_in = 0.66, 0.6  # inches per measure and for the legend (layout pass 2026-11-02c)
    fig, axs = plt.subplots(len(spec), 1, figsize=(5.9, row_in * len(spec) + top_in))
    n = len(rows)
    for ax, (lab, note, c, fmt) in zip(axs, spec, strict=True):
        dist = P[c].drop_nulls().to_numpy()
        ax.scatter(dist, np.zeros_like(dist), s=8, color="0.82", zorder=2, lw=0)
        ax.plot([np.median(dist)] * 2, [-0.6, 0.6], color="0.45", lw=0.8, zorder=1)
        for i, (r, col) in enumerate(zip(rows, cols, strict=False)):
            dy = 0.32 - 0.64 * i / max(n - 1, 1)
            v, lo, hi = r[c], r[f"{c}_lo"], r[f"{c}_hi"]
            ax.errorbar([v], [dy], xerr=[[v - lo], [hi - v]], fmt="o", color=col, ms=5, mec="k",
                        mew=0.5, capsize=2, lw=1.2, zorder=3)  # fmt: skip
            ax.text(1.02, 0.85 - 0.7 * i / max(n - 1, 1), f"{fmt.format(v)}  "
                    f"({_ordinal(r[f'{c}_pos_pct'])} of {r['position']}s)", transform=ax.transAxes,
                    fontsize=7, color=col, va="center")  # fmt: skip
        ax.set_ylim(-0.7, 0.7)
        ax.set_yticks([])
        for sp in ("left", "top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="x", labelsize=7, length=2)
        ax.text(0.0, 1.04, f"{lab}   ({note})", transform=ax.transAxes, fontsize=8.5,
                va="bottom")  # fmt: skip
    hand = [plt.Line2D([], [], marker="o", ls="", color=col, mec="k", mew=0.5,
                       label=f"{r['firstname'][0]}. {r['lastname']} ({r['position']}, "
                             f"{r['n_poss']:,} poss.)")
            for r, col in zip(rows, cols, strict=False)]  # fmt: skip
    hand += [plt.Line2D([], [], marker="o", ls="", color="0.82", label=f"all {P['player_id'].n_unique()} players"),
             plt.Line2D([], [], color="0.45", lw=0.8, label="median")]  # fmt: skip
    fig.legend(handles=hand, loc="upper left", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.0, 1.02))  # fmt: skip
    fig.subplots_adjust(left=0.02, right=0.72, hspace=1.35, top=1 - 0.5 / (row_in * len(spec) + top_in))
    out = OUT_F / f"player_card_{name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out.name


def clip_list(base: Path, cfg, game: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Breakdowns of one game against the learned individual ghost (D-023), as clips for a video
    room (review h, 2026-11-02). One clip per cascade (the initiator; 1 s window) with the number
    of defenders involved; left out: starts in the last CLIP_MIN_CLOCK_S of a period, and runs
    longer than CLIP_MAX_DURATION_S. Ranked by the xFG of the possession's shot, then points;
    the top CLIP_TOP kept. Defenders and the man they guarded at the start are given by name,
    team and listed position (the paper shows team and position only). Final review 2026-11-02
    (R1, R2, R4): clips whose defender had no man-to-man state from the start of the possession
    to the flag (``has_matchup`` false; often just after the ball crosses midcourt) are listed
    separately. Returns (all breakdowns with the filter reasons, the top CLIP_TOP clips with a
    matchup, the top CLIP_NO_MATCHUP_TOP clips without one), each in game order."""
    from ghost.tensors import load_game_set

    B = pl.read_parquet(base / "analysis" / f"breakdowns_learned_{CF}_nll_ind.parquet").filter(
        pl.col("game_id") == game)  # fmt: skip
    P = pl.read_parquet(base / "possessions.parquet").filter(pl.col("game_id") == game)
    led = D.L.ledger(cfg.version).select(["poss_uid", "points"])
    xfg = (pl.read_parquet(base / "analysis" / "ghost_points.parquet")
           .filter(pl.col("game_id") == game).select(["game_id", "possession_id", "xfg_real"])
           .unique(subset=["game_id", "possession_id"], keep="first"))  # fmt: skip
    ros = (D.L.rosters(cfg.version).filter(pl.col("game_id") == game)
           .select(pl.col("player_id"), pl.concat_str(["firstname", "lastname"], separator=" ")
                   .alias("name"), pl.col("abbreviation").alias("team"),
                   pl.col("position").replace_strict(POSITION_WORDS, default="player")
                   .alias("pos"))
           .unique(subset=["player_id"], keep="first", maintain_order=True))  # fmt: skip
    # the man each defender guarded at the start of the breakdown (HMM matchup)
    pt = load_game_set(base / "frames", [game], P)
    row = {int(p): i for i, p in enumerate(pt.possession_id)}
    M = pl.read_parquet(base / "matchups" / "hmm_strat_help" / f"{game}.parquet")
    # last man-to-man state (< 5) at or before the start: a breakdown often starts in help
    mm: dict = {}
    for p, s, d, st in M.filter(pl.col("state") < 5).sort("step").select(
            ["possession_id", "step", "def_id", "state"]).iter_rows():  # fmt: skip
        mm.setdefault((int(p), int(d)), []).append((int(s), int(st)))

    def man(p: int, s: int, d: int) -> int | None:
        prev = [st for k, st in mm.get((p, d), []) if k <= s]
        return int(pt.off_ids[row[p], prev[-1]]) if prev and p in row else None

    B = B.with_columns(pl.struct(["possession_id", "start_step", "def_id"]).map_elements(
        lambda r: man(r["possession_id"], r["start_step"], r["def_id"]), return_dtype=pl.Int64)
        .alias("man_id"),
        pl.len().over(["possession_id", "cascade_1s"]).alias("defenders_involved"))  # fmt: skip
    B = B.with_columns(pl.col("man_id").is_not_null().alias("has_matchup"))
    who = lambda pre: ros.rename({"player_id": f"{pre}_id", "name": f"{pre}", "team": f"{pre}_team",  # noqa: E731
                                  "pos": f"{pre}_position"})  # fmt: skip
    A = (B.join(P.select(["game_id", "possession_id", "period", "game_clock_start", "poss_uid",
                          "terminal_type", "terminal_desc"]), on=["game_id", "possession_id"],
                how="left")
         .join(led, on="poss_uid", how="left").join(xfg, on=["game_id", "possession_id"], how="left")
         .join(who("defender").rename({"defender_id": "def_id"}), on="def_id", how="left")
         .join(who("man"), on="man_id", how="left")
         .with_columns((pl.col("game_clock_start") - 0.2 * pl.col("start_step")).alias("clock_s"))
         .with_columns(pl.format("{}:{}", (pl.col("clock_s") // 60).cast(pl.Int64),
                                 (pl.col("clock_s") % 60).cast(pl.Int64).cast(pl.Utf8).str.zfill(2))
                       .alias("game_clock"),
                       pl.when(pl.col("clock_s") < CLIP_MIN_CLOCK_S).then(pl.lit("end of period"))
                       .when(pl.col("duration_s") > CLIP_MAX_DURATION_S).then(pl.lit("longer than 6 s"))
                       .when(~pl.col("initiator_1s")).then(pl.lit("follows another in its cascade"))
                       .otherwise(None).alias("left_out"))
         .sort(["period", "clock_s"], descending=[False, True]))  # fmt: skip
    A = A.select(["game_id", "possession_id", "period", "game_clock", "clock_s", "defender",
                  "def_id", "defender_team", "defender_position", "has_matchup", "man", "man_id",
                  "man_team",
                  "man_position", "duration_s", "peak_nll_ind", "initiator_1s", "cascade_1s",
                  "defenders_involved", "terminal_type", "terminal_desc", "points", "xfg_real",
                  "left_out"])  # fmt: skip
    def top(ok: pl.Expr, k: int) -> pl.DataFrame:
        return (A.filter(pl.col("left_out").is_null() & ok)
                .sort(["xfg_real", "points"], descending=True, nulls_last=True).head(k)
                .with_columns(pl.int_range(1, pl.len() + 1).alias("rank_by_shot_quality"))
                .sort(["period", "clock_s"], descending=[False, True])
                .drop(["left_out", "initiator_1s", "cascade_1s"]))  # fmt: skip

    return A, top(pl.col("has_matchup"), CLIP_TOP), top(~pl.col("has_matchup"), CLIP_NO_MATCHUP_TOP)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="SAS")
    ap.add_argument("--players", default="", help="comma-separated player ids for card figures")
    ap.add_argument("--game", default="0021500569")
    ap.add_argument("--team-pair", default="SAS,POR",
                    help="two teams on one card (review 2026-11-02: best points allowed, highest "
                    "early help)")  # fmt: skip
    ap.add_argument("--player-group", default="DET:203083,203484,1626169",
                    help="name:ids, players of one team on one card")  # fmt: skip
    ap.add_argument("--from-tables", action="store_true",
                    help="draw the cards from the shipped reports/ssac27/*.csv instead of the "
                    "processed tracking data (automatic when that data is absent)")  # fmt: skip
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    from_tables = args.from_tables or not (base / "analysis" / "onball_screens.parquet").exists()
    OUT_R.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    if from_tables:
        print("processed data absent or --from-tables: cards drawn from reports/ssac27/*.csv; "
              "tables and the clip list are not rebuilt")  # fmt: skip
        T = pl.read_csv(OUT_R / "team_cards.csv")
    else:
        T = team_cards(base)
        T.write_csv(OUT_R / "team_cards.csv")
    print("team card:", team_card_figure(T, args.team))
    if args.team_pair:
        print("team card:", team_pair_figure(T, args.team_pair.split(",")))
        print("team card:", team_pair_figure(T, args.team_pair.split(","), MAIN_TEAM_ROWS, "_main"))
    if from_tables:
        P = pl.read_csv(OUT_R / "player_cards.csv")
    else:
        P = player_cards(cfg)
        P.write_csv(OUT_R / "player_cards.csv")
        Cn = candidates(P)
        Cn.write_csv(OUT_R / "player_card_candidates.csv")
        print(Cn)
    for pid in [int(x) for x in args.players.split(",") if x]:
        print("player card:", player_card_figure(P, pid))
    if args.player_group:
        name, ids = args.player_group.split(":")
        print("player card:", player_group_figure(P, [int(x) for x in ids.split(",")], name))
    if from_tables:
        return
    A, C, Nm = clip_list(base, cfg, args.game)
    A.write_csv(OUT_R / f"clip_list_{args.game}_all.csv")
    C.write_csv(OUT_R / f"clip_list_{args.game}.csv")
    Nm.write_csv(OUT_R / f"clip_list_{args.game}_no_matchup.csv")
    print(A.group_by("left_out").len())
    print(C)


if __name__ == "__main__":
    main()
