"""Face validity of team-level screen defence measures against points allowed (tracking data only, no external data).

Points allowed per possession per defence team from the nbacore ledger (all tracked games;
possessions with data gaps excluded; half-court subset reported separately), correlated with the
team measures of analysis_team_screen_defense.py. No external data.

Output: reports/analysis/team_validity.json

Usage:
    uv run python scripts/analysis_team_validity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ghost import data as D


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    led = D.L.ledger(cfg.version).filter(~pl.col("data_gap"))
    pts = led.group_by("defense_team_id").agg(
        (pl.col("points").sum() / pl.len()).alias("pts_allowed_per_poss"),
        pl.len().alias("n_poss"),
    )
    pts_hc = (
        led.filter(~pl.col("is_transition"))
        .group_by("defense_team_id")
        .agg((pl.col("points").sum() / pl.len()).alias("pts_allowed_per_hc_poss"))
    )
    T = (
        pl.read_csv("reports/analysis/team_screen_defense.csv")
        .join(pts, left_on="team_id", right_on="defense_team_id")
        .join(pts_hc, left_on="team_id", right_on="defense_team_id")
        .sort("team_id")  # fixed row order: permutation / bootstrap draws are then reproducible
    )
    out = Path("reports/analysis")
    # 95 % game bootstrap of points allowed per possession (resampling the team's games; 500
    # draws), matching the help / switch intervals of analysis_team_screen_defense.py
    rng = np.random.default_rng(9)
    # sorted, order-preserving groups: the draws must not depend on hash order (reproducibility)
    per_game = (
        led.group_by(["defense_team_id", "game_id"])
        .agg(pl.col("points").sum().alias("pts"), pl.len().alias("n"))
        .sort(["defense_team_id", "game_id"])
    )
    ci = []
    for (tid,), g in per_game.group_by(["defense_team_id"], maintain_order=True):
        p_, n_ = g["pts"].to_numpy(), g["n"].to_numpy()
        ix = rng.integers(0, len(p_), size=(500, len(p_)))
        bs = p_[ix].sum(1) / n_[ix].sum(1)
        ci.append((tid, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))))
    ci = pl.DataFrame(ci, schema=["defense_team_id", "pts_lo", "pts_hi"], orient="row")
    pts = pts.join(ci, on="defense_team_id")
    pts.rename({"defense_team_id": "team_id"}).select(
        ["team_id", "pts_allowed_per_poss", "pts_lo", "pts_hi", "n_poss"]
    ).write_csv(out / "team_points_allowed.csv")
    res = {"n_teams": T.height}
    for m in ("help_rate_3s", "switch_rate", "help_delay_median_s"):
        for y in ("pts_allowed_per_poss", "pts_allowed_per_hc_poss"):
            x_, y_ = T[m].to_numpy(), T[y].to_numpy()
            r, p = stats.pearsonr(x_, y_)
            rho, prho = stats.spearmanr(x_, y_)
            res[f"{m}__vs__{y}"] = {
                "pearson_r": float(r),
                "p": float(p),
                "spearman_rho": float(rho),
                "p_spearman": float(prho),
            }
    # robustness of the headline association: without San Antonio (the best defence and the
    # largest-leverage point) and leave-one-team-out range
    x_, y_ = T["help_rate_3s"].to_numpy(), T["pts_allowed_per_poss"].to_numpy()
    teams = T["team"].to_list()
    keep = np.array([t != "SAS" for t in teams])
    r, p = stats.pearsonr(x_[keep], y_[keep])
    loo = [stats.pearsonr(np.delete(x_, i), np.delete(y_, i))[0] for i in range(len(teams))]
    res["help_rate_3s__vs__pts_allowed_per_poss__robustness"] = {
        "without_SAS": {"pearson_r": float(r), "p": float(p), "n_teams": int(keep.sum())},
        "leave_one_team_out_r_range": [float(min(loo)), float(max(loo))],
        "leave_one_team_out_extremes": {
            "weakest_when_dropped": teams[int(np.argmax(loo))],
            "strongest_when_dropped": teams[int(np.argmin(loo))],
        },
    }
    # inference for the abstract: Fisher-z 95 % CIs, permutation p for help, and whether the help
    # and switch correlations differ (Steiger 1980, dependent correlations sharing y; team bootstrap)
    s_ = T["switch_rate"].to_numpy()
    n = len(y_)

    def fisher_ci(r: float) -> list[float]:
        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
        return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]

    r1, r2 = stats.pearsonr(x_, y_)[0], stats.pearsonr(s_, y_)[0]
    r12 = stats.pearsonr(x_, s_)[0]
    rm2 = (r1**2 + r2**2) / 2
    f = min((1 - r12) / (2 * (1 - rm2)), 1.0)
    h = (1 - f * rm2) / (1 - rm2)
    z = (np.arctanh(r1) - np.arctanh(r2)) * np.sqrt((n - 3) / (2 * (1 - r12) * h))
    rng2 = np.random.default_rng(9)
    perm = np.mean(
        [abs(np.corrcoef(rng2.permutation(x_), y_)[0, 1]) >= abs(r1) for _ in range(20000)]
    )
    boot = []
    for _ in range(5000):
        i = rng2.integers(0, n, n)
        boot.append(np.corrcoef(x_[i], y_[i])[0, 1] - np.corrcoef(s_[i], y_[i])[0, 1])
    res["inference_pts_allowed_per_poss"] = {
        "help_r_ci95": fisher_ci(r1),
        "help_r_permutation_p": float(perm),
        "switch_r_ci95": fisher_ci(r2),
        "help_vs_switch_r": float(r12),
        "help_minus_switch_r": float(r1 - r2),
        "help_minus_switch_bootstrap_ci95": [float(q) for q in np.quantile(boot, [0.025, 0.975])],
        "steiger_z": float(z),
        "steiger_p": float(2 * stats.norm.sf(abs(z))),
        "n_team_level_tests": 6,
        "help_p_bonferroni_6": float(
            min(1.0, res["help_rate_3s__vs__pts_allowed_per_poss"]["p"] * 6)
        ),
    }
    # rule-ghost shot value (phase4_ghost_points.py): team mean expected points given up relative
    # to the rule ghost vs points allowed; the fixed split (first / last game date per split)
    gp = Path("reports/phase4") / f"{cfg.version}_all" / "ghost_points.json"
    if gp.exists():
        dep = pl.DataFrame(json.loads(gp.read_text())["teams"]).select(
            pl.col("defense_team_id").alias("team_id"), "dEP_team_per_shot"
        )
        J = T.join(dep, on="team_id")
        a, b = J["dEP_team_per_shot"].to_numpy(), J["pts_allowed_per_poss"].to_numpy()
        r_, p_ = stats.pearsonr(a, b)
        z_, se_ = np.arctanh(r_), 1 / np.sqrt(len(a) - 3)
        res["rule_ghost_dEP__vs__pts_allowed_per_poss"] = {
            "pearson_r": float(r_),
            "p": float(p_),
            "ci95": [float(np.tanh(z_ - 1.96 * se_)), float(np.tanh(z_ + 1.96 * se_))],
            "n_teams": int(len(a)),
        }
    g = D.games(cfg).filter(pl.col("split").is_not_null())
    res["fixed_split"] = {
        r["split"]: {"n_games": r["n"], "first_date": r["first"], "last_date": r["last"]}
        for r in g.group_by("split")
        .agg(pl.len().alias("n"), pl.col("game_date").min().alias("first"),
             pl.col("game_date").max().alias("last"))
        .iter_rows(named=True)
    }  # fmt: skip
    res["pts_allowed_per_poss_range"] = [
        float(T["pts_allowed_per_poss"].min()),
        float(T["pts_allowed_per_poss"].max()),
    ]
    res["best_defences"] = (
        T.sort("pts_allowed_per_poss")
        .head(5)
        .select(["team", "pts_allowed_per_poss", "help_rate_3s", "switch_rate"])
        .to_dicts()
    )
    res["worst_defences"] = (
        T.sort("pts_allowed_per_poss", descending=True)
        .head(5)
        .select(["team", "pts_allowed_per_poss", "help_rate_3s", "switch_rate"])
        .to_dicts()
    )
    res["note"] = (
        "Descriptive association across 30 teams; not causal. Points allowed include "
        "all tracked possessions of the half season."
    )
    (out / "team_validity.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    np.set_printoptions(precision=3)


if __name__ == "__main__":
    main()
