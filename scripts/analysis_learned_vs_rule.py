"""Player-level agreement of the learned-ghost and rule-ghost deviations (writing request, v0.10).

Players with >= 300 defensive half-court possessions. For the learned metrics (cross-fitted, out
of fold: nll_ind, nll_team, d_mean, d_mode) against the rule-ghost dev and sag: Pearson and
Spearman correlations of player means, raw and in the main specification (context-adjusted with
the D-021 phase shares, within team), with player-bootstrap 95 % intervals.

Output: reports/phase4/<version>_all/learned_vs_rule_players_<ghost>.json

Usage:
    uv run python scripts/analysis_learned_vs_rule.py [--ghost cf_gpu_d128_lr1e-3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ghost import data as D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_reliability import MIN_POSS, PHASE_COVS, adjust, load_dev, with_phases  # noqa: E402

LEARNED = ("nll_ind", "nll_team", "d_mean", "d_mode")
RULE = ("dev_ft", "sag_ft")


def boot_ci(a: np.ndarray, b: np.ndarray, reps: int, rng) -> list[float]:
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    rs = [np.corrcoef(a[i], b[i])[0, 1] for i in idx]
    return [float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghost", default="cf_gpu_d128_lr1e-3")
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    df, _, _ = load_dev(base, f"learned:{args.ghost}")
    df = with_phases(df, base)
    players = df.group_by("def_id").len().filter(pl.col("len") >= MIN_POSS)["def_id"].to_list()
    cols = {}
    for m in LEARNED + RULE:
        res, _ = adjust(df, m, tuple(PHASE_COVS))
        d = df.with_columns(pl.Series("adj", res))
        tm = d.group_by("defense_team_id").agg(pl.col("adj").mean().alias("tm"))
        d = d.join(tm, on="defense_team_id").with_columns((pl.col("adj") - pl.col("tm")).alias("w"))
        pm = (
            d.filter(pl.col("def_id").is_in(players))
            .group_by("def_id")
            .agg(pl.col(m).mean().alias("raw"), pl.col("w").mean().alias("main"))
            .sort("def_id")
        )
        cols[m] = pm
    rng = np.random.default_rng(9)
    rep: dict = {"ghost": args.ghost, "n_players": len(players), "min_possessions": MIN_POSS,
                 "specifications": {"raw": "player means of the per-possession values",
                                    "main": "context-adjusted (D-021 phases), within team"},
                 "correlations": {}}  # fmt: skip
    for spec in ("raw", "main"):
        out = {}
        for a in LEARNED:
            for b in RULE:
                x, y = cols[a][spec].to_numpy(), cols[b][spec].to_numpy()
                out[f"{a}__{b}"] = {"pearson": float(np.corrcoef(x, y)[0, 1]),
                                    "pearson_ci95_player_bootstrap": boot_ci(x, y, args.reps, rng),
                                    "spearman": float(stats.spearmanr(x, y)[0])}  # fmt: skip
        rep["correlations"][spec] = out
    p = Path("reports/phase4") / f"{cfg.version}_all" / f"learned_vs_rule_players_{args.ghost}.json"
    p.write_text(json.dumps(rep, indent=1))
    for spec, out in rep["correlations"].items():
        for k, v in out.items():
            print(
                f"{spec:>4} {k:>16}: r {v['pearson']:+.3f} {np.round(v['pearson_ci95_player_bootstrap'], 3)}  rho {v['spearman']:+.3f}"
            )
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
