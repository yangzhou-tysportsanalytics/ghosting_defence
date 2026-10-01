"""Phase 4 hierarchical model on rule-ghost deviations (sag and |deviation|), with prior
sensitivity. Uses the context adjustment of phase4_reliability.adjust.

Outputs: reports/phase4/<version>_all/hier<suffix>.json and hier_players_<metric><suffix>.csv
(suffix "_phases" with --context phases: phase shares in the context term, D-021)
(players with >= 300 defensive possessions; NOT a best/worst defender ranking - a positioning
style descriptor with 95 % intervals).

Usage:
    uv run python scripts/phase4_hier.py [--context base|phases]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.hier.model import Cells, definitional, fit, summarize

spec = importlib.util.spec_from_file_location(
    "rel", Path(__file__).resolve().parent / "phase4_reliability.py"
)
rel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="base", choices=["base", "phases"], help="D-021")
    ap.add_argument("--ghost", default="rule", help='"rule" or "learned:<name>"')
    args = ap.parse_args()
    extra = tuple(rel.PHASE_COVS) if args.context == "phases" else ()
    suffix = "" if args.context == "base" else "_phases"  # prefixed by the ghost suffix below
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    df, metrics, gsfx = rel.load_dev(base, args.ghost)
    suffix = gsfx + suffix
    if extra:
        df = rel.with_phases(df, base)
    ros = (
        D.L.rosters(cfg.version)
        .select(["player_id", "firstname", "lastname"])
        .unique(subset=["player_id"], keep="first")
    )
    abbr = {}
    for r in D.games(cfg).iter_rows(named=True):
        abbr[r["home_team_id"]] = r["home_abbr"]
        abbr[r["visitor_team_id"]] = r["visitor_abbr"]
    out_dir = Path("reports/phase4") / f"{cfg.version}_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "note": "two-stage Gaussian hierarchical model; stage-1 context coefficients fixed",
        "metrics": {},
    }
    for y in ("sag_ft", "dev_ft") if args.ghost == "rule" else metrics:
        res, _ = rel.adjust(df, y, extra)
        d = df.with_columns(pl.Series("r", res))
        sigma_e = float(
            np.sqrt(
                d.group_by(["def_id", "defense_team_id"])
                .agg(pl.col("r").var())
                .drop_nulls()["r"]
                .mean()
            )
        )
        c = d.group_by(["def_id", "defense_team_id"]).agg(
            pl.col("r").mean().alias("ybar"), pl.len().alias("n")
        )
        teams = sorted(c["defense_team_id"].unique().to_list())
        players = sorted(c["def_id"].unique().to_list())
        ti = {t: i for i, t in enumerate(teams)}
        pi = {p: i for i, p in enumerate(players)}
        cells = Cells(
            ybar=c["ybar"].to_numpy(),
            n=c["n"].to_numpy().astype(float),
            team_idx=np.array([ti[t] for t in c["defense_team_id"]]),
            player_idx=np.array([pi[p] for p in c["def_id"]]),
            n_team=len(teams),
            n_player=len(players),
            sigma_e=sigma_e,
        )
        met = {
            "sigma_e_ft": sigma_e,
            "n_cells": c.height,
            "n_players": len(players),
            "prior_sensitivity": {},
        }
        for scale in (0.5, 1.0, 2.0):
            s = fit(cells, prior_scale=scale)
            tm, pw = definitional(s, cells)
            var_t = tm.var(axis=1)
            big = cells.n >= 300
            var_p = pw[:, big].var(axis=1)
            met["prior_sensitivity"][str(scale)] = {
                "tau_team": summarize(s["tau_team"]),
                "tau_player": summarize(s["tau_player"]),
                "var_team_mean_effect": summarize(var_t),
                "var_player_within_team_ge300": summarize(var_p),
                "team_share": summarize(var_t / (var_t + var_p)),
            }
            if scale == 1.0:
                keep = np.flatnonzero(big)
                rows = []
                for j in keep:
                    pid = int(c["def_id"][int(j)])
                    rows.append(
                        {
                            "player_id": pid,
                            "team": abbr.get(int(c["defense_team_id"][int(j)])),
                            "n_poss": int(cells.n[j]),
                            **{f"effect_{k}": v for k, v in summarize(pw[:, j]).items()},
                        }
                    )
                P = pl.DataFrame(rows).join(ros, on="player_id", how="left").sort("effect_mean")
                P.write_csv(out_dir / f"hier_players_{y}{suffix}.csv")
                met["n_player_cells_ge300"] = int(big.sum())
                met["share_player_intervals_excluding_zero"] = float(
                    ((P["effect_lo95"] > 0) | (P["effect_hi95"] < 0)).mean()
                )
                tmean = [{"team": abbr.get(t), **summarize(tm[:, ti[t]])} for t in teams]
                met["team_effects"] = sorted(tmean, key=lambda r: r["mean"])
        report["metrics"][y] = met
        print(y, json.dumps({k: v for k, v in met.items() if k != "team_effects"}, indent=1)[:2500])
    report["context"] = args.context
    (out_dir / f"hier{suffix}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
