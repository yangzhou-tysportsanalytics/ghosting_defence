"""Does departure measure how spread the ghost's map is? (SSAC final review R4 P1, 2026-11-02)

Inputs: the per-step spread files of phase4_learned_ghost_spread.py and the reliability,
meta-metric, hierarchical and predictive outputs for the departure (cf_<tag>) and its two
spread-free versions (cf_<tag>_excess = NLL minus the map's entropy; cf_<tag>_pit = coverage level
of the smallest region holding the real position).

Reports
* defender-step level: correlation of departure (NLL) with the map's entropy;
* player level (players with >= 300 defensive possessions): share of the between-player
  variance of mean departure explained by mean entropy (raw and within team);
* per version: split-half reliability within team, share unexplained by measured role
  (position, height, threat of the man guarded), predictive rank correlation with xFG allowed in
  the other half of the games (both directions), the Detroit card values, and the rank
  correlation of the player effects with those of the departure.

Output: reports/analysis/departure_spread.json

Usage:
    uv run python scripts/analysis_departure_spread.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ghost import data as D

TAG = "cf_gpu_d128_lr1e-3"
VERSIONS = {"departure": TAG, "excess": f"{TAG}_excess", "pit": f"{TAG}_pit"}
DETROIT = {203083: "Drummond", 203484: "Caldwell-Pope", 1626169: "S. Johnson"}
MIN_POSS = 300
R4, R5 = Path("reports/phase4/v1.6_all"), Path("reports/phase5/v1.6_all")


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def spread_levels(base: Path) -> dict:
    files = str(base / "learned_ghost" / f"{TAG}_spread" / "*.parquet")
    S = pl.scan_parquet(files)
    step = S.select(pl.corr("nll", "entropy").alias("r"), pl.len().alias("n"),
                    pl.col("entropy").mean().alias("entropy_mean"),
                    pl.col("nll").mean().alias("nll_mean"),
                    (pl.col("pit") <= 0.8).mean().alias("coverage_80")).collect().row(0, named=True)  # fmt: skip
    per = (S.group_by(["game_id", "possession_id", "def_id"])
           .agg(pl.col("nll").mean(), pl.col("entropy").mean()).collect())  # fmt: skip
    P = pl.read_parquet(base / "possessions.parquet").select(
        ["game_id", "possession_id", "defense_team_id"]).unique()  # fmt: skip
    per = per.join(P, on=["game_id", "possession_id"], how="left")
    team = (per.group_by(["def_id", "defense_team_id"]).len()
            .sort("len", descending=True).unique("def_id", keep="first")
            .select("def_id", pl.col("defense_team_id").alias("team")))  # fmt: skip
    pm = (per.group_by("def_id").agg(pl.col("nll").mean(), pl.col("entropy").mean(),
                                     pl.len().alias("n_poss"))
          .filter(pl.col("n_poss") >= MIN_POSS).join(team, on="def_id"))  # fmt: skip
    x, y = pm["entropy"].to_numpy(), pm["nll"].to_numpy()
    wt = pm.with_columns(*(pl.col(c) - pl.col(c).mean().over("team") for c in ("nll", "entropy")))
    return {"defender_step": {k: float(v) for k, v in step.items()},
            "player_level": {"n_players": pm.height,
                             "r2_mean_departure_on_mean_entropy": float(np.corrcoef(x, y)[0, 1] ** 2),
                             "r2_within_team": float(np.corrcoef(wt["entropy"], wt["nll"])[0, 1] ** 2)}}  # fmt: skip


def version_summary(name: str, base_effects: pl.DataFrame | None) -> dict:
    rel = _load(R4 / f"reliability_learned_{name}_phases.json")
    mm = _load(Path("reports/analysis") / f"meta_metrics_learned_{name}.json")
    pv = _load(R5 / f"predictive_validity_phases_learned_{name}.json")
    eff_p = R4 / f"hier_players_nll_ind_learned_{name}_phases_gclust.csv"
    out: dict = {"ghost": name}
    if rel:
        out["split_half_r_within_team"] = rel["metrics"]["nll_ind"]["split_half_r_player_within_team"]
    if mm:
        out["share_unexplained_by_role"] = mm["metrics"]["nll_ind_within_team"]["independence"][
            "position_height_guarded_threat"]["independence"]  # fmt: skip
    if pv:
        pl_ = pv["player_level"]
        out["predictive_rho_xfg_m3"] = {
            d: {"rho": pl_[d]["xfg_m3_cf"]["rho"], "ci95": pl_[d]["xfg_m3_cf"]["ci95"]}
            for d in ("trait_parity0_outcome_parity1", "trait_parity1_outcome_parity0")}  # fmt: skip
    if eff_p.exists():
        E = pl.read_csv(eff_p)
        out["detroit"] = {DETROIT[r["player_id"]]: {k: r[k] for k in ("effect_mean", "effect_lo95", "effect_hi95")}
                          for r in E.filter(pl.col("player_id").is_in(list(DETROIT))).iter_rows(named=True)}  # fmt: skip
        out["detroit_percentile"] = {
            DETROIT[r["player_id"]]: float((E["effect_mean"] <= r["effect_mean"]).mean())
            for r in E.filter(pl.col("player_id").is_in(list(DETROIT))).iter_rows(named=True)}  # fmt: skip
        if base_effects is not None:
            J = E.join(base_effects, on="player_id", suffix="_base")
            out["rank_corr_with_departure_effects"] = float(
                stats.spearmanr(J["effect_mean"], J["effect_mean_base"]).statistic)
    return out


def main() -> None:
    base = D.DataConfig.load(game_set="all").processed_dir / "all"
    rep = {"spread": spread_levels(base), "versions": {}}
    bp = R4 / f"hier_players_nll_ind_learned_{TAG}_phases_gclust.csv"
    base_eff = pl.read_csv(bp).select("player_id", "effect_mean") if bp.exists() else None
    for k, name in VERSIONS.items():
        rep["versions"][k] = version_summary(name, base_eff if k != "departure" else None)
    out = Path("reports/analysis/departure_spread.json")
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
