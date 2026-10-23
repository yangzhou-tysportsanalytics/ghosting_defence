"""xFG calibration by shooter subgroup on the test games (writing request A4a).

A shared expected-make model can be biased for the best shooters (Davis & Robberechts 2024).
The models are refitted on the train games exactly as scripts/phase4_xfg.py (and xfg-v1.0):
M2 (no shooter term) and M3 (shrunk shooter-by-zone skill, the chosen model). On the test
shots, by group: n, mean prediction, observed make rate, observed - predicted with a 95 %
game-bootstrap interval, and the calibration slope (logistic regression of the outcome on the
model's logit). Groups:
* deciles of the shooter's shrunk zone skill (the M3 offset of each shot; train counts only);
* the 50 shooters with the most train attempts vs the rest;
* zone x (top 50 / rest).

Output: reports/analysis/xfg_subgroup_calibration.json

Usage:
    uv run python scripts/analysis_xfg_subgroups.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.xfg.model import FEATURE_SETS, Logistic, fit_xfg

B = 200


def calib_slope(y: np.ndarray, p: np.ndarray) -> float:
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    m = Logistic(C=1e6, tol=1e-10, max_iter=200).fit(lg, y.astype(int))
    return float(m.coef_[0][0])


def group_stats(d: pl.DataFrame, rng: np.random.Generator) -> dict:
    y, games = d["made"].to_numpy().astype(float), d["game_id"].to_numpy()
    out = {"n": d.height}
    ug, inv = np.unique(games, return_inverse=True)
    idx = rng.integers(0, len(ug), size=(B, len(ug)))
    for m in ("M2", "M3"):
        p = d[f"p_{m}"].to_numpy()
        gap = y - p
        s, c = np.bincount(inv, weights=gap), np.bincount(inv).astype(float)
        boot = s[idx].sum(1) / c[idx].sum(1)
        out[m] = {
            "mean_pred": float(p.mean()),
            "obs": float(y.mean()),
            "obs_minus_pred": float(gap.mean()),
            "obs_minus_pred_ci95": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],  # fmt: skip
            "calibration_slope": calib_slope(y, p) if d.height >= 200 else None,
        }
    return out


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all" / "analysis"
    S = pl.read_parquet(base / "shots.parquet").filter(pl.col("def1_ft").is_not_null())
    S = S.filter(pl.col("release_method").is_in(["rim", "apex"]))  # as phase4_xfg.py
    tr, te = S.filter(pl.col("split") == "train"), S.filter(pl.col("split") == "test")
    m2 = fit_xfg(tr, FEATURE_SETS["M2_plus_dribble"])
    m3 = fit_xfg(tr, FEATURE_SETS["M3_plus_shooter"])
    ref = json.loads((Path("reports/phase4") / f"{cfg.version}_all" / "xfg.json").read_text())
    c_ref = ref["models"]["M3_plus_shooter"]["coefficients"]
    assert all(math.isclose(v, c_ref[k], rel_tol=1e-6, abs_tol=1e-9)
               for k, v in m3.coefficients().items()), "refit differs from the frozen xFG"  # fmt: skip
    skill = m3.prior.skill(te, leave_one_out=False)
    top = (tr.group_by("shooter_id").len().sort("len", descending=True).head(50))["shooter_id"]
    te = te.with_columns(
        pl.Series("p_M2", m2.predict(te)),
        pl.Series("p_M3", m3.predict(te)),
        pl.Series("skill", skill),
        pl.col("shooter_id").is_in(top.to_list()).alias("top50"),
    )
    edges = np.quantile(skill, np.linspace(0.1, 0.9, 9))
    te = te.with_columns(pl.Series("skill_decile", np.searchsorted(edges, skill, side="right")))
    rng = np.random.default_rng(9)
    rep: dict = {"n_test_shots": te.height, "models": {"M2": "M2_plus_dribble (no shooter term)",
                 "M3": "M3_plus_shooter (chosen; xfg-v1.0)"},
                 "overall": group_stats(te, rng), "by_skill_decile": {}, "top50_vs_rest": {},
                 "zone_by_top50": {}}  # fmt: skip
    for k in range(10):
        d = te.filter(pl.col("skill_decile") == k)
        rep["by_skill_decile"][str(k + 1)] = {
            "skill_range": [float(d["skill"].min()), float(d["skill"].max())],
            **group_stats(d, rng),
        }
    for lab, flag in (("top50", True), ("rest", False)):
        rep["top50_vs_rest"][lab] = group_stats(te.filter(pl.col("top50") == flag), rng)
        for z in sorted(te["zone"].unique().to_list()):
            d = te.filter((pl.col("top50") == flag) & (pl.col("zone") == z))
            rep["zone_by_top50"][f"{z}_{lab}"] = group_stats(d, rng)
    p = Path("reports/analysis/xfg_subgroup_calibration.json")
    p.write_text(json.dumps(rep, indent=1))
    for g, v in (
        [("overall", rep["overall"])]
        + [(f"decile {k}", v) for k, v in rep["by_skill_decile"].items()]
        + list(rep["top50_vs_rest"].items())
    ):
        print(f"{g:>10} n={v['n']:5d}  M2 gap {v['M2']['obs_minus_pred']:+.3f}  "
              f"M3 gap {v['M3']['obs_minus_pred']:+.3f} {np.round(v['M3']['obs_minus_pred_ci95'], 3)}  "
              f"slope M3 {v['M3']['calibration_slope']}")  # fmt: skip
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
