"""Fit and evaluate the xFG models (train games) on val and test games; save the chosen model.

Output: reports/phase4/<version>_all/xfg.json, <processed_dir>/all/analysis/xfg_model.pkl

Usage:
    uv run python scripts/phase4_xfg.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import polars as pl

from ghost import data as D
from ghost.xfg.model import FEATURE_SETS, evaluate, fit_xfg


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all" / "analysis"
    S = pl.read_parquet(base / "shots.parquet").filter(pl.col("def1_ft").is_not_null())
    if "release_method" in S.columns:
        # only releases located from the ball with the shooter in hand ("rim" / "apex"); excluded:
        # pbp fallback, *_nohand / *_anyhand, the widened-window *_wide variants and "hand"
        # (nbacore v1.6). Same training population as under v1.5, where only rim / apex passed.
        S = S.filter(pl.col("release_method").is_in(["rim", "apex"]))
    tr, va, te = (S.filter(pl.col("split") == s) for s in ("train", "val", "test"))
    rep = {
        "n": {"train": tr.height, "val": va.height, "test": te.height},
        "base_rate_train": float(tr["made"].mean()),
        "models": {},
    }
    models = {}
    for name, groups in FEATURE_SETS.items():
        m = fit_xfg(tr, groups)
        models[name] = m
        rep["models"][name] = {
            "val": evaluate(va["made"].to_numpy(), m.predict(va)),
            "test": evaluate(te["made"].to_numpy(), m.predict(te)),
            "coefficients": m.coefficients(),
        }
        v = rep["models"][name]["val"]
        print(
            f"{name}: val logloss {v['log_loss']:.4f} brier {v['brier']:.4f} auc {v['auc']:.3f}",
            flush=True,
        )
    best = "M3_plus_shooter"
    rep["chosen"] = best
    rep["shooter_prior"] = models[best].prior.to_dict()
    # effect of 2 extra feet of space for a typical catch-and-shoot above-the-break three
    ex = te.filter((pl.col("zone") == "above3") & pl.col("catch_and_shoot"))
    if ex.height:
        base_p = models[best].predict(ex).mean()
        wider = ex.with_columns(
            (pl.col("def1_ft") + 2).alias("def1_ft"),
            (pl.col("def_front_ft") + 2).alias("def_front_ft"),
        )
        rep["effect_plus_2ft_catch_shoot_above3"] = {
            "xfg": float(base_p),
            "xfg_plus_2ft": float(models[best].predict(wider).mean()),
        }
    out = Path("reports/phase4") / f"{cfg.version}_all"
    out.mkdir(parents=True, exist_ok=True)
    (out / "xfg.json").write_text(json.dumps(rep, indent=2))
    with open(base / "xfg_model.pkl", "wb") as fp:
        pickle.dump(models[best], fp)
    print(json.dumps({k: v for k, v in rep.items() if k != "models"}, indent=2))
    print(json.dumps(rep["models"][best]["coefficients"], indent=1))


if __name__ == "__main__":
    main()
