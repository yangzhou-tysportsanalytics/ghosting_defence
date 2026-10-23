"""Nearest-defender height and closing speed at the release; xFG variant (writing request A4b).

For every shot of shots.parquet (release located from the ball, as the xFG training data):
* def1_id: the nearest opponent at the release frame (the defender of ``def1_ft``);
* def1_closing_fps: decrease of his distance to the shooter over the 0.2 s before the release
  (nearest frame to t - 200 ms within 80 ms), feet per second, positive = closing;
* def1_height_cm: his height (nbacore player_bio; missing -> mean, flagged).
No coordinates are stored.

Variant M3D = M3 (xfg-v1.0) + standardised height + clipped closing speed (+-20 ft/s) + missing
flags, fitted on the train games like M3; test log loss, Brier, AUC against M3; the change in
predictions (correlation, mean absolute difference); coefficients. xfg-v1.0 is not changed.

Outputs: <processed_dir>/all/analysis/shots_def1.parquet,
         reports/analysis/xfg_defender_variant.json

Usage:
    uv run python scripts/analysis_xfg_defender_features.py [--rebuild]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.xfg.model import FEATURE_SETS, Logistic, ShooterPrior, design, evaluate, fit_xfg

BALL_ID = -1


def features(cfg, S: pl.DataFrame) -> pl.DataFrame:
    from nbacore import paths

    rows, t0 = [], time.time()
    games = sorted(S["game_id"].unique().to_list())
    for gi, gid in enumerate(games, 1):
        sh = S.filter(pl.col("game_id") == gid)
        p = paths.release_dir(cfg.version) / "frames" / f"{str(gid).zfill(10)}.parquet"
        lo, hi = int(sh["t_release_ms"].min()) - 400, int(sh["t_release_ms"].max())
        F = (
            pl.scan_parquet(p)
            .filter(pl.col("unix_ms").is_between(lo, hi))
            .select(["period", "unix_ms", "team_id", "player_id", "x", "y"])
            .collect()
        )
        times = F["unix_ms"].unique().sort().to_numpy()
        for r in sh.iter_rows(named=True):
            t = int(r["t_release_ms"])
            k = int(np.argmin(np.abs(times - (t - 200)))) if times.size else 0
            tp = int(times[k]) if times.size else t
            if abs(tp - (t - 200)) > 80:
                continue
            fr = F.filter((pl.col("unix_ms") == t) & (pl.col("period") == r["period"]))
            fp = F.filter((pl.col("unix_ms") == tp) & (pl.col("period") == r["period"]))
            me = fr.filter(pl.col("player_id") == r["shooter_id"])
            opp = fr.filter(
                (pl.col("team_id") != BALL_ID) & (pl.col("team_id") != r["offense_team_id"])
            )
            if me.height == 0 or opp.height == 0:
                continue
            sx, sy = float(me["x"][0]), float(me["y"][0])
            d = np.hypot(opp["x"].to_numpy() - sx, opp["y"].to_numpy() - sy)
            j = int(np.argmin(d))
            did = int(opp["player_id"][j])
            mp, dp = (
                fp.filter(pl.col("player_id") == r["shooter_id"]),
                fp.filter(pl.col("player_id") == did),
            )
            closing = None
            if mp.height and dp.height:
                d_prev = float(
                    np.hypot(
                        float(dp["x"][0]) - float(mp["x"][0]), float(dp["y"][0]) - float(mp["y"][0])
                    )
                )
                closing = (d_prev - float(d[j])) / ((t - tp) / 1000)
            rows.append({"game_id": gid, "event_uid": r["event_uid"], "def1_id": did,
                         "def1_ft_check": float(d[j]), "def1_closing_fps": closing})  # fmt: skip
        if gi % 100 == 0 or gi == len(games):
            print(f"[{gi}/{len(games)}] {time.time() - t0:.0f}s rows={len(rows)}", flush=True)
    return pl.DataFrame(rows)


TERMS = {"m3d": ("height", "closing"), "m3h": ("height",), "m3c": ("closing",)}


def extra_columns(
    df: pl.DataFrame, h_mean: float, h_sd: float, terms=("height", "closing")
) -> tuple[np.ndarray, list[str]]:
    h = df["def1_height_cm"].to_numpy().astype(float)
    hm = np.isnan(h)
    c = df["def1_closing_fps"].to_numpy().astype(float)
    cm = np.isnan(c)
    cols, names = [], []
    if "height" in terms:
        cols += [np.where(hm, 0.0, (h - h_mean) / h_sd), hm.astype(float)]
        names += ["def1_height_z", "def1_height_missing"]
    if "closing" in terms:
        cols += [np.clip(np.nan_to_num(c, nan=0.0), -20, 20) / 10, cm.astype(float)]
        names += ["def1_closing_10fps", "def1_closing_missing"]
    return np.column_stack(cols), names


def with_def1(cfg, S: pl.DataFrame) -> pl.DataFrame:
    """Join the cached def1 features (shots_def1.parquet) and the defender's height; rows in
    a fixed order (bit-identical refits)."""
    base = cfg.processed_dir / "all" / "analysis"
    F = pl.read_parquet(base / "shots_def1.parquet")
    bio = D.L.player_bio(cfg.version).select(
        pl.col("player_id").cast(pl.Int64).alias("def1_id"),
        pl.col("height_cm").alias("def1_height_cm"),
    )
    S = S.join(F, on=["game_id", "event_uid"], how="left").join(bio, on="def1_id", how="left")
    return S.sort(["game_id", "event_uid"])


@dataclass
class M3D:
    prior: ShooterPrior
    clf: Logistic
    h_mean: float
    h_sd: float
    terms: tuple = ("height", "closing")

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        X, _ = design(df, FEATURE_SETS["M3_plus_shooter"], self.prior.skill(df, False))
        E, _ = extra_columns(df, self.h_mean, self.h_sd, self.terms)
        return self.clf.predict_proba(np.column_stack([X, E]))[:, 1]


def fit_m3d(tr: pl.DataFrame, terms=("height", "closing")) -> tuple[M3D, list[str]]:
    """M3 (xfg-v1.0 design, leave-one-out shooter skill on train) + the def1 terms."""
    prior = ShooterPrior.fit(tr)
    h_mean, h_sd = float(tr["def1_height_cm"].mean()), float(tr["def1_height_cm"].std())
    X, names = design(tr, FEATURE_SETS["M3_plus_shooter"], prior.skill(tr, leave_one_out=True))
    E, enames = extra_columns(tr, h_mean, h_sd, terms)
    clf = Logistic(C=1.0).fit(np.column_stack([X, E]), tr["made"].to_numpy().astype(int))
    return M3D(prior, clf, h_mean, h_sd, tuple(terms)), names + enames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all" / "analysis"
    S = pl.read_parquet(base / "shots.parquet").filter(pl.col("def1_ft").is_not_null())
    S = S.filter(pl.col("release_method").is_in(["rim", "apex"]))  # as phase4_xfg.py
    cache = base / "shots_def1.parquet"
    if cache.exists() and not args.rebuild:
        F = pl.read_parquet(cache)
    else:
        F = features(cfg, S)
        F.write_parquet(cache)
    S = with_def1(cfg, S)
    check = S.drop_nulls("def1_ft_check")
    rep: dict = {"n_shots": S.height, "share_with_def1_id": float(S["def1_id"].is_not_null().mean()),
                 "share_with_closing": float(S["def1_closing_fps"].is_not_null().mean()),
                 "share_with_height": float(S["def1_height_cm"].is_not_null().mean()),
                 "max_abs_diff_def1_ft_vs_shots_table": float((check["def1_ft_check"] - check["def1_ft"]).abs().max())}  # fmt: skip
    tr, te = S.filter(pl.col("split") == "train"), S.filter(pl.col("split") == "test")
    m3 = fit_xfg(tr, FEATURE_SETS["M3_plus_shooter"])
    m3d, names = fit_m3d(tr)
    y_te = te["made"].to_numpy()
    p3, p3d = m3.predict(te), m3d.predict(te)
    strip = lambda e: {k: v for k, v in e.items() if k != "calibration_deciles"}  # noqa: E731
    rep["test"] = {"M3": strip(evaluate(y_te, p3)), "M3D": strip(evaluate(y_te, p3d))}
    rep["test_prediction_change"] = {"corr": float(np.corrcoef(p3, p3d)[0, 1]),
                                     "mean_abs_diff": float(np.abs(p3 - p3d).mean()),
                                     "p95_abs_diff": float(np.quantile(np.abs(p3 - p3d), 0.95))}  # fmt: skip
    rep["coefficients_M3D"] = {
        "intercept": float(m3d.clf.intercept_[0]),
        **{n: float(c) for n, c in zip(names, m3d.clf.coef_[0], strict=True)},
    }
    rep["height_standardisation_cm"] = {"mean": m3d.h_mean, "sd": m3d.h_sd}
    cs = tr["def1_closing_fps"].drop_nulls().to_numpy()
    rep["closing_speed_fps_quantiles_train_5_25_50_75_95"] = [
        float(x) for x in np.quantile(cs, [0.05, 0.25, 0.5, 0.75, 0.95])
    ]
    p = Path("reports/analysis/xfg_defender_variant.json")
    p.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
