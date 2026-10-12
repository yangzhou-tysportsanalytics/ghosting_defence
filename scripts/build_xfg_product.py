"""Freeze the xFG models and register them as the shared product ``ghost_xfg`` (for other projects).

Output: <NBA_DATA_ROOT>/products/ghost_xfg/<version>/
    shots_xfg.parquet        one row per nbacore shot_release (key event_uid): features, flags,
                             frozen predictions (train-only models) and cross-fitted predictions
    M2.json, M3.json         frozen models (train games, trusted releases); M3 needs
    M3_shooter_prior.parquet the shooter-by-zone train counts (unseen shooters -> offset 0)
    calibration.json         out-of-sample metrics and decile calibration
    consistency.json         ghost.xfg.api.features at real releases vs the shot table
    README.md, MANIFEST.json

Usage:
    uv run python scripts/build_xfg_product.py [--version v1.0] [--force]
    uv run python scripts/build_xfg_product.py --out-dir <dir>   # rebuild elsewhere (checks)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from nbacore import paths

from ghost import data as D
from ghost.xfg import api
from ghost.xfg.model import evaluate

PRODUCT = "ghost_xfg"
ID_COLS = ["event_uid", "game_id", "pbp_event_num", "period", "t_release_ms", "shooter_id",
           "offense_team_id", "defense_team_id", "poss_uid", "made", "blocked", "release_method"]  # fmt: skip
NUMERIC = ["x", "y", "dist_ft", "angle_deg", "def1_ft", "def2_ft", "def_front_ft", "touch_s"]
EXACT = ["is_three", "n_dribbles", "catch_and_shoot", "is_transition", "zone"]


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def consistency(shots: pl.DataFrame, cfg, games: list[str]) -> dict:
    """Recompute features with the interface at the real releases of ``games``."""
    worst, mism, n = 0.0, {c: 0 for c in EXACT}, 0
    for gid in games:
        s = shots.filter(pl.col("game_id") == gid)
        if s.height == 0:
            continue
        f = api.features(gid, s.select(pl.col("t_release_ms").alias("unix_ms"), "shooter_id"), cfg)
        n += s.height
        for c in NUMERIC:
            a, b = s[c].to_numpy().astype(float), f[c].to_numpy().astype(float)
            both = ~(np.isnan(a) & np.isnan(b))
            worst = max(worst, float(np.nanmax(np.abs(a[both] - b[both]), initial=0.0)))
            mism["nan_pattern_" + c] = mism.get("nan_pattern_" + c, 0) + int(
                (np.isnan(a) != np.isnan(b)).sum()
            )
        for c in EXACT:
            mism[c] += int(
                (s[c].cast(pl.Utf8).fill_null("~") != f[c].cast(pl.Utf8).fill_null("~")).sum()
            )
    return {"games": len(games), "shots": n, "max_abs_numeric_diff": worst, "mismatches": mism}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--out-dir", default=None, help="write here instead of the shared products directory"
    )
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    root = Path(args.out_dir) if args.out_dir else paths.products_dir() / PRODUCT
    out = root / args.version
    if out.exists() and not args.force:
        raise SystemExit(
            f"{out} exists; published product versions are immutable (--force to rebuild)"
        )
    tmp = out.with_name(args.version + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    shots = pl.read_parquet(cfg.processed_dir / "all" / "analysis" / "shots.parquet")
    # the model reads a missing is_transition as False (475 shots whose poss_uid is not in the
    # ledger); publish it as False, as the interface does
    shots = shots.with_columns(pl.col("is_transition").fill_null(False))
    games = D.games(cfg).select(["game_id", "fold"])
    shots = shots.join(games, on="game_id", how="left").with_columns(
        (
            pl.col("release_method").is_in(api.TRUSTED_METHODS) & pl.col("def1_ft").is_not_null()
        ).alias("release_trusted")
    )
    shots = shots.with_columns(
        (pl.col("release_trusted") & (pl.col("split") == "train")).alias("in_training")
    )
    train_games = shots.filter(pl.col("split") == "train")["game_id"].unique().to_list()

    t0 = time.time()
    frozen = {v: api.fit(shots, train_games, v) for v in api.VERSIONS}
    fit_seconds = (time.time() - t0) / len(frozen)
    for v, m in frozen.items():
        api.save(m, tmp, v)
        shots = shots.with_columns(pl.Series(f"xfg_{v}", api.predict(shots, m)))
    # cross-fitting over nbacore folds: every shot predicted by models that never saw its game
    for v in api.VERSIONS:
        cf = np.full(shots.height, np.nan)
        for k in sorted(shots["fold"].drop_nulls().unique().to_list()):
            other = shots.filter(pl.col("fold").is_not_null() & (pl.col("fold") != k))
            m = api.fit(other, other["game_id"].unique().to_list(), v)
            idx = (shots["fold"] == k).fill_null(False).to_numpy()
            cf[idx] = api.predict(shots.filter(pl.col("fold") == k), m)
        shots = shots.with_columns(pl.Series(f"xfg_{v}_cf", cf))

    cal = {"fit_seconds_per_model_cpu": round(fit_seconds, 2), "frozen": {}, "cross_fit": {}}
    tr = shots.filter(pl.col("release_trusted"))
    for v in api.VERSIONS:
        cal["frozen"][v] = {
            s: evaluate(d["made"].to_numpy().astype(float), d[f"xfg_{v}"].to_numpy())
            for s in ("val", "test")
            if (d := tr.filter(pl.col("split") == s)).height
        }
        cal["cross_fit"][v] = {
            s: evaluate(d["made"].to_numpy().astype(float), d[f"xfg_{v}_cf"].to_numpy())
            for s in ("train", "val", "test")
            if (d := tr.filter((pl.col("split") == s) & pl.col(f"xfg_{v}_cf").is_not_nan())).height
        }
    (tmp / "calibration.json").write_text(json.dumps(cal, indent=2))
    sample = sorted(shots["game_id"].unique().to_list())[::20]
    cons = consistency(shots, cfg, sample)
    (tmp / "consistency.json").write_text(json.dumps(cons, indent=2))
    if cons["max_abs_numeric_diff"] > 1e-9 or any(cons["mismatches"].values()):
        raise SystemExit(f"interface and shot table disagree: {cons}")

    keep = (
        ID_COLS
        + api.FEATURE_COLUMNS
        + ["release_trusted", "in_training", "split", "fold", "parity"]
    )
    keep += [f"xfg_{v}{s}" for v in api.VERSIONS for s in ("", "_cf")]
    shots.select([c for c in keep if c in shots.columns]).write_parquet(
        tmp / "shots_xfg.parquet", compression="zstd"
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    (tmp / "README.md").write_text(
        README.format(version=args.version, data=cfg.version, commit=commit, **cal_summary(cal)),
        encoding="utf-8",
    )
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
        | (
            {"rows": pl.scan_parquet(p).select(pl.len()).collect().item()}
            if p.suffix == ".parquet"
            else {}
        )
        for p in sorted(tmp.iterdir())
    }
    manifest = {
        "product": PRODUCT,
        "version": args.version,
        "producer": "ghost-defense",
        "code_commit": commit,
        "nbacore_data_version": cfg.version,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }
    (tmp / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(json.dumps({"out": str(out), "consistency": cons, "fit_s": cal["fit_seconds_per_model_cpu"],
                      "test": {v: round(cal["frozen"][v]["test"]["log_loss"], 4) for v in api.VERSIONS}}, indent=1))  # fmt: skip


def cal_summary(cal: dict) -> dict:
    f = cal["frozen"]
    return {
        "m2_test": f"log loss {f['M2']['test']['log_loss']:.4f}, Brier {f['M2']['test']['brier']:.4f}, AUC {f['M2']['test']['auc']:.3f}",
        "m3_test": f"log loss {f['M3']['test']['log_loss']:.4f}, Brier {f['M3']['test']['brier']:.4f}, AUC {f['M3']['test']['auc']:.3f}",
        "fit_s": cal["fit_seconds_per_model_cpu"],
    }  # fmt: skip


README = """# ghost_xfg {version} — expected field-goal percentage (ghost-defense)

Producer: ghost-defense (code commit `{commit}`); input: nbacore data {data}.
Make probability of a field-goal attempt given the situation at the release. Two versions:

- **M2** (league level): zone, distance, distance², angle, log nearest-defender distance, log
  front-defender distance (with three-point and rim interactions), dribble bins (unknown, 1,
  2–4, 5+), catch-and-shoot, transition.
- **M3** = M2 + shooter skill: an empirical-Bayes shrunken shooter-by-zone make rate from the
  training shots (offset = logit(shrunk rate) − logit(zone rate)); shooters without training
  shots get offset 0.

L2-penalised logistic regression (C = 1, intercept unpenalised), fitted with Newton's method.

## Training data (no leakage into nbacore `test`)

Frozen models (`M2.json`, `M3.json`) use only the nbacore `train` games and only shots whose
release was located from the ball with the shooter in hand (`release_method` rim / apex). Test
(126 games from 2016-01-04) and val games are never used. Test metrics: M2 {m2_test};
M3 {m3_test}.

## Files

| file | content |
|---|---|
| `shots_xfg.parquet` | one row per nbacore `shot_release` (key `event_uid`, plus `game_id`, `pbp_event_num`): features, `release_trusted` (rim / apex and a defender distance), `in_training`, `split` / `fold` / `parity`, `xfg_M2` / `xfg_M3` (frozen models) and `xfg_M2_cf` / `xfg_M3_cf` (cross-fitted over the nbacore folds: each shot predicted by models fitted without its game's fold) |
| `M2.json`, `M3.json` | coefficients (`feature_names`, `coef`, `intercept`), feature groups, prior hyper-parameters |
| `M3_shooter_prior.parquet` | train counts `n`, `m` per (`shooter_id`, `zone`) |
| `calibration.json` | log loss, Brier, AUC, decile calibration: frozen (val, test) and cross-fitted (train, val, test) |
| `consistency.json` | the interface recomputed at the real releases of 1 in 20 games equals the table |

Predictions are given for every shot with features, including untrusted release methods
(`release_trusted = false`); consumers decide whether to use them.

## Features at any time (`ghost.xfg.api`, same code as the table)

`features(game_id, rows)` with `rows` = (`unix_ms`, `shooter_id`): the time is snapped to the
nearest 25 Hz frame within 40 ms; geometry after mapping the offence to the left basket
(`x`, `y`, `dist_ft`, `angle_deg`, `is_three`, `def1_ft`, `def2_ft`, `def_front_ft` = nearest
opponent within 45° of the shooter→hoop direction, 30 ft if none); the shooter's current
`possession_touch` (start within 15 s before t, end ≥ t − 1 s): `n_dribbles` = its dribbles
started by t, `touch_s` = t − touch start (null without a touch); `catch_and_shoot` = no dribble
so far and touch_s < 2 s; `is_transition` from the ledger possession of the shooter's team that
contains t (False if none; in the table, False for the 475 shots whose possession is not in the ledger); `zone` = rim (< 4 ft) / paint (< 14 ft) / mid / corner3 / above3.
`predict(features, api.load(dir, "M3"))`; `fit(shots, game_ids, "M3")` refits in about
{fit_s} s on a laptop CPU (parity halves, bootstrap by game).
"""


if __name__ == "__main__":
    main()
