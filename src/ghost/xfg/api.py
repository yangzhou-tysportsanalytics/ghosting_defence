"""xFG interface for other projects (frozen product ``ghost_xfg``).

- ``features(game_id, rows, cfg)``: xFG features at arbitrary tracking times for given
  would-be shooters ("if he shot now"), from nbacore L1 frames and L2 touches / dribbles; the
  same code (``ghost.xfg.features``) builds the shot table, so at real releases the two agree.
- ``predict(feats, model)``: make probability from a fitted or frozen model (M2 = league level,
  M3 = + shrunken shooter-by-zone offset; unseen shooters get offset 0).
- ``fit(shots, game_ids, version)``: refit on the trusted shots of the given games (~1 s).
- ``save`` / ``load``: frozen models as JSON (+ shooter prior counts as parquet).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.xfg.features import catch_and_shoot, geometry, touch_state, zone_of
from ghost.xfg.model import FEATURE_SETS, XFG, Logistic, ShooterPrior, fit_xfg

VERSIONS = {"M2": "M2_plus_dribble", "M3": "M3_plus_shooter"}
TRUSTED_METHODS = ("rim", "apex")  # releases located from the ball with the shooter in hand
SNAP_MS = 40  # a requested time is matched to the nearest 25 Hz frame within this tolerance
FEATURE_COLUMNS = [
    "x", "y", "dist_ft", "angle_deg", "is_three", "def1_ft", "def2_ft", "def_front_ft",
    "n_dribbles", "touch_s", "catch_and_shoot", "is_transition", "zone",
]  # fmt: skip


def trusted(shots: pl.DataFrame) -> pl.DataFrame:
    """Rows the models are trained on: trusted release method and a defender distance."""
    return shots.filter(
        pl.col("release_method").is_in(TRUSTED_METHODS) & pl.col("def1_ft").is_not_null()
    )


def features(game_id: str, rows: pl.DataFrame, cfg: D.DataConfig | None = None) -> pl.DataFrame:
    """Features for ``rows`` (columns ``unix_ms``, ``shooter_id``) of one game.

    Each time is snapped to the nearest 25 Hz frame within 40 ms (``frame_unix_ms``); rows
    without such a frame, or whose shooter is not on court, get null features. The shooter's team
    comes from the game roster, the attack direction from nbacore, ``is_transition`` from the
    ledger possession of the shooter's team that contains the time (False if none).
    """
    cfg = cfg or D.DataConfig.load(game_set="all")
    gid = str(game_id).zfill(10)
    req = rows.select(pl.col("unix_ms").cast(pl.Int64), pl.col("shooter_id").cast(pl.Int64))
    fr = D.frames(gid, cfg).select(["period", "unix_ms", "team_id", "player_id", "x", "y"])
    times = fr.select(["unix_ms", "period"]).unique().sort("unix_ms")
    snap = (
        req.with_row_index("_i")
        .sort("unix_ms")
        .join_asof(
            times.rename({"unix_ms": "frame_unix_ms"}),
            left_on="unix_ms",
            right_on="frame_unix_ms",
            strategy="nearest",
            tolerance=SNAP_MS,
        )
        .sort("_i")
    )
    ros = D.rosters(cfg, gid).select(["player_id", "team_id"]).unique("player_id")
    team_of = dict(zip(ros["player_id"].to_list(), ros["team_id"].to_list(), strict=True))
    ad = D.attack_direction(cfg, gid)
    left_of = {
        (t, p): a for t, p, a in zip(ad["team_id"], ad["period"], ad["attacks_left"], strict=True)
    }
    led = D.L.ledger(cfg.version).filter(pl.col("game_id") == gid)
    lo, lt = led["offense_team_id"].to_numpy(), led["is_transition"].fill_null(False).to_numpy()
    ls, le = led["t_start_ms"].to_numpy(), led["t_end_ms"].to_numpy()
    touch = D.events(gid, "possession_touch", cfg, offense_only=False)
    drib = D.events(gid, "dribble", cfg, offense_only=False)
    tk = [touch[c].to_numpy() for c in ("actor_id", "t_start_ms", "t_end_ms", "event_uid")]
    dk = (
        [drib[c].to_numpy() for c in ("touch_uid", "t_start_ms")]
        if drib.height
        else [np.array([], dtype=object), np.array([], dtype=np.int64)]
    )
    by_t = {
        k[0]: g
        for k, g in fr.filter(pl.col("unix_ms").is_in(snap["frame_unix_ms"].drop_nulls()))
        .partition_by("unix_ms", as_dict=True)
        .items()
    }
    out = []
    for r in snap.iter_rows(named=True):
        base = {"game_id": gid, "unix_ms": r["unix_ms"], "shooter_id": r["shooter_id"],
                "frame_unix_ms": r["frame_unix_ms"], "period": r["period"]}  # fmt: skip
        team = team_of.get(r["shooter_id"])
        g = None
        if r["frame_unix_ms"] is not None and team is not None:
            left = left_of.get((team, r["period"]))
            f = by_t[r["frame_unix_ms"]]
            if left is not None:
                g = geometry(
                    f["team_id"].to_numpy(),
                    f["player_id"].to_numpy(),
                    f["x"].to_numpy(),
                    f["y"].to_numpy(),
                    r["shooter_id"],
                    team,
                    bool(left),
                )
        if g is None:
            out.append(base)
            continue
        t = r["frame_unix_ms"]
        n_drib, touch_s = touch_state(*tk, *dk, r["shooter_id"], t)
        inside = np.flatnonzero((lo == team) & (ls <= t) & (le >= t))
        out.append(
            {
                **base,
                **g,
                "n_dribbles": n_drib,
                "touch_s": touch_s,
                "catch_and_shoot": catch_and_shoot(n_drib, touch_s),
                "is_transition": bool(lt[inside[0]]) if inside.size else False,
            }
        )
    df = pl.DataFrame(out, infer_schema_length=None)
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    ok = df["dist_ft"].is_not_null().to_numpy()
    z: list[str | None] = [None] * df.height
    if ok.any():
        zz = zone_of(
            df["dist_ft"].to_numpy()[ok], df["is_three"].to_numpy()[ok].astype(bool),
            df["y"].to_numpy()[ok],
        )  # fmt: skip
        for i, v in zip(np.flatnonzero(ok), zz, strict=True):
            z[i] = str(v)
    return df.with_columns(pl.Series("zone", z, dtype=pl.Utf8))


def fit(shots: pl.DataFrame, game_ids, version: str = "M3") -> XFG:
    """Refit a version on the trusted shots of ``game_ids`` (the shooter prior too)."""
    tr = trusted(shots).filter(pl.col("game_id").is_in(list(game_ids)))
    return fit_xfg(tr, FEATURE_SETS[VERSIONS[version]])


def predict(feats: pl.DataFrame, model: XFG) -> np.ndarray:
    """Make probability; NaN where the features are null (no frame / shooter off court)."""
    ok = feats["dist_ft"].is_not_null() & feats["def1_ft"].is_not_null()
    out = np.full(feats.height, np.nan)
    if ok.any():
        sub = feats.filter(ok)
        if "made" not in sub.columns:  # the shooter prior's leave-one-out needs it; unused here
            sub = sub.with_columns(pl.lit(False).alias("made"))
        out[ok.to_numpy()] = model.predict(sub, leave_one_out=False)
    return out


def save(model: XFG, directory: Path, name: str) -> list[Path]:
    """Write ``<name>.json`` (coefficients, groups, prior hyper-parameters) and, for models with
    a shooter prior, ``<name>_shooter_prior.parquet`` (train counts n, m per shooter and zone)."""
    directory.mkdir(parents=True, exist_ok=True)
    d = {
        "groups": model.groups,
        "feature_names": model.names,
        "intercept": float(model.clf.intercept_[0]),
        "coef": [float(c) for c in model.clf.coef_[0]],
        "C": model.clf.C,
        "prior": model.prior.to_dict() if model.prior else None,
    }
    paths = [directory / f"{name}.json"]
    paths[0].write_text(json.dumps(d, indent=2))
    if model.prior:
        keys = list(model.prior.counts)
        pl.DataFrame(
            {
                "shooter_id": [k[0] for k in keys],
                "zone": [k[1] for k in keys],
                "n": [model.prior.counts[k][0] for k in keys],
                "m": [model.prior.counts[k][1] for k in keys],
            }
        ).sort(["shooter_id", "zone"]).write_parquet(directory / f"{name}_shooter_prior.parquet")
        paths.append(directory / f"{name}_shooter_prior.parquet")
    return paths


def load(directory: Path, name: str) -> XFG:
    d = json.loads((Path(directory) / f"{name}.json").read_text())
    clf = Logistic(C=d["C"])
    clf.intercept_ = np.array([d["intercept"]])
    clf.coef_ = np.array([d["coef"]])
    prior = None
    if d["prior"]:
        c = pl.read_parquet(Path(directory) / f"{name}_shooter_prior.parquet")
        prior = ShooterPrior(
            d["prior"]["p_zone"],
            d["prior"]["k_zone"],
            {
                (int(s), z): (int(n), int(m))
                for s, z, n, m in zip(c["shooter_id"], c["zone"], c["n"], c["m"], strict=True)
            },
        )
    return XFG(d["groups"], clf, d["feature_names"], prior)
