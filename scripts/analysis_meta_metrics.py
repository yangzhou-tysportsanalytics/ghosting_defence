"""Reliability in the meta-analytics framework of Franks, D'Amour, Cervone & Bornn (2016) (A2).

Player-level metrics (players with >= 300 defensive half-court possessions), main specification:
* sag_ft, dev_ft: context-adjusted per-possession values (phase4_reliability.adjust with the
  D-021 phase shares), player value = mean; also the within-team version (minus the defence
  team's mean), which is the player effect of the definitional decomposition (D-005);
* help_b: help onsets (definition B) per defensive possession, raw.
Sampling variance of a player's value: bootstrap over his games (B = 200).

Meta-metrics (one half-season, so "seasons" are replaced by halves):
* discrimination D = 1 - E_p[tau2_p] / Var_p(X_p): share of between-player variance that is not
  sampling noise;
* stability S = 1 - E_p[Var_h(X_ph) - tau2_ph] / (Var(X_ph) - E[tau2_ph]) over two halves:
  game parity (a random split: S near 1 means the noise model is right) and date (games before /
  on or after the median date: change over the half-season);
* independence I(m | M) = (Var(X_m | M) - E[tau2]) / (Var(X_m) - E[tau2]): share of signal
  variance not explained by listed position, + height (player_bio), + the mean shooting threat
  of the men guarded (shrunk 3P%, 3PA rate; guarded_skill.parquet); linear, noise-corrected.
External metrics (RAPTOR defense, DBPM): correlations only (Pearson, Spearman, and Pearson
disattenuated by D), no redistribution of the external values.

Output: reports/analysis/meta_metrics[_learned_<name>].json

Usage:
    uv run python scripts/analysis_meta_metrics.py [--ghost learned:<name>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ghost import data as D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_reliability import (  # noqa: E402
    CONTEXT_SUFFIX,
    CONTEXTS,
    MIN_POSS,
    adjust,
    listed_positions,
    load_dev,
    with_phases,
)

B = 200


def boot_var(v: np.ndarray, games: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """(mean, bootstrap variance of the mean) resampling games."""
    ug, inv = np.unique(games, return_inverse=True)
    s, c = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    idx = rng.integers(0, len(ug), size=(B, len(ug)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(means.var(ddof=1))


def player_table(d: pl.DataFrame, col: str, by: str | None, rng) -> pl.DataFrame:
    """Per player (and half ``by``): mean and its bootstrap variance."""
    keys = ["def_id"] + ([by] if by else [])
    rows = []
    for k, g in d.group_by(keys, maintain_order=True):
        m, t2 = boot_var(g[col].to_numpy(), g["game_id"].to_numpy(), rng)
        rows.append({**dict(zip(keys, k, strict=True)), "x": m, "tau2": t2, "n": g.height})
    return pl.DataFrame(rows)


def discrimination(pt: pl.DataFrame) -> float:
    return float(1 - pt["tau2"].mean() / pt["x"].var())


def stability(ph: pl.DataFrame, by: str) -> dict:
    w = ph.pivot(on=by, index="def_id", values=["x", "tau2"]).drop_nulls()
    xs = [c for c in w.columns if c.startswith("x_")]
    ts = [c for c in w.columns if c.startswith("tau2_")]
    a, b = w[xs[0]].to_numpy(), w[xs[1]].to_numpy()
    tau = (w[ts[0]].to_numpy() + w[ts[1]].to_numpy()) / 2
    within = (a - b) ** 2 / 2  # variance of two values (ddof 1)
    total = np.concatenate([a, b]).var(ddof=1)
    return {"stability": float(1 - (within - tau).mean() / (total - tau.mean())),
            "r_between_halves": float(np.corrcoef(a, b)[0, 1]), "n_players": w.height}  # fmt: skip


def independence(pt: pl.DataFrame, covs: list[str]) -> dict:
    d = pt.drop_nulls(covs)
    y = d["x"].to_numpy()
    X = np.column_stack([np.ones(d.height)] + [d[c].to_numpy().astype(float) for c in covs])
    r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    k = np.linalg.matrix_rank(X)
    var_res = float((r**2).sum() / (d.height - k))  # unbiased residual variance
    tau = float(d["tau2"].mean())
    return {"independence": (var_res - tau) / (float(y.var(ddof=1)) - tau),
            "r2_raw": float(1 - (r**2).sum() / ((y - y.mean()) ** 2).sum()),
            "n_players": d.height}  # fmt: skip


def external(pt: pl.DataFrame, ext: pl.DataFrame, disc: float) -> dict:
    out = {}
    for c in ("raptor_defense", "dbpm"):
        d = pt.join(ext.select(["def_id", c]), on="def_id", how="inner").drop_nulls(c)
        r = float(np.corrcoef(d["x"], d[c])[0, 1])
        out[c] = {"pearson": r, "spearman": float(stats.spearmanr(d["x"], d[c])[0]),
                  "pearson_disattenuated_by_D": r / math.sqrt(disc) if disc > 0 else None,
                  "n_players": d.height}  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghost", default="rule", help='"rule" or "learned:<name>"')
    ap.add_argument("--context", default="phases", choices=["phases", "offense_phases"],
                    help="phases = main specification (D-021); offense_phases = review A6")  # fmt: skip
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    df, metrics, sfx = load_dev(base, args.ghost)
    df = with_phases(df, base)
    poss = pl.read_parquet(base / "possessions.parquet").select(["game_id", "game_date"]).unique()
    df = df.join(poss, on="game_id", how="left")
    dates = df.select("game_id", "game_date").unique()["game_date"].sort()
    cutoff = dates[len(dates) // 2]  # as in phase4_context_time_robustness.py
    help_b = (
        pl.read_parquet(base / "events" / "help.parquet")
        .filter(pl.col("definition") == "B")
        .group_by(["game_id", "possession_id", "def_id"])
        .len()
        .rename({"len": "help_b"})
    )
    df = df.join(help_b, on=["game_id", "possession_id", "def_id"], how="left").with_columns(
        pl.col("help_b").fill_null(0).cast(pl.Float64),
        (pl.col("game_date") >= cutoff).cast(pl.Int8).alias("date_half"),
    )
    players = df.group_by("def_id").len().filter(pl.col("len") >= MIN_POSS)["def_id"].to_list()
    df = df.filter(pl.col("def_id").is_in(players))

    pos = listed_positions(cfg)
    bio = D.L.player_bio(cfg.version).select(
        pl.col("player_id").cast(pl.Int64).alias("def_id"), "height_cm"
    )
    gs = (
        pl.read_parquet(base / "analysis" / "guarded_skill.parquet")
        .group_by("def_id")
        .agg(pl.col("g_three_pct").mean(), pl.col("g_three_rate").mean())
    )
    pos_df = pl.DataFrame({"def_id": list(pos), "pos": list(pos.values())}).with_columns(
        pl.col("def_id").cast(pl.Int64),
        (pl.col("pos") == "G").cast(pl.Float64).alias("pos_G"),
        (pl.col("pos") == "C").cast(pl.Float64).alias("pos_C"),
    )
    adv = D.L.advanced(cfg.version).sort("is_total", descending=True)
    adv = adv.unique(subset=["player_id"], keep="first")
    ext = adv.select(pl.col("player_id").cast(pl.Int64).alias("def_id"), "dbpm").join(
        D.L.raptor(cfg.version).select(
            pl.col("player_id").cast(pl.Int64).alias("def_id"), "raptor_defense"
        ),
        on="def_id",
        how="full",
        coalesce=True,
    )
    cov_sets = {
        "position": ["pos_G", "pos_C"],
        "position_height": ["pos_G", "pos_C", "height_cm"],
        "position_height_guarded_threat": ["pos_G", "pos_C", "height_cm", "g_three_pct",
                                           "g_three_rate"],
    }  # fmt: skip

    rng = np.random.default_rng(9)
    rep: dict = {"ghost": args.ghost, "n_players": len(players), "bootstrap": f"games, B={B}",
                 "date_cutoff": str(cutoff), "context": args.context,
                 "metrics": {}}  # fmt: skip
    versions = []
    for y in metrics:
        res, _ = adjust(df, y, CONTEXTS[args.context])
        d = df.with_columns(pl.Series("adj", res))
        tm = d.group_by("defense_team_id").agg(pl.col("adj").mean().alias("tm"))
        d = d.join(tm, on="defense_team_id").with_columns((pl.col("adj") - pl.col("tm")).alias("w"))
        versions += [(f"{y}_adjusted", d, "adj"), (f"{y}_within_team", d, "w")]
    versions.append(("help_b_rate", df, "help_b"))
    for name, d, col in versions:
        pt = player_table(d, col, None, rng)
        disc = discrimination(pt)
        pt = pt.join(pos_df, on="def_id", how="left").join(bio, on="def_id", how="left")
        pt = pt.join(gs, on="def_id", how="left")
        rep["metrics"][name] = {
            "discrimination": disc,
            "stability_parity": stability(player_table(d, col, "parity", rng), "parity"),
            "stability_date": stability(player_table(d, col, "date_half", rng), "date_half"),
            "independence": {k: independence(pt, v) for k, v in cov_sets.items()},
            "external_correlations": external(pt, ext, disc),
        }
        m = rep["metrics"][name]
        print(f"{name}: D {disc:.3f}  S_parity {m['stability_parity']['stability']:.3f}  "
              f"S_date {m['stability_date']['stability']:.3f}  I(pos,height,threat) "
              f"{m['independence']['position_height_guarded_threat']['independence']:.3f}",
              flush=True)  # fmt: skip
    rep["sources"] = {"basketball_reference": "local research use only; do not redistribute",
                      "raptor": "FiveThirtyEight, CC BY 4.0 (attribution required)"}  # fmt: skip
    sfx += "" if args.context == "phases" else CONTEXT_SUFFIX[args.context]
    p = Path("reports/analysis") / f"meta_metrics{sfx}.json"
    p.write_text(json.dumps(rep, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
