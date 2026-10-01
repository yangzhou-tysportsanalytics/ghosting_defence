"""Phase 4 reliability on the rule-ghost deviations (minimal path, CPU).

Per (possession, defender) rows from phase4_rule_ghost.py. For each metric (dev_ft, sag_ft):
1. Context adjustment: OLS on man-to-hoop and man-to-ball distance (+ squares), strong-side share,
   on-ball share, several men guarded, log steps; residuals are the adjusted metric.
2. Definitional decomposition (D-005): team effect := mean residual of the defence team;
   player effect := player mean residual minus his team's mean.
3. Split-half reliability by game parity (nbacore C-009 ``parity``) for players with >= 300
   defensive possessions: Pearson r across players, Spearman-Brown full-sample reliability.
4. Stability curve: r between halves when each player contributes n random possessions to each
   parity half (so 2n in total); pooled across positions and within listed position.
   Within-position: player means centred on their listed-position (G/F/C) mean in each half.
5. Variance shares (method of moments, sampling noise removed): team vs player-within-team.
Decision rule (analysis plan, fixed before any result): split-half r < 0.3 -> methods paper + league-level findings.

Output: reports/phase4/<version>_<game_set>/reliability.json

Usage:
    uv run python scripts/phase4_reliability.py [--game-set all]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D

MIN_POSS = 300
COVS = ["man_to_hoop_ft", "man_to_ball_ft", "share_strong", "share_on_ball"]


# D-021: the context term adds the phase shares of D-020 ("other" is the reference)
PHASE_COVS = ["share_pre_screen", "share_screen", "share_help", "share_closeout", "share_recovery"]


def with_phases(df: pl.DataFrame, base) -> pl.DataFrame:
    """Join the per-(possession, defender) phase shares (phase4_phase_context.py, definitions A)."""
    ph = pl.read_parquet(base / "analysis" / "phase_context_A.parquet").select(
        ["game_id", "possession_id", "def_id", *PHASE_COVS]
    )
    out = df.join(ph, on=["game_id", "possession_id", "def_id"], how="inner")
    if out.height != df.height:
        raise ValueError(f"phase shares missing for {df.height - out.height} rows")
    return out


LEARNED_METRICS = ("nll_ind", "nll_team", "d_mean", "d_mode")


def load_dev(base, ghost: str = "rule") -> tuple[pl.DataFrame, tuple[str, ...], str]:
    """Per-(possession, defender) deviations. ``ghost`` "rule" (default): rule-ghost dev and sag;
    "learned:<name>": the learned-ghost metrics of phase4_learned_ghost_dev.py joined onto the
    same rows (same covariates, parity and teams), plus dev and sag for comparison.
    Returns (table, metrics, output-name suffix)."""
    df = pl.read_parquet(base / "analysis" / "rule_ghost_dev.parquet")
    if ghost == "rule":
        return df, ("dev_ft", "sag_ft"), ""
    name = ghost.split(":", 1)[1]
    lg = pl.read_parquet(base / "analysis" / f"learned_ghost_dev_{name}.parquet")
    df = df.join(
        lg.select(["game_id", "possession_id", "def_id", *LEARNED_METRICS]),
        on=["game_id", "possession_id", "def_id"],
        how="inner",
    )
    return df, (*LEARNED_METRICS, "dev_ft", "sag_ft"), f"_learned_{name}"


def adjust(df: pl.DataFrame, y: str, extra: tuple[str, ...] = ()) -> np.ndarray:
    X = np.column_stack(
        [np.ones(df.height)]
        + [df[c].to_numpy() for c in COVS]
        + [df["man_to_hoop_ft"].to_numpy() ** 2, df["man_to_ball_ft"].to_numpy() ** 2]
        + [(df["n_men"].to_numpy() > 1).astype(float), np.log(df["n_steps"].to_numpy())]
        + [df[c].to_numpy() for c in extra]
    )
    yy = df[y].to_numpy()
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    r2 = 1 - ((yy - X @ beta) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    return yy - X @ beta, float(r2)


def split_half(df: pl.DataFrame, col: str, players: list[int]) -> tuple[float, int]:
    g = (
        df.filter(pl.col("def_id").is_in(players))
        .group_by(["def_id", "parity"])
        .agg(pl.col(col).mean())
        .pivot(on="parity", index="def_id", values=col)
        .drop_nulls()
    )
    a, b = g["0"].to_numpy(), g["1"].to_numpy()
    return float(np.corrcoef(a, b)[0, 1]), g.height


def _center(v, groups: list[str]) -> np.ndarray:
    """Subtract the group mean (listed position) from each player's value."""
    v = np.asarray(v, dtype=float)
    g = np.asarray(groups)
    out = v.copy()
    for k in np.unique(g):
        out[g == k] -= v[g == k].mean()
    return out


def stability_curve(
    df: pl.DataFrame, col: str, ns, reps: int, rng, pos: dict[int, str] | None = None
) -> dict:
    """Split-half r when each player contributes n random possessions to EACH parity half.

    With ``pos``, player means are centred on their listed-position mean within each half
    (within-position reliability); players without a listed position are dropped.
    """
    out = {}
    by = {(p, h): sub[col].to_numpy() for (p, h), sub in df.group_by(["def_id", "parity"])}
    players = sorted({p for p, _ in by})
    if pos is not None:
        players = [p for p in players if p in pos]
    for n in ns:
        elig = [p for p in players if len(by.get((p, 0), [])) >= n and len(by.get((p, 1), [])) >= n]
        if len(elig) < 20:
            continue
        rs = []
        for _ in range(reps):
            a = [rng.choice(by[(p, 0)], n, replace=False).mean() for p in elig]
            b = [rng.choice(by[(p, 1)], n, replace=False).mean() for p in elig]
            if pos is not None:
                g = [pos[p] for p in elig]
                a, b = _center(a, g), _center(b, g)
            rs.append(np.corrcoef(a, b)[0, 1])
        out[int(n)] = {"r": float(np.mean(rs)), "n_players": len(elig)}
    return out


def listed_positions(cfg: D.DataConfig) -> dict[int, str]:
    """First letter of the roster position (G, F, C; 'G-F' -> G), first listing per player."""
    ros = D.rosters(cfg).filter(
        pl.col("position").is_not_null() & (pl.col("position").str.len_chars() > 0)
    )
    p = ros.group_by("player_id").agg(pl.col("position").first().str.slice(0, 1).alias("pos"))
    return dict(zip(p["player_id"].cast(pl.Int64).to_list(), p["pos"].to_list(), strict=True))


def within_position(df: pl.DataFrame, col: str, players: list[int], pos: dict[int, str]) -> dict:
    """Split-half r of player means centred on their listed-position mean in each half."""
    g = (
        df.filter(pl.col("def_id").is_in(players))
        .group_by(["def_id", "parity"])
        .agg(pl.col(col).mean())
        .pivot(on="parity", index="def_id", values=col)
        .drop_nulls()
        .filter(pl.col("def_id").is_in(list(pos)))
    )
    grp = [pos[p] for p in g["def_id"].to_list()]
    a, b = _center(g["0"].to_numpy(), grp), _center(g["1"].to_numpy(), grp)
    by_pos = {}
    for k in sorted(set(grp)):
        m = np.asarray(grp) == k
        by_pos[k] = {"r": float(np.corrcoef(a[m], b[m])[0, 1]), "n_players": int(m.sum())}
    full = (g["0"].to_numpy() + g["1"].to_numpy()) / 2
    return {
        "split_half_r_within_position": float(np.corrcoef(a, b)[0, 1]),
        "n_players": g.height,
        "share_player_variance_explained_by_position": float(
            1 - _center(full, grp).var() / full.var()
        ),
        "by_position": by_pos,
    }


def variance_shares(df: pl.DataFrame, col: str, players: list[int]) -> dict:
    d = df.filter(pl.col("def_id").is_in(players))
    team = d.group_by("defense_team_id").agg(
        pl.col(col).mean().alias("m"), pl.col(col).var().alias("v"), pl.len().alias("n")
    )
    var_team = float(team["m"].var() - (team["v"] / team["n"]).mean())
    d = d.join(team.select(["defense_team_id", "m"]), on="defense_team_id")
    d = d.with_columns((pl.col(col) - pl.col("m")).alias("w"))
    pl_ = d.group_by("def_id").agg(
        pl.col("w").mean().alias("pm"), pl.col("w").var().alias("pv"), pl.len().alias("pn")
    )
    var_player = float(pl_["pm"].var() - (pl_["pv"] / pl_["pn"]).mean())
    total = float(df[col].var())
    return {
        "var_team_effect": var_team,
        "var_player_within_team": var_player,
        "var_total_possession_level": total,
        "share_team_of_systematic": var_team / (var_team + var_player)
        if var_team + var_player > 0
        else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-set", default="all")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--context", default="base", choices=["base", "phases"], help="D-021")
    ap.add_argument("--ghost", default="rule", help='"rule" or "learned:<name>"')
    args = ap.parse_args()
    extra = tuple(PHASE_COVS) if args.context == "phases" else ()
    cfg = D.DataConfig.load(game_set=args.game_set)
    base = cfg.processed_dir / cfg.game_set
    df, metrics, gsfx = load_dev(base, args.ghost)
    suffix = gsfx + ("" if args.context == "base" else "_phases")
    if extra:
        df = with_phases(df, base)
    rng = np.random.default_rng(9)
    counts = df.group_by("def_id").len()
    players = counts.filter(pl.col("len") >= MIN_POSS)["def_id"].to_list()
    pos = listed_positions(cfg)
    rep = {
        "game_set": cfg.game_set,
        "n_rows_possession_defender": df.height,
        "n_players_total": counts.height,
        "n_players_ge_300": len(players),
        "metrics": {},
    }
    for y in metrics:
        res, r2 = adjust(df, y, extra)
        d = df.with_columns(pl.Series("adj", res))
        r_raw, n_raw = split_half(d, y, players)
        r_adj, n_adj = split_half(d, "adj", players)
        # within-team player effect: subtract the team mean of the possession's defence team
        tm = d.group_by("defense_team_id").agg(pl.col("adj").mean().alias("tm"))
        d = d.join(tm, on="defense_team_id").with_columns(
            (pl.col("adj") - pl.col("tm")).alias("adj_w")
        )
        r_w, _ = split_half(d, "adj_w", players)
        rt = (
            d.group_by(["defense_team_id", "parity"])
            .agg(pl.col("adj").mean())
            .pivot(on="parity", index="defense_team_id", values="adj")
            .drop_nulls()
        )
        r_team = float(np.corrcoef(rt["0"], rt["1"])[0, 1])
        rep["metrics"][y] = {
            "context_r2": r2,
            "split_half_r_raw": r_raw,
            "split_half_r_adjusted": r_adj,
            "spearman_brown_adjusted": 2 * r_adj / (1 + r_adj),
            "split_half_r_player_within_team": r_w,
            "spearman_brown_player_within_team": 2 * r_w / (1 + r_w),
            "split_half_r_team_effect": r_team,
            "n_players": n_adj,
            "stability_curve_adjusted": stability_curve(
                d, "adj", [25, 50, 100, 150, 200, 300], args.reps, rng
            ),
            "within_position": within_position(d, "adj", players, pos),
            "stability_curve_within_position": stability_curve(
                d, "adj", [25, 50, 100, 150, 200, 300], args.reps, rng, pos
            ),
            "variance_shares_adjusted": variance_shares(d, "adj", players),
            "decision_rule": "methods+league findings"
            if r_w < 0.3
            else "player effects reportable",
        }
        print(
            y,
            json.dumps(
                {k: v for k, v in rep["metrics"][y].items() if not k.startswith("stability_curve")},
                indent=1,
            ),
        )
        print("  stability:", rep["metrics"][y]["stability_curve_adjusted"])
        print("  stability within position:", rep["metrics"][y]["stability_curve_within_position"])
    out = Path("reports/phase4") / f"{cfg.version}_{cfg.game_set}"
    out.mkdir(parents=True, exist_ok=True)
    rep["context"] = args.context
    rep["ghost"] = args.ghost
    (out / f"reliability{suffix}.json").write_text(json.dumps(rep, indent=2))
    # summary cited by the abstract (previously written by hand; now generated here)
    (out / f"reliability_within_position{suffix}.json").write_text(
        json.dumps({y: rep["metrics"][y]["within_position"] for y in rep["metrics"]}, indent=2)
    )


if __name__ == "__main__":
    main()
