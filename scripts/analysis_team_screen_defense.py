"""League-level finding (Phase 5 task 3a, D-014): how teams defend on-ball screens.

For every on-ball screen group (nbacore physical-screen grouping: same screener, contact <= 1 s)
inside a half-court possession:
  switched      the group's candidates produced a screen_switch (switch A)
  helped_3s     a help-B onset by a defender other than the two screen defenders within
                [contact, contact + 3 s]
  help_delay_s  onset - contact of the first such help
Per defence team: rates with 95 % bootstrap intervals (resampling games, 500 draws).

Caveats written into the output: candidate precision (on-ball 74 %), help B is a geometric
definition not yet validated against annotations.

Outputs: reports/analysis/team_screen_defense.{json,csv,png}

Usage:
    uv run python scripts/analysis_team_screen_defense.py [--game-set all]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ghost import data as D  # noqa: E402

HELP_WINDOW_MS = 3000


def group_screens(sc: pl.DataFrame) -> pl.DataFrame:
    s = sc.sort(["game_id", "screener_id", "t_contact_ms"])
    new = (
        (pl.col("screener_id") != pl.col("screener_id").shift(1))
        | (pl.col("game_id") != pl.col("game_id").shift(1))
        | ((pl.col("t_contact_ms") - pl.col("t_contact_ms").shift(1)) > 1000)
    ).fill_null(True)
    return s.with_columns(new.cast(pl.Int64).cum_sum().alias("screen_group"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-set", default="all")
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set=args.game_set)
    base = cfg.processed_dir / cfg.game_set
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    SW = pl.read_parquet(base / "events" / "switches.parquet")
    H = pl.read_parquet(base / "events" / "help.parquet").filter(pl.col("definition") == "B")
    w = P.select(
        [
            "game_id",
            "possession_id",
            "period",
            "t_start",
            "t_end",
            "defense_team_id",
            "offense_team_id",
            "attacks_left",
            "defense_player_ids",
        ]
    )

    parts = []
    for gid in sorted(P["game_id"].unique().to_list()):
        sc = (
            D.events(gid, "screen_candidate", cfg)
            .filter(pl.col("on_ball"))
            .select(
                [
                    "game_id",
                    "period",
                    "event_uid",
                    "screener_id",
                    "screened_def_id",
                    "screener_def_id",
                    "t_contact_ms",
                    "x",
                    "y",
                    "user_def_id",
                    "screen_group_uid",
                    "screen_confidence",
                ]
            )
        )
        j = sc.join(w.filter(pl.col("game_id") == gid), on=["game_id", "period"]).filter(
            (pl.col("t_contact_ms") >= pl.col("t_start") - 1000)
            & (pl.col("t_contact_ms") <= pl.col("t_end"))
        )
        # a contact can fall in two windows (1 s lead of the next one): prefer the window that
        # contains it, then the earlier window; sort first so the choice is deterministic
        j = j.with_columns((pl.col("t_contact_ms") < pl.col("t_start")).alias("_lead")).sort(
            ["event_uid", "_lead", "t_start"]
        )
        parts.append(
            j.unique(subset=["event_uid"], keep="first", maintain_order=True).drop("_lead")
        )
    SC = (
        pl.concat(parts)
        .with_columns(pl.coalesce("user_def_id", "screened_def_id").alias("screened_def_id"))
        .rename({"screen_group_uid": "screen_group"})
    )  # nbacore v1.3: physical-screen groups and the user's defender
    sw_uids = set(
        SW.filter(pl.col("event_class") == "screen_switch")["screen_uid"].drop_nulls().to_list()
    )
    SC = SC.with_columns(pl.col("event_uid").is_in(list(sw_uids)).alias("switched_cand"))
    # "first" = the group's earliest candidate (deterministic: sorted, order-preserving groups)
    SC = SC.sort(["screen_group", "t_contact_ms", "event_uid"])
    G = SC.group_by("screen_group", maintain_order=True).agg(
        pl.col("game_id").first(),
        pl.col("possession_id").first(),
        pl.col("defense_team_id").first(),
        pl.col("t_contact_ms").min(),
        pl.col("screened_def_id").first(),
        pl.col("screener_def_id").first(),
        pl.col("switched_cand").any().alias("switched"),
        pl.col("offense_team_id").first(),
        pl.col("attacks_left").first(),
        pl.col("x").first(),
        pl.col("y").first(),
        pl.col("screen_confidence").max(),
        pl.col("defense_player_ids").first(),
    )
    # help by a third defender after contact
    Hj = G.join(
        H.select(["game_id", "possession_id", "def_id", "t_start_ms"]),
        on=["game_id", "possession_id"],
        how="left",
    ).with_columns(((pl.col("t_start_ms") - pl.col("t_contact_ms")) / 1000).alias("delay_s"))
    Hj = Hj.filter(
        pl.col("delay_s").is_between(0, HELP_WINDOW_MS / 1000)
        & (pl.col("def_id") != pl.col("screened_def_id"))
        & (pl.col("def_id") != pl.col("screener_def_id"))
    )
    first_help = (
        Hj.sort(["screen_group", "delay_s", "def_id"])
        .group_by("screen_group", maintain_order=True)
        .agg(
            pl.col("delay_s").first().alias("help_delay_s"),
            pl.col("def_id").first().alias("first_helper_id"),
        )
    )
    G = G.join(first_help, on="screen_group", how="left").with_columns(
        pl.col("help_delay_s").is_not_null().alias("helped_3s")
    )
    # per-screen table for the value test (ids, times and screen location only)
    ana = cfg.processed_dir / cfg.game_set / "analysis"
    ana.mkdir(parents=True, exist_ok=True)
    G.write_parquet(ana / "onball_screens.parquet")

    gt = D.games(cfg)
    abbr = {}
    for r in gt.iter_rows(named=True):
        if r["home_abbr"]:
            abbr[r["home_team_id"]] = r["home_abbr"]
        if r["visitor_abbr"]:
            abbr[r["visitor_team_id"]] = r["visitor_abbr"]

    rng = np.random.default_rng(9)
    rows = []
    for tid in sorted(G["defense_team_id"].unique().to_list()):
        Gt = G.filter(pl.col("defense_team_id") == tid)
        gids = sorted(Gt["game_id"].unique().to_list())  # fixed order: reproducible bootstrap
        by_game = {g: Gt.filter(pl.col("game_id") == g) for g in gids}
        sw_rate = float(Gt["switched"].mean())
        help_rate = float(Gt["helped_3s"].mean())
        delay = float(Gt["help_delay_s"].median()) if Gt["helped_3s"].any() else None
        bs_sw, bs_h = [], []
        for _ in range(args.boot):
            pick = rng.choice(len(gids), size=len(gids), replace=True)
            bb = pl.concat([by_game[gids[k]] for k in pick])
            bs_sw.append(bb["switched"].mean())
            bs_h.append(bb["helped_3s"].mean())
        rows.append(
            {
                "team": abbr.get(tid, str(tid)),
                "team_id": tid,
                "n_games": len(gids),
                "n_on_ball_screens": Gt.height,
                "switch_rate": sw_rate,
                "switch_lo": float(np.quantile(bs_sw, 0.025)),
                "switch_hi": float(np.quantile(bs_sw, 0.975)),
                "help_rate_3s": help_rate,
                "help_lo": float(np.quantile(bs_h, 0.025)),
                "help_hi": float(np.quantile(bs_h, 0.975)),
                "help_delay_median_s": delay,
            }
        )
    T = pl.DataFrame(rows).sort("help_rate_3s", descending=True)
    out = Path("reports/analysis")
    out.mkdir(parents=True, exist_ok=True)
    T.write_csv(out / "team_screen_defense.csv")
    league = {
        "n_on_ball_screen_groups": G.height,
        "switch_rate": float(G["switched"].mean()),
        "help_rate_3s": float(G["helped_3s"].mean()),
        "help_delay_median_s": float(G["help_delay_s"].median()),
        "team_help_rate_range": [float(T["help_rate_3s"].min()), float(T["help_rate_3s"].max())],
        "team_switch_rate_range": [float(T["switch_rate"].min()), float(T["switch_rate"].max())],
        "corr_team_switch_vs_help": float(np.corrcoef(T["switch_rate"], T["help_rate_3s"])[0, 1]),
        "caveats": [
            "screen candidates: reviewed precision on-ball 74 % (nbacore)",
            "help B is geometric and not yet validated against human annotation",
            "rates per candidate group (same screener, contact <= 1 s)",
        ],
    }
    (out / "team_screen_defense.json").write_text(
        json.dumps({"league": league, "teams": rows}, indent=2)
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    x, y = T["switch_rate"].to_numpy(), T["help_rate_3s"].to_numpy()
    ax.errorbar(
        x,
        y,
        xerr=[x - T["switch_lo"].to_numpy(), T["switch_hi"].to_numpy() - x],
        yerr=[y - T["help_lo"].to_numpy(), T["help_hi"].to_numpy() - y],
        fmt="none",
        ecolor="0.8",
        lw=0.8,
        zorder=1,
    )
    ax.scatter(x, y, s=18, color="#1f77b4", zorder=2)
    for t, xi, yi in zip(T["team"], x, y, strict=True):
        ax.annotate(
            t,
            (xi, yi),
            fontsize=7,
            xytext=(3, 2),
            textcoords="offset points",
            color="#d62728" if t == "MIA" else "0.2",
        )
    ax.set_xlabel("switch rate on on-ball screens")
    ax.set_ylabel("share of on-ball screens with third-defender help within 3 s")
    ax.set_title(
        "How teams defend on-ball screens, 2015-16 (631 games, 95% game bootstrap)", fontsize=9
    )
    fig.savefig(out / "team_screen_defense.png", dpi=150, bbox_inches="tight")
    print(json.dumps(league, indent=2))
    print(
        T.select(
            ["team", "n_on_ball_screens", "switch_rate", "help_rate_3s", "help_delay_median_s"]
        ).head(8)
    )
    print(
        T.select(
            ["team", "n_on_ball_screens", "switch_rate", "help_rate_3s", "help_delay_median_s"]
        ).tail(5)
    )


if __name__ == "__main__":
    main()
