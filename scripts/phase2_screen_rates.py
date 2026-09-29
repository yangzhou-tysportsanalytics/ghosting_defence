"""Switch rates per physical screen (Phase 2; denominator = grouped screens).

Groups are nbacore v1.3 ``screen_group_uid`` (same screener, contact within 1 s).
``group_screens`` keeps the earlier interim rule for older releases.
A group is on-ball if any of its candidates is on-ball.

Reads events/switches.parquet (from phase2_screen_events.py) for the outcome of each candidate.
Reported precision caveat (nbacore manual review): real screens are 74 % of on-ball and 33 % of
off-ball candidates, so rates are per *candidate group*, not per verified screen.

Usage:
    uv run python scripts/phase2_screen_rates.py [--game-set small]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ghost import data as D

GAP_MS = 1000


def group_screens(sc: pl.DataFrame) -> pl.DataFrame:
    s = sc.sort(["game_id", "screener_id", "t_contact_ms"])
    new = (
        (pl.col("screener_id") != pl.col("screener_id").shift(1))
        | (pl.col("game_id") != pl.col("game_id").shift(1))
        | ((pl.col("t_contact_ms") - pl.col("t_contact_ms").shift(1)) > GAP_MS)
    ).fill_null(True)
    return s.with_columns(new.cast(pl.Int64).cum_sum().alias("screen_group"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-set", default=None)
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set=args.game_set)
    base = cfg.processed_dir / cfg.game_set
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    SW = pl.read_parquet(base / "events" / "switches.parquet")
    w = P.select(["game_id", "possession_id", "period", "t_start", "t_end"])
    parts = []
    for gid in sorted(P["game_id"].unique().to_list()):
        sc = D.events(gid, "screen_candidate", cfg).select(
            [
                "game_id",
                "period",
                "event_uid",
                "screener_id",
                "t_contact_ms",
                "on_ball",
                "screen_group_uid",
            ]
        )
        j = sc.join(w.filter(pl.col("game_id") == gid), on=["game_id", "period"]).filter(
            (pl.col("t_contact_ms") >= pl.col("t_start") - 1000)
            & (pl.col("t_contact_ms") <= pl.col("t_end"))
        )
        parts.append(j.unique(subset=["event_uid"], keep="first"))
    SC = pl.concat(parts).rename({"screen_group_uid": "screen_group"})  # nbacore v1.3
    outcome = (
        SW.filter(pl.col("screen_uid").is_not_null())
        .group_by("screen_uid")
        .agg(
            (pl.col("event_class") == "screen_switch").any().alias("switch"),
            (pl.col("event_class") == "one_sided").any().alias("one_sided"),
        )
        .rename({"screen_uid": "event_uid"})
    )
    SC = SC.join(outcome, on="event_uid", how="left").with_columns(
        pl.col("switch").fill_null(False), pl.col("one_sided").fill_null(False)
    )
    G = SC.group_by("screen_group").agg(
        pl.col("on_ball").any(),
        pl.col("switch").any(),
        (pl.col("one_sided").any() & ~pl.col("switch").any()).alias("one_sided_only"),
        pl.len().alias("n_candidates"),
    )

    def rates(df: pl.DataFrame) -> dict:
        return {
            "n_groups": df.height,
            "switch_rate": float(df["switch"].mean()) if df.height else None,
            "one_sided_rate": float(df["one_sided_only"].mean()) if df.height else None,
        }

    out = {
        "game_set": cfg.game_set,
        "n_candidates_in_windows": SC.height,
        "n_physical_screen_groups": G.height,
        "candidates_per_group": {
            "mean": float(G["n_candidates"].mean()),
            "median": float(G["n_candidates"].median()),
            "p90": float(G["n_candidates"].quantile(0.9)),
        },
        "on_ball": {
            **rates(G.filter(pl.col("on_ball"))),
            # the reported rate (analysis_team_screen_defense.py): same groups, but only their
            # on-ball candidates count as the switch
            "switch_rate_on_ball_candidates_only": float(
                SC.filter(pl.col("on_ball"))
                .group_by("screen_group")
                .agg(pl.col("switch").any())["switch"]
                .mean()
            ),
            "note": "switch_rate counts a group as switched if any of its candidates switched, "
            "including off-ball candidates grouped with the on-ball one; the reported on-ball "
            "switch rate uses on-ball candidates only (switch_rate_on_ball_candidates_only)",
        },
        "off_ball": rates(G.filter(~pl.col("on_ball"))),
        "groups_per_halfcourt_possession": G.height / P.height,
        "caveat": "rates per candidate group; nbacore-reviewed precision of real screens: "
        "on-ball 74 %, off-ball 33 % (48 % overall after offense_match)",
    }
    rep = Path("reports/phase2") / f"{cfg.version}_{cfg.game_set}"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "screen_rates.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
