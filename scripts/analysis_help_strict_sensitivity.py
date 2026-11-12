"""Section 4 with stricter help events (quick-check consensus; D-036).

The quick check found that 12 of 48 detected help events (early, late, corner) were judged help
(full or stunt) by the two analysts' consensus. Post hoc, confirmed events had the defender
farther from his man and closer to the ball handler. This refits the Section 4 panel with only
help events that look more like help, to see whether the within-team early-help association
depends on the loose events:

  base         all help-B events (the paper's definition; reproduces help_threshold.json)
  near_handler min distance to the handler during the event <= 8 ft
  far_from_man max distance from his man during the event >= 12 ft
  strict       both

For each variant, the first third-defender help after each on-ball screen is recomputed exactly
as in analysis_team_screen_defense.py (onset within [contact, contact + 3 s], neither screen
defender), then early (<= 1 s) and late (1-3 s) rates per team-game, and the fits of
analysis_help_threshold.py: single early-help model and the joint early / late model, with
defence-team FE and with defence-team + opponent FE, CR1 and two-way intervals; also without
fourth-quarter garbage time. Thresholds were read off the same 48 clips, so the in-sample
precision of each variant on those clips is descriptive only.

Output: reports/analysis/help_strict_sensitivity.json

Usage:
    uv run python scripts/analysis_help_strict_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_help_threshold as HT  # noqa: E402
from ghost import data as D  # noqa: E402

HELP_WINDOW_MS = 3000
VARIANTS = {
    "base": pl.lit(True),
    "near_handler": pl.col("min_dist_to_handler_ft") <= 8.0,
    "far_from_man": pl.col("max_dist_from_man_ft") >= 12.0,
    "strict": (pl.col("min_dist_to_handler_ft") <= 8.0) & (pl.col("max_dist_from_man_ft") >= 12.0),
}


def first_help(S: pl.DataFrame, H: pl.DataFrame) -> pl.DataFrame:
    """help_delay_s of the first third-defender help onset within 3 s of contact (or null)."""
    j = (S.select(["screen_group", "game_id", "possession_id", "t_contact_ms", "screened_def_id",
                   "screener_def_id"])
         .join(H.select(["game_id", "possession_id", "def_id", "t_start_ms"]),
               on=["game_id", "possession_id"], how="inner")
         .with_columns(((pl.col("t_start_ms") - pl.col("t_contact_ms")) / 1000).alias("delay_s"))
         .filter(pl.col("delay_s").is_between(0, HELP_WINDOW_MS / 1000)
                 & (pl.col("def_id") != pl.col("screened_def_id"))
                 & (pl.col("def_id") != pl.col("screener_def_id"))))  # fmt: skip
    f = j.group_by("screen_group").agg(pl.col("delay_s").min().alias("help_delay_s"))
    return S.drop("help_delay_s").join(f, on="screen_group", how="left")


def consensus_precision(H: pl.DataFrame) -> dict:
    """Share confirmed (full or stunt) among the quick-check help clips kept by each variant."""
    root = Path("data/annotations")
    try:
        al = json.loads((root / "quick" / "gdq_v1" / "allocation.json").read_text())["items"]
        ans = {c: json.loads((root / "raw" / f"{c}_gdq_v1_answers.json").read_text(
            encoding="utf-8"))["answers"] for c in ("A1", "A2", "ADJ")}  # fmt: skip
    except FileNotFoundError:
        return {}
    rows = []
    for i, a in al.items():
        if a["task"] != "E" or a["category"] not in ("help_early", "help_late", "help_corner"):
            continue
        x, y = ans["A1"][i]["a"], ans["A2"][i]["a"]
        c = x if x == y else ans["ADJ"][i]["a"]
        if c == "cant":
            continue
        h = (H.filter((pl.col("game_id") == a["game_id"])
                      & (pl.col("possession_id") == a["possession_id"])
                      & (pl.col("def_id") == a["defs"][0]))
             .with_columns((pl.col("t_start_ms") - a["t_ms"]).abs().alias("d")).sort("d")
             .row(0, named=True))  # fmt: skip
        rows.append(h | {"yes": c in ("full", "stunt")})
    E = pl.DataFrame(rows)
    out = {}
    for name, cond in VARIANTS.items():
        k = E.filter(cond)
        out[name] = {"kept": k.height, "confirmed": int(k["yes"].sum()), "of": E.height}
    return out


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    S0 = HT.screens(cfg)
    H0 = pl.read_parquet(base / "events" / "help.parquet").filter(pl.col("definition") == "B")
    pts = HT.points(cfg, drop_garbage=False)
    pts_ng = HT.points(cfg, drop_garbage=True)
    rep: dict = {"units": "points per 100 possessions per 10 percentage points of the rate",
                 "variants": {"base": "all help-B", "near_handler": "min dist to handler <= 8 ft",
                              "far_from_man": "max dist from man >= 12 ft",
                              "strict": "both"},
                 "quick_check_precision_in_sample": consensus_precision(H0), "results": {}}  # fmt: skip
    for name, cond in VARIANTS.items():
        H = H0.filter(cond)
        S = first_help(S0, H)
        d = S["help_delay_s"].fill_null(99)
        rep["results"][name] = {
            "help_events": H.height,
            "share_screens_early": float((d <= 1.0).mean()),
            "share_screens_late": float(((d > 1.0) & (d <= 3.0)).mean()),
            "main": HT.curve(S, pts, [1.0])["1.0"],
            "no_garbage_time": HT.curve(S.filter(~pl.col("garbage")), pts_ng, [1.0])["1.0"],
        }
    out = Path("reports/analysis/help_strict_sensitivity.json")
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["quick_check_precision_in_sample"]))
    for name, r in rep["results"].items():
        for part in ("main", "no_garbage_time"):
            a, b = r[part]["team_fe"], r[part]["team_and_opponent_fe"]
            print(f"{name:13s} {part:15s} early {r['share_screens_early']:.3f} "
                  f"FE {a['single']['early']['coef']:+.2f} {[round(v, 2) for v in a['single']['early']['ci95_twoway']]} "
                  f"FE+opp {b['single']['early']['coef']:+.2f} {[round(v, 2) for v in b['single']['early']['ci95_twoway']]} "
                  f"| joint diff FE {a['joint']['difference']['coef']:+.2f} "
                  f"{[round(v, 2) for v in a['joint']['difference']['ci95_twoway']]} "
                  f"FE+opp {b['joint']['difference']['coef']:+.2f} "
                  f"{[round(v, 2) for v in b['joint']['difference']['ci95_twoway']]}")  # fmt: skip


if __name__ == "__main__":
    main()
