"""Sample for the quick check of matchups and events (D-036).

Replaces the full possession annotation (gd_v2_core100) with two short tasks, all drawn from
TEST games (the matchup model and the event rules were fitted / set on training games):

M  40 possessions, stratified like sample v1 (pnr_switch 10, pnr_no_switch 10, drive 8,
   ball_movement 6, other 6). Two moments per possession drawn at random (one in each half,
   from step 3), the annotator picks whom each defender guards (or nobody / can't tell),
   without any model output. Pool counts per stratum (test games) are kept for weighting.
E  90 short event clips, each with a highlighted defender (or pair) and a marked window:
     help_early    25  first third-defender help (help B) within 1 s of an on-ball screen
     help_late     15  first third-defender help 1-3 s after an on-ball screen
     help_corner   10  help B leaving a strong-side strict-corner man during a drive episode
                       (analysis_corner_help_opportunity.py definitions)
     switch        25  on-ball screen with a screen switch (switch A): the two screen defenders
     decoy_help     8  third defender at an on-ball screen with no help within 3 s, and no help
                       onset of his within the window
     decoy_switch   7  on-ball screen without a screen switch: the two screen defenders
   The annotator is not told which clips are decoys. Help clips ask whether the defender left
   his man to help (full help / stunt / no / can't tell); switch clips whether the two defenders
   exchanged men. Gives the precision of each detected event type, the stunt share of detected
   help, and the yes-rate on decoys (a check of yes-bias; not a recall estimate).
Practice (not scored): one possession and one early-help clip from training games.

At most 2 items per game; every item is its own possession (M and E disjoint). Possessions of
earlier annotation bundles (bundles/*.json, not distributed) are left out, so a rerun without
them can draw a different sample; the sample that was labelled is in the derived-data release
(quick_check_matchups.csv, quick_check_events.csv).

Outputs (private: ids and times only; coordinates come from annotate_export.py):
    data/annotations/sample_quick.parquet
    data/annotations/quick/key.json

Usage:
    uv run python scripts/select_quick_check.py [--seed 17]
    uv run python scripts/annotate_export.py --name gdq_v1 --sample data/annotations/sample_quick.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.court import HOOP_LEFT, is_three_point_location
from ghost.tensors import load_game_set

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_corner_help_opportunity import DRIVE_FT, LEAD_STEPS, MIDDLE_FT, runs  # noqa: E402
from select_annotation_sample import proxies  # noqa: E402

M_ALLOC = {"pnr_switch": 10, "pnr_no_switch": 10, "drive": 8, "ball_movement": 6, "other": 6}
E_ALLOC = {"help_early": 25, "help_late": 15, "help_corner": 10, "switch": 25,
           "decoy_help": 8, "decoy_switch": 7}  # fmt: skip
EARLY_S, LATE_S = 1.0, 3.0
MAX_PER_GAME = 2
MIN_STEP = 3  # first step a moment can fall on (0.6 s of context)
EDGE_MS = 1000  # an event time needs >= 1 s of possession before and after it
ROOT = Path("data/annotations")


def strata(base: Path, P: pl.DataFrame) -> pl.DataFrame:
    """Stratum and number of valid steps per possession (select_annotation_sample rules)."""
    SW = pl.read_parquet(base / "events" / "switches.parquet")
    sw = SW.filter(pl.col("event_class") == "screen_switch").select(["game_id", "possession_id"])
    ob = pl.read_parquet(base / "analysis" / "onball_screens.parquet").select(
        ["game_id", "possession_id"]
    )
    rows = []
    for gid in sorted(P["game_id"].unique().to_list()):
        pt = load_game_set(base / "frames", [gid], P.filter(pl.col("game_id") == gid))
        ch, dr = proxies(pt)
        rows.append(pl.DataFrame({"game_id": pt.game_id, "possession_id": pt.possession_id,
                                  "handler_changes": ch, "drive": dr,
                                  "n_steps": pt.valid.sum(1).astype(int)}))  # fmt: skip
    S = pl.concat(rows).with_columns(pl.col("possession_id").cast(pl.Int32))
    key = ["game_id", "possession_id"]
    S = (S.join(sw.unique().with_columns(pl.lit(True).alias("has_switch")), on=key, how="left")
         .join(ob.unique().with_columns(pl.lit(True).alias("has_onball")), on=key, how="left")
         .with_columns(pl.col("has_switch").fill_null(False), pl.col("has_onball").fill_null(False)))  # fmt: skip
    return S.with_columns(
        pl.when(pl.col("has_switch")).then(pl.lit("pnr_switch"))
        .when(pl.col("has_onball")).then(pl.lit("pnr_no_switch"))
        .when(pl.col("drive")).then(pl.lit("drive"))
        .when(pl.col("handler_changes") >= 3).then(pl.lit("ball_movement"))
        .otherwise(pl.lit("other")).alias("stratum")
    )  # fmt: skip


def corner_helps(base: Path, P: pl.DataFrame, H: pl.DataFrame) -> list[dict]:
    """Help-B onsets leaving a strong-side strict-corner man during a drive episode."""
    out = []
    for gid in sorted(P["game_id"].unique().to_list()):
        Pg = P.filter(pl.col("game_id") == gid)
        pt = load_game_set(base / "frames", [gid], Pg)
        M = pl.read_parquet(base / "matchups" / "hmm_strat_help" / f"{gid}.parquet").filter(
            pl.col("state") < 5
        )
        mm = {(int(p), int(s), int(d)): int(st) for p, s, d, st in
              M.select(["possession_id", "step", "def_id", "state"]).iter_rows()}  # fmt: skip
        Hg = H.filter(pl.col("game_id") == gid)
        for j, pid in enumerate(pt.possession_id.tolist()):
            near = pt.valid[j] & (
                np.linalg.norm(pt.ball[j, :, :2] - HOOP_LEFT, axis=-1) <= DRIVE_FT
            )
            for s, e in runs(near):
                by = float(pt.ball[j, s, 1])
                if abs(by - 25.0) < MIDDLE_FT:
                    continue
                for d in [int(v) for v in pt.def_ids[j]]:
                    st = mm.get((int(pid), s, d))
                    if st is None:
                        continue
                    x, y = (float(c) for c in pt.off_xy[j, s, st])
                    if (not is_three_point_location(np.array(x), np.array(y), True)
                            or np.sign(y - 25.0) != np.sign(by - 25.0)
                            or abs(y - 25.0) < 21.0 or x > 14.0):  # fmt: skip
                        continue
                    man = int(pt.off_ids[j, st])
                    h = Hg.filter((pl.col("possession_id") == int(pid)) & (pl.col("def_id") == d)
                                  & (pl.col("man_off_id") == man)
                                  & pl.col("step_start").is_between(s - LEAD_STEPS, e)).sort("t_start_ms")  # fmt: skip
                    if h.height:
                        r = h.row(0, named=True)
                        out.append({"game_id": gid, "possession_id": int(pid), "def_id": d,
                                    "man_id": man, "t_ms": int(r["t_start_ms"])})  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    used = set()
    for name in ["gd_v2_pilot", "gd_v2_core100", "gdx_clips"]:
        f = ROOT / "bundles" / f"{name}.json"
        if f.exists():
            b = json.loads(f.read_text(encoding="utf-8"))
            used |= {(p["game_id"], int(p["possession_id"])) for p in b["possessions"]}
    PT = P.filter(pl.col("split") == "test")
    win = {(g, int(p)): (int(a), int(b)) for g, p, a, b in
           PT.select(["game_id", "possession_id", "t_start", "t_end"]).iter_rows()}  # fmt: skip

    def inside(g: str, p: int, t: int) -> bool:
        return (g, p) in win and win[(g, p)][0] + EDGE_MS <= t <= win[(g, p)][1] - EDGE_MS

    H = pl.read_parquet(base / "events" / "help.parquet").filter(pl.col("definition") == "B")
    G = pl.read_parquet(base / "analysis" / "onball_screens.parquet").filter(
        pl.col("game_id").is_in(PT["game_id"].unique().to_list()))  # fmt: skip
    cand: dict[str, list[dict]] = {k: [] for k in E_ALLOC}
    for r in G.iter_rows(named=True):
        g, p, tc = r["game_id"], int(r["possession_id"]), int(r["t_contact_ms"])
        pair = [int(r["screened_def_id"]), int(r["screener_def_id"])]
        base_row = {"game_id": g, "possession_id": p, "t_contact_ms": tc}
        if r["helped_3s"]:
            t = tc + int(round(1000 * r["help_delay_s"]))
            if inside(g, p, t):
                kind = "help_early" if r["help_delay_s"] <= EARLY_S else "help_late"
                cand[kind].append(base_row | {"defs": [int(r["first_helper_id"])], "t_ms": t,
                                              "delay_s": float(r["help_delay_s"])})  # fmt: skip
        elif inside(g, p, tc):
            third = [int(d) for d in r["defense_player_ids"] if int(d) not in pair]
            if len(third) == 3:
                d = third[int(rng.integers(3))]
                t = tc + int(rng.integers(0, 1001))
                hh = H.filter((pl.col("game_id") == g) & (pl.col("possession_id") == p)
                              & (pl.col("def_id") == d)
                              & pl.col("t_start_ms").is_between(t - 1500, t + 2500))  # fmt: skip
                if hh.height == 0 and inside(g, p, t):
                    cand["decoy_help"].append(base_row | {"defs": [d], "t_ms": t})
        if inside(g, p, tc):
            if r["switched"]:
                cand["switch"].append(base_row | {"defs": pair, "t_ms": tc})
            else:
                cand["decoy_switch"].append(base_row | {"defs": pair, "t_ms": tc})
    print("corner helps (test games) ...", flush=True)
    for c in corner_helps(base, PT, H):
        if inside(c["game_id"], c["possession_id"], c["t_ms"]):
            cand["help_corner"].append(c | {"defs": [c["def_id"]]})
    print({k: len(v) for k, v in cand.items()}, flush=True)

    print("strata (test games) ...", flush=True)
    S = strata(base, PT).join(PT.select(["game_id", "possession_id", "window_uid"]),
                              on=["game_id", "possession_id"])  # fmt: skip
    pool = dict(S.group_by("stratum").len().sort("stratum").rows())

    per_game: dict[str, int] = {}
    taken = set(used)
    items: list[dict] = []

    def ok(g: str, p: int) -> bool:
        return (g, p) not in taken and per_game.get(g, 0) < MAX_PER_GAME

    def take(g: str, p: int) -> None:
        taken.add((g, p))
        per_game[g] = per_game.get(g, 0) + 1

    # M first (stratified), then E categories, smallest pool first
    for st, k in M_ALLOC.items():
        Sg = S.filter((pl.col("stratum") == st) & (pl.col("n_steps") >= 12)).sort(
            ["game_id", "possession_id"])  # fmt: skip
        got = 0
        for i in rng.permutation(Sg.height):
            r = Sg.row(int(i), named=True)
            g, p, n = r["game_id"], int(r["possession_id"]), int(r["n_steps"])
            if not ok(g, p):
                continue
            h = n // 2
            k1 = int(rng.integers(MIN_STEP, h))
            k2 = int(rng.integers(h, n))
            take(g, p)
            items.append({"task": "M", "category": st, "game_id": g, "possession_id": p,
                          "window_uid": r["window_uid"], "n_steps": n, "keyframes": [k1, k2]})  # fmt: skip
            got += 1
            if got == k:
                break
        assert got == k, (st, got)
    for cat in sorted(E_ALLOC, key=lambda c: len(cand[c])):
        got = 0
        for i in rng.permutation(len(cand[cat])):
            c = cand[cat][int(i)]
            if not ok(c["game_id"], c["possession_id"]):
                continue
            take(c["game_id"], c["possession_id"])
            items.append({"task": "E", "category": cat, **{k: v for k, v in c.items()
                                                            if k not in ("def_id",)}})  # fmt: skip
            got += 1
            if got == E_ALLOC[cat]:
                break
        assert got == E_ALLOC[cat], (cat, got)

    # practice: one possession and one early help from training games
    PR = P.filter((pl.col("split") == "train") & (pl.col("duration_s").is_between(6, 12)))
    Gtr = pl.read_parquet(base / "analysis" / "onball_screens.parquet").join(
        PR.select(["game_id", "possession_id", "t_start", "t_end"]).with_columns(
            pl.col("possession_id").cast(pl.Int32)), on=["game_id", "possession_id"])  # fmt: skip
    Gtr = Gtr.filter((pl.col("help_delay_s") <= EARLY_S)
                     & (pl.col("t_contact_ms") >= pl.col("t_start") + 2 * EDGE_MS)
                     & (pl.col("t_contact_ms") <= pl.col("t_end") - 3 * EDGE_MS)).sort(
                         ["game_id", "possession_id"])  # fmt: skip
    pe = Gtr.row(int(rng.integers(Gtr.height)), named=True)
    prm = PR.filter(~((pl.col("game_id") == pe["game_id"])
                      & (pl.col("possession_id") == pe["possession_id"]))).sort(
                          ["game_id", "possession_id"])  # fmt: skip
    pm = prm.row(int(rng.integers(prm.height)), named=True)
    practice = [
        {"task": "M", "category": "practice", "game_id": pm["game_id"],
         "possession_id": int(pm["possession_id"]), "window_uid": pm["window_uid"], "practice": True},
        {"task": "E", "category": "practice_help", "game_id": pe["game_id"],
         "possession_id": int(pe["possession_id"]), "defs": [int(pe["first_helper_id"])],
         "t_ms": int(pe["t_contact_ms"]) + int(round(1000 * pe["help_delay_s"])),
         "t_contact_ms": int(pe["t_contact_ms"]), "practice": True},
    ]  # fmt: skip

    wuid = dict(zip(zip(P["game_id"], P["possession_id"].cast(pl.Int64), strict=True),
                    P["window_uid"], strict=True))  # fmt: skip
    rows = [{"game_id": it["game_id"], "possession_id": it["possession_id"],
             "window_uid": wuid[(it["game_id"], it["possession_id"])],
             "stratum": it["category"], "double_annotate": False}
            for it in items + practice]  # fmt: skip
    pl.DataFrame(rows).with_columns(
        pl.col("possession_id").cast(pl.Int32), pl.lit("quick_v1").alias("sample_version")
    ).write_parquet(ROOT / "sample_quick.parquet")
    meta = {"seed": args.seed, "split": "test", "m_alloc": M_ALLOC, "e_alloc": E_ALLOC,
            "m_pool_test": pool, "e_candidates": {k: len(v) for k, v in cand.items()},
            "n_games": len(per_game), "max_per_game": MAX_PER_GAME,
            "early_s": EARLY_S, "late_s": LATE_S, "min_step": MIN_STEP}  # fmt: skip
    (ROOT / "quick").mkdir(parents=True, exist_ok=True)
    (ROOT / "quick" / "key.json").write_text(
        json.dumps({"meta": meta, "items": items, "practice": practice}, indent=1)
    )
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
