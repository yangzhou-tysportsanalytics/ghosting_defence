"""Corner help per drive opportunity (SSAC27 review c, 2026-11-02).

analysis_help_left_man.py gives help rates per 1,000 steps spent guarding a man in a location.
Reviewers asked for rates per opportunity, a strict corner, intervals and the cost of leaving.

Opportunity: an episode in which the ball is within DRIVE_FT of the rim (a drive or another
paint touch; consecutive valid 5 Hz steps, at least 0.4 s), in a half-court possession. At the
first step of the episode, every defender whose HMM matchup is a man beyond the arc on the
ball's side of the rim-to-rim line (strong side; ball handler more than 4 ft from the line) has
one opportunity. Locations of the man:
  corner_strict  beyond the arc, |y - 25| >= 21 ft and within 14 ft of the baseline
  corner_wide    beyond the arc, |y - 25| >= 21 ft (the definition of the paper's Figure 4)
  above_break    beyond the arc, not corner_wide
Helped: the defender starts a help-B onset (D-018) leaving that man between 1 s before the
episode and its end.
Threat of the man: tercile of his three-point attempt rate (cuts from help_left_man.json, over
guarded-man steps) and, as a sensitivity, tercile of his shrunk three-point percentage (cuts
over the opportunities).
Outcomes per opportunity (possession level): points, the man takes a three in the possession.
Game-bootstrap intervals; helped minus not helped by location x threat.
Team rates (strict corner) vs points allowed per 100 (reports/ssac27/team_cards.csv).

Descriptive: help responds to the drive's threat, so the outcome contrasts are not effects.
Output: reports/analysis/corner_help_opportunity.json

Usage:
    uv run python scripts/analysis_corner_help_opportunity.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ghost import data as D
from ghost.court import HOOP_LEFT, is_three_point_location
from ghost.tensors import load_game_set

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_opponent_skill import shooting_profiles  # noqa: E402

DRIVE_FT = 12.0
MIN_STEPS = 2
LEAD_STEPS = 5  # a help onset up to 1 s before the episode counts
MIDDLE_FT = 4.0
LOCS = ("corner_strict", "corner_wide", "above_break")
B = 1000


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end] (inclusive) of runs of True of length >= MIN_STEPS."""
    out, t, n = [], 0, len(mask)
    while t < n:
        if mask[t]:
            s = t
            while t + 1 < n and mask[t + 1]:
                t += 1
            if t - s + 1 >= MIN_STEPS:
                out.append((s, t))
        t += 1
    return out


def collect(cfg: D.DataConfig, rate3: dict, pct3: dict) -> pl.DataFrame:
    base = cfg.processed_dir / "all"
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    H = pl.read_parquet(base / "events" / "help.parquet").filter(pl.col("definition") == "B")
    rows, t0 = [], time.time()
    games = sorted(P["game_id"].unique().to_list())
    for gi, gid in enumerate(games, 1):
        Pg = P.filter(pl.col("game_id") == gid)
        pt = load_game_set(base / "frames", [gid], Pg)
        key = {int(p): i for i, p in enumerate(pt.possession_id)}
        dteam = dict(
            zip(Pg["possession_id"].to_list(), Pg["defense_team_id"].to_list(), strict=True)
        )
        M = pl.read_parquet(base / "matchups" / "hmm_strat_help" / f"{gid}.parquet").filter(
            (pl.col("state") < 5) & pl.col("possession_id").is_in(list(key))
        )
        mm = {(int(p), int(s), int(d)): int(st) for p, s, d, st in
              M.select(["possession_id", "step", "def_id", "state"]).iter_rows()}  # fmt: skip
        Hg = H.filter(pl.col("game_id") == gid)
        helps: dict = {}
        for p, d, m, k in Hg.select(
            ["possession_id", "def_id", "man_off_id", "step_start"]
        ).iter_rows():
            helps.setdefault((int(p), int(d), int(m)), []).append(int(k))
        defs_of = {int(p): [int(x) for x in ids] for p, ids in
                   zip(Pg["possession_id"], Pg["defense_player_ids"], strict=True)}  # fmt: skip
        for pid, j in key.items():
            v = pt.valid[j]
            near = v & (np.linalg.norm(pt.ball[j, :, :2] - HOOP_LEFT, axis=-1) <= DRIVE_FT)
            for ep, (s, e) in enumerate(runs(near)):
                by = float(pt.ball[j, s, 1])
                if abs(by - 25.0) < MIDDLE_FT:
                    continue
                for d in defs_of.get(pid, []):
                    st = mm.get((pid, s, d))
                    if st is None:
                        continue
                    x, y = (float(c) for c in pt.off_xy[j, s, st])
                    if not is_three_point_location(np.array(x), np.array(y), True):
                        continue
                    if np.sign(y - 25.0) != np.sign(by - 25.0):
                        continue  # weak side
                    wide = abs(y - 25.0) >= 21.0
                    man = int(pt.off_ids[j, st])
                    ks = helps.get((pid, d, man), [])
                    rows.append({
                        "game_id": gid, "possession_id": pid, "episode": ep, "def_id": d,
                        "team": int(dteam[pid]), "man_id": man,
                        "loc": 0 if (wide and x <= 14.0) else (1 if wide else 2),
                        "helped": any(s - LEAD_STEPS <= k <= e for k in ks),
                        "rate3": rate3.get(man, 0.0), "pct3": pct3.get(man),
                    })  # fmt: skip
        if gi % 100 == 0 or gi == len(games):
            print(
                f"[{gi}/{len(games)}] {time.time() - t0:.0f}s opportunities={len(rows)}", flush=True
            )
    return pl.DataFrame(rows)


def boot(df: pl.DataFrame, col: str, rng, by: str | None = None) -> dict:
    """Mean of col (or helped - not helped difference when by='helped'), game bootstrap."""
    g = df["game_id"].to_numpy()
    ug, inv = np.unique(g, return_inverse=True)
    idx = rng.integers(0, len(ug), size=(B, len(ug)))
    v = df[col].cast(pl.Float64).to_numpy()
    if by is None:
        s, c = np.bincount(inv, v, len(ug)), np.bincount(inv, minlength=len(ug)).astype(float)
        m = s[idx].sum(1) / c[idx].sum(1)
        return {"mean": float(v.mean()), "ci95": [float(np.quantile(m, q)) for q in (0.025, 0.975)],
                "n": df.height}  # fmt: skip
    h = df[by].to_numpy().astype(bool)
    out = {}
    ms = []
    for flag in (True, False):
        w = (h == flag).astype(float)
        s, c = np.bincount(inv, v * w, len(ug)), np.bincount(inv, w, len(ug))
        ms.append(s[idx].sum(1) / c[idx].sum(1))
        out["helped" if flag else "not_helped"] = {"mean": float(v[h == flag].mean()),
                                                   "n": int((h == flag).sum())}  # fmt: skip
    diff = ms[0] - ms[1]
    out["difference"] = {"mean": out["helped"]["mean"] - out["not_helped"]["mean"],
                         "ci95": [float(np.nanquantile(diff, q)) for q in (0.025, 0.975)]}  # fmt: skip
    return out


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    prof, _ = shooting_profiles(pl.read_parquet(base / "analysis" / "shots.parquet"),
                                pl.read_parquet(base / "possessions.parquet"))  # fmt: skip
    rate3 = dict(zip(prof["shooter_id"].to_list(), prof["three_rate"].to_list(), strict=True))
    pct3 = dict(zip(prof["shooter_id"].to_list(), prof["three_pct_shrunk"].to_list(), strict=True))
    Opp = collect(cfg, rate3, pct3)
    cuts = json.loads(Path("reports/analysis/help_left_man.json").read_text())[
        "threat_terciles_three_rate"]  # fmt: skip
    pcuts = np.quantile(Opp["pct3"].drop_nulls().to_numpy(), [1 / 3, 2 / 3])

    def terc(c: str, k) -> pl.Expr:
        return pl.when(pl.col(c) <= k[0]).then(0).when(pl.col(c) <= k[1]).then(1).otherwise(2)

    Opp = Opp.with_columns(terc("rate3", cuts).alias("threat"),
                       pl.when(pl.col("pct3").is_null()).then(None)
                       .otherwise(terc("pct3", pcuts)).alias("threat_pct"))  # fmt: skip
    # outcomes of the possession
    P = pl.read_parquet(base / "possessions.parquet").select(
        ["game_id", "possession_id", "poss_uid"]
    )
    led = D.L.ledger(cfg.version).select(["poss_uid", "points"])
    shots = (pl.read_parquet(base / "analysis" / "shots.parquet")
             .filter(pl.col("possession_id").is_not_null())
             .select(["game_id", "possession_id", "shooter_id", "is_three"])
             .unique(subset=["game_id", "possession_id"], keep="first"))  # fmt: skip
    Opp = (Opp.join(P, on=["game_id", "possession_id"], how="left")
         .join(led, on="poss_uid", how="left")
         .join(shots, on=["game_id", "possession_id"], how="left")
         .with_columns(((pl.col("shooter_id") == pl.col("man_id")) & pl.col("is_three"))
                       .fill_null(False).alias("man_shoots_three"))
         .sort(["game_id", "possession_id", "episode", "def_id"]))  # fmt: skip
    rng = np.random.default_rng(11)
    rep: dict = {"drive_ft": DRIVE_FT, "n_opportunities": Opp.height,
                 "n_episodes": Opp.select(["game_id", "possession_id", "episode"]).n_unique(),
                 "threat_cuts_three_rate": cuts, "threat_cuts_three_pct": [float(c) for c in pcuts],
                 "help_rate": {}, "outcomes": {}}  # fmt: skip
    for lo, name in enumerate(LOCS):
        d0 = Opp.filter(pl.col("loc") == lo) if lo != 1 else Opp.filter(pl.col("loc") <= 1)
        rep["help_rate"][name] = {"all": boot(d0, "helped", rng)}
        for tcol in ("threat", "threat_pct"):
            for th in range(3):
                d = d0.filter(pl.col(tcol) == th)
                rep["help_rate"][name][f"{tcol}{th}"] = boot(d, "helped", rng)
            # top minus bottom tercile of the help rate, game bootstrap
            dd = d0.filter(pl.col(tcol).is_in([0, 2])).with_columns(
                (pl.col(tcol) == 2).alias("top")
            )
            rep["help_rate"][name][f"{tcol}_top_minus_bottom"] = boot(dd, "helped", rng, by="top")[
                "difference"]  # fmt: skip
        dp = d0.filter(pl.col("points").is_not_null())
        rep["outcomes"][name] = {"points": boot(dp, "points", rng, by="helped"),
                                 "man_shoots_three": boot(d0, "man_shoots_three", rng, by="helped")}  # fmt: skip
        for th in range(3):
            d = dp.filter(pl.col("threat") == th)
            rep["outcomes"][name][f"points_threat{th}"] = boot(d, "points", rng, by="helped")
    # team rates (strict corner) vs points allowed
    T = (Opp.filter(pl.col("loc") == 0).group_by("team")
         .agg(pl.col("helped").mean().alias("corner_help_per_opp"), pl.len().alias("n")))  # fmt: skip
    cards = pl.read_csv("reports/ssac27/team_cards.csv").select(["team_id", "team",
                                                                  "pts_allowed_per_100"])  # fmt: skip
    T = T.join(cards, left_on="team", right_on="team_id").sort("team")
    r = stats.pearsonr(T["corner_help_per_opp"], T["pts_allowed_per_100"])
    rep["team_strict_corner_help_vs_points_allowed"] = {
        "pearson_r": float(r[0]), "p": float(r[1]), "n_teams": T.height,
        "rate_min_max": [float(T["corner_help_per_opp"].min()), float(T["corner_help_per_opp"].max())],
    }  # fmt: skip
    T.write_csv("reports/analysis/corner_help_opportunity_by_team.csv")
    out = Path("reports/analysis/corner_help_opportunity.json")
    out.write_text(json.dumps(rep, indent=1))
    for name in LOCS:
        h = rep["help_rate"][name]
        print(name, f"all {h['all']['mean']:.3f} n={h['all']['n']}",
              " ".join(f"T{k}={h[f'threat{k}']['mean']:.3f}" for k in range(3)),
              "top-bottom", {k: round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]
                             for k, v in h["threat_top_minus_bottom"].items()},
              "pct top-bottom", [round(x, 3) for x in h["threat_pct_top_minus_bottom"]["ci95"]])  # fmt: skip
        o = rep["outcomes"][name]["points"]
        print("   points helped", round(o["helped"]["mean"], 3), "not", round(o["not_helped"]["mean"], 3),
              "diff", round(o["difference"]["mean"], 3), [round(x, 3) for x in o["difference"]["ci95"]])  # fmt: skip
    print(rep["team_strict_corner_help_vs_points_allowed"])


if __name__ == "__main__":
    main()
