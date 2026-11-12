"""Help by whom the helper leaves (writing request B1; Pelechrinis & Goldsberry 2021).

For every help-B onset (events/help.parquet; definition D-018), the man the helper leaves
(``man_off_id``) at the onset step:
* location: corner three (beyond the arc, |y - 25| >= 21 ft, as the xFG zones), above-the-break
  three, inside the arc;
* shooting threat: tercile of his three-point attempt rate (shrunk profiles of
  phase4_opponent_skill.shooting_profiles; terciles over the guarded-man steps).
Rates: help onsets per 1,000 defender-steps spent guarding a man of that location / threat
(exposure = HMM matchup states of all defenders, 5 Hz).
Outcomes (descriptive), per possession by its first help: points, xFG of the shot, share of
possessions in which the man left takes the shot, and takes a three; game-bootstrap intervals.
Team early help (on-ball screens, help within 1 s; analysis_team_screen_defense.py) recomputed
without early helps that leave a man in the corner, and without those that leave a corner man in
the top threat tercile; Spearman correlation of team rates with the original.

Review A1 (2026-10-30), added without changing the values above: the side of the man left,
relative to the ball, at the help onset: strong (same side of the rim-to-rim line y = 25 as the
ball), weak, or middle (ball within 4 ft of that line, side undefined); sensitivity with the
side 1 s before the onset (a drive across the middle can flip it). Rates by location x side and
location x side x threat; first help of the possession by side (points, the man left takes the
shot); team early-help ranking without strong-side corner helps. Separate random stream.

Output: reports/analysis/help_left_man.json

Usage:
    uv run python scripts/analysis_help_left_man.py
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

LOCS = ("corner3", "above3", "inside")
SIDES = ("strong", "weak", "middle")
MIDDLE_FT = 4.0  # ball this close to y = 25: no strong / weak side
B = 500


def side(man_y: np.ndarray, ball_y: np.ndarray) -> np.ndarray:
    """Index into SIDES: the man on the ball's side of y = 25 (strong), the other (weak), or
    the ball near the middle line."""
    strong = np.sign(man_y - 25.0) == np.sign(ball_y - 25.0)
    return np.where(np.abs(ball_y - 25.0) < MIDDLE_FT, 2, np.where(strong, 0, 1))


def location(xy: np.ndarray) -> np.ndarray:
    three = is_three_point_location(xy[..., 0], xy[..., 1], True)
    corner = three & (np.abs(xy[..., 1] - 25.0) >= 21.0)
    return np.where(three, np.where(corner, 0, 1), 2)  # index into LOCS


def boot_mean(v: np.ndarray, games: np.ndarray, rng) -> list[float]:
    ug, inv = np.unique(games, return_inverse=True)
    s, c = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    idx = rng.integers(0, len(ug), size=(B, len(ug)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def by_side(
    rep: dict, X: pl.DataFrame, E: pl.DataFrame, first: pl.DataFrame, S: pl.DataFrame, terc
) -> None:
    """Review A1: everything by the side of the man left (separate random stream)."""
    rng = np.random.default_rng(10)
    out: dict = {
        "sides": list(SIDES),
        "middle_band_ft": MIDDLE_FT,
        "rates_per_1000_guarding_steps": {},
    }
    xe = X.group_by(["loc", "side", "threat"]).len().rename({"len": "steps"})
    he = E.group_by(["loc", "side", "threat"]).len().rename({"len": "helps"})
    J = xe.join(he, on=["loc", "side", "threat"], how="left").fill_null(0)
    for lo in range(3):
        for sd in range(3):
            d = J.filter((pl.col("loc") == lo) & (pl.col("side") == sd))
            a, b = int(d["helps"].sum()), int(d["steps"].sum())
            out["rates_per_1000_guarding_steps"][f"{LOCS[lo]}_{SIDES[sd]}"] = {
                "rate": 1000 * a / b if b else None, "helps": a, "steps": b}  # fmt: skip
            for th in range(3):
                dd = d.filter(pl.col("threat") == th)
                a, b = int(dd["helps"].sum()), int(dd["steps"].sum())
                out["rates_per_1000_guarding_steps"][f"{LOCS[lo]}_{SIDES[sd]}_threat{th}"] = {
                    "rate": 1000 * a / b if b else None, "helps": a, "steps": b}  # fmt: skip
    corner = E.filter(pl.col("loc") == 0)
    out["corner_helps_by_side"] = {
        "at_onset": {SIDES[k]: float((corner["side"] == k).mean()) for k in range(3)},
        "one_second_before": {SIDES[k]: float((corner["side_1s_before"] == k).mean()) for k in range(3)},
        "n": corner.height,
    }  # fmt: skip
    out["outcomes_by_first_help"] = {}
    for lo in range(3):
        for sd in range(3):
            d = first.filter((pl.col("loc") == lo) & (pl.col("side") == sd))
            if d.height < 50:
                continue
            o = {"n_possessions": d.height}
            for c in ("points", "xfg_real", "man_shoots", "man_shoots_three"):
                v = d.filter(pl.col(c).is_not_null())
                vv = v[c].cast(pl.Float64).to_numpy()
                o[c] = {
                    "mean": float(vv.mean()),
                    "ci95": boot_mean(vv, v["game_id"].to_numpy(), rng),
                }
            out["outcomes_by_first_help"][f"{LOCS[lo]}_{SIDES[sd]}"] = o
    # team early-help ranking without strong-side corner helps (side at onset, and 1 s before)
    S2 = S.with_columns(
        (pl.col("early") & ~((pl.col("loc") == 0) & (pl.col("side") == 0)).fill_null(False))
        .alias("early_not_strong_corner"),
        (pl.col("early") & ~((pl.col("loc") == 0) & (pl.col("side_1s_before") == 0)).fill_null(False))
        .alias("early_not_strong_corner_1s"),
    )  # fmt: skip
    T = (S2.group_by("defense_team_id")
         .agg(pl.col("early").mean(), pl.col("early_not_strong_corner").mean(),
              pl.col("early_not_strong_corner_1s").mean())
         .sort("defense_team_id"))  # fmt: skip
    se = S2.filter(pl.col("early"))
    out["team_early_help_without_strong_corner"] = {
        "share_of_early_helps_leaving_strong_side_corner_man": float(((se["loc"] == 0) & (se["side"] == 0)).mean()),
        "spearman_original_vs_without": float(stats.spearmanr(T["early"], T["early_not_strong_corner"])[0]),
        "spearman_original_vs_without_side_1s_before": float(
            stats.spearmanr(T["early"], T["early_not_strong_corner_1s"])[0]),
        "league_rate": {c: float(S2[c].mean()) for c in ("early", "early_not_strong_corner")},
    }  # fmt: skip
    rep["by_side"] = out


def by_unit_tables(X: pl.DataFrame, E: pl.DataFrame) -> None:
    """Help rates per 1,000 guarding steps by location x side, per defending team and per
    defender (team cards and player cards of the SSAC paper). Counts only; no coordinates."""
    out = Path("reports/analysis")
    for unit, name in (("team", "help_side_by_team.csv"), ("def_id", "help_side_by_player.csv")):
        xs = X.group_by([unit, "loc", "side"]).len().rename({"len": "steps"})
        hs = E.group_by([unit, "loc", "side"]).len().rename({"len": "helps"})
        T = (xs.join(hs, on=[unit, "loc", "side"], how="left").fill_null(0)
             .with_columns(pl.col("loc").map_elements(lambda v: LOCS[v], return_dtype=pl.Utf8),
                           pl.col("side").map_elements(lambda v: SIDES[v], return_dtype=pl.Utf8),
                           (1000 * pl.col("helps") / pl.col("steps")).alias("rate_per_1000"))
             .sort([unit, "loc", "side"]))  # fmt: skip
        T.write_csv(out / name)


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    prof, _ = shooting_profiles(pl.read_parquet(base / "analysis" / "shots.parquet"),
                                pl.read_parquet(base / "possessions.parquet"))  # fmt: skip
    rate = dict(zip(prof["shooter_id"].to_list(), prof["three_rate"].to_list(), strict=True))
    H = pl.read_parquet(base / "events" / "help.parquet").filter(pl.col("definition") == "B")
    expo, ev_rows = [], []
    t0 = time.time()
    games = sorted(P["game_id"].unique().to_list())
    for gi, gid in enumerate(games, 1):
        Pg = P.filter(pl.col("game_id") == gid)
        pt = load_game_set(base / "frames", [gid], Pg)
        key = {int(p): i for i, p in enumerate(pt.possession_id)}
        dteam = dict(
            zip(Pg["possession_id"].to_list(), Pg["defense_team_id"].to_list(), strict=True)
        )
        team_of = np.array([dteam[int(p)] for p in pt.possession_id])
        M = pl.read_parquet(base / "matchups" / "hmm_strat_help" / f"{gid}.parquet").filter(
            (pl.col("state") < 5) & pl.col("possession_id").is_in(list(key))
        )
        i = np.array([key[int(p)] for p in M["possession_id"]])
        t, s = M["step"].to_numpy(), M["state"].to_numpy()
        ok = pt.valid[i, t]
        i, t, s = i[ok], t[ok], s[ok]
        dids = M["def_id"].to_numpy()[ok]
        loc = location(pt.off_xy[i, t, s])
        man = pt.off_ids[i, s]
        sd = side(pt.off_xy[i, t, s, 1], pt.ball[i, t, 1])
        expo.append(pl.DataFrame({"loc": loc.astype(np.int8), "side": sd.astype(np.int8),
                                  "team": team_of[i], "def_id": dids,
                                  "rate": [rate.get(int(m), 0.0) for m in man]}))  # fmt: skip
        Hg = H.filter((pl.col("game_id") == gid) & pl.col("possession_id").is_in(list(key)))
        for r in Hg.iter_rows(named=True):
            j = key[int(r["possession_id"])]
            k = int(r["step_start"])
            offs = [int(x) for x in pt.off_ids[j]]
            if (
                r["man_off_id"] not in offs
                or not (0 <= k < pt.valid.shape[1])
                or not pt.valid[j, k]
            ):
                continue
            mi = offs.index(r["man_off_id"])
            xy = pt.off_xy[j, k, mi]
            k1 = max(0, k - 5)  # 1 s before the onset
            sd0 = int(side(xy[1:2], pt.ball[j, k, 1:2])[0])
            sd1 = int(side(pt.off_xy[j, k1, mi, 1:2], pt.ball[j, k1, 1:2])[0])
            ev_rows.append({"game_id": gid, "possession_id": int(r["possession_id"]),
                            "def_id": r["def_id"], "man_off_id": r["man_off_id"],
                            "t_start_ms": r["t_start_ms"], "loc": int(location(xy[None])[0]),
                            "rate": rate.get(int(r["man_off_id"]), 0.0),
                            "man_dist_rim": float(np.linalg.norm(xy - HOOP_LEFT)),
                            "side": sd0, "side_1s_before": sd1, "team": int(team_of[j])})  # fmt: skip
        if gi % 100 == 0 or gi == len(games):
            print(f"[{gi}/{len(games)}] {time.time() - t0:.0f}s helps={len(ev_rows)}", flush=True)
    X = pl.concat(expo)
    E = pl.DataFrame(ev_rows)
    cuts = np.quantile(X["rate"].to_numpy(), [1 / 3, 2 / 3])

    def terc(c: str) -> pl.Expr:
        return pl.when(pl.col(c) <= cuts[0]).then(0).when(pl.col(c) <= cuts[1]).then(1).otherwise(2)

    X = X.with_columns(terc("rate").alias("threat"))
    E = E.with_columns(terc("rate").alias("threat"))
    rep: dict = {"n_help_onsets": E.height, "threat_terciles_three_rate": [float(c) for c in cuts],
                 "locations": list(LOCS), "rates_per_1000_guarding_steps": {}}  # fmt: skip
    xe = X.group_by(["loc", "threat"]).len().rename({"len": "steps"})
    he = E.group_by(["loc", "threat"]).len().rename({"len": "helps"})
    J = xe.join(he, on=["loc", "threat"], how="left").fill_null(0).sort(["loc", "threat"])
    for r in J.iter_rows(named=True):
        rep["rates_per_1000_guarding_steps"][f"{LOCS[r['loc']]}_threat{r['threat']}"] = {
            "rate": 1000 * r["helps"] / r["steps"], "helps": r["helps"], "steps": r["steps"]}  # fmt: skip
    for lo in range(3):
        a, b = (
            J.filter(pl.col("loc") == lo)["helps"].sum(),
            J.filter(pl.col("loc") == lo)["steps"].sum(),
        )
        rep["rates_per_1000_guarding_steps"][LOCS[lo]] = {"rate": 1000 * a / b, "helps": int(a),
                                                           "steps": int(b)}  # fmt: skip

    # outcomes by the first help of the possession
    led = D.L.ledger(cfg.version).select(["poss_uid", "points"])
    xfg = (pl.read_parquet(base / "analysis" / "ghost_points.parquet")
           .select(["game_id", "possession_id", "xfg_real"])
           .unique(subset=["game_id", "possession_id"], keep="first"))  # fmt: skip
    shots = pl.read_parquet(base / "analysis" / "shots.parquet").filter(
        pl.col("possession_id").is_not_null()
    ).select(["game_id", "possession_id", "shooter_id", "is_three"]).unique(
        subset=["game_id", "possession_id"], keep="first")  # fmt: skip
    first = (E.sort(["game_id", "possession_id", "t_start_ms"])
             .unique(subset=["game_id", "possession_id"], keep="first")
             .join(P.select(["game_id", "possession_id", "poss_uid"]), on=["game_id", "possession_id"])
             .join(led, on="poss_uid", how="left")
             .join(xfg, on=["game_id", "possession_id"], how="left")
             .join(shots, on=["game_id", "possession_id"], how="left")
             .with_columns((pl.col("shooter_id") == pl.col("man_off_id")).fill_null(False).alias("man_shoots"),
                           ((pl.col("shooter_id") == pl.col("man_off_id")) & pl.col("is_three"))
                           .fill_null(False).alias("man_shoots_three")))  # fmt: skip
    rng = np.random.default_rng(9)
    rep["outcomes_by_first_help"] = {}
    for lo in range(3):
        for th in (None, 0, 1, 2):
            d = first.filter(pl.col("loc") == lo)
            if th is not None:
                d = d.filter(pl.col("threat") == th)
            if d.height < 50:
                continue
            g = d["game_id"].to_numpy()
            out = {"n_possessions": d.height}
            for c in ("points", "xfg_real", "man_shoots", "man_shoots_three"):
                v = d.filter(pl.col(c).is_not_null())
                vv = v[c].cast(pl.Float64).to_numpy()
                out[c] = {
                    "mean": float(vv.mean()),
                    "ci95": boot_mean(vv, v["game_id"].to_numpy(), rng),
                }
            name = LOCS[lo] + ("" if th is None else f"_threat{th}")
            rep["outcomes_by_first_help"][name] = out
            del g

    # team early help without helps that leave a corner man
    S = pl.read_parquet(base / "analysis" / "onball_screens.parquet")
    S = S.with_columns((pl.col("help_delay_s").fill_null(99) <= 1.0).alias("early"),
                       (pl.col("t_contact_ms") + (pl.col("help_delay_s") * 1000).round()).alias("t_help"))  # fmt: skip
    Eh = E.select(
        [
            "game_id",
            "possession_id",
            "def_id",
            "t_start_ms",
            "loc",
            "threat",
            "side",
            "side_1s_before",
        ]
    ).rename({"def_id": "first_helper_id", "t_start_ms": "t_help"})
    S = S.join(Eh.with_columns(pl.col("t_help").cast(pl.Float64)),
               on=["game_id", "possession_id", "first_helper_id", "t_help"], how="left")  # fmt: skip
    se = S.filter(pl.col("early"))
    rep["early_help_screens"] = {"n": se.height,
                                 "share_matched_to_help_event": float(se["loc"].is_not_null().mean()),
                                 "share_leaving_corner_man": float((se["loc"] == 0).mean()),
                                 "share_leaving_corner_top_threat": float(((se["loc"] == 0) & (se["threat"] == 2)).mean())}  # fmt: skip
    S = S.with_columns(
        (pl.col("early") & ~(pl.col("loc") == 0).fill_null(False)).alias("early_not_corner"),
        (pl.col("early") & ~((pl.col("loc") == 0) & (pl.col("threat") == 2)).fill_null(False))
        .alias("early_not_corner_shooter"))  # fmt: skip
    T = S.group_by("defense_team_id").agg(pl.col("early").mean(), pl.col("early_not_corner").mean(),
                                          pl.col("early_not_corner_shooter").mean(), pl.len())  # fmt: skip
    corner_share = (S.filter(pl.col("early")).group_by("defense_team_id")
                    .agg((pl.col("loc") == 0).mean().alias("s"))["s"])  # fmt: skip
    rho = lambda c: float(stats.spearmanr(T["early"], T[c])[0])  # noqa: E731
    rep["team_early_help"] = {
        "n_teams": T.height,
        "spearman_original_vs_without_corner": rho("early_not_corner"),
        "spearman_original_vs_without_corner_shooter": rho("early_not_corner_shooter"),
        "league_rate": {c: float(S[c].mean())
                        for c in ("early", "early_not_corner", "early_not_corner_shooter")},
        "team_share_of_early_help_leaving_corner_man_min_max": [float(corner_share.min()),
                                                                float(corner_share.max())],
    }  # fmt: skip
    by_side(rep, X, E, first, S, terc)
    by_unit_tables(X, E)
    p = Path("reports/analysis/help_left_man.json")
    p.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "outcomes_by_first_help"}, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
