"""Phase 4 task 2: phases and help-geometry covariates (D-020), and what they do to the player
and team components of the rule-ghost deviation.

Per (possession, defender): share of man-guarding steps in each phase (help > closeout > recovery >
screen > pre_screen > other), share of off-ball steps inside the ball handler's sightline cone,
share of steps in long paint stretches, longest paint stretch. League means of |dev| and sag by
phase. Then the context adjustment of phase4_reliability.py is refitted with these covariates
added: context R², split-half reliability and the team share of the systematic variance, with and
without them.

Outputs:
    <processed_dir>/all/analysis/phase_context_<defs>.parquet
    reports/phase4/<version>_all/phase_context_<defs>.json

Usage:
    uv run python scripts/phase4_phase_context.py [--defs A|B]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from ghost import data as D
from ghost.deviation.phases import (
    PHASES,
    PhaseDefs,
    in_paint,
    in_sightline_cone,
    long_paint_runs,
    phase_labels,
    rule_ghost_deviation,
)
from ghost.matchup.events import EventConfig, stable_path
from ghost.tensors import load_game_set

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_reliability import COVS, MIN_POSS, split_half, variance_shares  # noqa: E402

MIN_STEPS = 5
EXTRA = [f"share_{p}" for p in PHASES if p != "other"] + ["cone_share_offball", "paint_long_share"]


def intervals(df: pl.DataFrame, s: str, e: str) -> dict:
    out: dict = {}
    for pid, did, a, b in zip(df["possession_id"], df["def_id"], df[s], df[e], strict=True):
        out.setdefault((int(pid), int(did)), []).append((int(a), int(b)))
    return out


def adjust(df: pl.DataFrame, y: str, extra: list[str]) -> tuple[np.ndarray, float]:
    """The reliability script's context OLS, optionally with extra covariates."""
    cols = [np.ones(df.height)] + [df[c].to_numpy() for c in COVS]
    cols += [df["man_to_hoop_ft"].to_numpy() ** 2, df["man_to_ball_ft"].to_numpy() ** 2]
    cols += [(df["n_men"].to_numpy() > 1).astype(float), np.log(df["n_steps"].to_numpy())]
    cols += [df[c].to_numpy() for c in extra]
    X, yy = np.column_stack(cols), df[y].to_numpy()
    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    res = yy - X @ beta
    return res, float(1 - (res**2).sum() / ((yy - yy.mean()) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defs", default="A", choices=["A", "B"])
    args = ap.parse_args()
    defs = PhaseDefs.variant(args.defs)
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    prm = json.loads((base / "matchups" / "hmm_strat_help_params.json").read_text())
    g_strong, g_weak = np.array(prm["gamma"]["strong"]), np.array(prm["gamma"]["weak"])
    hold = EventConfig().steps(EventConfig().min_hold_s)
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    H = pl.read_parquet(base / "events" / "help.parquet").filter(pl.col("definition") == "B")
    C = pl.read_parquet(base / "events" / "closeouts.parquet")
    S = pl.read_parquet(base / "analysis" / "onball_screens.parquet")

    rows, t0 = [], time.time()
    acc = {p: np.zeros(3) for p in PHASES}  # sum |dev|, sum sag, n steps
    games = sorted(P["game_id"].unique().to_list())
    for gi, gid in enumerate(games, 1):
        Pg = P.filter(pl.col("game_id") == gid)
        pt = load_game_set(base / "frames", [gid], Pg)
        M = pl.read_parquet(base / "matchups" / "hmm_strat_help" / f"{gid}.parquet")
        n, T = pt.valid.shape
        key = {int(p): i for i, p in enumerate(pt.possession_id)}
        path = np.full((n, 5, T), -1, dtype=np.int8)
        ri = np.array([key[int(p)] for p in M["possession_id"]])
        path[ri, M["def_slot"].to_numpy(), M["step"].to_numpy()] = M["state"].to_numpy()
        h_iv = intervals(H.filter(pl.col("game_id") == gid), "t_start_ms", "t_end_ms")
        c_iv = intervals(C.filter(pl.col("game_id") == gid), "t_catch_ms", "t_closed_ms")
        Sg = S.filter(pl.col("game_id") == gid)
        contacts: dict = {}
        for pid, c in zip(Sg["possession_id"], Sg["t_contact_ms"], strict=True):
            contacts.setdefault(int(pid), []).append(int(c))
        for i in range(n):
            pid, valid = int(pt.possession_id[i]), pt.valid[i]
            stable = stable_path(path[i], valid, hold)
            dxy, oxy = pt.def_xy[i].astype(np.float64), pt.off_xy[i].astype(np.float64)
            dev, sag = rule_ghost_deviation(
                dxy, oxy, pt.ball[i, :, :2].astype(np.float64), stable, valid, g_strong, g_weak
            )
            ids = [int(x) for x in pt.def_ids[i]]
            lab = phase_labels(
                pt.t[i].astype(np.int64),
                [h_iv.get((pid, d), []) for d in ids],
                [c_iv.get((pid, d), []) for d in ids],
                contacts.get(pid, []),
                defs,
            )
            h = pt.handler[i].astype(int)
            hxy = np.where((h >= 0)[:, None], oxy[np.arange(T), np.clip(h, 0, 4)], np.nan)
            cone = in_sightline_cone(hxy, dxy, defs.cone_half_angle_deg)  # (T, 5)
            long_flag, longest = long_paint_runs(in_paint(dxy), valid, defs.paint_long_s)
            for d in range(5):
                man = ~np.isnan(dev[d])
                if man.sum() < MIN_STEPS:
                    continue
                offball = man & (stable[d].astype(int) != h) & (h >= 0)
                row = {"game_id": gid, "possession_id": pid, "def_id": ids[d]}
                for k, p in enumerate(PHASES):
                    sel = man & (lab[d] == k)
                    row[f"share_{p}"] = float(sel.sum() / man.sum())
                    acc[p] += [np.nansum(dev[d, sel]), np.nansum(sag[d, sel]), sel.sum()]
                row["cone_share_offball"] = float(cone[offball, d].mean()) if offball.any() else 0.0
                row["paint_long_share"] = float(long_flag[valid, d].mean())
                row["paint_longest_s"] = float(longest[d])
                rows.append(row)
        if gi % 100 == 0 or gi == len(games):
            print(f"[{gi}/{len(games)}] {time.time() - t0:.0f}s rows={len(rows)}", flush=True)
    ctx = pl.DataFrame(rows)
    ctx.write_parquet(base / "analysis" / f"phase_context_{args.defs}.parquet")

    rep: dict = {
        "defs": args.defs,
        "definitions": defs.__dict__,
        "league_by_phase": {
            p: {
                "share_of_steps": float(v[2] / sum(a[2] for a in acc.values())),
                "mean_dev_ft": float(v[0] / max(v[2], 1)),
                "mean_sag_ft": float(v[1] / max(v[2], 1)),
            }
            for p, v in acc.items()
        },  # fmt: skip
        "covariate_means": {c: float(ctx[c].mean()) for c in EXTRA + ["paint_longest_s"]},
    }
    dev_tab = pl.read_parquet(base / "analysis" / "rule_ghost_dev.parquet").join(
        ctx, on=["game_id", "possession_id", "def_id"], how="inner"
    )
    rep["n_rows_joined"] = dev_tab.height
    players = dev_tab.group_by("def_id").len().filter(pl.col("len") >= MIN_POSS)["def_id"].to_list()
    rep["effect_of_covariates"] = {}
    for y in ("sag_ft", "dev_ft"):
        out = {}
        for lab_, extra in (("base", []), ("with_phase_and_geometry", EXTRA)):
            res, r2 = adjust(dev_tab, y, extra)
            d = dev_tab.with_columns(pl.Series("adj", res))
            r, npl = split_half(d, "adj", players)
            vs = variance_shares(d, "adj", players)
            out[lab_] = {"context_r2": r2, "split_half_r": r, "n_players": npl,
                         "team_share_of_systematic": vs["share_team_of_systematic"],
                         "var_player_within_team": vs["var_player_within_team"]}  # fmt: skip
        a = adjust(dev_tab, y, [])[0]
        b = adjust(dev_tab, y, EXTRA)[0]
        pm = dev_tab.with_columns(pl.Series("a", a), pl.Series("b", b)).filter(
            pl.col("def_id").is_in(players)
        )
        pm = pm.group_by("def_id").agg(pl.col("a").mean(), pl.col("b").mean())
        out["corr_player_means_base_vs_extended"] = float(np.corrcoef(pm["a"], pm["b"])[0, 1])
        rep["effect_of_covariates"][y] = out
    out_dir = Path("reports/phase4") / f"{cfg.version}_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"phase_context_{args.defs}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
