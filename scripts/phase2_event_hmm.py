"""B2: matchup HMM with event-dependent switching vs the constant-rho model (hmm_strat_help).

Same train / val subsets as phase2_matchups.py (seed 9: 20,000 train and 5,000 val half-court
possessions). Both models have the stratified emissions with the help state; the constant model
is the event model with only the intercept free, so the comparison is nested.

Covariates of the transition of defender j into step t (ghost.matchup.hmm_event): screen
contact within 1 s; j is the screened defender or the screener's defender of that screen;
completed pass / handoff in flight or caught within 1 s; drive (the same handler gains >= 8 ft
toward the rim within 2 s). Screens and passes from nbacore (offence_match), as in
phase2_screen_events.py.

Reports (no annotation needed): val log-likelihood per observation, fitted switch
probabilities per event pattern, reassignments per possession on the val subset (Viterbi
changes; man-to-man reassignments holding >= 1 s = switch B) and the share of switch-B changes
within 1 s of a screen. Accuracy against the annotation is added after it is complete.

Output: reports/phase2/<version>_<game_set>/event_hmm.json

Usage:
    uv run python scripts/phase2_event_hmm.py [--game-set all] [--fit-max 20000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ghost import data as D  # noqa: E402
from ghost.matchup import hmm as H  # noqa: E402
from ghost.matchup import hmm_event as E  # noqa: E402
from ghost.matchup.events import HELP, EventConfig, stable_path  # noqa: E402
from phase2_matchups import MODELS, _load  # noqa: E402


def codes_for(pt, cfg: D.DataConfig) -> np.ndarray:
    """(N, 5d, T) pattern codes for the possessions of ``pt`` (events loaded game by game)."""
    N, T = pt.valid.shape
    code = np.zeros((N, 5, T), dtype=np.uint8)
    hoop = H.HOOP_LEFT.astype(np.float64)
    for gid in np.unique(pt.game_id):
        sc = D.events(gid, "screen_candidate", cfg).with_columns(
            pl.coalesce("user_def_id", "screened_def_id").alias("screened_def_id")
        )
        ps = D.events(gid, ["pass", "handoff"], cfg).filter(pl.col("completed"))
        sc_t = sc["t_contact_ms"].to_numpy()
        sc_rows = list(
            zip(
                sc_t,
                sc["screened_def_id"].fill_null(-1).to_numpy(),
                sc["screener_def_id"].fill_null(-1).to_numpy(),
                strict=True,
            )
        )
        ps_r, ps_c = ps["t_release_ms"].to_numpy(), ps["t_catch_ms"].to_numpy()
        for i in np.flatnonzero(pt.game_id == gid):
            tv = pt.t[i][pt.valid[i]]
            if tv.size == 0:
                continue
            lo, hi = tv.min() - 1000, tv.max() + 1000
            scr = [r for r, x in zip(sc_rows, sc_t, strict=True) if lo <= x <= hi]
            m = (ps_c >= lo) & (ps_r <= hi)
            code[i] = E.event_codes(
                pt.t[i],
                pt.valid[i],
                pt.handler[i],
                pt.off_xy[i],
                pt.def_ids[i],
                scr,
                list(zip(ps_r[m], ps_c[m], strict=True)),
                hoop,
            )
    return code


def reassignments(inp: H.MatchupInputs, path: np.ndarray, code: np.ndarray) -> dict:
    """Viterbi changes and switch-B (man to man, holding >= 1 s) per possession."""
    hold = EventConfig().steps(1.0)
    N = path.shape[0]
    pv = np.zeros_like(inp.valid)
    pv[:, 1:] = inp.valid[:, 1:] & inp.valid[:, :-1]
    changes = (path[:, :, 1:] != path[:, :, :-1]) & pv[:, None, 1:]
    n_b = n_b_screen = 0
    for i in range(N):
        s = stable_path(path[i], inp.valid[i], hold)
        a, b = s[:, :-1], s[:, 1:]
        sw = (a >= 0) & (b >= 0) & (a != b) & (a != HELP) & (b != HELP)
        n_b += int(sw.sum())
        n_b_screen += int((sw & ((code[i, :, 1:] & 1) > 0)).sum())
    return {
        "viterbi_changes_per_possession": float(changes.sum() / N),
        "switch_b_per_possession": n_b / N,
        "switch_b_share_within_1s_of_screen": n_b_screen / max(n_b, 1),
    }


def pattern_table(code: np.ndarray, valid: np.ndarray, rho_by: dict[str, np.ndarray]) -> list:
    pv = np.zeros_like(valid)
    pv[:, 1:] = valid[:, 1:] & valid[:, :-1]
    counts = np.bincount(code[np.repeat(pv[:, None, :], 5, 1)], minlength=E.N_PATTERNS)
    rows = []
    for c in range(E.N_PATTERNS):
        if counts[c] == 0:
            continue
        on = [n for k, n in enumerate(E.COVARIATES) if c >> k & 1]
        rows.append(
            {
                "pattern": "+".join(on) or "none",
                "share_of_transitions": float(counts[c] / counts.sum()),
                **{f"switch_prob_per_step_{m}": float(1 - r[c]) for m, r in rho_by.items()},
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-set", default=None)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--fit-max", type=int, default=20000)
    ap.add_argument("--val-max", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    cfg = D.DataConfig.load(game_set=args.game_set)
    base = cfg.processed_dir / cfg.game_set
    P = pl.read_parquet(base / "possessions.parquet").filter(~pl.col("is_transition"))
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    def sample(split: str, cap: int) -> pl.DataFrame:  # identical draws to phase2_matchups.py
        S = P.filter(pl.col("split") == split)
        if S.height > cap:
            S = S[np.sort(rng.choice(S.height, size=cap, replace=False))]
        return S

    Ptr, Pva = sample("train", args.fit_max), sample("val", args.val_max)
    data = {}
    for name, S in (("train", Ptr), ("val", Pva)):
        pt, _ = _load(base, S, sorted(S["game_id"].unique().to_list()))
        data[name] = (H.MatchupInputs.from_tensors(pt), codes_for(pt, cfg))
        del pt
    print(f"loaded {Ptr.height} train / {Pva.height} val in {time.time() - t0:.0f}s", flush=True)
    inp_tr, code_tr = data["train"]
    inp_va, code_va = data["val"]
    n_tr, n_va = int(inp_tr.valid.sum()) * 5, int(inp_va.valid.sum()) * 5

    k = 1 + len(E.COVARIATES)
    specs = {"constant_rho": (True,) + (False,) * (k - 1), "event_dependent": (True,) * k}
    report = {
        "nbacore_version": cfg.version,
        "game_set": cfg.game_set,
        "emissions": "stratified (strong / weak side) with help state, as hmm_strat_help",
        "covariates": {
            "screen": "screen contact (nbacore screen_candidate, offence_match) within 1 s",
            "involved": "defender is the screened defender or the screener's defender",
            "pass": "completed pass / handoff from release to catch + 1 s",
            "drive": "same ball handler gains >= 8 ft toward the rim within 2 s",
        },
        "n_fit_train": Ptr.height,
        "n_eval_val": Pva.height,
        "models": {},
    }
    rho_by = {}
    for name, free in specs.items():
        t1 = time.time()
        p0 = E.EventHMMParams(emis=H.HMMParams.initial(**MODELS["hmm_strat_help"]), free=free)
        fit = E.fit_em(inp_tr, code_tr, p0, max_iter=args.max_iter)
        rho_by[name] = E.rho_of_beta(fit.params.beta)
        path = E.paths(inp_va, code_va, fit.params)
        report["models"][name] = {
            "params": fit.params.to_dict(),
            "em_iterations": fit.n_iter,
            "converged": fit.converged,
            "train_loglik_per_obs": fit.loglik[-1] / n_tr,
            "train_loglik": fit.loglik[-1],
            "val_loglik_per_obs": E.loglik(inp_va, code_va, fit.params) / n_va,
            "val": reassignments(inp_va, path, code_va),
            "fit_seconds": round(time.time() - t1, 1),
        }
        print(f"{name}: {json.dumps(report['models'][name])}", flush=True)
    m = report["models"]
    report["lr_statistic_train"] = 2 * (
        m["event_dependent"]["train_loglik"] - m["constant_rho"]["train_loglik"]
    )
    report["val_loglik_gain_per_obs"] = (
        m["event_dependent"]["val_loglik_per_obs"] - m["constant_rho"]["val_loglik_per_obs"]
    )
    report["patterns_val"] = pattern_table(code_va, inp_va.valid, rho_by)
    report["total_seconds"] = round(time.time() - t0, 1)
    rep = Path("reports/phase2") / f"{cfg.version}_{cfg.game_set}"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "event_hmm.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
