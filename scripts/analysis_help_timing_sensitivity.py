"""Sensitivity of the early-help comparisons to unmeasured confounding (writing request A3).

For the four comparisons of analysis_help_timing_matched.py (A: early help vs rest, B: early vs
late help; outcomes points and xFG of the shot), two summaries of how strong an unmeasured
confounder would have to be:

* E-value (VanderWeele & Ding 2017) of the overlap-weighted estimate and of the interval bound
  nearer the null. Continuous outcome: the standardised difference d = estimate / SD(outcome) is
  converted with RR = exp(0.91 d); E = RR + sqrt(RR (RR - 1)) (RR taken >= 1).
* Omitted-variable bounds (Cinelli & Hazlett 2020) for the regression-adjusted estimate (OLS of
  the outcome on treatment and the same design as the propensity model: standardised covariates,
  their deciles, team indicators). Robustness values RV (share of residual variance of both
  treatment and outcome a confounder must explain to move the estimate to zero; RV_a: to make
  it insignificant at 5 %), and bounds for a confounder k = 1, 2, 3 times as strong as an
  observed covariate group (each covariate with its decile terms): the handler's velocity
  toward the rim (named in the request), and the covariate with the largest partial R2 with
  the treatment.
Classical OLS standard errors as in Cinelli & Hazlett (games as clusters are not reflected; the
matched analysis's game-bootstrap intervals are the reference for uncertainty).

Output: reports/analysis/help_timing_sensitivity.json

Usage:
    uv run python scripts/analysis_help_timing_sensitivity.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_help_timing_matched import (  # noqa: E402
    NUM,
    analysis_frame,
    comparisons,
    design,
    outcome_samples,
)

Z = 1.959963984540054  # normal 97.5 % quantile (df in the tens of thousands)


def e_value(est: float, sd: float) -> float:
    rr = math.exp(0.91 * abs(est) / sd)
    return rr + math.sqrt(rr * (rr - 1))


def resid(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]


def rv(f: float, df: int, alpha: float | None = None) -> float:
    """Robustness value for q = 1 (Cinelli & Hazlett 2020, eq. 18-19)."""
    if alpha is not None:
        f = f - Z / math.sqrt(df - 1)
        if f <= 0:
            return 0.0
    return 0.5 * (math.sqrt(f**4 + 4 * f**2) - f**2)


def group_columns(n_lin: int, X: np.ndarray) -> dict[str, list[int]]:
    """Column indices of each covariate (linear term + its decile indicators) in design()."""
    groups, pos = {c: [j] for j, c in enumerate(NUM[:n_lin])}, n_lin
    for j, c in enumerate(NUM[:n_lin]):
        # design(): deciles of covariate j follow in order; count them from the same edges
        lin = X[:, j]
        k = len(np.unique(np.quantile(lin, np.linspace(0.1, 0.9, 9))))
        groups[c] += list(range(pos, pos + k))
        pos += k
    return groups


def ovb(X: np.ndarray, t: np.ndarray, y: np.ndarray) -> dict:
    n = len(y)
    A = np.column_stack([np.ones(n), X])
    rank = int(np.linalg.matrix_rank(A))
    rt, ry = resid(A, t), resid(A, y)
    beta = float(rt @ ry / (rt @ rt))
    rss = float(((ry - beta * rt) ** 2).sum())
    df = n - rank - 1
    se = math.sqrt(rss / df / float(rt @ rt))
    tstat = beta / se
    r2yd = tstat**2 / (tstat**2 + df)
    f = abs(tstat) / math.sqrt(df)
    out = {"estimate_ols_adjusted": beta, "se_classical": se, "t": tstat, "df": df,
           "partial_r2_outcome_treatment": r2yd,
           "rv_q1": rv(f, df), "rv_q1_alpha05": rv(f, df, 0.05)}  # fmt: skip
    # benchmark groups: partial R2 of each covariate group with the treatment and the outcome
    groups = group_columns(len(NUM), X)
    rss_t_full = float(rt @ rt)
    ryd_full = rss  # residual of y on [1, t, X]
    bench = {}
    for name, cols in groups.items():
        keep = np.setdiff1d(np.arange(X.shape[1]), cols)
        Ar = np.column_stack([np.ones(n), X[:, keep]])
        rt_r = resid(Ar, t)
        r2dx = 1 - rss_t_full / float(rt_r @ rt_r)
        ry_r = resid(np.column_stack([Ar, t]), y)
        r2yx = 1 - ryd_full / float(ry_r @ ry_r)
        bench[name] = (max(r2dx, 0.0), max(r2yx, 0.0))
    strongest = max(bench, key=lambda c: bench[c][0])
    out["benchmark_partial_r2"] = {
        c: {"with_treatment": round(v[0], 5), "with_outcome": round(v[1], 5)}
        for c, v in sorted(bench.items(), key=lambda kv: -kv[1][0])
    }
    out["bounds"] = {}
    for name in dict.fromkeys(["h_v_to_rim", strongest]):
        r2dxj, r2yxj = bench[name]
        rows = []
        for k in (1, 2, 3):
            if k * r2dxj >= 1:
                rows.append({"k": k, "note": "confounder would explain all treatment variance"})
                continue
            r2dz = k * r2dxj / (1 - r2dxj)
            r2zxj = k * r2dxj**2 / ((1 - k * r2dxj) * (1 - r2dxj))
            if r2zxj >= 1:  # as sensemakr: no confounder of that strength exists
                rows.append({"k": k, "note": "bound undefined (implied R2 of Z with X_j >= 1)"})
                continue
            r2yz = min(((math.sqrt(k) + math.sqrt(r2zxj)) / math.sqrt(1 - r2zxj)) ** 2
                       * (r2yxj / (1 - r2yxj)), 1.0)  # fmt: skip
            if r2dz >= 1:
                rows.append({"k": k, "note": "bound undefined (R2 with treatment >= 1)"})
                continue
            bias = se * math.sqrt(df) * math.sqrt(r2yz * r2dz / (1 - r2dz))
            adj = beta - math.copysign(bias, beta)
            adj_se = se * math.sqrt((1 - r2yz) / (1 - r2dz)) * math.sqrt(df / (df - 1))
            rows.append({"k": k, "r2_confounder_treatment": r2dz, "r2_confounder_outcome": r2yz,
                         "adjusted_estimate": adj,
                         "adjusted_ci95": [adj - Z * adj_se, adj + Z * adj_se]})  # fmt: skip
        out["bounds"][name] = rows
    out["strongest_observed_covariate"] = strongest
    return out


def main() -> None:
    df = analysis_frame()
    matched = json.loads(Path("reports/analysis/help_timing_matched.json").read_text())
    rep: dict = {
        "method": "E-values (VanderWeele & Ding 2017; continuous outcome via RR = exp(0.91 d)) for "
        "the overlap-weighted estimates; omitted-variable bounds and robustness values (Cinelli & "
        "Hazlett 2020) for the OLS-adjusted estimate with the propensity design",
        "comparisons": {},
    }
    for comp, sub in comparisons(df):
        rep["comparisons"][comp] = {}
        for yname, d, y in outcome_samples(sub):
            X, _ = design(d)
            t = d["early"].to_numpy().astype(float)
            m = matched["comparisons"][comp][yname]
            est, (lo, hi) = m["ato_overlap_weighted"], m["ato_ci95_game_bootstrap"]
            sd = float(y.std())
            near = 0.0 if lo <= 0 <= hi else (lo if est > 0 else hi)
            res = {
                "n": len(y),
                "ato_overlap_weighted": est,
                "ato_ci95_game_bootstrap": [lo, hi],
                "sd_outcome": sd,
                "standardised_difference": est / sd,
                "e_value_estimate": e_value(est, sd),
                "e_value_ci_bound": 1.0 if near == 0.0 else e_value(near, sd),
            }
            res.update(ovb(X, t, y))
            rep["comparisons"][comp][yname] = res
            b = res["bounds"].get("h_v_to_rim", [])
            print(f"{comp} {yname}: ATO {est:+.4f} E {res['e_value_estimate']:.2f} "
                  f"(CI {res['e_value_ci_bound']:.2f}); OLS {res['estimate_ols_adjusted']:+.4f} "
                  f"RV {res['rv_q1']:.3f} RVa {res['rv_q1_alpha05']:.3f}; strongest "
                  f"{res['strongest_observed_covariate']}; 1x h_v_to_rim -> "
                  f"{b[0].get('adjusted_estimate', float('nan')):+.4f}", flush=True)  # fmt: skip
    p = Path("reports/analysis/help_timing_sensitivity.json")
    p.write_text(json.dumps(rep, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
