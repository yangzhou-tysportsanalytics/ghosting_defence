"""Matchup HMM with event-dependent switching (B2).

Same emissions as :mod:`ghost.matchup.hmm` (stratified, optional help state). The transition of
defender j from step t-1 to t keeps the sticky form

    P(stay) = rho_jt,   P(move to any other state) = (1 - rho_jt) / (K - 1),

but the switch probability depends on observed events around step t:

    logit(1 - rho_jt) = b0 + b1 screen_t + b2 involved_jt + b3 pass_t + b4 drive_t.

The covariates are binary, so each (chain, step) carries a pattern code (bit i = covariate i)
and rho is a lookup per pattern. The uniform off-diagonal makes each forward / backward step
O(K) instead of O(K^2). M-step: emissions as in the constant model (expected sufficient
statistics); b by Newton-Raphson on the expected switch / stay counts aggregated per pattern
(a weighted logistic regression with fractional outcomes). With only b0 it reduces to the
constant-rho model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ghost.matchup import hmm as H

COVARIATES = ("screen", "involved", "pass", "drive")
N_PATTERNS = 1 << len(COVARIATES)
RHO_MIN, RHO_MAX = 0.5, 0.99999


@dataclass
class CovariateConfig:
    screen_window_ms: int = 1000  # |t - t_contact| <= window: screen (and involved) transitions
    pass_after_ms: int = 1000  # from release to catch + this: pass transitions
    drive_steps: int = 10  # 2 s at 5 Hz
    drive_gain_ft: float = 8.0  # same handler gains >= this toward the rim (annotation sample)


def event_codes(
    t: np.ndarray,
    valid: np.ndarray,
    handler: np.ndarray,
    off_xy: np.ndarray,
    def_ids: np.ndarray,
    screens: list[tuple[int, int, int]],
    passes: list[tuple[int, int]],
    hoop: np.ndarray,
    cfg: CovariateConfig | None = None,
) -> np.ndarray:
    """Pattern codes (5d, T) uint8 of one possession.

    ``screens``: (t_contact_ms, screened_def_id, screener_def_id); ``passes``: (t_release_ms,
    t_catch_ms) of completed passes / handoffs. Bit 0 screen, 1 defender involved in that screen
    (screened or screener's defender), 2 pass, 3 drive.
    """
    cfg = cfg or CovariateConfig()
    T = t.shape[0]
    code = np.zeros((5, T), dtype=np.uint8)
    for tc, d1, d2 in screens:
        near = valid & (np.abs(t - tc) <= cfg.screen_window_ms)
        code[:, near] |= 1
        for r in np.flatnonzero(np.isin(def_ids, [d for d in (d1, d2) if d >= 0])):
            code[r, near] |= 2
    for tr, tcat in passes:
        code[:, valid & (t >= tr) & (t <= tcat + cfg.pass_after_ms)] |= 4
    k = cfg.drive_steps
    hv = (handler >= 0) & valid
    xy = off_xy[np.arange(T), np.clip(handler, 0, 4)]
    dh = np.linalg.norm(xy - hoop, axis=-1)
    for s in range(T - k):
        if (
            hv[s]
            and hv[s + k]
            and handler[s] == handler[s + k]
            and dh[s] - dh[s + k] >= (cfg.drive_gain_ft)
        ):
            code[:, s + 1 : s + k + 1] |= 8
    return code


def design() -> np.ndarray:
    """(N_PATTERNS, 1 + n_cov) design matrix: intercept and the bits of each pattern code."""
    codes = np.arange(N_PATTERNS)
    bits = (codes[:, None] >> np.arange(len(COVARIATES))) & 1
    return np.concatenate([np.ones((N_PATTERNS, 1)), bits], axis=1).astype(np.float64)


def rho_of_beta(beta: np.ndarray) -> np.ndarray:
    """P(stay) per pattern code."""
    sw = 1.0 / (1.0 + np.exp(-(design() @ beta)))
    return np.clip(1.0 - sw, RHO_MIN, RHO_MAX)


def beta_constant(rho: float) -> np.ndarray:
    b = np.zeros(1 + len(COVARIATES))
    b[0] = np.log((1.0 - rho) / rho)
    return b


@dataclass
class EventHMMParams:
    emis: H.HMMParams
    beta: np.ndarray = field(default_factory=lambda: beta_constant(0.98))
    free: tuple[bool, ...] = (True,) * (1 + len(COVARIATES))  # False: coefficient fixed at 0

    def to_dict(self) -> dict:
        d = self.emis.to_dict()
        d.pop("rho")
        d["beta"] = dict(
            zip(("intercept", *COVARIATES), [round(float(b), 4) for b in self.beta], strict=True)
        )
        return d


def forward_backward_tv(
    logB: np.ndarray, code: np.ndarray, rho: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """logB (C, T, K); code (C, T) pattern of the transition into step t (t = 0 unused);
    rho (N_PATTERNS,). Returns (log posteriors (C, T, K), log-likelihood per chain (C,),
    expected stay probability of each transition (C, T); column 0 is 0)."""
    C, T, K = logB.shape
    lr = np.log(rho)[code]  # (C, T)
    lq = np.log((1.0 - rho) / (K - 1))[code]

    def step(prev: np.ndarray, t: int) -> np.ndarray:
        # log( rho * a_k + q * (sum(a) - a_k) ) for log-vector prev (C, K)
        m = prev.max(axis=1, keepdims=True)
        a = np.exp(prev - m)
        s = a.sum(axis=1, keepdims=True)
        r, q = np.exp(lr[:, t : t + 1]), np.exp(lq[:, t : t + 1])
        return np.log(r * a + q * (s - a)) + m

    la = np.empty_like(logB)
    la[:, 0] = -np.log(K) + logB[:, 0]
    for t in range(1, T):
        la[:, t] = logB[:, t] + step(la[:, t - 1], t)
    lb = np.zeros_like(logB)
    for t in range(T - 2, -1, -1):
        lb[:, t] = step(logB[:, t + 1] + lb[:, t + 1], t + 1)  # A is symmetric
    ll = H._lse(la[:, -1], axis=1)
    post = la + lb - ll[:, None, None]
    stay = np.zeros((C, T))
    stay[:, 1:] = np.exp(
        la[:, :-1] + lr[:, 1:, None] + logB[:, 1:] + lb[:, 1:] - ll[:, None, None]
    ).sum(-1)
    return post, ll, stay


def viterbi_tv(logB: np.ndarray, code: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """MAP path (C, T) int8 under the time-varying sticky transitions."""
    C, T, K = logB.shape
    lr = np.log(rho)[code]
    lq = np.log((1.0 - rho) / (K - 1))[code]
    delta = -np.log(K) + logB[:, 0]
    back = np.empty((C, T, K), dtype=np.int8)
    idx = np.arange(K)
    for t in range(1, T):
        # best "other" predecessor for each target: argmax over j != k
        order = np.argsort(-delta, axis=1)
        best, second = order[:, 0], order[:, 1]
        other = np.where(idx[None, :] == best[:, None], second[:, None], best[:, None])
        v_other = np.take_along_axis(delta, other, 1) + lq[:, t : t + 1]
        v_stay = delta + lr[:, t : t + 1]
        use_stay = v_stay >= v_other
        back[:, t] = np.where(use_stay, idx[None, :], other)
        delta = np.where(use_stay, v_stay, v_other) + logB[:, t]
    path = np.empty((C, T), dtype=np.int8)
    path[:, -1] = delta.argmax(axis=1)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = np.take_along_axis(back[:, t], path[:, t : t + 1].astype(np.int64), 1)[
            :, 0
        ]
    return path


def _pairs(valid: np.ndarray, D: int) -> np.ndarray:
    """(N*D, T) True where the transition into step t joins two valid steps."""
    pv = np.zeros_like(valid)
    pv[:, 1:] = valid[:, 1:] & valid[:, :-1]
    return np.repeat(pv, D, axis=0)


def e_step_chunk(inp: H.MatchupInputs, code: np.ndarray, p: EventHMMParams):
    """Posteriors (N, 5d, T, K), total log-likelihood, and per-pattern (stay, pairs) sums."""
    logB = H.log_emissions(inp, p.emis)
    N, D, T, K = logB.shape
    c = code.reshape(N * D, T)
    post, ll, stay = forward_backward_tv(H._chains(logB), c, rho_of_beta(p.beta))
    pv = _pairs(inp.valid, D)
    s_stay = np.bincount(c[pv], weights=stay[pv], minlength=N_PATTERNS)
    s_pairs = np.bincount(c[pv], minlength=N_PATTERNS).astype(np.float64)
    return np.exp(post).reshape(N, D, T, K), float(ll.sum()), s_stay, s_pairs


def update_beta(
    s_stay: np.ndarray, s_pairs: np.ndarray, beta: np.ndarray, free: tuple[bool, ...]
) -> np.ndarray:
    """Newton-Raphson for the logistic switch model on aggregated expected counts."""
    X = design()[:, np.asarray(free)]
    e = s_pairs - s_stay  # expected switches per pattern
    b = beta[np.asarray(free)].copy()

    def objective(b: np.ndarray) -> float:  # expected complete-data log-likelihood
        eta = X @ b
        return float((e * -np.logaddexp(0, -eta) + (s_pairs - e) * -np.logaddexp(0, eta)).sum())

    f = objective(b)
    for _ in range(100):
        pi = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -50, 50)))
        g = X.T @ (e - s_pairs * pi)
        Hm = (X * (s_pairs * pi * (1 - pi))[:, None]).T @ X + 1e-6 * np.eye(len(b))
        d = np.linalg.solve(Hm, g)
        step = 1.0
        while step > 1e-6 and objective(b + step * d) < f - 1e-12 * abs(f):
            step /= 2  # damped Newton: never decrease the objective
        b += step * d
        f = objective(b)
        if np.abs(step * d).max() < 1e-9:
            break
    out = np.zeros_like(beta)
    out[np.asarray(free)] = b
    return out


@dataclass
class EventFitResult:
    params: EventHMMParams
    loglik: list[float]
    n_iter: int
    converged: bool


def fit_em(
    inp: H.MatchupInputs,
    code: np.ndarray,
    params: EventHMMParams,
    max_iter: int = 40,
    tol: float = 1e-5,
    chunk: int = 1500,
    verbose: bool = False,
) -> EventFitResult:
    """Baum-Welch; ``code`` (N, 5d, T) uint8 pattern of each defender's transition into t."""
    p = params
    lls: list[float] = []
    converged = False
    N = inp.valid.shape[0]
    for it in range(max_iter):
        st = H._new_stats(p.emis)
        s_stay = np.zeros(N_PATTERNS)
        s_pairs = np.zeros(N_PATTERNS)
        total = 0.0
        for a in range(0, N, chunk):
            sl = slice(a, a + chunk)
            ci = H._chunk(inp, sl)
            post, ll, ss, sp = e_step_chunk(ci, code[sl], p)
            total += ll
            s_stay += ss
            s_pairs += sp
            H.accumulate_stats(st, ci, post, np.zeros(1), p.emis)
            del post
        lls.append(total)
        if verbose:
            print(f"iter {it}: loglik {total:.1f} {p.to_dict()}", flush=True)
        p = EventHMMParams(
            emis=H.update_params(st, p.emis),
            beta=update_beta(s_stay, s_pairs, p.beta, p.free),
            free=p.free,
        )
        if it > 0 and abs(lls[-1] - lls[-2]) < tol * abs(lls[-2]):
            converged = True
            break
    return EventFitResult(params=p, loglik=lls, n_iter=len(lls), converged=converged)


def loglik(inp: H.MatchupInputs, code: np.ndarray, p: EventHMMParams, chunk: int = 1500) -> float:
    total = 0.0
    for a in range(0, inp.valid.shape[0], chunk):
        sl = slice(a, a + chunk)
        total += e_step_chunk(H._chunk(inp, sl), code[sl], p)[1]
    return total


def paths(inp: H.MatchupInputs, code: np.ndarray, p: EventHMMParams, chunk: int = 1500):
    """Viterbi paths (N, 5d, T) int8."""
    N, T = inp.valid.shape
    out = np.empty((N, 5, T), dtype=np.int8)
    rho = rho_of_beta(p.beta)
    for a in range(0, N, chunk):
        sl = slice(a, a + chunk)
        logB = H.log_emissions(H._chunk(inp, sl), p.emis)
        n = logB.shape[0]
        out[sl] = viterbi_tv(H._chains(logB), code[sl].reshape(n * 5, T), rho).reshape(n, 5, T)
    return out
