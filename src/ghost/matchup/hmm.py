"""Matchup inference: min-residual baseline and the Franks et al. (2015) HMM.

Each defender j follows an independent hidden Markov chain whose state is the offensive player
(slot 0-4) he is guarding; an optional extra state ``HELP`` (index 5) means "not attached to a
man" (help / zone). Two defenders may guard the same attacker (double team).

Emission (isotropic Gaussian):  z_jt ~ N(mu_kt, sigma_g^2 I)
    man state k:   mu_kt = g_o * O_kt + g_b * B_t + g_h * H,   g_o + g_b + g_h = 1
    help state:    mu_t  = lam * B_t + (1 - lam) * H
``H`` is the hoop (left half after the Phase 1 flip), ``B`` the ball. Parameters can be stratified
by a group ``g`` of the (step, state) pair: ``"strong"`` when the attacker is on the ball side of
the court's long axis (y = 25), ``"weak"`` otherwise.

Transitions: sticky, P(stay) = rho, P(move to any other state) = (1 - rho) / (K - 1).

EM (Baum-Welch): E-step forward-backward in log space, vectorised over all (possession,
defender) chains; M-step closed-form weighted least squares for the convex weights (with the
sum-to-one constraint substituted), weighted mean squared residual for sigma^2, expected
self-transition share for rho. Invalid (padded) steps carry no observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ghost.court import HOOP_LEFT

FRANKS_GAMMA = (0.62, 0.11, 0.27)  # (offensive player, ball, hoop); Franks et al. 2015
LOG2PI = float(np.log(2 * np.pi))
GROUPS = ("all", "strong", "weak")


@dataclass
class HMMParams:
    gamma: dict[str, np.ndarray]  # group -> (g_o, g_b, g_h)
    sigma2: dict[str, float]  # group -> variance per axis (ft^2)
    rho: float = 0.95
    help_lam: float = 0.5
    help_sigma2: float = 25.0
    stratify: bool = False
    help_state: bool = False

    @classmethod
    def initial(cls, stratify: bool = False, help_state: bool = False) -> HMMParams:
        groups = ("strong", "weak") if stratify else ("all",)
        return cls(
            gamma={g: np.array(FRANKS_GAMMA, dtype=np.float64) for g in groups},
            sigma2={g: 9.0 for g in groups},
            stratify=stratify,
            help_state=help_state,
        )

    @property
    def n_states(self) -> int:
        return 6 if self.help_state else 5

    def to_dict(self) -> dict:
        return {
            "gamma": {g: [round(float(v), 4) for v in x] for g, x in self.gamma.items()},
            "sigma_ft": {g: round(float(np.sqrt(s)), 3) for g, s in self.sigma2.items()},
            "rho": round(float(self.rho), 5),
            "help_lam": round(float(self.help_lam), 4) if self.help_state else None,
            "help_sigma_ft": round(float(np.sqrt(self.help_sigma2)), 3)
            if self.help_state
            else None,
            "stratify": self.stratify,
            "help_state": self.help_state,
        }


@dataclass
class MatchupInputs:
    """Arrays for N possessions, T steps. ``d`` defenders are flattened into chains later."""

    off_xy: np.ndarray  # (N, T, 5, 2)
    def_xy: np.ndarray  # (N, T, 5, 2)
    ball: np.ndarray  # (N, T, 2)  (x, y)
    valid: np.ndarray  # (N, T) bool
    hoop: np.ndarray = field(default_factory=lambda: HOOP_LEFT.astype(np.float64))

    @classmethod
    def from_tensors(cls, pt) -> MatchupInputs:
        return cls(
            off_xy=pt.off_xy.astype(np.float64),
            def_xy=pt.def_xy.astype(np.float64),
            ball=pt.ball[..., :2].astype(np.float64),
            valid=pt.valid.astype(bool),
        )

    def side_group(self) -> np.ndarray:
        """(N, T, 5) True where attacker k is on the ball's side of y = 25 ("strong")."""
        by = self.ball[..., 1][..., None] - 25.0
        oy = self.off_xy[..., 1] - 25.0
        return np.sign(by) == np.sign(oy)


# ------------------------------------------------------------------------------------------
# emission means and log-likelihoods
# ------------------------------------------------------------------------------------------


def man_means(inp: MatchupInputs, gamma: np.ndarray) -> np.ndarray:
    """(N, T, 5, 2) means for the five man states with weights ``gamma`` (g_o, g_b, g_h)."""
    g_o, g_b, g_h = gamma
    return g_o * inp.off_xy + g_b * inp.ball[:, :, None, :] + g_h * inp.hoop


def _gamma_per_state(inp: MatchupInputs, p: HMMParams) -> tuple[np.ndarray, np.ndarray]:
    """Per (N, T, 5) gamma triple and sigma^2 according to the stratification."""
    N, T = inp.valid.shape
    if not p.stratify:
        g = np.broadcast_to(p.gamma["all"], (N, T, 5, 3))
        s2 = np.full((N, T, 5), p.sigma2["all"])
        return g, s2
    strong = inp.side_group()
    g = np.where(strong[..., None], p.gamma["strong"], p.gamma["weak"])
    s2 = np.where(strong, p.sigma2["strong"], p.sigma2["weak"])
    return g, s2


def log_emissions(inp: MatchupInputs, p: HMMParams) -> np.ndarray:
    """(N, 5 defenders, T, K) log-likelihood of each defender's position under each state."""
    g, s2 = _gamma_per_state(inp, p)
    mu = (
        g[..., 0:1] * inp.off_xy + g[..., 1:2] * inp.ball[:, :, None, :] + g[..., 2:3] * inp.hoop
    )  # (N, T, 5 states, 2)
    z = inp.def_xy  # (N, T, 5 defenders, 2)
    d2 = ((z[:, :, :, None, :] - mu[:, :, None, :, :]) ** 2).sum(-1)  # (N, T, 5d, 5k)
    ll = -0.5 * d2 / s2[:, :, None, :] - np.log(s2)[:, :, None, :] - LOG2PI
    if p.help_state:
        mh = p.help_lam * inp.ball + (1 - p.help_lam) * inp.hoop  # (N, T, 2)
        dh = ((z - mh[:, :, None, :]) ** 2).sum(-1)  # (N, T, 5d)
        llh = -0.5 * dh / p.help_sigma2 - np.log(p.help_sigma2) - LOG2PI
        ll = np.concatenate([ll, llh[..., None]], axis=-1)
    ll = np.where(inp.valid[:, :, None, None], ll, 0.0)
    return np.transpose(ll, (0, 2, 1, 3))  # (N, 5d, T, K)


# ------------------------------------------------------------------------------------------
# forward-backward / Viterbi (chains flattened to (C, T, K))
# ------------------------------------------------------------------------------------------


def _log_trans(rho: float, K: int) -> np.ndarray:
    off = (1.0 - rho) / (K - 1)
    A = np.full((K, K), off)
    np.fill_diagonal(A, rho)
    return np.log(A)


def _lse(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.exp(a - m).sum(axis=axis))


def forward_backward(
    logB: np.ndarray, logA: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """logB (C, T, K). Returns (log posteriors (C,T,K), log-likelihood per chain (C,),
    expected self-transition counts per chain (C,))."""
    C, T, K = logB.shape
    logpi = -np.log(K)
    la = np.empty_like(logB)
    la[:, 0] = logpi + logB[:, 0]
    for t in range(1, T):
        la[:, t] = logB[:, t] + _lse(la[:, t - 1][:, :, None] + logA[None], axis=1)
    lb = np.zeros_like(logB)
    for t in range(T - 2, -1, -1):
        lb[:, t] = _lse(logA[None] + (logB[:, t + 1] + lb[:, t + 1])[:, None, :], axis=2)
    ll = _lse(la[:, -1], axis=1)
    post = la + lb - ll[:, None, None]
    # expected number of self transitions: sum_t sum_i xi_t(i, i)
    diag = np.diag(logA)
    xi_self = la[:, :-1] + diag + logB[:, 1:] + lb[:, 1:] - ll[:, None, None]
    self_counts = np.exp(xi_self).sum(axis=(1, 2))
    return post, ll, self_counts


def viterbi(logB: np.ndarray, logA: np.ndarray) -> np.ndarray:
    """MAP state path (C, T) int8."""
    C, T, K = logB.shape
    delta = -np.log(K) + logB[:, 0]
    back = np.empty((C, T, K), dtype=np.int8)
    for t in range(1, T):
        cand = delta[:, :, None] + logA[None]  # (C, from, to)
        back[:, t] = cand.argmax(axis=1)
        delta = cand.max(axis=1) + logB[:, t]
    path = np.empty((C, T), dtype=np.int8)
    path[:, -1] = delta.argmax(axis=1)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = np.take_along_axis(back[:, t], path[:, t : t + 1].astype(np.int64), 1)[
            :, 0
        ]
    return path


# ------------------------------------------------------------------------------------------
# M-step via sufficient statistics (so the E-step can run in chunks of possessions)
# ------------------------------------------------------------------------------------------


def _new_stats(p: HMMParams) -> dict:
    groups = ("strong", "weak") if p.stratify else ("all",)
    st = {g: {"xx": np.zeros((2, 2)), "xy": np.zeros(2), "yy": 0.0, "w": 0.0} for g in groups}
    st["_help"] = {"xx": 0.0, "xy": 0.0, "yy": 0.0, "w": 0.0}
    st["_self"] = 0.0
    st["_pairs"] = 0
    return st


def accumulate_stats(
    st: dict, inp: MatchupInputs, post: np.ndarray, self_counts: np.ndarray, p: HMMParams
) -> None:
    """Add one chunk's expected sufficient statistics. post: (N, 5d, T, K) probabilities.

    Regression per axis: y = z - H, X = [O - H, B - H], coef = (g_o, g_b), weights = posterior.
    """
    w = post[..., :5] * inp.valid[:, None, :, None]  # (N, 5d, T, 5k)
    y = (np.transpose(inp.def_xy, (0, 2, 1, 3)) - inp.hoop)[:, :, :, None, :]  # (N,5d,T,1,2)
    xo = (inp.off_xy - inp.hoop)[:, None, :, :, :]  # (N,1,T,5k,2)
    xb = (inp.ball - inp.hoop)[:, None, :, None, :]  # (N,1,T,1,2)
    if p.stratify:
        strong = inp.side_group()[:, None, :, :]  # (N,1,T,5k)
        masks = {"strong": strong, "weak": ~strong}
    else:
        masks = {"all": None}
    for g, m in masks.items():
        wg = w if m is None else w * m
        wg2 = wg[..., None]  # same weight on both axes
        so = float((wg2 * xo * xo).sum())
        sb = float((wg2 * xb * xb).sum())
        sob = float((wg2 * xo * xb).sum())
        st[g]["xx"] += np.array([[so, sob], [sob, sb]])
        st[g]["xy"] += np.array([float((wg2 * xo * y).sum()), float((wg2 * xb * y).sum())])
        st[g]["yy"] += float((wg2 * y * y).sum())
        st[g]["w"] += float(2.0 * wg.sum())  # two axes per observation
    if p.help_state:
        wh = post[..., 5] * inp.valid[:, None, :]  # (N,5d,T)
        yh = np.transpose(inp.def_xy, (0, 2, 1, 3)) - inp.hoop  # (N,5d,T,2)
        xh = (inp.ball - inp.hoop)[:, None, :, :]  # (N,1,T,2)
        whh = wh[..., None]
        st["_help"]["xx"] += float((whh * xh * xh).sum())
        st["_help"]["xy"] += float((whh * xh * yh).sum())
        st["_help"]["yy"] += float((whh * yh * yh).sum())
        st["_help"]["w"] += float(2.0 * wh.sum())
    st["_self"] += float(self_counts.sum())
    st["_pairs"] += int((inp.valid[:, 1:] & inp.valid[:, :-1]).sum()) * 5


def update_params(st: dict, p: HMMParams) -> HMMParams:
    new = HMMParams(
        gamma=dict(p.gamma),
        sigma2=dict(p.sigma2),
        rho=p.rho,
        help_lam=p.help_lam,
        help_sigma2=p.help_sigma2,
        stratify=p.stratify,
        help_state=p.help_state,
    )
    for g in p.gamma:
        s = st[g]
        if s["w"] < 20:
            continue
        coef = np.linalg.solve(s["xx"] + 1e-9 * np.eye(2), s["xy"])
        new.gamma[g] = np.array([coef[0], coef[1], 1.0 - coef[0] - coef[1]])
        rss = s["yy"] - 2.0 * coef @ s["xy"] + coef @ s["xx"] @ coef
        new.sigma2[g] = float(max(rss, 1e-9) / s["w"])
    h = st["_help"]
    if p.help_state and h["w"] > 20:
        lam = float(np.clip(h["xy"] / max(h["xx"], 1e-12), 0.0, 1.0))
        new.help_lam = lam
        rss = h["yy"] - 2 * lam * h["xy"] + lam * lam * h["xx"]
        new.help_sigma2 = float(max(rss, 1e-9) / h["w"])
    new.rho = float(np.clip(st["_self"] / max(st["_pairs"], 1), 0.5, 0.9999))
    return new


def m_step(
    inp: MatchupInputs, post: np.ndarray, self_counts: np.ndarray, p: HMMParams
) -> HMMParams:
    """Single-chunk M-step (kept for tests and small inputs)."""
    st = _new_stats(p)
    accumulate_stats(st, inp, post, self_counts, p)
    return update_params(st, p)


# ------------------------------------------------------------------------------------------
# public API
# ------------------------------------------------------------------------------------------


@dataclass
class FitResult:
    params: HMMParams
    loglik: list[float]
    n_iter: int
    converged: bool


def _chains(logB: np.ndarray) -> np.ndarray:
    N, D, T, K = logB.shape
    return logB.reshape(N * D, T, K)


def _chunk(inp: MatchupInputs, sl: slice) -> MatchupInputs:
    return MatchupInputs(
        off_xy=inp.off_xy[sl],
        def_xy=inp.def_xy[sl],
        ball=inp.ball[sl],
        valid=inp.valid[sl],
        hoop=inp.hoop,
    )


def e_step_chunk(inp: MatchupInputs, p: HMMParams):
    """Posteriors (N,5d,T,K), total log-likelihood and self-transition counts for one chunk."""
    logB = log_emissions(inp, p)
    N, D, T, K = logB.shape
    post, ll, self_counts = forward_backward(_chains(logB), _log_trans(p.rho, K))
    # Padding is always trailing. A transition into an unobserved step has expected
    # self-transition probability exactly rho (the future carries no evidence), so the padded
    # pairs contribute n_pad * rho to self_counts; remove that to count valid pairs only.
    n_pad = np.repeat((~(inp.valid[:, 1:] & inp.valid[:, :-1])).sum(axis=1), D)
    self_counts = self_counts - n_pad * p.rho
    return np.exp(post).reshape(N, D, T, K), float(ll.sum()), self_counts


def fit_em(
    inp: MatchupInputs,
    params: HMMParams | None = None,
    max_iter: int = 50,
    tol: float = 1e-5,
    chunk: int = 1500,
    verbose: bool = False,
) -> FitResult:
    """Baum-Welch with the E-step in chunks of ``chunk`` possessions (bounded memory)."""
    p = params or HMMParams.initial()
    lls: list[float] = []
    converged = False
    N = inp.valid.shape[0]
    for it in range(max_iter):
        st = _new_stats(p)
        total = 0.0
        for a in range(0, N, chunk):
            ci = _chunk(inp, slice(a, a + chunk))
            post, ll, sc = e_step_chunk(ci, p)
            total += ll
            accumulate_stats(st, ci, post, sc, p)
            del post
        lls.append(total)
        if verbose:
            print(f"iter {it}: loglik {total:.1f} {p.to_dict()}", flush=True)
        p = update_params(st, p)
        if it > 0 and abs(lls[-1] - lls[-2]) < tol * abs(lls[-2]):
            converged = True
            break
    return FitResult(params=p, loglik=lls, n_iter=len(lls), converged=converged)


def loglik(inp: MatchupInputs, p: HMMParams, chunk: int = 1500) -> float:
    """Total marginal log-likelihood (e.g. on held-out possessions)."""
    total = 0.0
    for a in range(0, inp.valid.shape[0], chunk):
        _, ll, _ = e_step_chunk(_chunk(inp, slice(a, a + chunk)), p)
        total += ll
    return total


def posteriors(
    inp: MatchupInputs, p: HMMParams, chunk: int = 1500
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior probabilities (N, 5d, T, K) float32 and Viterbi paths (N, 5d, T) int8."""
    N, T = inp.valid.shape
    K = p.n_states
    post_all = np.empty((N, 5, T, K), dtype=np.float32)
    path_all = np.empty((N, 5, T), dtype=np.int8)
    logA = _log_trans(p.rho, K)
    for a in range(0, N, chunk):
        ci = _chunk(inp, slice(a, a + chunk))
        logB = log_emissions(ci, p)
        n = logB.shape[0]
        post, _, _ = forward_backward(_chains(logB), logA)
        post_all[a : a + n] = np.exp(post).reshape(n, 5, T, K)
        path_all[a : a + n] = viterbi(_chains(logB), logA).reshape(n, 5, T)
    return post_all, path_all


def min_residual(
    inp: MatchupInputs, gamma: tuple[float, float, float] = FRANKS_GAMMA
) -> np.ndarray:
    """Baseline: each defender's assignment = argmin_k ||z_d - mu_k|| at every step. (N, 5d, T)."""
    mu = man_means(inp, np.asarray(gamma))  # (N, T, 5k, 2)
    d2 = ((inp.def_xy[:, :, :, None, :] - mu[:, :, None, :, :]) ** 2).sum(-1)  # (N,T,5d,5k)
    a = d2.argmin(-1).astype(np.int8)
    return np.transpose(a, (0, 2, 1))
