"""Matchup HMM and baseline on data simulated from the model itself."""

from __future__ import annotations

import numpy as np
import pytest

from ghost.court import HOOP_LEFT
from ghost.matchup.hmm import (
    FRANKS_GAMMA,
    HMMParams,
    MatchupInputs,
    _log_trans,
    fit_em,
    forward_backward,
    min_residual,
    posteriors,
    viterbi,
)


def simulate(n=120, T=60, gamma=(0.6, 0.15, 0.25), sigma=2.0, rho=0.97, seed=0, n_valid=None):
    rng = np.random.default_rng(seed)
    off = np.zeros((n, T, 5, 2))
    off[:, 0] = rng.uniform([5, 3], [40, 47], size=(n, 5, 2))
    steps = rng.normal(0, 1.2, size=(n, T - 1, 5, 2))
    off[:, 1:] = off[:, :1] + np.cumsum(steps, axis=1)
    off = np.clip(off, [0, 0], [47, 50])
    handler = rng.integers(0, 5, size=n)
    ball = off[np.arange(n), :, handler] + rng.normal(0, 0.5, size=(n, T, 2))
    state = np.zeros((n, 5, T), dtype=int)
    state[:, :, 0] = np.stack([rng.permutation(5) for _ in range(n)])
    for t in range(1, T):
        move = rng.random((n, 5)) > rho
        new = rng.integers(0, 5, size=(n, 5))
        state[:, :, t] = np.where(move, new, state[:, :, t - 1])
    g_o, g_b, g_h = gamma
    mu_all = g_o * off + g_b * ball[:, :, None] + g_h * HOOP_LEFT  # (n, T, 5k, 2)
    idx = np.transpose(state, (0, 2, 1))  # (n, T, 5d)
    mu = np.take_along_axis(mu_all, idx[..., None].repeat(2, -1), axis=2)
    dfn = mu + rng.normal(0, sigma, size=mu.shape)
    valid = np.ones((n, T), dtype=bool)
    if n_valid is not None:
        valid[:, n_valid:] = False
    inp = MatchupInputs(off_xy=off, def_xy=dfn, ball=ball, valid=valid)
    return inp, state


def test_forward_backward_matches_brute_force():
    rng = np.random.default_rng(1)
    K, T = 3, 4
    logB = np.log(rng.uniform(0.1, 1.0, size=(1, T, K)))
    logA = _log_trans(0.8, K)
    post, ll, _ = forward_backward(logB, logA)
    # brute force over all K**T paths
    A = np.exp(logA)
    B = np.exp(logB[0])
    total = 0.0
    marg = np.zeros((T, K))
    for code in range(K**T):
        path = [(code // K**i) % K for i in range(T)]
        pr = (1 / K) * B[0, path[0]]
        for t in range(1, T):
            pr *= A[path[t - 1], path[t]] * B[t, path[t]]
        total += pr
        for t, s in enumerate(path):
            marg[t, s] += pr
    assert ll[0] == pytest.approx(np.log(total), rel=1e-10)
    assert np.allclose(np.exp(post[0]), marg / total, atol=1e-10)
    # Viterbi = most probable path
    best = viterbi(logB, logA)[0]
    assert best.shape == (T,)


def test_em_recovers_parameters_and_states():
    inp, state = simulate()
    fit = fit_em(inp, HMMParams.initial(), max_iter=40)
    g = fit.params.gamma["all"]
    assert g == pytest.approx([0.6, 0.15, 0.25], abs=0.03)
    assert np.sqrt(fit.params.sigma2["all"]) == pytest.approx(2.0, abs=0.15)
    assert fit.params.rho == pytest.approx(0.97, abs=0.01)
    assert all(
        b >= a - 1e-6 * abs(a) for a, b in zip(fit.loglik, fit.loglik[1:], strict=False)
    )  # EM monotone
    post, path = posteriors(inp, fit.params)
    acc_hmm = (path == state).mean()
    acc_base = (min_residual(inp, FRANKS_GAMMA) == state).mean()
    assert acc_hmm > 0.9
    assert acc_hmm > acc_base
    assert np.allclose(post.sum(-1), 1.0)


def test_padding_does_not_bias_rho():
    inp, _ = simulate(n_valid=30, seed=3)
    fit = fit_em(inp, HMMParams.initial(), max_iter=40)
    assert fit.params.rho == pytest.approx(0.97, abs=0.015)


def test_stratified_and_help_state_run():
    inp, _ = simulate(n=40, T=30, seed=5)
    fit = fit_em(inp, HMMParams.initial(stratify=True, help_state=True), max_iter=10)
    assert set(fit.params.gamma) == {"strong", "weak"}
    post, path = posteriors(inp, fit.params)
    assert post.shape[-1] == 6
    # data contain no help defenders: the help state should take little mass
    assert post[..., 5].mean() < 0.1
    assert 0.0 <= fit.params.help_lam <= 1.0
