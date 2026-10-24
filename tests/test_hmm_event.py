"""Event-dependent switching HMM (B2): reduces to the constant model; recovers event effects."""

import numpy as np

from ghost.matchup import hmm as H
from ghost.matchup import hmm_event as E


def _random_logB(rng, C=7, T=30, K=6):
    return rng.normal(size=(C, T, K)) * 2.0


def test_constant_rho_matches_constant_model():
    rng = np.random.default_rng(0)
    logB = _random_logB(rng)
    C, T, K = logB.shape
    rho = 0.93
    code = rng.integers(0, E.N_PATTERNS, size=(C, T))
    post0, ll0, self0 = H.forward_backward(logB, H._log_trans(rho, K))
    post1, ll1, stay = E.forward_backward_tv(logB, code, np.full(E.N_PATTERNS, rho))
    assert np.allclose(ll0, ll1)
    assert np.allclose(post0, post1)
    assert np.allclose(self0, stay[:, 1:].sum(1))
    v0 = H.viterbi(logB, H._log_trans(rho, K))
    v1 = E.viterbi_tv(logB, code, np.full(E.N_PATTERNS, rho))
    assert (v0 == v1).all()


def test_time_varying_matches_brute_force():
    rng = np.random.default_rng(1)
    logB = _random_logB(rng, C=2, T=4, K=3)
    code = rng.integers(0, E.N_PATTERNS, size=(2, 4))
    rho = rng.uniform(0.6, 0.99, size=E.N_PATTERNS)
    _, ll, _ = E.forward_backward_tv(logB, code, rho)
    import itertools

    for c in range(2):
        tot = []
        for s in itertools.product(range(3), repeat=4):
            lp = -np.log(3) + logB[c, 0, s[0]]
            for t in range(1, 4):
                r = rho[code[c, t]]
                lp += np.log(r if s[t] == s[t - 1] else (1 - r) / 2) + logB[c, t, s[t]]
            tot.append(lp)
        assert np.isclose(ll[c], np.logaddexp.reduce(tot))


def test_update_beta_recovers_logistic():
    rng = np.random.default_rng(2)
    beta = np.array([-4.0, 1.0, 1.5, 0.5, 0.3])
    pi = 1 / (1 + np.exp(-(E.design() @ beta)))
    pairs = rng.integers(20_000, 40_000, size=E.N_PATTERNS).astype(float)
    switches = rng.binomial(pairs.astype(int), pi)
    b = E.update_beta(pairs - switches, pairs, np.zeros(5), (True,) * 5)
    assert np.allclose(b, beta, atol=0.15)
    b0 = E.update_beta(pairs - switches, pairs, np.zeros(5), (True, False, False, False, False))
    assert np.isclose(1 / (1 + np.exp(-b0[0])), switches.sum() / pairs.sum())
    assert (b0[1:] == 0).all()


def test_em_recovers_event_effect():
    """Simulated chains whose matchups change mostly at 'screen' steps."""
    rng = np.random.default_rng(3)
    N, T = 300, 60
    off = rng.uniform([5, 5], [40, 45], size=(N, 1, 5, 2)) + np.cumsum(
        rng.normal(scale=0.4, size=(N, T, 5, 2)), axis=1
    )
    ball = off[:, :, 0, :].copy()
    screen = rng.random((N, T)) < 0.08
    code = np.repeat(screen[:, None, :].astype(np.uint8), 5, axis=1)  # bit 0 = screen
    true_rho = E.rho_of_beta(np.array([-5.0, 3.5, 0, 0, 0]))
    state = np.tile(np.arange(5), (N, 1))
    def_xy = np.empty((N, T, 5, 2))
    p = H.HMMParams.initial()
    hoop = H.HOOP_LEFT.astype(float)
    for t in range(T):
        if t:
            r = true_rho[code[:, :, t]]
            move = rng.random((N, 5)) > r
            state = np.where(move, (state + rng.integers(1, 5, size=(N, 5))) % 5, state)
        mu = H.man_means(
            H.MatchupInputs(off[:, t : t + 1], off[:, t : t + 1], ball[:, t : t + 1], None),
            p.gamma["all"],
        )[:, 0]
        def_xy[:, t] = np.take_along_axis(mu, state[..., None], 1) + rng.normal(
            scale=1.5, size=(N, 5, 2)
        )
    inp = H.MatchupInputs(off, def_xy, ball, np.ones((N, T), bool), hoop)
    fit = E.fit_em(inp, code, E.EventHMMParams(emis=H.HMMParams.initial()), max_iter=15)
    b = fit.params.beta
    assert 2.5 < b[1] < 4.5
    assert -5.8 < b[0] < -4.3
    assert all(np.diff(fit.loglik) > -1e-6 * abs(fit.loglik[0]))


def test_event_codes():
    T = 30
    t = 1_000_000 + 200 * np.arange(T)
    valid = np.arange(T) < 25
    handler = np.zeros(T, dtype=np.int8)
    off_xy = np.zeros((T, 5, 2)) + [40.0, 25.0]
    off_xy[12:, 0, 0] = 25.0  # handler 15 ft closer to the rim from step 12 on
    hoop = np.array([5.25, 25.0])
    def_ids = np.array([11, 12, 13, 14, 15])
    code = E.event_codes(
        t, valid, handler, off_xy, def_ids, [(t[5], 12, 14)], [(t[20], t[21])], hoop
    )
    assert (code[:, 0:11] & 1 > 0).sum(1).tolist() == [11] * 5  # steps 0..10 within 1 s of 5
    assert (code[:, 11] & 1 == 0).all()
    assert ((code[:, :11] & 2) > 0).any(1).tolist() == [False, True, False, True, False]
    assert ((code[:, 20:25] & 4) > 0).all() and not (code[:, 25:] & 4).any()  # padded: none
    # drive windows start at steps 2..11 (reach step 12 within 2 s): transitions 3..21
    assert ((code[:, 3:22] & 8) > 0).all() and not (code[:, :3] & 8).any()
    assert not (code[:, 22:] & 8).any()
