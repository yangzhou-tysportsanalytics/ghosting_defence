"""Rule-ghost deviation, phase labels, sightline cone and paint runs (D-020)."""

from __future__ import annotations

import numpy as np
import pytest

from ghost.court import HOOP_LEFT
from ghost.deviation.phases import (
    PHASES,
    PhaseDefs,
    in_paint,
    in_sightline_cone,
    long_paint_runs,
    phase_labels,
    rule_ghost_deviation,
)

G = np.array([0.6, 0.1, 0.3])


def test_rule_ghost_deviation_zero_at_ghost_and_sag_sign():
    T = 3
    off = np.tile(np.array([[30.0, 25.0]] * 5)[None], (T, 1, 1))  # every attacker at (30, 25)
    ball = np.tile(np.array([30.0, 30.0]), (T, 1))
    mu = G[0] * off[0, 0] + G[1] * ball[0] + G[2] * HOOP_LEFT
    dfn = np.tile(mu, (T, 5, 1))
    dfn[:, 1] += np.array([-2.0, 0.0])  # defender 1 is 2 ft closer to the hoop (x decreases)
    m = np.zeros((5, T), dtype=np.int8)
    m[4] = 5  # help state
    dev, sag = rule_ghost_deviation(dfn, off, ball, m, np.ones(T, bool), G, G)
    assert np.allclose(dev[0], 0) and np.allclose(sag[0], 0)
    assert np.allclose(dev[1], 2.0) and np.allclose(sag[1], 2.0)  # attacker -> hoop is -x
    assert np.isnan(dev[4]).all()


def test_phase_priority_and_windows():
    t = np.arange(0, 6000, 200)  # 30 steps, 6 s
    help_iv = [[(3000, 3600)], [], [], [], []]
    clo_iv = [[], [(1000, 1400)], [], [], []]
    lab = phase_labels(t, help_iv, clo_iv, [2000], PhaseDefs.variant("A"))
    name = np.array(PHASES)[lab]
    assert name[2, 0] == "pre_screen" and name[2, t.tolist().index(1600)] == "screen"
    # after the screen window (ends at 3000, inclusive)
    assert name[2, t.tolist().index(3200)] == "other"
    assert name[0, t.tolist().index(3200)] == "help"
    assert (
        name[0, t.tolist().index(5400)] == "recovery" and name[0, t.tolist().index(5800)] == "other"
    )
    assert name[1, t.tolist().index(1200)] == "closeout"  # overrides pre_screen
    assert (
        name[1, t.tolist().index(2000)] == "recovery"
    )  # within 2 s after the closeout, over screen
    none = phase_labels(t, [[]] * 5, [[]] * 5, [], PhaseDefs())
    assert (np.array(PHASES)[none] == "other").all()


def test_sightline_cone():
    h = np.array([[HOOP_LEFT[0] + 20.0, HOOP_LEFT[1]]])  # handler 20 ft straight out
    pts = [
        [HOOP_LEFT[0] + 10.0, HOOP_LEFT[1]],  # on the line, between -> in
        [HOOP_LEFT[0] + 10.0, HOOP_LEFT[1] + 10.0],  # 45 degrees off -> out
        [HOOP_LEFT[0] + 25.0, HOOP_LEFT[1]],  # behind the handler -> out
        [HOOP_LEFT[0] - 2.0, HOOP_LEFT[1]],  # beyond the hoop (22 ft > 20 ft) -> out
        [HOOP_LEFT[0] + 10.0, HOOP_LEFT[1] + 3.0],
    ]  # ~16.7 degrees -> in at 20, in at 30
    d = np.array(pts)[None]
    assert in_sightline_cone(h, d, 20.0)[0].tolist() == [True, False, False, False, True]
    assert not in_sightline_cone(np.full((1, 2), np.nan), d, 20.0).any()


def test_paint_runs():
    T = 30
    xy = np.full((T, 5, 2), 40.0)
    xy[2:15, 0] = [10.0, 25.0]  # 13 steps = 2.6 s in the lane
    xy[16:28, 1] = [10.0, 25.0]  # 12 steps = 2.4 s
    p = in_paint(xy)
    flag, longest = long_paint_runs(p, np.ones(T, bool), 2.5)
    assert flag[2:15, 0].all() and not flag[:, 1].any()
    assert longest[0] == pytest.approx(2.6) and longest[1] == pytest.approx(2.4) and longest[2] == 0
    v = np.ones(T, bool)
    v[8] = False  # an invalid step breaks the stretch
    assert not long_paint_runs(p, v, 2.5)[0][:, 0].any()


def test_variant_b():
    b = PhaseDefs.variant("B")
    assert (b.screen_before_s, b.screen_after_s, b.recovery_s, b.cone_half_angle_deg) == (
        1.0,
        2.0,
        3.0,
        30.0,
    )
