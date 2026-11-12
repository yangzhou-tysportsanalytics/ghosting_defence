"""Velocity features: backward differences are causal (review 2026-10-30)."""

import numpy as np

from ghost.ghost.dataset import backward_velocity
from ghost.possession.resample import _velocity


def test_backward_velocity_is_causal_and_correct():
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(2, 10, 3, 2))
    valid = np.ones((2, 10), bool)
    valid[1, 7:] = False
    v = backward_velocity(xy, valid)
    assert np.allclose(v[:, 0], 0)
    assert np.allclose(v[0, 4], (xy[0, 4] - xy[0, 3]) / 0.2)
    assert np.allclose(v[1, 7:], 0)  # invalid steps
    xy2 = xy.copy()
    xy2[:, 5] += 3.0  # change the future of step 4
    assert np.allclose(backward_velocity(xy2, valid)[:, :5], v[:, :5])


def test_central_velocity_looks_ahead():
    """The stored velocities (central differences) depend on the next step: the reason for the
    backward option."""
    xy = np.cumsum(np.ones((10, 2)), axis=0)
    a = _velocity(xy, 0.2, np.ones(10, bool))
    xy2 = xy.copy()
    xy2[5] += 3.0
    b = _velocity(xy2, 0.2, np.ones(10, bool))
    assert not np.allclose(a[4], b[4])
