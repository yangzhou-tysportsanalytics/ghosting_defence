"""Learned-ghost shot value: vectorised defence geometry equals the xFG feature definition."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from ghost.xfg.features import geometry

_p = Path(__file__).resolve().parents[1] / "scripts" / "phase4_learned_ghost_points.py"
spec = importlib.util.spec_from_file_location("phase4_learned_ghost_points", _p)
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)


def test_defense_geometry_matches_xfg_features():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = rng.uniform([5, 3], [40, 47])
        dfn = rng.uniform([0, 0], [47, 50], size=(5, 2))
        off = np.vstack([s, rng.uniform([0, 0], [47, 50], size=(4, 2))])
        xs = np.concatenate([[20.0], off[:, 0], dfn[:, 0]])
        ys = np.concatenate([[25.0], off[:, 1], dfn[:, 1]])
        team = np.array([-1] + [1] * 5 + [2] * 5)
        pid = np.array([-1, 7, 8, 9, 10, 11, 21, 22, 23, 24, 25])
        ref = geometry(team, pid, xs, ys, 7, 1, True)
        d1, d2, fr = lp.defense_geometry(s, dfn[None])
        assert np.isclose(d1[0], ref["def1_ft"]) and np.isclose(d2[0], ref["def2_ft"])
        assert np.isclose(fr[0], ref["def_front_ft"])


def test_sample_mixture_moments():
    rng = np.random.default_rng(1)
    x = lp.sample_mixture(np.array([1.0, 0.0]), np.array([[10.0, 20.0], [0.0, 0.0]]),
                          np.array([2.0, 1.0]), 20000, rng)  # fmt: skip
    assert np.allclose(x.mean(0), [10, 20], atol=0.1) and np.allclose(x.std(0), 2.0, atol=0.1)
