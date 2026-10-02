"""Phase 3 training helpers: seeded game subsets and the individual rule baseline."""

from __future__ import annotations

import importlib.util
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ghost.court import HOOP_LEFT

_p = Path(__file__).resolve().parents[1] / "scripts" / "phase3_train.py"
spec = importlib.util.spec_from_file_location("phase3_train", _p)
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)

GAMMA = (0.62, 0.11, 0.27)


def test_pick_games_nested_and_independent():
    games = {"train": [f"t{i:03d}" for i in range(100)], "val": [f"v{i:02d}" for i in range(20)]}
    a = p3.pick_games(games, 12, 8, 9)
    b = p3.pick_games(games, 32, 8, 9)
    assert a["train"] == b["train"][:12]  # a larger train subset contains the smaller one
    assert a["val"] == b["val"]  # val does not depend on the train size
    assert p3.pick_games(games, None, None, 9) == games


def _arrays(n=6, T=7, seed=0):
    rng = np.random.default_rng(seed)
    feats = np.zeros((n, 11, T, 8), dtype=np.float32)
    feats[:, :6, :, :2] = rng.uniform(-0.5, 0.5, size=(n, 6, T, 2))
    target = rng.uniform([0, 0], [47, 50], size=(n, 5, T, 2)).astype(np.float32)
    valid = np.ones((n, T), dtype=bool)
    valid[0, -2:] = False
    return {"feats": feats, "target": target, "valid": valid}


def test_individual_baseline_equals_hungarian_loop():
    arrs = _arrays()
    f = arrs["feats"]
    off = np.stack([(f[:, 1:6, :, 0] + 0.5) * 47.0, (f[:, 1:6, :, 1] + 0.5) * 50.0], -1)
    ball = np.stack([(f[:, 0, :, 0] + 0.5) * 47.0, (f[:, 0, :, 1] + 0.5) * 50.0], -1)
    mu = GAMMA[0] * off + GAMMA[1] * ball[:, None] + GAMMA[2] * HOOP_LEFT
    se, n = 0.0, 0
    for i in range(len(arrs["valid"])):
        for t in np.flatnonzero(arrs["valid"][i]):
            D2 = ((arrs["target"][i, :, t][:, None] - mu[i, :, t][None]) ** 2).sum(-1)
            for d in range(5):
                vis = [x for x in range(5) if x != d]
                _, c = linear_sum_assignment(np.sqrt(D2[vis]))
                k = ({0, 1, 2, 3, 4} - set(c.tolist())).pop()
                se += D2[d, k]
                n += 1
    res = p3.individual_baseline(arrs, GAMMA, chunk=4)
    assert np.isclose(res["rmse_ft"], np.sqrt(se / n))
    assert len(list(permutations(range(5)))) == 120


def test_baseline_rmse_is_best_permutation():
    arrs = _arrays()
    r = p3.baseline_rmse(arrs, GAMMA)
    assert np.isfinite(r) and r > 0
