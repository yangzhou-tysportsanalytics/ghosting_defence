"""Breakdown runs (>= 1 s above threshold) and cascade attribution."""

from __future__ import annotations

import importlib.util

import numpy as np

spec = importlib.util.spec_from_file_location("phase4_breakdowns", "scripts/phase4_breakdowns.py")
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)


def test_runs_need_five_steps():
    m = np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1], bool)
    assert bd.runs(m) == [(1, 5), (10, 15)]  # the 2-step run is not a breakdown
    assert bd.runs(np.ones(5, bool)) == [(0, 4)]
    assert bd.runs(np.zeros(8, bool)) == []


def test_cascade_initiator_is_earliest():
    events = [(1, 5), (8, 12), (20, 24), (3, 9)]
    cid, init = bd.cascades(events, gap=5)
    assert cid == [0, 0, 1, 0]  # 8 <= 9 + 5 joins; 20 > 12 + 5 starts a new cascade
    assert init == [True, False, True, False]
    cid, init = bd.cascades(events, gap=0)  # no gap: only overlapping events chain
    assert cid[0] == cid[3] and cid[1] == cid[0] and init == [True, False, True, False]
