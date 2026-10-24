"""Formulas of the robustness analyses: E-values and omitted-variable bounds, meta-metrics, help
location classes, the release-density baseline and the compute-log parser."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_e_value_and_robustness_value():
    s = _load("analysis_help_timing_sensitivity")
    # E-value of a risk ratio of 2 (VanderWeele & Ding 2017): 2 + sqrt(2)
    assert math.isclose(s.e_value(math.log(2) / 0.91, 1.0), 2 + math.sqrt(2), rel_tol=1e-9)
    # Cinelli & Hazlett (2020) Darfur example: t = 4.18 on 783 df -> RV 13.9 %, RV(5 %) 7.6 %
    f = 4.18 / math.sqrt(783)
    assert abs(s.rv(f, 783) - 0.139) < 0.001 and abs(s.rv(f, 783, 0.05) - 0.076) < 0.001


def test_ovb_recovers_effect_and_bounds_shrink_it():
    s = _load("analysis_help_timing_sensitivity")
    rng = np.random.default_rng(0)
    n, k = 4000, len(s.NUM)
    lin = rng.normal(size=(n, k))
    t = (lin[:, 2] + rng.normal(size=n) > 0).astype(float)  # treatment driven by covariate 2
    y = 0.5 * t + 0.8 * lin[:, 2] + rng.normal(size=n)
    out = s.ovb(lin, t, y)  # design = the linear terms only (no deciles): one column per group
    assert abs(out["estimate_ols_adjusted"] - 0.5) < 0.1
    assert out["strongest_observed_covariate"] == s.NUM[2]
    adj = [b["adjusted_estimate"] for b in out["bounds"][s.NUM[2]] if "adjusted_estimate" in b]
    assert all(a < out["estimate_ols_adjusted"] for a in adj) and adj == sorted(adj, reverse=True)


def test_meta_metrics_noise_model():
    m = _load("analysis_meta_metrics")
    rng = np.random.default_rng(1)
    rows = []
    for p in range(60):  # true player values, game-level noise, two parities
        true = rng.normal(0, 1.0)
        for g in range(40):
            v = true + rng.normal(0, 3.0, size=10)
            rows += [{"def_id": p, "game_id": f"{p}_{g}", "parity": g % 2, "v": x} for x in v]
    d = pl.DataFrame(rows)
    pt = m.player_table(d, "v", None, rng)
    disc = m.discrimination(pt)
    # signal variance 1, sampling variance 9 / 400 per player -> discrimination near 0.98
    assert 0.9 < disc < 1.0
    st = m.stability(m.player_table(d, "v", "parity", rng), "parity")
    assert 0.8 < st["stability"] < 1.2  # a random split: no change between halves
    pt = pt.with_columns(pl.col("x").alias("cov"))  # a covariate that is the signal itself
    assert m.independence(pt, ["cov"])["independence"] < 0.05


def test_help_location_classes():
    h = _load("analysis_help_left_man")
    xy = np.array([[2.0, 48.0], [30.0, 25.0], [10.0, 25.0], [2.0, 2.0]])
    assert h.location(xy).tolist() == [0, 1, 2, 0]  # corner, above the break, inside, corner


def test_release_rule_nll_matches_closed_form():
    r = _load("analysis_release_density_baseline")
    rng = np.random.default_rng(2)
    n = 6
    off = rng.uniform([5, 5], [40, 45], size=(n, 5, 2))
    ball = off[:, 0].copy()
    g = (0.62, 0.11, 0.27)
    mu = g[0] * off + g[1] * ball[:, None] + g[2] * np.array([5.25, 25.0])
    snap = {"off": off, "ball": ball, "def": mu.copy()}  # defenders exactly on the rule positions
    team, ind = r.rule_nll(snap, g, 3.0, 5.0)
    # at the left-over attacker's rule position the individual NLL is log(2 pi s^2)
    assert np.allclose(ind, math.log(2 * math.pi * 25.0))
    assert team.shape == (n, 5) and np.isfinite(team).all()


def test_compute_log_line():
    c = _load("phase3_compute_summary")
    m = c.LINE.match("2026-10-21 06:47:58 [66550423] start crossfit:fold1")
    assert m and m[2] == "66550423" and m[3] == "start crossfit:fold1"
