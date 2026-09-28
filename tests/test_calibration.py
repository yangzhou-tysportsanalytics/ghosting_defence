"""HPD-PIT is uniform for a correctly specified mixture and detects under-dispersion."""

from __future__ import annotations

import torch

from ghost.ghost.calibration import (
    coverage_curve,
    hpd_pit,
    mixture_mode,
    sample_mixture,
)


def _mix(B=2000):
    logits = torch.log(torch.tensor([0.7, 0.3])).expand(B, 2).clone()
    mu = torch.tensor([[0.0, 0.0], [10.0, 0.0]]).expand(B, 2, 2).clone()
    sigma = torch.tensor([1.0, 2.0]).expand(B, 2).clone()
    return logits, mu, sigma


def test_pit_uniform_when_calibrated_and_detects_overconfidence():
    g = torch.Generator().manual_seed(0)
    logits, mu, sigma = _mix()
    z = sample_mixture(logits, mu, sigma, 1, g)[:, 0]
    u = hpd_pit(logits, mu, sigma, z, 512, g)
    cov = coverage_curve(u)
    for q, c in cov.items():
        assert abs(c - q) < 0.04, (q, c)
    # an over-confident model (sigmas halved) under-covers
    u2 = hpd_pit(logits, mu, sigma / 2, z, 512, g)
    assert coverage_curve(u2)[0.9] < 0.8


def test_mode_is_a_component_not_the_mean():
    logits, mu, sigma = _mix(4)
    m = mixture_mode(logits, mu, sigma)
    assert torch.allclose(m, torch.tensor([0.0, 0.0]).expand(4, 2))
