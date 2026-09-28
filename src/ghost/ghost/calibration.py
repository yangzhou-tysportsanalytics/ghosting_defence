"""Calibration, sampling and mode extraction for the Gaussian-mixture ghost (D-007).

HPD-PIT for a 2-D density: for an observation z with predictive density f, the value
    u = P_f( f(Z) >= f(z) )
is the probability mass of the highest-density region that just contains z. If the model is
calibrated, u ~ Uniform(0, 1) and the empirical coverage of the nominal-q HPD region equals q.
u is estimated by Monte Carlo from ``n_samples`` draws of the mixture.

``mixture_mode`` returns the component mean with the highest mixture density (a proper point
from the distribution, never the average of two modes - mean ghosts are excluded by design).
"""

from __future__ import annotations

import math

import torch


def mixture_log_density(
    logits: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """logits (..., M), mu (..., M, 2), sigma (..., M), z (..., K, 2) -> log f(z) (..., K)."""
    lw = torch.log_softmax(logits, dim=-1)[..., None, :]  # (..., 1, M)
    d2 = ((z[..., :, None, :] - mu[..., None, :, :]) ** 2).sum(-1)  # (..., K, M)
    s = sigma[..., None, :]
    logn = -0.5 * d2 / s**2 - 2 * torch.log(s) - math.log(2 * math.pi)
    return torch.logsumexp(lw + logn, dim=-1)


def sample_mixture(
    logits: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    n: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw n samples per mixture: returns (..., n, 2)."""
    shape = logits.shape[:-1]
    M = logits.shape[-1]
    flat_logits = logits.reshape(-1, M)
    comp = torch.multinomial(
        torch.softmax(flat_logits, -1), n, replacement=True, generator=generator
    )
    comp = comp.reshape(*shape, n)
    m = torch.gather(mu, -2, comp[..., None].expand(*shape, n, 2))
    s = torch.gather(sigma, -1, comp)
    eps = torch.randn(*shape, n, 2, generator=generator, device=mu.device)
    return m + s[..., None] * eps


def mixture_mode(logits: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Component mean with the highest mixture density: (..., 2)."""
    ld = mixture_log_density(logits, mu, sigma, mu)  # evaluate at each component mean
    best = ld.argmax(-1)
    return torch.gather(mu, -2, best[..., None, None].expand(*best.shape, 1, 2))[..., 0, :]


def hpd_pit(
    logits: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    z: torch.Tensor,
    n_samples: int = 256,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """HPD-PIT values u (...) for observations z (..., 2)."""
    samp = sample_mixture(logits, mu, sigma, n_samples, generator)  # (..., S, 2)
    ld_s = mixture_log_density(logits, mu, sigma, samp)  # (..., S)
    ld_z = mixture_log_density(logits, mu, sigma, z[..., None, :])[..., 0]  # (...)
    return (ld_s >= ld_z[..., None]).float().mean(-1)


def coverage_curve(u: torch.Tensor, levels=(0.5, 0.8, 0.9, 0.95)) -> dict[float, float]:
    """Empirical coverage of nominal HPD regions: share of u <= q."""
    u = u.flatten()
    return {float(q): float((u <= q).float().mean()) for q in levels}


def pit_histogram(u: torch.Tensor, bins: int = 10) -> list[float]:
    h = torch.histc(u.flatten(), bins=bins, min=0.0, max=1.0)
    return (h / h.sum()).tolist()
