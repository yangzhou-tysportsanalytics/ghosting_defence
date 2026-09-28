"""Hierarchical decomposition of defensive deviation (Phase 4 task 4, numpyro).

Two-stage Gaussian model (exact for the Gaussian likelihood up to the stage-1 uncertainty of the
context coefficients, which is ignored):
  stage 1  context adjustment by OLS on possession-level rows (done by the caller)
  stage 2  cell c = (player p, team t) with n_c possessions and mean residual ybar_c:
           ybar_c ~ Normal(mu + beta_t + alpha_p, sigma_e^2 / n_c)
           beta_t ~ Normal(0, tau_team), alpha_p ~ Normal(0, tau_player)
           tau ~ HalfNormal(prior_scale)
Definitional decomposition (D-005): after sampling, player effects are centred within each team
(possession-weighted) and the team means are moved into beta_t, so team effect := team mean and
player effect := deviation from the own team mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


@dataclass
class Cells:
    ybar: np.ndarray
    n: np.ndarray
    team_idx: np.ndarray
    player_idx: np.ndarray
    n_team: int
    n_player: int
    sigma_e: float


def model(cells: Cells, prior_scale: float = 1.0):
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    tau_t = numpyro.sample("tau_team", dist.HalfNormal(prior_scale))
    tau_p = numpyro.sample("tau_player", dist.HalfNormal(prior_scale))
    with numpyro.plate("teams", cells.n_team):
        z_t = numpyro.sample("z_team", dist.Normal(0.0, 1.0))
    with numpyro.plate("players", cells.n_player):
        z_p = numpyro.sample("z_player", dist.Normal(0.0, 1.0))
    beta = numpyro.deterministic("beta_team", tau_t * z_t)
    alpha = numpyro.deterministic("alpha_player", tau_p * z_p)
    loc = mu + beta[cells.team_idx] + alpha[cells.player_idx]
    scale = cells.sigma_e / jnp.sqrt(cells.n)
    numpyro.sample("ybar", dist.Normal(loc, scale), obs=cells.ybar)


def fit(
    cells: Cells, prior_scale: float = 1.0, warmup: int = 500, samples: int = 1000, seed: int = 9
) -> dict:
    mcmc = MCMC(NUTS(model), num_warmup=warmup, num_samples=samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), cells=cells, prior_scale=prior_scale)
    s = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    return s


def definitional(s: dict, cells: Cells) -> tuple[np.ndarray, np.ndarray]:
    """Per-draw team effects (draws, n_team) and within-team player effects (draws, n_cells):
    team := possession-weighted team mean of mu+beta+alpha; player := cell value - team mean."""
    val = (
        s["mu"][:, None]
        + s["beta_team"][:, cells.team_idx]
        + s["alpha_player"][:, cells.player_idx]
    )
    w = cells.n.astype(float)
    team_mean = np.zeros((val.shape[0], cells.n_team))
    for t in range(cells.n_team):
        m = cells.team_idx == t
        team_mean[:, t] = (val[:, m] * w[m]).sum(1) / w[m].sum()
    player_within = val - team_mean[:, cells.team_idx]
    return team_mean, player_within


def summarize(draws: np.ndarray) -> dict:
    return {
        "mean": float(draws.mean()),
        "lo95": float(np.quantile(draws, 0.025)),
        "hi95": float(np.quantile(draws, 0.975)),
    }
