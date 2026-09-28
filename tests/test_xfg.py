"""xFG model: shooter prior shrinkage, leave-one-out, and learning the space effect."""

from __future__ import annotations

import numpy as np
import polars as pl

from ghost.xfg.model import ShooterPrior, design, fit_xfg


def _shots(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    zone = rng.choice(["rim", "paint", "mid", "corner3", "above3"], size=n)
    base = {"rim": 0.62, "paint": 0.40, "mid": 0.40, "corner3": 0.39, "above3": 0.35}
    def1 = rng.gamma(2.0, 2.0, size=n)
    shooter = rng.integers(0, 40, size=n)
    skill = (shooter - 20) / 100
    logit = np.log(np.array([base[z] for z in zone]) / (1 - np.array([base[z] for z in zone])))
    p = 1 / (1 + np.exp(-(logit + 0.35 * np.log(def1 + 0.5) - 0.35 + skill)))
    made = rng.random(n) < p
    dist = np.where(
        zone == "rim", 2.0, np.where(zone == "paint", 8.0, np.where(zone == "mid", 17.0, 24.0))
    )
    return pl.DataFrame(
        {
            "shooter_id": shooter,
            "zone": zone,
            "made": made,
            "dist_ft": dist,
            "angle_deg": rng.uniform(0, 90, n),
            "is_three": np.isin(zone, ["corner3", "above3"]),
            "def1_ft": def1,
            "def_front_ft": def1 + 1.0,
            "n_dribbles": rng.integers(0, 6, n),
            "catch_and_shoot": rng.random(n) < 0.3,
            "is_transition": rng.random(n) < 0.1,
        }
    )


def test_prior_shrinks_and_loo_removes_own_shot():
    S = _shots()
    pr = ShooterPrior.fit(S)
    assert set(pr.p_zone) == {"rim", "paint", "mid", "corner3", "above3"}
    assert all(5 <= k <= 2000 for k in pr.k_zone.values())
    one = S[:1]
    a = pr.skill(one, leave_one_out=False)[0]
    b = pr.skill(one, leave_one_out=True)[0]
    # removing a make lowers the skill estimate, removing a miss raises it
    assert (b < a) if one["made"][0] else (b > a)
    # unseen shooter -> zero offset
    unseen = one.with_columns(pl.lit(99999).alias("shooter_id"))
    assert pr.skill(unseen, False)[0] == 0.0


def test_design_and_space_effect():
    S = _shots(8000, seed=1)
    m = fit_xfg(S, ["zone", "geom", "defense", "dribble", "shooter"])
    X, names = design(S, m.groups, m.prior.skill(S, True))
    assert X.shape == (S.height, len(names))
    coef = m.coefficients()
    assert coef["log_def1"] + coef["log_def_front"] > 0  # more space -> higher make probability
    wider = S.with_columns(pl.col("def1_ft") + 3, pl.col("def_front_ft") + 3)
    assert m.predict(wider).mean() > m.predict(S).mean()
