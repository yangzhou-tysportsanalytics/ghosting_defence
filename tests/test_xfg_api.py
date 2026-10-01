"""xFG feature module and interface: geometry, touch state, save / load, prediction."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ghost.court import HOOP_LEFT
from ghost.xfg import api
from ghost.xfg.features import NO_FRONT_FT, catch_and_shoot, geometry, touch_state

HX, HY = HOOP_LEFT


def _frame(shooter_xy, defenders_xy, offence_extra=4):
    """Ball + shooter (id 1, team 10) + teammates + defenders (team 20)."""
    team = [-1, 10] + [10] * offence_extra + [20] * len(defenders_xy)
    pid = [-1, 1] + list(range(2, 2 + offence_extra)) + list(range(100, 100 + len(defenders_xy)))
    xy = [shooter_xy, shooter_xy] + [(40.0, 45.0)] * offence_extra + list(defenders_xy)
    xy = np.array(xy, dtype=float)
    return np.array(team), np.array(pid), xy[:, 0], xy[:, 1]


def test_geometry_front_defender_and_nearest():
    s = (HX + 15.0, HY)  # 15 ft straight out from the hoop
    far = [(80.0, 45.0)] * 3
    t, p, x, y = _frame(s, [(HX + 11.0, HY), (HX + 17.0, HY)] + far)  # 4 ft in front, 2 ft behind
    g = geometry(t, p, x, y, shooter_id=1, shooter_team_id=10, attacks_left=True)
    assert g["dist_ft"] == pytest.approx(15.0)
    assert g["def1_ft"] == pytest.approx(2.0) and g["def_front_ft"] == pytest.approx(4.0)
    assert g["angle_deg"] == pytest.approx(0.0) and not g["is_three"]
    # nobody in the 45-degree cone -> 30 ft
    t, p, x, y = _frame(s, [(HX + 17.0, HY)] + far + [(80.0, 5.0)])
    assert geometry(t, p, x, y, 1, 10, True)["def_front_ft"] == NO_FRONT_FT
    # incomplete frame or shooter absent -> None
    assert geometry(t[:8], p[:8], x[:8], y[:8], 1, 10, True) is None
    assert geometry(t, p, x, y, 999, 10, True) is None


def test_touch_state_counts_dribbles_so_far():
    touches = [np.array([1, 1]), np.array([0, 10_000]), np.array([5_000, 20_000]),
               np.array(["a", "b"], dtype=object)]  # fmt: skip
    dribbles = [
        np.array(["b", "b", "b", "a"], dtype=object),
        np.array([11_000, 12_000, 14_000, 1_000]),
    ]
    assert touch_state(*touches, *dribbles, 1, 13_000) == (2, 3.0)  # 2 dribbles so far in touch b
    assert touch_state(*touches, *dribbles, 1, 15_000) == (3, 5.0)
    assert touch_state(*touches, *dribbles, 2, 13_000) == (None, None)  # other player
    assert touch_state(*touches, *dribbles, 1, 40_000) == (None, None)  # touch ended too long ago
    assert catch_and_shoot(0, 1.0) and not catch_and_shoot(1, 1.0) and not catch_and_shoot(0, 3.0)
    assert catch_and_shoot(None, 1.5) and not catch_and_shoot(None, None)


def _shots(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    zone = rng.choice(["rim", "paint", "mid", "corner3", "above3"], size=n)
    d1 = rng.gamma(2.0, 2.0, size=n)
    made = rng.random(n) < 1 / (1 + np.exp(-(0.4 * np.log(d1 + 0.5) - 0.6)))
    dist = np.where(zone == "rim", 2.0, np.where(zone == "paint", 8.0, 20.0))
    return pl.DataFrame(
        {
            "game_id": rng.choice(["g1", "g2", "g3"], size=n),
            "release_method": rng.choice(["rim", "apex", "pbp"], size=n, p=[0.8, 0.15, 0.05]),
            "shooter_id": rng.integers(0, 30, size=n),
            "zone": zone,
            "made": made,
            "dist_ft": dist,
            "angle_deg": rng.uniform(0, 90, n),
            "is_three": np.isin(zone, ["corner3", "above3"]),
            "def1_ft": d1,
            "def_front_ft": d1 + 1.0,
            "n_dribbles": rng.integers(0, 6, n),
            "catch_and_shoot": rng.random(n) < 0.3,
            "is_transition": rng.random(n) < 0.1,
        }
    )


def test_fit_save_load_predict(tmp_path):
    S = _shots()
    for version in ("M2", "M3"):
        m = api.fit(S, ["g1", "g2"], version)
        assert m.prior is None if version == "M2" else m.prior is not None
        api.save(m, tmp_path, version)
        m2 = api.load(tmp_path, version)
        test = S.filter(pl.col("game_id") == "g3")
        assert np.allclose(api.predict(test, m), api.predict(test, m2), atol=1e-12)
    # untrusted release methods are not used for fitting
    assert api.trusted(S)["release_method"].is_in(["rim", "apex"]).all()
    # rows without features -> NaN
    bad = S.head(3).with_columns(pl.lit(None, dtype=pl.Float64).alias("dist_ft"))
    assert np.isnan(api.predict(bad, m)).all()
