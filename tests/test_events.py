"""Switch / help / closeout derivation on synthetic 5 Hz possessions."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ghost.matchup.events import (
    EventConfig,
    classify_switches,
    closeout_events,
    help_a_events,
    help_b_events,
    stable_path,
    switch_b_events,
)

T = 40
T0 = 1_000_000
t_ms = T0 + 200 * np.arange(T)
OFF = np.array([101, 102, 103, 104, 105])
DEF = np.array([201, 202, 203, 204, 205])


def test_stable_path_removes_flicker():
    path = np.zeros((1, T), dtype=np.int8)
    path[0, 10:12] = 3  # 0.4 s flicker
    path[0, 20:] = 1  # real change
    valid = np.ones(T, bool)
    valid[35:] = False
    s = stable_path(path, valid, min_hold=5)
    assert (s[0, :20] == 0).all()
    assert (s[0, 20:35] == 1).all()
    assert (s[0, 35:] == -1).all()


def _switch_case():
    path = np.tile(np.arange(5, dtype=np.int8)[:, None], (1, T))  # d guards attacker d
    # screen at step 15: user = attacker 0 (defended by d0), screener = attacker 1 (d1)
    path[0, 16:] = 1  # d0: user -> screener
    path[1, 17:] = 0  # d1: screener -> user
    path[3, 30:] = 4  # d3: unrelated rotation 3 -> 4
    valid = np.ones(T, bool)
    s = stable_path(path, valid, 5)
    ev = pl.DataFrame(switch_b_events(s, t_ms, OFF, DEF))
    screens = pl.DataFrame(
        {
            "game_id": ["g"],
            "event_uid": ["g:screen_candidate:x"],
            "screener_id": [102],
            "user_id": [101],
            "screened_def_id": [201],
            "screener_def_id": [202],
            "t_contact_ms": [int(t_ms[15])],
            "on_ball": [True],
        }
    )
    return ev, screens


def test_switch_a_b_classification():
    ev, screens = _switch_case()
    assert ev.height == 3
    out = classify_switches(ev, screens)
    cls = dict(zip(out["def_id"].to_list(), out["event_class"].to_list(), strict=True))
    assert cls == {201: "screen_switch", 202: "screen_switch", 204: "rotation"}
    assert out.filter(pl.col("event_class") == "screen_switch")[
        "screen_uid"
    ].unique().to_list() == ["g:screen_candidate:x"]


def test_one_sided_screen_reaction():
    ev, screens = _switch_case()
    ev = ev.filter(pl.col("def_id") != 202)  # the screener's defender stays home
    out = classify_switches(ev, screens)
    assert out.filter(pl.col("def_id") == 201)["event_class"].item() == "one_sided"


def test_overlapping_candidate_does_not_downgrade_screen_switch():
    ev, screens = _switch_case()
    # a second, overlapping candidate that matches only the screener's defender's change
    extra = screens.with_columns(
        pl.lit("g:screen_candidate:y").alias("event_uid"),
        pl.lit(101, dtype=pl.Int64).alias("screener_id"),
        pl.lit(102, dtype=pl.Int64).alias("user_id"),
        pl.lit(202, dtype=pl.Int64).alias("screened_def_id"),
        pl.lit(999, dtype=pl.Int64).alias("screener_def_id"),
    )
    out = classify_switches(ev, pl.concat([screens, extra]))
    sw = out.filter(pl.col("event_class") == "screen_switch")
    assert sorted(sw["def_id"].to_list()) == [201, 202]
    assert sw["screen_uid"].unique().to_list() == ["g:screen_candidate:x"]


def test_schema_is_enforced():
    ev, screens = _switch_case()
    with pytest.raises(ValueError):
        classify_switches(ev, screens.drop("screened_def_id"))


def _help_case():
    off = np.zeros((T, 5, 2))
    off[:, :, 0] = 25.0
    off[:, :, 1] = np.array([5, 15, 25, 35, 45])
    off[:, 0] = [15.0, 25.0]  # handler (slot 0) near the hoop (~10 ft)
    dfn = off.copy()
    dfn[:, :, 0] -= 2.0  # everyone tight on his man
    # defender 3 (guarding attacker 3 at (25, 35)) helps toward the handler from step 10 to 20
    for k in range(10, 20):
        f = (k - 9) / 10
        dfn[k, 3] = (1 - f) * np.array([23.0, 35.0]) + f * np.array([17.0, 27.0])
    handler = np.zeros(T, dtype=np.int8)
    valid = np.ones(T, bool)
    stable = np.tile(np.arange(5, dtype=np.int16)[:, None], (1, T))
    return off, dfn, handler, valid, stable


def test_help_b():
    off, dfn, handler, valid, stable = _help_case()
    ev = help_b_events(stable, handler, off, dfn, valid, t_ms, OFF, DEF)
    assert len(ev) == 1
    e = ev[0]
    assert e["def_id"] == 204 and e["man_off_id"] == 104 and e["handler_id"] == 101
    assert e["max_dist_from_man_ft"] >= 6.0
    # handler far from the hoop -> no help event
    off2 = off.copy()
    off2[:, 0] = [40.0, 25.0]
    dfn2 = dfn.copy()
    assert help_b_events(stable, handler, off2, dfn2, valid, t_ms, OFF, DEF) == []


def test_help_a():
    off, dfn, handler, valid, _ = _help_case()
    p_help = np.zeros((5, T))
    p_help[3, 10:20] = 0.9
    ev = help_a_events(p_help, handler, off, dfn, valid, t_ms, DEF, OFF)
    assert len(ev) == 1 and ev[0]["def_id"] == 204


def test_closeout():
    off = np.zeros((T, 5, 2))
    off[:, :, 0] = 20.0
    off[:, :, 1] = np.array([5, 15, 25, 35, 45])
    dfn = off.copy()
    dfn[:, :, 0] -= 3.0
    # receiver = attacker 4 in the corner; its defender sags 12 ft off and closes in 0.8 s
    catch = 20
    dfn[: catch + 1, 4] = [12.0, 40.0]
    for k in range(1, 5):
        f = k / 4
        dfn[catch + k, 4] = (1 - f) * np.array([12.0, 40.0]) + f * np.array([18.0, 44.0])
    dfn[catch + 5 :, 4] = [18.0, 44.0]
    stable = np.tile(np.arange(5, dtype=np.int16)[:, None], (1, T))
    passes = pl.DataFrame(
        {
            "game_id": ["g"],
            "event_uid": ["g:pass:1"],
            "passer_id": [101],
            "receiver_id": [105],
            "t_release_ms": [int(t_ms[catch - 3])],
            "t_catch_ms": [int(t_ms[catch])],
            "completed": [True],
        }
    )
    ev = closeout_events(passes, stable, off, dfn, np.ones(T, bool), t_ms, OFF, DEF, EventConfig())
    assert len(ev) == 1
    e = ev[0]
    assert e["def_id"] == 205 and e["matched_defender"]
    assert e["dist_at_catch_ft"] > 8 and e["dist_closed_ft"] < 4
    assert e["closing_speed_fts"] >= 8
