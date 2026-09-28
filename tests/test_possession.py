"""Tests for possession segmentation, resampling and ball-handler inference (synthetic)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ghost.ballhandler.infer import (
    HandlerConfig,
    candidate_handlers,
    handler_on_grid,
    last_handler_before,
    sticky_handler,
)
from ghost.clean.dedup import build_frame_index, dedup_entities
from ghost.io.raw import BALL_ID, FRAME_SCHEMA
from ghost.possession.resample import ResampleConfig, arrays_to_long, resample_possession
from ghost.possession.segment import (
    SegmentConfig,
    _crossing_index,
    locate_clock,
    segment_game,
)

T0 = 1_447_287_043_000
HOME, AWAY = 100, 200
HOME_IDS = [1000 + i for i in range(5)]
AWAY_IDS = [2000 + i for i in range(5)]
GID = "0021500001"


# ------------------------------------------------------------------------------------------
# synthetic game: HOME attacks LEFT in period 1.
#   t = 0..9.96 s   : ball in HOME backcourt (x = 70), then crosses at t = 10 s to x = 30
#   t = 10..19.96 s : HOME frontcourt possession, HOME player 1000 holds the ball
#   pbp: FG made by 1000 at clock 700 (=> t in [19, 20) s... we set gc = 720 - t)
# ------------------------------------------------------------------------------------------


def _frame_rows(event_id, mi, period, t_ms, gc, sc, ball_xyz, off_xy, def_xy):
    rows = [(event_id, mi, period, t_ms, gc, sc, BALL_ID, BALL_ID, *ball_xyz)]
    for pid, (x, y) in zip(HOME_IDS, off_xy, strict=True):
        rows.append((event_id, mi, period, t_ms, gc, sc, HOME, pid, x, y, 0.0))
    for pid, (x, y) in zip(AWAY_IDS, def_xy, strict=True):
        rows.append((event_id, mi, period, t_ms, gc, sc, AWAY, pid, x, y, 0.0))
    return rows


def _df(rows):
    cols = [c for c in FRAME_SCHEMA if c != "game_id"]
    data = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    data["game_id"] = [GID] * len(rows)
    return pl.DataFrame(data, schema=FRAME_SCHEMA)


@pytest.fixture
def synthetic_game():
    rows = []
    n = 500  # 20 s at 25 Hz
    for i in range(n):
        t = i * 0.04
        gc = 720.0 - t
        if t < 10:
            bx = 70.0 - 2.0 * t  # moving towards midcourt but still backcourt (>47)
            hx = 70.0 - 2.0 * t
        else:
            bx = 30.0 - 0.5 * (t - 10)  # frontcourt (<47)
            hx = 30.0 - 0.5 * (t - 10)
        ball = (bx, 25.0, 2.0)
        off = [(hx, 25.0)] + [(20.0, 5.0 + 10 * k) for k in range(1, 5)]  # slot 0 = 1000 holds
        dfn = [(hx - 4.0, 25.0)] + [(15.0, 5.0 + 10 * k) for k in range(1, 5)]
        rows += _frame_rows(1, i, 1, T0 + 40 * i, gc, 24.0 - (t % 24), ball, off, dfn)
    ent = dedup_entities(_df(rows))
    fi = build_frame_index(ent)
    pbp = pl.DataFrame(
        {
            "game_id": [GID, GID],
            "event_num": [1, 2],
            "period": [1, 1],
            "pctime_sec": [720.0, 700.0],
            "msg_type": [12, 1],
            "action_type": [0, 1],
            "p1_team_id": [None, HOME],
            "p1_id": [0, 1000],
            "home_desc": [None, "Player 25' 3PT Jump Shot (3 PTS)"],
            "visitor_desc": [None, None],
            "score_margin": [None, 3],
        },
        schema={
            "game_id": pl.Utf8,
            "event_num": pl.Int32,
            "period": pl.Int8,
            "pctime_sec": pl.Float32,
            "msg_type": pl.Int16,
            "action_type": pl.Int16,
            "p1_team_id": pl.Int64,
            "p1_id": pl.Int64,
            "home_desc": pl.Utf8,
            "visitor_desc": pl.Utf8,
            "score_margin": pl.Int16,
        },
    )
    direction = pl.DataFrame(
        {
            "game_id": [GID, GID],
            "team_id": [HOME, AWAY],
            "period": [1, 1],
            "attacks_left": [True, False],
        },
        schema={
            "game_id": pl.Utf8,
            "team_id": pl.Int64,
            "period": pl.Int8,
            "attacks_left": pl.Boolean,
        },
    )
    return ent, fi, pbp, direction


def test_crossing_index():
    assert _crossing_index(np.array([False, False, True, True])) == 2
    assert _crossing_index(np.array([False, True, False, True])) == 3
    assert _crossing_index(np.array([True, True])) is None  # never in backcourt
    assert _crossing_index(np.array([False, True, False])) is None  # ends in backcourt
    assert _crossing_index(np.array([], dtype=bool)) is None


def test_locate_clock(synthetic_game):
    _, fi, _, _ = synthetic_game
    # clock reading 700 -> frames with gc in [700, 701) -> centre 700.5 -> t = 19.5 s
    t = locate_clock(fi, 700.0, None)
    assert t == T0 + 19_480  # 19.48 and 19.52 tie on |gc - 700.5|; earlier frame wins
    assert locate_clock(fi, 100.0, None) is None  # untracked
    # after_unix excludes earlier candidates
    assert locate_clock(fi, 700.0, T0 + 19_600) == T0 + 19_640


def test_segment_game(synthetic_game):
    ent, fi, pbp, direction = synthetic_game
    poss, rej = segment_game(ent, fi, pbp, direction, HOME, SegmentConfig())
    assert poss.height == 1, rej
    r = poss.row(0, named=True)
    assert r["offense_team_id"] == HOME and r["defense_team_id"] == AWAY
    assert r["attacks_left"] is True
    assert r["t_cross"] == T0 + 10_000
    assert r["t_terminal"] == T0 + 19_480 and r["t_end"] == T0 + 19_980
    assert r["duration_s"] == pytest.approx(9.98)
    assert r["terminal_type"] == "fg_made" and r["terminal_player_id"] == 1000
    assert r["offense_player_ids"] == HOME_IDS and r["defense_player_ids"] == AWAY_IDS
    assert r["is_transition"] is False and r["cropped"] is False
    assert r["score_margin_offense"] == 0  # margin before the shot
    assert r["game_clock_start"] == pytest.approx(710.0, abs=0.05)


def test_segment_rejects_when_lineup_changes(synthetic_game):
    ent, fi, pbp, direction = synthetic_game
    # swap one defender's id for the last 2 s -> lineup change
    ent2 = ent.with_columns(
        pl.when((pl.col("player_id") == 2004) & (pl.col("unix_ms") > T0 + 18_000))
        .then(pl.lit(2999, dtype=pl.Int64))
        .otherwise(pl.col("player_id"))
        .alias("player_id")
    )
    poss, rej = segment_game(ent2, fi, pbp, direction, HOME)
    assert poss.height == 0 and rej == {"lineup_change": 1}


def test_resample_and_flip(synthetic_game):
    ent, fi, pbp, direction = synthetic_game
    poss, _ = segment_game(ent, fi, pbp, direction, HOME)
    row = poss.row(0, named=True)
    pa = resample_possession(ent, fi, row, ResampleConfig())
    assert pa.t.shape == (121,) and pa.valid.sum() == 50  # 9.98 s at 5 Hz -> steps 0..49
    assert not pa.flipped
    # ball x at grid step k = 30 - 0.5 * (0.2 k)
    assert pa.ball[0, 0] == pytest.approx(30.0, abs=0.02)
    assert pa.ball[49, 0] == pytest.approx(25.1, abs=0.02)
    assert pa.ball_v[10, 0] == pytest.approx(-0.5, abs=0.05)
    assert pa.off_ids.tolist() == HOME_IDS and pa.def_ids.tolist() == AWAY_IDS
    assert pa.game_clock[0] == pytest.approx(710.0, abs=0.05)
    # flip: pretend the offence attacked right
    row2 = dict(row, attacks_left=False)
    pb = resample_possession(ent, fi, row2, ResampleConfig())
    assert pb.flipped
    assert pb.ball[0, 0] == pytest.approx(94.0 - 30.0, abs=0.02)
    assert pb.ball[0, 1] == pytest.approx(50.0 - 25.0, abs=0.02)
    assert pb.ball_v[10, 0] == pytest.approx(0.5, abs=0.05)
    long = arrays_to_long(pa, np.zeros(121, dtype=np.int8))
    assert long.height == 121 * 11
    assert (
        long.filter(pl.col("step") == 0)["role"].to_list() == ["ball"] + ["off"] * 5 + ["def"] * 5
    )
    assert long.filter(pl.col("is_handler"))["player_id"].unique().to_list() == [1000]
    assert long.filter(~pl.col("valid")).height == 71 * 11


def test_handler_rules():
    cfg = HandlerConfig(hz=25.0, min_hold_s=0.4)
    n = 60
    t = np.arange(n) * 40
    ball = np.tile(np.array([[10.0, 10.0, 2.0]]), (n, 1))
    off = np.zeros((n, 5, 2))
    off[:, 0] = (10.5, 10.0)  # slot 0 within 3 ft
    off[:, 1] = (20.0, 10.0)
    cand = candidate_handlers(t, ball, off, cfg)
    assert (cand == 0).all()
    # ball in the air -> no candidate
    ball_air = ball.copy()
    ball_air[:, 2] = 12.0
    assert (candidate_handlers(t, ball_air, off, cfg) == -1).all()
    # sticky: a 5-frame blip to slot 1 is ignored, a 12-frame run is accepted
    seq = np.array([0] * 20 + [1] * 5 + [0] * 5 + [1] * 12 + [-1] * 3 + [-1] * 15, dtype=np.int8)
    out = sticky_handler(seq, cfg)
    assert out[:30].tolist() == [0] * 30
    assert out[30:42].tolist() == [1] * 12
    # the two -1 runs are one run of 18 frames (identical values) -> accepted as no handler
    assert out[42:].tolist() == [-1] * 18
    # catch-and-shoot: a 3-frame touch followed by flight is accepted
    cs = sticky_handler(np.array([0] * 20 + [1] * 3 + [-1] * 15, dtype=np.int8), cfg)
    assert cs[20:23].tolist() == [1, 1, 1] and cs[23:].tolist() == [-1] * 15
    grid = handler_on_grid(t, out, np.array([0, 1200, 1700, 2359]))
    assert grid.tolist() == [0, 1, -1, -1]  # t=1700 -> nearest raw frame 1680 (index 42) = -1
    assert last_handler_before(t, out, 2359) == 1
    assert last_handler_before(t, out, 100) == 0
