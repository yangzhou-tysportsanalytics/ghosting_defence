"""Shot features at any tracking time: the single implementation behind the shot table and the
xFG interface (``ghost.xfg.api``).

At a time t, for a would-be shooter s of team T (offence attacking the left basket after
mapping):

- geometry (25 Hz frame at t): shot location, ``dist_ft`` and ``angle_deg`` to the left hoop,
  ``is_three``; ``def1_ft`` / ``def2_ft`` = nearest / second-nearest opponent; ``def_front_ft`` =
  nearest opponent within 45 degrees of the shooter -> hoop direction (30 ft if none);
- touch state: the shooter's last ``possession_touch`` with t - 15 s <= t_start <= t and
  t_end >= t - 1 s; ``n_dribbles`` = dribbles of that touch whose ``t_start_ms`` <= t (so far),
  ``touch_s`` = t - touch start; both null without such a touch;
- ``catch_and_shoot`` = no dribble so far (null counts as 0) and touch_s < 2 s (False without
  a touch); ``zone`` from distance, three-point location and the corner band.
"""

from __future__ import annotations

import numpy as np

from ghost.court import HOOP_LEFT, is_three_point_location, to_left_half
from ghost.io.raw import BALL_ID

TOUCH_LOOKBACK_MS = 15_000
TOUCH_END_SLACK_MS = 1_000
CATCH_SHOOT_S = 2.0
FRONT_CONE_RAD = np.pi / 4
NO_FRONT_FT = 30.0


def zone_of(dist: np.ndarray, three: np.ndarray, y: np.ndarray) -> np.ndarray:
    corner = three & (np.abs(y - 25.0) >= 21.0)
    z = np.where(dist < 4, "rim", np.where(dist < 14, "paint", "mid"))
    return np.where(three, np.where(corner, "corner3", "above3"), z)


def geometry(
    team_ids: np.ndarray,
    player_ids: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    shooter_id: int,
    shooter_team_id: int,
    attacks_left: bool,
) -> dict | None:
    """Features of one frame (rows = entities of one timestamp). None if the frame is incomplete
    (< 10 entities) or the shooter is not on court."""
    if len(team_ids) < 10:
        return None
    # float64 throughout: in float32 a defender on the 45-degree cone edge can flip in or out
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    me = np.flatnonzero(player_ids == shooter_id)
    if me.size == 0:
        return None
    sx, sy = to_left_half(xs[me[:1]], ys[me[:1]], attacks_left)
    opp = (team_ids != BALL_ID) & (team_ids != shooter_team_id)
    dx, dy = to_left_half(xs[opp], ys[opp], attacks_left)
    s = np.array([sx[0], sy[0]])
    dd = np.hypot(dx - s[0], dy - s[1])
    order = np.sort(dd)
    to_rim = HOOP_LEFT - s
    dist = float(np.linalg.norm(to_rim))
    vecs = np.column_stack([dx - s[0], dy - s[1]])
    cosang = (vecs @ to_rim) / np.maximum(np.linalg.norm(vecs, axis=1) * max(dist, 1e-6), 1e-6)
    front = dd[cosang >= np.cos(FRONT_CONE_RAD)]
    return {
        "x": float(s[0]),
        "y": float(s[1]),
        "dist_ft": dist,
        "angle_deg": float(
            np.degrees(np.arctan2(abs(s[1] - 25.0), max(s[0] - HOOP_LEFT[0], 1e-3)))
        ),
        "is_three": bool(is_three_point_location(np.array([s[0]]), np.array([s[1]]), True)[0]),
        "def1_ft": float(order[0]) if order.size else None,
        "def2_ft": float(order[1]) if order.size > 1 else None,
        "def_front_ft": float(front.min()) if front.size else NO_FRONT_FT,
    }


def touch_state(
    touch_actor: np.ndarray,
    touch_start: np.ndarray,
    touch_end: np.ndarray,
    touch_uid: np.ndarray,
    dribble_touch_uid: np.ndarray,
    dribble_start: np.ndarray,
    shooter_id: int,
    t_ms: int,
) -> tuple[int | None, float | None]:
    """(n_dribbles so far, touch_s) of the shooter's current touch at t (see module docstring)."""
    sel = np.flatnonzero(
        (touch_actor == shooter_id)
        & (touch_start <= t_ms)
        & (touch_start >= t_ms - TOUCH_LOOKBACK_MS)
        & (touch_end >= t_ms - TOUCH_END_SLACK_MS)
    )
    if sel.size == 0:
        return None, None
    k = sel[np.argmax(touch_start[sel])]  # the latest such touch
    n = int(np.sum((dribble_touch_uid == touch_uid[k]) & (dribble_start <= t_ms)))
    return n, (t_ms - int(touch_start[k])) / 1000


def catch_and_shoot(n_dribbles, touch_s) -> bool:
    return (0 if n_dribbles is None else n_dribbles) == 0 and (
        9.0 if touch_s is None else touch_s
    ) < CATCH_SHOOT_S
