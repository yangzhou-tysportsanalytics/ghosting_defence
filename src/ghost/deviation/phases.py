"""Per-step context of a defender (D-020): rule-ghost deviation, phase, sightline cone, paint time.

All inputs are one possession on the 5 Hz grid in left-half coordinates (offence attacks the left
basket). Pure functions; the script ``phase4_phase_context.py`` feeds them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ghost.court import HOOP_LEFT, HOOP_Y, LANE_LENGTH, LANE_WIDTH

PHASES = ["help", "closeout", "recovery", "screen", "pre_screen", "other"]  # priority order
STEP_S = 0.2


@dataclass(frozen=True)
class PhaseDefs:
    screen_before_s: float = 0.5
    screen_after_s: float = 1.0
    recovery_s: float = 2.0
    cone_half_angle_deg: float = 20.0
    paint_long_s: float = 2.5

    @classmethod
    def variant(cls, name: str) -> PhaseDefs:
        if name == "A":
            return cls()
        if name == "B":
            return cls(
                screen_before_s=1.0, screen_after_s=2.0, recovery_s=3.0, cone_half_angle_deg=30.0
            )
        raise ValueError(name)


def rule_ghost_deviation(
    def_xy: np.ndarray,
    off_xy: np.ndarray,
    ball_xy: np.ndarray,
    matchup: np.ndarray,
    valid: np.ndarray,
    g_strong: np.ndarray,
    g_weak: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(|d|, sag), each (5, T), NaN where the defender has no stable man (help state included).

    Rule ghost of defender d guarding attacker k: mu = g_o O_k + g_b B + g_h H with strong / weak
    side weights (ball on the attacker's side of the court's long axis = strong); sag = d . u with
    u the unit vector from the attacker to the hoop."""
    T = valid.shape[0]
    dev, sag = np.full((5, T), np.nan), np.full((5, T), np.nan)
    for d in range(5):
        m = matchup[d].astype(int)
        tt = np.flatnonzero((m >= 0) & (m < 5) & valid)
        if tt.size == 0:
            continue
        om, B = off_xy[tt, m[tt]], ball_xy[tt]
        strong = np.sign(om[:, 1] - HOOP_Y) == np.sign(B[:, 1] - HOOP_Y)
        g = np.where(strong[:, None], g_strong, g_weak)
        mu = g[:, 0:1] * om + g[:, 1:2] * B + g[:, 2:3] * HOOP_LEFT
        dv = def_xy[tt, d] - mu
        u = HOOP_LEFT - om
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
        dev[d, tt] = np.linalg.norm(dv, axis=1)
        sag[d, tt] = (dv * u).sum(1)
    return dev, sag


def phase_labels(
    t_ms: np.ndarray,
    help_iv: list[list[tuple[int, int]]],
    closeout_iv: list[list[tuple[int, int]]],
    screen_contacts: list[int],
    defs: PhaseDefs,
) -> np.ndarray:
    """(5, T) index into PHASES. ``help_iv`` / ``closeout_iv``: per defender slot, (start, end)
    unix-ms intervals; ``screen_contacts``: unix-ms contact times of the possession's on-ball
    screens. Higher-priority phases overwrite lower ones."""
    T = t_ms.shape[0]
    lab = np.full((5, T), PHASES.index("other"), dtype=np.int8)
    pre, scr = np.zeros(T, bool), np.zeros(T, bool)
    if screen_contacts:
        first = min(screen_contacts) - defs.screen_before_s * 1000
        pre = t_ms < first
        for c in screen_contacts:
            scr |= (t_ms >= c - defs.screen_before_s * 1000) & (
                t_ms <= c + defs.screen_after_s * 1000
            )
    lab[:, pre] = PHASES.index("pre_screen")
    lab[:, scr] = PHASES.index("screen")
    for d in range(5):
        rec, clo, hlp = np.zeros(T, bool), np.zeros(T, bool), np.zeros(T, bool)
        for s, e in closeout_iv[d]:
            clo |= (t_ms >= s) & (t_ms <= e)
            rec |= (t_ms > e) & (t_ms <= e + defs.recovery_s * 1000)
        for s, e in help_iv[d]:
            hlp |= (t_ms >= s) & (t_ms <= e)
            rec |= (t_ms > e) & (t_ms <= e + defs.recovery_s * 1000)
        lab[d, rec] = PHASES.index("recovery")
        lab[d, clo] = PHASES.index("closeout")
        lab[d, hlp] = PHASES.index("help")
    return lab


def in_sightline_cone(
    handler_xy: np.ndarray, def_xy: np.ndarray, half_angle_deg: float
) -> np.ndarray:
    """(T, 5): defender inside the cone from the ball handler toward the hoop (half-angle given)
    and no farther from the handler than the handler is from the hoop. ``handler_xy`` (T, 2) may
    hold NaN (no handler) -> False."""
    to_hoop = HOOP_LEFT - handler_xy  # (T, 2)
    reach = np.linalg.norm(to_hoop, axis=1)  # (T,)
    v = def_xy - handler_xy[:, None, :]  # (T, 5, 2)
    dist = np.linalg.norm(v, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v @ to_hoop[:, :, None])[..., 0] / (dist * reach[:, None])
        out = (cos >= np.cos(np.radians(half_angle_deg))) & (dist <= reach[:, None]) & (dist > 0)
    return np.nan_to_num(out, nan=False).astype(bool)


def in_paint(def_xy: np.ndarray) -> np.ndarray:
    """(T, 5): inside the lane of the left basket."""
    x, y = def_xy[..., 0], def_xy[..., 1]
    return (x >= 0) & (x <= LANE_LENGTH) & (np.abs(y - HOOP_Y) <= LANE_WIDTH / 2)


def long_paint_runs(
    paint: np.ndarray, valid: np.ndarray, long_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per defender: (T, 5) flag of steps inside a continuous paint stretch lasting >= ``long_s``,
    and (5,) longest continuous stretch in seconds. Invalid steps break a stretch."""
    T = paint.shape[0]
    flag, longest = np.zeros_like(paint, dtype=bool), np.zeros(paint.shape[1])
    need = int(np.ceil(long_s / STEP_S - 1e-9))
    for d in range(paint.shape[1]):
        p = paint[:, d] & valid
        t = 0
        while t < T:
            if not p[t]:
                t += 1
                continue
            s = t
            while t < T and p[t]:
                t += 1
            n = t - s
            longest[d] = max(longest[d], n * STEP_S)
            if n >= need:
                flag[s:t, d] = True
    return flag, longest
