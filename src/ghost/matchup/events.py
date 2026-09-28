"""Switch, help and closeout events derived from matchup paths (Phase 2 task 3).

All computations run on the 5 Hz possession grid (left half). Screens and passes are **inputs**
from nbacore (>= v1.1); this module never detects them itself (D-016).

Definitions (both switch variants are computed; A is reported, D-018):

* ``stable_path``: a defender's matchup run shorter than ``min_hold_s`` inherits the previous
  stable matchup (removes 1-2 step flickers of the Viterbi path).
* **Switch B** (reassignment): a defender's stable matchup changes from attacker a to attacker b
  (both man states) and the new matchup holds >= ``min_hold_s`` (1 s).
* **Switch A** (screen switch): around a screen (|t - t_contact| <= ``window_s``, 1 s) the
  screened defender changes user -> screener *and* the screener's defender changes
  screener -> user (both are switch-B changes). If only one of the two changes it is recorded as
  ``one_sided`` (hedge / show / scram, not a switch). Switch-B changes not explained by a screen
  are labelled ``rotation``.
* **Help B** (geometric, primary): defender j, not guarding the ball handler, is >= ``help_leave_ft``
  (6 ft) from his stable matchup, closes on the handler at >= ``help_speed_fts`` (3 ft/s), while the
  handler is within ``help_zone_ft`` (20 ft) of the hoop; sustained >= ``help_min_s`` (0.4 s). The
  onset is the first step of the run.
* **Help A** (model): posterior of the HMM help state >= 0.5 for >= 0.4 s while closing on the
  handler. Needs a help-state model.
* **Closeout**: after a completed pass, the receiver's defender (stable matchup at the catch;
  nearest defender if nobody is matched) is > ``co_far_ft`` (8 ft) away at the catch, gets within
  ``co_near_ft`` (4 ft) within ``co_window_s`` (1 s), at a mean closing speed >= ``co_speed_fts``
  (8 ft/s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ghost.court import HOOP_LEFT

HELP = 5  # help state index in the matchup paths

# Required input columns (nbacore >= v1.1 event schema)
SCREEN_COLUMNS = [
    "game_id",
    "event_uid",
    "screener_id",
    "user_id",
    "screened_def_id",
    "screener_def_id",
    "t_contact_ms",
    "on_ball",
]
PASS_COLUMNS = [
    "game_id",
    "event_uid",
    "passer_id",
    "receiver_id",
    "t_release_ms",
    "t_catch_ms",
    "completed",
]


@dataclass
class EventConfig:
    hz: float = 5.0
    min_hold_s: float = 1.0
    window_s: float = 1.0
    help_leave_ft: float = 6.0
    help_speed_fts: float = 3.0
    help_zone_ft: float = 20.0
    help_min_s: float = 0.4
    help_post: float = 0.5
    co_far_ft: float = 8.0
    co_near_ft: float = 4.0
    co_window_s: float = 1.0
    co_speed_fts: float = 8.0

    def steps(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.hz)))


def _require(df: pl.DataFrame, cols: list[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} table lacks columns {missing} (nbacore >= v1.1 event schema)")


# ------------------------------------------------------------------------------------------
# stable paths and switch B
# ------------------------------------------------------------------------------------------


def stable_path(path: np.ndarray, valid: np.ndarray, min_hold: int) -> np.ndarray:
    """(D, T) matchup path of one possession -> flicker-free path (-1 on invalid steps)."""
    D, T = path.shape
    n = int(valid.sum())
    out = np.full((D, T), -1, dtype=np.int16)
    for d in range(D):
        seq = path[d, :n].astype(np.int16)
        if n == 0:
            continue
        change = np.flatnonzero(np.diff(seq)) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [n]])
        current = int(seq[0])  # the first run is accepted whatever its length
        for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
            if i == 0 or e - s >= min_hold:
                current = int(seq[s])
            out[d, s:e] = current
    return out


def switch_b_events(
    stable: np.ndarray, t_ms: np.ndarray, off_ids: np.ndarray, def_ids: np.ndarray
) -> list[dict]:
    """Man-to-man reassignments of one possession (help entries/exits are not switches)."""
    events = []
    D, T = stable.shape
    for d in range(D):
        s = stable[d]
        for k in range(1, T):
            a, b = int(s[k - 1]), int(s[k])
            if a < 0 or b < 0 or a == b or a == HELP or b == HELP:
                continue
            events.append(
                {
                    "def_id": int(def_ids[d]),
                    "from_off_id": int(off_ids[a]),
                    "to_off_id": int(off_ids[b]),
                    "t_ms": int(t_ms[k]),
                    "step": k,
                }
            )
    return events


def classify_switches(
    switches: pl.DataFrame, screens: pl.DataFrame, cfg: EventConfig | None = None
) -> pl.DataFrame:
    """Label switch-B events of one possession as screen_switch / one_sided / rotation.

    ``switches``: output of :func:`switch_b_events` as a frame. ``screens``: nbacore
    ``screen_candidate`` rows of the same possession window.
    Returns one row per switch-B event (``event_class``, ``screen_uid``) plus one row per
    ``one_sided`` screen reaction that matched only one side.
    """
    cfg = cfg or EventConfig()
    _require(screens, SCREEN_COLUMNS, "screen")
    win = int(cfg.window_s * 1000)
    sw = switches.with_columns(
        pl.lit("rotation").alias("event_class"), pl.lit(None, dtype=pl.Utf8).alias("screen_uid")
    )
    rows = sw.to_dicts()
    extra = []
    for sc in screens.iter_rows(named=True):
        near = [i for i, r in enumerate(rows) if abs(r["t_ms"] - sc["t_contact_ms"]) <= win]
        a = [  # screened defender: user -> screener
            i
            for i in near
            if rows[i]["def_id"] == sc["screened_def_id"]
            and rows[i]["from_off_id"] == sc["user_id"]
            and rows[i]["to_off_id"] == sc["screener_id"]
        ]
        b = [  # screener's defender: screener -> user
            i
            for i in near
            if rows[i]["def_id"] == sc["screener_def_id"]
            and rows[i]["from_off_id"] == sc["screener_id"]
            and rows[i]["to_off_id"] == sc["user_id"]
        ]
        if a and b:
            for i in (a[0], b[0]):
                rows[i]["event_class"] = "screen_switch"
                rows[i]["screen_uid"] = sc["event_uid"]
        elif a or b:
            i = (a or b)[0]
            if rows[i]["event_class"] == "screen_switch":
                continue  # already half of a screen switch; an overlapping candidate must not downgrade it
            rows[i]["event_class"] = "one_sided"
            rows[i]["screen_uid"] = sc["event_uid"]
    schema = {
        "def_id": pl.Int64,
        "from_off_id": pl.Int64,
        "to_off_id": pl.Int64,
        "t_ms": pl.Int64,
        "step": pl.Int32,
        "event_class": pl.Utf8,
        "screen_uid": pl.Utf8,
    }
    return pl.DataFrame(rows + extra, schema=schema)


# ------------------------------------------------------------------------------------------
# help
# ------------------------------------------------------------------------------------------


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of True runs."""
    if not mask.any():
        return []
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def help_b_events(
    stable: np.ndarray,
    handler: np.ndarray,
    off_xy: np.ndarray,
    def_xy: np.ndarray,
    valid: np.ndarray,
    t_ms: np.ndarray,
    off_ids: np.ndarray,
    def_ids: np.ndarray,
    cfg: EventConfig | None = None,
    hoop: np.ndarray = HOOP_LEFT,
) -> list[dict]:
    """Geometric help events of one possession. Arrays: stable (D,T), handler (T,),
    off_xy (T,5,2), def_xy (T,5,2), valid (T,), t_ms (T,)."""
    cfg = cfg or EventConfig()
    T = valid.shape[0]
    dt = 1.0 / cfg.hz
    h = handler.astype(int)
    has_h = (h >= 0) & valid
    hxy = np.where(has_h[:, None], off_xy[np.arange(T), np.clip(h, 0, 4)], np.nan)
    h_zone = np.linalg.norm(hxy - hoop, axis=-1) <= cfg.help_zone_ft
    out = []
    for d in range(def_xy.shape[1]):
        m = stable[d].astype(int)
        man = (m >= 0) & (m < 5)
        man_xy = np.where(man[:, None], off_xy[np.arange(T), np.clip(m, 0, 4)], np.nan)
        d_man = np.linalg.norm(def_xy[:, d] - man_xy, axis=-1)
        d_h = np.linalg.norm(def_xy[:, d] - hxy, axis=-1)
        closing = np.full(T, np.nan)
        closing[1:] = -(d_h[1:] - d_h[:-1]) / dt  # ft/s toward the handler
        cond = (
            has_h
            & man
            & (m != h)
            & h_zone
            & (d_man >= cfg.help_leave_ft)
            & (closing >= cfg.help_speed_fts)
        )
        cond = np.nan_to_num(cond, nan=False).astype(bool)
        for s, e in _runs(cond):
            if e - s < cfg.steps(cfg.help_min_s):
                continue
            out.append(
                {
                    "def_id": int(def_ids[d]),
                    "man_off_id": int(off_ids[m[s]]),
                    "handler_id": int(off_ids[h[s]]),
                    "t_start_ms": int(t_ms[s]),
                    "t_end_ms": int(t_ms[e - 1]),
                    "step_start": int(s),
                    "max_dist_from_man_ft": float(np.nanmax(d_man[s:e])),
                    "min_dist_to_handler_ft": float(np.nanmin(d_h[s:e])),
                    "definition": "B",
                }
            )
    return out


def help_a_events(
    p_help: np.ndarray,
    handler: np.ndarray,
    off_xy: np.ndarray,
    def_xy: np.ndarray,
    valid: np.ndarray,
    t_ms: np.ndarray,
    def_ids: np.ndarray,
    off_ids: np.ndarray,
    cfg: EventConfig | None = None,
) -> list[dict]:
    """Model-based help: help-state posterior >= cfg.help_post while closing on the handler."""
    cfg = cfg or EventConfig()
    T = valid.shape[0]
    h = handler.astype(int)
    has_h = (h >= 0) & valid
    hxy = np.where(has_h[:, None], off_xy[np.arange(T), np.clip(h, 0, 4)], np.nan)
    out = []
    for d in range(def_xy.shape[1]):
        d_h = np.linalg.norm(def_xy[:, d] - hxy, axis=-1)
        closing = np.full(T, np.nan)
        closing[1:] = -(d_h[1:] - d_h[:-1]) * cfg.hz
        cond = has_h & (p_help[d] >= cfg.help_post) & (np.nan_to_num(closing) > 0)
        for s, e in _runs(cond):
            if e - s < cfg.steps(cfg.help_min_s):
                continue
            out.append(
                {
                    "def_id": int(def_ids[d]),
                    "handler_id": int(off_ids[h[s]]),
                    "t_start_ms": int(t_ms[s]),
                    "t_end_ms": int(t_ms[e - 1]),
                    "step_start": int(s),
                    "definition": "A",
                }
            )
    return out


# ------------------------------------------------------------------------------------------
# closeouts
# ------------------------------------------------------------------------------------------


def closeout_events(
    passes: pl.DataFrame,
    stable: np.ndarray,
    off_xy: np.ndarray,
    def_xy: np.ndarray,
    valid: np.ndarray,
    t_ms: np.ndarray,
    off_ids: np.ndarray,
    def_ids: np.ndarray,
    cfg: EventConfig | None = None,
) -> list[dict]:
    """Closeouts after the completed passes of one possession (nbacore ``pass`` rows)."""
    cfg = cfg or EventConfig()
    _require(passes, PASS_COLUMNS, "pass")
    n = int(valid.sum())
    win = cfg.steps(cfg.co_window_s)
    out = []
    for p in passes.filter(pl.col("completed")).iter_rows(named=True):
        rid = p["receiver_id"]
        if rid is None or rid not in set(off_ids.tolist()):
            continue
        r = int(np.flatnonzero(off_ids == rid)[0])
        c = int(np.searchsorted(t_ms[:n], p["t_catch_ms"]))
        if c >= n:
            continue
        guards = np.flatnonzero(stable[:, c] == r)
        dist_c = np.linalg.norm(def_xy[c] - off_xy[c, r], axis=-1)  # (5,)
        d = int(guards[np.argmin(dist_c[guards])]) if guards.size else int(dist_c.argmin())
        e = min(n, c + win + 1)
        dd = np.linalg.norm(def_xy[c:e, d] - off_xy[c:e, r], axis=-1)
        if dd[0] <= cfg.co_far_ft or not (dd <= cfg.co_near_ft).any():
            continue
        k = int(np.flatnonzero(dd <= cfg.co_near_ft)[0])
        speed = (dd[0] - dd[k]) * cfg.hz / max(k, 1)
        if speed < cfg.co_speed_fts:
            continue
        out.append(
            {
                "def_id": int(def_ids[d]),
                "receiver_id": int(rid),
                "pass_uid": p["event_uid"],
                "t_catch_ms": int(p["t_catch_ms"]),
                "t_closed_ms": int(t_ms[c + k]),
                "dist_at_catch_ft": float(dd[0]),
                "dist_closed_ft": float(dd[k]),
                "closing_speed_fts": float(speed),
                "matched_defender": bool(guards.size),
            }
        )
    return out
