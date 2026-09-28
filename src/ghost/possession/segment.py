"""Half-court possession segmentation (v1 rules, decision D-009).

A *possession window* is delimited by consecutive *terminal* play-by-play events. Inside the
window (previous terminal, this terminal + ``end_pad_s``] the possession proper starts at the
**last** moment the ball crosses from the offence's backcourt into its frontcourt and stays
there. Windows are rejected (and counted) when:

* the terminal event cannot be placed on the tracking timeline (untracked clock reading),
* the offensive team cannot be determined (ball too close to midcourt for ambiguous events),
* the ball never crosses into the frontcourt inside the window (putbacks after offensive
  rebounds, and-one fouls, ...),
* the frontcourt stretch is shorter than ``min_frontcourt_s``,
* the frames between crossing and end are not continuous (dead ball / tracking hole / official
  clock correction),
* the on-court lineup changes or a player is missing from more than ``max_missing_frac`` of
  frames.

Possessions longer than ``max_window_s`` are cropped to their last ``max_window_s`` seconds
(flag ``cropped``). Possessions shorter than ``transition_s`` are flagged ``is_transition``
(excluded downstream in v1, for every terminal type - a superset of "shot within 4 s").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ghost.court import HALF, HOOP_LEFT, HOOP_RIGHT
from ghost.io.raw import BALL_ID
from ghost.possession.shots import ShotTimeConfig, refine_shot_release

TERMINAL_TYPES: dict[int, str] = {
    1: "fg_made",
    2: "fg_missed",
    5: "turnover",
    6: "foul",
    7: "violation",
    9: "timeout",
    13: "period_end",
}
SHOT_TYPES = (1, 2)
# terminal types whose pbp PLAYER1 team is the offence
OFFENSE_IS_P1 = (1, 2, 5)


@dataclass
class SegmentConfig:
    max_window_s: float = 24.0
    transition_s: float = 4.0
    end_pad_s: float = 0.5
    min_frontcourt_s: float = 1.0
    midcourt_margin_ft: float = 3.0  # ball within this of midcourt => offence ambiguous
    max_missing_frac: float = 0.05
    clock_bin_s: float = 1.0  # pbp clock is floored to whole seconds
    refine_shots: bool = True  # replace pbp time of FG attempts by the ball-derived release
    shot_cfg: ShotTimeConfig = field(default_factory=ShotTimeConfig)


@dataclass
class RejectCounts:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1


POSSESSION_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "possession_id": pl.Int32,
    "period": pl.Int8,
    "offense_team_id": pl.Int64,
    "defense_team_id": pl.Int64,
    "attacks_left": pl.Boolean,
    "t_start": pl.Int64,  # unix_ms of the (possibly cropped) crossing
    "t_cross": pl.Int64,  # unix_ms of the actual crossing
    "t_end": pl.Int64,  # unix_ms of terminal event + end_pad
    "t_terminal": pl.Int64,  # unix_ms of the terminal event (release time for shots)
    "t_pbp_terminal": pl.Int64,  # unix_ms matched to the pbp clock reading
    "shot_time_method": pl.Utf8,  # rim | apex | pbp (+ _nohand/_nooff) | null for non-shots
    "shooter_inferred_id": pl.Int64,  # offensive player holding the ball at release
    "duration_s": pl.Float32,
    "terminal_event_num": pl.Int32,
    "terminal_type": pl.Utf8,
    "terminal_msg_type": pl.Int16,
    "terminal_action_type": pl.Int16,
    "terminal_desc": pl.Utf8,
    "terminal_player_id": pl.Int64,
    "prev_terminal_event_num": pl.Int32,
    "game_clock_start": pl.Float32,
    "game_clock_end": pl.Float32,
    "shot_clock_start": pl.Float32,
    "score_margin_offense": pl.Int16,
    "is_transition": pl.Boolean,
    "cropped": pl.Boolean,
    "offense_player_ids": pl.List(pl.Int64),
    "defense_player_ids": pl.List(pl.Int64),
    "n_raw_frames": pl.Int32,
}


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------


def terminal_events(pbp_game: pl.DataFrame) -> pl.DataFrame:
    """Terminal pbp rows in event order with a forward-filled score margin (home - visitor)."""
    # margin *before* the event: forward-fill the scoring rows, then look at the previous row
    ff = pbp_game.sort("event_num").with_columns(
        pl.col("score_margin").forward_fill().shift(1).fill_null(0).alias("margin_home"),
    )
    return ff.filter(pl.col("msg_type").is_in(list(TERMINAL_TYPES))).select(
        [
            "event_num",
            "period",
            "pctime_sec",
            "msg_type",
            "action_type",
            "p1_team_id",
            "p1_id",
            "home_desc",
            "visitor_desc",
            "margin_home",
        ]
    )


def locate_clock(
    idx_period: pl.DataFrame, clock_s: float, after_unix: int | None, bin_s: float = 1.0
) -> int | None:
    """unix_ms of the frame best matching a floored pbp clock reading, after ``after_unix``.

    Candidates are frames with game_clock in [clock, clock + bin). Because official clock
    corrections can make the same reading occur twice, candidates are clustered by
    ``segment_id`` and the first cluster after ``after_unix`` is used; inside it the frame whose
    clock is closest to the bin centre is returned.
    """
    cand = idx_period.filter(
        (pl.col("game_clock") >= clock_s) & (pl.col("game_clock") < clock_s + bin_s)
    )
    if after_unix is not None:
        cand = cand.filter(pl.col("unix_ms") > after_unix)
    if cand.height == 0:
        # clock frozen exactly on the reading, or untracked: fall back to nearest reading
        cand = idx_period.filter((pl.col("game_clock") - clock_s).abs() <= bin_s)
        if after_unix is not None:
            cand = cand.filter(pl.col("unix_ms") > after_unix)
        if cand.height == 0:
            return None
    first_seg = cand["segment_id"].min()
    cand = cand.filter(pl.col("segment_id") == first_seg)
    centre = clock_s + bin_s / 2
    best = cand.with_columns((pl.col("game_clock") - centre).abs().alias("_d")).sort(
        ["_d", "unix_ms"]
    )
    return int(best["unix_ms"][0])


def _ball_x_at(entities_ball: pl.DataFrame, unix_ms: int, tol_ms: int = 500) -> float | None:
    near = entities_ball.filter((pl.col("unix_ms") - unix_ms).abs() <= tol_ms)
    if near.height == 0:
        return None
    return float(near["x"].median())


def _crossing_index(front: np.ndarray) -> int | None:
    """Index of the last back->front transition after which the ball stays in front."""
    if front.size == 0 or not front[-1]:
        return None
    # last index where front is False
    back_idx = np.flatnonzero(~front)
    if back_idx.size == 0:
        return None  # never in the backcourt inside the window
    return int(back_idx[-1] + 1)


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------


def segment_game(
    entities: pl.DataFrame,
    frame_index: pl.DataFrame,
    pbp_game: pl.DataFrame,
    direction: pl.DataFrame,
    home_team_id: int,
    cfg: SegmentConfig | None = None,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Segment one game into half-court possessions.

    Returns (possessions table with POSSESSION_SCHEMA, reject counts).
    """
    cfg = cfg or SegmentConfig()
    game_id = str(entities["game_id"][0])
    rejects = RejectCounts()
    ball = entities.filter(pl.col("team_id") == BALL_ID).select(["period", "unix_ms", "x"])
    players = entities.filter(pl.col("team_id") != BALL_ID).select(
        ["period", "unix_ms", "team_id", "player_id"]
    )
    dir_map = {
        (r["team_id"], r["period"]): r["attacks_left"] for r in direction.iter_rows(named=True)
    }
    teams = sorted({t for t, _ in dir_map})
    if len(teams) != 2:
        raise ValueError(f"{game_id}: expected two teams in direction table, got {teams}")
    other = {teams[0]: teams[1], teams[1]: teams[0]}

    ent_by_period = {
        int(p): entities.filter(pl.col("period") == p)
        for p in entities["period"].unique().to_list()
    }
    terms = terminal_events(pbp_game)
    records: list[dict] = []
    prev_unix: int | None = None
    prev_event: int | None = None
    prev_period: int | None = None
    pid = 0

    for row in terms.iter_rows(named=True):
        period = int(row["period"])
        if period != prev_period:
            prev_unix, prev_event, prev_period = None, None, period
        idx_p = frame_index.filter(pl.col("period") == period)
        if idx_p.height == 0:
            rejects.add("period_untracked")
            continue
        t_term = locate_clock(idx_p, float(row["pctime_sec"]), prev_unix, cfg.clock_bin_s)
        if t_term is None:
            rejects.add(f"terminal_not_located:{TERMINAL_TYPES[int(row['msg_type'])]}")
            continue
        t_end = t_term + int(cfg.end_pad_s * 1000)
        msg = int(row["msg_type"])

        # ---- offence
        ball_p = ball.filter(pl.col("period") == period)
        if msg in OFFENSE_IS_P1 and row["p1_team_id"] is not None:
            offense = int(row["p1_team_id"])
        else:
            bx = _ball_x_at(ball_p, t_term)
            if bx is None or abs(bx - HALF) < cfg.midcourt_margin_ft:
                rejects.add("offense_ambiguous")
                prev_unix, prev_event = t_term, int(row["event_num"])
                continue
            left_team = [t for t in teams if dir_map.get((t, period)) is True]
            if len(left_team) != 1:
                rejects.add("direction_missing")
                prev_unix, prev_event = t_term, int(row["event_num"])
                continue
            offense = left_team[0] if bx < HALF else other[left_team[0]]
        if offense not in other:
            rejects.add("offense_unknown_team")
            prev_unix, prev_event = t_term, int(row["event_num"])
            continue
        attacks_left = dir_map.get((offense, period))
        if attacks_left is None:
            rejects.add("direction_missing")
            prev_unix, prev_event = t_term, int(row["event_num"])
            continue

        # ---- shots: replace the pbp time by the ball-derived release time
        t_pbp = t_term
        method: str | None = None
        shooter_inf: int | None = None
        if msg in SHOT_TYPES:
            method = "pbp"
            if cfg.refine_shots:
                hoop = HOOP_LEFT if attacks_left else HOOP_RIGHT
                t_term, shooter_inf, method = refine_shot_release(
                    ent_by_period[period], t_pbp, offense, hoop, cfg.shot_cfg, t_min=prev_unix
                )
                t_end = t_term + int(cfg.end_pad_s * 1000)

        # ---- window frames and crossing
        w_start = prev_unix if prev_unix is not None else int(idx_p["unix_ms"].min())
        win = ball_p.filter((pl.col("unix_ms") > w_start) & (pl.col("unix_ms") <= t_end)).sort(
            "unix_ms"
        )
        if win.height == 0:
            rejects.add("no_ball_frames")
            prev_unix, prev_event = t_term, int(row["event_num"])
            continue
        x = win["x"].to_numpy()
        front = (x < HALF) if attacks_left else (x > HALF)
        ci = _crossing_index(front)
        if ci is None:
            # finer accounting: paired events with an (almost) empty window, second chances
            # that never left the frontcourt, or windows ending in the backcourt
            if (t_end - w_start) < 1500:
                why = "no_crossing:short_window"
            elif front.all():
                why = "no_crossing:always_frontcourt"
            else:
                why = "no_crossing:ends_backcourt"
            rejects.add(f"{why}:{TERMINAL_TYPES[msg]}")
            prev_unix, prev_event = t_term, int(row["event_num"])
            continue
        t_cross = int(win["unix_ms"][ci])
        if (t_end - t_cross) < cfg.min_frontcourt_s * 1000:
            rejects.add("frontcourt_too_short")
            prev_unix, prev_event = t_term, int(row["event_num"])
            continue
        cropped = False
        t_start = t_cross
        if (t_end - t_cross) > cfg.max_window_s * 1000:
            t_start = t_end - int(cfg.max_window_s * 1000)
            cropped = True

        # ---- continuity and lineup
        fi = idx_p.filter((pl.col("unix_ms") >= t_start) & (pl.col("unix_ms") <= t_end))
        if fi.height < 2 or fi["segment_id"].n_unique() != 1:
            rejects.add("gap_in_window")
            prev_unix, prev_event = t_term, int(row["event_num"])
            continue
        pw = players.filter(
            (pl.col("period") == period)
            & (pl.col("unix_ms") >= t_start)
            & (pl.col("unix_ms") <= t_end)
        )
        n_frames = fi.height
        presence = pw.group_by(["team_id", "player_id"]).len().sort(["team_id", "player_id"])
        good = presence.filter(pl.col("len") >= (1 - cfg.max_missing_frac) * n_frames)
        off_ids = good.filter(pl.col("team_id") == offense)["player_id"].to_list()
        def_ids = good.filter(pl.col("team_id") == other[offense])["player_id"].to_list()
        if presence.height != 10 or len(off_ids) != 5 or len(def_ids) != 5:
            rejects.add("lineup_change" if presence.height > 10 else "missing_players")
            prev_unix, prev_event = t_term, int(row["event_num"])
            continue

        # ---- context
        gc_start = float(fi["game_clock"][0])
        gc_end = float(fi["game_clock"][-1])
        sc_start = fi["shot_clock"][0]
        margin_home = int(row["margin_home"])  # before the terminal event (see terminal_events)
        margin_off = margin_home if offense == home_team_id else -margin_home
        duration = (t_end - t_start) / 1000.0
        pid += 1
        records.append(
            {
                "game_id": game_id,
                "possession_id": pid,
                "period": period,
                "offense_team_id": offense,
                "defense_team_id": other[offense],
                "attacks_left": bool(attacks_left),
                "t_start": t_start,
                "t_cross": t_cross,
                "t_end": t_end,
                "t_terminal": t_term,
                "t_pbp_terminal": t_pbp,
                "shot_time_method": method,
                "shooter_inferred_id": shooter_inf,
                "duration_s": duration,
                "terminal_event_num": int(row["event_num"]),
                "terminal_type": TERMINAL_TYPES[msg],
                "terminal_msg_type": msg,
                "terminal_action_type": row["action_type"],
                "terminal_desc": row["home_desc"] or row["visitor_desc"],
                "terminal_player_id": row["p1_id"] if msg in OFFENSE_IS_P1 else None,
                "prev_terminal_event_num": prev_event,
                "game_clock_start": gc_start,
                "game_clock_end": gc_end,
                "shot_clock_start": None if sc_start is None else float(sc_start),
                "score_margin_offense": margin_off,
                "is_transition": (t_end - t_cross) < cfg.transition_s * 1000,
                "cropped": cropped,
                "offense_player_ids": sorted(off_ids),
                "defense_player_ids": sorted(def_ids),
                "n_raw_frames": n_frames,
            }
        )
        prev_unix, prev_event = t_term, int(row["event_num"])

    table = pl.DataFrame(records, schema=POSSESSION_SCHEMA)
    return table, rejects.counts
