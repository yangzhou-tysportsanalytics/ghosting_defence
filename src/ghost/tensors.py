"""Dense possession tensors from the 5 Hz long frame table (Phase 1 output).

The long table has 11 rows per step (``role`` ball / off / def, ``slot`` 0-4, players ordered by
player_id). This module stacks possessions into fixed-shape numpy arrays, left-half coordinates:

    off_xy  (N, T, 5, 2)    def_xy (N, T, 5, 2)    ball (N, T, 3)
    off_v   (N, T, 5, 2)    def_v  (N, T, 5, 2)    ball_v (N, T, 2)
    t       (N, T) int64 unix_ms of each grid step
    valid   (N, T) bool     handler (N, T) int8 offence slot or -1
    game_clock, shot_clock (N, T) float32 (shot clock NaN when off)
    off_ids, def_ids (N, 5) int64;  keys: game_id / possession_id per row
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

T_STEPS = 121


@dataclass
class PossessionTensors:
    game_id: np.ndarray
    possession_id: np.ndarray
    t: np.ndarray  # (N, T) int64 unix_ms of the grid points
    valid: np.ndarray
    ball: np.ndarray
    ball_v: np.ndarray
    off_xy: np.ndarray
    off_v: np.ndarray
    def_xy: np.ndarray
    def_v: np.ndarray
    handler: np.ndarray
    game_clock: np.ndarray
    shot_clock: np.ndarray
    off_ids: np.ndarray
    def_ids: np.ndarray
    shot_clock_imputed: np.ndarray | None = None  # (N, T) bool, set by fill_shot_clock
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.valid.shape[0])

    def subset(self, idx: np.ndarray) -> PossessionTensors:
        kw = {
            k: (None if getattr(self, k) is None else getattr(self, k)[idx])
            for k in self.__dataclass_fields__
            if k != "meta"
        }
        return PossessionTensors(**kw, meta=dict(self.meta))


def from_long(frames: pl.DataFrame, n_steps: int = T_STEPS) -> PossessionTensors:
    """Stack a long 5 Hz frame table (one or many games) into tensors."""
    f = frames.sort(["game_id", "possession_id", "step", "role", "slot"])
    keys = f.select(["game_id", "possession_id"]).unique(maintain_order=True)
    n = keys.height
    rows_per = n_steps * 11
    if f.height != n * rows_per:
        raise ValueError(f"expected {n} x {rows_per} rows, got {f.height}")
    # after sorting by role name: "ball" < "def" < "off"  -> order ball, def0..4, off0..4
    role = f["role"].to_numpy().reshape(n, n_steps, 11)
    if not (role[..., 0] == "ball").all() or not (role[..., 1:6] == "def").all():
        raise ValueError("unexpected role ordering in frame table")
    x = f["x"].to_numpy().reshape(n, n_steps, 11).astype(np.float32)
    y = f["y"].to_numpy().reshape(n, n_steps, 11).astype(np.float32)
    z = f["z"].to_numpy().reshape(n, n_steps, 11).astype(np.float32)
    vx = f["vx"].to_numpy().reshape(n, n_steps, 11).astype(np.float32)
    vy = f["vy"].to_numpy().reshape(n, n_steps, 11).astype(np.float32)
    xy = np.stack([x, y], axis=-1)
    v = np.stack([vx, vy], axis=-1)
    is_h = f["is_handler"].to_numpy().reshape(n, n_steps, 11)[..., 6:11]
    handler = np.where(is_h.any(-1), is_h.argmax(-1), -1).astype(np.int8)
    pid = f["player_id"].to_numpy().reshape(n, n_steps, 11)[:, 0, :]
    sc = f["shot_clock"].cast(pl.Float32).fill_null(np.nan).to_numpy()
    return PossessionTensors(
        game_id=keys["game_id"].to_numpy(),
        possession_id=keys["possession_id"].to_numpy(),
        t=f["t_unix"].to_numpy().reshape(n, n_steps, 11)[..., 0].astype(np.int64),
        valid=f["valid"].to_numpy().reshape(n, n_steps, 11)[..., 0],
        ball=np.concatenate([xy[:, :, 0], z[:, :, 0:1]], axis=-1),
        ball_v=v[:, :, 0],
        def_xy=xy[:, :, 1:6],
        def_v=v[:, :, 1:6],
        off_xy=xy[:, :, 6:11],
        off_v=v[:, :, 6:11],
        handler=handler,
        game_clock=f["game_clock"].to_numpy().reshape(n, n_steps, 11)[..., 0].astype(np.float32),
        shot_clock=sc.reshape(n, n_steps, 11)[..., 0].astype(np.float32),
        off_ids=pid[:, 6:11].astype(np.int64),
        def_ids=pid[:, 1:6].astype(np.int64),
    )


def load_game_set(
    frames_dir: str | Path,
    game_ids: list[str],
    possessions: pl.DataFrame | None = None,
) -> PossessionTensors:
    """Load and stack the 5 Hz frames of ``game_ids``; optionally keep only the possessions in
    ``possessions`` (e.g. half-court only), matched on (game_id, possession_id)."""
    parts = []
    for gid in game_ids:
        p = Path(frames_dir) / f"{gid}.parquet"
        if not p.exists():
            continue
        fr = pl.read_parquet(p)
        if possessions is not None:
            keep = possessions.filter(pl.col("game_id") == gid).select(["game_id", "possession_id"])
            fr = fr.join(keep, on=["game_id", "possession_id"], how="semi")
        if fr.height:
            parts.append(fr)
    if not parts:
        raise ValueError("no frames loaded")
    return from_long(pl.concat(parts))


def fill_shot_clock(pt: PossessionTensors, shot_clock_by_game: dict[str, pl.DataFrame]) -> None:
    """Replace ``pt.shot_clock`` by nbacore's per-frame filled shot clock (v1.2) at the nearest
    25 Hz frame of each grid step, and set ``pt.shot_clock_imputed``. In place.

    ``shot_clock_by_game[game_id]`` has columns period, unix_ms, shot_clock_filled,
    shot_clock_imputed (``nbacore.load.shot_clock``). Steps without a value stay NaN.
    """
    N, T = pt.valid.shape
    sc = np.full((N, T), np.nan, dtype=np.float32)
    imp = np.zeros((N, T), dtype=bool)
    for gid in np.unique(pt.game_id):
        tab = shot_clock_by_game[str(gid)].sort("unix_ms")
        u = tab["unix_ms"].to_numpy()
        val = tab["shot_clock_filled"].cast(pl.Float32).fill_null(np.nan).to_numpy()
        im = tab["shot_clock_imputed"].fill_null(False).to_numpy()
        rows = np.flatnonzero(pt.game_id == gid)
        tt = pt.t[rows]  # (n, T)
        j = np.searchsorted(u, tt).clip(1, len(u) - 1)
        left = j - 1
        pick = np.where(np.abs(tt - u[left]) <= np.abs(u[j] - tt), left, j)
        sc[rows] = val[pick]
        imp[rows] = im[pick]
    sc[~pt.valid] = np.nan
    pt.shot_clock = sc
    pt.shot_clock_imputed = imp & pt.valid
