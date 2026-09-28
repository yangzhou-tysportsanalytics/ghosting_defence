"""Single access point to the shared nbacore release (decision D-016).

All L1 inputs (deduped 25 Hz frames, frame index, rosters, attack direction, typed pbp, tracking
event windows) and the fixed L4 split come from ``nbacore.load`` at the version pinned in
``configs/data.yaml``. Research code in this repo should import from here, not from
``nbacore.load`` directly, so that the pinned version is applied everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nbacore.load as L
import polars as pl
from omegaconf import DictConfig, OmegaConf

CONFIG_PATH = Path("configs/data.yaml")


@dataclass(frozen=True)
class DataConfig:
    version: str
    game_set: str
    windows_release: str
    game_sets_dir: Path
    processed_dir: Path
    seed: int

    @classmethod
    def load(cls, path: str | Path = CONFIG_PATH, **overrides) -> DataConfig:
        cfg: DictConfig = OmegaConf.load(path)
        vals = {
            "version": str(cfg.nbacore_data_version),
            "windows_release": str(cfg.get("windows_release", "ghost")),
            "game_set": str(cfg.game_set),
            "game_sets_dir": Path(cfg.game_sets_dir),
            "processed_dir": Path(cfg.processed_dir),
            "seed": int(cfg.seed),
        }
        vals.update({k: v for k, v in overrides.items() if v is not None})
        vals["game_sets_dir"] = Path(vals["game_sets_dir"])
        vals["processed_dir"] = Path(vals["processed_dir"])
        if vals["version"] == "latest":
            raise ValueError("pin a concrete nbacore data version (e.g. v1.0), not 'latest'")
        return cls(**vals)


def games(cfg: DataConfig) -> pl.DataFrame:
    """Games table of the pinned release (all statuses)."""
    return L.games(cfg.version)


def game_ids(cfg: DataConfig) -> list[str]:
    """Game ids of the configured game set, restricted to games with usable tracking."""
    ok = games(cfg).filter(pl.col("status") == "ok")["game_id"].to_list()
    if cfg.game_set == "all":
        return sorted(ok)
    path = cfg.game_sets_dir / f"{cfg.game_set}.txt"
    wanted = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    missing = sorted(set(wanted) - set(ok))
    if missing:
        raise ValueError(f"game set {cfg.game_set}: not usable in {cfg.version}: {missing}")
    return sorted(wanted)


def splits(cfg: DataConfig) -> pl.DataFrame:
    """Fixed split over all usable games: game_id, game_date, split, fold, parity (nbacore C-009)."""
    return (
        games(cfg)
        .filter(pl.col("status") == "ok")
        .select(["game_id", "game_date", "split", "fold", "parity"])
    )


def frames(game_id: str, cfg: DataConfig) -> pl.DataFrame:
    return L.frames(game_id, cfg.version)


def frame_index(game_id: str, cfg: DataConfig) -> pl.DataFrame:
    return L.frame_index(game_id, cfg.version)


def rosters(cfg: DataConfig, game_id: str | None = None) -> pl.DataFrame:
    return L.rosters(cfg.version, game_id)


def pbp(cfg: DataConfig, game_id: str | None = None) -> pl.DataFrame:
    return L.pbp(cfg.version, game_id)


def attack_direction(cfg: DataConfig, game_id: str | None = None) -> pl.DataFrame:
    return L.attack_direction(cfg.version, game_id)


def home_team_id(game_id: str, cfg: DataConfig) -> int:
    g = games(cfg).filter(pl.col("game_id") == str(game_id).zfill(10))
    return int(g["home_team_id"][0])


def manifest(cfg: DataConfig) -> dict:
    return L.manifest(cfg.version)


# ------------------------------------------------------------------------------------------
# v1.1 / v1.2 layers
# ------------------------------------------------------------------------------------------

WINDOW_KEYS = ["window_uid", "poss_uid", "overlap_frac", "offense_match"]


def ghost_v1(cfg: DataConfig) -> pl.DataFrame:
    """This project's v1 possession windows as built by nbacore, plus stable keys
    ``window_uid`` / ``poss_uid``. ``cfg.windows_release``: "l2" (D-017, corrected shot release,
    nbacore >= v1.4) or "ghost" (the original D-011 rule, identical to our own segmentation)."""
    if cfg.windows_release == "l2":
        return L.ghost_v1(cfg.version, release="l2")
    return L.ghost_v1(cfg.version)


def events(game_id: str, types, cfg: DataConfig, offense_only: bool = True) -> pl.DataFrame:
    """L2 events of one game. With ``offense_only`` keep only events whose team is the ledger
    offence (``event_possession.offense_match``); nbacore recommends this filter because the L2
    ball-possession side is sometimes wrong for > 1 s."""
    ev = L.events(game_id, types, cfg.version)
    if not offense_only or ev.height == 0:
        return ev
    ep = L.event_possession(cfg.version, game_id).select(["event_uid", "poss_uid", "offense_match"])
    return ev.join(ep, on="event_uid", how="left").filter(pl.col("offense_match").fill_null(False))


def shot_clock(game_id: str, cfg: DataConfig) -> pl.DataFrame:
    """Per-frame filled shot clock with ``shot_clock_imputed`` (v1.2)."""
    return L.shot_clock(game_id, cfg.version)


def frames_at(game_id: str, unix_ms: list[int], cfg: DataConfig) -> pl.DataFrame:
    """L1 rows of one game at the given timestamps only (predicate pushdown; much faster than
    reading the whole 25 Hz game)."""
    from nbacore import paths

    p = paths.release_dir(cfg.version) / "frames" / f"{str(game_id).zfill(10)}.parquet"
    return pl.scan_parquet(p).filter(pl.col("unix_ms").is_in(unix_ms)).collect()
