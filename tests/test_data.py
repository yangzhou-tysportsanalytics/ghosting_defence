"""Tests for the nbacore access layer (ghost.data). Skipped when the release is not present."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ghost import data as D

try:
    from nbacore import paths

    HAVE_RELEASE = (paths.data_root() / "releases" / "v1.2" / "MANIFEST.json").exists()
except Exception:  # noqa: BLE001
    HAVE_RELEASE = False

pytestmark = pytest.mark.skipif(not HAVE_RELEASE, reason="nbacore v1.2 release not present")


def test_config_pins_version():
    cfg = D.DataConfig.load()
    # the pinned data version must match the nbacore code tag in pyproject.toml
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert f'tag = "data-{cfg.version}"' in pyproject
    assert cfg.windows_release == "l2"
    with pytest.raises(ValueError):
        D.DataConfig.load(version="latest")


def test_game_sets_and_split():
    cfg = D.DataConfig.load(game_set="small")
    ids = D.game_ids(cfg)
    assert len(ids) == 25
    s = D.splits(cfg)
    assert s.height == 631
    assert set(s["split"].unique().to_list()) == {"train", "val", "test"}
    # fixed split (nbacore C-009): the five changes against the old 25-game split
    small = s.filter(pl.col("game_id").is_in(ids))
    got = dict(zip(small["game_id"].to_list(), small["split"].to_list(), strict=True))
    assert got["0021500535"] == "test"
    assert got["0021500230"] == "train" and got["0021500477"] == "train"
    assert got["0021500333"] == "val" and got["0021500504"] == "val"
    tiny = D.game_ids(D.DataConfig.load(game_set="tiny"))
    assert set(tiny) <= set(ids)


def test_l1_matches_legacy_interim():
    """nbacore L1 must equal this project's pre-nbacore interim tables (same code, D-016)."""
    legacy = Path("data/interim/frames/0021500115.parquet")
    if not legacy.exists():
        pytest.skip("legacy interim not present")
    cfg = D.DataConfig.load()
    new = D.frames("0021500115", cfg)
    old = pl.read_parquet(legacy)
    assert new.equals(old.select(new.columns))


def test_ghost_v1_matches_our_possessions():
    """nbacore v1.2 ghost_v1 reproduces this project's windows column for column (small)."""
    ours = Path("data/processed/nbacore-v1.0/small/possessions.parquet")  # D-011 windows
    if not ours.exists():
        pytest.skip("small possessions not built")
    mine = pl.read_parquet(ours)
    gv = D.ghost_v1(D.DataConfig.load(windows_release="ghost")).filter(
        pl.col("game_id").is_in(mine["game_id"].unique())
    )
    derived = set(D.WINDOW_KEYS)  # nbacore keys/flags, regenerated per release
    cols = [c for c in mine.columns if c in gv.columns and c not in derived]
    key = ["game_id", "possession_id"]
    assert mine.select(cols).sort(key).equals(gv.select(cols).sort(key))
    assert gv["window_uid"].n_unique() == gv.height
