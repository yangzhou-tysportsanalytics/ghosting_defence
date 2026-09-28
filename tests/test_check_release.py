"""check_release: flags raw data and frame-level coordinates, accepts aggregates."""

from __future__ import annotations

import importlib.util
import json

import polars as pl

spec = importlib.util.spec_from_file_location("check_release", "scripts/check_release.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def test_flags_coordinates_and_raw(tmp_path):
    # frame-level table with x/y -> violation
    frames = pl.DataFrame(
        {
            "game_id": ["g"] * 50,
            "possession_id": [1] * 50,
            "step": range(50),
            "x": [1.0] * 50,
            "y": [2.0] * 50,
        }
    )
    frames.write_parquet(tmp_path / "frames.parquet")
    # renamed trajectory: ball_x / ball_y with a time column
    pl.DataFrame({"t_ms": [1, 2], "ball_x": [1.0, 2.0], "ball_y": [3.0, 4.0]}).write_csv(
        tmp_path / "b.csv"
    )
    (tmp_path / "g.json").write_text(json.dumps({"gameid": "1", "events": []}))
    (tmp_path / "a.7z").write_bytes(b"7z")
    bad = cr.check_dir(tmp_path)
    assert any("frames.parquet" in b for b in bad)
    assert any("b.csv" in b for b in bad)
    assert any("g.json" in b for b in bad)
    assert any("a.7z" in b for b in bad)


def test_point_coordinates_policy(tmp_path):
    # D-019 (R3 = shot): only the shot location, one row per shot, may carry coordinates
    shots = pl.DataFrame(
        {
            "game_id": ["g", "g"],
            "t_release_ms": [1, 2],
            "shot_x": [5.0, 20.0],
            "shot_y": [25.0, 10.0],
            "def1_ft": [3.2, 6.0],
        }
    )
    shots.write_parquet(tmp_path / "shots.parquet")
    assert cr.check_dir(tmp_path) == []
    # screen location without a time column is still a point coordinate -> violation
    pl.DataFrame({"game_id": ["g"], "screen_x": [10.0], "screen_y": [20.0]}).write_parquet(
        tmp_path / "screens.parquet"
    )
    # nbacore-style event columns x_catch / y_catch -> violation
    pl.DataFrame({"game_id": ["g"], "x_catch": [1.0], "y_catch": [2.0]}).write_csv(
        tmp_path / "passes.csv"
    )
    # shot location repeated per shot (trajectory-like) -> violation
    pl.concat([shots, shots]).write_parquet(tmp_path / "dup_shots.parquet")
    bad = cr.check_dir(tmp_path)
    assert not any(b.startswith("shots.parquet") for b in bad)
    assert any("screens.parquet" in b for b in bad)
    assert any("passes.csv" in b for b in bad)
    assert any("dup_shots.parquet" in b for b in bad)


def test_accepts_aggregates(tmp_path):
    pl.DataFrame(
        {
            "game_id": ["g"],
            "possession_id": [1],
            "dev_ft": [2.3],
            "sag_ft": [0.4],
            "window_uid": ["g:1"],
        }
    ).write_parquet(tmp_path / "dev.parquet")
    pl.DataFrame({"player_id": [1], "effect_mean": [0.1]}).write_csv(tmp_path / "p.csv")
    assert cr.check_dir(tmp_path) == []
