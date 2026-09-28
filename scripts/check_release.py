"""Check that a release directory contains no raw or frame-level coordinates (data policy, D-019).

Rules (any violation fails):
  1. no raw archives or raw game JSON (*.7z, *.json with SportVU keys "moments"/"events")
  2. tabular files (parquet / csv) must not contain coordinate columns: x, y, z, vx, vy (any
     case), *_x / *_y, x_* / y_*. The only exception (D-019, release policy R3 = "shot") is the
     shot location ``shot_x`` / ``shot_y`` in a shot-level table (one row per game_id +
     t_release_ms).
  3. no table may have more rows per possession than MAX_ROWS_PER_POSSESSION when it carries any
     coordinate-like column (guards against renamed trajectories)
Possession- or shot-level aggregates (e.g. a shot distance, a mean deviation) are allowed.

Usage:
    uv run python scripts/check_release.py [data/derived_release]
Exit code 0 = clean, 1 = violations (listed).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import polars as pl

COORD = re.compile(r"^(x|y|z|vx|vy)$", re.I)
PAIR = re.compile(r"^(.+)_(x|y)$", re.I)
PREFIX = re.compile(r"^(x|y)_.+$", re.I)  # nbacore event style: x_release, y_catch
ALLOWED_POINTS = {"shot_x", "shot_y"}  # D-019: shot location only
SHOT_KEY = ["game_id", "t_release_ms"]
TIME = {"unix_ms", "t_unix", "step", "t_ms", "moment_idx"}
MAX_ROWS_PER_POSSESSION = 30


def check_table(df: pl.DataFrame, name: str) -> list[str]:
    bad = []
    cols = df.columns
    lc = {c.lower() for c in cols}
    coord_cols = [c for c in cols if COORD.match(c)]
    if coord_cols:
        bad.append(f"{name}: coordinate columns {coord_cols}")
    points = [
        c for c in cols if (PAIR.match(c) or PREFIX.match(c)) and c.lower() not in ALLOWED_POINTS
    ]
    if points:
        bad.append(f"{name}: point coordinates other than the shot location (D-019) {points}")
    both = points
    if lc & ALLOWED_POINTS:
        if not set(SHOT_KEY) <= lc:
            bad.append(f"{name}: shot location without a shot key {SHOT_KEY}")
        elif df.select(SHOT_KEY).is_duplicated().any():
            bad.append(f"{name}: shot location with more than one row per shot")
    if (coord_cols or both) and "possession_id" in lc and "game_id" in lc and df.height:
        per = df.group_by(["game_id", "possession_id"]).len()["len"].max()
        if per and per > MAX_ROWS_PER_POSSESSION:
            bad.append(f"{name}: {per} rows per possession with coordinates (trajectory-like)")
    return bad


def check_dir(root: Path) -> list[str]:
    bad = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if p.suffix.lower() == ".7z":
            bad.append(f"{rel}: raw archive")
        elif p.suffix.lower() == ".json":
            try:
                head = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(head, dict) and {"events", "gameid"} <= set(head):
                    bad.append(f"{rel}: raw SportVU game JSON")
            except Exception:  # noqa: BLE001
                pass
        elif p.suffix.lower() in (".parquet", ".csv"):
            df = pl.read_parquet(p) if p.suffix.lower() == ".parquet" else pl.read_csv(p)
            bad += check_table(df, str(rel))
    return bad


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/derived_release")
    if not root.exists():
        print(f"{root} does not exist")
        sys.exit(1)
    bad = check_dir(root)
    if bad:
        print("RELEASE CHECK FAILED:")
        for b in bad:
            print("  -", b)
        sys.exit(1)
    print(f"{root}: clean (no raw or frame-level coordinates)")


if __name__ == "__main__":
    main()
