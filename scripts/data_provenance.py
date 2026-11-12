"""Data provenance (review request A10): games per team, missing games, raw-file checksums.

* Games: the tracking archive against the 2015-16 regular-season schedule (nbacore box_games,
  all 1,230 games) between the first and last archive dates; games per team among the games
  used (status ok).
* Raw files: SHA-256 and size of every SportVU archive and of the play-by-play CSV; for the
  archives also the git blob SHA-1, compared with the upstream GitHub listing saved at download
  time (raw/github_listing.json), so each local file is verified byte for byte against the
  public repository.

Outputs (no data, only names and hashes): reports/phase0/data_provenance.json,
reports/phase0/raw_checksums.csv

Usage:
    uv run python scripts/data_provenance.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
from nbacore import paths

from ghost import data as D

FIRST, LAST = "2015-10-27", "2016-01-23"


def hashes(p: Path) -> tuple[str, str, int]:
    """SHA-256, git blob SHA-1 and size of a file."""
    b = p.read_bytes()
    blob = hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()  # noqa: S324 (git object id)
    return hashlib.sha256(b).hexdigest(), blob, len(b)


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    G = D.L.games(cfg.version)
    box = D.L.box_games(cfg.version)
    season = box.filter(
        (pl.col("game_date").cast(pl.Utf8) >= FIRST) & (pl.col("game_date").cast(pl.Utf8) <= LAST)
    )
    ok = G.filter(pl.col("status") == "ok")
    used_ids = set(ok["game_id"].to_list())
    missing = season.filter(~pl.col("game_id").is_in(list(used_ids)))
    status = dict(G.filter(pl.col("game_id").is_in(missing["game_id"].to_list()))
                  .group_by("status").len().sort("status").rows())  # fmt: skip
    status["not_in_archive_table"] = missing.height - sum(status.values())
    teams = (
        pl.concat(
            [
                ok.select(pl.col("home_abbr").alias("team")),
                ok.select(pl.col("visitor_abbr").alias("team")),
            ]
        )  # fmt: skip
        .group_by("team")
        .len()
        .sort("team")
    )
    raw = paths.data_root() / "raw"
    listing = {r["name"]: r for r in json.loads((raw / "github_listing.json").read_text())}
    used_arch = set(ok["archive_name"].drop_nulls().to_list())
    rows = []
    for name in sorted(listing):
        f = raw / "sportvu" / name
        sha256, blob, size = hashes(f) if f.exists() else (None, None, None)
        rows.append({"file": f"sportvu/{name}", "size_bytes": size, "sha256": sha256,
                     "git_blob_sha1": blob, "upstream_git_sha1": listing[name]["sha"],
                     "matches_upstream": blob == listing[name]["sha"],
                     "used": name in used_arch})  # fmt: skip
    pbp = raw / "pbp" / "2015-16_pbp.csv"
    s, b, n = hashes(pbp)
    rows.append({"file": "pbp/2015-16_pbp.csv", "size_bytes": n, "sha256": s, "git_blob_sha1": b,
                 "upstream_git_sha1": None, "matches_upstream": None, "used": True})  # fmt: skip
    R = pl.DataFrame(rows)
    out = Path("reports/phase0")
    out.mkdir(parents=True, exist_ok=True)
    R.write_csv(out / "raw_checksums.csv")
    arch = R.filter(pl.col("file").str.starts_with("sportvu/"))
    rep = {
        "nbacore_version": cfg.version,
        "period": [FIRST, LAST],
        "regular_season_games_in_period": season.height,
        "games_used": ok.height,
        "games_missing_from_analysis": missing.height,
        "missing_by_reason": status,
        "games_per_team_used": dict(teams.rows()),
        "games_per_team_min_max": [int(teams["len"].min()), int(teams["len"].max())],
        "raw_archives": {
            "n_in_upstream_listing": len(listing),
            "n_present_locally": int(arch["sha256"].is_not_null().sum()),
            "n_matching_upstream_git_sha": int(arch["matches_upstream"].sum()),
            "n_used": int(arch["used"].sum()),
            "upstream": "github.com/linouk23/NBA-Player-Movements, data/2016.NBA.Raw.SportVU.Game.Logs",
        },
        "pbp_csv_sha256": s,
        "checksums_file": "reports/phase0/raw_checksums.csv",
    }
    (out / "data_provenance.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "games_per_team_used"}, indent=1))


if __name__ == "__main__":
    main()
