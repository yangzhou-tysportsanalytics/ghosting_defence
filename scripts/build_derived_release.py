"""Build the public derived-data package (the data behind the reported results; D-019).

No player or ball tracks: possession index, per-step matchup labels, events, per-(possession,
defender) deviation summaries, screen / shot tables and team / player result tables. The only
point coordinates are shot locations (``shot_x`` / ``shot_y`` in ``shots.parquet``); screen
locations are dropped. The package is checked with ``check_release.py`` and zipped for a GitHub
release asset; it is never committed to git.

Output: data/derived_release/<package_id>/ and data/derived_release/<package_id>.zip

Usage:
    uv run python scripts/build_derived_release.py [--package-id ghost-defense-derived-v1.0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_release import check_dir  # noqa: E402
from ghost import data as D  # noqa: E402

NBACORE_PACKAGE = {
    "package_id": "nbacore-derived-v1.5-shot-p1",
    "release": "https://github.com/yangzhou-tysportsanalytics/nbacore/releases/tag/data-v1.5",
    "zip_sha256": "e742442c68b0701d31cf65152a4c1b4b7abc0a93b1ab88da695a58935b9d8f31",
    "manifest_sha256": "6a7080088b3d00dde818551bf2d5c5b54251440d27283fabd9d923b0372ac4e0",
}
MATCHUP_MODEL = "hmm_strat_help"

DESCRIPTIONS = {
    "possessions.parquet": "possession windows (one row per window, half-court and transition); "
    "keys window_uid (stable across releases) and poss_uid (nbacore ledger); times in unix ms",
    "matchups.parquet": "per 5 Hz step and defender: guarded attacker from the matchup HMM "
    "(strong/weak side + help state); state 0-4 = attacker slot, 5 = help / no man; step k is "
    "t_start + 200 k ms of the possession window; post_max = posterior of the chosen state; "
    "baseline_state = min-residual baseline",
    "events_switches.parquet": "changes of guarded attacker held >= 1 s (switch B) with class "
    "screen_switch (= reported switch, definition A) / one_sided / rotation",
    "events_help.parquet": "help events; definition A (help state) and B (geometric, reported)",
    "events_closeouts.parquet": "closeouts after a catch",
    "onball_screens.parquet": "one row per on-ball screen group: switched, first third-defender "
    "help delay and helped within 3 s (screen location dropped)",
    "rule_ghost_deviations.parquet": "per (possession, defender): mean |deviation| and sag toward "
    "the rim relative to the league-average rule ghost, plus context covariates",
    "shots.parquet": "field-goal attempts with xFG features; shot_x / shot_y = shot location "
    "(half-court, ft; the only point coordinates in the package)",
    "ghost_points.parquet": "per shot: xFG at the real and at the rule-ghost defender positions",
    "team_screen_defense.csv": "team switch and help rates on on-ball screens (95 % game bootstrap)",
    "team_points_allowed.csv": "team points allowed per possession in the tracked games (95 % game "
    "bootstrap)",
    "hier_players_sag_ft.csv": "player sag effects from the hierarchical model (95 % intervals); "
    "players with >= 300 defensive possessions; not a ranking of defenders",
    "hier_players_dev_ft.csv": "as above for |deviation|",
    "matchup_params.json": "fitted matchup-HMM parameters (all three model variants)",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-id", default="ghost-defense-derived-v1.0")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    base = cfg.processed_dir / "all"
    out = Path("data/derived_release") / args.package_id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    zstd = {"compression": "zstd", "compression_level": 9}

    pl.read_parquet(base / "possessions.parquet").write_parquet(out / "possessions.parquet", **zstd)
    files = sorted((base / "matchups" / MATCHUP_MODEL).glob("*.parquet"))
    pl.scan_parquet(files).sort(["game_id", "possession_id", "def_slot", "step"]).sink_parquet(
        out / "matchups.parquet", **zstd
    )
    for name in ("switches", "help", "closeouts"):
        pl.read_parquet(base / "events" / f"{name}.parquet").write_parquet(
            out / f"events_{name}.parquet", **zstd
        )
    ana = base / "analysis"
    pl.read_parquet(ana / "onball_screens.parquet").drop(["x", "y"]).write_parquet(
        out / "onball_screens.parquet", **zstd
    )
    pl.read_parquet(ana / "rule_ghost_dev.parquet").write_parquet(
        out / "rule_ghost_deviations.parquet", **zstd
    )
    shots = pl.read_parquet(ana / "shots.parquet").rename({"x": "shot_x", "y": "shot_y"})
    # a few shots were attached to two windows of the same ledger possession by
    # phase4_xfg_data.py; publish one row per shot (the lower possession_id)
    n_before = shots.height
    shots = shots.sort(["game_id", "t_release_ms", "possession_id"]).unique(
        subset=["game_id", "t_release_ms"], keep="first", maintain_order=True
    )
    shots_dropped = n_before - shots.height
    shots.write_parquet(out / "shots.parquet", **zstd)
    pl.read_parquet(ana / "ghost_points.parquet").write_parquet(
        out / "ghost_points.parquet", **zstd
    )
    rep = Path("reports")
    for src in (
        rep / "analysis" / "team_screen_defense.csv",
        rep / "analysis" / "team_points_allowed.csv",
        rep / "phase4" / f"{cfg.version}_all" / "hier_players_sag_ft.csv",
        rep / "phase4" / f"{cfg.version}_all" / "hier_players_dev_ft.csv",
    ):
        shutil.copy2(src, out / src.name)
    params = {
        p.name.removesuffix("_params.json"): json.loads(p.read_text())
        for p in sorted((base / "matchups").glob("*_params.json"))
    }
    (out / "matchup_params.json").write_text(json.dumps(params, indent=2))

    bad = check_dir(out)
    if bad:
        sys.exit("release check failed:\n  " + "\n  ".join(bad))

    manifest = {
        "package_id": args.package_id,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": {
            "nbacore_data_version": cfg.version,
            "windows_release": cfg.windows_release,
            "matchup_model": MATCHUP_MODEL,
            "nbacore_public_package": NBACORE_PACKAGE,
        },
        "coordinate_policy": "no player or ball tracks; only shots.parquet shot_x / shot_y",
        "notes": {
            "shots_duplicate_rows_dropped": shots_dropped,
            "shots_duplicate_reason": "shot attached to two windows of one ledger possession; "
            "the xFG fit used both rows (effect negligible)",
        },
        "files": {},
    }
    for p in sorted(out.iterdir()):
        entry = {
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
            "description": DESCRIPTIONS[p.name],
        }
        if p.suffix == ".parquet":
            lf = pl.scan_parquet(p)
            entry["rows"] = lf.select(pl.len()).collect().item()
            entry["columns"] = lf.collect_schema().names()
        elif p.suffix == ".csv":
            df = pl.read_csv(p)
            entry["rows"], entry["columns"] = df.height, df.columns
        manifest["files"][p.name] = entry
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    lines = [
        f"# {args.package_id}",
        "",
        "Derived data behind the ghost-defense results on the public 2015-16 NBA SportVU release",
        f"(631 tracked games; nbacore data {cfg.version}). No player or ball tracks are included; the",
        "only point coordinates are shot locations. Row counts, columns and sha256: `MANIFEST.json`.",
        "",
        "| file | content |",
        "|---|---|",
        *[f"| `{k}` | {v} |" for k, v in DESCRIPTIONS.items()],
        "",
        "Keys: `game_id` (NBA game id), `possession_id` (window index within a game), `window_uid`",
        "(`<game_id>:<terminal pbp event>`), `poss_uid` (nbacore ledger possession), player and team",
        "ids are NBA ids. Split columns follow the fixed nbacore split (train / val / test by date).",
        "",
        "Inputs: nbacore public data package "
        f"`{NBACORE_PACKAGE['package_id']}` ({NBACORE_PACKAGE['release']}).",
        "",
        "Terms: underlying tracking and play-by-play data (c) NBA / NBA.com; these derived tables are",
        "shared for research reproducibility. Point estimates of individual players describe observed",
        "positioning relative to a league-average reference; they are not a ranking of defenders.",
    ]
    (out / "README_DATA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zpath = out.parent / f"{args.package_id}.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        for p in sorted(out.iterdir()):
            z.write(p, arcname=f"{args.package_id}/{p.name}")
    print(
        json.dumps(
            {k: (v["rows"] if "rows" in v else v["bytes"]) for k, v in manifest["files"].items()},
            indent=1,
        )
    )
    print(f"{zpath}: {zpath.stat().st_size:,} bytes, sha256 {sha256(zpath)}")
    print(f"MANIFEST.json sha256 {sha256(out / 'MANIFEST.json')}")


if __name__ == "__main__":
    main()
