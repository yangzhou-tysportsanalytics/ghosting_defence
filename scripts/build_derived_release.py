"""Build the public derived-data package (the data behind the reported results; D-019).

No player or ball tracks: possession index, per-step matchup labels, events, per-(possession,
defender) deviation summaries, screen / shot tables and team / player result tables. The only
point coordinates are shot locations (``shot_x`` / ``shot_y`` in ``shots.parquet``); screen
locations are dropped. The package is checked with ``check_release.py`` and zipped for a GitHub
release asset; it is never committed to git.

Output: data/derived_release/<package_id>/ and data/derived_release/<package_id>.zip

Usage:
    uv run python scripts/build_derived_release.py [--package-id ghost-defense-derived-v1.1]
        [--ghost-crossfit <config tag>] [--ghost-final <run tag>]   # v1.2 learned-ghost outputs
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
    "package_id": "nbacore-derived-v1.6-shot-p1",
    "release": "https://github.com/yangzhou-tysportsanalytics/nbacore/releases/tag/data-v1.6",
    "zip_sha256": "d44a4a92f584204f10972d56a3ed5c3c9cc278df39fbe53d5c305aacd7c0a92f",
    "manifest_sha256": "1551d916078c58df702cf9d32a438eed6c54e48a3c1accd4fd582ae0db855fda",
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
    # v1.2 additions (each included when its source exists; see MANIFEST "optional")
    "phase_context.parquet": "per (possession, defender): share of man-guarding steps in each "
    "phase (pre-screen, screen, help, closeout, recovery, other; D-020), sightline-cone share "
    "and paint-time descriptors",
    "breakdowns_rule_ghost.parquet": "breakdown events against the rule ghost (D-023 threshold; "
    "reference only, D-020): defender, start / end step, peak, cascade and initiator flags",
    "help_timing_teams.csv": "team help delay after on-ball screen contact and early-help rates",
    "help_timing_players.csv": "player help propensity as a potential third defender",
    "hier_players_sag_ft_phases.csv": "player sag effects, main specification (context with "
    "phase shares, D-021); not a ranking of defenders",
    "hier_players_dev_ft_phases.csv": "as above for |deviation|",
    "learned_ghost_deviations.parquet": "per (possession, defender): mean negative log density "
    "under the individual and the identity-free team ghost, distance to the mixture mean and to "
    "the nearest mode (cross-fitted: each fold scored by a model trained without it)",
    "learned_ghost_points.parquet": "per shot and defender: shooter-aware xFG at the real "
    "positions and its expectation with the defender drawn from his ghost (and with the team ghost)",
    "breakdowns_learned_ghost.parquet": "breakdown events against the learned individual ghost "
    "(D-023 threshold on the negative log density)",
    "ghost_model_final.pt": "learned-ghost weights (PyTorch state dict), the model scored on the "
    "test games",
    "ghost_model_config.json": "its configuration (model and training) and evaluation summary",
    "ghost_models_crossfit.zip": "the five cross-fitting models (state dicts + configuration)",
    "annotations.json": "the 200 annotated possessions: reviewed matchup corrections and events "
    "per annotator (anonymised A1, A2); notes and video links removed",
    "annotation_eval.json": "accuracy of the matchup model and events against the annotations",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def add_optional(out: Path, base: Path, cfg, args, zstd: dict) -> tuple[list[str], list[str]]:
    """v1.2 additions, each copied when its source exists. Returns (included, skipped)."""
    ana, rep = base / "analysis", Path("reports")
    r4 = rep / "phase4" / f"{cfg.version}_all"
    items: list[tuple[Path, str]] = [
        (ana / "phase_context_A.parquet", "phase_context.parquet"),
        (ana / "breakdowns.parquet", "breakdowns_rule_ghost.parquet"),
        (rep / "analysis" / "help_timing_teams.csv", "help_timing_teams.csv"),
        (rep / "analysis" / "help_timing_players.csv", "help_timing_players.csv"),
        (r4 / "hier_players_sag_ft_phases.csv", "hier_players_sag_ft_phases.csv"),
        (r4 / "hier_players_dev_ft_phases.csv", "hier_players_dev_ft_phases.csv"),
    ]
    if args.ghost_crossfit:
        n = f"cf_{args.ghost_crossfit}"
        items += [
            (ana / f"learned_ghost_dev_{n}.parquet", "learned_ghost_deviations.parquet"),
            (ana / f"learned_ghost_points_{n}.parquet", "learned_ghost_points.parquet"),
            (ana / f"breakdowns_learned_{n}_nll_ind.parquet", "breakdowns_learned_ghost.parquet"),
        ]
    included, skipped = [], []
    for src, name in items:
        if not src.exists():
            skipped.append(name)
            continue
        if src.suffix == ".parquet":
            pl.read_parquet(src).write_parquet(out / name, **zstd)
        else:
            shutil.copy2(src, out / name)
        included.append(name)
    runs = Path("runs/phase3")
    if args.ghost_final:
        run = runs / f"{cfg.version}_all_{args.ghost_final}"
        if (run / "model.pt").exists():
            shutil.copy2(run / "model.pt", out / "ghost_model_final.pt")
            ev = json.loads((run / "eval.json").read_text())
            keep = {k: ev[k] for k in ("config", "n_params", "val_team", "val_individual", "test")
                    if k in ev}  # fmt: skip
            (out / "ghost_model_config.json").write_text(json.dumps(keep, indent=1))
            included += ["ghost_model_final.pt", "ghost_model_config.json"]
        else:
            skipped.append("ghost_model_final.pt")
    if args.ghost_crossfit:
        folds = [runs / f"{cfg.version}_all_{args.ghost_crossfit}_fold{k}" for k in range(5)]
        if all((f / "model.pt").exists() for f in folds):
            with zipfile.ZipFile(out / "ghost_models_crossfit.zip", "w") as z:
                for k, f in enumerate(folds):
                    z.write(f / "model.pt", arcname=f"fold{k}/model.pt")
                    cfg_k = json.loads((f / "eval.json").read_text())["config"]
                    z.writestr(f"fold{k}/config.json", json.dumps(cfg_k, indent=1))
            included.append("ghost_models_crossfit.zip")
        else:
            skipped.append("ghost_models_crossfit.zip")
    raw = sorted(Path("data/annotations/raw").glob("gd_v2_all__*.json"))
    if raw:
        anns = []
        for j, f in enumerate(raw, 1):  # anonymise; drop free text and video links
            a = json.loads(f.read_text(encoding="utf-8"))
            poss = {}
            for key, x in a["possessions"].items():
                segs = [{k: v for k, v in sg.items() if k != "note"} for sg in x["segments"]]
                poss[key] = {"segments": segs, "status": x.get("status"),
                             "rejected_prelabels": x.get("rejected_prelabels", []),
                             "time_spent_s": x.get("time_spent_s")}  # fmt: skip
            anns.append({"annotator": f"A{j}", "possessions": poss})
        data = {"bundle": "gd_v2_all", "annotators": anns}
        (out / "annotations.json").write_text(json.dumps(data))
        included.append("annotations.json")
    else:
        skipped.append("annotations.json")
    ev_p = Path("reports/phase2") / f"{cfg.version}_all" / "annotation_eval.json"
    if ev_p.exists():
        e = json.loads(ev_p.read_text())
        e.pop("files", None)
        for a in e.get("per_annotator", []):
            a.pop("annotator", None)
        for a in e.get("agreement", []):
            a.pop("pair", None)
        (out / "annotation_eval.json").write_text(json.dumps(e, indent=1))
        included.append("annotation_eval.json")
    else:
        skipped.append("annotation_eval.json")
    return included, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-id", default="ghost-defense-derived-v1.1")
    ap.add_argument("--ghost-crossfit", default=None, help="cross-fitting config tag (cf_<tag>)")
    ap.add_argument("--ghost-final", default=None, help="run tag of the model scored on test")
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

    included, skipped = add_optional(out, base, cfg, args, zstd)

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
        "optional": {"included": included, "skipped_missing_source": skipped},
        "notes": (
            {
                "shots_duplicate_rows_dropped": shots_dropped,
                "shots_duplicate_reason": "shot attached to two windows of one ledger possession",
            }
            if shots_dropped
            else {}
        ),
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
        *[f"| `{k}` | {v} |" for k, v in DESCRIPTIONS.items() if (out / k).exists()],
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
