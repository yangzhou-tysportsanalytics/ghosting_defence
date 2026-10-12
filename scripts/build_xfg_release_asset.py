"""Release asset of the frozen xFG product (ghost_xfg) for the public repository.

From the product directory (scripts/build_xfg_product.py; default
<NBA_DATA_ROOT>/products/ghost_xfg/v1.0) to data/derived_release/ghost-xfg-<version>/ and its zip:
  shots_xfg.parquet         one row per shot: ids, xFG features, flags, frozen and cross-fitted
                            predictions; the shot location as shot_x / shot_y (D-019: the only
                            point coordinates; ghost.xfg.api.read_release_shots restores x / y)
  M2.json, M3.json,         frozen models (train games, trusted releases), loadable with
  M3_shooter_prior.parquet  ghost.xfg.api.load
  calibration.json          out-of-sample metrics
  README_DATA.md, MANIFEST.json
Checked with check_release.py (no tracks). Also verifies that refitting M2 / M3 from the released
table alone reproduces the frozen predictions.

Usage:
    uv run python scripts/build_xfg_release_asset.py [--version v1.0] [--product-dir <dir>]
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

import numpy as np
import polars as pl
from nbacore import paths

from ghost.xfg import api

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_release import check_dir  # noqa: E402

COPY = ["M2.json", "M3.json", "M3_shooter_prior.parquet", "calibration.json"]

README = """# ghost-xfg-{version}

Frozen expected field-goal (xFG) models of ghost-defense on the public 2015-16 NBA SportVU
release (nbacore data {data}): M2 = league level (location, defence geometry, dribbles,
catch-and-shoot), M3 = M2 + shrunken shooter-by-zone offset. Code: the repository's tag
`xfg-{version}` (`ghost.xfg`). Row counts and sha256: `MANIFEST.json`.

| file | content |
|---|---|
| `shots_xfg.parquet` | one row per field-goal attempt: ids, features, `release_trusted`, `in_training`, split / fold / parity, frozen (`xfg_M2`, `xfg_M3`) and cross-fitted (`xfg_M2_cf`, `xfg_M3_cf`) predictions; shot location `shot_x` / `shot_y` (ft, half court; the only point coordinates) |
| `M2.json`, `M3.json`, `M3_shooter_prior.parquet` | frozen models, `ghost.xfg.api.load(dir, "M3")` |
| `calibration.json` | out-of-sample log loss, Brier score, AUC and decile calibration |

Refit without tracking data:

```python
from ghost.xfg import api
shots = api.read_release_shots("shots_xfg.parquet")      # restores x / y for the models
games = shots.filter(shots["split"] == "train")["game_id"].unique().to_list()
m3 = api.fit(shots, games, "M3")
p = api.predict(shots, m3)                               # = shots["xfg_M3"] (to ~1e-12)
```

Terms: underlying tracking and play-by-play data (c) NBA / NBA.com; derived tables shared for
research reproducibility; code MIT.
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0")
    ap.add_argument("--product-dir", default=None)
    args = ap.parse_args()
    src = (
        Path(args.product_dir)
        if args.product_dir
        else paths.products_dir() / "ghost_xfg" / args.version
    )
    pid = f"ghost-xfg-{args.version}"
    out = Path("data/derived_release") / pid
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    prod_manifest = json.loads((src / "MANIFEST.json").read_text())
    shots = pl.read_parquet(src / "shots_xfg.parquet").rename({"x": "shot_x", "y": "shot_y"})
    shots.write_parquet(out / "shots_xfg.parquet", compression="zstd")
    for f in COPY:
        shutil.copy2(src / f, out / f)

    # refitting from the released table alone reproduces the frozen predictions
    rel = api.read_release_shots(out / "shots_xfg.parquet")
    train = rel.filter(pl.col("split") == "train")["game_id"].unique().to_list()
    refit = {}
    for v in api.VERSIONS:
        p = api.predict(rel, api.fit(rel, train, v))
        ref = rel[f"xfg_{v}"].to_numpy()
        ok = ~np.isnan(ref) & ~np.isnan(p)
        refit[v] = float(np.abs(p[ok] - ref[ok]).max())
    if max(refit.values()) > 1e-9:
        raise SystemExit(f"refit from the released table does not reproduce the product: {refit}")

    (out / "README_DATA.md").write_text(
        README.format(version=args.version, data=prod_manifest["nbacore_data_version"]),
        encoding="utf-8",
    )
    bad = check_dir(out)
    if bad:
        raise SystemExit("release check failed:\n  " + "\n  ".join(bad))
    manifest = {
        "package_id": pid,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "product": {
            k: prod_manifest[k]
            for k in ("product", "version", "code_commit", "nbacore_data_version", "created")
        },  # fmt: skip
        "refit_from_release_max_abs_diff": refit,
        "coordinate_policy": "no player or ball tracks; only shots_xfg.parquet shot_x / shot_y",
        "files": {},
    }
    for p in sorted(out.iterdir()):
        e = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        if p.suffix == ".parquet":
            e["rows"] = pl.scan_parquet(p).select(pl.len()).collect().item()
        manifest["files"][p.name] = e
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    z = out.parent / f"{pid}.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as f:
        for p in sorted(out.iterdir()):
            f.write(p, arcname=f"{pid}/{p.name}")
    print(json.dumps({"zip": str(z), "bytes": z.stat().st_size, "sha256": sha256(z),
                      "refit_max_abs_diff": refit}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
