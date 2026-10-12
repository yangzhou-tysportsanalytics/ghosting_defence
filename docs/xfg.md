# xFG: expected field-goal percentage (`ghost.xfg`)

Make probability of a field-goal attempt given the situation at the release, on the public
2015-16 NBA SportVU release (nbacore data v1.6). Two frozen versions:

- **M2** (league level): zone, distance, distance², angle, log nearest-defender and
  front-defender distance (with three-point and rim interactions), dribble bins,
  catch-and-shoot, transition.
- **M3** = M2 + an empirical-Bayes shrunken shooter-by-zone offset (unseen shooters get 0).

L2-penalised logistic regression fitted by Newton's method (no scikit-learn). Frozen models use
only the training games of the fixed nbacore split and releases located from the ball with the
shooter in hand (`release_method` rim / apex).

## Versions and citation

- Code: tag `xfg-v1.0` of this repository (`src/ghost/xfg/`; MIT licence). The model and
  feature code is the code that produced the frozen product `ghost_xfg v1.0`; the public tag
  adds packaging only (light core dependencies, `read_release_shots`).
- Data: release asset `ghost-xfg-v1.0.zip` on the GitHub release `xfg-v1.0` (shot table,
  frozen models, calibration; see its `README_DATA.md` and `MANIFEST.json`).

## Use as a dependency

The package's core dependencies are numpy, polars, scipy, omegaconf and nbacore; the modelling
stack of the rest of the project (torch, jax, numpyro, ...) is a development group and is not
installed with the package.

With uv (nbacore's GitHub tag comes from this package's `tool.uv.sources`):

```bash
uv add "ghost-defense @ git+https://github.com/yangzhou-tysportsanalytics/ghosting_defence@xfg-v1.0"
```

With pip, which does not read `tool.uv.sources`, install nbacore first (it is not on PyPI):

```bash
pip install "nbacore @ git+https://github.com/yangzhou-tysportsanalytics/nbacore@data-v1.6"
pip install "ghost-defense @ git+https://github.com/yangzhou-tysportsanalytics/ghosting_defence@xfg-v1.0"
```

Checked in a clean environment: the install brings no torch / jax / numpyro, and refitting from
the release asset reproduces the frozen predictions (M2 exactly, M3 to 5e-16).

## Refit without tracking data

```python
from ghost.xfg import api

shots = api.read_release_shots("ghost-xfg-v1.0/shots_xfg.parquet")  # shot_x / shot_y -> x / y
train = shots.filter(shots["split"] == "train")["game_id"].unique().to_list()
m3 = api.fit(shots, train, "M3")          # ~1 s; trusted releases of the given games
p = api.predict(shots, m3)                 # equals shots["xfg_M3"] (max difference ~1e-15)
frozen = api.load("ghost-xfg-v1.0", "M3")  # or use the frozen model directly
```

`fit(shots, game_ids, version)` refits on any subset of games (e.g. cross-fitting by fold).

## Features at arbitrary times (needs the nbacore data release)

`api.features(game_id, rows)` computes the same features at any tracking time for given
would-be shooters (`rows` with `unix_ms`, `shooter_id`), from nbacore's frames and touch /
dribble events. Set `NBA_DATA_ROOT` to the nbacore data release (built from the public raw logs
with the nbacore code). About 2.7 s per game for all of a game's frames of interest.

## Rebuild the product from the tracking data

With `NBA_DATA_ROOT` set and the possession windows built (`phase1_possessions.py`):

| step | command | time | peak memory |
|---|---|---|---|
| possession windows (once) | `uv run python scripts/phase1_possessions.py --game-set all --workers 3` | ~35 min | fits in 16 GB with 3 workers |
| shot table (features of every attempt) | `uv run python scripts/phase4_xfg_data.py` | 3.5 min | 0.9 GB |
| frozen models, predictions, calibration | `uv run python scripts/build_xfg_product.py --out-dir <dir>` | 26 s | 1.5 GB |
| release asset | `uv run python scripts/build_xfg_release_asset.py --product-dir <dir>/v1.0` | < 1 min | < 1 GB |

Measured on a laptop CPU (Intel Core Ultra 7, 16 GB). A rebuild reproduces the published
`ghost_xfg v1.0`: the shot table is identical, M2 predictions are identical, and M3 predictions
and coefficients differ by at most 3e-14 (floating-point summation order).

## Version consistency (frozen product vs public tag)

The frozen product `ghost_xfg v1.0` was built from an internal commit of this project; the public
tag `xfg-v1.0` (commit `15ded21`) carries the same model and feature code. Two checks:

1. **Code**, `reports/analysis/xfg_code_consistency.json`, made with
   `scripts/compare_code_trees.py` (Python syntax trees with docstrings removed; comments never
   enter the tree). `ghost.xfg.features`, `ghost.xfg.model`, `ghost.court` and `ghost.io.schema`
   are identical; `ghost.xfg.api` is identical apart from the added `read_release_shots`; the two
   other differences (`ghost.data`, `ghost.io.raw`) are outside the code paths of `ghost.xfg`.
   The file lists the sha256 of each normalised file on both sides.
2. **Numbers**, which anyone can repeat from public material only: the predictions in the
   release asset were produced by the frozen build; refitting with the public tag reproduces
   them (snippet above; max difference 0 for M2, 4e-16 for M3).

Published tags and release assets are not moved or replaced; changes get a new tag.

## Terms

Underlying tracking and play-by-play data (c) NBA / NBA.com; the derived tables are shared for
research reproducibility and contain no player or ball tracks (the only point coordinates are
shot locations). Code: MIT.
