"""Blind annotation subset (decision D-027; writing request A1b).

Two possessions per stratum (10) of the core bundle gd_v2_core100, drawn at random (seeded) from
those not flagged for double annotation, copied without any model output: no event pre-labels
and every matchup lane "not labelled yet" (-2). The annotator labels them first, blind, and
labels the same possessions again later with the pre-labels inside gd_v2_core100 (a few days
apart); scripts/phase2_annotation_eval.py --blind compares the two (anchoring).

Output: data/annotations/bundles/gd_v2_blind10.json (private: contains coordinates)

Usage:
    uv run python scripts/make_blind_bundle.py [--seed 9]
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PER_STRATUM = 2
UNLABELLED = -2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()
    root = Path("data/annotations/bundles")
    core = json.loads((root / "gd_v2_core100.json").read_text())
    rng = np.random.default_rng(args.seed)
    chosen = []
    for stratum in sorted({p["stratum"] for p in core["possessions"]}):
        pool = sorted(
            (p for p in core["possessions"]
             if p["stratum"] == stratum and not p.get("double_annotate")),
            key=lambda p: p["key"],
        )  # fmt: skip
        chosen += [pool[i] for i in sorted(rng.choice(len(pool), PER_STRATUM, replace=False))]
    out = {k: v for k, v in core.items() if k != "possessions"}
    out.update(bundle="gd_v2_blind10", blind=True, paired_with="gd_v2_core100",
               created=datetime.now(UTC).isoformat(timespec="seconds"))  # fmt: skip
    out["possessions"] = []
    for p in sorted(chosen, key=lambda p: p["key"]):
        q = copy.deepcopy(p)
        q["prelabels"] = []
        q["matchup_prelabel"] = [[UNLABELLED] * q["n_steps"] for _ in range(5)]
        q["double_annotate"] = False
        out["possessions"].append(q)
    (root / "gd_v2_blind10.json").write_text(json.dumps(out, separators=(",", ":")))
    print(json.dumps({"bundle": out["bundle"], "n": len(out["possessions"]),
                      "keys": [(p["key"], p["stratum"]) for p in out["possessions"]]}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
