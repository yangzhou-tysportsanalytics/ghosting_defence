"""Consensus review page for the quick check (D-036).

The two annotators' Part 2 answers (data/annotations/raw/A1_ and A2_gdq_v1_answers.json) differ
on a number of clips. This page holds exactly those clips (help and switch, decoys included, in a
seeded random order), shows both original answers and notes, and records one consensus answer
per clip, chosen by the two annotators together. Clips on which they agreed keep their common
answer; quick_check_eval.py combines the two into the consensus labels.

Output (git-ignored; contains coordinates): data/annotations/quick/gdq_v1/ADJ.html
The page saves ADJ_gdq_v1_answers.json, to be put into data/annotations/raw/.

Usage:
    uv run python scripts/make_quick_adjudication.py [--lang zh] [--seed 23]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ghost import data as D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_quick_check import ROOT, SLOT, TEMPLATE, VERSION, item_e  # noqa: E402


def disagreements(alloc: dict, a1: dict, a2: dict) -> list[str]:
    """Part 2 item ids whose two answers differ (either missing counts as a difference)."""
    return sorted(i for i, a in alloc.items() if a["task"] == "E"
                  and (a1.get(i) or {}).get("a") != (a2.get(i) or {}).get("a"))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zh", choices=["en", "zh"])
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--raw-dir", default="data/annotations/raw")
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    a1 = json.loads((raw / f"A1_{VERSION}_answers.json").read_text(encoding="utf-8"))["answers"]
    a2 = json.loads((raw / f"A2_{VERSION}_answers.json").read_text(encoding="utf-8"))["answers"]
    alloc = json.loads((ROOT / "quick" / VERSION / "allocation.json").read_text())["items"]
    K = json.loads((ROOT / "quick" / "key.json").read_text())["items"]
    bundle = json.loads((ROOT / "bundles" / f"{VERSION}.json").read_text(encoding="utf-8"))
    by = {(p["game_id"], int(p["possession_id"])): p for p in bundle["possessions"]}
    cfg = D.DataConfig.load(game_set="all")
    r = (D.rosters(cfg).select(["player_id", "position"]).drop_nulls()
         .unique("player_id", keep="first"))  # fmt: skip
    pos = {int(a): str(b) for a, b in r.iter_rows()}

    ids = disagreements(alloc, a1, a2)
    items = []
    for i in ids:
        k = K[alloc[i]["index"]]
        it = item_e(k, by[(k["game_id"], int(k["possession_id"]))], pos)
        assert it["id"] == i, (it["id"], i)
        it["prior"] = {"answers": {"A1": a1[i].get("a"), "A2": a2[i].get("a")},
                       "notes": {"A1": a1[i].get("note", ""), "A2": a2[i].get("note", "")}}  # fmt: skip
        items.append(it)
    rng = np.random.default_rng(args.seed)
    items = [items[j] for j in rng.permutation(len(items))]
    data = {"version": VERSION, "annotator": "ADJ", "mode": "adjudicate", "hz": 5,
            "lang": args.lang, "items": items,
            "estimate_min": {"guide": 3.0, "M": 1.0, "E": 0.75}}  # fmt: skip
    template = TEMPLATE.read_text(encoding="utf-8")
    out = ROOT / "quick" / VERSION / "ADJ.html"
    out.write_text(
        template.replace(SLOT, json.dumps(data, separators=(",", ":"))), encoding="utf-8"
    )
    kinds = {}
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    print(f"wrote {out}: {len(items)} clips {kinds}, {round(out.stat().st_size / 1024)} KB")


if __name__ == "__main__":
    main()
