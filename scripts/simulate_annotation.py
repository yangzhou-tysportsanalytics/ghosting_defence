"""Dry run of the annotation evaluation: a simulated annotator for the core and blind bundles.

The simulated annotator confirms every model pre-label and, on a random 30 % of possessions,
re-assigns one defender to his nearest attacker for 2 s (a stand-in for a correction). For the
blind bundle it draws the same lanes as matchup segments (blind lanes start unlabelled) and
copies the events. The files only exercise phase2_annotation_eval.py end to end; they are not
annotations and must never go to data/annotations/raw (refused).

Usage:
    uv run python scripts/simulate_annotation.py --out-dir <scratch dir>
    uv run python scripts/phase2_annotation_eval.py --raw-dir <scratch dir> --blind gd_v2_blind10 \
        --out <scratch dir>/annotation_eval_dryrun.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_annotation_eval import EVENT_TYPES, nearest_defender, truth_lanes  # noqa: E402

BUNDLES = Path("data/annotations/bundles")
NAME = "SIM"


def runs_to_segments(lane: np.ndarray, off_ids: list, def_id: int) -> list[dict]:
    out, n = [], len(lane)
    starts = [0] + [k for k in range(1, n) if lane[k] != lane[k - 1]]
    for a, b in zip(starts, starts[1:] + [n], strict=True):
        k = int(lane[a])
        out.append({"type": "matchup", "source": "human", "start_step": a, "end_step": b - 1,
                    "participants": [{"role": "defender", "player_id": def_id},
                                     {"role": "attacker",
                                      "player_id": off_ids[k] if k >= 0 else None}]})  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--core", default="gd_v2_core100")
    ap.add_argument("--blind", default="gd_v2_blind10")
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    if out == Path("data/annotations/raw").resolve():
        raise SystemExit("refusing to write simulated annotations to data/annotations/raw")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    core = json.loads((BUNDLES / f"{args.core}.json").read_text())
    ann = {"annotator": NAME, "simulated": True, "possessions": {}}
    for p in core["possessions"]:
        segs = [dict(copy.deepcopy(s), source="model_confirmed", pre_id=i)
                for i, s in enumerate(p["prelabels"]) if s["type"] in EVENT_TYPES]  # fmt: skip
        n = p["n_steps"]
        if rng.random() < 0.3 and n > 12:
            d = int(rng.integers(5))
            a = int(rng.integers(0, n - 10))
            k = int(nearest_defender(p)[d, a])
            segs.append({"type": "matchup", "source": "human", "start_step": a,
                         "end_step": a + 9,
                         "participants": [{"role": "defender", "player_id": p["defense"][d]["id"]},
                                          {"role": "attacker",
                                           "player_id": p["offense"][k]["id"]}]})  # fmt: skip
        ann["possessions"][p["key"]] = {"segments": segs, "rejected_prelabels": [],
                                        "status": "done", "time_spent_s": 300}  # fmt: skip
    (out / f"{args.core}__{NAME}.json").write_text(json.dumps(ann))
    blind = json.loads((BUNDLES / f"{args.blind}.json").read_text())
    by_key = {p["key"]: p for p in core["possessions"]}
    bann = {"annotator": NAME, "simulated": True, "possessions": {}}
    for bp in blind["possessions"]:
        p = by_key[bp["key"]]
        lanes = truth_lanes(p, ann["possessions"][p["key"]]["segments"])
        off_ids = [q["id"] for q in p["offense"]]
        segs = [s for d, q in enumerate(p["defense"])
                for s in runs_to_segments(lanes[d], off_ids, q["id"])]  # fmt: skip
        segs += [dict(copy.deepcopy(s), source="human") for s in p["prelabels"]
                 if s["type"] in EVENT_TYPES]  # fmt: skip
        bann["possessions"][bp["key"]] = {"segments": segs, "rejected_prelabels": [],
                                          "status": "done", "time_spent_s": 600}  # fmt: skip
    (out / f"{args.blind}__{NAME}.json").write_text(json.dumps(bann))
    print(f"wrote {out}/{args.core}__{NAME}.json and {args.blind}__{NAME}.json")


if __name__ == "__main__":
    main()
