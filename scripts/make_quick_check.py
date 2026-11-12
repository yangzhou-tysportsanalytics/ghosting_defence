"""Pages for the quick check of matchups and events (D-036; sample: select_quick_check.py).

One offline page per annotator (tools/quick_check/index.html with the items embedded): A1 and
A2. Both pages hold the same items; the order is shuffled
per annotator (practice items first in each part).

Part 1 (M): the possession from its start to 1.2 s after the second moment; moments as drawn
by the sampler. Names, jerseys and listed positions shown; no model output.
Part 2 (E): a clip from 3 s before to 2.4 s after the event time (help) or 2.4 s before to 2.4 s
after screen contact (switch), with the asked defender(s) ringed and a shaded window
[t - 1 s, t + 2 s] (help) or [contact - 0.6 s, contact + 2 s] (switch). Decoys are built the same
way from their reference time.

Item ids are opaque (hash of game, possession and time); allocation.json maps them back to the
key items (private).

Outputs (git-ignored; pages contain coordinates):
    data/annotations/quick/gdq_v1/A1.html, A2.html
    data/annotations/quick/gdq_v1/allocation.json

Usage:
    uv run python scripts/make_quick_check.py [--annotators A1 A2] [--seed 19]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ghost import data as D

VERSION = "gdq_v1"
SALT = "gdq-v1"
ROOT = Path("data/annotations")
TEMPLATE = Path("tools/quick_check/index.html")
SLOT = "/*__GDQ_DATA__*/null"
EST = {"guide": 5.0, "M": 1.0, "E": 0.33}  # minutes (estimates; the page records actual time)
M_TAIL = 6  # steps shown after the second moment
HELP_CLIP, HELP_WIN = (15, 12), (5, 10)  # steps before / after the event time
SWITCH_CLIP, SWITCH_WIN = (12, 12), (3, 10)


def item_id(task: str, key: str) -> str:
    return f"{task}-" + hashlib.sha1(f"{SALT}:{key}".encode()).hexdigest()[:7]


def step_of(t_ms: int, t0: int, n: int) -> int:
    return int(np.clip(round((t_ms - t0) / 200), 0, n - 1))


def people(p: dict, pos: dict[int, str]) -> dict:
    one = lambda x: {"name": x["name"].split(" ")[-1], "jersey": str(x["jersey"]),  # noqa: E731
                     "pos": pos.get(int(x["id"]), "")}  # fmt: skip
    return {"off": [one(x) for x in p["offense"]], "def": [one(x) for x in p["defense"]]}


def arrays(p: dict, a: int, b: int) -> dict:
    return {"off": p["off"][a : b + 1], "def": p["def"][a : b + 1], "ball": p["ball"][a : b + 1],
            "handler": p["handler"][a : b + 1], "n": b - a + 1}  # fmt: skip


def item_m(k: dict, p: dict, pos: dict, practice: bool = False) -> dict:
    n = p["n_steps"]
    kf = k.get("keyframes") or [int(round(0.35 * (n - 1))), int(round(0.75 * (n - 1)))]
    end = min(n - 1, kf[1] + M_TAIL)
    key = f"M:{k['game_id']}:{k['possession_id']}"
    return {"id": item_id("P" if practice else "M", key), "task": "M", "practice": practice,
            "keyframes": kf, "people": people(p, pos), "offset": 0, **arrays(p, 0, end)}  # fmt: skip


def item_e(k: dict, p: dict, pos: dict, practice: bool = False) -> dict:
    n, t0 = p["n_steps"], p["t_start_ms"]
    kind = "switch" if k["category"] in ("switch", "decoy_switch") else "help"
    (cb, ca), (wb, wa) = (SWITCH_CLIP, SWITCH_WIN) if kind == "switch" else (HELP_CLIP, HELP_WIN)
    r = step_of(k["t_ms"], t0, n)
    a, b = max(0, r - cb), min(n - 1, r + ca)
    def_ids = [x["id"] for x in p["defense"]]
    ring = [def_ids.index(d) for d in k["defs"]]
    key = f"E:{k['game_id']}:{k['possession_id']}:{k['t_ms']}"
    return {"id": item_id("Q" if practice else "E", key), "task": "E", "kind": kind,
            "practice": practice, "ring": ring, "window": [max(0, r - wb - a), min(b, r + wa) - a],
            "people": people(p, pos), "offset": a, "ref_step": r, **arrays(p, a, b)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--lang", default="zh", choices=["en", "zh"], help="interface language")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    r = (
        D.rosters(cfg)
        .select(["player_id", "position"])
        .drop_nulls()
        .unique("player_id", keep="first")
    )
    pos = {int(a): str(b) for a, b in r.iter_rows()}
    bundle = json.loads((ROOT / "bundles" / f"{VERSION}.json").read_text(encoding="utf-8"))
    by = {(p["game_id"], int(p["possession_id"])): p for p in bundle["possessions"]}
    K = json.loads((ROOT / "quick" / "key.json").read_text())
    template = TEMPLATE.read_text(encoding="utf-8")
    assert SLOT in template

    build = lambda k, pr=False: (item_m if k["task"] == "M" else item_e)(  # noqa: E731
        k, by[(k["game_id"], int(k["possession_id"]))], pos, pr)  # fmt: skip
    items = [build(k) for k in K["items"]]
    prac = {k["task"]: build(k, True) for k in K["practice"]}
    ids = [x["id"] for x in items]
    assert len(set(ids)) == len(ids), "item id collision"
    alloc = {x["id"]: {"index": i, "task": k["task"], "category": k["category"],
                       "game_id": k["game_id"], "possession_id": int(k["possession_id"]),
                       **({"keyframes": x["keyframes"]} if x["task"] == "M" else
                          {"ring_slots": x["ring"], "ref_step": x["ref_step"], "offset": x["offset"],
                           "window": x["window"], "defs": k["defs"], "t_ms": k["t_ms"]})}
             for i, (k, x) in enumerate(zip(K["items"], items, strict=True))}  # fmt: skip

    out = ROOT / "quick" / VERSION
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary = {"version": VERSION, "created": datetime.now(UTC).isoformat(timespec="seconds"),
               "annotators": {}}  # fmt: skip
    for code in args.annotators:
        m = [x for x in items if x["task"] == "M"]
        e = [x for x in items if x["task"] == "E"]
        order = ([prac["M"]] + [m[i] for i in rng.permutation(len(m))]
                 + [prac["E"]] + [e[i] for i in rng.permutation(len(e))])  # fmt: skip
        data = {"version": VERSION, "annotator": code, "hz": 5, "lang": args.lang,
                "items": order, "estimate_min": EST}  # fmt: skip
        f = out / f"{code}.html"
        f.write_text(
            template.replace(SLOT, json.dumps(data, separators=(",", ":"))), encoding="utf-8"
        )
        summary["annotators"][code] = {
            "M": len(m), "E": len(e), "kb": round(f.stat().st_size / 1024),
            "minutes_est": round(EST["guide"] + EST["M"] * len(m) + EST["E"] * len(e)),
            "order": [x["id"] for x in order]}  # fmt: skip
    (out / "allocation.json").write_text(json.dumps({"items": alloc, "summary": summary}, indent=1))
    # a two-item sample page (the practice items of training games, shown as ordinary items) for
    # reviewing the format; it does not touch the scored items
    for lang, name in (("en", "sample.html"), ("zh", "sample_zh.html")):
        sample = {"version": f"{VERSION}_sample_{lang}", "annotator": "SAMPLE", "hz": 5,
                  "lang": lang, "estimate_min": EST,
                  "items": [prac["M"] | {"practice": False}, prac["E"] | {"practice": False}]}  # fmt: skip
        (out / name).write_text(
            template.replace(SLOT, json.dumps(sample, separators=(",", ":"))), encoding="utf-8"
        )
    print(json.dumps({c: {k: v for k, v in s.items() if k != "order"}
                      for c, s in summary["annotators"].items()}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
