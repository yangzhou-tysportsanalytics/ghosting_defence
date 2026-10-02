"""Phase 2 task 5: evaluate the matchup model and its events against the annotated possessions.

Inputs: the bundle given to annotators (data/annotations/bundles/<bundle>.json) and the files
they saved (data/annotations/raw/<bundle>__<initials>.json). Only possessions marked ``done``.

* Matchup lanes. Truth per defender and 5 Hz step = the model lane overridden by the annotator's
  ``matchup`` segments (the tool's own rule). Compared: the HMM lane (the pre-label) and the
  min-residual baseline (each step, the attacker whose rule position g_o O_k + g_b B + g_h H is
  nearest; it never says "no man"). Frame accuracy over all defender-steps and over steps with a
  man; accuracy within 1 s of annotated switches; share of defender-steps the annotator changed
  (anchoring: an untouched lane counts as agreement).
* Events (switch, help_rotation, closeout). A model pre-label matches an annotated event of the
  same type with the same principal player (switch defender_1, help helper, closeout defender)
  starting within 2 steps (0.4 s). Precision, recall, F1 by type and kind; pre-label review
  counts (confirmed / edited / deleted); help kinds (annotated pre_positioned / leave_man /
  recovery_scramble) against model definitions A and B (D-018).
* Two annotators on the double-annotated possessions: lane agreement and event F1 between them.

Output (aggregates only): reports/phase2/<version>_all/annotation_eval.json

Usage:
    uv run python scripts/phase2_annotation_eval.py --bundle gd_v2_all
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ghost import data as D
from ghost.court import HOOP_LEFT

EVENT_TYPES = ("switch", "help_rotation", "closeout")
PRINCIPAL = {"switch": "defender_1", "help_rotation": "helper", "closeout": "defender"}
KIND = {"switch": "switch_kind", "help_rotation": "help_kind"}
TOL = 2  # steps (0.4 s)
GAMMA = (0.62, 0.11, 0.27)  # Franks et al. (2015)


def principal(seg: dict) -> int | None:
    role = PRINCIPAL.get(seg["type"])
    for r in seg.get("participants", []):
        if r["role"] == role:
            return r["player_id"]
    return None


def kind(seg: dict) -> str:
    return (seg.get("attributes") or {}).get(KIND.get(seg["type"], ""), "") or ""


def truth_lanes(poss: dict, segments: list[dict]) -> np.ndarray:
    """(5, n) attacker slot per defender and step (-1 = no man): the model lane, overridden by
    the annotator's matchup segments in segment order (as the tool draws it)."""
    lane = np.array(poss["matchup_prelabel"], dtype=int)
    n = lane.shape[1]
    off_ids = [p["id"] for p in poss["offense"]]
    def_ids = [p["id"] for p in poss["defense"]]
    for s in segments:
        if s["type"] != "matchup":
            continue
        did = s["participants"][0]["player_id"]
        if did not in def_ids:
            continue
        att = s["participants"][1]["player_id"] if len(s["participants"]) > 1 else None
        k = off_ids.index(att) if att in off_ids else -1
        a, b = max(0, s["start_step"]), min(n - 1, s["end_step"])
        lane[def_ids.index(did), a : b + 1] = k
    return lane


def min_residual(poss: dict, gamma=GAMMA) -> np.ndarray:
    """(5, n) nearest attacker by rule position (Phase 2 baseline)."""
    off = np.asarray(poss["off"], dtype=float)  # (n, 5, 2)
    dfn = np.asarray(poss["def"], dtype=float)  # (n, 5, 2)
    ball = np.asarray(poss["ball"], dtype=float)[:, :2]  # (n, 2)
    mu = gamma[0] * off + gamma[1] * ball[:, None] + gamma[2] * HOOP_LEFT  # (n, 5k, 2)
    d = np.linalg.norm(dfn[:, :, None] - mu[:, None], axis=-1)  # (n, 5d, 5k)
    return d.argmin(-1).T  # (5, n)


def match_events(pred: list[dict], ref: list[dict], tol: int = TOL) -> list[tuple[int, int]]:
    """Greedy one-to-one matches (index in pred, index in ref): same type and principal player,
    start within ``tol`` steps; closest starts first."""
    cand = []
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            if p["type"] == r["type"] and principal(p) == principal(r):
                gap = abs(p["start_step"] - r["start_step"])
                if gap <= tol:
                    cand.append((gap, i, j))
    used_i, used_j, out = set(), set(), []
    for _, i, j in sorted(cand):
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            out.append((i, j))
    return out


def prf(tp: int, n_pred: int, n_ref: int) -> dict:
    p = tp / n_pred if n_pred else None
    r = tp / n_ref if n_ref else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": p, "recall": r, "f1": f, "n_pred": n_pred, "n_ref": n_ref, "tp": tp}


def final_events(segments: list[dict]) -> list[dict]:
    return [s for s in segments if s["type"] in EVENT_TYPES and s.get("source") != "model"]


def evaluate(bundle: dict, ann: dict) -> dict:
    """Metrics of one annotator file against the model (lanes, baseline, events)."""
    poss = {p["key"]: p for p in bundle["possessions"]}
    lanes = Counter()
    sw_acc = Counter()
    ev = defaultdict(lambda: [0, 0, 0])  # (type, kind) -> tp, n_pred, n_ref
    review = defaultdict(Counter)  # (type, kind) -> confirmed / edited / deleted
    help_map = defaultdict(Counter)  # annotated help_kind -> which model kinds matched
    times, n_done = [], 0
    for key, a in ann["possessions"].items():
        if a.get("status") != "done" or key not in poss:
            continue
        n_done += 1
        p = poss[key]
        segs = a["segments"]
        truth = truth_lanes(p, segs)
        hmm = np.array(p["matchup_prelabel"], dtype=int)
        base = min_residual(p)
        man = truth >= 0
        lanes["n"] += truth.size
        lanes["n_man"] += int(man.sum())
        lanes["hmm"] += int((hmm == truth).sum())
        lanes["hmm_man"] += int(((hmm == truth) & man).sum())
        lanes["base_man"] += int(((base == truth) & man).sum())
        lanes["changed"] += int((hmm != truth).sum())
        def_ids = [q["id"] for q in p["defense"]]
        fin = final_events(segs)
        for s in fin:  # lane accuracy within 1 s of annotated switches, switching defenders
            if s["type"] != "switch":
                continue
            a0, b0 = max(0, s["start_step"] - 5), min(truth.shape[1] - 1, s["end_step"] + 5)
            for r in s["participants"]:
                if r["player_id"] in def_ids:
                    d = def_ids.index(r["player_id"])
                    sw_acc["n"] += b0 - a0 + 1
                    sw_acc["hmm"] += int((hmm[d, a0 : b0 + 1] == truth[d, a0 : b0 + 1]).sum())
                    sw_acc["base"] += int((base[d, a0 : b0 + 1] == truth[d, a0 : b0 + 1]).sum())
        pre = [s for s in p["prelabels"] if s["type"] in EVENT_TYPES]
        for s in segs:
            if s.get("source") in ("model_confirmed", "model_edited"):
                o = s.get("model_original") or s
                review[(o["type"], kind(o))][s["source"].split("_")[1]] += 1
        for pid in a.get("rejected_prelabels", []):
            if 0 <= pid < len(p["prelabels"]):
                o = p["prelabels"][pid]
                if o["type"] in EVENT_TYPES:
                    review[(o["type"], kind(o))]["deleted"] += 1
        m = match_events(pre, fin)
        for t in EVENT_TYPES:
            ps = [i for i, s in enumerate(pre) if s["type"] == t]
            rs = [j for j, s in enumerate(fin) if s["type"] == t]
            ev[(t, "")][0] += sum(1 for i, j in m if pre[i]["type"] == t)
            ev[(t, "")][1] += len(ps)
            ev[(t, "")][2] += len(rs)
        for i, j in m:  # by kind (types with kinds only): the annotated kind is the reference
            if kind(fin[j]):
                ev[(fin[j]["type"], kind(fin[j]))][0] += int(kind(pre[i]) == kind(fin[j]))
        for s in pre:
            ev[(s["type"], kind(s))][1] += 1 if kind(s) else 0
        for s in fin:
            ev[(s["type"], kind(s))][2] += 1 if kind(s) else 0
        for s in fin:  # D-018: which model help definitions find each annotated kind
            if s["type"] != "help_rotation":
                continue
            hits = {kind(q) for q in pre if q["type"] == "help_rotation"
                    and principal(q) == principal(s)
                    and abs(q["start_step"] - s["start_step"]) <= TOL}  # fmt: skip
            tag = "+".join(sorted(hits)) if hits else "none"
            help_map[kind(s) or "unspecified"][tag] += 1
        times.append(a.get("time_spent_s") or 0)
    acc = lambda x, n: x / n if n else None  # noqa: E731
    return {
        "annotator": ann.get("annotator"),
        "n_possessions_done": n_done,
        "lanes": {
            "frame_accuracy_hmm": acc(lanes["hmm"], lanes["n"]),
            "frame_accuracy_hmm_steps_with_man": acc(lanes["hmm_man"], lanes["n_man"]),
            "frame_accuracy_min_residual_steps_with_man": acc(lanes["base_man"], lanes["n_man"]),
            "share_defender_steps_changed_by_annotator": acc(lanes["changed"], lanes["n"]),
            "accuracy_within_1s_of_switches_hmm": acc(sw_acc["hmm"], sw_acc["n"]),
            "accuracy_within_1s_of_switches_min_residual": acc(sw_acc["base"], sw_acc["n"]),
            "n_defender_steps": lanes["n"],
        },
        "events": {f"{t}{'/' + k if k else ''}": prf(*v) for (t, k), v in sorted(ev.items())},
        "prelabel_review": {
            f"{t}{'/' + k if k else ''}": dict(c) for (t, k), c in sorted(review.items())
        },
        "help_kind_vs_model_definitions": {k: dict(v) for k, v in help_map.items()},
        "time_per_possession_s_median": float(np.median(times)) if times else None,
    }


def agreement(bundle: dict, a1: dict, a2: dict) -> dict:
    """Two annotators on the possessions both marked done."""
    poss = {p["key"]: p for p in bundle["possessions"]}
    same, n, tp, n1, n2, k = 0, 0, 0, 0, 0, 0
    for key, x in a1["possessions"].items():
        y = a2["possessions"].get(key)
        if not y or x.get("status") != "done" or y.get("status") != "done" or key not in poss:
            continue
        k += 1
        t1, t2 = truth_lanes(poss[key], x["segments"]), truth_lanes(poss[key], y["segments"])
        same += int((t1 == t2).sum())
        n += t1.size
        e1, e2 = final_events(x["segments"]), final_events(y["segments"])
        tp += len(match_events(e1, e2))
        n1, n2 = n1 + len(e1), n2 + len(e2)
    return {"n_possessions": k, "lane_agreement": same / n if n else None,
            "event_agreement": prf(tp, n1, n2)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="gd_v2_all")
    ap.add_argument("--raw-dir", default="data/annotations/raw")
    ap.add_argument("--out", default=None, help="output path (default: reports/phase2/...)")
    args = ap.parse_args()
    cfg = D.DataConfig.load(game_set="all")
    bundle = json.loads(Path(f"data/annotations/bundles/{args.bundle}.json").read_text())
    files = sorted(Path(args.raw_dir).glob(f"{args.bundle}__*.json"))
    if not files:
        raise SystemExit(f"no annotation files {args.raw_dir}/{args.bundle}__*.json")
    anns = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    rep = {"bundle": args.bundle, "files": [f.name for f in files], "tolerance_steps": TOL,
           "per_annotator": [evaluate(bundle, a) for a in anns], "agreement": []}  # fmt: skip
    for i in range(len(anns)):
        for j in range(i + 1, len(anns)):
            rep["agreement"].append(
                {
                    "pair": [anns[i].get("annotator"), anns[j].get("annotator")],
                    **agreement(bundle, anns[i], anns[j]),
                }  # fmt: skip
            )
    out = Path(args.out or Path("reports/phase2") / f"{cfg.version}_all" / "annotation_eval.json")
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
