"""Evaluate the quick check of matchups and events (D-036; pages: make_quick_check.py).

Inputs: data/annotations/quick/gdq_v1/allocation.json, the bundle gdq_v1, key.json and the
answer files data/annotations/raw/<code>_gdq_v1_answers.json (one per annotator).

Part 1 (matchups, test games, blind). Per possession, moment and defender the annotator's choice
(attacker, nobody = -1, can't tell = excluded) against
  hmm               the matchup model's stable path (help state = nobody), as pre-labelled
  min_residual, nearest_defender, franks_fixed    baselines of phase2_annotation_eval
Accuracy over all answered defender-moments and over those where the annotator named a man;
raw and weighted by the stratum shares of the test-game pool; 95 % intervals from a bootstrap
of possessions within strata.
Part 2 (events, test games). Per category the share confirmed (help: full help or stunt; switch:
yes) among clips not answered "can't tell", with Wilson and possession-bootstrap intervals; the
stunt share among confirmed help; the can't-tell share. Help answered "already in a help spot,
only sank toward the ball" (prepositioned) counts as not confirmed in the main share; its share
is reported, and a lenient share counts it as confirmed. Decoys give the yes-rate on clips
without a detected event (a check of yes-bias, not recall).
Consensus (when ADJ_gdq_v1_answers.json exists, make_quick_adjudication.py): Part 2 statistics on
the agreed answers plus the jointly reviewed answers where the two differed.
Two annotators (when both files exist): agreement on matchups (Cohen's kappa over the 7
answer values) and on events (raw agreement), and the main numbers for each annotator.

Output (aggregates only): reports/phase2/quick_check_eval.json

Usage:
    uv run python scripts/quick_check_eval.py [--raw-dir data/annotations/raw]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_annotation_eval import baseline_lanes  # noqa: E402

VERSION = "gdq_v1"
ROOT = Path("data/annotations")
CANT = -9
METHODS = ("hmm", "min_residual", "nearest_defender", "franks_fixed")
YES = {"help": {"full", "stunt"}, "switch": {"yes"}}
REPS = 2000


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def matchup_rows(alloc: dict, by: dict, answers: dict) -> list[dict]:
    """One row per answered possession: counts of correct per method, by moment-defender."""
    rows = []
    for iid, a in alloc.items():
        if a["task"] != "M" or iid not in answers:
            continue
        p = by[(a["game_id"], a["possession_id"])]
        lanes = {"hmm": np.asarray(p["matchup_prelabel"], dtype=int)} | baseline_lanes(p)
        r = {"stratum": a["category"], "n": 0, "n_man": 0, "cant": 0}
        r |= {m: 0 for m in METHODS} | {f"{m}_man": 0 for m in METHODS}
        for j, k in enumerate(a["keyframes"]):
            for d, v in enumerate(answers[iid]["km"][j]):
                if v is None:
                    continue
                if v == CANT:
                    r["cant"] += 1
                    continue
                r["n"] += 1
                r["n_man"] += v >= 0
                for m in METHODS:
                    hit = int(lanes[m][d, k] == v)
                    r[m] += hit
                    r[f"{m}_man"] += hit * (v >= 0)
        rows.append(r)
    return rows


def matchup_stats(rows: list[dict], weights: dict) -> dict:
    strata = sorted({r["stratum"] for r in rows})
    idx = {s: np.array([i for i, r in enumerate(rows) if r["stratum"] == s]) for s in strata}
    keys = ["n", "n_man", "cant"] + [m for m in METHODS] + [f"{m}_man" for m in METHODS]
    X = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    c = {k: i for i, k in enumerate(keys)}
    w = {s: weights[s] for s in strata}
    tot = sum(w.values())

    def stats(sel: dict) -> dict:
        S = {s: X[sel[s]].sum(0) for s in strata}
        T = sum(S.values())
        out = {}
        for m in METHODS:
            out[f"{m}"] = T[c[m]] / T[c["n"]]
            out[f"{m}_steps_with_man"] = T[c[f"{m}_man"]] / T[c["n_man"]]
            out[f"{m}_pool_weighted"] = sum(w[s] / tot * S[s][c[m]] / S[s][c["n"]] for s in strata)
        out["hmm_minus_best_baseline"] = out["hmm"] - max(out[m] for m in METHODS[1:])
        return out

    est = stats(idx)
    rng = np.random.default_rng(9)
    boot = [stats({s: rng.choice(idx[s], len(idx[s])) for s in strata}) for _ in range(REPS)]
    res = {k: {"estimate": float(v),
               "ci95": [float(np.quantile([b[k] for b in boot], q)) for q in (0.025, 0.975)]}
           for k, v in est.items()}  # fmt: skip
    T = X.sum(0)
    res["counts"] = {"possessions": len(rows), "defender_moments": int(T[c["n"]]),
                     "with_man": int(T[c["n_man"]]), "cant_tell": int(T[c["cant"]]),
                     "by_stratum": {s: len(idx[s]) for s in strata}}  # fmt: skip
    res["weights"] = w
    return res


def event_stats(alloc: dict, answers: dict) -> dict:
    out = {}
    cats = sorted({a["category"] for a in alloc.values() if a["task"] == "E"})
    rng = np.random.default_rng(9)
    for cat in cats:
        ids = [i for i, a in alloc.items() if a["category"] == cat and i in answers
               and answers[i].get("a")]  # fmt: skip
        kind = "switch" if "switch" in cat else "help"
        v = [answers[i]["a"] for i in ids]
        dec = [x for x in v if x != "cant"]
        yes = [x in YES[kind] for x in dec]
        k, n = int(sum(yes)), len(dec)
        r = {"answered": len(v), "cant_tell": len(v) - n, "decided": n, "confirmed": k,
             "share_confirmed": k / n if n else None, "wilson95": wilson(k, n)}  # fmt: skip
        if n:
            b = rng.choice(np.array(yes, dtype=float), size=(REPS, n)).mean(1)
            r["bootstrap95"] = [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
        if kind == "help":
            pp = sum(x == "prepositioned" for x in dec)
            r["prepositioned"] = pp
            r["share_prepositioned"] = pp / n if n else None
            r["share_confirmed_lenient"] = (k + pp) / n if n else None
            r["wilson95_lenient"] = wilson(k + pp, n)
            st = sum(x == "stunt" for x in dec)
            r["stunt"] = st
            r["stunt_share_of_confirmed"] = st / k if k else None
            r["stunt_wilson95"] = wilson(st, k)
        else:
            r["only_one_or_other"] = sum(x == "other" for x in dec)
        out[cat] = r
    # pooled detected help (early + late + corner) and detected events overall
    for name, group in (("help_detected", ("help_early", "help_late", "help_corner")),
                        ("all_detected", ("help_early", "help_late", "help_corner", "switch"))):  # fmt: skip
        k = sum(out[c]["confirmed"] for c in group if c in out)
        n = sum(out[c]["decided"] for c in group if c in out)
        out[name] = {"confirmed": k, "decided": n, "share_confirmed": k / n if n else None,
                     "wilson95": wilson(k, n), "note": "unweighted pool of the sampled clips"}  # fmt: skip
    return out


def kappa(a: list, b: list) -> float | None:
    if not a:
        return None
    cats = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b, strict=True)])
    pe = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def agreement(alloc: dict, A: dict, B: dict) -> dict:
    ma, mb, ea, eb = [], [], [], []
    for iid, a in alloc.items():
        if iid not in A or iid not in B:
            continue
        if a["task"] == "M":
            for ra, rb in zip(A[iid]["km"], B[iid]["km"], strict=True):
                for x, y in zip(ra, rb, strict=True):
                    if x is not None and y is not None and CANT not in (x, y):
                        ma.append(x)
                        mb.append(y)
        elif A[iid].get("a") and B[iid].get("a") and "cant" not in (A[iid]["a"], B[iid]["a"]):
            kind = "switch" if "switch" in a["category"] else "help"
            ea.append(A[iid]["a"] in YES[kind])
            eb.append(B[iid]["a"] in YES[kind])
    return {"matchups": {"n": len(ma), "agreement": float(np.mean(np.array(ma) == np.array(mb)))
                         if ma else None, "kappa": kappa(ma, mb)},
            "events_confirmed": {"n": len(ea), "agreement": float(np.mean(np.array(ea) == np.array(eb)))
                                 if ea else None, "kappa": kappa(ea, eb)}}  # fmt: skip


def consensus(alloc: dict, a1: dict, a2: dict, review: dict) -> dict:
    """Part 2 consensus: the common answer where both agree, else the review answer."""
    out = {}
    for i, a in alloc.items():
        if a["task"] != "E":
            continue
        x, y = (a1.get(i) or {}).get("a"), (a2.get(i) or {}).get("a")
        out[i] = {"a": x} if x == y else {"a": (review.get(i) or {}).get("a")}
    return out


def evaluate(alloc: dict, by: dict, answers: dict, weights: dict) -> dict:
    rows = matchup_rows(alloc, by, answers)
    return {"matchups": matchup_stats(rows, weights) if rows else None,
            "events": event_stats(alloc, answers)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/annotations/raw")
    ap.add_argument("--out", default="reports/phase2/quick_check_eval.json")
    args = ap.parse_args()
    alloc = json.loads((ROOT / "quick" / VERSION / "allocation.json").read_text())["items"]
    bundle = json.loads((ROOT / "bundles" / f"{VERSION}.json").read_text(encoding="utf-8"))
    by = {(p["game_id"], int(p["possession_id"])): p for p in bundle["possessions"]}
    pool = json.loads((ROOT / "quick" / "key.json").read_text())["meta"]["m_pool_test"]
    weights = {k: v / sum(pool.values()) for k, v in pool.items()}
    files = sorted(Path(args.raw_dir).glob(f"*_{VERSION}_answers.json"))
    if not files:
        raise SystemExit(f"no answer files in {args.raw_dir}")
    ans = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        ans[d["annotator"]] = {k: v for k, v in d["answers"].items() if k in alloc}
    secs = {c: json.loads(f.read_text(encoding="utf-8")).get("seconds", {}) for c, f in
            zip(ans, files, strict=True)}  # fmt: skip
    adj = ans.pop("ADJ", None)  # consensus review of the Part 2 disagreements
    secs.pop("ADJ", None)
    rep = {"version": VERSION, "annotators": sorted(ans), "split": "test",
           "primary": "A1" if "A1" in ans else sorted(ans)[0]}  # fmt: skip
    rep["by_annotator"] = {c: evaluate(alloc, by, a, weights) for c, a in ans.items()}
    rep["minutes"] = {c: {t: round(sum(v for k, v in s.items() if k in alloc
                                       and alloc[k]["task"] == t) / 60, 1) for t in ("M", "E")}
                      for c, s in secs.items()}  # fmt: skip
    if len(ans) >= 2:
        c1, c2 = sorted(ans)[:2]
        rep["agreement"] = agreement(alloc, ans[c1], ans[c2])
        if adj is not None:
            cons = consensus(alloc, ans[c1], ans[c2], adj)
            rep["consensus"] = {"events": event_stats(alloc, cons),
                                "n_reviewed": sum(1 for i in adj if i in alloc),
                                "n_unresolved": sum(1 for i in alloc if alloc[i]["task"] == "E"
                                                    and not (cons.get(i) or {}).get("a")),
                                "note": "agreed answers of the two annotators, review answer "
                                        "where they differed"}  # fmt: skip
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    p = rep["by_annotator"][rep["primary"]]
    if p["matchups"]:
        mm = p["matchups"]
        print("matchups:", {m: round(mm[m]["estimate"], 3) for m in METHODS}, mm["counts"])
    print("events:", {k: (v["confirmed"], v["decided"]) for k, v in p["events"].items()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
