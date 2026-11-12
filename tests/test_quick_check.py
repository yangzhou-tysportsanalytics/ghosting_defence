"""Helpers of the quick check (scripts/make_quick_check.py, scripts/quick_check_eval.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_S = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _S / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MK = _load("make_quick_check")
EV = _load("quick_check_eval")


def _poss(n: int = 40) -> dict:
    off = [{"id": 10 + k, "name": f"A B{k}", "jersey": str(k)} for k in range(5)]
    dfn = [{"id": 20 + k, "name": f"C D{k}", "jersey": str(k + 5)} for k in range(5)]
    xy = [[[float(k), float(t)] for k in range(5)] for t in range(n)]
    return {"offense": off, "defense": dfn, "n_steps": n, "t_start_ms": 1_000_000,
            "off": xy, "def": xy, "ball": [[0.0, 0.0, 1.0]] * n, "handler": [0] * n}  # fmt: skip


def test_step_of_clips():
    assert MK.step_of(1_000_000, 1_000_000, 40) == 0
    assert MK.step_of(1_002_000, 1_000_000, 40) == 10
    assert MK.step_of(2_000_000, 1_000_000, 40) == 39


def test_help_clip_window_and_ring():
    k = {"category": "help_early", "game_id": "g", "possession_id": 1, "defs": [22],
         "t_ms": 1_004_000}  # fmt: skip
    it = MK.item_e(k, _poss(), {})
    assert it["kind"] == "help" and it["ring"] == [2]
    assert it["offset"] == 20 - 15 and it["n"] == 15 + 12 + 1
    w0, w1 = it["window"]
    assert w0 == 15 - 5 and w1 == 15 + 10 and 0 <= w0 < w1 <= it["n"] - 1


def test_switch_and_decoy_share_the_form():
    base = {"game_id": "g", "possession_id": 1, "defs": [20, 21], "t_ms": 1_004_000}
    a = MK.item_e(base | {"category": "switch"}, _poss(), {})
    b = MK.item_e(base | {"category": "decoy_switch"}, _poss(), {})
    assert a["kind"] == b["kind"] == "switch" and a["window"] == b["window"]
    assert a["id"] == b["id"]  # the id does not depend on the category


def test_matchup_clip_ends_after_second_moment():
    it = MK.item_m({"game_id": "g", "possession_id": 1, "keyframes": [5, 20]}, _poss(), {})
    assert it["n"] == 20 + MK.M_TAIL + 1 and it["keyframes"] == [5, 20]


def test_wilson_and_kappa():
    lo, hi = EV.wilson(8, 10)
    assert 0.4 < lo < 0.8 < hi < 1.0
    assert EV.wilson(0, 0) is None
    assert EV.kappa([0, 1, 2, 0], [0, 1, 2, 0]) == 1.0


def test_template_has_data_slot():
    html = (_S.parent / "tools" / "quick_check" / "index.html").read_text(encoding="utf-8")
    assert MK.SLOT in html


def test_consensus_uses_review_only_where_annotators_differ():
    alloc = {"E-1": {"task": "E"}, "E-2": {"task": "E"}, "M-1": {"task": "M"}}
    a1 = {"E-1": {"a": "no"}, "E-2": {"a": "stunt"}}
    a2 = {"E-1": {"a": "no"}, "E-2": {"a": "full"}}
    c = EV.consensus(alloc, a1, a2, {"E-1": {"a": "full"}, "E-2": {"a": "full"}})
    assert c == {"E-1": {"a": "no"}, "E-2": {"a": "full"}}


def test_disagreements_lists_differing_part2_items():
    AD = _load("make_quick_adjudication")
    alloc = {"E-1": {"task": "E"}, "E-2": {"task": "E"}, "M-1": {"task": "M"}}
    a1 = {"E-1": {"a": "no"}, "E-2": {"a": "stunt"}, "M-1": {"km": []}}
    a2 = {"E-1": {"a": "no"}, "E-2": {"a": "full"}, "M-1": {"km": [1]}}
    assert AD.disagreements(alloc, a1, a2) == ["E-2"]
