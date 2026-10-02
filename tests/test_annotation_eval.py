"""Annotation evaluation: lane truth, event matching, review counts, baseline and agreement."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "scripts" / "phase2_annotation_eval.py"
spec = importlib.util.spec_from_file_location("phase2_annotation_eval", _p)
ae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae)

OFF = [{"id": 10 + k} for k in range(5)]
DEF = [{"id": 20 + d} for d in range(5)]
N = 10


def _poss():
    rng = np.random.default_rng(0)
    off = rng.uniform([5, 5], [40, 45], size=(N, 5, 2))
    ball = np.concatenate([off[:, 0], np.full((N, 1), 5.0)], 1)
    # each defender d stands at attacker d's rule position: the baseline must recover d
    mu = 0.62 * off + 0.11 * ball[:, None, :2] + 0.27 * np.array([5.25, 25.0])
    return {
        "key": "g:1", "offense": OFF, "defense": DEF, "off": off.tolist(), "def": mu.tolist(),
        "ball": ball.tolist(), "matchup_prelabel": [[d] * N for d in range(5)],
        "prelabels": [
            {"type": "switch", "participants": [{"role": "defender_1", "player_id": 20},
                                                {"role": "defender_2", "player_id": 21}],
             "start_step": 4, "end_step": 6, "attributes": {"switch_kind": "screen_switch"}},
            {"type": "help_rotation", "participants": [{"role": "helper", "player_id": 22},
                                                       {"role": "threatened_attacker", "player_id": 10}],
             "start_step": 2, "end_step": 3, "attributes": {"help_kind": "leave_man"}},
        ],
    }  # fmt: skip


def _confirm_all(p):
    segs = [
        dict(copy.deepcopy(s), source="model_confirmed", pre_id=i)
        for i, s in enumerate(p["prelabels"])
    ]
    return {"annotator": "A", "possessions": {p["key"]: {"segments": segs, "rejected_prelabels": [],
                                                          "status": "done", "time_spent_s": 120}}}  # fmt: skip


def test_lanes_baseline_and_perfect_agreement():
    p = _poss()
    assert (ae.min_residual(p) == np.arange(5)[:, None]).all()
    r = ae.evaluate({"possessions": [p]}, _confirm_all(p))
    assert r["lanes"]["frame_accuracy_hmm"] == 1.0
    assert r["lanes"]["frame_accuracy_min_residual_steps_with_man"] == 1.0
    assert r["events"]["switch"]["f1"] == 1.0 and r["events"]["help_rotation"]["f1"] == 1.0
    assert r["prelabel_review"]["switch/screen_switch"] == {"confirmed": 1}
    assert r["help_kind_vs_model_definitions"] == {"leave_man": {"leave_man": 1}}


def test_corrections_deletions_and_additions():
    p = _poss()
    a = _confirm_all(p)
    pa = a["possessions"]["g:1"]
    pa["segments"] = [pa["segments"][0]]  # delete the help pre-label
    pa["rejected_prelabels"] = [1]
    pa["segments"].append({"type": "matchup", "participants": [{"role": "defender", "player_id": 23},
                           {"role": "attacker", "player_id": None}], "start_step": 0, "end_step": 4,
                           "source": "human"})  # fmt: skip
    pa["segments"].append({"type": "closeout", "participants": [{"role": "defender", "player_id": 24},
                           {"role": "attacker", "player_id": 14}], "start_step": 7, "end_step": 8,
                           "attributes": {}, "source": "human"})  # fmt: skip
    t = ae.truth_lanes(p, pa["segments"])
    assert (t[3, :5] == -1).all() and (t[3, 5:] == 3).all()
    r = ae.evaluate({"possessions": [p]}, a)
    assert np.isclose(r["lanes"]["share_defender_steps_changed_by_annotator"], 5 / 50)
    assert r["events"]["help_rotation"]["precision"] == 0.0
    assert r["events"]["closeout"]["recall"] == 0.0 and r["events"]["closeout"]["n_ref"] == 1
    assert r["prelabel_review"]["help_rotation/leave_man"] == {"deleted": 1}


def test_agreement_between_annotators():
    p = _poss()
    a1, a2 = _confirm_all(p), _confirm_all(p)
    a2["annotator"] = "B"
    g = ae.agreement({"possessions": [p]}, a1, a2)
    assert g["lane_agreement"] == 1.0 and g["event_agreement"]["f1"] == 1.0


def test_closeout_counted_once():
    p = _poss()
    p["prelabels"].append({"type": "closeout", "participants": [
        {"role": "defender", "player_id": 24}, {"role": "attacker", "player_id": 14}],
        "start_step": 7, "end_step": 8, "attributes": {}})  # fmt: skip
    r = ae.evaluate({"possessions": [p]}, _confirm_all(p))
    assert r["events"]["closeout"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_pred": 1,
                                       "n_ref": 1, "tp": 1}  # fmt: skip
