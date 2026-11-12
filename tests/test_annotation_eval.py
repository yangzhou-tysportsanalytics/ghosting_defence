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


def test_population_weighted_accuracy():
    from collections import Counter

    by = {
        "a": Counter(hmm=9, n=10, hmm_man=9, n_man=10),
        "b": Counter(hmm=5, n=10, hmm_man=5, n_man=10),
    }
    w = ae.weighted(by, {"a": 0.75, "b": 0.25, "c": 0.5})  # "c" absent from the sample: ignored
    assert np.isclose(w["frame_accuracy_hmm"], 0.75 * 0.9 + 0.25 * 0.5)
    assert ae.weighted(by, None) is None


def test_blind_vs_assisted_anchoring_gap():
    p = _poss()
    bp = copy.deepcopy(p)
    bp["prelabels"], bp["matchup_prelabel"] = [], [[ae.UNLABELLED] * N for _ in range(5)]
    # blind: four defenders as the model, defender 23 guards nobody; events re-created by hand
    segs = [
        {
            "type": "matchup",
            "participants": [
                {"role": "defender", "player_id": 20 + d},
                {"role": "attacker", "player_id": None if d == 3 else 10 + d},
            ],
            "start_step": 0,
            "end_step": N - 1,
            "source": "human",
        }
        for d in range(5)
    ]
    segs += [dict(copy.deepcopy(s), source="human") for s in p["prelabels"]]
    blind = {"annotator": "A", "possessions": {"g:1": {"segments": segs, "status": "done"}}}
    r = ae.blind_eval({"possessions": [p]}, {"possessions": [bp]}, blind, _confirm_all(p), reps=50)
    assert r["n_possessions"] == 1 and r["n_defender_steps"] == 5 * N
    assert np.isclose(r["hmm_accuracy_vs_blind"], 0.8) and r["hmm_accuracy_vs_assisted"] == 1.0
    assert np.isclose(r["anchoring_gap_assisted_minus_blind"], 0.2)
    assert r["events_model_vs_blind"]["f1"] == 1.0


def test_unlabelled_steps_are_excluded():
    p = _poss()
    bp = copy.deepcopy(p)
    bp["prelabels"], bp["matchup_prelabel"] = [], [[ae.UNLABELLED] * N for _ in range(5)]
    seg = {
        "type": "matchup",
        "participants": [
            {"role": "defender", "player_id": 20},
            {"role": "attacker", "player_id": 10},
        ],
        "start_step": 0,
        "end_step": 4,
    }
    t = ae.truth_lanes(bp, [seg])
    assert (t[0, :5] == 0).all() and (t[0, 5:] == ae.UNLABELLED).all() and (t[1:] == -2).all()


def test_baselines_recover_simple_matchups():
    p = _poss()
    # defenders on their attackers' rule positions: the Franks HMM and min-residual recover d
    assert (ae.franks_fixed(p) == np.arange(5)[:, None]).all()
    # defenders 0.1 ft from their own attacker (fixture attackers can be < 1 ft apart)
    q = copy.deepcopy(p)
    q["def"] = (np.asarray(p["off"]) + np.array([0.1, 0.0])).tolist()
    assert (ae.nearest_defender(q) == np.arange(5)[:, None]).all()
    r = ae.evaluate({"possessions": [p]}, _confirm_all(p), {"x": 1.0})
    b = r["lanes"]["baselines"]
    assert b["franks_fixed"]["frame_accuracy"] == 1.0 and b["min_residual"]["frame_accuracy"] == 1.0
    assert set(b) == {"min_residual", "nearest_defender", "franks_fixed"}


def test_extra_lanes_are_evaluated_like_baselines():
    p = _poss()
    wrong = np.array(p["matchup_prelabel"])
    wrong[0, :5] = 4  # B2 lane wrong for defender 0 on half the steps
    p["_extra_lanes"] = {"event_hmm": wrong, "event_hmm_constant": np.array(p["matchup_prelabel"])}
    r = ae.evaluate({"possessions": [p]}, _confirm_all(p))
    b = r["lanes"]["baselines"]
    assert np.isclose(b["event_hmm"]["frame_accuracy"], 45 / 50)
    assert b["event_hmm_constant"]["frame_accuracy"] == 1.0
    assert (
        "event_hmm"
        not in ae.evaluate({"possessions": [_poss()]}, _confirm_all(_poss()))["lanes"]["baselines"]
    )


def test_event_params_roundtrip():
    d = {"gamma": {"strong": [0.64, 0.1, 0.26], "weak": [0.58, 0.1, 0.32]},
         "sigma_ft": {"strong": 2.5, "weak": 2.6}, "help_lam": 0.67, "help_sigma_ft": 14.8,
         "stratify": True, "help_state": True,
         "beta": {"intercept": -4.1, "screen": -0.3, "involved": 0.9, "pass": 0.3, "drive": 0.5}}  # fmt: skip
    prm = ae._event_params(d)
    assert np.isclose(prm.emis.sigma2["weak"], 2.6**2) and prm.emis.n_states == 6
    assert np.allclose(prm.beta, [-4.1, -0.3, 0.9, 0.3, 0.5])
