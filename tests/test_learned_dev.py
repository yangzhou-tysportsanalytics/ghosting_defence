"""Learned-ghost deviations: per-step metrics agree with the model's own losses."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from ghost.ghost.dataset import mask_defenders
from ghost.ghost.model import N_CTX, N_FEAT, GhostConfig, GhostModel, mixture_nll, slot_mixture_nll

_p = Path(__file__).resolve().parents[1] / "scripts" / "phase4_learned_ghost_dev.py"
spec = importlib.util.spec_from_file_location("phase4_learned_ghost_dev", _p)
lg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lg)

T = 12


def _batch(B=3):
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(B, 11, T, N_FEAT, generator=g) * 0.3
    feats[..., 5:8] = 0.0
    feats[..., 6] = 1.0
    ctx = torch.zeros(B, T, N_CTX)
    target = torch.rand(B, 5, T, 2, generator=g) * torch.tensor([47.0, 50.0])
    return {"feats": feats, "ctx": ctx, "target": target}


def test_score_matches_model_losses():
    torch.manual_seed(0)
    m = GhostModel(GhostConfig(d_model=16, n_blocks=1, dropout=0.0, anchor="rule"), T).eval()
    with torch.no_grad():
        m.head.weight.normal_(0, 0.1)
    b = _batch()
    res = lg.score(m, b)
    assert all(v.shape == (3, 5, T) and torch.isfinite(v).all() for v in res.values())
    # team: mean over valid steps equals the identity-free slot-mixture NLL
    with torch.no_grad():
        out = m(mask_defenders(b["feats"], torch.ones(3, 5, dtype=torch.bool)), b["ctx"])
    tot, n = slot_mixture_nll(out, b["target"], torch.ones(3, T, dtype=torch.bool))
    assert math.isclose(float(res["nll_team"].mean()), tot / n, rel_tol=1e-5)
    # individual: defender 2 hidden alone, his own slot
    one = torch.zeros(3, 5, dtype=torch.bool)
    one[:, 2] = True
    with torch.no_grad():
        nll = mixture_nll(m(mask_defenders(b["feats"], one), b["ctx"]), b["target"])[:, 2, 2]
    assert torch.allclose(res["nll_ind"][:, 2], nll, atol=1e-5)
    assert (res["d_mode"] >= 0).all() and (res["d_mean"] >= 0).all()
