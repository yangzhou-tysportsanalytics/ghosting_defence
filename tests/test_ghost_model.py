"""Ghost model: shapes, causality (D-006), masking (D-013), set loss, a tiny overfit."""

from __future__ import annotations

import numpy as np
import torch

from ghost.ghost.dataset import apply_mask
from ghost.ghost.model import (
    N_CTX,
    N_FEAT,
    PERMS,
    GhostConfig,
    GhostModel,
    ghost_loss,
    set_loss,
)
from ghost.ghost.train import TrainConfig, fit

T = 20


def _batch(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(B, 11, T, N_FEAT, generator=g) * 0.3
    feats[..., 5:8] = 0.0
    feats[..., 6] = 1.0
    ctx = torch.randn(B, T, N_CTX, generator=g) * 0.1
    target = torch.rand(B, 5, T, 2, generator=g) * torch.tensor([47.0, 50.0])
    valid = torch.ones(B, T, dtype=torch.bool)
    return feats, ctx, target, valid


def test_shapes_and_positive_sigma():
    m = GhostModel(GhostConfig(d_model=32, n_blocks=1, dropout=0.0), n_steps=T)
    feats, ctx, _, _ = _batch()
    out = m(feats, ctx)
    assert out["logits"].shape == (3, 5, T, 4)
    assert out["mu"].shape == (3, 5, T, 4, 2)
    assert (out["sigma"] > 0).all()


def test_online_is_causal_offline_is_not():
    torch.manual_seed(0)
    feats, ctx, _, _ = _batch()
    f2 = feats.clone()
    f2[:, :, 12:, :4] += 5.0  # change the future (all agents) after step 11
    for mode, causal in (("online", True), ("offline", False)):
        m = GhostModel(
            GhostConfig(d_model=32, n_blocks=2, dropout=0.0, mode=mode), n_steps=T
        ).eval()
        with torch.no_grad():
            a = m(feats, ctx)["mu"][:, :, :12]
            b = m(f2, ctx)["mu"][:, :, :12]
        same = torch.allclose(a, b, atol=1e-5)
        assert same == causal, mode


def test_masking_hides_defenders():
    feats, _, _, _ = _batch()
    f, masked = apply_mask(feats, "team")
    assert masked.all()
    assert (f[:, 6:, :, :4] == 0).all() and (f[:, 6:, :, 5] == 1).all()
    assert torch.equal(f[:, :6], feats[:, :6])  # ball and offence untouched
    f, masked = apply_mask(feats, "individual", torch.Generator().manual_seed(1))
    assert (masked.sum(-1) == 1).all()
    for b in range(3):
        j = int(masked[b].nonzero())
        assert (f[b, 6 + j, :, :4] == 0).all()
        others = [k for k in range(5) if k != j]
        assert torch.equal(f[b, [6 + k for k in others]], feats[b, [6 + k for k in others]])


def test_set_loss_finds_permutation():
    perm = torch.tensor([2, 0, 4, 1, 3])
    cost = torch.ones(1, 5, 5, T)
    cost[0, torch.arange(5), perm] = 0.0
    loss, best = set_loss(cost, torch.ones(1, T, dtype=torch.bool))
    assert float(loss) == 0.0 and torch.equal(best[0], perm)
    assert PERMS.shape == (120, 5)


def test_tiny_overfit_reduces_loss():
    rng = np.random.default_rng(0)
    N = 24
    off = rng.uniform(0, 1, size=(N, T, 5, 2)).astype(np.float32)
    feats = np.zeros((N, 11, T, N_FEAT), dtype=np.float32)
    feats[:, 1:6, :, 0:2] = np.transpose(off, (0, 2, 1, 3)) - 0.5
    feats[..., 6] = 1.0
    # defenders stand 3 ft toward the hoop from their man (a learnable rule)
    target = np.transpose(off, (0, 2, 1, 3)) * np.array([47.0, 50.0], dtype=np.float32)
    target[..., 0] -= 3.0
    arrs = {
        "feats": feats,
        "ctx": np.zeros((N, T, N_CTX), dtype=np.float32),
        "target": target.astype(np.float32),
        "valid": np.ones((N, T), dtype=bool),
    }
    arrs["feats"][:, 6:, :, 0:2] = (target / np.array([47.0, 50.0]) - 0.5).astype(np.float32)
    mcfg = GhostConfig(d_model=32, n_blocks=1, dropout=0.0, n_comp=2)
    _, hist = fit(
        arrs,
        None,
        mcfg,
        TrainConfig(epochs=15, batch_size=8, lr=3e-3, schedule="individual"),
        log=lambda r: None,
    )
    assert hist[-1]["train_loss"] < hist[0]["train_loss"] - 0.5


def test_ghost_loss_individual_uses_own_slot():
    m = GhostModel(GhostConfig(d_model=16, n_blocks=1, dropout=0.0), n_steps=T)
    feats, ctx, target, valid = _batch()
    f, masked = apply_mask(feats, "individual", torch.Generator().manual_seed(0))
    loss, info = ghost_loss(m(f, ctx), target, valid, masked, "nll")
    assert torch.isfinite(loss) and info["n_obs"] == 3 * T


def test_fill_shot_clock_nearest_frame():
    import polars as pl

    from ghost.tensors import PossessionTensors, fill_shot_clock

    n, T = 1, 4
    z = np.zeros
    pt = PossessionTensors(
        game_id=np.array(["g"]),
        possession_id=np.array([1]),
        t=np.array([[1000, 1200, 1400, 1600]]),
        valid=np.array([[True, True, True, False]]),
        ball=z((n, T, 3)),
        ball_v=z((n, T, 2)),
        off_xy=z((n, T, 5, 2)),
        off_v=z((n, T, 5, 2)),
        def_xy=z((n, T, 5, 2)),
        def_v=z((n, T, 5, 2)),
        handler=z((n, T), dtype=np.int8),
        game_clock=z((n, T)),
        shot_clock=np.full((n, T), np.nan),
        off_ids=z((n, 5)),
        def_ids=z((n, 5)),
    )
    tab = pl.DataFrame(
        {
            "period": [1] * 4,
            "unix_ms": [990, 1210, 1390, 1610],
            "shot_clock_filled": [20.0, 19.8, 19.6, 19.4],
            "shot_clock_imputed": [False, True, False, False],
        }
    )
    fill_shot_clock(pt, {"g": tab})
    assert np.allclose(pt.shot_clock[0, :3], [20.0, 19.8, 19.6]) and np.isnan(pt.shot_clock[0, 3])
    assert pt.shot_clock_imputed[0].tolist() == [False, True, False, False]


def test_checkpoint_resume(tmp_path):
    rng = np.random.default_rng(0)
    N = 8
    arrs = {
        "feats": rng.normal(0, 0.3, size=(N, 11, T, N_FEAT)).astype(np.float32),
        "ctx": np.zeros((N, T, N_CTX), dtype=np.float32),
        "target": rng.uniform(0, 40, size=(N, 5, T, 2)).astype(np.float32),
        "valid": np.ones((N, T), dtype=bool),
    }
    mcfg = GhostConfig(d_model=16, n_blocks=1, dropout=0.0, n_comp=2)
    ck = str(tmp_path / "ck.pt")
    _, h1 = fit(
        arrs,
        None,
        mcfg,
        TrainConfig(epochs=2, batch_size=4, checkpoint_path=ck),
        log=lambda r: None,
    )
    _, h2 = fit(
        arrs,
        None,
        mcfg,
        TrainConfig(epochs=3, batch_size=4, checkpoint_path=ck),
        log=lambda r: None,
    )
    assert len(h1) == 2 and len(h2) == 3 and h2[:2] == h1
