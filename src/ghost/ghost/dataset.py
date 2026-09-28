"""Model inputs from possession tensors, and masking schedules (D-013)."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from ghost.ghost.model import N_CTX, N_FEAT, V_SCALE, X_SCALE, Y_SCALE
from ghost.tensors import PossessionTensors


def build_arrays(pt: PossessionTensors, possessions: pl.DataFrame) -> dict[str, np.ndarray]:
    """Unmasked features (N, 11, T, N_FEAT), context (N, T, N_CTX), targets (N, 5, T, 2) feet,
    valid (N, T). Agent order: ball, offence 0-4, defence 0-4."""
    N, T = pt.valid.shape
    xy = np.concatenate([pt.ball[:, :, None, :2], pt.off_xy, pt.def_xy], axis=2)  # (N,T,11,2)
    v = np.concatenate([pt.ball_v[:, :, None], pt.off_v, pt.def_v], axis=2)
    f = np.zeros((N, T, 11, N_FEAT), dtype=np.float32)
    f[..., 0] = xy[..., 0] / X_SCALE - 0.5
    f[..., 1] = xy[..., 1] / Y_SCALE - 0.5
    f[..., 2:4] = v / V_SCALE
    h = pt.handler.astype(int)
    for k in range(5):
        f[:, :, 1 + k, 4] = (h == k).astype(np.float32)
    f[..., 6] = 1.0  # visible
    f[~pt.valid, :, :] = 0.0
    f[~pt.valid, :, 7] = 1.0  # padding flag
    feats = np.transpose(f, (0, 2, 1, 3))  # (N,11,T,F)

    key = pl.DataFrame({"game_id": pt.game_id, "possession_id": pt.possession_id})
    ctxdf = key.join(
        possessions.select(["game_id", "possession_id", "score_margin_offense", "period"]),
        on=["game_id", "possession_id"],
        how="left",
    )
    ctx = np.zeros((N, T, N_CTX), dtype=np.float32)
    ctx[..., 0] = ctxdf["score_margin_offense"].fill_null(0).to_numpy()[:, None] / 10.0
    ctx[..., 1] = pt.game_clock / 720.0
    sc = pt.shot_clock
    ctx[..., 2] = np.nan_to_num(sc, nan=0.0) / 24.0
    # flag: shot clock missing, or imputed by rule (nbacore v1.2 shot_clock_imputed)
    flag = np.isnan(sc)
    if pt.shot_clock_imputed is not None:
        flag = flag | pt.shot_clock_imputed
    ctx[..., 3] = flag.astype(np.float32)
    ctx[..., 4] = ctxdf["period"].fill_null(1).to_numpy()[:, None] / 4.0
    target = np.transpose(pt.def_xy, (0, 2, 1, 3)).astype(np.float32)  # (N,5,T,2)
    out = {"feats": feats, "ctx": ctx, "target": target, "valid": pt.valid.copy()}
    if pt.shot_clock_imputed is not None:
        out["sc_imputed"] = pt.shot_clock_imputed.copy()
    return out


def apply_mask(
    feats: torch.Tensor, schedule: str, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hide defenders. schedule: 'team' (all five), 'individual' (one random defender per
    possession), or 'mixed' (half the batch each). Returns (masked feats, masked (B, 5) bool)."""
    B = feats.shape[0]
    masked = torch.zeros(B, 5, dtype=torch.bool, device=feats.device)
    if schedule == "team":
        masked[:] = True
    elif schedule in ("individual", "mixed"):
        who = torch.randint(0, 5, (B,), generator=generator, device=feats.device)
        masked[torch.arange(B), who] = True
        if schedule == "mixed":
            team = torch.rand(B, generator=generator, device=feats.device) < 0.5
            masked[team] = True
    else:
        raise ValueError(schedule)
    f = feats.clone()
    dm = masked[:, :, None, None]  # (B,5,1,1)
    d = f[:, 6:]
    pad = d[..., 7:8]
    d = torch.where(dm, torch.zeros_like(d), d)
    d[..., 5:6] = torch.where(dm, 1.0 - pad, torch.zeros_like(pad))  # masked flag on real steps
    d[..., 6:7] = torch.where(dm, torch.zeros_like(pad), d[..., 6:7])
    d[..., 7:8] = pad
    f[:, 6:] = d
    return f, masked
