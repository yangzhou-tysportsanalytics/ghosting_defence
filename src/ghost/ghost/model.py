"""Ghost model: conditional density of defender positions given the offence (Phase 3).

Architecture: per-token embedding of (ball, 5 offence, 5 defence) x T steps, then blocks of
(temporal self-attention per agent, agent self-attention per step). Temporal attention is causal
in ``online`` mode (decision D-006: the ghost at t only sees the offence and ball up to t) and
bidirectional in ``offline`` mode (visualisation only).

Masking schedules (D-013):
* ``team``: all five defenders masked at every step (team ghost).
* ``individual``: one defender masked; the other four real defenders are visible (up to t in
  online mode).

Symmetry: masked defender tokens would be identical and the agent attention is permutation
equivariant, so defender tokens carry learned slot embeddings, and the team-mode loss matches the
five predicted slots to the five real defenders by the best of the 120 permutations per
possession (set loss, as in DETR). In individual mode the target is the masked defender itself.

Head: per defender token and step, a mixture of ``n_comp`` isotropic Gaussians in feet (analytic
density, D-007). ``loss="l2"`` trains the component-weighted mean only (deterministic v1).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
from torch import nn

PERMS = torch.tensor(list(itertools.permutations(range(5))), dtype=torch.long)  # (120, 5)
X_SCALE, Y_SCALE, V_SCALE = 47.0, 50.0, 20.0
N_FEAT = 8  # x, y, vx, vy, is_handler, masked, visible, pad
N_CTX = 5  # margin, game clock, shot clock, shot clock missing, period


@dataclass
class GhostConfig:
    d_model: int = 64
    n_heads: int = 4
    n_blocks: int = 2
    n_comp: int = 4
    dropout: float = 0.1
    mode: str = "online"  # online | offline
    min_sigma_ft: float = 0.5


class Block(nn.Module):
    def __init__(self, d: int, h: int, p: float):
        super().__init__()
        self.t_attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.a_attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, t_mask: torch.Tensor | None) -> torch.Tensor:
        B, A, T, D = x.shape
        y = self.n1(x).reshape(B * A, T, D)
        y, _ = self.t_attn(y, y, y, attn_mask=t_mask, need_weights=False)
        x = x + y.reshape(B, A, T, D)
        y = self.n2(x).transpose(1, 2).reshape(B * T, A, D)
        y, _ = self.a_attn(y, y, y, need_weights=False)
        x = x + y.reshape(B, T, A, D).transpose(1, 2)
        return x + self.ff(self.n3(x))


class GhostModel(nn.Module):
    def __init__(self, cfg: GhostConfig, n_steps: int = 121):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.inp = nn.Linear(N_FEAT, d)
        self.ctx = nn.Linear(N_CTX, d)
        self.role = nn.Embedding(3, d)  # ball, offence, defence
        self.def_slot = nn.Embedding(5, d)
        self.time = nn.Embedding(n_steps, d)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_blocks))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.n_comp * 4)  # logit, mu_x, mu_y, log_sigma
        causal = torch.triu(torch.ones(n_steps, n_steps, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", causal, persistent=False)
        roles = torch.tensor([0] + [1] * 5 + [2] * 5)
        self.register_buffer("roles", roles, persistent=False)

    def forward(self, feats: torch.Tensor, ctx: torch.Tensor) -> dict[str, torch.Tensor]:
        """feats (B, 11, T, N_FEAT) with defenders already masked; ctx (B, T, N_CTX).
        Returns mixture parameters for the 5 defender tokens: logits (B,5,T,M),
        mu (B,5,T,M,2) in feet, sigma (B,5,T,M) in feet."""
        B, A, T, _ = feats.shape
        x = self.inp(feats) + self.role(self.roles)[None, :, None, :]
        x = x + self.time.weight[:T][None, None]
        x = x + self.ctx(ctx)[:, None]
        slot = torch.zeros(A, x.shape[-1], device=x.device)
        slot[6:] = self.def_slot.weight
        x = x + slot[None, :, None, :]
        t_mask = self.causal[:T, :T] if self.cfg.mode == "online" else None
        for blk in self.blocks:
            x = blk(x, t_mask)
        out = self.head(self.norm(x[:, 6:]))  # (B, 5, T, 4M)
        M = self.cfg.n_comp
        out = out.reshape(B, 5, T, M, 4)
        logits = out[..., 0]
        mu = torch.stack([(out[..., 1] + 0.5) * X_SCALE, (out[..., 2] + 0.5) * Y_SCALE], dim=-1)
        sigma = self.cfg.min_sigma_ft + nn.functional.softplus(out[..., 3]) * 5.0
        return {"logits": logits, "mu": mu, "sigma": sigma}


# ------------------------------------------------------------------------------------------
# losses
# ------------------------------------------------------------------------------------------


def mixture_nll(out: dict, target: torch.Tensor) -> torch.Tensor:
    """Per (B, slot, target_defender, T) NLL in feet units: every predicted slot against every
    real defender. target (B, 5, T, 2) feet. Returns (B, 5, 5, T)."""
    logits = torch.log_softmax(out["logits"], dim=-1)[:, :, None]  # (B,5,1,T,M)
    mu = out["mu"][:, :, None]  # (B,5,1,T,M,2)
    s = out["sigma"][:, :, None]  # (B,5,1,T,M)
    z = target[:, None, :, :, None, :]  # (B,1,5,T,1,2)
    d2 = ((z - mu) ** 2).sum(-1)
    logn = -0.5 * d2 / s**2 - 2 * torch.log(s) - math.log(2 * math.pi)
    return -torch.logsumexp(logits + logn, dim=-1)


def mixture_mean(out: dict) -> torch.Tensor:
    w = torch.softmax(out["logits"], dim=-1)[..., None]
    return (w * out["mu"]).sum(-2)  # (B,5,T,2)


def set_loss(cost: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cost (B, 5 slots, 5 defenders, T); valid (B, T). Best permutation per possession.
    Returns (loss per possession summed over matched pairs and valid steps, best perm (B,5))."""
    c = (cost * valid[:, None, None, :]).sum(-1)  # (B,5,5)
    perms = PERMS.to(cost.device)
    total = c[:, torch.arange(5, device=cost.device)[None, :], perms].sum(-1)  # (B,120)
    best = total.argmin(-1)
    return total.gather(1, best[:, None])[:, 0], perms[best]


def ghost_loss(
    out: dict, target: torch.Tensor, valid: torch.Tensor, masked: torch.Tensor, kind: str = "nll"
) -> tuple[torch.Tensor, dict]:
    """masked (B, 5) bool: which defenders were hidden. In team mode all five are masked and the
    set loss is used; otherwise only masked defenders contribute, matched to their own slot.
    Returns the equal-weight average of the per-observation team and individual losses."""
    if kind == "nll":
        cost = mixture_nll(out, target)  # (B,5,5,T)
    else:
        mean = mixture_mean(out)  # (B,5,T,2)
        cost = ((mean[:, :, None] - target[:, None]) ** 2).sum(-1)  # (B,5,5,T)
    n_obs = valid.sum(-1)  # (B,)
    team = masked.all(-1)
    parts, n_total = [], torch.zeros((), device=target.device)
    if team.any():
        l_team, _ = set_loss(cost[team], valid[team])
        n_t = (n_obs[team] * 5).sum()
        parts.append(l_team.sum() / n_t.clamp(min=1))
        n_total = n_total + n_t
    ind = ~team
    if ind.any():
        diag = torch.diagonal(cost[ind], dim1=1, dim2=2).permute(0, 2, 1)  # (b,5,T)
        w = masked[ind][:, :, None] * valid[ind][:, None, :]
        parts.append((diag * w).sum() / w.sum().clamp(min=1))
        n_total = n_total + w.sum()
    # team and individual terms are averaged with equal weight (a team possession has five
    # targets, an individual one has one; summing would drown the individual ghost)
    return torch.stack(parts).mean(), {"n_obs": float(n_total)}


def slot_mixture_nll(out: dict, target: torch.Tensor, valid: torch.Tensor) -> tuple[float, float]:
    """Identity-free team-ghost density of each real defender: equal-weight mixture over the five
    predicted slots, p(z_d) = (1/5) sum_s p_s(z_d). Unlike the set loss it does not pick the best
    permutation with hindsight, so it is comparable with the attacker-mixture baseline and is the
    definition to use for team-ghost deviations. Returns (sum of NLL over valid defender-steps, n)."""
    cost = mixture_nll(out, target)  # (B,5 slots,5 defenders,T)
    nll = -(torch.logsumexp(-cost, dim=1) - math.log(5.0))  # (B,5d,T)
    v = valid[:, None, :].expand_as(nll)
    return float((nll * v).sum()), float(v.sum())
