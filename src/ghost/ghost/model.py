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

Anchored head (``anchor="rule"``): every component mean is a learned convex combination of the
five attackers' rule positions g_o O_k + g_b B + g_h H (the fitted HMM emission means) plus a
learned offset in feet. The combination weights come from query-key attention between the
defender token and the attacker tokens (so the head is equivariant to the attackers' order), plus
two learned priors: in team mode a bias towards "slot s -> attacker s" (one ghost per attacker);
when some defenders are visible, a bias towards attackers far from every visible defender
(u_k = distance from attacker k's rule position to the nearest visible defender: a soft version
of "the masked defender takes the attacker the others leave"). The offset and the rest of the
head start at zero, so an untrained model is the rule ghost and training learns corrections.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
from torch import nn

from ghost.court import HOOP_LEFT

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
    anchor: str = "none"  # none | rule
    gamma: tuple[float, float, float] = (0.62, 0.11, 0.27)  # rule weights (man, ball, hoop)
    anchor_id_bias: float = 5.0  # initial bias of slot s towards attacker s (team mode)
    anchor_cover_bias: float = (
        1.0  # initial score per ft of distance to the nearest visible defender
    )
    offset_scale_ft: float = 10.0
    # Phase 3 task 4: "league" (no identity), "lineup" (the five defenders, no team) or
    # "scheme" (defending team). Shrinkage: small embeddings and identity dropout; index 0 =
    # unknown (also what the model sees when no ids are passed). Vocab sizes include index 0.
    condition: str = "league"  # league | lineup | scheme
    n_players: int = 1
    n_teams: int = 1
    id_dim: int = 8
    id_dropout: float = 0.3


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
        if cfg.anchor == "rule":
            # mu_x, mu_y are offsets from the anchor; all-zero start = the rule ghost
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.q = nn.Linear(d, cfg.n_comp * d)
            self.k = nn.Linear(d, d)
            self.id_bias = nn.Parameter(torch.tensor(float(cfg.anchor_id_bias)))
            self.cover_bias = nn.Parameter(torch.tensor(float(cfg.anchor_cover_bias)))
            self.register_buffer("hoop", torch.tensor(HOOP_LEFT, dtype=torch.float32), False)
        elif cfg.anchor != "none":
            raise ValueError(cfg.anchor)
        causal = torch.triu(torch.ones(n_steps, n_steps, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", causal, persistent=False)
        roles = torch.tensor([0] + [1] * 5 + [2] * 5)
        self.register_buffer("roles", roles, persistent=False)
        if cfg.condition == "lineup":
            self.id_emb = nn.Embedding(cfg.n_players, cfg.id_dim)
        elif cfg.condition == "scheme":
            self.id_emb = nn.Embedding(cfg.n_teams, cfg.id_dim)
        elif cfg.condition != "league":
            raise ValueError(cfg.condition)
        if cfg.condition != "league":
            nn.init.normal_(self.id_emb.weight, std=0.01)
            self.id_proj = nn.Linear(cfg.id_dim, d, bias=False)

    def identity(self, ids: dict | None, B: int, device) -> torch.Tensor | None:
        """(B, d_model) condition vector, or None for the league ghost. ids: "players" (B, 5) or
        "team" (B,) vocabulary indices; missing ids = unknown (0); dropout in training."""
        c = self.cfg.condition
        if c == "league":
            return None
        key = "players" if c == "lineup" else "team"
        shape = (B, 5) if c == "lineup" else (B,)
        idx = ids[key] if ids is not None and key in ids else None
        if idx is None:
            idx = torch.zeros(shape, dtype=torch.long, device=device)
        if self.training and self.cfg.id_dropout > 0:
            keep = torch.rand(shape, device=device) >= self.cfg.id_dropout
            idx = torch.where(keep, idx, torch.zeros_like(idx))
        e = self.id_emb(idx)
        if c == "lineup":
            e = e.mean(1)  # who is on the floor, not which slot is whom
        return self.id_proj(e)

    def forward(
        self, feats: torch.Tensor, ctx: torch.Tensor, ids: dict | None = None
    ) -> dict[str, torch.Tensor]:
        """feats (B, 11, T, N_FEAT) with defenders already masked; ctx (B, T, N_CTX); ids:
        identity indices for the lineup / scheme ghosts (ignored by the league ghost).
        Returns mixture parameters for the 5 defender tokens: logits (B,5,T,M),
        mu (B,5,T,M,2) in feet, sigma (B,5,T,M) in feet."""
        B, A, T, _ = feats.shape
        x = self.inp(feats) + self.role(self.roles)[None, :, None, :]
        x = x + self.time.weight[:T][None, None]
        x = x + self.ctx(ctx)[:, None]
        cond = self.identity(ids, B, feats.device)
        if cond is not None:
            x = x + cond[:, None, None, :]
        slot = torch.zeros(A, x.shape[-1], device=x.device)
        slot[6:] = self.def_slot.weight
        x = x + slot[None, :, None, :]
        t_mask = self.causal[:T, :T] if self.cfg.mode == "online" else None
        for blk in self.blocks:
            x = blk(x, t_mask)
        h = self.norm(x)
        out = self.head(h[:, 6:])  # (B, 5, T, 4M)
        M = self.cfg.n_comp
        out = out.reshape(B, 5, T, M, 4)
        logits = out[..., 0]
        sigma = self.cfg.min_sigma_ft + nn.functional.softplus(out[..., 3]) * 5.0
        if self.cfg.anchor == "none":
            mu = torch.stack([(out[..., 1] + 0.5) * X_SCALE, (out[..., 2] + 0.5) * Y_SCALE], -1)
            return {"logits": logits, "mu": mu, "sigma": sigma}
        anchors = self.rule_anchors(feats)  # (B, 5 attackers, T, 2) ft
        d = h.shape[-1]
        q = self.q(h[:, 6:]).reshape(B, 5, T, M, d)
        k = self.k(h[:, 1:6])  # (B, 5 attackers, T, d)
        score = torch.einsum("bstmc,bktc->bstmk", q, k) / math.sqrt(d)
        u, team = self.uncovered(feats, anchors)  # (B, T, 5 attackers) ft, (B, T) bool
        eye = torch.eye(5, device=feats.device)[None, :, None, None, :]  # slot s, attacker k
        prior = self.id_bias * eye * team[:, None, :, None, None].float()
        prior = prior + self.cover_bias * u[:, None, :, None, :]
        w = torch.softmax(score.float() + prior, dim=-1)  # (B, 5, T, M, 5)
        # positions in float32 even under autocast (bfloat16 resolves ~0.2 ft at 47 ft)
        with torch.autocast(device_type=feats.device.type, enabled=False):
            base = torch.einsum("bstmk,bktx->bstmx", w.float(), anchors.float())
            mu = base + out[..., 1:3].float() * self.cfg.offset_scale_ft
        return {"logits": logits, "mu": mu, "sigma": sigma, "anchor_w": w}

    def rule_anchors(self, feats: torch.Tensor) -> torch.Tensor:
        """Rule positions g_o O_k + g_b B + g_h H (feet) of the five attackers, (B, 5, T, 2).
        Offence and ball are never masked; at step t they use only step t (causal)."""
        scale = torch.tensor([X_SCALE, Y_SCALE], device=feats.device)
        off = (feats[:, 1:6, :, :2] + 0.5) * scale
        ball = (feats[:, 0, :, :2] + 0.5) * scale
        g_o, g_b, g_h = self.cfg.gamma
        return g_o * off + g_b * ball[:, None] + g_h * self.hoop

    @staticmethod
    def uncovered(feats: torch.Tensor, anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """u (B, T, 5): distance (ft, capped at 30) from each attacker's rule position to the
        nearest visible defender at the same step, 0 where no defender is visible; and the
        team-mode flag (B, T): no defender visible."""
        scale = torch.tensor([X_SCALE, Y_SCALE], device=feats.device)
        dxy = (feats[:, 6:, :, :2] + 0.5) * scale  # (B, 5 defenders, T, 2)
        vis = feats[:, 6:, :, 6] > 0.5  # (B, 5, T)
        dist = torch.linalg.vector_norm(dxy[:, :, None] - anchors[:, None], dim=-1)  # (B,5d,5k,T)
        dist = torch.where(vis[:, :, None], dist, torch.full_like(dist, 30.0))
        u = dist.amin(1).clamp(max=30.0).transpose(1, 2)  # (B, T, 5k)
        team = ~vis.any(1)  # (B, T)
        return torch.where(team[..., None], torch.zeros_like(u), u), team


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
