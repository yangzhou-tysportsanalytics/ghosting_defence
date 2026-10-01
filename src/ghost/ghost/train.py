"""Training / evaluation loop for the ghost model (CPU or GPU)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ghost.ghost.dataset import apply_mask
from ghost.ghost.model import GhostConfig, GhostModel, ghost_loss, mixture_mean, slot_mixture_nll


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    loss: str = "nll"  # nll | l2
    schedule: str = "mixed"  # team | individual | mixed
    seed: int = 9
    max_batches_per_epoch: int | None = None
    device: str = "cpu"
    checkpoint_path: str | None = None  # saved after every epoch; resumed if it exists
    amp: bool = False  # bfloat16 autocast on CUDA (ignored on CPU)
    lr_cosine: bool = True


def _batches(n: int, bs: int, rng: np.random.Generator, shuffle: bool):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for a in range(0, n, bs):
        yield idx[a : a + bs]


def ids_of(b: dict) -> dict | None:
    """Identity indices of a batch for the lineup / scheme ghosts (None if absent)."""
    out = {k: b[a] for k, a in (("players", "player_idx"), ("team", "team_idx")) if a in b}
    return out or None


def _to(arrs: dict, idx: np.ndarray, device: str) -> dict:
    return {k: torch.from_numpy(np.ascontiguousarray(v[idx])).to(device) for k, v in arrs.items()}


def evaluate(
    model: GhostModel,
    arrs: dict,
    schedule: str,
    device: str = "cpu",
    bs: int = 32,
    seed: int = 0,
    exclude_imputed: bool = False,
) -> dict:
    """Mean NLL (feet) and mean-position error over masked defender-steps."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    tot_nll = tot_l2 = tot_n = mix_sum = mix_n = 0.0
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for idx in _batches(len(arrs["valid"]), bs, rng, shuffle=False):
            b = _to(arrs, idx, device)
            if exclude_imputed and "sc_imputed" in b:
                b["valid"] = b["valid"] & ~b["sc_imputed"]
            f, masked = apply_mask(b["feats"], schedule, g)
            out = model(f, b["ctx"], ids_of(b))
            nll, info = ghost_loss(out, b["target"], b["valid"], masked, "nll")
            l2, _ = ghost_loss(out, b["target"], b["valid"], masked, "l2")
            tot_nll += float(nll) * info["n_obs"]
            tot_l2 += float(l2) * info["n_obs"]
            tot_n += info["n_obs"]
            if schedule == "team":
                a, n = slot_mixture_nll(out, b["target"], b["valid"])
                mix_sum += a
                mix_n += n
    res = {
        "nll_ft": tot_nll / max(tot_n, 1),
        "rmse_ft": float(np.sqrt(tot_l2 / max(tot_n, 1))),
        "n_obs": tot_n,
        "schedule": schedule,
    }
    if schedule == "team":
        # nll_ft uses the best permutation per possession (hindsight); this one does not and is
        # the one comparable with the attacker-mixture baseline
        res["slot_mixture_nll_ft"] = mix_sum / max(mix_n, 1)
    return res


def fit(
    train: dict, val: dict | None, mcfg: GhostConfig, tcfg: TrainConfig, log=print
) -> tuple[GhostModel, list[dict]]:
    torch.manual_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)
    g = torch.Generator(device="cpu").manual_seed(tcfg.seed)
    T = train["valid"].shape[1]
    model = GhostModel(mcfg, n_steps=T).to(tcfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(tcfg.epochs, 1))
        if tcfg.lr_cosine
        else None
    )
    history: list[dict] = []
    start = 0
    ckpt = Path(tcfg.checkpoint_path) if tcfg.checkpoint_path else None
    if ckpt is not None and ckpt.exists():
        state = torch.load(ckpt, map_location=tcfg.device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if sched is not None and state.get("sched"):
            sched.load_state_dict(state["sched"])
        history = state["history"]
        start = state["epoch"] + 1
        rng = state["rng"]
        g.set_state(state["g"])
        log({"resumed_from_epoch": state["epoch"]})
    use_amp = tcfg.amp and tcfg.device.startswith("cuda")
    for ep in range(start, tcfg.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for bi, idx in enumerate(_batches(len(train["valid"]), tcfg.batch_size, rng, True)):
            if tcfg.max_batches_per_epoch and bi >= tcfg.max_batches_per_epoch:
                break
            b = _to(train, idx, tcfg.device)
            f, masked = apply_mask(b["feats"], tcfg.schedule, g)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(f, b["ctx"], ids_of(b))
            out = {k: v.float() for k, v in out.items()}
            loss, _ = ghost_loss(out, b["target"], b["valid"], masked, tcfg.loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        rec = {
            "epoch": ep,
            "train_loss": float(np.mean(losses)),
            "seconds": round(time.time() - t0, 1),
        }
        if val is not None:
            for sch in ("team", "individual"):
                rec[f"val_{sch}"] = evaluate(model, val, sch, tcfg.device)
        if sched is not None:
            sched.step()
        history.append(rec)
        log(rec)
        if ckpt is not None:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None,
                    "history": history,
                    "epoch": ep,
                    "rng": rng,
                    "g": g.get_state(),
                    "mcfg": asdict(mcfg),
                    "tcfg": asdict(tcfg),
                },
                ckpt,
            )
    return model, history


def predict_mean(model: GhostModel, arrs: dict, schedule: str, device: str = "cpu") -> np.ndarray:
    """Component-weighted mean positions (N, 5, T, 2) for visualisation."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(0)
    outs = []
    with torch.no_grad():
        for idx in _batches(len(arrs["valid"]), 32, np.random.default_rng(0), False):
            b = _to(arrs, idx, device)
            f, _ = apply_mask(b["feats"], schedule, g)
            outs.append(mixture_mean(model(f, b["ctx"], ids_of(b))).cpu().numpy())
    return np.concatenate(outs)


def config_dict(mcfg: GhostConfig, tcfg: TrainConfig) -> dict:
    return {"model": asdict(mcfg), "train": asdict(tcfg)}


def calibration_eval(
    model: GhostModel,
    arrs: dict,
    device: str = "cpu",
    n_poss: int = 256,
    n_samples: int = 256,
    seed: int = 0,
) -> dict:
    """HPD-PIT coverage on the first ``n_poss`` possessions (D-007).
    individual: the masked defender's own mixture; team: equal-weight mixture over the five slots
    (identity-free), evaluated for every real defender."""
    from ghost.ghost.calibration import coverage_curve, hpd_pit, pit_histogram

    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    sub = {k: v[:n_poss] for k, v in arrs.items()}
    b = _to(sub, np.arange(len(sub["valid"])), device)
    res = {}
    with torch.no_grad():
        for sch in ("individual", "team"):
            f, masked = apply_mask(b["feats"], sch, g)
            out = model(f, b["ctx"], ids_of(b))
            if sch == "individual":
                j = masked.float().argmax(-1)  # (B,)
                ix = torch.arange(len(j))
                lg, mu, sg = out["logits"][ix, j], out["mu"][ix, j], out["sigma"][ix, j]
                z = b["target"][ix, j]  # (B,T,2)
                v = b["valid"]
            else:
                B, S, T, M = out["logits"].shape
                lg = out["logits"].permute(0, 2, 1, 3).reshape(B, T, S * M)  # equal slot weights
                lg = lg - torch.logsumexp(out["logits"], -1).permute(0, 2, 1).repeat_interleave(
                    M, -1
                )
                mu = out["mu"].permute(0, 2, 1, 3, 4).reshape(B, T, S * M, 2)
                sg = out["sigma"].permute(0, 2, 1, 3).reshape(B, T, S * M)
                lg, mu, sg = (x[:, None].expand(B, 5, *x.shape[1:]) for x in (lg, mu, sg))
                z = b["target"]  # (B,5,T,2)
                v = b["valid"][:, None].expand(B, 5, T)
            u = hpd_pit(
                lg.cpu().float(), mu.cpu().float(), sg.cpu().float(), z.cpu().float(), n_samples, g
            )
            u = u[v.cpu()]
            res[sch] = {
                "coverage": coverage_curve(u),
                "pit_hist": pit_histogram(u),
                "n": int(u.numel()),
            }
    return res
