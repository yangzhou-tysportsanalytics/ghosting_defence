"""The simple rule's region at an actual 80 % coverage (SSAC final review R4, 2026-11-02).

phase3_lookahead_sharpness.py compares the learned ghost's 80 % region (73 sq ft, holding 78.9 %
of real positions) with the rule's nominal 80 % region (346 sq ft, holding 86.0 %). Here the
rule's region is resized to hold exactly 80 % of the same defender-steps (same 2,000 test
possessions, same hidden defender and steps, seed 0): radius = 80th percentile of the distance
from the real defender to his rule position (left-over attacker), area = pi r^2. No model is run.

Output: reports/phase3/<version>_all_rule_area_actual_coverage.json

Usage:
    uv run python scripts/analysis_rule_area_actual_coverage.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_lookahead_sharpness as S  # noqa: E402
from ghost import data as D  # noqa: E402
from ghost.ghost.dataset import apply_mask, load_arrays  # noqa: E402


def main(n_poss: int = 2000, seed: int = 0, run: str = "gpu_d128_lr1e-3_final") -> None:
    cfg = D.DataConfig.load(game_set="all")
    ev = json.loads(Path(f"reports/phase3/{cfg.version}_all_{run}_eval.json").read_text())
    gamma = ev["baseline_attacker_mixture"]["gamma"]
    sigma = ev["baseline_individual_leftover"]["sigma_ft"]
    A, _ = load_arrays(cfg.processed_dir / "all" / "ghost_arrays", "test")
    tu = S.tu
    rng = np.random.default_rng(seed)  # the sample and masks of S.sharpness
    idx = np.sort(rng.choice(len(A["valid"]), size=n_poss, replace=False))
    g = torch.Generator(device="cpu").manual_seed(seed)
    dists = []
    for a in range(0, n_poss, 16):
        sl = idx[a : a + 16]
        sub = {k: A[k][sl] for k in ("feats", "ctx", "target", "valid")}
        _, masked = apply_mask(torch.from_numpy(np.ascontiguousarray(sub["feats"])), "individual", g)
        mu_rule = tu._rule_positions(sub, gamma)
        for b in range(len(sl)):
            d = int(torch.nonzero(masked[b])[0, 0])
            steps = np.flatnonzero(sub["valid"][b])[::5]
            if steps.size == 0:
                continue
            zr, mr = sub["target"][b][:, steps], mu_rule[b][:, steps]
            dist = np.sqrt(((zr[:, None] - mr[None]) ** 2).sum(-1))
            per = dist[np.arange(5)[None, :], tu.PERMS]
            best = (per.sum(1) - per[:, d]).argmin(0)
            dists.append(dist[d, tu.PERMS[best, d], np.arange(steps.size)])
    e = np.concatenate(dists)
    r2 = 2 * math.log(1 / (1 - S.LEVEL))
    r80 = float(np.quantile(e, S.LEVEL))
    rep = {"level": S.LEVEL, "n_possessions": n_poss, "n_defender_steps": int(e.size),
           "rule_sigma_ft": sigma,
           "rule_nominal_area_sqft": math.pi * sigma**2 * r2,
           "rule_nominal_coverage": float((e**2 <= r2 * sigma**2).mean()),
           "rule_radius_at_actual_level_ft": r80,
           "rule_area_at_actual_level_sqft": math.pi * r80**2}  # fmt: skip
    out = Path(f"reports/phase3/{cfg.version}_all_rule_area_actual_coverage.json")
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
