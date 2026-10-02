"""Rule baselines of the learned ghost on the validation and test games (no model involved).

Exactly the baselines that phase3_train.py reports for a full run (--arrays-dir, all val games,
--eval-test): team = equal-weight mixture over the five attackers' rule positions (σ of the
matchup model) and its RMSE with the per-possession best permutation; individual = Gaussian at the
left-over attacker's rule position, σ fitted (MLE) on all validation games and applied to test.

Output: reports/phase3/<version>_all_rule_baselines.json

Usage:
    uv run python scripts/phase3_rule_baselines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ghost import data as D
from ghost.ghost.dataset import load_arrays

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_train import baseline_nll, baseline_rmse, individual_baseline  # noqa: E402


def main() -> None:
    cfg = D.DataConfig.load(game_set="all")
    root = cfg.processed_dir / "all" / "ghost_arrays"
    prm = json.loads(
        (Path("reports/phase2") / f"{cfg.version}_all" / "matchup_fit.json").read_text()
    )["models"]["hmm"]["params"]
    gamma, sigma = prm["gamma"]["all"], prm["sigma_ft"]["all"]
    out: dict = {"gamma": gamma, "team_sigma_ft": sigma}
    ind_val = None
    for split in ("val", "test"):
        a, k = load_arrays(root, split)
        ind = individual_baseline(a, gamma, None if split == "val" else ind_val["sigma_ft"])
        if split == "val":
            ind_val = ind
        out[split] = {
            "n_games": k["game_id"].n_unique(),
            "n_possessions": len(a["valid"]),
            "team_identity_free_nll_ft": baseline_nll(a, gamma, sigma),
            "team_rmse_ft_best_permutation": baseline_rmse(a, gamma),
            "individual_leftover": ind,
        }
        del a
    out["note"] = "individual sigma fitted (MLE) on all validation games, applied to test"
    p = Path("reports/phase3") / f"{cfg.version}_all_rule_baselines.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
