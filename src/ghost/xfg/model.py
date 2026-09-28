"""Shot-quality (xFG) model: logistic regression with a shrunken shooter-by-zone skill offset.

Shooter skill (decision D-015): for shooter s in zone z with n attempts and m makes in the TRAIN
games, the empirical-Bayes rate is (m + k_z p_z) / (n + k_z), with league zone rate p_z and prior
strength k_z from a beta-binomial method of moments over shooters. Training rows use leave-one-out
counts (the shot's own outcome is removed) so the offset does not leak the label.
Feature ``skill`` = logit(shrunk rate) - logit(p_z).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ZONES = ["rim", "paint", "mid", "corner3", "above3"]
FEATURE_SETS = {
    "M0_zone": ["zone"],
    "M1_geometry_defense": ["zone", "geom", "defense"],
    "M2_plus_dribble": ["zone", "geom", "defense", "dribble"],
    "M3_plus_shooter": ["zone", "geom", "defense", "dribble", "shooter"],
}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


@dataclass
class ShooterPrior:
    p_zone: dict[str, float]
    k_zone: dict[str, float]
    counts: dict[tuple[int, str], tuple[int, int]] = field(default_factory=dict)  # (n, m)

    @classmethod
    def fit(cls, train: pl.DataFrame, min_n: int = 20) -> ShooterPrior:
        p_zone, k_zone = {}, {}
        g = train.group_by(["shooter_id", "zone"]).agg(
            pl.len().alias("n"), pl.col("made").sum().alias("m")
        )
        for z in ZONES:
            gz = g.filter(pl.col("zone") == z)
            n, m = gz["n"].to_numpy().astype(float), gz["m"].to_numpy().astype(float)
            p = m.sum() / max(n.sum(), 1)
            p_zone[z] = float(p)
            sel = n >= min_n
            if sel.sum() >= 10:
                rates = m[sel] / n[sel]
                var_obs = rates.var()
                var_noise = np.mean(p * (1 - p) / n[sel])
                var_true = max(var_obs - var_noise, 1e-5)
                k_zone[z] = float(np.clip(p * (1 - p) / var_true - 1, 5, 2000))
            else:
                k_zone[z] = 200.0
        counts = {
            (int(r["shooter_id"]), r["zone"]): (int(r["n"]), int(r["m"]))
            for r in g.iter_rows(named=True)
        }
        return cls(p_zone, k_zone, counts)

    def skill(self, df: pl.DataFrame, leave_one_out: bool) -> np.ndarray:
        out = np.zeros(df.height)
        for i, (s, z, made) in enumerate(
            zip(df["shooter_id"], df["zone"], df["made"], strict=True)
        ):
            n, m = self.counts.get((int(s), z), (0, 0))
            if leave_one_out and n > 0:
                n, m = n - 1, m - int(made)
            p, k = self.p_zone[z], self.k_zone[z]
            out[i] = _logit((m + k * p) / (n + k)) - _logit(p)
        return out

    def to_dict(self) -> dict:
        return {"p_zone": self.p_zone, "k_zone": self.k_zone}


def design(
    df: pl.DataFrame, groups: list[str], skill: np.ndarray | None = None
) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    z = df["zone"].to_numpy()
    if "zone" in groups:
        for zz in ZONES[1:]:
            cols.append((z == zz).astype(float))
            names.append(f"zone_{zz}")
    if "geom" in groups:
        d = df["dist_ft"].to_numpy()
        cols += [d, d**2 / 100, df["angle_deg"].to_numpy() / 90]
        names += ["dist", "dist2", "angle"]
    if "defense" in groups:
        l1 = np.log(df["def1_ft"].fill_null(30).to_numpy() + 0.5)
        lf = np.log(df["def_front_ft"].fill_null(30).to_numpy() + 0.5)
        three = df["is_three"].to_numpy().astype(float)
        rim = (z == "rim").astype(float)
        cols += [l1, lf, l1 * three, l1 * rim]
        names += ["log_def1", "log_def_front", "log_def1_x_three", "log_def1_x_rim"]
    if "dribble" in groups:
        nd = df["n_dribbles"].to_numpy()
        unk = (
            np.isnan(nd.astype(float)) if nd.dtype != object else np.array([v is None for v in nd])
        )
        ndf = np.nan_to_num(nd.astype(float), nan=0.0)
        cols += [
            unk.astype(float),
            ((ndf >= 1) & (ndf <= 1) & ~unk).astype(float),
            ((ndf >= 2) & (ndf <= 4) & ~unk).astype(float),
            ((ndf >= 5) & ~unk).astype(float),
            df["catch_and_shoot"].to_numpy().astype(float),
            df["is_transition"].fill_null(False).to_numpy().astype(float),
        ]
        names += ["drib_unknown", "drib_1", "drib_2_4", "drib_5plus", "catch_shoot", "transition"]
    if "shooter" in groups:
        cols.append(skill)
        names.append("shooter_skill")
    return np.column_stack(cols), names


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    bins = np.quantile(p, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(bins, p, side="right") - 1, 0, 9)
    calib = [
        {
            "p_mean": float(p[idx == b].mean()),
            "y_mean": float(y[idx == b].mean()),
            "n": int((idx == b).sum()),
        }
        for b in range(10)
        if (idx == b).any()
    ]
    return {
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "mean_pred": float(p.mean()),
        "mean_obs": float(y.mean()),
        "calibration_deciles": calib,
    }


@dataclass
class XFG:
    groups: list[str]
    clf: LogisticRegression
    names: list[str]
    prior: ShooterPrior | None

    def predict(self, df: pl.DataFrame, leave_one_out: bool = False) -> np.ndarray:
        sk = self.prior.skill(df, leave_one_out) if self.prior is not None else None
        X, _ = design(df, self.groups, sk)
        return self.clf.predict_proba(X)[:, 1]

    def coefficients(self) -> dict:
        return {
            "intercept": float(self.clf.intercept_[0]),
            **{n: float(c) for n, c in zip(self.names, self.clf.coef_[0], strict=True)},
        }


def fit_xfg(train: pl.DataFrame, groups: list[str], C: float = 1.0) -> XFG:
    prior = ShooterPrior.fit(train) if "shooter" in groups else None
    sk = prior.skill(train, leave_one_out=True) if prior is not None else None
    X, names = design(train, groups, sk)
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(X, train["made"].to_numpy().astype(int))
    return XFG(groups, clf, names, prior)


def save(model: XFG, path) -> None:
    d = {
        "groups": model.groups,
        "coefficients": model.coefficients(),
        "prior": model.prior.to_dict() if model.prior else None,
    }
    path.write_text(json.dumps(d, indent=2))
