"""
Personalized calibration for CABINGUARD-ADI.

c_{i,s} = median(x_{i,s,t})
sigma_{i,s} = 1.4826 * median(|x - c_{i,s}|)   [MAD-based robust std]
c_tilde_{i,s} = w_i * c_{i,s} + (1-w_i) * mu_pop
w_i = N_good / (N_good + kappa),  kappa=300, cal_duration=60s
"""
from __future__ import annotations
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)
KAPPA = 300
CAL_DURATION = 60.0


def compute_individual_baseline(x: np.ndarray) -> tuple[float, float]:
    """Returns (median, 1.4826*MAD)."""
    c_i = float(np.median(x))
    sigma_i = 1.4826 * float(np.median(np.abs(x - c_i)))
    return c_i, sigma_i


def compute_shrinkage_weight(N_good: int, kappa: float = KAPPA) -> float:
    return N_good / (N_good + kappa)


def compute_shrunk_baseline(c_i, sigma_i, mu_pop, sigma_pop, N_good, kappa=KAPPA):
    w = compute_shrinkage_weight(N_good, kappa)
    return float(w * c_i + (1 - w) * mu_pop), float(w * sigma_i + (1 - w) * sigma_pop)


class FeatureCalibrator:
    """Per-subject, per-feature calibration with shrinkage."""

    def __init__(self, feature_names: list[str], kappa=KAPPA, fs=100.0):
        self.feature_names = feature_names
        self.kappa = kappa
        self.fs = fs
        self.c_tilde = {f: 0.0 for f in feature_names}
        self.sigma_tilde = {f: 1.0 for f in feature_names}
        self.N_good = {f: 0 for f in feature_names}
        self.is_calibrated = False

    def fit(self, X: np.ndarray, population_centers=None, population_scales=None,
            quality_mask=None) -> "FeatureCalibrator":
        if quality_mask is None:
            quality_mask = np.ones(len(X), dtype=bool)
        pop_c = population_centers or {}
        pop_s = population_scales or {}
        for i, feat in enumerate(self.feature_names):
            x_feat = X[quality_mask, i]
            N_good = int(np.sum(quality_mask & ~np.isnan(X[:, i])))
            self.N_good[feat] = N_good
            valid = x_feat[~np.isnan(x_feat)]
            if len(valid) < 5:
                c_i, sigma_i = pop_c.get(feat, 0.0), pop_s.get(feat, 1.0)
            else:
                c_i, sigma_i = compute_individual_baseline(valid)
            self.c_tilde[feat], self.sigma_tilde[feat] = compute_shrunk_baseline(
                c_i, sigma_i, pop_c.get(feat, 0.0), pop_s.get(feat, 1.0), N_good, self.kappa)
        self.is_calibrated = True
        return self

    def transform(self, x_vector: np.ndarray, O_s: float = 1.0) -> np.ndarray:
        """z_{i,s,t} = O_s * (x - c_tilde) / sigma_tilde"""
        z = np.zeros(len(self.feature_names), dtype=np.float64)
        for i, feat in enumerate(self.feature_names):
            denom = self.sigma_tilde[feat]
            z[i] = 0.0 if denom < 1e-10 else O_s * (x_vector[i] - self.c_tilde[feat]) / denom
        return z

