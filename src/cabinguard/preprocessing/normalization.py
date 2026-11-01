"""
Feature normalization and severity bands for CABINGUARD-ADI.
NORMAL: |z|<2, ELEVATED: 2<=|z|<3, LARGE: 3<=|z|<4.5, EXTREME: |z|>=4.5
"""
from __future__ import annotations
import numpy as np
from enum import IntEnum


class SeverityBand(IntEnum):
    NORMAL = 0
    ELEVATED = 1
    LARGE = 2
    EXTREME = 3


SEVERITY_THRESHOLDS = [2.0, 3.0, 4.5]


def classify_severity(z: float) -> SeverityBand:
    az = abs(z)
    if az < 2.0: return SeverityBand.NORMAL
    if az < 3.0: return SeverityBand.ELEVATED
    if az < 4.5: return SeverityBand.LARGE
    return SeverityBand.EXTREME


def classify_severity_array(z: np.ndarray) -> np.ndarray:
    az = np.abs(z)
    out = np.zeros(az.shape, dtype=np.int32)
    out[az >= 2.0] = 1
    out[az >= 3.0] = 2
    out[az >= 4.5] = 3
    return out


def compute_source_availability(mask: np.ndarray) -> float:
    """A_sources = (1/N_s) * sum_s m_s(t)"""
    return float(np.mean(mask.astype(float))) if len(mask) > 0 else 0.0

