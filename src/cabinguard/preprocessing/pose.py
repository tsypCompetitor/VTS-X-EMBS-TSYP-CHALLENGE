"""Pose-based feature computation."""
from __future__ import annotations
import numpy as np


def compute_perclos_proxy(eye_closure: np.ndarray, window_size: int, threshold: float = 0.7) -> np.ndarray:
    """PERCLOS proxy — fraction of window with eye_closure > threshold. nan where insufficient data."""
    n = len(eye_closure)
    out = np.full(n, np.nan)
    for i in range(window_size - 1, n):
        w = eye_closure[i - window_size + 1:i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) >= window_size // 2:
            out[i] = float(np.mean(valid > threshold))
    return out


def compute_gaze_stability(head_pitch: np.ndarray, head_roll: np.ndarray, window_size: int) -> np.ndarray:
    """Gaze stability proxy = combined head motion variance in window."""
    n = len(head_pitch)
    out = np.full(n, np.nan)
    for i in range(window_size - 1, n):
        vp = head_pitch[i - window_size + 1:i + 1]
        vr = head_roll[i - window_size + 1:i + 1]
        vp = vp[~np.isnan(vp)]
        vr = vr[~np.isnan(vr)]
        if len(vp) >= 3 and len(vr) >= 3:
            out[i] = float(np.std(vp) + np.std(vr))
    return out

