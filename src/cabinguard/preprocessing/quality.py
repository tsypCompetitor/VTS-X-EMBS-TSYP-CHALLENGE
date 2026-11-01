"""Signal quality assessment."""
from __future__ import annotations
import numpy as np


def assess_signal_quality(signal: np.ndarray, fs: float = 100.0) -> dict:
    nan_frac = float(np.mean(np.isnan(signal)))
    valid = signal[~np.isnan(signal)]
    if len(valid) == 0:
        return {"quality_score": 0.0, "is_clipped": False, "is_flatline": True,
                "nan_fraction": 1.0, "flags": ["all_nan"]}
    flags = []
    if nan_frac > 0: flags.append(f"nan_fraction={nan_frac:.3f}")
    max_val = np.max(np.abs(valid))
    is_clipped = bool(max_val > 0 and np.mean(np.abs(valid) >= 0.95 * max_val) > 0.01)
    if is_clipped: flags.append("clipped")
    is_flatline = bool(len(valid) >= 50 and np.std(valid[-50:]) < 1e-6)
    if is_flatline: flags.append("flatline")
    q = (1.0 - nan_frac) * (0.5 if is_clipped else 1.0) * (0.2 if is_flatline else 1.0)
    return {"quality_score": float(np.clip(q, 0.0, 1.0)), "is_clipped": is_clipped,
            "is_flatline": is_flatline, "nan_fraction": nan_frac, "flags": flags}

