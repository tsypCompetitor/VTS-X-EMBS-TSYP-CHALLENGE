"""
Temporal envelope functions for CABINGUARD-ADI.

g_k(t) = 0                              if t < t_p
          (t-t_p)/(t_a-t_p)             if t_p <= t < t_a
          1                              if t >= t_a
"""
from __future__ import annotations
import numpy as np


def ramp_envelope(t: np.ndarray, t_p: float, t_a: float) -> np.ndarray:
    assert t_p < t_a, f"t_p ({t_p}) must be < t_a ({t_a})"
    g = np.where(t < t_p, 0.0,
            np.where(t < t_a, (t - t_p) / (t_a - t_p), 1.0))
    return g.astype(np.float64)


def oscillatory_envelope(t, t_p, t_a, amplitude, frequency_hz, phase_rad=0.0) -> np.ndarray:
    """A * g_k(t) * sin(2*pi*f*t + phi). Frequencies: seizure 3-6 Hz, cog 4-8 Hz."""
    return amplitude * ramp_envelope(t, t_p, t_a) * np.sin(2*np.pi*frequency_hz*t + phase_rad)


def degradation_mask(n: int, rng: np.random.Generator,
                     fault_prob: float = 0.1, min_dur: int = 5, max_dur: int = 50) -> np.ndarray:
    """Binary mask: 1=available, 0=missing/failed."""
    mask = np.ones(n, dtype=np.float64)
    t = 0
    while t < n:
        if rng.random() < fault_prob:
            dur = int(rng.integers(min_dur, max_dur + 1))
            mask[t:min(t + dur, n)] = 0.0
            t += dur
        else:
            t += 1
    return mask

