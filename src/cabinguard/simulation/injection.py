"""Feature-level injection for CABINGUARD-ADI synthetic scenarios."""
from __future__ import annotations
import numpy as np
from .envelopes import ramp_envelope, oscillatory_envelope, degradation_mask


def additive_injection(x: np.ndarray, t: np.ndarray, alpha: float, t_p: float, t_a: float) -> np.ndarray:
    """x_syn(t) = x_b(t) + alpha * g_k(t)"""
    return x + alpha * ramp_envelope(t, t_p, t_a)


def target_injection(x: np.ndarray, t: np.ndarray, x_target: float, t_p: float, t_a: float) -> np.ndarray:
    """x_syn(t) = (1-g)*x_b(t) + g*x_target"""
    g = ramp_envelope(t, t_p, t_a)
    return (1.0 - g) * x + g * x_target


def oscillatory_injection(x: np.ndarray, t: np.ndarray, amplitude: float, frequency_hz: float,
                           t_p: float, t_a: float, phase_rad: float = 0.0,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """x_syn(t) = x_b(t) + A*g_k(t)*sin(2*pi*f*t+phi)"""
    if rng is not None and phase_rad == 0.0:
        phase_rad = float(rng.uniform(0, 2 * np.pi))
    return x + oscillatory_envelope(t, t_p, t_a, amplitude, frequency_hz, phase_rad)


def sensor_degradation_injection(x: np.ndarray, rng: np.random.Generator,
                                  fault_prob: float = 0.1, noise_std: float = 0.1):
    """x_deg(t) = m(t)*x(t) + (1-m(t))*noise(t)"""
    mask = degradation_mask(len(x), rng, fault_prob=fault_prob)
    noise = rng.normal(0, noise_std, size=len(x))
    return mask * x + (1.0 - mask) * noise, mask

