"""
Signal filtering for CABINGUARD-ADI.

NLMS adaptive filter:
    s_hat[n] = x[n] - sum_{k=0}^{M-1} w_k[n] * d[n-k]
    w[n+1] = w[n] + (mu / (eps + ||d[n]||^2)) * s_hat[n] * d[n]

Parameters: M=32, fs=100 Hz, mu=0.05, epsilon=1e-6
Butterworth: 4th-order, 0.3-15 Hz bandpass
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt
import logging

logger = logging.getLogger(__name__)

NLMS_M = 32
NLMS_MU = 0.05
NLMS_EPSILON = 1e-6
DEFAULT_FS = 100.0
BP_LOW = 0.3
BP_HIGH = 15.0
BP_ORDER = 4


def nlms_adaptive_filter(
    x: np.ndarray,
    d: np.ndarray,
    M: int = NLMS_M,
    mu: float = NLMS_MU,
    epsilon: float = NLMS_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """NLMS adaptive filter for BCG/SCG motion artifact removal."""
    N = len(x)
    assert len(d) == N, "x and d must have same length"
    w = np.zeros(M, dtype=np.float64)
    s_hat = np.zeros(N, dtype=np.float64)
    d_buf = np.zeros(M, dtype=np.float64)
    for n in range(N):
        d_buf = np.roll(d_buf, 1)
        d_buf[0] = d[n]
        s_hat[n] = x[n] - np.dot(w, d_buf)
        norm = epsilon + np.dot(d_buf, d_buf)
        w = w + (mu / norm) * s_hat[n] * d_buf
    return s_hat, w


def butterworth_bandpass(
    signal: np.ndarray,
    fs: float = DEFAULT_FS,
    lowcut: float = BP_LOW,
    highcut: float = BP_HIGH,
    order: int = BP_ORDER,
) -> np.ndarray:
    """4th-order Butterworth bandpass, zero-phase."""
    nyq = fs / 2.0
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR in dB. Returns -999.0 if not computable."""
    noise = noisy - clean
    p_signal = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(p_signal / p_noise))

