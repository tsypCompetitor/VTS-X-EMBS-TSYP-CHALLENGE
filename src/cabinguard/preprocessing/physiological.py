"""Physiological signal processing — heart rate, respiration, IBI."""
from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)


def estimate_heart_rate_from_peaks(signal, fs=100.0, min_hr=30.0, max_hr=200.0) -> float:
    from scipy.signal import find_peaks
    try:
        peaks, _ = find_peaks(signal, distance=int(fs * 60.0 / max_hr))
        if len(peaks) < 2: return np.nan
        ibi = np.diff(peaks) / fs
        valid = ibi[(ibi >= 60/max_hr) & (ibi <= 60/min_hr)]
        return float(60.0 / np.mean(valid)) if len(valid) > 0 else np.nan
    except Exception:
        return np.nan


def estimate_respiratory_rate(signal, fs=100.0, min_rr=4.0, max_rr=40.0) -> float:
    from scipy.signal import butter, sosfiltfilt, find_peaks
    try:
        nyq = fs / 2.0
        sos = butter(2, [(min_rr/60)/nyq, (max_rr/60)/nyq], btype="band", output="sos")
        rr_sig = sosfiltfilt(sos, signal)
        peaks, _ = find_peaks(rr_sig, distance=int(fs * 60/max_rr))
        if len(peaks) < 2: return np.nan
        ibi = np.diff(peaks) / fs
        valid = ibi[(ibi >= 60/max_rr) & (ibi <= 60/min_rr)]
        return float(60.0 / np.mean(valid)) if len(valid) > 0 else np.nan
    except Exception:
        return np.nan


def compute_ibi_variability(signal, fs=100.0) -> float:
    from scipy.signal import find_peaks
    try:
        peaks, _ = find_peaks(signal, distance=int(fs * 0.3))
        if len(peaks) < 3: return np.nan
        ibi = np.diff(peaks) / fs
        return float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
    except Exception:
        return np.nan

