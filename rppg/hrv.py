"""
Heart Rate Variability (HRV) Estimation
=========================================
Computes time-domain HRV metrics from a filtered rPPG pulse signal.

Pipeline
--------
    Filtered rPPG signal
        ↓  detect_peaks()          — scipy peak detection with prominence
        ↓  compute_rr_intervals()  — inter-peak distances → RR (ms)
        ↓  compute_sdnn()          — SD of all NN intervals (global HRV)
        ↓  compute_rmssd()         — RMSSD (parasympathetic HRV)
        ↓  compute_hrv_metrics()   — aggregated dict for export/logging

Clinical note
-------------
Camera-derived rPPG carries an inherent timing uncertainty of ±(1/FPS)
seconds per peak (e.g., ±33 ms at 30 FPS).  SDNN and RMSSD values are
therefore approximations of ECG-derived HRV.  This implementation is
suitable for research and relative comparison across conditions, not for
clinical diagnosis.

Reference:
    Task Force of the European Society of Cardiology and the North
    American Society of Pacing and Electrophysiology. (1996).
    Heart rate variability: Standards of measurement, physiological
    interpretation, and clinical use.
    Circulation, 93(5), 1043–1065.
"""

import numpy as np
from scipy.signal import find_peaks, welch
from typing import Optional, Dict, Any, Tuple


def detect_peaks(
    signal: np.ndarray,
    fs: float,
    min_distance_sec: float = 0.4,
    min_prominence: float = 0.2,
) -> np.ndarray:
    """
    Detect systolic peaks in a normalised rPPG signal.

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        Filtered, zero-mean rPPG signal.
    fs : float
        Sampling frequency in Hz.
    min_distance_sec : float
        Minimum time between consecutive peaks (seconds).
        Default 0.4 s corresponds to a maximum rate of 150 BPM.
    min_prominence : float
        Minimum peak prominence as a fraction of the signal's peak-to-peak
        range.  Suppresses noise-induced micro-peaks.

    Returns
    -------
    peaks : np.ndarray of int
        Sample indices of detected peaks, sorted ascending.
    """
    min_dist_samples = max(1, int(min_distance_sec * fs))

    # Absolute prominence threshold scaled to the signal's dynamic range
    signal_range = float(signal.max() - signal.min())
    abs_prominence = max(1e-6, min_prominence * signal_range)

    peaks, _ = find_peaks(
        signal,
        distance=min_dist_samples,
        prominence=abs_prominence,
    )
    return peaks


def compute_rr_intervals(
    peaks: np.ndarray, fs: float
) -> Optional[np.ndarray]:
    """
    Convert peak sample indices to RR intervals in milliseconds.

    Parameters
    ----------
    peaks : np.ndarray
        Sample indices of detected beats.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    rr_ms : np.ndarray, shape (N-1,) or None
        Successive inter-beat intervals in milliseconds.
        Returns None if fewer than 2 peaks are present.
    """
    if len(peaks) < 2:
        return None
    rr_samples = np.diff(peaks).astype(np.float64)
    rr_ms = (rr_samples / fs) * 1000.0
    return rr_ms


def compute_sdnn(rr_ms: Optional[np.ndarray]) -> float:
    """
    Compute SDNN — standard deviation of all NN (RR) intervals.

    SDNN is the primary time-domain HRV metric reflecting total autonomic
    variability over the recording window.  Higher SDNN indicates greater
    heart rate flexibility and generally better autonomic regulation.

    Uses ddof=1 (sample standard deviation) as required by the Task Force
    guidelines for finite-length recordings.

    Parameters
    ----------
    rr_ms : np.ndarray or None
        RR intervals in milliseconds.

    Returns
    -------
    sdnn : float
        SDNN in milliseconds.  Returns 0.0 if insufficient data.
    """
    if rr_ms is None or len(rr_ms) < 2:
        return 0.0
    return float(np.std(rr_ms, ddof=1))


def compute_rmssd(rr_ms: Optional[np.ndarray]) -> float:
    """
    Compute RMSSD — root mean square of successive RR differences.

    RMSSD quantifies short-term, beat-to-beat HRV and is a sensitive
    marker of cardiac vagal (parasympathetic) modulation.  It is the
    recommended short-window HRV metric when recordings are < 5 minutes.

    Parameters
    ----------
    rr_ms : np.ndarray or None
        RR intervals in milliseconds.

    Returns
    -------
    rmssd : float
        RMSSD in milliseconds.  Returns 0.0 if insufficient data.
    """
    if rr_ms is None or len(rr_ms) < 2:
        return 0.0
    successive_diffs = np.diff(rr_ms)
    return float(np.sqrt(np.mean(successive_diffs ** 2)))


def compute_mean_rr(rr_ms: Optional[np.ndarray]) -> float:
    """
    Compute the mean RR (NN) interval in milliseconds.

    Parameters
    ----------
    rr_ms : np.ndarray or None

    Returns
    -------
    float
        Mean RR in ms, or 0.0 if insufficient data.
    """
    if rr_ms is None or len(rr_ms) == 0:
        return 0.0
    return float(np.mean(rr_ms))


def compute_pnn50(rr_ms: Optional[np.ndarray]) -> float:
    """
    Compute pNN50 — percentage of successive RR differences > 50 ms.

    pNN50 is a robust time-domain marker of cardiac parasympathetic
    activity and is highly correlated with HF spectral power.

    Parameters
    ----------
    rr_ms : np.ndarray or None
        RR intervals in milliseconds.

    Returns
    -------
    float
        pNN50 in percent [0, 100].  Returns 0.0 if insufficient data.
    """
    if rr_ms is None or len(rr_ms) < 2:
        return 0.0
    diffs = np.abs(np.diff(rr_ms))
    return float(100.0 * np.sum(diffs > 50.0) / len(diffs))


def compute_frequency_domain_hrv(
    rr_ms: Optional[np.ndarray],
    fs_rr: float = 4.0,
) -> Dict[str, float]:
    """
    Estimate LF and HF power from the RR-interval tachogram using
    Welch's power spectral density method.

    The tachogram (uniformly re-sampled RR series) is required because
    Welch's method assumes evenly spaced samples.  The RR series is
    linearly interpolated onto a uniform grid at ``fs_rr`` Hz (4 Hz is
    the standard choice per Task Force guidelines and gives adequate
    resolution for both LF and HF bands from short windows).

    Frequency bands (Task Force 1996):
        VLF  0.000 – 0.040 Hz  (not computed — too short a window)
        LF   0.040 – 0.150 Hz  (sympathetic + parasympathetic)
        HF   0.150 – 0.400 Hz  (parasympathetic / respiratory)

    Note: reliable frequency-domain HRV requires ≥ 5 min of data.  For
    the 10-second rPPG buffer used here the values are indicative only.

    Parameters
    ----------
    rr_ms : np.ndarray or None
        RR intervals in milliseconds.
    fs_rr : float
        Re-sampling frequency for the tachogram (Hz).  Default 4 Hz.

    Returns
    -------
    dict with keys:
        lf_power   float  — LF band power (ms²)
        hf_power   float  — HF band power (ms²)
        lf_hf      float  — LF/HF ratio (0.0 if HF is negligible)
    """
    _empty = {"lf_power": 0.0, "hf_power": 0.0, "lf_hf": 0.0}

    if rr_ms is None or len(rr_ms) < 4:
        return _empty

    # Cumulative time axis (seconds) for the beat series
    rr_sec    = rr_ms / 1000.0
    t_beats   = np.concatenate([[0.0], np.cumsum(rr_sec[:-1])])
    total_sec = float(t_beats[-1])

    if total_sec < 2.0:
        return _empty

    # Uniform grid
    t_uniform = np.arange(0.0, total_sec, 1.0 / fs_rr)
    if len(t_uniform) < 8:
        return _empty

    # Linear interpolation onto the uniform grid
    rr_uniform = np.interp(t_uniform, t_beats, rr_ms)

    # Welch PSD — nperseg capped at signal length
    nperseg = min(len(rr_uniform), max(8, int(fs_rr * 60)))  # up to 60-s segments
    freqs, psd = welch(rr_uniform, fs=fs_rr, nperseg=nperseg, scaling="density")

    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)

    lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
    lf_hf    = lf_power / hf_power if hf_power > 1e-10 else 0.0

    return {
        "lf_power": max(0.0, lf_power),
        "hf_power": max(0.0, hf_power),
        "lf_hf":    lf_hf,
    }


def compute_hrv_metrics(
    signal: np.ndarray,
    fs: float,
    min_distance_sec: float = 0.4,
    min_prominence: float = 0.2,
) -> Dict[str, Any]:
    """
    Full HRV analysis pipeline for a single rPPG signal window.

    Runs peak detection, RR-interval extraction, and computes all
    standard time-domain HRV metrics in one call.

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        Filtered, normalised rPPG signal.
    fs : float
        Sampling frequency in Hz.
    min_distance_sec : float
        Minimum inter-peak interval in seconds.
    min_prominence : float
        Minimum relative peak prominence.

    Returns
    -------
    metrics : dict
        ┌─────────────┬────────────────────────────────────────────────┐
        │ Key         │ Value                                          │
        ├─────────────┼────────────────────────────────────────────────┤
        │ peaks       │ np.ndarray — peak sample indices               │
        │ rr_ms       │ np.ndarray | None — RR intervals (ms)          │
        │ sdnn        │ float — SDNN (ms)                              │
        │ rmssd       │ float — RMSSD (ms)                             │
        │ hr_mean_bpm │ float — mean HR derived from RR intervals      │
        │ n_peaks     │ int   — number of detected beats               │
        └─────────────┴────────────────────────────────────────────────┘
    """
    peaks = detect_peaks(signal, fs, min_distance_sec, min_prominence)
    rr_ms = compute_rr_intervals(peaks, fs)

    sdnn    = compute_sdnn(rr_ms)
    rmssd   = compute_rmssd(rr_ms)
    mean_rr = compute_mean_rr(rr_ms)
    pnn50   = compute_pnn50(rr_ms)
    freq_hrv = compute_frequency_domain_hrv(rr_ms)

    if rr_ms is not None and len(rr_ms) > 0:
        hr_mean_bpm = 60_000.0 / float(np.mean(rr_ms))
    else:
        hr_mean_bpm = 0.0

    return {
        "peaks":       peaks,
        "rr_ms":       rr_ms,
        "sdnn":        sdnn,
        "rmssd":       rmssd,
        "mean_rr":     mean_rr,
        "pnn50":       pnn50,
        "lf_power":    freq_hrv["lf_power"],
        "hf_power":    freq_hrv["hf_power"],
        "lf_hf":       freq_hrv["lf_hf"],
        "hr_mean_bpm": hr_mean_bpm,
        "n_peaks":     int(len(peaks)),
    }
