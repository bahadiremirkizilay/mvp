"""
Signal Processing for rPPG
===========================
Provides the complete signal-conditioning chain applied to raw RGB traces
before and after POS projection:

    Raw RGB buffer
        ↓  detrend_signal()      — remove linear trend per channel
        ↓  pos_sliding_window()  — POS projection  [see pos_method.py]
        ↓  bandpass_filter()     — zero-phase Butterworth (0.7–4 Hz)
        ↓  normalize_signal()    — z-score normalisation
        ↓  estimate_bpm()        — FFT peak → dominant frequency → BPM

Design choices:
    • filtfilt() is used throughout for zero-phase (no temporal shift)
      filtering, which is critical for accurate peak-timing in HRV
      estimation.
    • A Hanning window is applied before FFT to reduce spectral leakage.
    • Frequency estimation is restricted to the physiological HR band
      [low_hz, high_hz] to suppress noise-driven false peaks.
"""

import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.ndimage import uniform_filter1d
from typing import Tuple


# ---------------------------------------------------------------------------
# Detrending
# ---------------------------------------------------------------------------

def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """
    Remove the linear (least-squares) trend from a 1-D or 2-D signal.

    For 2-D input of shape (T, C) — as produced by the RGB buffer —
    detrending is applied independently to each column (channel), which
    is the standard pre-processing step in rPPG literature to compensate
    for slow illumination drift.

    Parameters
    ----------
    signal : np.ndarray, shape (T,) or (T, C)

    Returns
    -------
    np.ndarray
        Detrended signal, same shape as input.
    """
    if signal.ndim == 1:
        return detrend(signal, type="linear")
    # Column-wise detrending for multi-channel (R, G, B) input
    return np.apply_along_axis(lambda x: detrend(x, type="linear"), 0, signal)


# ---------------------------------------------------------------------------
# Butterworth bandpass filter
# ---------------------------------------------------------------------------

def _design_bandpass(
    low_hz: float, high_hz: float, fs: float, order: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a digital Butterworth bandpass filter.

    Parameters
    ----------
    low_hz, high_hz : float
        Cutoff frequencies in Hz (3 dB points).
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order.  Order 4 provides a good roll-off / phase-linearity
        trade-off for typical rPPG sampling rates (25–30 Hz).

    Returns
    -------
    b, a : IIR filter coefficients (numerator, denominator).
    """
    nyq = 0.5 * fs
    # Normalise and clamp to the open interval (0, 1) to prevent
    # ValueError from scipy.signal.butter on edge-case configs
    low  = float(np.clip(low_hz  / nyq, 1e-4, 1.0 - 1e-4))
    high = float(np.clip(high_hz / nyq, 1e-4, 1.0 - 1e-4))
    return butter(order, [low, high], btype="band")


def bandpass_filter(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter using forward-backward
    filtering (filtfilt), preserving temporal alignment of all features.

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        1-D signal to filter.
    low_hz, high_hz : float
        Passband cutoff frequencies (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int
        Filter order (default 4).

    Returns
    -------
    np.ndarray, shape (T,)
        Filtered signal.  Returns a copy of the input unchanged if it is
        too short to meet filtfilt's minimum-length requirement.
    """
    b, a = _design_bandpass(low_hz, high_hz, fs, order)

    # filtfilt requires signal length > padlen = 3 * max(len(a), len(b))
    min_len = 3 * max(len(a), len(b))
    if len(signal) < min_len:
        return signal.copy()

    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Spectral analysis & BPM estimation
# ---------------------------------------------------------------------------

def compute_fft(
    signal: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the single-sided power spectrum of a real-valued signal.

    A Hanning window is applied before the FFT to reduce spectral leakage
    from the finite observation window.

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        Pre-processed (detrended, filtered) signal.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis in Hz, length ⌊T/2⌋ + 1.
    power : np.ndarray
        Power (|FFT|² / T), same length as freqs.
    """
    n = len(signal)
    windowed = signal * np.hanning(n)
    fft_vals = np.fft.rfft(windowed)
    power = (np.abs(fft_vals) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, power


def estimate_bpm(
    signal: np.ndarray,
    fs: float,
    low_hz: float = 0.7,
    high_hz: float = 3.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate heart rate (BPM) from a processed rPPG signal via FFT peak
    detection within the physiologically valid frequency band.

    Strategy:
        1. Compute power spectrum of the full signal window.
        2. Zero out spectral components outside [low_hz, high_hz].
        3. Return the frequency with maximum power as the dominant HR.

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        Processed rPPG signal (detrended, POS-projected, bandpassed).
    fs : float
        Sampling rate in Hz.
    low_hz, high_hz : float
        Physiological HR search band (Hz).
        Default: 0.7 Hz (42 BPM) – 4.0 Hz (240 BPM).

    Returns
    -------
    bpm : float
        Estimated heart rate in beats per minute.
    freqs : np.ndarray
        Full frequency axis (Hz) for downstream plotting.
    power : np.ndarray
        Full power spectrum for downstream plotting.
    """
    freqs, power = compute_fft(signal, fs)

    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not band_mask.any():
        return 0.0, freqs, power

    # Restrict search to valid band; leave full spectrum untouched for plots
    band_power = np.where(band_mask, power, 0.0)
    dominant_idx = int(np.argmax(band_power))
    dominant_freq = float(freqs[dominant_idx])
    bpm = dominant_freq * 60.0

    return bpm, freqs, power


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalise a signal to zero mean and unit variance.

    Parameters
    ----------
    signal : np.ndarray

    Returns
    -------
    np.ndarray
        Normalised signal.  If the standard deviation is negligible
        (constant signal), only the mean is subtracted.
    """
    mu = signal.mean()
    sigma = signal.std()
    if sigma < 1e-8:
        return signal - mu
    return (signal - mu) / sigma


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def moving_average_filter(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Apply a uniform causal moving average to a 1-D rPPG signal.

    Reduces frame-to-frame noise while preserving the dominant cardiac
    frequency.  The convolution uses ``mode='same'`` so the output length
    equals the input length; edge effects are confined to the first and
    last ``window_samples // 2`` samples and are negligible for long
    buffers (> 5 s at typical frame rates).

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        Raw 1-D rPPG signal (e.g., POS output before bandpass).
    window_samples : int
        Number of samples in the averaging window.
        At 30 FPS, ``window_samples = 9`` corresponds to 0.3 s.

    Returns
    -------
    np.ndarray, shape (T,)
        Smoothed signal.  Returns an unmodified copy when
        ``window_samples < 2`` or the signal is shorter than the window.
    """
    if window_samples < 2 or len(signal) < window_samples:
        return signal.copy()
    kernel = np.ones(window_samples, dtype=np.float64) / window_samples
    return np.convolve(signal, kernel, mode="same")


# ---------------------------------------------------------------------------
# Temporal RGB normalisation
# ---------------------------------------------------------------------------

def temporal_normalize_rgb(
    rgb_buffer: np.ndarray,
    window_samples: int = 30,
) -> np.ndarray:
    """
    Suppress illumination drift by applying a rolling-window normalisation
    to each RGB channel: subtract the rolling mean, divide by the rolling
    standard deviation.

    Uses ``scipy.ndimage.uniform_filter1d`` for an efficient single-pass
    computation of both the rolling mean and variance (via E[X²] - E[X]²),
    making the operation O(T × C) regardless of window size.

    Parameters
    ----------
    rgb_buffer : np.ndarray, shape (T, 3)
        Raw detrended RGB buffer (R, G, B columns), any value scale.
    window_samples : int
        Width of the uniform rolling window in samples.
        Default 30 ≈ 1 s at 30 FPS.

    Returns
    -------
    np.ndarray, shape (T, 3), dtype float64
        Rolling-normalised RGB buffer with approximately zero mean and
        unit standard deviation per channel.
    """
    buf = rgb_buffer.astype(np.float64)
    rolling_mean    = uniform_filter1d(buf,      size=window_samples, axis=0, mode="nearest")
    rolling_mean_sq = uniform_filter1d(buf ** 2, size=window_samples, axis=0, mode="nearest")
    rolling_var     = rolling_mean_sq - rolling_mean ** 2
    rolling_std     = np.sqrt(np.maximum(rolling_var, 1e-8))
    return (buf - rolling_mean) / rolling_std


# ---------------------------------------------------------------------------
# Spectral quality metrics
# ---------------------------------------------------------------------------

def compute_peak_confidence(
    freqs: np.ndarray,
    power: np.ndarray,
    low_hz: float = 0.8,
    high_hz: float = 2.2,
) -> float:
    """
    Estimate a confidence score [0, 1] for the dominant spectral peak being
    a true cardiac signal rather than noise.

    The metric is the SNR of the peak against the surrounding band (a ±0.15 Hz
    exclusion zone around the peak removes its immediate neighbourhood from
    the baseline), mapped through tanh compression so that SNR = 1 (flat
    spectrum) yields 0.0 and SNR ≥ 5 yields ≥ 0.9.

    Parameters
    ----------
    freqs, power : np.ndarray
        Full frequency axis (Hz) and power spectrum from ``estimate_bpm()``.
    low_hz, high_hz : float
        Cardiac band boundaries (Hz).

    Returns
    -------
    float
        Confidence in [0, 1].  Returns 0.0 if the band is empty.
    """
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not band_mask.any():
        return 0.0

    band_power = power[band_mask]
    band_freqs = freqs[band_mask]

    peak_idx   = int(np.argmax(band_power))
    peak_power = float(band_power[peak_idx])
    peak_freq  = float(band_freqs[peak_idx])

    # Surrounding mean — exclude ±0.15 Hz neighbourhood around the peak
    surround = band_power[np.abs(band_freqs - peak_freq) >= 0.15]
    if len(surround) == 0 or surround.mean() < 1e-14:
        return 0.5   # insufficient context — return neutral score

    snr = peak_power / surround.mean()
    return float(np.clip(np.tanh((snr - 1.0) / 4.0), 0.0, 1.0))


def compute_signal_quality(
    freqs: np.ndarray,
    power: np.ndarray,
    low_hz: float = 0.8,
    high_hz: float = 2.2,
) -> float:
    """
    Compute spectral purity: the fraction of total spectral power that falls
    inside the cardiac band [low_hz, high_hz].

    Higher values indicate that most signal energy is concentrated at cardiac
    frequencies, suggesting a clean recording with little motion or lighting
    contamination.  Used as a per-ROI fusion weight in the multi-ROI pipeline.

    Parameters
    ----------
    freqs, power : np.ndarray
        Frequency axis and power spectrum.
    low_hz, high_hz : float
        Cardiac band boundaries (Hz).

    Returns
    -------
    float
        Quality score in [0, 1].  Returns 0.0 if total power is negligible.
    """
    total_power = float(power.sum())
    if total_power < 1e-14:
        return 0.0
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.clip(power[band_mask].sum() / total_power, 0.0, 1.0))


def compute_sqi(
    freqs: np.ndarray,
    power: np.ndarray,
    low_hz: float = 0.8,
    high_hz: float = 2.2,
) -> float:
    """
    Compute a Signal Quality Index (SQI) from the ratio of cardiac-band
    power to out-of-band (noise) power.

    Formula
    -------
    SQI_raw  = signal_power / max(noise_power, ε)
    SQI_norm = SQI_raw / (1 + SQI_raw)          # maps [0, ∞) → [0, 1)

    The normalisation ensures the output is always in [0, 1]:
        SQI = 0.5  →  signal and noise power are equal.
        SQI → 1    →  nearly all spectral energy is in the cardiac band.
        SQI → 0    →  signal energy is negligible relative to noise.

    Interpretive thresholds (empirical guidelines for webcam rPPG):
        SQI ≥ 0.6  — reliable estimate
        0.4 ≤ SQI < 0.6  — moderate quality; treat BPM with caution
        SQI < 0.4  — low quality (motion or lighting artefacts likely)

    Parameters
    ----------
    freqs, power : np.ndarray
        Frequency axis (Hz) and power spectrum from ``estimate_bpm()``.
    low_hz, high_hz : float
        Cardiac band boundaries (Hz).

    Returns
    -------
    float
        SQI in [0, 1].  Returns 0.0 if total power is negligible.
    """
    band_mask  = (freqs >= low_hz) & (freqs <= high_hz)
    noise_mask = ~band_mask

    signal_power = float(power[band_mask].sum())  if band_mask.any()  else 0.0
    noise_power  = float(power[noise_mask].sum()) if noise_mask.any() else 1e-14

    if signal_power < 1e-14:
        return 0.0

    sqi_raw = signal_power / max(noise_power, 1e-14)
    return float(np.clip(sqi_raw / (1.0 + sqi_raw), 0.0, 1.0))
