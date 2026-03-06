"""
POS (Plane-Orthogonal-to-Skin) rPPG Method
============================================
Implements the Plane-Orthogonal-to-Skin projection algorithm for
contactless pulse-rate estimation from facial video.

The core insight of POS is that haemoglobin-driven colour changes in
skin lie in the plane orthogonal to the DC skin-tone vector in RGB
colour space. By projecting the temporally normalised RGB signal onto
this plane, the dominant pulsatile component is isolated from
illumination and skin-tone variation.

Algorithm (Wang et al., 2017):
    1. Temporal normalisation: C_n(t) = C(t) / μ_C removes the DC
       skin-tone baseline while preserving the AC pulsatile component.
    2. Skin-plane projection via fixed matrix P_skin:
           S1 = 0·R + 1·G − 1·B  =  G − B
           S2 = −2·R + 1·G + 1·B
    3. Adaptive combination:  P = S1 + α·S2
       where α = σ(S1)/σ(S2) equalises the two projection channels.
    4. Sub-window overlap-add aggregation (Section IV) improves SNR by
       averaging independent estimates from short overlapping windows.

Reference:
    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
    Algorithmic principles of remote PPG.
    IEEE Transactions on Biomedical Engineering, 64(7), 1479–1491.
    https://doi.org/10.1109/TBME.2016.2609282
"""

import numpy as np


def pos_project(rgb_window: np.ndarray) -> np.ndarray:
    """
    Apply the POS projection to one temporal window of RGB observations.

    Parameters
    ----------
    rgb_window : np.ndarray, shape (T, 3)
        Sequence of mean RGB observations [R, G, B] over T frames.
        Values may be in any consistent scale (e.g., 0–255 or 0–1).
        A minimum of 2 frames is required.

    Returns
    -------
    pulse : np.ndarray, shape (T,)
        Zero-mean estimated rPPG pulse signal (arbitrary amplitude units).

    Raises
    ------
    ValueError
        If ``rgb_window`` does not have shape (T, 3).
    """
    if rgb_window.ndim != 2 or rgb_window.shape[1] != 3:
        raise ValueError(
            f"Expected rgb_window of shape (T, 3), got {rgb_window.shape}."
        )

    T = rgb_window.shape[0]
    if T < 2:
        return np.zeros(T, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 1 — Temporal normalisation
    # Divides each channel by its window mean so that absolute intensity
    # differences between windows are removed.  ε prevents division by
    # zero for near-black regions or saturated channels.
    # ------------------------------------------------------------------
    channel_means = rgb_window.mean(axis=0)          # (3,)
    eps = 1e-8
    C_n = rgb_window / (channel_means + eps)          # (T, 3), unit-normalised

    # ------------------------------------------------------------------
    # Step 2 — Skin-plane projection  (Wang et al. 2017, Table I)
    # Projection matrix P_skin (rows):
    #   [ 0,  1, -1 ]   →   S1 = G - B
    #   [-2,  1,  1 ]   →   S2 = -2R + G + B
    # ------------------------------------------------------------------
    S1 = C_n[:, 1] - C_n[:, 2]                       # G − B
    S2 = -2.0 * C_n[:, 0] + C_n[:, 1] + C_n[:, 2]   # −2R + G + B

    # ------------------------------------------------------------------
    # Step 3 — Adaptive combination
    # α balances the two channels by equalising their standard deviations.
    # When S2 carries no signal (near-zero variance) we use S1 alone.
    # ------------------------------------------------------------------
    std_s2 = S2.std()
    if std_s2 < eps:
        pulse = S1.copy()
    else:
        alpha = S1.std() / std_s2
        pulse = S1 + alpha * S2

    # Zero-mean the output so subsequent bandpass filtering is well-centred
    pulse -= pulse.mean()

    return pulse.astype(np.float64)


def pos_sliding_window(
    rgb_buffer: np.ndarray,
    fs: float,
    window_sec: float = 1.6,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Apply POS over overlapping sub-windows and aggregate via overlap-add.

    Short sub-windows improve temporal stationarity: within 1–2 s the
    skin-tone baseline is approximately constant, making the temporal
    normalisation of POS maximally effective.  The overlap-add step
    fills the full buffer length H while weighting each sample by a
    Hanning window to suppress edge discontinuities.

    Parameters
    ----------
    rgb_buffer : np.ndarray, shape (T, 3)
        Full sliding-buffer of mean RGB observations.
    fs : float
        Sampling frequency in Hz (camera FPS).
    window_sec : float
        Sub-window duration in seconds.  Wang et al. recommend 1.6 s.
    overlap : float
        Fractional overlap between consecutive sub-windows (0 < overlap < 1).
        0.5 (50 % overlap) is the standard choice.

    Returns
    -------
    H : np.ndarray, shape (T,)
        Overlap-added, Hanning-weighted aggregated rPPG signal.
    """
    T = len(rgb_buffer)
    win_len = max(2, int(window_sec * fs))
    hop = max(1, int(win_len * (1.0 - overlap)))

    hann_win = np.hanning(win_len)

    H = np.zeros(T, dtype=np.float64)
    weight = np.zeros(T, dtype=np.float64)

    # ---- Full sub-windows ----
    start = 0
    while start + win_len <= T:
        end = start + win_len
        chunk = rgb_buffer[start:end]                 # (win_len, 3)

        pulse = pos_project(chunk)                    # (win_len,)

        # Normalise to unit variance before accumulation so that
        # windows with high-intensity motion do not dominate the sum
        std_p = pulse.std()
        if std_p > 1e-8:
            pulse /= std_p

        H[start:end] += pulse * hann_win
        weight[start:end] += hann_win
        start += hop

    # ---- Partial trailing window (if any) ----
    if start < T:
        chunk = rgb_buffer[start:]
        pulse = pos_project(chunk)
        std_p = pulse.std()
        if std_p > 1e-8:
            pulse /= std_p
        trunc_hann = np.hanning(len(chunk))
        H[start:] += pulse * trunc_hann
        weight[start:] += trunc_hann

    # Normalise by accumulated Hanning weight; avoid near-zero division at edges
    valid = weight > 1e-8
    H[valid] /= weight[valid]

    return H
