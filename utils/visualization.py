"""
Real-Time rPPG Visualizer
==========================
Provides a three-panel live matplotlib figure updated from the main
capture loop without blocking frame acquisition.

Layout
------
    Panel 1 (top):    Raw mean RGB signals vs. time (R=red, G=green, B=blue)
    Panel 2 (middle): Filtered POS-rPPG signal with detected peak overlay
    Panel 3 (bottom): Single-sided power spectrum (0.4–5.0 Hz) with the
                      dominant frequency marked and the physiological HR
                      band [0.7–4.0 Hz] shaded

Usage
-----
    viz = Visualizer(fs=30.0, window_sec=10.0)

    # Inside the capture loop (call every vis_interval frames):
    viz.update(rgb_buffer, rppg_signal, freqs, power, bpm,
               peaks=hrv["peaks"], hrv_metrics=hrv)

    viz.close()   # on exit
"""

import collections
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

# Select an interactive backend that works on Windows with Python 3.10.
# Preference order: TkAgg (most portable) -> Qt5Agg -> fallback to Agg
# (Agg is non-interactive but at least won't crash on headless setups).
def _select_backend() -> None:
    current = matplotlib.get_backend().lower()
    if current in ("tkagg", "qt5agg", "qt6agg", "wxagg"):
        return  # already interactive
    for backend in ("TkAgg", "Qt5Agg", "WxAgg"):
        try:
            matplotlib.use(backend)
            return
        except Exception:
            continue
    # Fall back to non-interactive Agg if no windowing toolkit available
    matplotlib.use("Agg")

_select_backend()


# Palette constants (dark-theme, colourblind-friendly choices)
_BG_FIGURE   = "#1a1a2e"
_BG_AXES     = "#0f0f1e"
_GRID_COLOR  = "#222244"
_SPINE_COLOR = "#444466"
_TEXT_COLOR  = "white"
_COLOR_R     = "#ff4444"
_COLOR_G     = "#44dd44"
_COLOR_B     = "#4488ff"
_COLOR_RPPG  = "#00e5ff"
_COLOR_PEAK  = "#ff9900"
_COLOR_PSD   = "#b388ff"
_COLOR_BAND  = "#00e5ff"


class Visualizer:
    """
    Non-blocking live visualizer for the rPPG pipeline.

    Matplotlib interactive mode (``plt.ion()``) is used so that
    ``fig.canvas.draw_idle()`` + ``plt.pause(0.001)`` refresh the plot
    without blocking the OpenCV capture loop.
    """

    def __init__(self, fs: float, window_sec: float = 10.0) -> None:
        """
        Parameters
        ----------
        fs : float
            Camera sampling frequency in Hz.  Used to construct time axes.
        window_sec : float
            Display window length in seconds (should match rPPG buffer).
        """
        self.fs = float(fs)
        self.window_sec = float(window_sec)

        plt.ion()
        self.fig = plt.figure(figsize=(12, 8), facecolor=_BG_FIGURE)
        try:
            self.fig.canvas.manager.set_window_title(
                "Psychophysiological Analysis — rPPG / POS Method"
            )
        except AttributeError:
            pass  # Some backends do not expose the window title

        gs = gridspec.GridSpec(
            3, 1, figure=self.fig, hspace=0.50, top=0.92, bottom=0.07
        )

        # ── Panel 1: Raw RGB ─────────────────────────────────────────
        self.ax_rgb = self.fig.add_subplot(gs[0])
        _style_axes(self.ax_rgb,
                    title="Raw RGB Mean Signals",
                    xlabel="Time (s)",
                    ylabel="Intensity")
        (self.ln_r,) = self.ax_rgb.plot([], [], color=_COLOR_R,  lw=1.0, label="R")
        (self.ln_g,) = self.ax_rgb.plot([], [], color=_COLOR_G,  lw=1.0, label="G")
        (self.ln_b,) = self.ax_rgb.plot([], [], color=_COLOR_B,  lw=1.0, label="B")
        self.ax_rgb.legend(
            loc="upper right", fontsize=8,
            framealpha=0.3, facecolor="#2a2a4a", labelcolor=_TEXT_COLOR
        )

        # ── Panel 2: Filtered rPPG ───────────────────────────────────
        self.ax_sig = self.fig.add_subplot(gs[1])
        _style_axes(self.ax_sig,
                    title="Filtered rPPG Signal  (POS + Butterworth 0.7–4 Hz)",
                    xlabel="Time (s)",
                    ylabel="Amplitude (a.u.)")
        (self.ln_sig,) = self.ax_sig.plot(
            [], [], color=_COLOR_RPPG, lw=1.5, label="rPPG"
        )
        self.sc_peaks = self.ax_sig.scatter(
            [], [], color=_COLOR_PEAK, s=45, zorder=5, label="Peaks"
        )
        self.ax_sig.legend(
            loc="upper right", fontsize=8,
            framealpha=0.3, facecolor="#2a2a4a", labelcolor=_TEXT_COLOR
        )

        # ── Panel 3: Power spectrum ──────────────────────────────────
        self.ax_psd = self.fig.add_subplot(gs[2])
        _style_axes(self.ax_psd,
                    title="Power Spectral Density",
                    xlabel="Frequency (Hz)",
                    ylabel="Power")
        (self.ln_psd,) = self.ax_psd.plot([], [], color=_COLOR_PSD, lw=1.5)

        # Shaded physiological HR band
        self.ax_psd.axvspan(0.7, 4.0, alpha=0.10, color=_COLOR_BAND, label="HR band")

        # Dominant frequency marker (vertical dashed line)
        self.vl_peak = self.ax_psd.axvline(
            x=0, color=_COLOR_PEAK, lw=1.8, ls="--", alpha=0.90
        )
        # Annotation inside the PSD panel
        self.txt_bpm = self.ax_psd.text(
            0.02, 0.90, "",
            transform=self.ax_psd.transAxes,
            color=_COLOR_PEAK, fontsize=10, fontweight="bold",
            verticalalignment="top"
        )

        # ── Header: summary metrics ──────────────────────────────────
        self.txt_header = self.fig.text(
            0.5, 0.965, "Initialising — buffering signal…",
            ha="center", va="top",
            color=_TEXT_COLOR, fontsize=11, fontweight="bold",
            transform=self.fig.transFigure,
        )

        plt.draw()
        plt.pause(0.05)

    # ------------------------------------------------------------------
    # Public update method
    # ------------------------------------------------------------------

    def update(
        self,
        rgb_buffer: np.ndarray,
        rppg_signal: np.ndarray,
        freqs: np.ndarray,
        power: np.ndarray,
        bpm: float,
        peaks: Optional[np.ndarray] = None,
        hrv_metrics: Optional[dict] = None,
    ) -> None:
        """
        Refresh all panels with the latest data.

        Parameters
        ----------
        rgb_buffer : np.ndarray, shape (T, 3)
            Raw per-channel mean values (R, G, B).
        rppg_signal : np.ndarray, shape (T,)
            Processed (filtered, normalised) rPPG signal.
        freqs : np.ndarray
            FFT frequency axis in Hz.
        power : np.ndarray
            FFT power spectrum.
        bpm : float
            Estimated heart rate in BPM.
        peaks : np.ndarray or None
            Peak sample indices into ``rppg_signal``.
        hrv_metrics : dict or None
            Output of ``hrv.compute_hrv_metrics()``; used for the header.
        """
        T = len(rppg_signal)
        t = np.arange(T) / self.fs

        # ---- Panel 1: RGB ----
        self.ln_r.set_data(t, rgb_buffer[:T, 0])
        self.ln_g.set_data(t, rgb_buffer[:T, 1])
        self.ln_b.set_data(t, rgb_buffer[:T, 2])
        _autoscale(self.ax_rgb, t, rgb_buffer[:T])

        # ---- Panel 2: rPPG ----
        self.ln_sig.set_data(t, rppg_signal)
        _autoscale(self.ax_sig, t, rppg_signal[:, np.newaxis])

        if peaks is not None and len(peaks) > 0:
            valid = peaks[peaks < T]
            if len(valid) > 0:
                self.sc_peaks.set_offsets(
                    np.column_stack([t[valid], rppg_signal[valid]])
                )
            else:
                self.sc_peaks.set_offsets(np.empty((0, 2)))
        else:
            self.sc_peaks.set_offsets(np.empty((0, 2)))

        # ---- Panel 3: PSD ----
        # Show only the 0.4–5.0 Hz range for clarity
        band_mask = (freqs >= 0.4) & (freqs <= 5.0)
        if band_mask.any():
            self.ln_psd.set_data(freqs[band_mask], power[band_mask])
            self.ax_psd.set_xlim(0.4, 5.0)
            y_max = power[band_mask].max()
            self.ax_psd.set_ylim(0, max(y_max * 1.20, 1e-10))

        dom_freq = bpm / 60.0
        self.vl_peak.set_xdata([dom_freq])
        self.txt_bpm.set_text(f"HR = {bpm:.1f} BPM  ({dom_freq:.3f} Hz)")

        # ---- Header ----
        if hrv_metrics and hrv_metrics.get("n_peaks", 0) >= 2:
            header = (
                f"HR: {bpm:.1f} BPM  |  "
                f"SDNN: {hrv_metrics['sdnn']:.1f} ms  |  "
                f"RMSSD: {hrv_metrics['rmssd']:.1f} ms  |  "
                f"Beats detected: {hrv_metrics['n_peaks']}"
            )
        else:
            header = (
                f"HR: {bpm:.1f} BPM  |  "
                "SDNN: ---  |  RMSSD: ---  |  "
                "Buffering signal…"
            )
        self.txt_header.set_text(header)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        """Close the matplotlib figure and release resources."""
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the public API)
# ---------------------------------------------------------------------------

def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the consistent dark-theme style to a single Axes object."""
    ax.set_facecolor(_BG_AXES)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=7)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)
    ax.set_title(title, color=_TEXT_COLOR, fontsize=9,
                 fontweight="bold", pad=4)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=_GRID_COLOR, lw=0.5, linestyle="--", alpha=0.8)


def _autoscale(ax, xdata: np.ndarray, ydata: np.ndarray) -> None:
    """Set axis limits with a 5% margin around the data range."""
    if len(xdata) == 0:
        return
    ax.set_xlim(float(xdata[0]), float(xdata[-1]) + 1e-9)
    y_min, y_max = float(ydata.min()), float(ydata.max())
    margin = max((y_max - y_min) * 0.06, 0.5)
    ax.set_ylim(y_min - margin, y_max + margin)


# ---------------------------------------------------------------------------
# Rolling-window metrics visualizer
# ---------------------------------------------------------------------------

_COLOR_BPM   = "#ffb347"   # warm orange — heart rate
_COLOR_RMSSD = "#90ee90"   # light green — HRV
_COLOR_WAVE  = "#00e5ff"   # cyan        — rPPG waveform (matches existing)


class MetricsVisualizer:
    """
    Separate rolling-window live chart for debugging signal stability.

    Displays three stacked panels in a dedicated figure window, updated
    from the main capture loop without blocking frame acquisition:

        Panel 1: Smoothed heart rate (BPM) — rolling 30 s history
        Panel 2: RMSSD (ms)                — rolling 30 s history
        Panel 3: Current rPPG waveform     — latest signal buffer

    Usage
    -----
        mv = MetricsVisualizer(history_sec=30.0)

        # Inside the processing block:
        mv.update(smoothed_bpm, rmssd, rppg_signal, fs_actual)

        mv.close()   # on exit
    """

    # Maximum stored data points for the rolling metrics deques.
    # 600 points ≈ 10 min of 1-Hz estimates — ample headroom.
    _MAXLEN = 600

    def __init__(self, history_sec: float = 30.0) -> None:
        """
        Parameters
        ----------
        history_sec : float
            Width of the rolling display window for BPM and RMSSD, in
            seconds.  Default 30 s.
        """
        self.history_sec = float(history_sec)

        # Rolling metric stores — parallel deques (wall-clock time, value)
        self._bpm_times:   collections.deque = collections.deque(maxlen=self._MAXLEN)
        self._bpm_vals:    collections.deque = collections.deque(maxlen=self._MAXLEN)
        self._rmssd_times: collections.deque = collections.deque(maxlen=self._MAXLEN)
        self._rmssd_vals:  collections.deque = collections.deque(maxlen=self._MAXLEN)

        self._t0: float = time.perf_counter()   # reference for x-axis

        # ── Figure setup ─────────────────────────────────────────────
        self.fig = plt.figure(figsize=(10, 7), facecolor=_BG_FIGURE)
        try:
            self.fig.canvas.manager.set_window_title(
                "rPPG — Metrics Monitor (30 s rolling)"
            )
        except AttributeError:
            pass

        gs = gridspec.GridSpec(
            3, 1, figure=self.fig, hspace=0.55, top=0.91, bottom=0.07
        )

        # Panel 1: Smoothed BPM
        self.ax_bpm = self.fig.add_subplot(gs[0])
        _style_axes(self.ax_bpm,
                    title="Smoothed Heart Rate",
                    xlabel="Time (s)",
                    ylabel="BPM")
        self.ax_bpm.set_ylim(40, 130)
        # Physiologically normal resting HR band
        self.ax_bpm.axhspan(60, 100, alpha=0.08, color=_COLOR_BPM)
        (self.ln_bpm,) = self.ax_bpm.plot(
            [], [], color=_COLOR_BPM, lw=2.0, marker="o",
            markersize=3, label="Smoothed BPM"
        )
        self.ax_bpm.legend(
            loc="upper right", fontsize=7,
            framealpha=0.3, facecolor="#2a2a4a", labelcolor=_TEXT_COLOR,
        )

        # Panel 2: RMSSD
        self.ax_rmssd = self.fig.add_subplot(gs[1])
        _style_axes(self.ax_rmssd,
                    title="RMSSD (Short-term HRV)",
                    xlabel="Time (s)",
                    ylabel="ms")
        self.ax_rmssd.set_ylim(0, 120)
        (self.ln_rmssd,) = self.ax_rmssd.plot(
            [], [], color=_COLOR_RMSSD, lw=2.0, marker="s",
            markersize=3, label="RMSSD"
        )
        self.ax_rmssd.legend(
            loc="upper right", fontsize=7,
            framealpha=0.3, facecolor="#2a2a4a", labelcolor=_TEXT_COLOR,
        )

        # Panel 3: rPPG waveform
        self.ax_wave = self.fig.add_subplot(gs[2])
        _style_axes(self.ax_wave,
                    title="rPPG Waveform (current buffer)",
                    xlabel="Time (s)",
                    ylabel="Amplitude (a.u.)")
        (self.ln_wave,) = self.ax_wave.plot(
            [], [], color=_COLOR_WAVE, lw=1.2, label="rPPG"
        )
        self.sc_peaks_mv = self.ax_wave.scatter(
            [], [], color=_COLOR_PEAK, s=35, zorder=5, label="Peaks"
        )
        self.ax_wave.legend(
            loc="upper right", fontsize=7,
            framealpha=0.3, facecolor="#2a2a4a", labelcolor=_TEXT_COLOR,
        )

        # Header text
        self.txt_hdr = self.fig.text(
            0.5, 0.965, "Buffering…",
            ha="center", va="top",
            color=_TEXT_COLOR, fontsize=10, fontweight="bold",
            transform=self.fig.transFigure,
        )

        plt.draw()
        plt.pause(0.05)

    # ------------------------------------------------------------------

    def update(
        self,
        smoothed_bpm: float,
        rmssd: float,
        rppg_signal: np.ndarray,
        fs: float,
        peaks: Optional[np.ndarray] = None,
    ) -> None:
        """
        Append the latest estimates and redraw all three panels.

        Parameters
        ----------
        smoothed_bpm : float
            Median-filtered heart rate estimate in BPM.
        rmssd : float
            RMSSD in milliseconds.
        rppg_signal : np.ndarray, shape (T,)
            Current normalised rPPG waveform buffer.
        fs : float
            Sampling rate in Hz (used to build the waveform time axis).
        peaks : np.ndarray or None
            Peak sample indices into ``rppg_signal`` for overlay markers.
        """
        now_rel = time.perf_counter() - self._t0   # seconds since start

        # ── Append to rolling metric stores ──────────────────────────
        if smoothed_bpm > 0:
            self._bpm_times.append(now_rel)
            self._bpm_vals.append(smoothed_bpm)
        if rmssd > 0:
            self._rmssd_times.append(now_rel)
            self._rmssd_vals.append(rmssd)

        # ── Panel 1: BPM (rolling window) ────────────────────────────
        self._update_rolling(
            self.ax_bpm, self.ln_bpm,
            self._bpm_times, self._bpm_vals,
            now_rel, fixed_ylim=(40.0, 130.0),
        )

        # ── Panel 2: RMSSD (rolling window) ──────────────────────────
        self._update_rolling(
            self.ax_rmssd, self.ln_rmssd,
            self._rmssd_times, self._rmssd_vals,
            now_rel, fixed_ylim=None,
        )

        # ── Panel 3: rPPG waveform ────────────────────────────────────
        T = len(rppg_signal)
        if T > 0 and fs > 0:
            t_wave = np.arange(T) / float(fs)
            self.ln_wave.set_data(t_wave, rppg_signal)
            _autoscale(self.ax_wave, t_wave, rppg_signal[:, np.newaxis])

            if peaks is not None and len(peaks) > 0:
                valid = peaks[peaks < T]
                if len(valid) > 0:
                    self.sc_peaks_mv.set_offsets(
                        np.column_stack([t_wave[valid], rppg_signal[valid]])
                    )
                else:
                    self.sc_peaks_mv.set_offsets(np.empty((0, 2)))
            else:
                self.sc_peaks_mv.set_offsets(np.empty((0, 2)))

        # ── Header ───────────────────────────────────────────────────
        bpm_str  = f"{smoothed_bpm:.1f} BPM" if smoothed_bpm > 0 else "--- BPM"
        rmssd_str = f"{rmssd:.1f} ms"        if rmssd > 0        else "--- ms"
        self.txt_hdr.set_text(
            f"HR: {bpm_str}   |   RMSSD: {rmssd_str}   "
            f"|   window: last {self.history_sec:.0f} s"
        )

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        """Close the metrics figure and release resources."""
        plt.close(self.fig)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_rolling(
        self,
        ax,
        line,
        t_deque: "collections.deque",
        v_deque: "collections.deque",
        now: float,
        fixed_ylim,
    ) -> None:
        """
        Trim the stored (time, value) pairs to the rolling window and
        refresh the line artist.  ``fixed_ylim`` is a (lo, hi) tuple or
        None to use autoscale.
        """
        if not t_deque:
            return

        t_arr = np.array(t_deque)
        v_arr = np.array(v_deque)

        # Keep only last `history_sec` of data
        cutoff = now - self.history_sec
        mask   = t_arr >= cutoff
        if not mask.any():
            return

        t_win = t_arr[mask]
        v_win = v_arr[mask]

        line.set_data(t_win, v_win)
        ax.set_xlim(max(0.0, cutoff), now + 0.5)

        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)
        else:
            margin = max((v_win.max() - v_win.min()) * 0.15, 2.0)
            ax.set_ylim(v_win.min() - margin, v_win.max() + margin)
