"""
Multimodal Psychophysiological Analysis System — Main Entry Point
=================================================================
Phase 1: Classical rPPG engine (POS method, no deep learning).

Full pipeline per frame
-----------------------
    VideoCapture (30 FPS)
        ↓  ROIExtractor.process()
              MediaPipe Face Mesh → affine stabilization → ROI mask
              → mean RGB               [roi_extractor.py]
        ↓  Sliding window buffer  (deque, 10 s / 300 frames)
        ↓  [every vis_interval frames, once buffer ≥ min_frames]
        ↓  detrend_signal()            [signal_processing.py]
        ↓  pos_sliding_window()        [pos_method.py]
        ↓  bandpass_filter()           [signal_processing.py]
        ↓  normalize_signal()          [signal_processing.py]
        ↓  estimate_bpm()   (FFT)      [signal_processing.py]
        ↓  compute_hrv_metrics()       [hrv.py]
        ↓  Visualizer.update()         [utils/visualization.py]

Controls
--------
    Press  q  or  Esc  in the OpenCV preview window to quit.
"""

import sys

# ---------------------------------------------------------------------------
# Force UTF-8 stdout/stderr so non-ASCII characters in log messages never
# crash on Windows terminals with narrow codepages (e.g. cp1254 / cp1252).
# Must happen before any other import that might print to the console.
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Python version gate
# mediapipe 0.10.x officially supports Python 3.8 - 3.11 only.
# Python 3.12+ has known GIL / threading incompatibilities with mediapipe.
# ---------------------------------------------------------------------------
_ver = sys.version_info
if _ver >= (3, 12):
    print(
        "[ERROR] Python 3.12+ is NOT supported.\n"
        "        mediapipe 0.10.x requires Python 3.10 (recommended) or 3.11.\n"
        f"        Detected: Python {_ver.major}.{_ver.minor}.{_ver.micro}\n"
        "        Install Python 3.10 from https://www.python.org/downloads/"
        " and re-run with that interpreter.",
        file=sys.stderr,
    )
    sys.exit(1)
if _ver < (3, 9):
    print(
        f"[ERROR] Python {_ver.major}.{_ver.minor} is too old. Use Python 3.10.",
        file=sys.stderr,
    )
    sys.exit(1)
if _ver < (3, 10):
    print(
        f"[WARN]  Running on Python {_ver.major}.{_ver.minor}. "
        "Python 3.10 is strongly recommended for full mediapipe compatibility."
    )

import time
import csv
import collections
from datetime import datetime, timezone
from pathlib import Path

import yaml
import numpy as np
import cv2

# ── Project modules ───────────────────────────────────────────────────────────
from rppg.roi_extractor import ROIExtractor
from rppg.pos_method import pos_sliding_window
from rppg.signal_processing import (
    detrend_signal,
    bandpass_filter,
    estimate_bpm,
    normalize_signal,
    moving_average_filter,
    temporal_normalize_rgb,
    compute_peak_confidence,
    compute_signal_quality,
    compute_sqi,
)
from rppg.hrv import compute_hrv_metrics
from utils.visualization import Visualizer, MetricsVisualizer
from behavioral.stress import StressEstimator

# ── Configuration ─────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
_CSV_PATH    = Path(__file__).parent / "rppg_data.csv"

_CSV_COLUMNS = [
    "timestamp",
    "raw_bpm",
    "smoothed_bpm",
    "sdnn",
    "rmssd",
    "signal_quality",
]


def _open_csv_log(path: Path):
    """
    Open (or create) the CSV log file in append mode.

    Writes the header row only when the file is new/empty so that
    restarting the program never overwrites previous recordings.

    Returns
    -------
    (file_handle, csv.writer)
        Caller is responsible for closing the file handle on exit.
    """
    is_new = not path.exists() or path.stat().st_size == 0
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(_CSV_COLUMNS)
        fh.flush()
    return fh, writer


def load_config(path: Path) -> dict:
    """Load the YAML configuration file and return it as a dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def rgb_deque_to_array(rgb_deque: collections.deque) -> np.ndarray:
    """Convert the RGB deque snapshot to a contiguous (T, 3) float64 array."""
    return np.array(list(rgb_deque), dtype=np.float64)


# ---------------------------------------------------------------------------
# OpenCV overlay helpers
# ---------------------------------------------------------------------------

def _draw_hud(
    frame: np.ndarray,
    bpm: float,
    hrv_metrics: dict,
    fps: float,
    buffer_pct: int,
    peak_conf: float = 0.0,
    signal_quality: float = 0.0,
) -> None:
    """
    Render a minimal heads-up display onto the preview frame (in-place).

    Displays: HR (smoothed), SDNN, FPS, buffer fill, peak confidence,
    signal quality.  A semi-transparent dark banner sits behind the text.
    """
    h, w = frame.shape[:2]

    # Semi-transparent top banner (72 px tall to fit extra metrics row)
    roi = frame[0:72, 0:w]
    banner = roi.copy()
    cv2.rectangle(banner, (0, 0), (w, 72), (10, 10, 28), -1)
    cv2.addWeighted(banner, 0.65, roi, 0.35, 0, roi)
    frame[0:72, 0:w] = roi

    # HR
    bpm_str = f"HR: {bpm:.1f} BPM" if bpm > 0 else "HR: --- BPM"
    cv2.putText(
        frame, bpm_str, (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 229, 255), 2, cv2.LINE_AA
    )

    # SDNN
    sdnn = hrv_metrics.get("sdnn", 0.0)
    sdnn_str = f"SDNN: {sdnn:.1f} ms" if sdnn > 0 else "SDNN: ---"
    cv2.putText(
        frame, sdnn_str, (w // 2 - 80, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 255, 160), 1, cv2.LINE_AA
    )

    # FPS + buffer
    status = f"FPS: {fps:.1f}   Buffer: {buffer_pct:3d}%"
    cv2.putText(
        frame, status, (10, 46),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 150, 200), 1, cv2.LINE_AA
    )

    # Peak confidence + signal quality
    conf_str = f"Conf: {peak_conf:.2f}  SQ: {signal_quality:.2f}"
    cv2.putText(
        frame, conf_str, (10, 66),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 100), 1, cv2.LINE_AA
    )


# ---------------------------------------------------------------------------
# Per-ROI signal processing helper
# ---------------------------------------------------------------------------

def _run_roi_pipeline(
    rgb_buffer: np.ndarray,
    fs_actual: float,
    bp_low: float,
    bp_high: float,
    bp_order: int,
    ma_window_sec: float,
):
    """
    Full single-ROI rPPG extraction chain:
        detrend → temporal_normalize → POS → moving_avg
        → bandpass → normalize → FFT → spectral quality score

    Parameters
    ----------
    rgb_buffer : np.ndarray, shape (T, 3)
        Raw per-ROI mean RGB buffer.
    fs_actual : float
        Measured sampling rate in Hz.
    bp_low, bp_high : float
        Bandpass cutoff frequencies (Hz).
    bp_order : int
        Butterworth filter order.
    ma_window_sec : float
        Moving-average window length in seconds.

    Returns
    -------
    tuple (rppg_signal, freqs, power, quality) or None
        Returns None if the buffer is too short (< 32 samples).
    """
    if len(rgb_buffer) < 32:
        return None

    # 1. Remove linear trend per channel
    rgb_detrended = detrend_signal(rgb_buffer)

    # 2. Rolling normalisation — suppresses illumination drift
    tn_window = max(3, int(0.5 * fs_actual))
    rgb_norm  = temporal_normalize_rgb(rgb_detrended, tn_window)

    # 3. POS projection (overlap-add sub-windows)
    rppg_raw = pos_sliding_window(rgb_norm, fs_actual)

    # 4. Temporal moving average (~0.3 s)
    ma_samples    = max(3, int(ma_window_sec * fs_actual))
    rppg_smoothed = moving_average_filter(rppg_raw, ma_samples)

    # 5. Zero-phase Butterworth bandpass
    rppg_filtered = bandpass_filter(
        rppg_smoothed, bp_low, bp_high, fs_actual, bp_order
    )

    # 6. Z-score normalise
    rppg_sig = normalize_signal(rppg_filtered)

    # 7. Power spectrum (for fusion weighting and plotting)
    _, freqs_out, power_out = estimate_bpm(rppg_sig, fs_actual, bp_low, bp_high)

    # 8. Spectral purity quality score
    quality = compute_signal_quality(freqs_out, power_out, bp_low, bp_high)

    return rppg_sig, freqs_out, power_out, quality


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Load configuration ─────────────────────────────────────────────────
    config = load_config(_CONFIG_PATH)
    input_cfg = config.get("input", {})
    cam_cfg  = config["camera"]
    rppg_cfg = config["rppg"]
    hrv_cfg  = config.get("hrv", {})
    vis_cfg  = config.get("visualization", {})

    use_video_file = input_cfg.get("use_video_file", False)
    video_path = input_cfg.get("video_path", "")

    fs              = float(cam_cfg["fps"])
    window_samples  = int(rppg_cfg["window_seconds"] * fs)
    min_frames      = int(rppg_cfg.get("min_frames", 64))
    bp_low          = float(rppg_cfg["bandpass_low"])
    bp_high         = float(rppg_cfg["bandpass_high"])
    bp_order        = int(rppg_cfg.get("bandpass_order", 4))
    vis_interval    = int(vis_cfg.get("update_interval", 30))
    ma_window_sec   = float(rppg_cfg.get("ma_window_sec", 0.3))

    # Per-ROI sliding-window RGB buffers (deque maxlen auto-evicts oldest).
    # Kept separate so each ROI can be quality-scored and fused independently.
    forehead_deque:    collections.deque = collections.deque(maxlen=window_samples)
    left_cheek_deque:  collections.deque = collections.deque(maxlen=window_samples)
    right_cheek_deque: collections.deque = collections.deque(maxlen=window_samples)

    # Minimum motion confidence required to accept a frame into the buffers.
    # Frames below this threshold are silently dropped.
    _MOTION_CONF_THRESHOLD: float = 0.30

    # ── Initialise subsystems ──────────────────────────────────────────────
    extractor    = ROIExtractor(config)
    visualizer   = Visualizer(fs=fs, window_sec=rppg_cfg["window_seconds"])
    metrics_viz  = MetricsVisualizer(history_sec=30.0)
    stress_est   = StressEstimator()

    # ── CSV log ────────────────────────────────────────────────────────────
    csv_fh, csv_writer = _open_csv_log(_CSV_PATH)
    print(f"[INFO] CSV log → {_CSV_PATH}")

    # ── Video capture initialization (camera or file) ─────────────────────
    if use_video_file:
        if not video_path:
            print("[ERROR] use_video_file=true but video_path is not set in config/config.yaml")
            visualizer.close()
            sys.exit(1)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video file: {video_path}")
            visualizer.close()
            sys.exit(1)
        # Read actual FPS from video file
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0:
            print(f"[WARN] Video file FPS not detected, using config value: {fs:.1f}")
            actual_fps = fs
        else:
            # Update fs to match video file FPS
            fs = actual_fps
            window_samples = int(rppg_cfg["window_seconds"] * fs)
        
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] Video file opened: {video_path}")
        print(f"[INFO] Video FPS: {actual_fps:.1f}, Total frames: {total_frames_in_video}")
    else:
        cap = cv2.VideoCapture(cam_cfg["device_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        cap.set(cv2.CAP_PROP_FPS,          cam_cfg["fps"])
        
        if not cap.isOpened():
            print("[ERROR] Cannot open camera. "
                  "Check 'device_id' in config/config.yaml.")
            visualizer.close()
            sys.exit(1)
        
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Camera opened  —  reported FPS: {actual_fps:.1f}")
    
    print(f"[INFO] Buffering {rppg_cfg['window_seconds']} s "
          f"({window_samples} frames) before first HR estimate.")
    print("[INFO] Press  q  or  Esc  in the preview window to quit.\n")

    # ── State ──────────────────────────────────────────────────────────────
    bpm:         float = 0.0
    hrv_metrics: dict  = {}
    freqs:  np.ndarray = np.array([0.0])
    power:  np.ndarray = np.array([0.0])
    rppg_signal: np.ndarray = np.zeros(min_frames)

    # BPM stabilisation state
    # raw_bpm       — latest unfiltered candidate from peak-based HR.
    # bpm_history   — rolling window of last 8 accepted estimates;
    #                 median is used to suppress remaining outliers.
    # last_accepted_bpm — used for inter-estimate jump gating:
    #                 if |Δbpm| > 20 BPM the reading is discarded.
    raw_bpm:           float = 0.0
    bpm_history:       collections.deque = collections.deque(maxlen=8)
    last_accepted_bpm: float = 0.0

    # Spectral quality metrics — updated each processing cycle
    peak_conf:      float = 0.0
    signal_quality: float = 0.0   # fusion weight (band-power fraction)
    sqi:            float = 0.0   # user-facing Signal Quality Index
    stress_result:  dict  = {"stress_score": 0.0, "stress_level": "---", "partial_scores": {}}

    # Per-frame wall-clock timestamps — updated only for frames accepted
    # into the per-ROI deques (motion confidence ≥ threshold).
    # Used to compute actual sampling rate for downstream DSP.
    frame_timestamps: collections.deque = collections.deque(maxlen=window_samples)

    # Monotonically increasing frame counter for vis_interval modulo gating
    total_frames: int = 0

    # FPS display state
    fps_counter: int   = 0
    fps_display: float = 0.0
    t_fps_ref:   float = time.perf_counter()

    # ── Capture loop ───────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            if use_video_file:
                print("[INFO] End of video file reached.")
                break
            else:
                print("[WARN] Frame capture failed — retrying…")
                continue

        total_frames += 1
        fps_counter  += 1

        # Live FPS measurement (updated every second)
        now = time.perf_counter()
        elapsed = now - t_fps_ref
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            t_fps_ref   = now

        # ── ROI extraction ────────────────────────────────────────────
        roi_signals, motion_conf, debug_frame = extractor.process(frame)
        if roi_signals is not None and motion_conf >= _MOTION_CONF_THRESHOLD:
            frame_timestamps.append(time.perf_counter())
            if roi_signals.get("forehead")   is not None:
                forehead_deque.append(roi_signals["forehead"])
            if roi_signals.get("left_cheek")  is not None:
                left_cheek_deque.append(roi_signals["left_cheek"])
            if roi_signals.get("right_cheek") is not None:
                right_cheek_deque.append(roi_signals["right_cheek"])

        n_buffered  = max(len(forehead_deque), len(left_cheek_deque), len(right_cheek_deque))
        buffer_pct  = min(100, int(100 * n_buffered / window_samples))

        # ── HUD overlay + preview ─────────────────────────────────────
        _draw_hud(debug_frame, bpm, hrv_metrics, fps_display, buffer_pct,
                  peak_conf, signal_quality)
        cv2.imshow("rPPG — POS Method  (q / Esc to quit)", debug_frame)

        # ── Signal processing (gated by buffer fill and frame cadence) ─
        if n_buffered >= min_frames and (total_frames % vis_interval == 0):

            # Compute actual sampling rate from wall-clock timestamps.
            if len(frame_timestamps) >= 2:
                _ts = list(frame_timestamps)
                fs_actual = (len(_ts) - 1) / (_ts[-1] - _ts[0])
                # Clamp to ±50 % of nominal to guard against timestamp bursts
                fs_actual = float(np.clip(fs_actual, fs * 0.5, fs * 2.0))
            else:
                fs_actual = fs

            # ── Per-ROI signal extraction ────────────────────────────────
            # Each ROI independently: detrend → temporal_normalize → POS
            # → moving_avg → bandpass → normalize → FFT → quality_score
            _pipeline_args = (fs_actual, bp_low, bp_high, bp_order, ma_window_sec)
            roi_results: dict = {}
            for _rname, _dq in [
                ("forehead",    forehead_deque),
                ("left_cheek",  left_cheek_deque),
                ("right_cheek", right_cheek_deque),
            ]:
                if len(_dq) >= min_frames:
                    _res = _run_roi_pipeline(
                        rgb_deque_to_array(_dq), *_pipeline_args
                    )
                    if _res is not None:
                        roi_results[_rname] = _res  # (sig, freqs, power, quality)

            if roi_results:
                # ── Quality-weighted signal fusion ──────────────────────
                # Signals from all available ROIs are aligned to the
                # shortest buffer, then fused by spectral purity weight.
                _min_T   = min(len(v[0]) for v in roi_results.values())
                _total_w = sum(v[3]      for v in roi_results.values())

                if _total_w < 1e-8:
                    # All qualities zero → equal-weight fallback
                    _rois       = list(roi_results.values())
                    rppg_signal = np.mean(
                        np.stack([r[0][-_min_T:] for r in _rois], axis=0), axis=0
                    )
                    freqs, power   = _rois[0][1], _rois[0][2]
                    signal_quality = 0.0
                else:
                    fused  = np.zeros(_min_T, dtype=np.float64)
                    best_q = -1.0
                    for _sig, _fq, _pw, _q in roi_results.values():
                        fused += (_q / _total_w) * _sig[-_min_T:]
                        if _q > best_q:
                            freqs, power = _fq, _pw  # keep best-ROI spectrum
                            best_q = _q
                    rppg_signal    = normalize_signal(fused)
                    signal_quality = best_q

                # Re-derive combined spectrum from the fused signal
                _, freqs, power = estimate_bpm(
                    rppg_signal, fs_actual, bp_low, bp_high
                )

                # ── Spectral peak confidence & SQI ────────────────────
                peak_conf = compute_peak_confidence(freqs, power, bp_low, bp_high)
                sqi       = compute_sqi(freqs, power, bp_low, bp_high)

                # ── Forehead RGB for visualizer panel 1 (time-aligned) ───
                _T = len(rppg_signal)
                if len(forehead_deque) >= _T:
                    rgb_buffer_display = rgb_deque_to_array(forehead_deque)[-_T:]
                elif forehead_deque:
                    rgb_buffer_display = rgb_deque_to_array(forehead_deque)
                else:
                    rgb_buffer_display = np.zeros((_T, 3), dtype=np.float64)

                # ── HRV metrics on fused signal ───────────────────────
                hrv_metrics = compute_hrv_metrics(
                    rppg_signal,
                    fs_actual,
                    min_distance_sec=hrv_cfg.get("min_peak_distance_sec", 0.4),
                    min_prominence=hrv_cfg.get("min_peak_prominence", 0.2),
                )

                # ── BPM with outlier rejection and rolling median ──────
                #    Jump rejection: discard |Δbpm| > 20 BPM
                #    Rolling median (maxlen=8) absorbs residual outliers.
                candidate_bpm = hrv_metrics.get("hr_mean_bpm", 0.0)
                raw_bpm = candidate_bpm

                if candidate_bpm > 0:
                    if (
                        last_accepted_bpm > 0
                        and abs(candidate_bpm - last_accepted_bpm) > 20.0
                    ):
                        pass  # reject — jump exceeds 20 BPM
                    else:
                        last_accepted_bpm = candidate_bpm
                        bpm_history.append(candidate_bpm)

                bpm = float(np.median(list(bpm_history))) if bpm_history else 0.0

                # ── Refresh matplotlib plots ───────────────────────────
                visualizer.update(
                    rgb_buffer=rgb_buffer_display,
                    rppg_signal=rppg_signal,
                    freqs=freqs,
                    power=power,
                    bpm=bpm,
                    peaks=hrv_metrics.get("peaks"),
                    hrv_metrics=hrv_metrics,
                )
                metrics_viz.update(
                    smoothed_bpm=bpm,
                    rmssd=hrv_metrics.get("rmssd", 0.0),
                    rppg_signal=rppg_signal,
                    fs=fs_actual,
                    peaks=hrv_metrics.get("peaks"),
                )

                # ── Stress estimation ───────────────────────────────────
                stress_result = stress_est.estimate({
                    "heart_rate": bpm,
                    "rmssd":      hrv_metrics.get("rmssd",    0.0),
                    "sdnn":       hrv_metrics.get("sdnn",     0.0),
                    "lf_hf":      hrv_metrics.get("lf_hf",    0.0),
                })

                # ── Console output ────────────────────────────────────
                n_peaks  = hrv_metrics.get("n_peaks",  0)
                sdnn     = hrv_metrics.get("sdnn",     0.0)
                rmssd    = hrv_metrics.get("rmssd",    0.0)
                mean_rr  = hrv_metrics.get("mean_rr",  0.0)
                pnn50    = hrv_metrics.get("pnn50",    0.0)
                lf_power = hrv_metrics.get("lf_power", 0.0)
                hf_power = hrv_metrics.get("hf_power", 0.0)
                lf_hf    = hrv_metrics.get("lf_hf",    0.0)
                raw_str  = f"{raw_bpm:.1f}" if raw_bpm > 0 else "---"
                smth_str = f"{bpm:.1f}"     if bpm     > 0 else "---"
                rois_str = "/".join(roi_results.keys())
                sqi_warn = (
                    "\n  [!] Low signal quality – please reduce motion or improve lighting"
                    if sqi < 0.4 else ""
                )
                stress_score = stress_result["stress_score"]
                stress_level = stress_result["stress_level"]
                print(
                    f"\n[rPPG]"
                    f"\n  Raw BPM:         {raw_str:>6}"
                    f"\n  Smoothed BPM:    {smth_str:>6}"
                    f"\n  Peak Confidence: {peak_conf:.2f}"
                    f"\n  SDNN:    {sdnn:5.1f} ms"
                    f"\n  RMSSD:   {rmssd:5.1f} ms"
                    f"\n  Mean RR: {mean_rr:6.1f} ms"
                    f"\n  pNN50:   {pnn50:5.1f} %"
                    f"\n  LF Power:   {lf_power:8.1f}  |  HF Power: {hf_power:8.1f}  |  LF/HF: {lf_hf:.2f}"
                    f"\n  Stress Level: {stress_level}  |  Stress Score: {stress_score:.2f}"
                    f"\n  Signal Quality:  {sqi:.2f}  [{rois_str}]"
                    f"{sqi_warn}"
                    f"\n  Peaks: {n_peaks:3d}  |  Buffer: {n_buffered}/{window_samples}",
                )

                # ── CSV logging ───────────────────────────────────────
                csv_writer.writerow([
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                    f"{raw_bpm:.2f}",
                    f"{bpm:.2f}",
                    f"{sdnn:.2f}",
                    f"{rmssd:.2f}",
                    f"{sqi:.4f}",
                ])
                csv_fh.flush()

        # ── Exit condition ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):          # 'q' or Esc
            break

    # ── Graceful shutdown ──────────────────────────────────────────────────
    print("\n[INFO] Shutting down…")
    cap.release()
    extractor.release()
    visualizer.close()
    metrics_viz.close()
    csv_fh.close()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
