"""
Validation Script — UBFC-RPPG Dataset
=====================================
Compare system predictions against ground truth to compute:
  • MAE (Mean Absolute Error)
  • RMSE (Root Mean Squared Error)
  • Pearson Correlation

Usage
-----
python validate.py --video vid.avi --ground_truth ground_truth.txt
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import yaml
import collections
import time
from scipy import stats

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rppg.roi_extractor import ROIExtractor
from rppg.pos_method import pos_sliding_window
from rppg.signal_processing import (
    detrend_signal,
    bandpass_filter,
    normalize_signal,
    moving_average_filter,
    temporal_normalize_rgb,
    compute_signal_quality,
)
from rppg.hrv import compute_hrv_metrics


def load_ground_truth(gt_path: Path):
    """
    Load UBFC-RPPG ground truth file.
    
    Format:
      Line 1: rPPG signal (normalized)
      Line 2: BPM values (ground truth HR)
      Line 3: Timestamps (seconds)
    
    Returns
    -------
    dict with keys: 'rppg_signal', 'bpm', 'timestamps'
    """
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    
    rppg_signal = np.array([float(x) for x in lines[0].strip().split()])
    bpm_values  = np.array([float(x) for x in lines[1].strip().split()])
    timestamps  = np.array([float(x) for x in lines[2].strip().split()])
    
    return {
        'rppg_signal': rppg_signal,
        'bpm': bpm_values,
        'timestamps': timestamps
    }


def run_rppg_on_video(video_path: Path, config: dict):
    """
    Process video and extract BPM values at each processing interval.
    
    Returns
    -------
    list of tuples: [(timestamp, bpm, frame_idx), ...]
    """
    cam_cfg  = config["camera"]
    rppg_cfg = config["rppg"]
    hrv_cfg  = config.get("hrv", {})
    vis_cfg  = config.get("visualization", {})
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get actual video FPS
    fs = cap.get(cv2.CAP_PROP_FPS)
    if fs <= 0:
        print(f"[WARN] Video FPS not detected, using config default: {cam_cfg['fps']}")
        fs = float(cam_cfg['fps'])
    
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames_in_video / fs
    
    print(f"[INFO] Video: {video_path.name}")
    print(f"[INFO]   FPS: {fs:.2f}")
    print(f"[INFO]   Total frames: {total_frames_in_video}")
    print(f"[INFO]   Duration: {video_duration:.2f} s")
    
    window_samples  = int(rppg_cfg["window_seconds"] * fs)
    min_frames      = int(rppg_cfg.get("min_frames", 64))
    bp_low          = float(rppg_cfg["bandpass_low"])
    bp_high         = float(rppg_cfg["bandpass_high"])
    bp_order        = int(rppg_cfg.get("bandpass_order", 4))
    vis_interval    = int(vis_cfg.get("update_interval", 30))
    ma_window_sec   = float(rppg_cfg.get("ma_window_sec", 0.3))
    
    # rPPG buffers
    forehead_deque    = collections.deque(maxlen=window_samples)
    left_cheek_deque  = collections.deque(maxlen=window_samples)
    right_cheek_deque = collections.deque(maxlen=window_samples)
    frame_timestamps  = collections.deque(maxlen=window_samples)
    
    extractor = ROIExtractor(config)
    
    results = []  # [(timestamp, bpm, frame_idx), ...]
    
    frame_idx = 0
    start_time = time.perf_counter()
    
    bpm_history = collections.deque(maxlen=8)
    last_accepted_bpm = 0.0
    
    # Level 2: Enhanced motion filtering
    _MOTION_CONF_THRESHOLD = 0.55  # Increased for cleaner signals
    _SIGNAL_STABILITY_THRESHOLD = 0.15  # Max allowed temporal variance
    
    # Track previous ROI signals for frame-to-frame stability
    prev_roi_signal = None
    
    print(f"[INFO] Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # ROI extraction
        roi_signals, motion_conf, _ = extractor.process(frame)
        
        # Level 2: Frame-to-frame stability check
        signal_stable = True
        if roi_signals is not None and prev_roi_signal is not None:
            # Calculate frame-to-frame signal change
            signal_change = np.abs(roi_signals.get("forehead", np.zeros(3)) - prev_roi_signal)
            signal_stability = np.mean(signal_change) / (np.mean(np.abs(prev_roi_signal)) + 1e-6)
            signal_stable = signal_stability < _SIGNAL_STABILITY_THRESHOLD
        
        if roi_signals is not None and motion_conf >= _MOTION_CONF_THRESHOLD and signal_stable:
            prev_roi_signal = roi_signals.get("forehead", np.zeros(3))
            frame_timestamps.append(time.perf_counter())
            if roi_signals.get("forehead") is not None:
                forehead_deque.append(roi_signals["forehead"])
            if roi_signals.get("left_cheek") is not None:
                left_cheek_deque.append(roi_signals["left_cheek"])
            if roi_signals.get("right_cheek") is not None:
                right_cheek_deque.append(roi_signals["right_cheek"])
        
        n_buffered = max(len(forehead_deque), len(left_cheek_deque), len(right_cheek_deque))
        
        # Process signal when buffer is full and at interval
        if n_buffered >= min_frames and (frame_idx % vis_interval == 0):
            # Compute actual sampling rate
            if len(frame_timestamps) >= 2:
                _ts = list(frame_timestamps)
                fs_actual = (len(_ts) - 1) / (_ts[-1] - _ts[0])
                fs_actual = float(np.clip(fs_actual, fs * 0.5, fs * 2.0))
            else:
                fs_actual = fs
            
            # Per-ROI signal extraction
            roi_results = {}
            for _rname, _dq in [
                ("forehead", forehead_deque),
                ("left_cheek", left_cheek_deque),
                ("right_cheek", right_cheek_deque),
            ]:
                if len(_dq) >= min_frames:
                    rgb_buffer = np.array(list(_dq), dtype=np.float64)
                    
                    # Signal processing pipeline
                    rgb_detrended = detrend_signal(rgb_buffer)
                    tn_window = max(3, int(0.5 * fs_actual))
                    rgb_norm = temporal_normalize_rgb(rgb_detrended, tn_window)
                    rppg_raw = pos_sliding_window(rgb_norm, fs_actual)
                    ma_samples = max(3, int(ma_window_sec * fs_actual))
                    rppg_smoothed = moving_average_filter(rppg_raw, ma_samples)
                    rppg_filtered = bandpass_filter(rppg_smoothed, bp_low, bp_high, fs_actual, bp_order)
                    rppg_sig = normalize_signal(rppg_filtered)
                    
                    # Quality score
                    from rppg.signal_processing import estimate_bpm
                    _, freqs_out, power_out = estimate_bpm(rppg_sig, fs_actual, bp_low, bp_high)
                    quality = compute_signal_quality(freqs_out, power_out, bp_low, bp_high)
                    
                    roi_results[_rname] = (rppg_sig, freqs_out, power_out, quality)
            
            if roi_results:
                # Level 2: Quality-weighted fusion
                _min_T = min(len(v[0]) for v in roi_results.values())
                _total_w = sum(v[3] for v in roi_results.values())
                
                if _total_w < 1e-8:
                    _rois = list(roi_results.values())
                    rppg_signal = np.mean(
                        np.stack([r[0][-_min_T:] for r in _rois], axis=0), axis=0
                    )
                else:
                    fused = np.zeros(_min_T, dtype=np.float64)
                    for _sig, _fq, _pw, _q in roi_results.values():
                        fused += (_q / _total_w) * _sig[-_min_T:]
                    rppg_signal = normalize_signal(fused)
                
                # HRV & BPM estimation
                hrv_metrics = compute_hrv_metrics(
                    rppg_signal,
                    fs_actual,
                    min_distance_sec=hrv_cfg.get("min_peak_distance_sec", 0.4),
                    min_prominence=hrv_cfg.get("min_peak_prominence", 0.2),
                )
                
                candidate_bpm = hrv_metrics.get("hr_mean_bpm", 0.0)
                
                # Outlier rejection (relaxed threshold for dynamic HR changes in Dataset 2)
                if candidate_bpm > 0 and 40 <= candidate_bpm <= 180:
                    if last_accepted_bpm > 0 and abs(candidate_bpm - last_accepted_bpm) > 35.0:
                        pass  # reject only extreme outliers
                    else:
                        last_accepted_bpm = candidate_bpm
                        bpm_history.append(candidate_bpm)
                
                bpm = float(np.median(list(bpm_history))) if bpm_history else 0.0
                
                # Calculate timestamp (seconds from start)
                timestamp = frame_idx / fs
                
                if bpm > 0:
                    results.append((timestamp, bpm, frame_idx))
                    
        # Progress indicator
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames_in_video) * 100
            print(f"[INFO] Progress: {frame_idx}/{total_frames_in_video} ({progress:.1f}%)", end='\r')
    
    cap.release()
    extractor.release()
    
    print(f"\n[INFO] Processing complete. Extracted {len(results)} BPM estimates.")
    
    return results, fs


def interpolate_to_ground_truth(predictions, ground_truth):
    """
    Match predictions to nearest ground truth measurements.
    
    Instead of interpolation, find the closest GT sample for each prediction.
    This is more appropriate when GT has step-like changes.
    
    Parameters
    ----------
    predictions : list of (timestamp, bpm, frame_idx)
    ground_truth : dict with 'timestamps' and 'bpm'
    
    Returns
    -------
    tuple: (pred_bpm_aligned, gt_bpm_aligned)
    """
    if len(predictions) == 0:
        raise ValueError("No predictions available for alignment")
    
    pred_times = np.array([p[0] for p in predictions])
    pred_bpms  = np.array([p[1] for p in predictions])
    
    gt_times = ground_truth['timestamps']
    gt_bpms  = ground_truth['bpm']
    
    # Find overlapping time range
    t_min = max(pred_times.min(), gt_times.min())
    t_max = min(pred_times.max(), gt_times.max())
    
    # Filter predictions to overlapping range
    mask_pred = (pred_times >= t_min) & (pred_times <= t_max)
    pred_times_valid = pred_times[mask_pred]
    pred_bpms_valid = pred_bpms[mask_pred]
    
    # For each prediction, find nearest GT sample
    gt_bpms_matched = []
    for pred_t in pred_times_valid:
        # Find index of nearest GT timestamp
        idx = np.argmin(np.abs(gt_times - pred_t))
        gt_bpms_matched.append(gt_bpms[idx])
    
    gt_bpms_matched = np.array(gt_bpms_matched)
    
    print(f"[INFO] Matched {len(pred_bpms_valid)} predictions to GT")
    print(f"[INFO] Time range: {t_min:.2f} - {t_max:.2f} s")
    
    return pred_bpms_valid, gt_bpms_matched


def compute_metrics(pred_bpm, gt_bpm):
    """
    Compute MAE, RMSE, and Pearson correlation.
    
    Parameters
    ----------
    pred_bpm : np.ndarray
        Predicted BPM values
    gt_bpm : np.ndarray
        Ground truth BPM values
    
    Returns
    -------
    dict with metrics
    """
    mae = np.mean(np.abs(pred_bpm - gt_bpm))
    rmse = np.sqrt(np.mean((pred_bpm - gt_bpm) ** 2))
    
    # Pearson correlation
    correlation, p_value = stats.pearsonr(pred_bpm, gt_bpm)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'p_value': p_value,
        'n_samples': len(pred_bpm),
        'mean_pred': np.mean(pred_bpm),
        'mean_gt': np.mean(gt_bpm),
        'std_pred': np.std(pred_bpm),
        'std_gt': np.std(gt_bpm),
    }


def print_metrics_report(metrics: dict):
    """Print validation metrics in a formatted report."""
    print("\n" + "="*60)
    print("VALIDATION METRICS — UBFC-RPPG Dataset")
    print("="*60)
    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-"*60)
    print(f"{'MAE (BPM)':<25} {metrics['mae']:>15.3f}")
    print(f"{'RMSE (BPM)':<25} {metrics['rmse']:>15.3f}")
    print(f"{'Pearson Correlation':<25} {metrics['correlation']:>15.4f}")
    print(f"{'p-value':<25} {metrics['p_value']:>15.6f}")
    print(f"\n{'Samples Aligned':<25} {metrics['n_samples']:>15d}")
    print(f"\n{'Mean Predicted BPM':<25} {metrics['mean_pred']:>15.2f}")
    print(f"{'Mean Ground Truth BPM':<25} {metrics['mean_gt']:>15.2f}")
    print(f"{'Std Predicted BPM':<25} {metrics['std_pred']:>15.2f}")
    print(f"{'Std Ground Truth BPM':<25} {metrics['std_gt']:>15.2f}")
    print("="*60)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if metrics['mae'] < 5:
        mae_quality = "Excellent"
    elif metrics['mae'] < 10:
        mae_quality = "Good"
    elif metrics['mae'] < 15:
        mae_quality = "Acceptable"
    else:
        mae_quality = "Needs Improvement"
    print(f"  MAE Quality: {mae_quality}")
    
    if metrics['correlation'] > 0.9:
        corr_quality = "Excellent"
    elif metrics['correlation'] > 0.7:
        corr_quality = "Good"
    elif metrics['correlation'] > 0.5:
        corr_quality = "Moderate"
    else:
        corr_quality = "Weak"
    print(f"  Correlation Quality: {corr_quality}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate rPPG system against UBFC-RPPG ground truth"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="vid.avi",
        help="Path to video file (default: vid.avi)"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="ground_truth.txt",
        help="Path to ground truth file (default: ground_truth.txt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    gt_path = Path(args.ground_truth)
    config_path = Path(args.config)
    
    # Verify files exist
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)
    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        sys.exit(1)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override to use video file input
    if 'input' not in config:
        config['input'] = {}
    config['input']['use_video_file'] = True
    config['input']['video_path'] = str(video_path)
    
    # Load ground truth
    print(f"[INFO] Loading ground truth: {gt_path}")
    ground_truth = load_ground_truth(gt_path)
    print(f"[INFO] Ground truth samples: {len(ground_truth['bpm'])}")
    print(f"[INFO] GT BPM range: {ground_truth['bpm'].min():.1f} - {ground_truth['bpm'].max():.1f}")
    print(f"[INFO] GT time range: {ground_truth['timestamps'][0]:.2f} - {ground_truth['timestamps'][-1]:.2f} s")
    
    # Run rPPG on video
    print(f"\n[INFO] Running rPPG analysis on: {video_path}")
    predictions, fs = run_rppg_on_video(video_path, config)
    
    if len(predictions) == 0:
        print("[ERROR] No BPM predictions generated. Check video quality and face detection.")
        sys.exit(1)
    
    # Align predictions with ground truth
    print(f"\n[INFO] Aligning predictions with ground truth...")
    pred_aligned, gt_aligned = interpolate_to_ground_truth(predictions, ground_truth)
    
    # Save aligned time-series data for visualization
    pred_times = np.array([p[0] for p in predictions if p[2] < len(ground_truth['timestamps'])])[:len(pred_aligned)]
    data_path = Path("validation_data.csv")
    np.savetxt(data_path, np.column_stack((pred_times, pred_aligned, gt_aligned)),
               delimiter=',', header='Time(s),Predicted_BPM,GT_BPM', comments='',
               fmt='%.3f')
    print(f"[INFO] Time-series data saved to: {data_path}")
    
    # Compute metrics
    print(f"[INFO] Computing validation metrics...")
    metrics = compute_metrics(pred_aligned, gt_aligned)
    
    # Print report
    print_metrics_report(metrics)
    
    # Save results
    results_path = Path("validation_results.txt")
    with open(results_path, 'w') as f:
        f.write("VALIDATION METRICS — UBFC-RPPG Dataset\n")
        f.write("="*60 + "\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Ground Truth: {gt_path}\n")
        f.write(f"\nMAE (BPM): {metrics['mae']:.3f}\n")
        f.write(f"RMSE (BPM): {metrics['rmse']:.3f}\n")
        f.write(f"Pearson Correlation: {metrics['correlation']:.4f}\n")
        f.write(f"p-value: {metrics['p_value']:.6f}\n")
        f.write(f"\nSamples: {metrics['n_samples']}\n")
        f.write(f"Mean Predicted BPM: {metrics['mean_pred']:.2f}\n")
        f.write(f"Mean Ground Truth BPM: {metrics['mean_gt']:.2f}\n")
    
    print(f"[INFO] Results saved to: {results_path}")


if __name__ == "__main__":
    main()
