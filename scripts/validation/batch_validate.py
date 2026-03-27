"""
Batch Validation Script - Multiple UBFC-RPPG Subjects
======================================================
Process multiple subjects sequentially and generate comparative metrics.

Usage
-----
python batch_validate.py --subjects subject1 subject3 subject4 subject5 subject8
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import yaml
import collections
import time
from scipy import stats
import pandas as pd

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
from rppg.landmark_cache import (
    extract_and_cache_landmarks,
    cache_exists,
    clear_cache,
)


def load_ground_truth(gt_path: Path):
    """
    Load UBFC-RPPG ground truth file (Dataset 2 format).
    
    Format:
      Line 1: PPG signal (normalized) - not used
      Line 2: BPM values (ground truth HR from contact PPG)
      Line 3: Timestamps (seconds)
    
    Returns
    -------
    dict with keys: 'bpm', 'timestamps'
    """
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    
    bpm_values = np.array([float(x) for x in lines[1].strip().split()])
    timestamps = np.array([float(x) for x in lines[2].strip().split()])
    
    return {
        'bpm': bpm_values,
        'timestamps': timestamps
    }


def run_rppg_on_video(video_path: Path, config: dict, use_cache: bool = True, filtering_level: int = 2):
    """
    Process video and extract BPM values at each processing interval.
    
    Parameters
    ----------
    video_path : Path
        Path to video file
    config : dict
        Configuration dictionary
    use_cache : bool
        If True, use cached landmarks for deterministic results
    filtering_level : int
        0 = No filtering (raw), 1 = Basic motion filtering only, 2 = Enhanced (motion + stability)
    
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
        print(f"  [WARN] Video FPS not detected, using config default: {cam_cfg['fps']}")
        fs = float(cam_cfg['fps'])
    
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames_in_video / fs
    
    print(f"  Video: {total_frames_in_video} frames at {fs:.1f} FPS ({video_duration:.1f}s)")
    
    # Check for cached landmarks
    cached_landmarks = None
    if use_cache:
        if cache_exists(video_path):
            print(f"  [CACHE] Using cached landmarks for deterministic results")
            cached_landmarks, _ = extract_and_cache_landmarks(
                video_path, 
                ROIExtractor(config),  # Temporary extractor for cache
                force_recompute=False
            )
        else:
            print(f"  [CACHE] No cache found, will create on first run")
            cached_landmarks, _ = extract_and_cache_landmarks(
                video_path,
                ROIExtractor(config),
                force_recompute=False,
                progress_callback=lambda idx, total: print(f"  Caching: {int(idx/total*100)}%", end='\r', flush=True)
            )
            print()  # New line after caching progress
    
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
    
    # Filtering levels
    if filtering_level == 0:
        # Level 0: No filtering (raw)
        _MOTION_CONF_THRESHOLD = 0.0
        _SIGNAL_STABILITY_THRESHOLD = 1.0
    elif filtering_level == 1:
        # Level 1: Basic motion filtering
        _MOTION_CONF_THRESHOLD = 0.3
        _SIGNAL_STABILITY_THRESHOLD = 1.0
    else:
        # Level 2: Enhanced filtering (motion + stability)
        _MOTION_CONF_THRESHOLD = 0.55
        _SIGNAL_STABILITY_THRESHOLD = 0.15
    
    # Track previous ROI signals for frame-to-frame stability
    prev_roi_signal = None
    
    last_progress = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Progress indicator (every 20%)
        progress = int((frame_idx / total_frames_in_video) * 100)
        if progress >= last_progress + 20:
            print(f"  Progress: {progress}%", end='\r', flush=True)
            last_progress = progress
        
        # ROI extraction (cached or live)
        if use_cache and cached_landmarks is not None:
            # Use cached landmarks for deterministic results
            frame_landmarks = cached_landmarks[frame_idx - 1]  # 0-indexed
            roi_signals, motion_conf, _ = extractor.process_from_cached_landmarks(frame, frame_landmarks)
        else:
            # Normal MediaPipe-based extraction
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
    
    print()  # New line after progress
    cap.release()
    extractor.release()
    
    print(f"  Extracted {len(results)} BPM estimates")
    
    return results, fs


def align_predictions_to_gt(predictions, ground_truth):
    """
    Match predictions to nearest ground truth measurements.
    
    Parameters
    ----------
    predictions : list of (timestamp, bpm, frame_idx)
    ground_truth : dict with 'timestamps' and 'bpm'
    
    Returns
    -------
    tuple: (pred_bpm_aligned, gt_bpm_aligned)
    """
    if len(predictions) == 0:
        return np.array([]), np.array([])
    
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
        idx = np.argmin(np.abs(gt_times - pred_t))
        gt_bpms_matched.append(gt_bpms[idx])
    
    gt_bpms_matched = np.array(gt_bpms_matched)
    
    print(f"  Matched {len(pred_bpms_valid)} predictions to GT (time range: {t_min:.1f} - {t_max:.1f}s)")
    
    return pred_bpms_valid, gt_bpms_matched


def compute_metrics(pred_bpm, gt_bpm):
    """
    Compute MAE, RMSE, and Pearson correlation.
    
    Returns
    -------
    dict with metrics
    """
    mae = np.mean(np.abs(pred_bpm - gt_bpm))
    rmse = np.sqrt(np.mean((pred_bpm - gt_bpm) ** 2))
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
        'std_gt': np.std(gt_bpm)
    }


def main():
    parser = argparse.ArgumentParser(description="Batch validation for UBFC-RPPG subjects")
    parser.add_argument('--subjects', nargs='+', required=True,
                       help='List of subject IDs (e.g., subject1 subject3 subject4)')
    parser.add_argument('--dataset', type=str, default=str(PROJECT_ROOT / 'data' / 'ubfc'),
                       help='Path to UBFC-RPPG dataset')
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'config' / 'config.yaml'),
                       help='Path to config file')
    parser.add_argument('--output', type=str, default=str(PROJECT_ROOT / 'results' / 'batch_validation_results.csv'),
                       help='Output CSV file for results')
    parser.add_argument('--use-cache', action='store_true', default=True,
                       help='Use cached landmarks for deterministic results (default: True)')
    parser.add_argument('--no-cache', dest='use_cache', action='store_false',
                       help='Disable landmark caching (non-deterministic)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cached landmarks and recompute')
    parser.add_argument('--level', type=int, default=2, choices=[0, 1, 2],
                       help='Filtering level: 0=None, 1=Motion only, 2=Motion+Stability (default: 2)')
    
    args = parser.parse_args()
    
    # Handle cache clearing
    if args.clear_cache:
        from rppg.landmark_cache import clear_all_caches
        dataset_path = Path(args.dataset)
        print("[CACHE] Clearing all cached landmarks...")
        num_cleared = clear_all_caches(dataset_path)
        print(f"[CACHE] Cleared {num_cleared} cache file(s)")
        if num_cleared == 0:
            print("[CACHE] No cache files found")
        return
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(args.dataset)
    
    print("="*70)
    print("BATCH VALIDATION - UBFC-RPPG Dataset")
    print("="*70)
    print(f"Dataset path: {dataset_path}")
    print(f"Subjects: {', '.join(args.subjects)}")
    print(f"Configuration: {config_path}")
    
    level_names = {0: "Level 0 (No filtering)", 1: "Level 1 (Motion only)", 2: "Level 2 (Motion+Stability)"}
    print(f"Filtering: {level_names[args.level]}")
    
    if args.use_cache:
        print(f"Cache mode: ENABLED (deterministic results)")
    else:
        print(f"Cache mode: DISABLED (non-deterministic)")
    print("="*70)
    print()
    
    results = []
    
    for idx, subject_id in enumerate(args.subjects, 1):
        print(f"[{idx}/{len(args.subjects)}] Processing {subject_id}")
        print("-" * 70)
        
        subject_dir = dataset_path / subject_id
        video_path = subject_dir / 'vid.avi'
        gt_path = subject_dir / 'ground_truth.txt'
        
        if not subject_dir.exists():
            print(f"  ✗ ERROR: Subject directory not found: {subject_dir}")
            print()
            continue
        
        if not video_path.exists():
            print(f"  ✗ ERROR: Video not found: {video_path}")
            print()
            continue
        
        if not gt_path.exists():
            print(f"  ✗ ERROR: Ground truth not found: {gt_path}")
            print()
            continue
        
        try:
            start_time = time.time()
            
            # Load ground truth
            print("  Loading ground truth...")
            ground_truth = load_ground_truth(gt_path)
            print(f"  GT samples: {len(ground_truth['bpm'])}, Duration: {ground_truth['timestamps'][-1]:.1f}s")
            
            # Process video
            print("  Processing video...")
            predictions, fps = run_rppg_on_video(
                video_path, config, 
                use_cache=args.use_cache,
                filtering_level=args.level
            )
            
            if len(predictions) == 0:
                print(f"  ✗ WARNING: No predictions generated (face not detected?)")
                print()
                continue
            
            # Align predictions with GT
            pred_aligned, gt_aligned = align_predictions_to_gt(predictions, ground_truth)
            
            if len(pred_aligned) < 10:
                print(f"  ✗ WARNING: Too few aligned samples ({len(pred_aligned)})")
                print()
                continue
            
            # Compute metrics
            metrics = compute_metrics(pred_aligned, gt_aligned)
            metrics['subject_id'] = subject_id
            metrics['video_fps'] = fps
            
            elapsed = time.time() - start_time
            
            print(f"  ✓ SUCCESS")
            print(f"    MAE:         {metrics['mae']:.2f} BPM")
            print(f"    RMSE:        {metrics['rmse']:.2f} BPM")
            print(f"    Correlation: {metrics['correlation']:.3f} (p={metrics['p_value']:.4f})")
            print(f"    Samples:     {metrics['n_samples']}")
            print(f"    Time:        {elapsed:.1f}s")
            
            results.append(metrics)
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Save results
    if len(results) == 0:
        print("\n❌ No results generated!")
        sys.exit(1)
    
    df = pd.DataFrame(results)
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    
    print("="*70)
    print("BATCH VALIDATION COMPLETE")
    print("="*70)
    print(f"Processed: {len(results)}/{len(args.subjects)} subjects")
    print(f"Results saved to: {output_path}")
    print()
    
    # Summary statistics
    print("OVERALL METRICS (Mean ± Std)")
    print("-" * 70)
    print(f"MAE:         {df['mae'].mean():.2f} ± {df['mae'].std():.2f} BPM")
    print(f"RMSE:        {df['rmse'].mean():.2f} ± {df['rmse'].std():.2f} BPM")
    print(f"Correlation: {df['correlation'].mean():.3f} ± {df['correlation'].std():.3f}")
    print(f"Samples:     {df['n_samples'].mean():.0f} ± {df['n_samples'].std():.0f}")
    print()
    
    # Per-subject results
    print("PER-SUBJECT RESULTS")
    print("-" * 70)
    print(f"{'Subject':<12} {'MAE':>8} {'RMSE':>8} {'Corr':>8} {'Samples':>10}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['subject_id']:<12} {row['mae']:>8.2f} {row['rmse']:>8.2f} "
              f"{row['correlation']:>8.3f} {row['n_samples']:>10.0f}")
    print("="*70)


if __name__ == '__main__':
    main()
