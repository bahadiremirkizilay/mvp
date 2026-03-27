"""
Debug script to compare cached vs non-cached ROI extraction
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import cv2
import numpy as np
from rppg.roi_extractor import ROIExtractor
from rppg.landmark_cache import extract_and_cache_landmarks, cache_exists

def compare_cached_vs_live():
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    video_path = Path("data/ubfc/subject1/vid.avi")
    
    # Extract first 50 frames with both methods
    print("=" * 60)
    print("Testing first 50 frames: CACHED vs LIVE")
    print("=" * 60)
    
    # Get cached landmarks
    extractor_for_cache = ROIExtractor(config)
    if cache_exists(video_path):
        print("[CACHE] Using existing cache")
        cached_landmarks, _ = extract_and_cache_landmarks(
            video_path, extractor_for_cache, force_recompute=False
        )
    else:
        print("[ERROR] No cache found, run batch_validate first")
        return
    
    # Process with cache
    print("\n[1] Processing with CACHE...")
    cap = cv2.VideoCapture(str(video_path))
    extractor_cached = ROIExtractor(config)
    
    cached_signals = []
    cached_motion = []
    
    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        
        roi_signals, motion_conf, _ = extractor_cached.process_from_cached_landmarks(
            frame, cached_landmarks[i]
        )
        
        if roi_signals and roi_signals.get("forehead") is not None:
            cached_signals.append(roi_signals["forehead"])
            cached_motion.append(motion_conf)
    
    cap.release()
    extractor_cached.release()
    
    # Process without cache
    print("[2] Processing WITHOUT cache (live MediaPipe)...")
    cap = cv2.VideoCapture(str(video_path))
    extractor_live = ROIExtractor(config)
    
    live_signals = []
    live_motion = []
    
    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        
        roi_signals, motion_conf, _ = extractor_live.process(frame)
        
        if roi_signals and roi_signals.get("forehead") is not None:
            live_signals.append(roi_signals["forehead"])
            live_motion.append(motion_conf)
    
    cap.release()
    extractor_live.release()
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\nFrames with valid ROI:")
    print(f"  Cached: {len(cached_signals)}")
    print(f"  Live:   {len(live_signals)}")
    
    if len(cached_signals) > 0 and len(live_signals) > 0:
        cached_arr = np.array(cached_signals)
        live_arr = np.array(live_signals)
        
        print(f"\nROI Signal Statistics (forehead RGB):")
        print(f"  Cached mean: R={cached_arr[:, 0].mean():.1f}, G={cached_arr[:, 1].mean():.1f}, B={cached_arr[:, 2].mean():.1f}")
        print(f"  Live mean:   R={live_arr[:, 0].mean():.1f}, G={live_arr[:, 1].mean():.1f}, B={live_arr[:, 2].mean():.1f}")
        
        print(f"\n  Cached std:  R={cached_arr[:, 0].std():.2f}, G={cached_arr[:, 1].std():.2f}, B={cached_arr[:, 2].std():.2f}")
        print(f"  Live std:    R={live_arr[:, 0].std():.2f}, G={live_arr[:, 1].std():.2f}, B={live_arr[:, 2].std():.2f}")
        
        # Compare frame-by-frame
        min_len = min(len(cached_signals), len(live_signals))
        differences = []
        for i in range(min_len):
            diff = np.abs(cached_arr[i] - live_arr[i])
            differences.append(np.mean(diff))
        
        print(f"\nFrame-by-frame difference (mean absolute RGB):")
        print(f"  Average: {np.mean(differences):.2f}")
        print(f"  Max:     {np.max(differences):.2f}")
        print(f"  Min:     {np.min(differences):.2f}")
        
        # Check first few frames in detail
        print(f"\nFirst 5 frames (forehead RGB):")
        for i in range(min(5, min_len)):
            print(f"  Frame {i}:")
            print(f"    Cached: {cached_arr[i]}")
            print(f"    Live:   {live_arr[i]}")
            print(f"    Diff:   {np.abs(cached_arr[i] - live_arr[i])}")
    
    print(f"\nMotion confidence:")
    if len(cached_motion) > 0:
        print(f"  Cached: mean={np.mean(cached_motion):.3f}, min={np.min(cached_motion):.3f}, max={np.max(cached_motion):.3f}")
    if len(live_motion) > 0:
        print(f"  Live:   mean={np.mean(live_motion):.3f}, min={np.min(live_motion):.3f}, max={np.max(live_motion):.3f}")

if __name__ == "__main__":
    compare_cached_vs_live()
